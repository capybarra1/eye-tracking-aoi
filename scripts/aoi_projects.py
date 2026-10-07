"""Local project index and lossless, video-verified review packages."""
from __future__ import annotations
import copy
import csv
import io
import os
import tempfile
from datetime import datetime
import hashlib
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from scripts.supervised_aoi import Session, VideoSource, validate_polygons, coverage_for, effective_aois
from scripts.aoi_import_status import prioritize_imports

ROOT=Path(__file__).resolve().parents[1]
TIMING=ROOT/'outputs/aoi_workflow_20260928/timing_v3'
VIDEO_ROOT=ROOT/'videos'
SEGMENTS=[f'{i}.{j}' for i in range(1,7) for j in (1,2)]


def fingerprint(path: Path, cancel=None) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            if cancel and cancel():raise InterruptedError('已暂停视频准备')
            h.update(block)
    return h.hexdigest()


def validate_manifest(rows: list, subject: int, video: Path, partial: bool=False) -> list:
    ids=[str(r.get('segment_id')) for r in rows] if isinstance(rows,list) else []
    if not ids or ids!=[s for s in SEGMENTS if s in ids] or (not partial and ids!=SEGMENTS):
        raise ValueError('请提供按 1.1 至 6.2 排列的 12 个切片')
    result=[];previous=0.
    for row in rows:
        start,end=float(row['start_s']),float(row['end_s'])
        if not np.isfinite([start,end]).all() or not previous<=start<end:
            raise ValueError('切片时间必须递增、不重叠，结束晚于开始')
        result.append({**row,'subject_id':subject,'video_path':str(video),'start_s':start,'end_s':end})
        previous=end
    return result


def make_package(session: Session, digest: str) -> dict:
    bounds={str(r['segment_id']):(r['start_s'],r['end_s']) for r in session.manifest}
    inside=lambda seg,frame: seg in bounds and bounds[seg][0]<=session.source.times[frame]<bounds[seg][1]
    records=[dict(segment=seg,frame=frame,payload=json.loads(p),approved=bool(a))
             for seg,frame,p,a in session.db.execute('SELECT segment,frame,payload,approved FROM records ORDER BY segment,frame') if inside(seg,frame)]
    anchors=[dict(segment=seg,frame=frame,payload=json.loads(p))
             for seg,frame,p in session.db.execute('SELECT segment,frame,payload FROM anchors ORDER BY segment,frame') if inside(seg,frame)]
    return dict(format='aoi-review-v1',video_sha256=digest,video_path=str(session.source.path),
                size=list(session.source.size),timestamps=session.source.times,manifest=session.manifest,
                image_adjustments=session._get('image_adjustments'),horizontal_mode=bool(session._get('horizontal_mode')),partition_gap=session._get('partition_gap'),fine_partition_seed=session._get('fine_partition_seed'),template=session.template,records=records,anchors=anchors,selected=session.segment,cursor=session.cursor,
                note='标注存档不含视频；导入独立副本。原始多边形、可见性、异常与审核标记保留。')


def validate_package(package: dict, source: VideoSource, digest: str, manifest: list) -> None:
    if package.get('format')!='aoi-review-v1' or package.get('video_sha256')!=digest:
        raise ValueError('视频内容与标注存档不一致，已拒绝导入')
    times=package.get('timestamps')
    if not isinstance(times,list) or len(times)!=len(source.times) or not np.allclose(times,source.times,rtol=0,atol=1e-6) or package.get('size')!=list(source.size):
        raise ValueError('视频时间轴或坐标尺寸不一致，已拒绝导入')
    slices={r['segment_id']:r for r in manifest}
    for name in ('records','anchors'):
        rows=package.get(name)
        if not isinstance(rows,list) or len(rows)>len(source.times):raise ValueError('复核记录数量无效')
        seen=set()
        for row in rows:
            seg,index=row['segment'],row['frame']
            if seg not in slices or type(index) is not int or not 0<=index<len(source.times) or (seg,index) in seen:
                raise ValueError('标注存档切片/帧号重复或越界')
            seen.add((seg,index));s=slices[seg];p=row['payload']
            if not s['start_s']<=source.times[index]<s['end_s']:raise ValueError('帧与切片时间不对应')
            validate_polygons(p['polygons'],p['visible'],source.size)
            if name=='records':
                if p.get('frame_index')!=index or abs(float(p.get('time_s',-1))-source.times[index])>1e-8:
                    raise ValueError('记录中的帧号与时间戳不一致')
                if not isinstance(p.get('problems'),list) or not all(isinstance(x,str) for x in p['problems']) or type(row.get('approved')) is not bool:
                    raise ValueError('异常或审核标记格式错误')
    record_keys={(r['segment'],r['frame']) for r in package['records']}
    if any((r['segment'],r['frame']) not in record_keys for r in package['anchors']):raise ValueError('人工关键帧没有对应标注记录')
    selected=package.get('selected');cursor=package.get('cursor')
    if selected not in slices or type(cursor) is not int or not 0<=cursor<len(source.times):raise ValueError('复核位置无效')
    if not slices[selected]['start_s']<=source.times[cursor]<slices[selected]['end_s']:raise ValueError('复核位置不在所选切片内')


class Projects:
    def __init__(self, session: Session, timestamps: Path, root: Path | None=None):
        self.root=root or ROOT/'outputs/aoi_local_projects';self.root.mkdir(parents=True,exist_ok=True)
        self.index_path=self.root/'projects.json'
        self.index=json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.eligible={r['subject_id']:r for r in json.loads((TIMING/'eligible_subject_inventory.json').read_text()) if r['eligible']}
        fallback='original-'+str(session.manifest[0]['subject_id'])+'-'+hashlib.sha256(str(session.folder.resolve()).encode()).hexdigest()[:8]
        self.current=next((k for k,v in self.index.items() if Path(v['folder']).resolve()==session.folder.resolve()),fallback)
        self.register(self.current,session.folder,session.base_manifest,timestamps,session.template,'标注')
        self.hashes={};self.job=None

    def register(self, key: str, folder: Path, manifest: list, timestamps: Path, template: dict, kind: str):
        folder=folder.resolve();mp=folder/'manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        self.index[key]=dict(id=key,subject=manifest[0]['subject_id'],folder=str(folder),manifest=str(mp),
                             timestamps=str(timestamps.resolve()),template=template,video=manifest[0]['video_path'],kind=kind,
                             video_stat=[Path(manifest[0]['video_path']).stat().st_size,Path(manifest[0]['video_path']).stat().st_mtime_ns])
        temp=self.index_path.with_suffix('.tmp');temp.write_text(json.dumps(self.index,ensure_ascii=False,indent=2));temp.replace(self.index_path)

    def digest(self, video: Path) -> str:
        stat=video.stat();key=(str(video),stat.st_size,stat.st_mtime_ns)
        if key not in self.hashes:self.hashes[key]=fingerprint(video,getattr(self,'cancel_check',None))
        return self.hashes[key]

    def catalog(self) -> dict:
        videos=[]
        prepared=self.prepared_imports()
        for row in prepared:
            existing=next((p['id'] for p in self.index.values() if p.get('import_key')==row['key']),None)
            fallback=row.get('existing_project_id')
            if not existing and fallback in self.index and self.index[fallback]['subject']==row['subject']:
                existing=fallback
            videos.append({**row,'project_id':existing,'available':Path(row['path']).is_file() or bool(existing and Path(self.index[existing]['video']).is_file())})
        for p in sorted(VIDEO_ROOT.glob('*/scenevideo.mp4')):
            if prepared:break
            try:
                name=json.loads((p.parent/'meta/participant').read_text())['name'];match=re.match(r'\s*(\d+)',name)
                subject=int(match[1]) if match else None
                if subject in self.eligible:videos.append(dict(path=str(p),subject=subject,name=name))
            except (OSError,ValueError,KeyError):continue
        return dict(projects=[{**{k:v[k] for k in ('id','subject','video','kind')},'label':v.get('label','')} for v in self.index.values()],
                    videos=prioritize_imports(videos,self.index),subjects=sorted(self.eligible),current=self.current,
                    preset25=json.loads((TIMING/'demo_segments.json').read_text()),timing_images={k:v['image_count'] for k,v in self.eligible.items()})

    def prepared_imports(self) -> list:
        path=ROOT/'imports/catalog.json'
        if not path.exists():return []
        data=json.loads(path.read_text())
        if data.get('version')!=1:raise ValueError('预配置导入清单版本不支持')
        return [r for r in data['videos'] if r['subject'] in self.eligible]

    def times(self, video: Path, digest: str) -> Path:
        # Reuse a verified existing project index for precisely the same content.
        for project in self.index.values():
            if Path(project['video']).resolve()==video and project.get('video_stat')==[video.stat().st_size,video.stat().st_mtime_ns] and Path(project['timestamps']).exists():
                return Path(project['timestamps'])
        path=self.root/f'pts-{digest}.json'
        if not path.exists():
            cap=cv2.VideoCapture(str(video));times=[]
            try:
                if not cap.isOpened():raise ValueError('无法打开视频')
                while cap.grab():
                    if getattr(self,'cancel_check',lambda:False)():raise InterruptedError('已暂停视频索引')
                    times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            finally:cap.release()
            if not times or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('视频时间轴无法可靠索引')
            path.write_text(json.dumps(times))
        return path

    def open(self, key: str) -> Session:
        if key not in self.index:raise ValueError('未找到本地项目')
        p=self.index[key]
        if not Path(p['video']).is_file():raise ValueError('视频未找到，请连接原始数据硬盘并确认文件夹未移动')
        source=VideoSource(Path(p['video']),Path(p['timestamps']))
        try:s=Session(json.loads(Path(p['manifest']).read_text()),Path(p['folder']),source,p['template'])
        except Exception:source.close();raise
        self.current=key;return s

    def create(self, data: dict) -> Session:
        video=Path(data['video']).expanduser().resolve();subject=int(data['subject'])
        if subject not in self.eligible:raise ValueError('该被试不在绿色可用名单内')
        imported=None
        if data.get('import_key'):
            imported=next((r for r in self.prepared_imports() if r['key']==data['import_key']),None)
            if not imported or imported['subject']!=subject or Path(imported['path']).resolve()!=video:
                raise ValueError('导入条目与被试或视频不一致，请重新选择')
            existing=next((p['id'] for p in self.index.values() if p.get('import_key')==imported['key']),None)
            fallback=imported.get('existing_project_id')
            if not existing and fallback in self.index and self.index[fallback]['subject']==subject:existing=fallback
            if existing:return self.open(existing)
        if not video.is_file() or video.suffix.lower() not in ('.mp4','.mov','.avi','.mkv'):raise ValueError('请输入有效的本地视频路径')
        participant=video.parent/'meta/participant'
        if participant.exists():
            name=json.loads(participant.read_text()).get('name','');match=re.match(r'\s*(\d+)',name)
            if match and int(match[1])!=subject:raise ValueError('所选被试与视频旁的 participant 编号不一致')
        rows=validate_manifest(data['manifest'],subject,video,partial=data.get('split_recording') is True)
        if imported:
            originals={r['segment_id']:r for r in imported['manifest']}
            rows=[{**originals.get(r['segment_id'],{}),**r} for r in rows]
        digest=self.digest(video);timestamps=self.times(video,digest)
        source=VideoSource(video,timestamps);w,h=source.size
        template=dict(screen=[[[0,h*.65],[w/3,h*.65],[w*2/3,h*.65],[w-1,h*.65]]],tablet=[[[w*.55,h*.65],[w*.85,h*.65],[w*.85,h*.95],[w*.55,h*.95]]])
        key=f'{subject}-{uuid.uuid4().hex[:10]}';folder=self.root/key
        try:s=Session(rows,folder,source,template)
        except Exception:source.close();raise
        self.register(key,folder,rows,timestamps,template,'标注')
        if imported:
            self.index[key].update(import_key=imported['key'],label=imported['name'],import_notes=imported['notes'])
            temp=self.index_path.with_suffix('.tmp');temp.write_text(json.dumps(self.index,ensure_ascii=False,indent=2));temp.replace(self.index_path)
        self.current=key;return s

    def saved_annotations(self, session: Session) -> dict:
        saved=session._get('annotation_save') or {}
        if not saved:return dict(status='none')
        folder=session.folder/'exports'
        result={**saved,'folder':str(folder.resolve())}
        result['status']='saved' if all((folder/saved[k]).is_file() for k in ('archive_name','csv_name')) else 'missing'
        return result

    def save_annotations(self, session: Session, reason: str='手动保存') -> dict:
        """Publish both files from one committed snapshot, retaining the prior pair on error.

        Called under the server lock, never from the frame-processing hot path.
        Does not pause: continuing after a correction must remain possible.
        """
        session.db.commit()
        package=make_package(session,self.digest(session.source.path))
        save_id=uuid.uuid4().hex
        saved_at=datetime.now().astimezone().isoformat(timespec='seconds')
        subject=session.manifest[0]['subject_id']
        recording=re.sub(r'[^\w.-]+','_',session.source.path.parent.name)[:40] or '视频'
        project=hashlib.sha256(session.folder.name.encode()).hexdigest()[:6]
        stem=f'{subject}号_{recording}_{project}'
        names=[stem+'_标注存档.aoi.json',stem+'_数据表.csv']
        summary=dict(status='saved',saved_at=saved_at,save_id=save_id,reason=reason,
                     frames=len(package['records']),archive_name=names[0],csv_name=names[1])
        package['saved_at']=saved_at;package['save_id']=save_id
        table=io.StringIO(newline='');writer=csv.writer(table)
        writer.writerow(['subject_id','segment_id','frame_index','time_s','approved','screen_visible','tablet_visible','problems','aois_json','save_id','aoi_definition','gap_px'])
        for row in package['records']:
            r=row['payload'];aois=effective_aois(r['polygons'],r['visible'],session.source.size)
            writer.writerow([subject,row['segment'],row['frame'],r['time_s'],row['approved'],bool(aois['screen']),bool(aois['tablet']),
                             ';'.join(r['problems']),json.dumps(aois,ensure_ascii=False),save_id,r['polygons'].get('partition',{}).get('mode','precise'),r['polygons'].get('partition',{}).get('gap_px','')])
        contents=[json.dumps(package,ensure_ascii=False).encode('utf-8'),table.getvalue().encode('utf-8-sig')]
        folder=session.folder/'exports';folder.mkdir(exist_ok=True)
        old=session._get('annotation_save')
        with tempfile.TemporaryDirectory(prefix='.saving-',dir=folder) as temp:
            stage=Path(temp)
            # Prepare both files before touching the published pair.
            for i,(name,content) in enumerate(zip(names,contents)):
                with (stage/name).open('wb') as f:
                    f.write(content);f.flush();os.fsync(f.fileno())
                target=folder/name
                if target.exists():(stage/f'{i}.bak').write_bytes(target.read_bytes())
            replaced=[]
            try:
                for i,name in enumerate(names):
                    (stage/name).replace(folder/name);replaced.append((i,name))
                session._set('annotation_save',summary);session.db.commit()
            except Exception:
                session.db.rollback()
                for i,name in reversed(replaced):
                    backup=stage/f'{i}.bak'
                    if backup.exists():backup.replace(folder/name)
                    else:(folder/name).unlink(missing_ok=True)
                if old is not None:session._set('annotation_save',old)
                else:session.db.execute("DELETE FROM kv WHERE key='annotation_save'")
                session.db.commit()
                raise
        return self.saved_annotations(session)

    def export(self, session: Session) -> Path:
        # Keep the existing Python entry point compatible with callers.
        saved=self.save_annotations(session)
        return session.folder/'exports'/saved['archive_name']

    def import_review(self, package: dict, video_override: str='') -> Session:
        video=Path(video_override or package['video_path']).expanduser().resolve()
        if not video.is_file():raise ValueError('标注存档中的视频不存在，请填写此电脑上的视频路径')
        subject=int(package['manifest'][0]['subject_id'])
        if subject not in self.eligible:raise ValueError('标注存档被试不在绿色可用名单')
        manifest=validate_manifest(package['manifest'],subject,video,partial=True)
        if any(int(r['subject_id'])!=subject for r in package['manifest']):raise ValueError('包内存在不同被试')
        digest=self.digest(video)
        if package.get('video_sha256')!=digest:raise ValueError('视频内容与标注存档不一致，已拒绝导入')
        timestamps=self.times(video,digest);source=VideoSource(video,timestamps)
        try:
            validate_package(package,source,digest,manifest)
            validate_polygons(package['template'],dict(screen=True,tablet=True),source.size)
            key=f'review-{subject}-{uuid.uuid4().hex[:10]}';folder=self.root/key
            s=Session(manifest,folder,source,package['template'])
            with s.db:
                from scripts.aoi_image_adjustments import validate
                settings=validate(package.get('image_adjustments') or {})
                s._set('image_adjustments',settings);s.source.image_adjustments=settings
                s._set('horizontal_mode',package.get('horizontal_mode') is True)
                gap=package.get('partition_gap')
                if gap is not None:
                    if not isinstance(gap,(float,int)) or not np.isfinite(gap) or not 0<=gap<=source.size[1]/2:raise ValueError('间隔宽度无效')
                    s._set('partition_gap',gap)
                if package.get('fine_partition_seed'):
                    s._set('fine_partition_seed',validate_polygons(package['fine_partition_seed'],dict(screen=True,tablet=True),source.size))
                for r in package['records']:s.db.execute('INSERT INTO records VALUES (?,?,?,?)',(r['segment'],r['frame'],json.dumps(r['payload']),int(r['approved'])))
                for r in package['anchors']:s.db.execute('INSERT INTO anchors VALUES (?,?,?)',(r['segment'],r['frame'],json.dumps(r['payload'])))
            selected=package.get('selected',manifest[0]['segment_id']);s.select(selected)
            cursor=package.get('cursor',s.first_index);s.seek(cursor)
        except Exception:source.close();raise
        self.register(key,folder,manifest,timestamps,package['template'],'复核副本');self.current=key;return s

    def analyze(self, session: Session) -> dict:
        if self.job and self.job['process'].poll() is None:raise ValueError('已有指标任务正在运行')
        if not (session.source.path.parent/'gazedata.gz').exists():raise ValueError('视频同目录缺少 gazedata.gz，无法计算眼动指标')
        session.pause();p=self.index[self.current];out=session.folder/'metrics'/uuid.uuid4().hex[:10];out.mkdir(parents=True)
        effective=out/'manifest.json';effective.write_text(json.dumps(session.manifest,ensure_ascii=False))
        log=(out/'run.log').open('w')
        command=[sys.executable,'-m','scripts.compute_supervised_aoi_metrics','--manifest',str(effective),'--database',str(session.folder/'session.sqlite3'),'--timestamps',p['timestamps'],'--out',str(out)]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);log.close()
        self.job=dict(process=process,folder=out,subject=p['subject']);return self.analysis_state()

    def analysis_state(self) -> dict:
        if not self.job:return dict(status='none')
        code=self.job['process'].poll();out=self.job['folder']
        return dict(status='running' if code is None else 'complete' if code==0 else 'failed',folder=str(out),subject=self.job['subject'],
                    csv=str(out/f'{self.job["subject"]}号_切片指标.csv'),
                    error=(out/'run.log').read_text()[-1500:] if code not in (None,0) else '')
