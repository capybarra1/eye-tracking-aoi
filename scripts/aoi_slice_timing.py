"""Non-destructive local slice overrides; raw manifest and all annotations survive."""
from __future__ import annotations
import base64,bisect,copy,json,math,sqlite3,uuid
from scripts.aoi_image_adjustments import display_frame, apply_settings
import cv2


def effective_manifest(manifest,db):
    rows=copy.deepcopy(manifest)
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv'").fetchone():return rows
    saved=db.execute("SELECT value FROM kv WHERE key='slice_timing'").fetchone()
    overrides=json.loads(saved[0]) if saved else {}
    for row in rows:
        change=overrides.get(str(row['segment_id']))
        if change:
            row.setdefault('timing_before_adjustment',dict(start_s=row['start_s'],end_s=row['end_s']))
            row.update(start_s=change['start_s'],end_s=change['end_s'],timing_adjusted=True)
    return rows


def timing_info(s):
    original=next(r for r in s.base_manifest if str(r['segment_id'])==s.segment)
    return dict(duration_s=s.source.times[-1]+1/s.source.fps,last_frame=len(s.source.times)-1,
                original_start_s=original['start_s'],original_end_s=original['end_s'],
                original_first=bisect.bisect_left(s.source.times,original['start_s']),original_last=bisect.bisect_left(s.source.times,original['end_s'])-1)


def timing_preview(s,index):
    if type(index) is not int or not 0<=index<len(s.source.times):raise ValueError('预览帧超出当前录像范围')
    ok,data=cv2.imencode('.jpg',display_frame(s.source,index),[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:raise ValueError('预览图生成失败')
    return dict(index=index,time_s=s.source.times[index],end_s=s.source.times[index+1] if index+1<len(s.source.times) else s.source.times[-1]+1/s.source.fps,
                image='data:image/jpeg;base64,'+base64.b64encode(data).decode())


def adjust_timing(s,start,end):
    start,end=float(start),float(end)
    if not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end<=s.source.times[-1]+1/s.source.fps+.2:
        raise ValueError('切片范围无效，开始须早于结束且位于当前录像内')
    first=bisect.bisect_left(s.source.times,start);last=bisect.bisect_left(s.source.times,end)-1
    if first>last:raise ValueError('切片没有可用帧')
    keys=list(s.slices);pos=keys.index(s.segment)
    if pos and start<s.slices[keys[pos-1]]['end_s']:raise ValueError('与前一个切片重叠，请先缩短前一个切片的结尾')
    if pos+1<len(keys) and end>s.slices[keys[pos+1]]['start_s']:raise ValueError('与后一个切片重叠，请先调整后一个切片的开头')
    s.pause();old=copy.deepcopy(s.slices[s.segment]);folder=s.folder/'backups';folder.mkdir(exist_ok=True)
    backup=folder/f'before-timing-{s.segment}-{uuid.uuid4().hex[:8]}.sqlite3'
    with sqlite3.connect(backup) as db:s.db.backup(db)
    changes=s._get('slice_timing') or {};changes[s.segment]=dict(start_s=start,end_s=end)
    s._set('slice_timing',changes)
    manifest=effective_manifest(s.base_manifest,s.db)
    row=next(r for r in manifest if str(r['segment_id'])==s.segment)
    s.manifest=manifest;s.slices[s.segment]={**row,'first':first,'last':last}
    s.first_index=first;s.last_index=last
    # Begin at the first hole, including a newly extended head or tail.
    frames={r[0] for r in s.db.execute('SELECT frame FROM records WHERE segment=? AND frame BETWEEN ? AND ?',(s.segment,first,last))}
    s.cursor=next((i for i in range(first,last+1) if i not in frames),min(last,max(first,s.cursor)))
    s.tracker=None;s.running=False;s.mode='paused' if s.get_record(s.cursor) else 'uninitialized'
    s.reason='切片范围已保存；已有标注保留，新增部分待标注'
    s.next_check=s.source.times[s.cursor]+s.checkpoint_s
    s._set('auto_retry_failed',None)
    s._event('adjust_timing',dict(before=dict(start_s=old['start_s'],end_s=old['end_s']),after=changes[s.segment],backup=str(backup)))
    s._save()
    return dict(backup=str(backup),first=first,last=last)


def adjust_frames(s,first,last):
    if type(first) is not int or type(last) is not int or not 0<=first<=last<len(s.source.times):
        raise ValueError('切片帧范围无效')
    end=s.source.times[last+1] if last+1<len(s.source.times) else s.source.times[-1]+1/s.source.fps
    row=s.slices[s.segment]
    start=row['start_s'] if first==row['first'] else s.source.times[first]
    if last==row['last']:end=row['end_s']
    keys=list(s.slices);pos=keys.index(s.segment)
    if pos+1<len(keys):
        following=s.slices[keys[pos+1]]['start_s']
        if s.source.times[last]<following:end=min(end,following)
    return adjust_timing(s,start,end)
