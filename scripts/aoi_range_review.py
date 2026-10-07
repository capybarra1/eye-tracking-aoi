"""Read-only bidirectional previews, then a backed-up, bounded commit."""
from __future__ import annotations
import copy
import hashlib
import json
import sqlite3
import uuid
import numpy as np
from scripts.aoi_image_adjustments import make_tracker
from scripts.aoi_progress import annotation_status
from scripts.supervised_aoi import validate_polygons, Tracker


def issue_ranges(db, slices, fps):
    groups=[]
    for segment,frame,payload in db.execute('SELECT segment,frame,payload FROM records ORDER BY segment,frame'):
        problems=json.loads(payload)['problems']
        if not problems:continue
        status=annotation_status(problems)
        if groups and groups[-1]['segment']==segment and groups[-1]['status']==status and frame-groups[-1]['issue_end']<=max(1,int(fps*.3)):
            g=groups[-1];g['issue_end']=frame;g['reasons']=sorted(set(g['reasons']+problems))
        else:groups.append(dict(segment=segment,status=status,issue_start=frame,issue_end=frame,reasons=problems))
    for g in groups:
        row=slices[g['segment']];pad=max(1,int(fps*.2))
        g['start']=max(row['first'],g['issue_start']-pad);g['end']=min(row['last'],g['issue_end']+pad)
    return groups


def signature(s, start, end):
    rows=[s.db.execute(f'SELECT * FROM {table} WHERE segment=? AND frame BETWEEN ? AND ? ORDER BY frame',(s.segment,start,end)).fetchall() for table in ('records','anchors')]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def prepare_review(s,start,end,head,tail,method='tracking'):
    if method not in ('tracking','keyframes'):raise ValueError('未知复核方式')
    if type(start) is not int or type(end) is not int or not s.first_index<=start<end<=s.last_index:
        raise ValueError('首尾帧必须在当前切片内，尾帧晚于首帧')
    keys={start:copy.deepcopy(head),end:copy.deepcopy(tail)}
    for frame,payload in s.db.execute('SELECT frame,payload FROM anchors WHERE segment=? AND frame>? AND frame<? ORDER BY frame',(s.segment,start,end)):
        keys[frame]=json.loads(payload)
    for key in keys.values():key['polygons']=validate_polygons(key['polygons'],key['visible'],s.source.size)
    modes={bool(k['polygons'].get('partition')) for k in keys.values()}
    if len(modes)>1:raise ValueError('复核首尾及中间关键帧需使用相同分区方式，请分段复核')
    return dict(id=uuid.uuid4().hex,folder=str(s.folder.resolve()),segment=s.segment,start=start,end=end,
                signature=signature(s,start,end),keys=keys,records={},status='computing',progress=0,method=method)


def interpolate_shape(a,b,fraction):
    """Similarity motion plus residual shape change, with shortest-path rotation."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape:raise ValueError('首尾轮廓点数不同，请分段修复')
    ca,cb=a.mean(axis=0),b.mean(axis=0);x,y=a-ca,b-cb
    angle=np.arctan2(np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0]),np.sum(x*y))
    def rotation(theta):return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    size_a=np.linalg.norm(x);size_b=np.linalg.norm(y)
    if min(size_a,size_b)<1e-8:raise ValueError('关键帧轮廓过小，请重新调整')
    # Undo final rotation/scale to interpolate only the non-rigid residual.
    local=(1-fraction)*x/size_a+fraction*(y@rotation(angle))/size_b
    return (local@rotation(fraction*angle).T*((1-fraction)*size_a+fraction*size_b)+(1-fraction)*ca+fraction*cb).tolist()


def generate_keyframes(s,draft,tick=None,cancel=None):
    """Deterministic geometry only: no video decoding or tracking between keys."""
    from scripts.aoi_horizontal import partition_polygons,center_height,center_slope
    keys=draft['keys'];ordered=sorted(keys);total=draft['end']-draft['start']+1
    for left,right in zip(ordered,ordered[1:]):
        a,b=keys[left],keys[right]
        for i in range(left,right+1):
            if cancel and cancel():raise InterruptedError('已取消关键帧预览；正式标注未改动')
            f=(s.source.times[i]-s.source.times[left])/(s.source.times[right]-s.source.times[left])
            problems=[]
            if i in keys:
                p=copy.deepcopy(keys[i]['polygons']);visible=copy.deepcopy(keys[i]['visible'])
            else:
                visible={k:a['visible'][k] if f<.5 else b['visible'][k] for k in ('screen','tablet')}
                for k,label in (('screen','屏幕'),('tablet','平板')):
                    if a['visible'][k]!=b['visible'][k]:problems.append(label+'可见性变化，请在出现或消失处补关键帧')
                pa,pb=a['polygons'],b['polygons']
                if pa.get('partition'):
                    y=(1-f)*center_height(pa,s.source.size)+f*center_height(pb,s.source.size)
                    angle=(1-f)*np.arctan(center_slope(pa,s.source.size))+f*np.arctan(center_slope(pb,s.source.size))
                    gap=(1-f)*pa['partition']['gap_px']+f*pb['partition']['gap_px']
                    p=partition_polygons(s.source.size,y,gap,float(np.tan(angle)))
                else:
                    p={}
                    for k in ('screen','tablet'):
                        if len(pa[k])!=len(pb[k]):raise ValueError('首尾轮廓数量不同，请分段修复')
                        p[k]=[interpolate_shape(x,y,f) for x,y in zip(pa[k],pb[k])]
                p=validate_polygons(p,visible,s.source.size)
            draft['records'][i]=dict(frame_index=i,time_s=s.source.times[i],polygons=p,visible=visible,
                problems=problems,tracking=dict(method='manual_keyframe' if i in keys else 'manual_keyframe_interpolation',left=left,right=right))
            draft['progress']=min(99,round(len(draft['records'])*100/total))
            if tick:tick()
    draft.update(progress=100,status='ready',uncertain=sum(bool(r['problems']) for r in draft['records'].values()))


def generate_review(s,draft,tick=None,cancel=None):
    """Track from both ends of each manual-keyframe interval. Never change DB."""
    if draft.get('method')=='keyframes':return generate_keyframes(s,draft,tick,cancel)
    keys=draft['keys'];ordered=sorted(keys);steps=0;total=max(1,2*(draft['end']-draft['start']))
    def checkpoint():
        nonlocal steps
        if cancel and cancel():raise InterruptedError('已取消复核预览；正式标注未改动')
        steps+=1;draft['progress']=min(99,round(100*steps/total))
        if tick:tick()
    def record(i,polygons,visible,problems):
        return dict(frame_index=i,time_s=s.source.times[i],polygons=copy.deepcopy(polygons),visible=copy.deepcopy(visible),problems=problems)
    for i,key in keys.items():draft['records'][i]=record(i,key['polygons'],key['visible'],[])
    for left,right in zip(ordered,ordered[1:]):
        tracks=[]
        for first,indices in ((left,range(left+1,right)),(right,range(right-1,left,-1))):
            key=keys[first];tracker=make_tracker(s.source,Tracker if key['polygons'].get('partition') else s.factory,first,key['polygons'],key['visible']);rows={}
            previous=key['polygons']
            for i in indices:
                checkpoint();p,problems=tracker.update(s.source.frame(i))
                try:p=validate_polygons(p,key['visible'],s.source.size)
                except ValueError:p=previous;problems=sorted(set(problems+['轮廓异常']))
                rows[i]=(copy.deepcopy(p),problems);previous=p
            tracks.append(rows)
        for i in range(left+1,right):
            fraction=(s.source.times[i]-s.source.times[left])/(s.source.times[right]-s.source.times[left])
            result={};problems=[];visible={}
            for aoi,label in (('screen','屏幕'),('tablet','平板')):
                va,vb=keys[left]['visible'][aoi],keys[right]['visible'][aoi]
                visible[aoi]=va if fraction<.5 else vb
                pa,fa=tracks[0][i];pb,fb=tracks[1][i]
                gooda=not any(label in p or '轮廓异常' in p for p in fa)
                goodb=not any(label in p or '轮廓异常' in p for p in fb)
                a,b=np.asarray(pa[aoi]),np.asarray(pb[aoi])
                if va!=vb:problems.append(label+'可见性在段内变化，需补关键帧')
                if gooda and goodb and a.shape==b.shape:
                    mismatch=np.max(np.linalg.norm(a-b,axis=-1))
                    result[aoi]=((1-fraction)*a+fraction*b).tolist()
                    if mismatch>max(s.source.size)*.035:problems.append(label+'双向跟踪不一致，需复核')
                elif gooda:result[aoi]=pa[aoi]
                elif goodb:result[aoi]=pb[aoi]
                else:
                    a,b=np.asarray(keys[left]['polygons'][aoi]),np.asarray(keys[right]['polygons'][aoi])
                    result[aoi]=((1-fraction)*a+fraction*b).tolist() if a.shape==b.shape else copy.deepcopy(keys[left]['polygons'][aoi])
                    if va or vb:problems.append(label+'跟踪不足，暂用关键帧插值；需复核')
            if keys[left]['polygons'].get('partition'):result['partition']=copy.deepcopy(keys[left]['polygons']['partition'])
            try:result=validate_polygons(result,visible,s.source.size)
            except ValueError:result=copy.deepcopy(keys[left]['polygons']);problems.append('插值轮廓异常，需复核')
            draft['records'][i]=record(i,result,visible,problems)
    draft['progress']=100;draft['status']='ready'
    draft['uncertain']=sum(bool(r['problems']) for r in draft['records'].values())


def commit_review(s,draft):
    if draft['status']!='ready':raise ValueError('请先生成并检查预览')
    if str(s.folder.resolve())!=draft['folder'] or s.segment!=draft['segment']:raise ValueError('项目或切片已变化，请重新预览')
    start,end=draft['start'],draft['end']
    if signature(s,start,end)!=draft['signature']:raise ValueError('选定范围已发生变化，请重新预览')
    if set(draft['records'])!=set(range(start,end+1)):raise ValueError('预览帧不完整，不能保存')
    s.pause();folder=s.folder/'backups';folder.mkdir(exist_ok=True)
    backup=folder/f'before-review-{s.segment}-{start}-{end}-{draft["id"][:8]}.sqlite3'
    with sqlite3.connect(backup) as db:s.db.backup(db)
    # Only this closed interval is replaced. Existing internal anchors are kept.
    with s.db:
        for i,r in draft['records'].items():s._put(i,r['polygons'],r['visible'],r['problems'],tracking=r.get('tracking'))
        for i in (start,end):s.db.execute('INSERT OR REPLACE INTO anchors VALUES (?,?,?)',(s.segment,i,json.dumps(draft['keys'][i])))
        s._event('range_review',dict(start=start,end=end,backup=str(backup),uncertain=draft['uncertain'],method=draft.get('method','tracking')))
    if draft.get('method')=='keyframes':
        s.cursor=end;s.next_check=s.source.times[end]+s.checkpoint_s
    s.tracker=None;s.mode='paused';s.reason=f'已只保存第 {start}–{end} 帧；其他标注未改动';s._save()
    draft['status']='saved'
