"""Read-only progress colours and a subject directory; never confirms annotations."""
from __future__ import annotations
from scripts.aoi_slice_timing import effective_manifest
from bisect import bisect_left
import copy
import json
from pathlib import Path
import sqlite3

SEED_PENDING='起始轮廓待确认'


def annotation_status(problems,approved=False):
    if any(p!=SEED_PENDING for p in problems):return 'issue'
    if SEED_PENDING in problems:return 'unverified'
    return 'approved' if approved else 'candidate'


def append_span(spans,index,status):
    if spans and spans[-1]['status']==status and spans[-1]['end']==index-1:spans[-1]['end']=index
    else:spans.append(dict(start=index,end=index,status=status))


_cache={}

def project_progress(project):
    dbpath=Path(project['folder'])/'session.sqlite3'
    paths=[dbpath,Path(str(dbpath)+'-wal'),Path(project['timestamps']),Path(project['manifest'])]
    stamp=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else (str(p),None) for p in paths)
    if stamp in _cache:return copy.deepcopy(_cache[stamp])
    times=json.loads(paths[2].read_text());manifest=json.loads(paths[3].read_text())
    with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:manifest=effective_manifest(manifest,db)
    segments=[];by_id={}
    for row in manifest:
        first=bisect_left(times,row['start_s']);last=bisect_left(times,row['end_s'])-1
        if first>last:continue
        seg=dict(segment=str(row['segment_id']),first=first,last=last,start_s=row['start_s'],end_s=row['end_s'],
                 total=last-first+1,computed=0,issues=0,unverified=0,spans=[])
        segments.append(seg);by_id[seg['segment']]=seg
    with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        db.execute('BEGIN')
        for segment,frame,approved,problems in db.execute("SELECT segment,frame,approved,json_extract(payload,'$.problems') FROM records ORDER BY segment,frame"):
            seg=by_id.get(str(segment))
            if not seg or not seg['first']<=frame<=seg['last']:continue
            status=annotation_status(json.loads(problems or '[]'),approved)
            seg['computed']+=1;seg['issues']+=status=='issue';seg['unverified']+=status=='unverified'
            append_span(seg['spans'],frame,status)
    for seg in segments:
        for span in seg['spans']:span.update(start_s=times[span['start']],end_s=times[span['end']])
    result=dict(segments=segments,total=sum(s['total'] for s in segments),computed=sum(s['computed'] for s in segments),
                issues=sum(s['issues'] for s in segments),unverified=sum(s['unverified'] for s in segments))
    # Keep one current cache entry per DB; stale snapshots cannot hide new saves.
    for old in list(_cache):
        if old[0][0]==str(dbpath):del _cache[old]
    if len(_cache)>=80:_cache.pop(next(iter(_cache)))
    _cache[stamp]=result
    return copy.deepcopy(result)


def subject_directory(manager):
    catalog=manager.catalog();recordings=[];seen=set()
    def add(entry,project=None):
        pid=project['id'] if project else None
        if pid and pid in seen:return
        if pid:seen.add(pid)
        old=bool(entry.get('annotation_history',{}).get('old_aoi_present'))
        row=dict(subject=entry.get('subject',project['subject'] if project else None),project_id=pid,import_key=entry.get('key'),
                 label=entry.get('name') or (project.get('label') if project else '') or f"{entry.get('subject')}号",
                 old_aoi_present=old,available=entry.get('available',True),segments=[],total=0,computed=0,issues=0,unverified=0)
        try:
            if project:row.update(project_progress(project))
            else:
                for m in entry.get('manifest',[]):
                    row['segments'].append(dict(segment=str(m['segment_id']),start_s=m['start_s'],end_s=m['end_s'],
                        total=0,computed=0,issues=0,unverified=0,spans=[]))
        except (OSError,ValueError,KeyError,sqlite3.Error) as exc:row['error']='进度暂不可读：'+str(exc)
        recordings.append(row)
    for entry in catalog['videos']:
        if entry['subject'] not in manager.eligible:continue
        p=manager.index.get(entry.get('project_id'))
        add(entry,p if p and p['kind']=='标注' else None)
    for p in manager.index.values():
        if p['subject'] in manager.eligible and p['kind']=='标注' and p['id'] not in seen:add(dict(subject=p['subject']),p)
    subjects=[]
    for subject in sorted(manager.eligible):
        rows=[r for r in recordings if r['subject']==subject]
        duration=sum(max(0,s['end_s']-s['start_s']) for r in rows for s in r['segments'])
        covered=sum(max(0,s['end_s']-s['start_s'])*s['computed']/s['total'] for r in rows for s in r['segments'] if s['total'])
        combined=[]
        for segment in [f'{n}.{half}' for n in range(1,7) for half in (1,2)]:
            parts=[dict(recording_index=i,**seg) for i,r in enumerate(rows) for seg in r['segments'] if seg['segment']==segment]
            combined.append(dict(segment=segment,parts=parts,missing=not parts))
        complete=all(seg['total'] and seg['computed']>=seg['total'] for r in rows for seg in r['segments'])
        percent=(100 if complete else min(99,int(100*covered/duration))) if duration else 0
        subjects.append(dict(subject=subject,recordings=rows,segments=combined,known_segments=sum(not s['missing'] for s in combined),percent=percent,
                             issues=sum(r['issues'] for r in rows),unverified=sum(r['unverified'] for r in rows)))
    return dict(subjects=subjects,current=manager.current)
