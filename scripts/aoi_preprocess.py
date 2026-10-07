"""Prepare reusable screen features for the existing eligible, unfinished queue.

No Session is opened and no annotation database, source video or queue is edited.
Restart this command to resume. Ctrl+C or a STOP file pauses after the current frame.
"""
from __future__ import annotations
import argparse,bisect,fcntl,hashlib,json,os,signal,sqlite3,time
from pathlib import Path
import cv2
from scripts.aoi_slice_timing import effective_manifest
from scripts.aoi_feature_cache import FeatureWriter,VERSION
from scripts.fast_aoi_tracker import prepare_gray

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)

def jobs_for_queue(root,mode="legacy",current_project=None):
 version=VERSION
 if mode=="line":
  from scripts.aoi_line_cache import VERSION as version
 registry=root/'outputs/aoi_local_projects';index=json.loads((registry/'projects.json').read_text())
 queue=json.loads((registry/'batch_state.json').read_text())['items']
 scope=json.loads((registry/'batch_scope.json').read_text()) if (registry/'batch_scope.json').exists() else {}
 eligible={r['subject_id'] for r in json.loads((root/'outputs/aoi_workflow_20260928/timing_v3/eligible_subject_inventory.json').read_text()) if r['eligible']}
 jobs=[]
 # A user can open/import a recording before the batch queue has assigned its
 # project id. Resolve existing annotation projects by subject + video first.
 resolved=[]
 for original in queue:
  item=dict(original)
  if item.get('project_id') not in index and item.get('entry',{}).get('path'):
   video=str(Path(item['entry']['path']).resolve())
   matches=[key for key,p in index.items() if p.get('kind','标注')=='标注' and str(Path(p['video']).resolve())==video and (p.get('subject',item['subject'])==item['subject'])]
   if matches:item['project_id']=current_project if current_project in matches else matches[0]
  resolved.append(item)
 queue=resolved
 position=next((i for i,item in enumerate(queue) if item.get('project_id')==current_project),0) if current_project else 0
 queue=queue[position:]+queue[:position]
 for item in queue:
  if item['subject'] not in eligible or item['subject'] in scope.get('excluded_subjects',[]) or item['status'] in ('complete','skipped'):continue
  p=index.get(item.get('project_id'));entry=item.get('entry',{})
  video=Path(p['video'] if p else entry['path'])
  manifest=json.loads(Path(p['manifest']).read_text()) if p else entry['manifest']
  times=json.loads(Path(p['timestamps']).read_text()) if p else None
  dbpath=Path(p['folder'])/'session.sqlite3' if p else None
  rows={};selected=None;cursor=-1;local=[]
  if dbpath and dbpath.exists():
   with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
    manifest=effective_manifest(manifest,db)
    if mode=='line':
     anchors={frame for frame, in db.execute('SELECT frame FROM anchors')}
     for segment,frame,payload in db.execute('SELECT segment,frame,payload FROM records ORDER BY segment,frame'):
      if frame not in anchors and not json.loads(payload).get('problems'):rows.setdefault(segment,set()).add(frame)
    else:
     for segment,frame in db.execute('SELECT segment,frame FROM records ORDER BY segment,frame'):rows.setdefault(segment,set()).add(frame)
    saved=db.execute("SELECT value FROM kv WHERE key='selected'").fetchone()
    selected=json.loads(saved[0]) if saved else None
    saved=db.execute('SELECT value FROM kv WHERE key=?',('state:'+str(selected),)).fetchone()
    cursor=json.loads(saved[0]).get('cursor',-1) if saved else -1
  try:st=video.stat();stamp=(st.st_size,st.st_mtime_ns)
  except OSError:stamp=None
  for segment in manifest:
   start,end=segment['start_s'],segment['end_s']
   ranges=[(start,end)]
   if times:
    first,last=bisect.bisect_left(times,start),bisect.bisect_left(times,end)
    present=rows.get(segment['segment_id'],());ranges=[];i=first
    while i<last:
     if i in present:i+=1;continue
     begin=i
     while i<last and i not in present:i+=1
     ranges.append((max(start,times[max(first,begin-2)]),min(end,times[i]) if i<len(times) else end))
   if mode=='line':
    merged=[]
    for a,b in ranges:
     if merged and a-merged[-1][1]<=1.:merged[-1]=(merged[-1][0],b)
     else:merged.append((a,b))
    ranges=merged
   for start,end in ranges:
    identity=[str(video),stamp,segment['segment_id'],start,end,version]
    local.append(dict(id=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),video=str(video),stat=stamp,
                     project_id=item.get('project_id'),subject=item['subject'],segment=segment['segment_id'],start=start,end=end))
  if times and 0<=cursor<len(times):local.sort(key=lambda j:(j['end']<=times[cursor],j['start']))
  jobs.extend(local)
 unique={}
 for job in jobs:unique.setdefault(job['id'],job)
 return list(unique.values())

def main():
 ap=argparse.ArgumentParser(description='预处理剩余绿色被试的切片，Ctrl+C暂停，再运行续做')
 ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--cache',type=Path,required=True)
 ap.add_argument('--mode',choices=['legacy','line'],default='legacy');ap.add_argument('--current-project');ap.add_argument('--recordings',type=int,default=2)
 ap.add_argument('--dry-run',action='store_true');ap.add_argument('--limit-frames',type=int);ap.add_argument('--max-gb',type=float,default=None)
 args=ap.parse_args();args.max_gb=args.max_gb if args.max_gb is not None else 12 if args.mode=='line' else 48
 if not 0<args.max_gb<=256:ap.error('缓存预算须在 0～256 GiB 之间')
 version=VERSION
 if args.mode=='line':
  from scripts.aoi_line_cache import LineWriter,scene_gray,VERSION as version
 jobs=jobs_for_queue(args.root.resolve(),args.mode,args.current_project)
 if args.mode=='line':
  if not 1<=args.recordings<=4:ap.error('滚动预处理范围为1～4份录像')
  videos=list(dict.fromkeys(j['video'] for j in jobs))[:args.recordings]
  jobs=[j for j in jobs if j['video'] in videos]
 folder=args.cache.expanduser().resolve().parent
 plan=dict(jobs=jobs,segments=len(jobs),recordings=len({j['video'] for j in jobs}),seconds=sum(j['end']-j['start'] for j in jobs))
 if args.dry_run:print(json.dumps(plan,ensure_ascii=False,indent=2));return
 folder.mkdir(parents=True,exist_ok=True)
 with (folder/'preprocess.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('预处理已经在运行，无需重复打开')
  cv2.setNumThreads(2)
  try:os.nice(10)
  except OSError:pass
  (folder/'STOP').unlink(missing_ok=True)
  progress=folder/'progress.json'
  state=json.loads(progress.read_text()) if progress.exists() else {}
  if state.get('version')!=version:state={}
  state.pop('error',None)
  state.update(version=version,mode=args.mode,current_project=args.current_project,max_gb=args.max_gb,status='running',pid=os.getpid(),segments=len(jobs),recordings=plan['recordings'],jobs=state.get('jobs',{}),started_at=time.time())
  write_json(folder/'plan.json',plan);write_json(progress,state)
  writer=None;count=0;cap=None
  def stop(*_):raise KeyboardInterrupt
  signal.signal(signal.SIGTERM,stop)
  try:
   writer=(LineWriter if args.mode=='line' else FeatureWriter)(args.cache,max_bytes=int(args.max_gb*1024**3))
   if args.mode=='line':
    # This database contains only reconstructible v1 line descriptors. Keep
    # the active two-recording window, reclaim other cached recordings only.
    active={str(Path(j['video']).resolve()) for j in jobs}
    owners={row[0] for row in writer.db.execute('SELECT DISTINCT owner FROM features')}
    writer.retire_owners(owners-active)
    # Progress from an evicted window must never masquerade as a warm cache.
    retained=set(state.get('retained_owners',[])) & owners
    for job in jobs:
     if str(Path(job['video']).resolve()) not in retained:state['jobs'].pop(job['id'],None)
    state['retained_owners']=sorted(active)
    write_json(progress,state)
   for job in jobs:
    record=state['jobs'].setdefault(job['id'],dict(status='pending',frames=0))
    if record['status']=='complete':continue
    state.update(subject=job['subject'],segment=job['segment'],video=job['video'],job_start=job['start'],job_end=job['end'],completed_jobs=sum(state['jobs'].get(j['id'],{}).get('status')=='complete' for j in jobs));record['status']='running'
    st=Path(job['video']).stat()
    if list(job['stat'] or ())!=[st.st_size,st.st_mtime_ns]:raise ValueError('视频已变化，请重新运行以更新预处理计划')
    cap=cv2.VideoCapture(job['video'])
    if not cap.isOpened():raise OSError('无法打开视频：'+job['video'])
    size=(960,round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*960/cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    start=max(job['start'],record.get('last_pts',job['start']));cap.set(cv2.CAP_PROP_POS_MSEC,max(0,start-.25)*1000)
    reached=False
    while True:
     if (folder/'STOP').exists():raise KeyboardInterrupt
     ok,image=cap.read()
     if not ok:break
     pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
     if pts<start-1e-6:continue
     if pts>=job['end']:reached=True;break
     resized=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
     added=writer.prepare(scene_gray(resized),owner=str(Path(job['video']).resolve())) if args.mode=='line' else writer.prepare(prepare_gray(resized))
     count+=1;record['frames']+=int(added);record['last_pts']=pts
     state.update(position_s=pts,frames_this_run=count,updated_at=time.time())
     if count%50==0:
      write_json(progress,state)
      print(f"{job['subject']}号 {job['segment']} · {pts:.1f}/{job['end']:.1f}秒 · 本次{count}帧",flush=True)
     if args.limit_frames and count>=args.limit_frames:raise KeyboardInterrupt
    cap.release();cap=None
    if not reached and record.get('last_pts',0)<job['end']-.2:raise OSError('视频提前结束，未把未处理部分标为完成')
    record['status']='complete';write_json(progress,state)
   state['status']='complete';state['completed_jobs']=len(jobs)
  except KeyboardInterrupt:state['status']='paused'
  except Exception as exc:state.update(status='space_paused' if args.mode=='line' and ('缓存预算' in str(exc) or '硬盘剩余' in str(exc)) else 'error',error=str(exc));print(str(exc),flush=True)
  finally:
   if cap is not None:cap.release()
   if writer is not None:writer.close()
   state['updated_at']=time.time();write_json(progress,state)
   print('预处理状态：'+state['status'],flush=True)

if __name__=='__main__':main()
