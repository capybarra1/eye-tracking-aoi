"""Background review previews and cross-project issue navigation."""
from __future__ import annotations
import bisect
import copy
import json
import sqlite3
import threading
from types import SimpleNamespace
from pathlib import Path
import cv2
from scripts.aoi_slice_timing import effective_manifest
from scripts.aoi_range_review import prepare_review,generate_review,commit_review,issue_ranges


class ReviewWorker:
    def __init__(self,server):
        self.server=server;self.draft=None;self.stop=threading.Event();self.thread=None
    @property
    def active(self):return bool(self.draft and self.draft['status']=='computing')
    def snapshot(self):
        return {k:v for k,v in self.draft.items() if k in ('id','segment','start','end','status','progress','uncertain','error')} if self.draft else dict(status='idle')
    def start(self,data):
        if self.active:raise ValueError('正在生成预览，请等待或取消')
        s=self.server.session;s.pause()
        self.draft=prepare_review(s,data['start'],data['end'],data['head'],data['tail'],method=data.get('method','keyframes'));self.stop=threading.Event();cancel=self.stop
        source=copy.copy(s.source)
        decode=self.draft['method']!='keyframes' and hasattr(source,'cap')
        if decode:
            source.cap=cv2.VideoCapture(str(s.source.path));source.last=-2;source.cached=None
        detached=SimpleNamespace(source=source,factory=s.factory)
        draft=self.draft
        def work():
            try:generate_review(detached,draft,cancel=cancel.is_set)
            except InterruptedError as exc:draft.update(status='cancelled',error=str(exc))
            except Exception as exc:draft.update(status='error',error=str(exc))
            finally:
                if decode:source.close()
        self.thread=threading.Thread(target=work,daemon=True,name='aoi-review');self.thread.start()
    def overlay(self,s,state):
        d=self.draft
        if d and d['status']=='ready' and d['folder']==str(s.folder.resolve()) and d['segment']==s.segment:
            r=d['records'].get(s.cursor)
            if r:
                from scripts.supervised_aoi import four_points,coverage_for
                state.update(polygons=r['polygons'],visible=r['visible'],screen_points=four_points(r['polygons']['screen'],s.source.size),
                             coverage=coverage_for(r['polygons']['screen'],s.source.size) if r['visible']['screen'] else [],
                             reason='复核预览，尚未写入正式标注',preview=True)
    def commit(self):
        if not self.draft:raise ValueError('没有可保存的预览')
        s=self.server.session;failed=s._get('auto_retry_failed')
        commit_review(s,self.draft)
        if failed and failed.get('segment')==s.segment and self.draft['start']<=failed['end'] and self.draft['end']>=failed['start']:
            s._set('auto_retry_failed',None);s._save()
            batch=getattr(self.server,'batch',None)
            if batch and not batch.active:
                items=batch.state.get('items',[]);position=batch.state.get('position',0)
                current=items[position] if position<len(items) else {}
                if current.get('project_id')==getattr(self.server.projects,'current',None):
                    batch.state.pop('needs_attention',None);batch.state.pop('error',None);batch.persist()
    def cancel(self):
        self.stop.set()
        self.draft=None


def all_issues(manager):
    items=[];unavailable=[]
    for p in manager.index.values():
        if p['kind']!='标注' or p['subject'] not in manager.eligible:continue
        try:
            times=json.loads(Path(p['timestamps']).read_text());rows=json.loads(Path(p['manifest']).read_text());fps=1/(times[1]-times[0])
            slices={str(r['segment_id']):dict(first=bisect.bisect_left(times,r['start_s']),last=bisect.bisect_left(times,r['end_s'])-1) for r in rows}
            db=sqlite3.connect('file:'+str(Path(p['folder'])/'session.sqlite3')+'?mode=ro',uri=True)
            try:
                rows=effective_manifest(rows,db)
                slices={str(r['segment_id']):dict(first=bisect.bisect_left(times,r['start_s']),last=bisect.bisect_left(times,r['end_s'])-1) for r in rows}
                ranges=issue_ranges(db,slices,fps)
            finally:db.close()
            for r in ranges:items.append({**r,'project_id':p['id'],'subject':p['subject'],'label':p.get('label') or f"{p['subject']}号",'start_s':times[r['start']],'end_s':times[r['end']]})
        except (OSError,ValueError,sqlite3.Error) as exc:unavailable.append(dict(project_id=p['id'],error=str(exc)))
    return dict(items=items,unavailable=unavailable)
