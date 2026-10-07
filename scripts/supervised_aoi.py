"""Durable, human-supervised slice tracking. No autonomous background worker."""
from __future__ import annotations

import bisect
import copy
import csv
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np

from scripts.aoi_image_adjustments import make_tracker, display_frame, validate as validate_image_adjustments
from scripts.aoi_progress import annotation_status
from scripts.aoi_horizontal import HorizontalTracker, partition_polygons, center_height, center_slope
from scripts.full_aoi_video import PilotState
from scripts.screen_structure_tracker import screen_coverage
from scripts.tablet_recovery import TabletRecovery, visible_fraction
from scripts.blur_recovery import BlurGate, ScreenRecovery
from scripts.screen_lower_edge import LowerEdgeRecovery
from scripts.fast_aoi_tracker import prepare_gray, reseed_region, enhance_frame_for_detection
from scripts.aoi_geometry import convex_clip, convex_difference, halfplane
from scripts.aoi_detector_rescue import DetectorRescue


def four_points(screens: list, size: tuple) -> list:
    if len(screens)==1:
        points=copy.deepcopy(screens[0])
        if len(points)==3:
            # Insert a collinear point, preserving existing three-point geometry.
            i=max(range(2),key=lambda n:points[n+1][0]-points[n][0])
            points.insert(i+1,[(a+b)/2 for a,b in zip(points[i],points[i+1])])
        return lock_screen_edges(points,size)
    old=screen_coverage([np.array(p,np.float32) for p in screens],size)
    return [old[0][3].tolist(),old[0][2].tolist(),old[1][2].tolist(),old[-1][2].tolist()]


def lock_screen_edges(points: list, size: tuple) -> list:
    """Keep edge intersections fixed in x, preserving the visible lower lines."""
    boundary=coverage_for([points],size)[0]
    # Camera motion can carry a join outside the visible image. Keep the
    # visible polyline and replace offscreen handles with collinear handles.
    # A two-pixel edge margin avoids nearly coincident controls.
    locked=[boundary[-1],*[copy.deepcopy(p) for p in points[1:-1]
                          if 2 <= p[0] <= size[0]-3],boundary[2]]
    while len(locked)<len(points):
        i=max(range(len(locked)-1),key=lambda n:locked[n+1][0]-locked[n][0])
        locked.insert(i+1,[(a+b)/2 for a,b in zip(locked[i],locked[i+1])])
    coverage_for([locked],size)
    return locked


def coverage_for(screens: list, size: tuple) -> list:
    if len(screens)!=1:
        return [p.tolist() for p in screen_coverage([np.array(p,np.float32) for p in screens],size)]
    p=np.asarray(screens[0],np.float32)
    if p.shape not in ((3,2),(4,2)) or not np.isfinite(p).all() or np.any(np.diff(p[:,0])<2):
        raise ValueError('屏幕下边缘点需从左到右排列，不能重叠或交叉')
    width,height=size
    def edge(a,b,x):return float(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]))
    left,right=edge(p[0],p[1],0),edge(p[-2],p[-1],width-1)
    if max(abs(left),abs(right))>height*8:raise ValueError('屏幕下边缘斜率过大，请调整四个点')
    # User-defined lower polyline, extended to both image sides and filled upwards.
    interior=[v.tolist() for v in p if 0<float(v[0])<width-1]
    return [[[0.,0.],[float(width-1),0.],[float(width-1),right],*reversed(interior),[0.,left]]]


def tracker_screens(points: list, size: tuple) -> list:
    coverage_for([points],size)
    p=np.asarray(points,np.float32);w,_=size
    if len(p)==4:
        first=float(np.clip(p[1,0],2,w-5))
        second=float(np.clip(p[2,0],first+2,w-3))
        xs=[0,first,second,w-1]
    else:
        middle=float(np.clip(p[1,0],2,w-3))
        xs=[0,middle/2,middle,w-1]
    def y(x):
        i=int(np.clip(np.searchsorted(p[:,0],x)-1,0,len(p)-2))
        a,b=p[i],p[i+1]
        return float(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]))
    return [np.array([[a,0],[b,0],[b,y(b)],[a,y(a)]],np.float32) for a,b in zip(xs,xs[1:])]


def effective_aois(polygons: dict, visible: dict, size: tuple) -> dict:
    """On-frame tablet takes precedence over screen; neither area overlaps."""
    w,h=size;frame=[[0.,0.],[w-1.,0.],[w-1.,h-1.],[0.,h-1.]]
    tablets=[q for p in polygons['tablet'] if (q:=convex_clip(p,frame))] if visible['tablet'] else []
    screens=[]
    if visible['screen']:
        if len(polygons['screen'])==1:
            raw=[]
            for p in tracker_screens(polygons['screen'][0],size):
                left,right=p[0,0],p[1,0]
                q=halfplane([[left,0],[right,0],[right,h-1],[left,h-1]],p[2],p[3],1)
                if q:raw.append(q)
        else:raw=coverage_for(polygons['screen'],size)
        for p in raw:
            q=convex_clip(p,frame)
            parts=[q] if q else []
            for tablet in tablets:parts=[part for region in parts for part in convex_difference(region,tablet)]
            screens.extend(parts)
    return dict(screen=screens,tablet=tablets)


class VideoSource:
    def __init__(self, path: Path, timestamps: Path):
        self.path=path.resolve()
        stat=path.stat()
        self.identity=f'{self.path}:{stat.st_size}:{stat.st_mtime_ns}'
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened(): raise ValueError('无法打开录像')
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.size=(960,round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*960/self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.times=json.loads(timestamps.read_text())
        if not self.times or not all(np.isfinite(self.times)) or not all(b>a for a,b in zip(self.times,self.times[1:])):
            raise ValueError('时间戳索引必须严格递增')
        self.identity+=':'+hashlib.sha256(timestamps.read_bytes()).hexdigest()
        self.last=-2
        self.cached=None

    def frame(self, index: int) -> np.ndarray:
        if index==self.last: return self.cached.copy()
        if not 0<=index<len(self.times): raise ValueError('帧号越界')
        target=self.times[index]
        # Frame seeking uses average FPS in OpenCV and drifts on variable-rate
        # recordings. Seek by PTS, decode forward, and verify the actual timestamp.
        seeks=([None] if index==self.last+1 else [])+[max(0,target-r) for r in (.25,2,30,120)]+[0]
        for seek in seeks:
            if seek is not None:self.cap.set(cv2.CAP_PROP_POS_MSEC,seek*1000)
            while True:
                ok,image=self.cap.read()
                if not ok:break
                pts=self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if pts<target-.0001:continue
                if abs(pts-target)<=.0001:
                    self.last=index
                    self.cached=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
                    return self.cached.copy()
                break
        raise ValueError(f'第 {index} 帧解码或时间戳不匹配，已暂停')

    def close(self): self.cap.release()


class Tracker:
    # Opt-in only. The server may explicitly configure a shared thread-safe
    # detector; normal imports and tests never load an ML runtime or weights.
    default_detector=None
    default_detector_parallel=False

    def __new__(cls, frame, polygons, visible, *args, **kwargs):
        if polygons.get("partition",{}).get("mode")=="straight_line":
            return HorizontalTracker(frame,polygons,visible)
        return super().__new__(cls)

    def __init__(self, frame: np.ndarray, polygons: dict, visible: dict, detector=None, detector_parallel=None):
        self.visible=visible
        seed={k:[np.array(p,np.float32) for p in v] for k,v in polygons.items()}
        self.screen_points=np.array(polygons['screen'][0],np.float32) if len(polygons['screen'])==1 else None
        if self.screen_points is not None:seed['screen']=tracker_screens(self.screen_points.tolist(),(frame.shape[1],frame.shape[0]))
        if not visible['tablet']: seed['tablet']=[]
        self.state=PilotState(frame,seed,tablet_padding=0.)
        self.tablet_tracker=TabletRecovery(frame,seed['tablet'][0]) if seed['tablet'] else None
        self.state.tablet_tracker=self.tablet_tracker
        if self.tablet_tracker is not None:
            self.tablet_tracker.candidate_filter=self._tablet_candidate
        if self.screen_points is not None:
            self.state.screen_projector=self._project_screen
        self.polygons=copy.deepcopy(polygons)
        self.blur_gate=BlurGate(frame)
        self.screen_recovery=ScreenRecovery() if self.screen_points is not None else None
        self.lower_edges=LowerEdgeRecovery(self.state.size,lock_screen_edges) if self.screen_points is not None else None
        self.clear_steps=0
        configured=self.default_detector if detector is None else detector
        parallel=self.default_detector_parallel if detector_parallel is None else detector_parallel
        self.detector_rescue=DetectorRescue(configured,self.state.size,parallel=parallel) if configured else None
        self.rescue_counts=dict(screen=0,tablet=0)
        self.last_rescue=None
        self.last_stats={}
        if self.screen_recovery is not None:
            self.screen_recovery.remember(self.state.gray,self.state.H,self.screen_points)

    def add_tablet_reference(self, frame, polygon):
        if self.tablet_tracker is not None:self.tablet_tracker.add_reference(frame,polygon)

    def cancel_prefetch(self):
        if self.detector_rescue is not None:self.detector_rescue.discard_prefetch()

    def close(self):
        if self.detector_rescue is not None:self.detector_rescue.close()

    def _tablet_candidate(self, polygon):
        if not self.visible['screen']:return True
        # Reject a match mainly on the driving display. Small foreground
        # occlusion is allowed and removed from the effective screen region.
        w,h=self.state.size
        p=convex_clip(polygon,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        if not p:return True
        area=abs(cv2.contourArea(np.float32(p)))
        overlap=sum(abs(cv2.contourArea(np.float32(q))) for s in self.state.screens if (q:=convex_clip(p,s)))
        return overlap<=area*.5

    def _project_screen(self, transform):
        moved=cv2.perspectiveTransform(self.screen_points[None],transform)[0].tolist()
        points=lock_screen_edges(moved,self.state.size)
        return tracker_screens(points,self.state.size),[np.array(p,np.float32) for p in coverage_for([points],self.state.size)]

    def _apply_lower_boundary(self, points, frame):
        # A direct boundary observation replaces the motion-derived geometry.
        # Rebase only the screen; tablet history and manual references survive.
        self.screen_points=np.float32(lock_screen_edges(points.tolist(),self.state.size))
        self.state.H=np.eye(3,dtype=np.float32)
        self.state.seed=tracker_screens(self.screen_points.tolist(),self.state.size)
        self.state.screens=[p.copy() for p in self.state.seed]
        self.state.coverage=[np.float32(p) for p in coverage_for([self.screen_points.tolist()],self.state.size)]
        self.state.gray=prepare_gray(frame);self.state.screen_lost=False
        self.lower_edges.remember_flow(frame,self.screen_points)
        self.screen_recovery=ScreenRecovery()
        self.screen_recovery.remember(self.state.gray,self.state.H,self.screen_points)

    def _learned_rescue(self, frame, screen, tablet):
        if self.detector_rescue is None:return screen,tablet
        if not self.visible['tablet']:
            self.detector_rescue.clear_tablet()
        needed=dict(screen=False,
                    tablet=self.visible['tablet'] and self.tablet_tracker is not None
                    and (tablet['status'] not in ('tracked','reacquired')
                         or self.detector_rescue.tablet_requires_redetection))
        accepted=self.detector_rescue.observe(frame,needed)
        used=[]
        if 'tablet' in accepted:
            points=accepted['tablet']
            # A learned navigation-screen false positive must not bypass the
            # same screen/tablet spatial restriction used by optical flow.
            # The traditional screen boundary is never replaced by this path.
            if not self._tablet_candidate(points):
                self.detector_rescue.pending['tablet']=None
                self.detector_rescue.cooldown['tablet']=2
                accepted.pop('tablet')
        if 'tablet' in accepted:
            points=accepted['tablet']
            recovery=self.tablet_tracker
            recovery.polygon=points.copy()
            recovery.gray=prepare_gray(frame)
            recovery.region.polygon=points.copy()
            recovery.region.points=None
            reseed_region(recovery.gray,recovery.region,self.state.size)
            recovery.pending=None
            recovery.searching=recovery.waiting=recovery.parked=False
            recovery.region.status='reacquired'
            recovery.quality=dict(method='learned_rescue',recovered=True,candidate_only=True)
            # Keep the retained appearance bank; these are candidate coordinates.
            region=self.state.tablets[0]
            region.polygon=points.copy()
            region.points=None if recovery.region.points is None else recovery.region.points.copy()
            region.status='reacquired'
            self.state.tablet_gray=prepare_gray(enhance_frame_for_detection(frame,1.45,34.))
            self.state.tablet_lost=False
            tablet=dict(status='reacquired',method='learned_rescue',recovered=True,candidate_only=True)
            self.detector_rescue.tablet_applied()
            recovery.allow_reference_update=not self.detector_rescue.tablet_requires_redetection
            tablet['requires_redetection']=self.detector_rescue.tablet_requires_redetection
            used.append('tablet')
        if used:
            for key in used:self.rescue_counts[key]+=1
            self.last_rescue=dict(step=self.state.steps,aois=used,**self.detector_rescue.audit())
        return screen,tablet

    def update(self, frame: np.ndarray) -> tuple[dict,list[str]]:
        try:
            return self._update(frame)
        finally:
            # An exception or early return cannot retain speculative work.
            self.cancel_prefetch()

    def _update(self, frame: np.ndarray) -> tuple[dict,list[str]]:
        before=self.polygons
        if self.detector_rescue is not None and not self.visible['tablet']:
            self.detector_rescue.clear_tablet()
        if self.detector_rescue is not None and self.tablet_tracker is not None:
            self.tablet_tracker.allow_reference_update=not self.detector_rescue.tablet_requires_redetection
        if self.blur_gate.blurred(frame):
            if self.screen_recovery is not None:self.screen_recovery.pending=None
            if self.lower_edges is not None:self.lower_edges.pending=None
            if self.detector_rescue is not None:self.detector_rescue.interrupt()
            self.last_stats={key:dict(status='blurred') for key in ('screen','tablet')}
            # A blurred/occluded foreground can lower global sharpness while
            # physical structures still provide strong camera-motion evidence.
            # Bridge only the screen; retain tablet and sharp recovery anchors.
            if self.visible['screen'] and self.screen_points is not None:
                gray=prepare_gray(frame)
                motion=self.state.update_screen(gray,require_strong=True)
                if motion['status']=='tracked':
                    self.state.gray=gray
                    before=copy.deepcopy(before)
                    moved=cv2.perspectiveTransform(self.screen_points[None],self.state.H)[0]
                    before['screen']=[lock_screen_edges(moved.tolist(),self.state.size)]
                    self.polygons=before
                    self.last_stats['screen']={**motion,'status':'blurred',
                                               'motion_supported':True,'candidate_only':True}
            problems=[label+'运动模糊，等待恢复' for aoi,label in [('screen','屏幕'),('tablet','平板')] if self.visible[aoi]]
            if self.detector_rescue is not None and self.detector_rescue.tablet_requires_redetection:
                problems.append('平板仅恢复可见部分，待完整重识别')
            return copy.deepcopy(before),problems
        if self.detector_rescue is not None and self.visible['tablet']:
            likely_needed=(self.detector_rescue.tablet_requires_redetection
                           or self.last_stats.get('tablet',{}).get('status') in ('lost','waiting'))
            self.detector_rescue.prefetch(frame,dict(tablet=likely_needed))
        reference=self.state.gray
        previous_H=self.state.H.copy()
        screen,tablet=self.state.update(frame)
        if screen['status']!='tracked':
            # Retry against the last reliable screen image, so recovery includes
            # motion during the missing frames instead of silently losing it.
            self.state.gray=reference
            if self.screen_recovery is not None and self.visible['screen']:
                gray=prepare_gray(frame)
                def project(H):
                    return lock_screen_edges(cv2.perspectiveTransform(self.screen_points[None],H)[0].tolist(),self.state.size)
                found=self.screen_recovery.locate(gray,project)
                if found is not None:
                    self.state.H,screen=found
                    self.state.screens,self.state.coverage=self._project_screen(self.state.H)
                    self.state.gray=gray
                    # Camera motion supplies only an absent-tablet prediction,
                    # never a confirmed tablet label. Its appearance bank remains.
                    if self.tablet_tracker is not None and tablet['status']=='lost':
                        motion=self.state.H@np.linalg.inv(previous_H)
                        predicted=cv2.perspectiveTransform(np.array(before['tablet'],np.float32),motion)[0]
                        if np.isfinite(predicted).all() and cv2.isContourConvex(predicted) and np.max(np.abs(predicted))<max(self.state.size)*6 and visible_fraction(predicted,self.state.size)<.15:
                            self.tablet_tracker.polygon=predicted
                            self.tablet_tracker.searching=True;self.tablet_tracker.waiting=True
                            self.state.tablets[0].polygon=predicted
                            tablet={**tablet,'status':'waiting','predicted_absent':True}
        if self.lower_edges is not None and self.visible['screen']:
            edge_follow=self.lower_edges.follow(frame,[r.polygon for r in self.state.tablets])
            if edge_follow is not None:
                points,screen=edge_follow
                self._apply_lower_boundary(points,frame)
            # Periodic independent checks also detect a confidently drifting
            # motion fit. A pending proposal is checked on the next clear frame.
            if screen['status']!='tracked' or self.state.steps%(15 if edge_follow is not None else 3)==0 or self.lower_edges.pending is not None:
                found=self.lower_edges.locate(frame,[r.polygon for r in self.state.tablets])
                if found is not None:
                    points,screen=found
                    self._apply_lower_boundary(points,frame)
        screen,tablet=self._learned_rescue(frame,screen,tablet)
        polygons={'screen':[p.tolist() for p in self.state.screens],
                  'tablet':[r.polygon.tolist() for r in self.state.tablets] if self.visible['tablet'] else before['tablet']}
        if self.screen_points is not None:
            moved=cv2.perspectiveTransform(self.screen_points[None],self.state.H)[0].tolist()
            try:polygons['screen']=[lock_screen_edges(moved,self.state.size)]
            except ValueError:
                polygons['screen']=before['screen']
                screen={**screen,'status':'geometry_failed'}
        if not self.visible['screen']: polygons['screen']=before['screen']
        problems=[]
        if self.detector_rescue is not None and self.detector_rescue.tablet_requires_redetection:
            problems.append('平板仅恢复可见部分，待完整重识别')
            tablet={**tablet,'requires_redetection':True,'geometry_scope':'visible_intersection_only'}
        try:coverage_for(polygons['screen'],self.state.size)
        except ValueError:problems.append('屏幕下边缘异常')
        for aoi,stats,label in [('screen',screen,'屏幕'),('tablet',tablet,'平板')]:
            if not self.visible[aoi]: continue
            if stats['status']=='lost': problems.append(label+'短暂失跟')
            elif stats['status']=='waiting': problems.append('平板离开画面，等待返回')
            elif stats['status'] not in ('tracked','reacquired'): problems.append(label+'几何异常')
            for old,new in zip(before[aoi],polygons[aoi]):
                old,new=np.array(old,np.float32),np.array(new,np.float32)
                ratio=1. if aoi=='screen' and self.screen_points is not None else abs(cv2.contourArea(new))/max(abs(cv2.contourArea(old)),1.)
                movement=np.max(np.linalg.norm(new-old,axis=1))
                if aoi=='screen' and self.screen_points is not None:
                    # Replacement edge handles can move far while the actual
                    # visible lower boundary remains almost unchanged.
                    xs=np.unique(np.concatenate([old[:,0],new[:,0]]))
                    movement=np.max(np.abs(np.interp(xs,new[:,0],new[:,1])-np.interp(xs,old[:,0],old[:,1])))
                supported=stats.get('confidence',0)>=.55 and stats.get('inliers',0)>=(30 if aoi=='screen' else 12)
                limit=max(self.state.size)*.25 if supported else 35
                parking=aoi=='tablet' and stats['status'] in ('lost','waiting') and visible_fraction(new,self.state.size)==0
                if not np.isfinite(new).all() or (not parking and stats['status']!='reacquired' and not stats.get('recovered') and not stats.get('predicted_absent') and (movement>limit or not .6<ratio<1.5)):
                    problems.append(label+'边界突变')
        self.polygons=polygons
        self.last_stats=dict(screen=screen.copy(),tablet=tablet.copy())
        if screen['status']=='tracked':
            if self.screen_recovery is not None:self.screen_recovery.pending=None
            self.blur_gate.accept(frame)
            self.clear_steps+=1
            if self.screen_recovery is not None and self.clear_steps%10==0 and not any('屏幕' in p for p in problems):
                self.screen_recovery.remember(self.state.gray,self.state.H,polygons['screen'][0])
        return polygons, sorted(set(problems))


def validate_polygons(polygons: dict, visible: dict, size: tuple) -> dict:
    if set(visible)!={'screen','tablet'} or any(type(v) is not bool for v in visible.values()):
        raise ValueError('请明确屏幕和平板是否可见')
    partition=polygons.get('partition')
    if partition is not None:
        if not isinstance(partition,dict) or partition.get('mode')!='straight_line':raise ValueError('未知分区方式')
        coverage_for(polygons['screen'],size)
        return partition_polygons(size,center_height(polygons,size),partition.get('gap_px',48.),center_slope(polygons,size))
    result={}
    for aoi,count in [('screen',3),('tablet',1)]:
        values=polygons.get(aoi,[])
        if aoi=='screen' and len(values)==1:
            coverage_for(values,size)
            result[aoi]=[lock_screen_edges(np.asarray(values[0],np.float32).tolist(),size)]
            continue
        if len(values)!=count: raise ValueError('需要三个屏幕轮廓和一个平板轮廓')
        result[aoi]=[]
        for value in values:
            p=np.asarray(value,np.float32)
            if p.shape!=(4,2) or not np.isfinite(p).all() or np.max(np.abs(p))>max(size)*8:
                raise ValueError('轮廓坐标无效')
            if not cv2.isContourConvex(p) or abs(cv2.contourArea(p))<4:
                raise ValueError('轮廓不能交叉或折叠，请按边缘顺序调整四角')
            result[aoi].append(p.tolist())
    if visible['screen']:coverage_for(result['screen'],size)
    return result


class Session:
    def __init__(self, manifest: list, folder: Path, source, template: dict,
                 tracker_factory=Tracker, checkpoint_s: float=10.):
        self.source=source
        self.template=template
        self.factory=tracker_factory
        self.checkpoint_s=checkpoint_s
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.base_manifest=copy.deepcopy(manifest)
        self.manifest=manifest
        self.slices={}
        subjects={r['subject_id'] for r in manifest}
        videos={r['video_path'] for r in manifest}
        if len(subjects)!=1 or len(videos)!=1:
            raise ValueError('一个工作区对应一个被试的一份录像；分录像请使用独立清单')
        for row in manifest:
            key=str(row['segment_id'])
            if key in self.slices: raise ValueError('切片编号重复')
            start,end=float(row['start_s']),float(row['end_s'])
            if not 0<=start<end<=source.times[-1]+1/source.fps+.2:
                raise ValueError('切片时间超出录像范围')
            first=bisect.bisect_left(source.times,start)
            last=bisect.bisect_left(source.times,end)-1
            if first>last: raise ValueError('切片没有可用帧')
            self.slices[key]={**row,'first':first,'last':last}
        if not self.slices: raise ValueError('切片清单为空')
        self.db=sqlite3.connect(self.folder/'session.sqlite3',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS records (segment TEXT, frame INTEGER, payload TEXT, approved INTEGER DEFAULT 0, PRIMARY KEY(segment,frame));
        CREATE TABLE IF NOT EXISTS anchors (segment TEXT, frame INTEGER, payload TEXT, PRIMARY KEY(segment,frame));
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time TEXT DEFAULT CURRENT_TIMESTAMP, segment TEXT, action TEXT, payload TEXT);''')
        signature=hashlib.sha256(json.dumps([manifest,source.identity,'supervised-v1'],sort_keys=True).encode()).hexdigest()
        old=self._get('identity')
        if old and old!=signature:
            self.db.close();raise ValueError('工作区对应的录像或清单已改变，请使用新工作区')
        self._set('identity',signature)
        self.source.image_adjustments=validate_image_adjustments(self._get('image_adjustments') or {})
        from scripts.aoi_slice_timing import effective_manifest
        self.manifest=effective_manifest(self.base_manifest,self.db)
        for row in self.manifest:
            self.slices[str(row['segment_id'])]={**row,'first':bisect.bisect_left(source.times,row['start_s']),'last':bisect.bisect_left(source.times,row['end_s'])-1}
        self.segment=self._get('selected') or next(iter(self.slices))
        self.running=False;self.token='';self.tracker=None;self.closed=False
        self._load_slice()

    def _get(self,key):
        row=self.db.execute('SELECT value FROM kv WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',(key,json.dumps(value)))

    def _event(self,action,payload):
        self.db.execute('INSERT INTO events(segment,action,payload) VALUES (?,?,?)',(self.segment,action,json.dumps(payload)))

    def _save(self):
        self._set('selected',self.segment)
        self._set('state:'+self.segment,dict(cursor=self.cursor,mode=self.mode,reason=self.reason,next_check=self.next_check))
        self.db.commit()

    def _load_slice(self):
        row=self.slices[self.segment]
        self.first_index,self.last_index=row['first'],row['last']
        saved=self._get('state:'+self.segment) or {}
        self.cursor=min(self.last_index,max(self.first_index,saved.get('cursor',self.first_index)))
        self.mode=saved.get('mode','uninitialized')
        self.reason=saved.get('reason','请检查并确认本切片的起始轮廓')
        if self.mode=='running': self.mode='paused';self.reason='已恢复保存的进度，等待继续'
        self.next_check=saved.get('next_check',self.source.times[self.cursor]+self.checkpoint_s)
        self.tracker=None;self.running=False
        self._save()

    def select(self, segment: str):
        if segment not in self.slices: raise ValueError('未知切片')
        self.pause();self.segment=segment;self._load_slice()

    def get_record(self,index):
        row=self.db.execute('SELECT payload,approved FROM records WHERE segment=? AND frame=?',(self.segment,index)).fetchone()
        if not row:return None
        return {**json.loads(row[0]),'approved':bool(row[1])}

    def get_anchor(self,index):
        row=self.db.execute('SELECT payload FROM anchors WHERE segment=? AND frame=?',(self.segment,index)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self,index,polygons,visible,problems,tracking=None):
        record=dict(frame_index=index,time_s=self.source.times[index],polygons=polygons,visible=visible,problems=problems)
        if tracking is not None:record['tracking']=tracking
        self.db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,0)',(self.segment,index,json.dumps(record)))

    def set_partition(self,enabled):
        """Change this seed only; all other frames and future anchors survive."""
        if type(enabled) is not bool:raise ValueError('分区开关无效')
        self.pause();snapshot=self.snapshot();original=snapshot['polygons']
        folder=self.folder/'backups';folder.mkdir(exist_ok=True)
        backup=folder/f'before-partition-{self.segment}-{self.cursor}-{uuid.uuid4().hex[:8]}.sqlite3'
        with sqlite3.connect(backup) as db:self.db.backup(db)
        if enabled:
            if not original.get('partition'):self._set('fine_partition_seed',original)
            boundary=four_points(original['screen'],self.source.size)
            polygons=partition_polygons(self.source.size,center_height({'screen':[boundary]},self.source.size),slope=center_slope({'screen':[boundary]},self.source.size))
        else:
            polygons=self._get('fine_partition_seed')
            if not polygons:raise ValueError('没有精细轮廓可恢复，请先打开原精细项目')
        self._set('horizontal_mode',enabled)
        if enabled:self._set('partition_gap',polygons['partition']['gap_px'])
        visible=dict(screen=True,tablet=True) if enabled else snapshot['visible']
        self._event('partition_mode',dict(enabled=enabled,frame=self.cursor,backup=str(backup)))
        self.db.execute('INSERT OR REPLACE INTO anchors VALUES (?,?,?)',
                        (self.segment,self.cursor,json.dumps(dict(polygons=polygons,visible=visible))))
        self._put(self.cursor,polygons,visible,[])
        self.tracker=None;self.mode='paused';self.reason='分区方式已切换；仅当前种子和后续新标注使用新方式，其他已存帧保持原样'
        self._save()

    def pause(self):
        if self.tracker is not None and hasattr(self.tracker,'cancel_prefetch'):
            self.tracker.cancel_prefetch()
        self.running=False;self.token=''
        if self.mode=='running':self.mode='paused';self.reason='已暂停计算并保存进度'
        self._save()

    def seek(self,index: int):
        self.pause()
        if not self.first_index<=index<=self.last_index:raise ValueError('不能跳出当前切片')
        self.cursor=index;self.tracker=None
        r=self.get_record(index)
        self.mode='issue' if r and annotation_status(r['problems'])=='issue' else 'paused' if r else 'uninitialized'
        self.reason='回看位置：继续可回放已有标注；从此处重新计算请点「从这里重标」'
        self._save()

    def correct(self,polygons: dict,visible: dict):
        self.pause()
        polygons=validate_polygons(polygons,visible,self.source.size)
        if polygons.get('partition'):self._set('partition_gap',polygons['partition']['gap_px'])
        index=self.cursor
        next_anchor=self.db.execute('SELECT MIN(frame) FROM anchors WHERE segment=? AND frame>?',(self.segment,index)).fetchone()[0]
        stop=next_anchor if next_anchor is not None else self.last_index+1
        with self.db:
            self._event('correction',dict(frame=index,old=self.get_record(index),polygons=polygons,visible=visible,invalidated_until=stop))
            self.db.execute('DELETE FROM records WHERE segment=? AND frame>=? AND frame<?',(self.segment,index,stop))
            # Later coordinates may survive an anchor, but review must be repeated.
            self.db.execute('UPDATE records SET approved=0 WHERE segment=? AND frame>=?',(self.segment,index))
            anchor=dict(polygons=polygons,visible=visible)
            self.db.execute('INSERT OR REPLACE INTO anchors VALUES (?,?,?)',(self.segment,index,json.dumps(anchor)))
            self._put(index,polygons,visible,[])
            self.mode='paused';self.reason='关键帧已保存，可以继续；请观察修正效果'
            self.next_check=self.source.times[index]+2.
            self.tracker=None;self._save()

    def continue_simple(self,polygons: dict|None=None,visible: dict|None=None):
        """One explicit continue action: save edits/reseed, then resume; never approve."""
        self.pause()
        record=self.get_record(self.cursor)
        if polygons is None and record and (len(record['polygons']['screen'])!=1 or len(record['polygons']['screen'][0])!=4):
            polygons={**record['polygons'],'screen':[four_points(record['polygons']['screen'],self.source.size)]}
            visible=record['visible']
        if polygons is not None:
            self.correct(polygons,visible)
        elif self.mode=='issue' and record:
            self.correct(record['polygons'],record['visible'])
        if self.mode=='checkpoint' or self.source.times[self.cursor]>=self.next_check:
            self.next_check=self.source.times[self.cursor]+self.checkpoint_s
            self.mode='paused';self._save()
        return self.start(require_review=False)

    def reset_from_here(self, polygons: dict, visible: dict):
        """Back up, then invalidate this slice from the chosen frame onward."""
        polygons=validate_polygons(polygons,visible,self.source.size)
        self.pause()
        folder=self.folder/'backups';folder.mkdir(exist_ok=True)
        backup=folder/f'before-reset-{self.segment}-{self.cursor}-{uuid.uuid4().hex[:8]}.sqlite3'
        with sqlite3.connect(backup) as destination:self.db.backup(destination)
        with self.db:
            self._event('reset_from_here',dict(frame=self.cursor,backup=str(backup)))
            self.db.execute('DELETE FROM records WHERE segment=? AND frame>=?',(self.segment,self.cursor))
            self.db.execute('DELETE FROM anchors WHERE segment=? AND frame>=?',(self.segment,self.cursor))
            self.db.execute('INSERT INTO anchors VALUES (?,?,?)',
                            (self.segment,self.cursor,json.dumps(dict(polygons=polygons,visible=visible))))
            self._put(self.cursor,polygons,visible,[])
            self.tracker=None;self.mode='paused';self.reason='已从当前帧重置，本切片后面的旧标注已备份；可继续重标'
            self.next_check=self.source.times[self.cursor]+self.checkpoint_s
            self._save()

    def _make_tracker(self, index, record):
        polygons=record['polygons']
        if self._get('horizontal_mode') is True and not polygons.get('partition'):
            boundary=four_points(polygons['screen'],self.source.size)
            polygons=partition_polygons(self.source.size,center_height({'screen':[boundary]},self.source.size),self._get('partition_gap') if self._get('partition_gap') is not None else None,slope=center_slope({'screen':[boundary]},self.source.size))
        if polygons.get('partition'):
            tracker=make_tracker(self.source,HorizontalTracker,index,polygons,record['visible'])
            # Cache at most four manually corrected scene views across pause/continue.
            # A generated frame must not become an absolute localization reference.
            if self.get_anchor(index) is None:tracker.scene.refs=[]
            if not hasattr(self,'line_reference_cache'):self.line_reference_cache={}
            count=0
            for frame,payload in self.db.execute('SELECT frame,payload FROM anchors WHERE frame<? ORDER BY frame DESC LIMIT 6',(index,)):
                anchor=json.loads(payload);old=anchor['polygons']
                if not old.get('partition') or abs(old['partition']['gap_px']-polygons['partition']['gap_px'])>3:continue
                key=(frame,payload)
                if key not in self.line_reference_cache:
                    reference=make_tracker(self.source,HorizontalTracker,frame,old,anchor['visible'])
                    self.line_reference_cache[key]=(reference.bank,reference.tablet_reference,reference.upper.reference,reference.scene.refs)
                    while len(self.line_reference_cache)>12:self.line_reference_cache.pop(next(iter(self.line_reference_cache)))
                bank,tablet_ref,upper_ref,scene_refs=self.line_reference_cache[key]
                tracker.scene.refs=(tracker.scene.refs+scene_refs)[-4:]
                tracker.bank=(tracker.bank+bank)[-3:]
                if tracker.tablet_reference is None and tablet_ref:tracker.tablet_reference=copy.deepcopy(tablet_ref)
                if tracker.upper.reference is None and upper_ref:tracker.upper.reference=copy.deepcopy(upper_ref)
                count+=1
                if count>=3:break
            return tracker
        tracker=make_tracker(self.source,self.factory,index,polygons,record['visible'])
        edge_recovery=getattr(tracker,'lower_edges',None)
        if edge_recovery is not None and record['visible']['screen']:
            # Same recording, past manual frames only. Cache descriptors across
            # pause/continue, so repeat corrections do not reprocess all images.
            if not hasattr(self,'screen_edge_references'):self.screen_edge_references={}
            selected=[];attempts=0
            for frame,payload in self.db.execute('SELECT frame,payload FROM anchors WHERE frame<=? ORDER BY frame DESC LIMIT 100',(index,)):
                if selected and any(abs(self.source.times[frame]-self.source.times[f])<1. for f in selected):continue
                anchor=json.loads(payload)
                if not anchor['visible']['screen'] or len(anchor['polygons']['screen'])!=1:continue
                key=(frame,payload);cache=self.screen_edge_references
                if key not in cache:
                    if attempts>=12:break
                    attempts+=1
                    cache[key]=edge_recovery.make_reference(display_frame(self.source,frame),anchor['polygons']['screen'][0],
                        anchor['polygons']['tablet'] if anchor['visible']['tablet'] else [])
                    while len(cache)>24:cache.pop(next(iter(cache)))
                ref=cache[key]
                if ref is not None:edge_recovery.add(ref);selected.append(frame)
                if len(selected)>=8:break
        if edge_recovery is not None and edge_recovery.references:
            current_frame=display_frame(self.source,index)
            if edge_recovery.refine(current_frame,record['polygons']['screen'][0],record['polygons']['tablet'] if record['visible']['tablet'] else []) is not None:
                edge_recovery.remember_flow(current_frame,record['polygons']['screen'][0])
        rescue=getattr(tracker,'detector_rescue',None)
        if rescue is not None and record['visible']['tablet'] and self.get_anchor(index) is None:
            rescue.tablet_requires_redetection=bool(record.get('tracking',{}).get('tablet_requires_redetection')
                or '平板仅恢复可见部分，待完整重识别' in record.get('problems',[]))
            if rescue.tablet_requires_redetection and tracker.tablet_tracker is not None:
                # Constructor saw only the visible crop; never learn it as the
                # complete tablet. Rehydrate old trustworthy references below.
                tracker.tablet_tracker.references.clear()
                tracker.tablet_tracker.allow_reference_update=False
        if hasattr(tracker,'add_tablet_reference') and record['visible']['tablet']:
            # Rehydrate a good appearance after pause/restart, including when
            # the current tablet polygon is entirely outside the image.
            for payload, in self.db.execute('SELECT payload FROM records WHERE frame<? ORDER BY frame DESC LIMIT 3000',(index,)):
                old=json.loads(payload)
                if not old['visible']['tablet'] or old['problems']:continue
                polygon=old['polygons']['tablet'][0]
                if visible_fraction(polygon,self.source.size)>=.9:
                    tracker.add_tablet_reference(self.source.frame(old['frame_index']),polygon)
                    break
        return tracker

    def start(self,require_review: bool=True):
        record=self.get_record(self.cursor)
        if self.mode in ('issue','checkpoint','complete'):
            raise ValueError('请先修正异常或确认检查点，再继续')
        if not record:raise ValueError('请先确认当前帧轮廓')
        if not require_review:
            # Simple view is continuously supervised on screen. Keep end-of-slice
            # and anomaly guards; do not interrupt every 10 s / 2 s after edits.
            self.next_check=self.source.times[self.last_index]+1.
        if self.source.times[self.cursor]>=self.next_check:
            self.mode='checkpoint';self.reason='检查点尚未确认，请先回看并确认';self._save()
            raise ValueError(self.reason)
        # A seek cannot bypass a checkpoint or leave an unreviewed region arbitrarily long.
        unapproved=self.db.execute('SELECT MIN(frame) FROM records WHERE segment=? AND approved=0 AND frame<=?',(self.segment,self.cursor)).fetchone()[0]
        if require_review and unapproved is not None and self.source.times[self.cursor]-self.source.times[unapproved]>=self.checkpoint_s:
            self.mode='checkpoint';self.reason='该范围尚未审核，请先确认';self._save()
            raise ValueError(self.reason)
        # step() creates a tracker only when the next frame needs computing.
        # A continue click can return immediately; replay needs no tracker.
        self.running=True;self.mode='running';self.reason='运行中：可随时暂停'
        self.loss_streaks={}
        self.blur_streaks=set()
        self.token=uuid.uuid4().hex;self._save();return self.token

    def _pause_problems(self, problems, index):
        transient={'屏幕短暂失跟','平板短暂失跟'}
        blur={'屏幕运动模糊，等待恢复','平板运动模糊，等待恢复'}
        now=self.source.times[index]
        # One clock per AOI: changing from blur to loss must not reset it.
        current={p[:2] for p in problems if p in transient|blur}
        self.loss_streaks={p:v for p,v in self.loss_streaks.items() if p[:2] in current}
        self.blur_streaks=getattr(self,'blur_streaks',set()) & current
        review_only={'平板离开画面，等待返回','平板仅恢复可见部分，待完整重识别'}
        severe=[p for p in problems if p not in transient|blur|review_only]
        for label in current:
            key=label+'短暂失跟'
            first,count=self.loss_streaks.get(key,(index,0))
            self.loss_streaks[key]=(first,count+1)
            if label+'运动模糊，等待恢复' in problems:self.blur_streaks.add(label)
            limit=2. if label in self.blur_streaks else .6
            if count+1>=3 and now-self.source.times[first]>=limit-1e-8:
                severe.append(label+('模糊后仍未找回' if label in self.blur_streaks else '持续失跟'))
        if self._get('horizontal_mode') is True:return [p for p in severe if not p.startswith('屏幕持续') and not p.startswith('屏幕模糊后')]
        return severe

    def step(self,token: str,defer_issues: bool=False):
        if not self.running or token!=self.token:raise ValueError('已暂停或运行令牌过期')
        if self.cursor>=self.last_index:
            self.running=False;self.mode='checkpoint';self.reason='切片结束，等待人工验收';self._save();return
        index=self.cursor+1
        anchor=self.get_anchor(index)
        previous=self.get_record(self.cursor)
        existing=self.get_record(index)
        if existing:
            # Seeking for inspection must never silently rewrite reviewed geometry.
            polygons,visible,problems=existing['polygons'],existing['visible'],existing['problems']
            self.tracker=None
        else:
            tracking=None
            if self.tracker is None:
                self.tracker=self._make_tracker(self.cursor,previous)
            image=self.source.frame(index)
            if anchor:
                polygons,visible=anchor['polygons'],anchor['visible'];problems=[]
                self.tracker=self._make_tracker(index,dict(polygons=polygons,visible=visible))
            else:
                previous_rescue=getattr(self.tracker,'last_rescue',None)
                polygons,problems=self.tracker.update(image);visible=previous['visible']
                rescue=getattr(self.tracker,'last_rescue',None)
                if rescue is not None and rescue is not previous_rescue:
                    tracking=dict(automatic_relocation=dict(frame=index,applied=True,**copy.deepcopy(rescue)))
                rescue_state=getattr(self.tracker,'detector_rescue',None)
                if rescue_state is not None and rescue_state.tablet_requires_redetection:
                    tracking={**(tracking or {}),'tablet_requires_redetection':True}
                screen_evidence=getattr(self.tracker,'last_stats',{}).get('screen',{})
                if screen_evidence.get('method') in ('bezel_flow','manual_bezel_edges','dual_edge_lock'):
                    tracking={**(tracking or {}),'screen_boundary':copy.deepcopy(screen_evidence)}
                if defer_issues:
                    try:polygons=validate_polygons(polygons,visible,self.source.size)
                    except ValueError:
                        polygons=copy.deepcopy(previous['polygons']);problems=sorted(set(problems+['轮廓异常，待复核']))
                        if tracking is not None and 'automatic_relocation' in tracking:tracking['automatic_relocation']['applied']=False
                    if '起始轮廓待确认' in previous['problems']:problems=sorted(set(problems+['起始轮廓待确认']))
            self._put(index,polygons,visible,problems,tracking=tracking)
        self.cursor=index
        pause_problems=self._pause_problems(problems,index)
        if pause_problems and not defer_issues:
            self.running=False;self.mode='issue';self.reason='；'.join(pause_problems)+'。请检查轮廓，必要时回到红色片段起点修正'
            self._event('auto_pause',dict(frame=index,problems=pause_problems,
                                          first_warning_frame=min((v[0] for v in self.loss_streaks.values()),default=index)))
        elif index==self.last_index or self.source.times[index]>=self.next_check:
            self.running=False;self.mode='checkpoint';self.reason='切片结束，请验收' if index==self.last_index else '到达检查点，请回看本段后确认'
        else:
            self.reason=('转头模糊，正在等待清晰并重新定位；这段已标记待复核' if any('运动模糊' in p for p in problems) else '平板暂离画面，正在等待并重新定位；这段已标记待复核' if '平板离开画面，等待返回' in problems else '短暂跟踪不稳定，正在尝试恢复；红色片段可回看') if problems else '正在标注，随时可暂停；红色片段可回看'
        self._save()

    def approve(self):
        self.pause()
        rows=self.db.execute('SELECT frame,payload FROM records WHERE segment=? AND frame<=? ORDER BY frame',(self.segment,self.cursor)).fetchall()
        if [r[0] for r in rows]!=list(range(self.first_index,self.cursor+1)):
            raise ValueError('之前仍有未计算的帧，不能越过空缺审核')
        if any(json.loads(r[1])['problems'] for r in rows):
            raise ValueError('该范围仍有异常，请先修正并重算')
        self.db.execute('UPDATE records SET approved=1 WHERE segment=? AND frame<=?',(self.segment,self.cursor))
        self._event('human_approved',dict(through=self.cursor))
        self.mode='complete' if self.cursor==self.last_index else 'paused'
        self.reason='整个切片已人工确认' if self.mode=='complete' else '已确认至当前帧，可以继续'
        self.next_check=self.source.times[self.cursor]+self.checkpoint_s
        self._save()

    def snapshot(self):
        record=self.get_record(self.cursor)
        inherited_from=None
        if record:
            polygons=record['polygons'];visible=record['visible']
        else:
            nearby=self.db.execute('SELECT payload FROM anchors WHERE segment=? ORDER BY ABS(frame-?) LIMIT 1',(self.segment,self.cursor)).fetchone()
            polygons=json.loads(nearby[0])['polygons'] if nearby else copy.deepcopy(self.template)
            visible={'screen':True,'tablet':True}
            if not nearby and self.cursor==self.first_index and not self.db.execute(
                    'SELECT 1 FROM records WHERE segment=? LIMIT 1',(self.segment,)).fetchone():
                # Same recording only; use the latest saved earlier frame, not
                # whichever frame happened to be on screen before switching.
                previous=self.db.execute(
                    'SELECT segment,payload FROM records WHERE frame<? AND segment<>? ORDER BY frame DESC LIMIT 1',
                    (self.first_index,self.segment)).fetchone()
                if previous:
                    inherited_from=previous[0]
                    saved=json.loads(previous[1])
                    polygons=saved['polygons'];visible=saved['visible']
        coverage=[]
        if visible['screen']:
            try:coverage=coverage_for(polygons['screen'],self.source.size)
            except ValueError:pass
        summary=[]
        for key,row in self.slices.items():
            n,approved=self.db.execute('SELECT COUNT(*),COALESCE(SUM(approved),0) FROM records WHERE segment=? AND frame BETWEEN ? AND ?',(key,row['first'],row['last'])).fetchone()
            summary.append(dict(id=key,total=row['last']-row['first']+1,computed=n,approved=approved))
        spans=[]
        for index,approved,payload in self.db.execute('SELECT frame,approved,payload FROM records WHERE segment=? AND frame BETWEEN ? AND ? ORDER BY frame',(self.segment,self.first_index,self.last_index)):
            status=annotation_status(json.loads(payload)['problems'],approved)
            if spans and spans[-1]['status']==status and spans[-1]['end']==index-1:spans[-1]['end']=index
            else:spans.append(dict(start=index,end=index,status=status))
        from scripts.aoi_slice_timing import timing_info
        return dict(image_adjustments=self.source.image_adjustments,timing=timing_info(self),segment=self.segment,subject=self.manifest[0]['subject_id'],cursor=self.cursor,time_s=self.source.times[self.cursor],
                    first=self.first_index,last=self.last_index,start_s=self.slices[self.segment]['start_s'],end_s=self.slices[self.segment]['end_s'],
                    fps=self.source.fps,size=self.source.size,mode=self.mode,running=self.running,reason=self.reason,
                    polygons=polygons,screen_points=four_points(polygons['screen'],self.source.size),visible=visible,coverage=coverage,record_exists=bool(record),spans=spans,slices=summary,
                    annotation_status=annotation_status(record['problems'],record['approved']) if record else 'missing',
                    horizontal_mode=bool(self._get('horizontal_mode')),token=self.token,approved=bool(record and record['approved']),inherited_from=inherited_from)

    def export(self,approved_only: bool):
        self.pause()
        label='approved' if approved_only else 'candidate'
        folder=self.folder/'exports';folder.mkdir(exist_ok=True)
        records=[]
        for segment,index,payload,approved in self.db.execute('SELECT segment,frame,payload,approved FROM records ORDER BY segment,frame'):
            if segment not in self.slices or not self.slices[segment]['first']<=index<=self.slices[segment]['last']:continue
            if approved_only and not approved:continue
            r=json.loads(payload)
            aois=effective_aois(r['polygons'],r['visible'],self.source.size)
            records.append({**r,'segment_id':segment,'subject_id':self.manifest[0]['subject_id'],'approved':bool(approved),'aois':aois})
        path=folder/f'{label}.jsonl'
        temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));temp.replace(path)
        csv_path=folder/f'{label}.csv'
        with csv_path.open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.writer(file);writer.writerow(['subject_id','segment_id','frame_index','time_s','approved','screen_visible','tablet_visible','problems','aois_json','aoi_definition','gap_px'])
            for r in records:writer.writerow([r['subject_id'],r['segment_id'],r['frame_index'],r['time_s'],r['approved'],r['visible']['screen'],r['visible']['tablet'],';'.join(r['problems']),json.dumps(r['aois']),r['polygons'].get('partition',{}).get('mode','precise'),r['polygons'].get('partition',{}).get('gap_px','')])
        metadata=dict(frames=len(records),approved_only=approved_only,coordinate_size=self.source.size,manifest=self.manifest,source_identity=self.source.identity,
                      note='人工确认与算法候选分开；不可见区域为空。时间为原视频 PTS，切片区间为 [start,end)。')
        (folder/f'{label}_meta.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        return dict(frames=len(records),path=str(path.resolve()),csv=str(csv_path.resolve()))

    def close(self):
        if self.closed:return
        self.pause()
        if self.tracker is not None and hasattr(self.tracker,'close'):self.tracker.close()
        self.db.close();self.closed=True
