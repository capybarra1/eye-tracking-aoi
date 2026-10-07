"""Full recording runner for the locally reviewed screen/tablet AOI pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from scripts.fast_aoi_tracker import (
    RegionTrack, StepStats, draw_polygons, enhance_frame_for_detection,
    estimate_region_motion, expand_polygon_outward, load_prompt_polygons,
    make_contact, prepare_gray, reseed_region, serialize_polygons, transform_polygon,
)
from scripts.screen_structure_tracker import (
    StructureReferenceCache, estimate_structure_motion, screen_coverage, structure_mask,
)


def direction_indices(count: int, seed: int) -> tuple[range, range]:
    if not 0 <= seed < count:
        raise ValueError('Seed frame is outside video')
    return range(seed-1,-1,-1), range(seed,count)


def validate_records(records: dict, expected_count: int, timestamps: list[float]|None=None) -> list[dict]:
    if set(records) != set(range(expected_count)):
        raise RuntimeError(f'Incomplete tracking journal: {len(records)} records, expected {expected_count}')
    ordered=[records[i] for i in range(expected_count)]
    if timestamps is not None and any(abs(r['time_s']-timestamps[i])>.0001 for i,r in enumerate(ordered)):
        raise RuntimeError('Tracking journal does not match canonical timestamps')
    if not all(b['time_s']>a['time_s'] for a,b in zip(ordered,ordered[1:])):
        raise RuntimeError('Video timestamps are not strictly increasing')
    return ordered


def review_intervals(records: list[dict], frame_duration: float) -> list[dict]:
    intervals=[]
    for aoi in ('screen','tablet'):
        start=None
        for i,record in enumerate(records):
            flagged=record[f'{aoi}_needs_review']
            if flagged and start is None:
                start=i
            if start is not None and (not flagged or i==len(records)-1):
                last=i-1 if not flagged else i
                end=record['time_s'] if not flagged else record['time_s']+frame_duration
                intervals.append(dict(aoi=aoi,start_frame=records[start]['frame_index'],
                                      end_frame=records[last]['frame_index'],
                                      start_s=records[start]['time_s'],end_s=round(end,6)))
                start=None
    return sorted(intervals,key=lambda r:(r['start_s'],r['aoi']))


def frame_stream(video: Path, indices: range, size: tuple[int,int], timestamps: list[float]|None=None) -> Iterator[tuple[int,float,np.ndarray]]:
    """Decode forwards; use bounded blocks for reverse tracking, avoiding per-frame seeking."""
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {video}')
    def read_at(index: int) -> tuple[float,np.ndarray]:
        while True:
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Decode failed at canonical frame {index}')
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamps is None or pts>=timestamps[index]-.0001: break
        if timestamps is not None and abs(pts-timestamps[index])>.0001:
            raise RuntimeError(f'Seek alignment failed at frame {index}: {pts} vs {timestamps[index]}')
        return pts,cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
    try:
        if indices.step>0:
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,indices.start-5) if timestamps else indices.start)
            for index in indices:
                if timestamps is not None:
                    pts,frame=read_at(index)
                    yield index,pts,frame
                    continue
                ok,frame=cap.read()
                if not ok:
                    # Container frame counts can overstate the actual decodable
                    # frame count. Only tolerate EOF at the declared tail.
                    if index >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-2:
                        return
                    raise RuntimeError(f'Decode failed at frame {index}')
                yield index,cap.get(cv2.CAP_PROP_POS_MSEC)/1000,cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
        else:
            high=indices.start
            while high>indices.stop:
                low=max(indices.stop+1,high-99)
                cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,low-5) if timestamps else low)
                block=[]
                for index in range(low,high+1):
                    if timestamps is not None:
                        pts,frame=read_at(index)
                        block.append((index,pts,frame))
                        continue
                    ok,frame=cap.read()
                    if not ok: raise RuntimeError(f'Decode failed at frame {index}')
                    block.append((index,cap.get(cv2.CAP_PROP_POS_MSEC)/1000,cv2.resize(frame,size,interpolation=cv2.INTER_AREA)))
                yield from reversed(block)
                high=low-1
    finally: cap.release()


class PilotState:
    def __init__(self, frame: np.ndarray, seed: dict[str,list[np.ndarray]], tablet_padding: float=36.):
        self.size=(frame.shape[1],frame.shape[0])
        self.seed=[p.copy() for p in seed['screen']]
        self.screens=[p.copy() for p in self.seed]
        self.coverage=screen_coverage(self.screens,self.size)
        self.tablets=[RegionTrack('tablet',i,expand_polygon_outward(p,tablet_padding)) for i,p in enumerate(seed['tablet'])]
        self.gray=prepare_gray(frame)
        self.tablet_gray=prepare_gray(enhance_frame_for_detection(frame,1.45,34.))
        for region in self.tablets: reseed_region(self.tablet_gray,region,self.size)
        self.H=np.eye(3,dtype=np.float32)
        self.screen_lost=False
        self.tablet_lost=False
        self.steps=0
        self.screen_projector=None
        self.tablet_tracker=None
        self.structure_reference_cache=StructureReferenceCache()

    def update_screen(self, gray: np.ndarray, require_strong: bool=False) -> dict:
        mask=structure_mask(gray.shape,self.screens,[r.polygon for r in self.tablets])
        H,screen_stats=estimate_structure_motion(self.gray,gray,mask,cache=self.structure_reference_cache)
        if require_strong and not (screen_stats.get('inliers',0)>=30
                                   and screen_stats.get('confidence',0)>=.55
                                   and screen_stats.get('support',0)>=.03):
            return {**screen_stats,'status':'lost'}
        if screen_stats['status']=='tracked':
            proposed=H@self.H
            proposed/=proposed[2,2]
            moved=[transform_polygon(p,proposed) for p in self.seed]
            sane=all(np.isfinite(p).all() and np.max(np.abs(p))<max(self.size)*6 for p in moved)
            try:
                if not sane: raise ValueError('Extreme projected polygon')
                if self.screen_projector is not None:
                    moved,coverage=self.screen_projector(proposed)
                else:
                    coverage=screen_coverage(moved,self.size)
                if not all(np.isfinite(p).all() and np.max(np.abs(p))<max(self.size)*8 for p in coverage):
                    raise ValueError('Extreme extrapolated lower edge')
                self.H,self.screens,self.coverage=proposed,moved,coverage
            except ValueError:
                screen_stats={**screen_stats,'status':'geometry_failed'}
                self.screen_lost=True
        else: self.screen_lost=True
        return screen_stats

    def update(self, frame: np.ndarray) -> tuple[dict,dict]:
        gray=prepare_gray(frame)
        screen_stats=self.update_screen(gray)
        # TabletRecovery consumes the original frame and maintains its own gray
        # reference. This enhanced image is used only by the legacy tracker.
        tablet_gray=(prepare_gray(enhance_frame_for_detection(frame,1.45,34.))
                     if self.tablet_tracker is None else None)
        for region in self.tablets:
            if self.tablet_tracker is not None:
                region.polygon,region.status=self.tablet_tracker.update(frame)
                continue
            motion,points,status,confidence,inliers,tracked=estimate_region_motion(
                self.tablet_gray,tablet_gray,region.points,region.polygon,self.size)
            if status=='tracked':
                region.polygon=transform_polygon(region.polygon,motion)
                region.points=points
            else: self.tablet_lost=True
            region.status,region.confidence,region.inliers,region.tracked_points=status,confidence,inliers,tracked
            if region.points is None or len(region.points)<45 or self.steps%18==0 or status!='tracked':
                reseed_region(tablet_gray,region,self.size)
        self.gray=gray
        if tablet_gray is not None:self.tablet_gray=tablet_gray
        self.steps+=1
        tablet_status=self.tablets[0].status if self.tablet_tracker is not None and self.tablets else ('tracked' if all(r.status=='tracked' for r in self.tablets) else 'lost')
        return screen_stats,{**(getattr(self.tablet_tracker,'quality',{}) if self.tablet_tracker is not None else {}), 'status':tablet_status}

    def record(self,index:int,pts:float,direction:str,screen_stats:dict,tablet_stats:dict)->dict:
        return dict(frame_index=index,time_s=round(pts,6),direction=direction,
                    screen_status=screen_stats['status'],tablet_status=tablet_stats['status'],
                    screen_needs_review=self.screen_lost,tablet_needs_review=self.tablet_lost,
                    screen_motion_inliers=screen_stats.get('inliers',0),
                    screen_motion_matches=screen_stats.get('tracked_points',0),
                    aois=serialize_polygons({'screen':self.coverage,'tablet':[r.polygon for r in self.tablets]}))


def overlay(frame:np.ndarray,record:dict)->np.ndarray:
    polys={k:[np.array(p,np.float32) for p in v] for k,v in record['aois'].items()}
    state='NEEDS REVIEW' if record['screen_needs_review'] or record['tablet_needs_review'] else 'candidate'
    stats=StepStats(record['frame_index'],record['time_s'],state,0,0,0)
    image=draw_polygons(frame,polys,stats)
    cv2.rectangle(image,(0,0),(frame.shape[1],36),(0,0,0),-1)
    color=(0,170,255) if state=='NEEDS REVIEW' else (255,255,255)
    cv2.putText(image,f"25-HLY | {record['time_s']:.2f}s | {state} | S:{record['screen_status']} T:{record['tablet_status']}",
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.58,color,2)
    return image


def run(video:Path,prompts:Path,out:Path,seed_time:float=90.,limit_frames:int|None=None,resume_export:bool=False,timestamp_file:Path|None=None)->dict:
    total_start=time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    samples=out/'samples';samples.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {video}')
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    declared_count=count
    count=min(count,limit_frames) if limit_frames else count
    size=(960,round(source_size[1]*960/source_size[0]))
    seed_index=round(seed_time*fps)
    _,seed_pts,seed_frame=next(frame_stream(video,range(seed_index,seed_index+1),size))
    timestamps=json.loads(timestamp_file.read_text()) if timestamp_file else None
    if timestamps is not None:
        count=min(count,len(timestamps))
        seed_index=min(range(count),key=lambda i:abs(timestamps[i]-seed_pts))
        if abs(timestamps[seed_index]-seed_pts)>.0001: raise ValueError('Seed image timestamp not in canonical index')
    backwards,forwards=direction_indices(count,seed_index)
    seeds=load_prompt_polygons(prompts,size)
    if len(seeds.get('screen',[]))!=3 or not seeds.get('tablet'): raise ValueError('Three screens and tablet required')
    (out/'user_prompts.json').write_bytes(prompts.read_bytes())
    records={}
    previous_tracking_sec=0.
    if resume_export:
        checkpoint=json.loads((out/'tracking_done.json').read_text())
        if checkpoint['frames'] != count or checkpoint.get('expected_frames',count) != count:
            raise RuntimeError('Checkpoint frame count differs from requested video range; use the original --limit-frames if applicable')
        for line in (out/'tracking_journal.jsonl').open():
            record=json.loads(line)
            if record['frame_index'] in records: raise RuntimeError('Duplicate frame in tracking journal')
            records[record['frame_index']]=record
        validate_records(records,count,timestamps)
        previous_tracking_sec=checkpoint['tracking_sec']
        total_start-=previous_tracking_sec
    track_start=time.perf_counter()
    journal=None if resume_export else (out/'tracking_journal.jsonl').open('w',buffering=1)
    for direction,indices in ([] if resume_export else [('backward',backwards),('forward',forwards)]):
        state=PilotState(seed_frame,seeds)
        for index,pts,frame in frame_stream(video,indices,size,timestamps):
            if index==seed_index:
                screen_stats=tablet_stats=dict(status='seed')
            else: screen_stats,tablet_stats=state.update(frame)
            record=state.record(index,pts,direction,screen_stats,tablet_stats)
            records[index]=record
            journal.write(json.dumps(record)+'\n')
            if index%round(fps*30)==0 or index==seed_index or index==count-1:
                cv2.imwrite(str(samples/f'frame_{index:06d}.jpg'),overlay(frame,record))
            if len(records)%1000==0 or len(records)==count:
                progress=dict(stage='tracking',processed=len(records),total=count,source_s=round(pts,1),
                              elapsed_s=round(time.perf_counter()-total_start,1),
                              screen_needs_review=state.screen_lost,tablet_needs_review=state.tablet_lost)
                (out/'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress),flush=True)
        (out/f'{direction}_records.jsonl').write_text(''.join(json.dumps(records[i])+'\n' for i in indices if i in records))
    if journal is not None: journal.close()
    tracking_sec=previous_tracking_sec if resume_export else time.perf_counter()-track_start
    ordered=validate_records(records,count,timestamps)
    (out/'tracking_done.json').write_text(json.dumps(dict(tracking_sec=tracking_sec,frames=len(records),expected_frames=count)))
    with (out/'aoi_frames.jsonl').open('w') as file:
        for r in ordered: file.write(json.dumps(r)+'\n')
    fields=['frame_index','time_s','aoi','polygon_index','point_index','x','y','needs_review','tracking_status']
    with (out/'aoi_polygons.csv').open('w',newline='',encoding='utf-8-sig') as file:
        csv_writer=csv.DictWriter(file,fieldnames=fields);csv_writer.writeheader()
        for r in ordered:
            for aoi,polygons in r['aois'].items():
                for p,polygon in enumerate(polygons):
                    for point,(x,y) in enumerate(polygon):
                        csv_writer.writerow(dict(frame_index=r['frame_index'],time_s=r['time_s'],aoi=aoi,
                            polygon_index=p,point_index=point,x=x,y=y,needs_review=r[f'{aoi}_needs_review'],
                            tracking_status=r[f'{aoi}_status']))
    intervals=review_intervals(ordered,1/fps)
    (out/'review_intervals.json').write_text(json.dumps(intervals,indent=2))
    render_start=time.perf_counter()
    writer=cv2.VideoWriter(str(out/'full_aoi_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened(): raise RuntimeError('Cannot create review video')
    for index,pts,frame in frame_stream(video,range(count),size,timestamps):
        if abs(pts-records[index]['time_s'])>.0001:
            raise RuntimeError(f'Render timestamp mismatch at frame {index}: {pts} vs {records[index]["time_s"]}')
        writer.write(overlay(frame,records[index]))
        if (index+1)%3000==0 or index==count-1:
            progress=dict(stage='rendering',processed=index+1,total=count,elapsed_s=round(time.perf_counter()-total_start,1))
            (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    writer.release()
    sample_paths=sorted(samples.glob('frame_*.jpg'))
    pages=[]
    for start in range(0,len(sample_paths),12):
        page=out/f'contact_{start//12+1:02d}.jpg'
        cv2.imwrite(str(page),make_contact([cv2.imread(str(p)) for p in sample_paths[start:start+12]],columns=3))
        pages.append(str(page.resolve()))
    meta=dict(video=str(video.resolve()),source_size=source_size,work_size=size,source_fps=fps,
              frame_count=count,container_declared_frames=declared_count,source_duration_s=ordered[-1]['time_s']+1/fps,
              seed_frame=seed_index,seed_time_s=seed_pts,prompt_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest(),
              timestamp_index=str(timestamp_file.resolve()) if timestamp_file else None,
              screen_method='Reviewed physical-structure incremental homography and lower-edge coverage',
              tablet_method='Reviewed padded polygon translation tracking',
              review_status='candidate_not_approved',
              screen_flagged_frames=sum(r['screen_needs_review'] for r in ordered),
              tablet_flagged_frames=sum(r['tablet_needs_review'] for r in ordered),
              screen_status_counts=dict(Counter(r['screen_status'] for r in ordered)),
              tablet_status_counts=dict(Counter(r['tablet_status'] for r in ordered)),
              tracking_sec=round(tracking_sec,3),rendering_sec=round(time.perf_counter()-render_start,3),
              total_sec=round(time.perf_counter()-total_start,3),contact_pages=pages,
              limitations=['Flags remain set after a lost step; no automatic relocalization.',
                           'No loss flag does not imply accurate AOI; gradual drift requires visual review.',
                           'Review video uses average source FPS; exported timestamps retain decoded frame PTS.',
                           'Full recording includes preparation and end screens; no trial statistics computed.'])
    (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    (out/'progress.json').write_text(json.dumps(dict(stage='complete',processed=count,total=count)))
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('video','prompts','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed-time',type=float,default=90.)
    parser.add_argument('--limit-frames',type=int)
    parser.add_argument('--opencv-threads',type=int,default=2)
    parser.add_argument('--resume-export',action='store_true')
    parser.add_argument('--timestamps',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(args.opencv_threads)
    run(args.video,args.prompts,args.out,args.seed_time,args.limit_frames,args.resume_export,args.timestamps)
