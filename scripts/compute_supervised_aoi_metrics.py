"""Reproducible I-VT pilot from a read-only AOI snapshot and Tobii raw samples.

Never fills missing AOI frames. This is an explicit alternative pipeline, not an
implementation of undocumented ErgoLAB processing settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

from scripts.supervised_aoi import coverage_for


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_gaze(path: Path) -> dict:
    times, xy, directions, pupils = [], [], [], []
    with gzip.open(path, 'rt') as f:
        for line in f:
            obj = json.loads(line)
            data = obj.get('data', {})
            times.append(float(obj['timestamp']))
            p = data.get('gaze2d', [np.nan, np.nan])
            xy.append(p if len(p) == 2 else [np.nan, np.nan])
            ds, ps = [], []
            for key in ('eyeleft', 'eyeright'):
                eye = data.get(key, {})
                d = np.asarray(eye.get('gazedirection', [np.nan] * 3), float)
                if d.shape == (3,) and np.isfinite(d).all() and np.linalg.norm(d) > 0:
                    ds.append(d / np.linalg.norm(d))
                pupil = eye.get('pupildiameter', np.nan)
                ps.append(pupil if isinstance(pupil, (float, int)) and pupil > 0 else np.nan)
            d = np.mean(ds, axis=0) if ds else np.full(3, np.nan)
            directions.append(d / np.linalg.norm(d) if ds else d)
            pupils.append(ps)
    t = np.asarray(times)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Gaze timestamps must be strictly increasing')
    return dict(t=t, xy=np.asarray(xy, float), direction=np.asarray(directions), pupil=np.asarray(pupils))


def prepare(gaze: dict, gap_s: float) -> dict:
    t, xy, direction = gaze['t'], gaze['xy'].copy(), gaze['direction'].copy()
    valid = np.isfinite(xy).all(axis=1) & ((xy >= 0) & (xy <= 1)).all(axis=1) & np.isfinite(direction).all(axis=1)
    raw_valid = valid.copy()
    # Interpolate only bounded short gaps; never extrapolate recording edges.
    for left, right in zip(np.flatnonzero(valid)[:-1], np.flatnonzero(valid)[1:]):
        if right > left + 1 and t[right] - t[left] <= gap_s + 1e-9:
            for i in range(left + 1, right):
                a = (t[i] - t[left]) / (t[right] - t[left])
                xy[i] = (1-a)*xy[left] + a*xy[right]
                d = (1-a)*direction[left] + a*direction[right]
                direction[i] = d / np.linalg.norm(d)
                valid[i] = True
    return {**gaze, 'xy': xy, 'direction': direction, 'valid': valid, 'raw_valid': raw_valid, 'interpolated': valid & ~raw_valid}


def fixations(gaze: dict, threshold: float = 30, minimum: float = .1,
              merge_gap: float = .1, merge_angle: float = 3) -> list[dict]:
    t, d, valid = gaze['t'], gaze['direction'], gaze['valid']
    dt = np.diff(t)
    dot = np.clip(np.sum(d[:-1]*d[1:], axis=1), -1, 1)
    velocity = np.degrees(np.arccos(dot)) / dt
    low = valid[:-1] & valid[1:] & (dt <= .075) & (velocity <= threshold)
    edges = np.diff(np.r_[False, low, False].astype(int))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    merged: list[list[int]] = []
    for start, end in runs:
        if merged:
            a, b = merged[-1]
            v1, v2 = np.mean(d[a:b+1], axis=0), np.mean(d[start:end+1], axis=0)
            angle = np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))
            # A missing-data stretch is a break, even if event merging is enabled.
            if t[start]-t[b] <= merge_gap and angle <= merge_angle and valid[b:start+1].all() and np.all(dt[b:start] <= .075):
                merged[-1][1] = int(end)
                continue
        merged.append([int(start), int(end)])
    return [dict(id=i, start_s=float(t[a]), end_s=float(t[b]), first=a, last=b)
            for i, (a,b) in enumerate(merged) if t[b]-t[a] >= minimum-1e-9]


def labels_for_frames(db: Path, size: tuple[int, int]) -> dict:
    c = sqlite3.connect(f'file:{db.resolve()}?mode=ro', uri=True)
    result = {}
    for segment, frame, payload, approved in c.execute('SELECT segment,frame,payload,approved FROM records'):
        row = json.loads(payload)
        shapes = dict(screen=[], tablet=[])
        if not row['problems']:
            if row['visible']['screen']:
                shapes['screen'] = coverage_for(row['polygons']['screen'], size)
            if row['visible']['tablet']:
                shapes['tablet'] = row['polygons']['tablet']
        result[(segment,frame)] = dict(shapes={k:[np.asarray(p,np.float32) for p in v] for k,v in shapes.items()},
                                      problem=bool(row['problems']), approved=bool(approved),aoi_definition=row['polygons'].get('partition',{}).get('mode','precise'))
    c.close()
    return result


def aoi_label(record: dict | None, xy: np.ndarray, size: tuple[int,int]) -> str:
    if record is None or record['problem']:
        return 'unknown'
    p = (float(xy[0]*size[0]), float(xy[1]*size[1]))
    # Foreground tablet takes priority; categories never double-count overlap.
    for label in ('tablet','screen'):
        if any(cv2.pointPolygonTest(poly,p,False) >= 0 for poly in record['shapes'][label]):
            return label
    return 'other'


def support_pieces(gaze: dict, pts: np.ndarray, frames: dict, segment: dict,
                   size: tuple[int,int]) -> list[dict]:
    """Intersect gaze support with [slice start,end) and actual frame intervals."""
    t = gaze['t']; start, end = segment['start_s'], segment['end_s']
    nominal = float(np.median(np.diff(t)))
    rows = []
    for i in range(max(0,int(np.searchsorted(t,start))-1), min(len(t),int(np.searchsorted(t,end)))):
        left = max(float(t[i]),start)
        # Do not extend a gaze sample through a missing timestamp interval.
        right = min(float(t[i+1]) if i+1 < len(t) else end, float(t[i])+nominal*1.5, end)
        while left < right-1e-10:
            frame = int(np.searchsorted(pts,left,side='right')-1)
            boundary = min(right,float(pts[frame+1]) if frame+1 < len(pts) else right)
            record = frames.get((segment['segment_id'],frame))
            label = aoi_label(record,gaze['xy'][i],size) if gaze['valid'][i] else 'invalid'
            rows.append(dict(start_s=left,end_s=boundary,sample=i,frame=frame,label=label,
                             annotated=record is not None and not record['problem'],
                             interpolated=bool(gaze['interpolated'][i])))
            left = boundary
    return rows


def transition_counts(events: list[dict], pieces: list[dict], bridge_other_s: float = 0) -> dict:
    counts = {'screen_to_tablet_n':0, 'tablet_to_screen_n':0}
    previous = None
    for event in events:
        label = event['aoi']
        if label not in ('screen','tablet'):
            if label != 'other' or bridge_other_s == 0:
                previous = None
            continue
        if previous:
            gap = event['start_s'] - previous['end_s']
            between = [p for p in pieces if p['end_s'] > previous['end_s']+1e-9 and p['start_s'] < event['start_s']-1e-9]
            uninterrupted = all(p['label'] not in ('unknown','invalid') for p in between)
            supported = sum(max(0,min(p['end_s'],event['start_s'])-max(p['start_s'],previous['end_s'])) for p in between)
            if previous['aoi'] != label and gap <= .3+1e-9 and uninterrupted and supported >= gap-1e-6:
                counts[previous['aoi']+'_to_'+label+'_n'] += 1
        previous = event
    return {**counts, 'switches_n':sum(counts.values())}


def summarize(segment: dict, pieces: list[dict], events: list[dict], pts: np.ndarray, frames: dict) -> tuple[dict,list[dict]]:
    start,end = segment['start_s'],segment['end_s']; sid = segment['segment_id']
    total = end-start
    frame_ids = list(range(int(np.searchsorted(pts,start)),int(np.searchsorted(pts,end))))
    available = [i for i in frame_ids if (sid,i) in frames and not frames[(sid,i)]['problem']]
    coverage = sum(max(0,min(end,pts[i+1] if i+1<len(pts) else end)-max(start,pts[i])) for i in available)
    out = dict(subject_id=segment['subject_id'],segment_id=sid,phase='normal' if sid.endswith('.1') else 'error',
               start_s=start,end_s=end,duration_s=total,annotated_frames=len(available),expected_frames=len(frame_ids),
               aoi_coverage_s=coverage,aoi_coverage_pct=100*coverage/total,
               status='candidate_complete_frames' if len(available)==len(frame_ids) else 'partial' if available else 'unannotated',
               raw_valid_gaze_s=sum(p['end_s']-p['start_s'] for p in pieces if p['label']!='invalid' and not p['interpolated']),
               interpolated_gaze_s=sum(p['end_s']-p['start_s'] for p in pieces if p['interpolated']))
    fixation_rows = []
    for ev in events:
        a,b = max(start,ev['start_s']),min(end,ev['end_s'])
        if a >= b:
            continue
        weights = Counter()
        for p in pieces:
            duration = min(b,p['end_s'])-max(a,p['start_s'])
            if duration > 0:
                weights[p['label']] += duration
        known = sum(weights.get(k,0) for k in ('screen','tablet','other'))
        label = max(('screen','tablet','other'),key=lambda k:weights.get(k,0)) if known >= b-a-1e-6 else 'unknown'
        fixation_rows.append(dict(segment_id=sid,fixation_id=ev['id'],start_s=a,end_s=b,duration_s=b-a,aoi=label,
                                  screen_support_s=weights['screen'],tablet_support_s=weights['tablet'],other_support_s=weights['other']))
    definitions=sorted({r.get('aoi_definition','precise') for (segment_id,_),r in frames.items() if segment_id==sid})
    out['aoi_definition']=';'.join(definitions)
    out['aoi_definition_note']='下方区域为平板近似区域，包含桌面和手；与精细平板指标口径不同' if 'straight_line' in definitions else '精细AOI'
    included = [e for e in fixation_rows if e['aoi']!='unknown']
    denominator = sum(e['duration_s'] for e in included)
    out['valid_fixation_duration_s'] = denominator
    out['valid_fixation_count_n'] = len(included)
    out['unknown_aoi_fixation_s'] = sum(e['duration_s'] for e in fixation_rows if e['aoi']=='unknown')
    def pct(value: float, base: float) -> float | None:
        return 100*value/base if base > 0 else None
    for label in ('screen','tablet','other'):
        hit_events = [e for e in included if e['aoi']==label]
        duration = sum(e['duration_s'] for e in hit_events)
        out[label+'_fixation_s'] = duration if available else None
        out[label+'_fixation_pct_valid'] = pct(duration,denominator)
        out[label+'_fixation_count_n'] = len(hit_events) if available else None
        out[label+'_fixation_count_pct'] = pct(len(hit_events),len(included))
        out[label+'_mean_fixation_s'] = duration/len(hit_events) if hit_events else None
        out[label+'_gaze_hit_s'] = sum(p['end_s']-p['start_s'] for p in pieces if p['label']==label) if available else None
    pair = sum(e['duration_s'] for e in included if e['aoi'] in ('screen','tablet'))
    out['screen_pair_pct'] = pct(out['screen_fixation_s'] or 0,pair)
    out['tablet_pair_pct'] = pct(out['tablet_fixation_s'] or 0,pair)
    out.update(transition_counts(fixation_rows,pieces))
    out['switches_bridge_other_300ms_n'] = transition_counts(fixation_rows,pieces,.3)['switches_n']
    if not available:
        for key in list(out):
            if 'switch' in key or '_to_' in key: out[key] = None
    return out,fixation_rows


def read_ergolab(path: Path) -> tuple[list[dict],list[dict]]:
    """Read cell values without touching malformed merge definitions in source."""
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    aoi_rows,global_rows=[],[]
    with zipfile.ZipFile(path) as z:
        strings=[''.join(n.text or '' for n in s.iter('{'+ns['m']+'}t')) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',ns):
            name=sheet.attrib['name'];rid=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target=rels[rid].lstrip('/');target=target if target.startswith('xl/') else 'xl/'+target
            cells={}
            for c in ET.fromstring(z.read(target)).findall('.//m:sheetData/m:row/m:c',ns):
                v=c.find('m:v',ns)
                if v is not None:cells[c.attrib['r']]=strings[int(v.text)] if c.attrib.get('t')=='s' else v.text
            subject=re.match(r'(\d+)',str(cells.get('B4',cells.get('B3',''))))
            segment=str(cells.get('C3',''))
            for ref,title in cells.items():
                if not ref.startswith('A') or title=='Data Info':continue
                row=int(ref[1:]);head=row+1;data=row+2
                for col in ('D','E'):
                    raw=cells.get(f'{col}{data}')
                    try:value=float(raw)
                    except (TypeError,ValueError):continue
                    result=dict(sheet=name,subject_id=int(subject[1]) if subject else None,segment_id=segment,
                                time_range=cells.get('C4',''),metric=title,aoi=cells.get(f'{col}{head}',''),value=value,cell=f'{col}{data}')
                    (aoi_rows if 'AOI Segment Type'==cells.get('B2') else global_rows).append(result)
    return aoi_rows,global_rows


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=Path('outputs/aoi_workflow_20260928/timing_v3/demo_segments.json'))
    ap.add_argument('--database',type=Path,default=Path('outputs/aoi_demo_session/session.sqlite3'))
    ap.add_argument('--timestamps',type=Path,default=Path('outputs/aoi_demo_video/source_pts.json'))
    ap.add_argument('--ergolab',type=Path,default=Path('examples/eye-events.xlsx'))
    ap.add_argument('--out',type=Path,default=Path('outputs/aoi_metrics_25_20260928'))
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    # SQLite backup captures a consistent point in time, including committed WAL.
    snapshot=out/'aoi_snapshot.sqlite3'
    if args.database.resolve()!=snapshot.resolve():
        source=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
        dest=sqlite3.connect(snapshot);source.backup(dest);dest.close();source.close()
    manifest=json.loads(args.manifest.read_text());pts=np.asarray(json.loads(args.timestamps.read_text()))
    subject=int(manifest[0]['subject_id'])
    rawdir=Path(manifest[0]['video_path']).parent
    cap=cv2.VideoCapture(manifest[0]['video_path'])
    if not cap.isOpened():raise ValueError('无法读取视频尺寸')
    size=(960,round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*960/cap.get(cv2.CAP_PROP_FRAME_WIDTH)));cap.release()
    raw=load_gaze(rawdir/'gazedata.gz');frames=labels_for_frames(snapshot,size)
    params=dict(velocity_deg_s=30,minimum_fixation_s=.1,interpolation_gap_s=.075,merge_gap_s=.1,merge_angle_deg=3,
                smoothing='none',velocity='adjacent normalized mean-eye direction',assignment='duration-weighted majority per fixation',
                overlap='tablet priority',missing_aoi='unknown; exclude entire affected fixation',interval='[start,end)',
                transition_gap_limit_s=.3)
    all_rows=[]
    for name,threshold,gap,merge in [('pilot_ivt',30,.075,.1),('no_gap_no_merge',30,0,0),('threshold_40',40,.075,.1)]:
        gaze=prepare(raw,gap);events=fixations(gaze,threshold=threshold,merge_gap=merge)
        summary,event_rows=[],[]
        for segment in manifest:
            pieces=support_pieces(gaze,pts,frames,segment,size)
            row,classified=summarize(segment,pieces,events,pts,frames)
            row['method']=name;summary.append(row);event_rows.extend(classified)
            if name=='pilot_ivt' and row['annotated_frames']:
                write_csv(out/f'gaze_aoi_{segment["segment_id"]}.csv',pieces)
        all_rows.extend(summary)
        if name=='pilot_ivt':
            write_csv(out/f'{subject}号_切片指标.csv',summary);write_csv(out/f'{subject}号_注视事件.csv',event_rows)
    write_csv(out/f'{subject}号_参数敏感性.csv',all_rows)
    aoi,global_rows=read_ergolab(args.ergolab) if args.ergolab.exists() else ([],[])
    own=[r for r in global_rows if r['subject_id']==subject]
    write_csv(out/'Ergolab_AOI原始指标.csv',aoi)
    write_csv(out/f'Ergolab_{subject}号整段指标.csv',own)
    p=raw['pupil'];count=np.isfinite(p).sum(axis=1);valid=count>0
    average=np.nansum(p,axis=1)[valid]/count[valid]
    if not len(average):average=np.array([np.nan])
    checks=[]
    for label,calculated in [('平均瞳孔直径(mm)',np.mean(average)),('最小瞳孔直径(mm)',np.min(average)),('最大瞳孔直径(mm)',np.max(average))]:
        matches=[r['value'] for r in own if r['metric']==label]
        if len(matches)!=1:continue
        old=matches[0]
        checks.append(dict(metric=label,raw_mean_available_eyes=float(calculated),ergolab=old,difference=float(calculated-old),same_processing=False))
    write_csv(out/f'{subject}号_整段瞳孔对照.csv',checks)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),parameters=params,raw_samples=len(raw['t']),
                  valid_pupil_samples=int(valid.sum()),raw_gaze_path=str(rawdir/'gazedata.gz'),manifest=str(args.manifest.resolve()),
                  aoi_snapshot=str(snapshot.resolve()),aoi_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  ergolab_source=str(args.ergolab),ergolab_same_subject_aoi_rows=sum(r['subject_id']==subject for r in aoi),
                  status='pilot; missing Ergolab filter settings and same-subject AOI reference')
    (out/'calculation_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps(dict(annotated=[{k:r[k] for k in ('segment_id','status','aoi_coverage_pct','screen_fixation_pct_valid','tablet_fixation_pct_valid','screen_pair_pct','switches_n')} for r in all_rows if r['method']=='pilot_ivt' and r['annotated_frames']],pupil_comparison=checks),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
