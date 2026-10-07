"""Bounded automatic retries. Draft first; only failed/missing frames may change."""
from __future__ import annotations
import bisect, copy, json, sqlite3, uuid
import cv2
from scripts.aoi_image_adjustments import make_tracker
from scripts.aoi_horizontal import HorizontalTracker
from scripts.aoi_range_review import signature
from scripts.supervised_aoi import validate_polygons


def tracking_failed(record: dict | None) -> bool:
    return bool(record and any(p != '起始轮廓待确认' for p in record.get('problems', [])))


def prepare_retry(s, start: int) -> dict:
    start=max(s.first_index,start)
    end=min(s.last_index,bisect.bisect_right(s.source.times,s.source.times[start]+8)-1)
    seed_index=start-1
    seed=s.get_record(seed_index) if seed_index>=s.first_index else None
    if not seed or tracking_failed(seed):
        raise ValueError('异常前没有可靠的定位参照，请先调整此段起点')
    if not seed['polygons'].get('partition'):
        raise ValueError('自动回退重试适用于直线分区，请先确认直线起点')
    records={i:json.loads(p) for i,p in s.db.execute('SELECT frame,payload FROM records WHERE segment=? AND frame BETWEEN ? AND ?',(s.segment,start,end))}
    anchors={i:json.loads(p) for i,p in s.db.execute('SELECT frame,payload FROM anchors WHERE segment=? AND frame BETWEEN ? AND ?',(s.segment,start,end))}
    references=[dict(frame=i,polygons=json.loads(payload)['polygons']) for i,payload in s.db.execute('SELECT frame,payload FROM anchors WHERE frame<=? ORDER BY frame DESC LIMIT 4',(seed_index,)) if json.loads(payload)['polygons'].get('partition')]
    return dict(manual_references=references,id=uuid.uuid4().hex,folder=str(s.folder.resolve()),segment=s.segment,start=start,end=end,seed_index=seed_index,seed=copy.deepcopy(seed),original_cursor=s.cursor,records=records,anchors=anchors,signature=signature(s,start,end))


def run_retry(source, plan: dict, cancel=lambda:False, progress=lambda attempt,frame:None, factory=HorizontalTracker) -> dict:
    """Try forward re-localization, then forward/backward with a wider window."""
    seed=plan['seed'];start=plan['start'];last_error='连续画面仍无法稳定定位'
    inherited=[p for p in seed.get('problems',[]) if p=='起始轮廓待确认']
    reference_bank=None
    def check():
        if cancel():raise InterruptedError('自动重试已取消，未改写标注')
    def make(index,key,level):
        nonlocal reference_bank
        check();tracker=make_tracker(source,factory,index,key['polygons'],key['visible']);tracker.recovery_level=level
        if hasattr(tracker,'scene') and plan.get('manual_references'):
            if reference_bank is None:
                from scripts.aoi_scene_anchor import SceneAnchor
                from scripts.aoi_horizontal import center_height,center_slope
                bank=SceneAnchor()
                for ref in plan['manual_references']:
                    check();p=ref['polygons'];bank.add(tracker.small(source.frame(ref['frame'])),center_height(p,source.size)/2,center_slope(p,source.size))
                reference_bank=bank.refs
            tracker.scene.refs=list(reference_bank)
        return tracker
    for attempt,seconds in ((1,3),(2,8)):
        check();progress(attempt,start)
        try:
            tracker=make(plan['seed_index'],seed,attempt);rows={};stable=0;recovered=None
            limit=min(plan['end'],bisect.bisect_right(source.times,source.times[start]+seconds)-1)
            for i in range(start,limit+1):
                check()
                anchor=plan['anchors'].get(i)
                if anchor:
                    tracker=make(i,anchor,attempt);polygons=anchor['polygons'];problems=[];visible=anchor['visible']
                else:
                    polygons,problems=tracker.update(source.frame(i));visible=seed['visible']
                polygons=validate_polygons(polygons,visible,source.size)
                rows[i]=dict(polygons=copy.deepcopy(polygons),visible=copy.deepcopy(visible),problems=list(problems)+inherited)
                stable=0 if problems else stable+1
                if (i-start)%8==0:progress(attempt,i)
                # Require a real run of clear frames, including at a slice end.
                if stable>=5 and i>=plan['original_cursor']:
                    recovered=i;break
            if recovered is None:continue
            if attempt==2:
                tracker=make(recovered,rows[recovered],2)
                for i in range(recovered-1,start-1,-1):
                    check()
                    if i in plan['anchors']:
                        tracker=make(i,plan['anchors'][i],2);continue
                    p,problems=tracker.update(source.frame(i))
                    if not problems and tracking_failed(rows[i]):
                        rows[i]=dict(polygons=validate_polygons(p,seed['visible'],source.size),visible=copy.deepcopy(seed['visible']),problems=list(inherited))
                    if (i-start)%8==0:progress(attempt,i)
            return dict(success=True,attempts=attempt,end=recovered,rows=rows)
        except (OSError,ValueError,cv2.error) as exc:
            last_error=str(exc)
    return dict(success=False,attempts=2,error=last_error)


def commit_retry(s,plan: dict,result: dict) -> int:
    if not result.get('success'):raise ValueError('重试未成功，不能保存')
    if str(s.folder.resolve())!=plan['folder'] or s.segment!=plan['segment'] or signature(s,plan['start'],plan['end'])!=plan['signature']:
        raise ValueError('标注已变化，自动重试草稿已作废')
    end=result['end']
    if set(result['rows'])!=set(range(plan['start'],end+1)):raise ValueError('重试帧不完整')
    folder=s.folder/'backups';folder.mkdir(exist_ok=True)
    backup=folder/f'before-auto-retry-{s.segment}-{plan["start"]}-{plan["id"][:8]}.sqlite3'
    with sqlite3.connect(backup) as db:s.db.backup(db)
    changed=0
    with s.db:
        for i,r in result['rows'].items():
            old=plan['records'].get(i)
            if i in plan['anchors'] or (old and not tracking_failed(old)):continue
            s._put(i,r['polygons'],r['visible'],r['problems'],tracking=dict(auto_retry=dict(attempts=result['attempts'],start=plan['start'],automatic=True)))
            changed+=1
        s._event('auto_retry',dict(start=plan['start'],end=end,attempts=result['attempts'],changed=changed,backup=str(backup)))
    s.cursor=end;s.tracker=None;s.mode='paused';s.running=False;s.reason='自动重试已找回，继续标注；模糊帧仍保留待复核标记';s._save()
    return changed


def clone_source(source):
    other=copy.copy(source)
    if hasattr(source,'cap'):
        other.cap=cv2.VideoCapture(str(source.path));other.last=-2;other.cached=None
        if not other.cap.isOpened():other.close();raise ValueError('自动重试无法读取视频')
    return other
