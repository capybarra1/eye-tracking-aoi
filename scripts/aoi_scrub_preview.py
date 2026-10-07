"""Read-only, exact-frame preview. The caller owns the session lock."""
import base64
from scripts.aoi_image_adjustments import display_frame, apply_settings
import cv2
from scripts.supervised_aoi import four_points, coverage_for


def preview_frame(server, index: int) -> dict:
    s=server.session
    if not s.first_index<=index<=s.last_index:raise ValueError('预览位置超出当前切片')
    if s.running or server.batch.active or server.review.active:raise ValueError('请先暂停，再拖动进度条复核')
    record=s.get_record(index)
    draft=server.review.draft
    is_draft=bool(draft and draft['status']=='ready' and draft['folder']==str(s.folder.resolve()) and draft['segment']==s.segment and index in draft['records'])
    if is_draft:record=draft['records'][index]
    key=(s.source,index);cache=server.scrub_images
    if key not in cache:
        ok,data=cv2.imencode('.jpg',display_frame(s.source,index),[cv2.IMWRITE_JPEG_QUALITY,75])
        if not ok:raise ValueError('预览画面解码失败')
        cache[key]='data:image/jpeg;base64,'+base64.b64encode(data).decode()
        while len(cache)>40:cache.popitem(last=False)
    cache.move_to_end(key)
    polygons=record['polygons'] if record else None
    visible=record['visible'] if record else dict(screen=False,tablet=False)
    return dict(cursor=index,time_s=s.source.times[index],segment=s.segment,revision=server.revision,
                project_id=server.projects.current if server.projects else '',image=cache[key],size=s.source.size,
                record_exists=bool(record),polygons=polygons,visible=visible,preview=is_draft,
                screen_points=four_points(polygons['screen'],s.source.size) if record else None,
                coverage=coverage_for(polygons['screen'],s.source.size) if record and visible['screen'] else [],
                problems=record.get('problems',[]) if record else [])
