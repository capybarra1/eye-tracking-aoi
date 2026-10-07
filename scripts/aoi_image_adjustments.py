"""Per-recording photometric controls; raw frames and AOI coordinates stay unchanged."""
import math
import cv2
import numpy as np

DEFAULT = dict(brightness=0, contrast=100)


def validate(settings):
    if not isinstance(settings, dict):raise ValueError('画面参数格式错误')
    result={}
    for key,lo,hi in [('brightness',-50,100),('contrast',50,200)]:
        value=settings.get(key,DEFAULT[key])
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('亮度范围 -50～100，对比度范围 50～200')
        result[key]=float(value)
    return result


def enhance(frame, settings=None):
    settings=settings or DEFAULT
    if settings==DEFAULT:return frame
    # Gamma lifts shadows without adding a grey veil to black borders.
    x=np.arange(256,dtype=np.float32)/255
    x=np.power(x,2**(-settings['brightness']/100))
    x=(x-.5)*(settings['contrast']/100)+.5
    return cv2.LUT(frame,np.uint8(np.clip(x*255,0,255)))


def display_frame(source,index):
    return enhance(source.frame(index),getattr(source,'image_adjustments',None))


def apply_settings(session,settings):
    settings=validate(settings)
    if session.running:raise ValueError('请先暂停，再调整画面')
    session.source.image_adjustments=settings
    session.tracker=None
    for name in ('line_reference_cache','screen_edge_references'):
        if hasattr(session,name):getattr(session,name).clear()
    session._set('image_adjustments',settings)
    session.db.commit()


class AdjustedTracker:
    def __init__(self,factory,frame,polygons,visible,settings):
        self.settings=settings
        self.inner=factory(enhance(frame,settings),polygons,visible)
    def __getattr__(self,name):return getattr(self.inner,name)
    def update(self,frame):return self.inner.update(enhance(frame,self.settings))
    def add_tablet_reference(self,frame,polygon):
        return self.inner.add_tablet_reference(enhance(frame,self.settings),polygon)


def make_tracker(source,factory,index,polygons,visible):
    from scripts.aoi_horizontal import HorizontalTracker
    settings=getattr(source,'image_adjustments',DEFAULT)
    frame=source.frame(index)
    if polygons.get('partition') and getattr(factory,'__name__','') in ('HorizontalTracker','Tracker'):
        return HorizontalTracker(frame,polygons,visible,image_adjustments=settings)
    if settings!=DEFAULT:return AdjustedTracker(factory,frame,polygons,visible,settings)
    return factory(frame,polygons,visible)
