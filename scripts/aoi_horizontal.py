"""Fast coarse AOIs: one central bezel line, two regions."""
from __future__ import annotations
import cv2
import numpy as np
from scripts.aoi_upper_edge import UpperEdge,enhance_dark

DEFAULT_GAP=48.


def partition_polygons(size, y, gap=None, slope=0.):
    w,h=size;y=float(y);gap=float(min(DEFAULT_GAP,h/2) if gap is None else gap);slope=float(slope)
    if not np.isfinite([y,gap,slope]).all() or not 0<=gap<=h/2 or abs(y)>h*4 or abs(slope)>1.5:
        raise ValueError('分界线或间隔带数值无效')
    # Keep a nondegenerate off-frame rectangle when the lower area disappears.
    left=y-slope*(w-1)/2;right=y+slope*(w-1)/2
    bottom=max(float(h-1),left+gap+2,right+gap+2)
    return dict(screen=[[[float(x),y+slope*(x-(w-1)/2)] for x in (0,(w-1)/3,2*(w-1)/3,w-1)]],
                tablet=[[[0.,left+gap],[w-1.,right+gap],[w-1.,bottom],[0.,bottom]]],
                partition=dict(mode='straight_line',gap_px=gap))


def center_height(polygons,size):
    p=np.float32(polygons['screen'][0])
    return float(np.interp((size[0]-1)/2,p[:,0],p[:,1]))


def center_slope(polygons,size):
    p=np.float32(polygons['screen'][0]);i=int(np.clip(np.searchsorted(p[:,0],(size[0]-1)/2)-1,0,len(p)-2))
    return float((p[i+1,1]-p[i,1])/(p[i+1,0]-p[i,0]))


class HorizontalTracker:
    """Calibrated upper/lower bezels and tablet; uncertain frames never teach templates."""
    def __init__(self,frame,polygons,visible,image_adjustments=None):
        self.image_adjustments=image_adjustments
        from scripts.aoi_edge_lock import strip,tablet_panels
        self.size=(frame.shape[1],frame.shape[0]);self.visible=visible
        self.gap=polygons['partition']['gap_px'];self.y=center_height(polygons,self.size);self.slope=center_slope(polygons,self.size)
        self.raw_gray=self.small(frame);self.gray,self.low_light=self.enhanced(self.raw_gray);self.hint_y=self.y/2;self.hint_slope=self.slope
        self.misses=0;self.pending=None;self.last_stats={};self.reference_source='screen'
        self.anchor_x=self.gray.shape[1]/2;self.tablet_x=self.anchor_x
        self.lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        patch=strip(self.gray,self.y/2,self.slope);self.bank=[patch] if patch is not None else []
        self.tablet_reference=None
        self.calibrate_tablet(tablet_panels(self.gray),self.y/2,self.slope,self.gap/2)
        self.upper=UpperEdge();self.upper.calibrate(self.gray,self.y/2,self.slope);self.steps=0
        from scripts.aoi_scene_anchor import SceneAnchor
        self.scene=SceneAnchor();self.scene.add(self.raw_gray,self.y/2,self.slope)

    def enhanced(self,raw):
        from scripts.aoi_image_adjustments import enhance, DEFAULT
        if self.image_adjustments and self.image_adjustments!=DEFAULT:
            # Manual adjustment replaces automatic brightening; physical bezel
            # checks still use original pixels, so lifting shadows cannot hide it.
            return enhance(raw,self.image_adjustments),bool(np.median(raw)<75)
        return enhance_dark(raw)

    @staticmethod
    def small(frame):
        from scripts.aoi_line_cache import scene_gray
        return scene_gray(frame)

    def calibrate_tablet(self,panels,y,slope,gap):
        matches=[p for p in panels if abs(p['y']-(y+gap))<18 and abs(p['slope']-slope)<.2 and p['y']>y+max(8,gap*.5)]
        if not matches:return False
        p=min(matches,key=lambda p:abs(p['y']-y-gap))
        self.tablet_reference=dict(offset=y+gap-p['y'],slope_offset=slope-p['slope'],width=p['width'],height=p['height'])
        return True

    def add_reference(self,frame,polygons):
        from scripts.aoi_edge_lock import strip,tablet_panels
        gray,_=self.enhanced(self.small(frame));y=center_height(polygons,self.size)/2;slope=center_slope(polygons,self.size)
        p=strip(gray,y,slope)
        if p is not None:self.bank=(self.bank+[p])[-3:]
        if self.tablet_reference is None:self.calibrate_tablet(tablet_panels(gray),y,slope,polygons['partition']['gap_px']/2)
        if self.upper.reference is None:self.upper.calibrate(gray,y,slope)
        self.scene.add(self.small(frame),y,slope)

    def motion(self,gray):
        h,w=gray.shape;y=self.hint_y;slope=self.hint_slope
        # Follow the last observed physical edge, not the entire screen/game.
        anchor_y=y+(self.gap/2 if self.reference_source=='tablet' else -self.upper.reference['height'] if self.reference_source=='screen_top' and self.upper.reference else 0)
        yy,xx=np.ogrid[:h,:w]
        cx=self.tablet_x if self.reference_source=='tablet' else self.upper.reference['x'] if self.reference_source=='screen_top' and self.upper.reference else self.anchor_x
        mask=np.uint8((abs(yy-(anchor_y+slope*(xx-w/2)))<18)&(abs(xx-cx)<w*.23))*255
        p=cv2.goodFeaturesToTrack(self.gray,90,.02,5,mask=mask)
        if p is None or len(p)<10:return y,slope,False
        opts=dict(winSize=(21,21),maxLevel=3)
        q,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,p,None,**opts)
        if q is None:return y,slope,False
        back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,q,None,**opts)
        if back is None:return y,slope,False
        good=(ok.ravel()==1)&(valid.ravel()==1)&(np.linalg.norm((back-p).reshape(-1,2),axis=1)<.8)
        a,b=p.reshape(-1,2)[good],q.reshape(-1,2)[good]
        if len(a)<10 or np.ptp(a[:,0])<w*.12:return y,slope,False
        A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5)
        if A is None or inliers is None or inliers.mean()<.65 or not .8<np.linalg.det(A[:,:2])<1.25:return y,slope,False
        ends=np.array([[w/2-30,y-slope*30,1],[w/2+30,y+slope*30,1]])@A.T
        if ends[1,0]-ends[0,0]<20:return y,slope,False
        new_slope=float(np.diff(ends[:,1])[0]/np.diff(ends[:,0])[0])
        new_y=float(ends[0,1]+(w/2-ends[0,0])*new_slope)
        if abs(new_y-y)>60 or abs(new_slope-slope)>.25:return y,slope,False
        point=np.array([self.anchor_x,anchor_y+slope*(self.anchor_x-w/2),1])@A.T
        self.proposed_anchor_x=float(point[0]) if abs(float(point[0])-self.anchor_x)<70 else self.anchor_x
        return new_y,new_slope,True

    def broad_motion(self,raw):
        """Use the desk/bezel band to bridge head turns, as a search hint only.

        The old narrow bezel mask loses all points in motion blur. Tracking a
        wider static band keeps the search near the physical edge instead of
        resetting to the stale pre-turn location. Weak fits never approve AOIs.
        """
        h,w=raw.shape;y=self.hint_y;slope=self.hint_slope
        yy,xx=np.ogrid[:h,:w];distance=yy-(y+slope*(xx-w/2))
        mask=np.uint8((distance>-15)&(distance<100))*255
        p=cv2.goodFeaturesToTrack(self.raw_gray,220,.008,5,mask=mask)
        if p is None or len(p)<12:return None
        opts=dict(winSize=(31,31),maxLevel=4)
        q,ok,error=cv2.calcOpticalFlowPyrLK(self.raw_gray,raw,p,None,**opts)
        if q is None:return None
        back,valid,_=cv2.calcOpticalFlowPyrLK(raw,self.raw_gray,q,None,**opts)
        if back is None:return None
        good=(ok.ravel()==1)&(valid.ravel()==1)&(error.ravel()<35)&(np.linalg.norm((back-p).reshape(-1,2),axis=1)<1.5)
        a,b=p.reshape(-1,2)[good],q.reshape(-1,2)[good]
        if len(a)<12:return None
        A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
        if A is None or inliers is None:return None
        keep=inliers.ravel()==1
        if keep.sum()<8 or keep.mean()<.22 or np.ptp(a[keep,0])<w*.3 or np.ptp(a[keep,1])<h*.08:return None
        if not .7<np.linalg.det(A[:,:2])<1.4:return None
        ends=np.array([[w/2-40,y-slope*40,1],[w/2+40,y+slope*40,1]])@A.T
        if ends[1,0]-ends[0,0]<30:return None
        s=float(np.diff(ends[:,1])[0]/np.diff(ends[:,0])[0]);cy=float(ends[0,1]+(w/2-ends[0,0])*s)
        if abs(cy-y)>80 or abs(s-slope)>.35:return None
        return cy,s

    def update(self,frame):
        from scripts.aoi_edge_lock import strip,similarity,tablet_panels
        raw=self.small(frame);gray,self.low_light=self.enhanced(raw);h,w=gray.shape;hint,slope,flow=self.motion(gray);self.steps+=1
        wide=None
        if not flow or self.misses:
            wide=self.broad_motion(raw)
            if wide is not None:hint,slope=wide
        recovery=getattr(self,'recovery_level',0)
        x0=max(0,int(min(self.anchor_x-w*.15,w*.35)));x1=min(w,int(max(self.anchor_x+w*.15,w*.65)));crop=gray[:,x0:x1]
        detected=self.lsd.detect(crop)[0] if x1-x0>30 and 0<=self.anchor_x<w else None;options=[]
        radius=8 if flow and not self.misses else min(36,18+self.misses*2)
        if recovery:radius=96 if recovery==1 else h
        if detected is not None:
            for a,b,c,d in detected.reshape(-1,4):
                if a>c:a,b,c,d=c,d,a,b
                if c-a<20 or abs(d-b)/(c-a)>1.:continue
                if not (a-8<=self.anchor_x-x0<=c+8 or a-8<=w/2-x0<=c+8):continue
                xs=np.linspace(a+1,c-1,20);ys=b+(xs-a)*(d-b)/(c-a)
                if ys.min()<4 or ys.max()>h-5:continue
                xx=np.clip(np.rint(xs).astype(int),0,crop.shape[1]-1);yy=np.rint(ys).astype(int)
                above=(crop[yy-2,xx].astype(float)+crop[yy-3,xx])/2
                below=(crop[yy+2,xx].astype(float)+crop[yy+3,xx])/2
                raw_above=(raw[yy-2,xx+x0].astype(float)+raw[yy-3,xx+x0])/2
                raw_below=(raw[yy+2,xx+x0].astype(float)+raw[yy+3,xx+x0])/2
                support=float(np.mean((raw_above<45)&(raw_below-raw_above>(4 if self.low_light else 12))&(below-above>(9 if self.low_light else 12))))
                y=float(b+((w/2-x0)-a)*(d-b)/(c-a));s=float((d-b)/(c-a));distance=abs(y-hint)
                if support<.7 or distance>radius or abs(s-slope)>(.4 if recovery else .14):continue
                candidate_x=float(x0+(a+c)/2)
                match=max(similarity(strip(gray,y,s,candidate_x),self.bank),similarity(strip(gray,y,s),self.bank))
                if match<(.55 if recovery else .28):continue
                quality=min(1,(c-a)/55)*support*max(0,match)
                score=quality/(1+distance*.12+abs(s-slope)*3)
                options.append(dict(score=score,y=y,slope=s,source='screen',quality=quality,appearance=match,x=candidate_x))
        panels=tablet_panels(gray);tablet=[]
        ref=self.tablet_reference
        if ref:
            for p in panels:
                scale=p['width']/ref['width']
                if not .6<scale<1.65 or not .6<p['height']/ref['height']<1.65:continue
                y=p['y']+ref['offset']*scale-self.gap/2;s=p['slope']+ref['slope_offset'];distance=abs(y-hint)
                if distance>radius+8 or abs(s-slope)>(.4 if recovery else .18) or p['y']<y+max(8,self.gap*.2):continue
                tablet.append(dict(score=p['quality']/(1+distance*.1),y=y,slope=s,source='tablet',quality=p['quality'],x=p['x']))
        options.sort(key=lambda p:p['score'],reverse=True);tablet.sort(key=lambda p:p['score'],reverse=True)
        screen=options[0] if options else None;tab=tablet[0] if tablet else None
        # Correlated but incompatible edges are uncertainty, not a vote by length.
        conflict=bool(screen and tab and (abs(screen['y']-tab['y'])>16 or abs(screen['slope']-tab['slope'])>.16))
        ambiguous=bool(len(options)>1 and options[1]['score']>screen['score']*.8 and abs(options[1]['y']-screen['y'])>8)
        top=self.upper.candidate(gray,hint,slope,radius,recovery)
        lower=screen or tab
        top_conflict=bool(top and lower and (abs(top['y']-lower['y'])>16 or abs(top['slope']-lower['slope'])>.16))
        conflict=conflict or top_conflict
        candidate=None if conflict or ambiguous else screen or tab or top
        anchor=None
        if self.scene.refs and (self.misses or not flow or conflict or ambiguous or self.steps%12==0 or self.reference_source=='scene_anchor'):
            anchor=self.scene.candidate(raw)
            if self.scene.last_conflict:candidate=None;conflict=True
        if anchor:
            # The manually calibrated scene is absolute evidence: do not follow
            # a tablet UI/keyboard line merely because it is closer to stale AOI.
            agrees_edge=candidate and abs(candidate['y']-anchor['y'])<10 and abs(candidate['slope']-anchor['slope'])<.08
            if not agrees_edge:candidate=anchor
            conflict=False;ambiguous=False
        problems=['屏幕短暂失跟'];confirmed=False
        if candidate:
            y,s=candidate['y'],candidate['slope']
            switched=candidate['source']!=self.reference_source
            stable_edge=getattr(self,'last_candidate',None)
            stationary=bool(stable_edge and not self.misses and stable_edge['source']==candidate['source'] and abs(y-stable_edge['y'])<3 and abs(s-stable_edge['slope'])<.03)
            needs_confirmation=switched or (not flow and not stationary) or abs(y-hint)>7 or abs(s-slope)>.08
            agrees=self.pending is not None and self.pending['source']==candidate['source'] and abs(y-self.pending['y'])<12 and abs(s-self.pending['slope'])<.10
            if not needs_confirmation or agrees:
                self.anchor_x=candidate['x'] if candidate['source']=='screen' else getattr(self,'proposed_anchor_x',self.anchor_x) if flow else self.anchor_x
                self.y=y*2;self.slope=s;self.reference_source=candidate['source'];self.misses=0;self.pending=None;problems=[];confirmed=True
                self.last_candidate=candidate
                if candidate['source']=='tablet':self.tablet_x=candidate['x']
                if screen and tab and screen['appearance']>.45:
                    patch=strip(gray,y,s,self.anchor_x)
                    if patch is not None:self.bank=(self.bank[:2]+[patch])[-3:]
                if screen and ref is None and screen['appearance']>.45:self.calibrate_tablet(panels,y,s,self.gap/2)
                if screen and screen['appearance']>.55 and self.steps%30==0:
                    self.upper.calibrate(gray,y,s)
            else:self.pending=candidate
        else:self.pending=None
        if not confirmed:self.misses+=1
        # Keep measured geometry if evidence is absent. Propagate only the search
        # hint through a brief gap; never export an unverified edge as trustworthy.
        self.hint_y=self.y/2 if confirmed else hint
        self.hint_slope=self.slope if confirmed else slope
        if self.misses>12 and wide is None and not flow:self.hint_y=self.y/2;self.hint_slope=self.slope
        self.gray=gray;self.raw_gray=raw
        self.last_stats={'screen':dict(method='dual_edge_lock',status='uncertain' if problems else 'tracked',
                                      reference=candidate['source'] if confirmed else None,conflict=conflict or ambiguous,broad_motion=wide is not None,scene_anchor_inliers=anchor['inliers'] if anchor else 0,
                                      tablet_reference=bool(self.tablet_reference),upper_reference=bool(self.upper.reference),low_light_enhanced=self.low_light,anchor_x=self.anchor_x*2)}
        return partition_polygons(self.size,self.y,self.gap,self.slope),problems
