"""Recover a boundary from immutable, manually located views of static hardware."""
import cv2
import numpy as np


class SceneAnchor:
    def __init__(self):
        self.refs=[];self.last_conflict=False
        from scripts.aoi_line_cache import scene_sift
        self.detector=scene_sift()
        self.matcher=cv2.BFMatcher()

    def add(self,gray,y,slope):
        h,w=gray.shape;yy,xx=np.ogrid[:h,:w];distance=yy-(y+slope*(xx-w/2))
        # Prefer the bezel, desk and supports. Exclude the moving game above,
        # and the hands/lap far below. Keep raw luminance across slider changes.
        mask=np.uint8((distance>-8)&(distance<100))*255
        keys,descriptors=self.detector.detectAndCompute(gray,mask)
        if descriptors is None or len(keys)<18:return
        self.refs=(self.refs+[dict(points=np.float32([p.pt for p in keys]),descriptors=descriptors,y=y,slope=slope)])[-4:]

    def candidate(self,gray):
        self.last_conflict=False
        if not self.refs:return None
        h,w=gray.shape;keys,descriptors=self.detector.detectAndCompute(gray,None)
        if descriptors is None or len(keys)<18:return None
        points=np.float32([p.pt for p in keys]);options=[]
        for ref in self.refs:
            pairs=self.matcher.knnMatch(ref['descriptors'],descriptors,k=2)
            matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
            # Repeated keyboard keys must not vote multiple times for one point.
            unique={}
            for m in matches:
                if m.trainIdx not in unique or m.distance<unique[m.trainIdx].distance:unique[m.trainIdx]=m
            matches=list(unique.values())
            if len(matches)<18:continue
            a=np.float32([ref['points'][m.queryIdx] for m in matches]);b=np.float32([points[m.trainIdx] for m in matches])
            A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
            if A is None or inliers is None:continue
            keep=inliers.ravel()==1;n=int(keep.sum());ratio=float(keep.mean())
            if n<16 or ratio<.48 or np.ptp(a[keep,0])<w*.24 or np.ptp(a[keep,1])<h*.10:continue
            scale=float(np.linalg.det(A[:,:2]))
            if not .6<scale<1.65:continue
            error=np.linalg.norm(a[keep]@A[:,:2].T+A[:,2]-b[keep],axis=1)
            if float(np.median(error))>1.1:continue
            y,s=ref['y'],ref['slope'];ends=np.array([[w/2-40,y-s*40,1],[w/2+40,y+s*40,1]])@A.T
            if ends[1,0]-ends[0,0]<30:continue
            slope=float((ends[1,1]-ends[0,1])/(ends[1,0]-ends[0,0]));cy=float(ends[0,1]+(w/2-ends[0,0])*slope)
            if abs(slope)>1.2 or not -h*.3<cy<h*1.3:continue
            score=n*ratio/(1+float(np.median(error)))
            options.append(dict(y=cy,slope=slope,source='scene_anchor',x=w/2,quality=ratio,score=score,inliers=n))
        if not options:return None
        options.sort(key=lambda p:p['score'],reverse=True);best=options[0]
        # Equally strong but inconsistent manual views require inspection.
        if len(options)>1 and options[1]['score']>.8*best['score'] and (abs(options[1]['y']-best['y'])>15 or abs(options[1]['slope']-best['slope'])>.15):
            self.last_conflict=True;return None
        return best
