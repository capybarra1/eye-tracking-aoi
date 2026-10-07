"""Keep sharp references through brief motion blur; uncertain frames stay flagged."""
from collections import deque
import cv2
import numpy as np


class BlurGate:
    def __init__(self, frame):
        self.scores=deque([self.score(frame)],maxlen=30)
        self.active=False
    @staticmethod
    def score(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray,cv2.CV_32F).var())
    def blurred(self,frame):
        score=self.score(frame)
        baseline=float(np.median(self.scores))
        # Relative to this recording, without treating a naturally low-texture
        # or black initialization image as evidence of motion blur.
        self.active=baseline>8 and score<baseline*(.50 if self.active else .35)
        return self.active
    def accept(self,frame):
        self.scores.append(self.score(frame))


class ScreenRecovery:
    """Sparse bezel/seam references, used only after normal tracking fails.

    Reacquisition needs distributed matches and agreement on two clear frames.
    Geometry is validated on the visible lower boundary by the caller, since
    extrapolating the full image corners exaggerates perspective during turns.
    """
    def __init__(self):
        self.references=[];self.pending=None;self.attempts=0
        self.detector=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.015)
    def remember(self,gray,transform,points):
        self.references.append([gray.copy(),transform.copy(),np.array(points,np.float32),None])
        if len(self.references)>3:self.references.pop(1)
    def _features(self,ref):
        gray,_,points,cached=ref
        if cached is None:
            mask=np.zeros(gray.shape,np.uint8)
            cv2.polylines(mask,[np.rint(points).astype(np.int32)],False,255,45)
            for x,y in np.rint(points[1:-1]).astype(int):cv2.line(mask,(x,0),(x,y),255,25)
            keys,desc=self.detector.detectAndCompute(gray,mask)
            ref[3]=(keys,desc)
        return ref[3]
    def locate(self,gray,project):
        self.attempts+=1
        # Expensive full-image matching only while recovery is needed.
        if self.pending is None and self.attempts%3!=1:return None
        keys,desc=self.detector.detectAndCompute(gray,None)
        if desc is None:self.pending=None;return None
        for ref in reversed(self.references):
            oldkeys,oldDesc=self._features(ref)
            if oldDesc is None or len(keys)<2:continue
            pairs=cv2.BFMatcher().knnMatch(oldDesc,desc,k=2)
            unique={}
            for m in sorted([a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance],key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
            matches=list(unique.values())
            if len(matches)<20:continue
            old=np.array([oldkeys[m.queryIdx].pt for m in matches],np.float32)
            new=np.array([keys[m.trainIdx].pt for m in matches],np.float32)
            H,mask=cv2.findHomography(old,new,cv2.RANSAC,3,maxIters=2000,confidence=.995)
            if H is None or mask is None or not np.isfinite(H).all():continue
            keep=mask.ravel().astype(bool);n=int(keep.sum());ratio=float(keep.mean())
            h,w=gray.shape
            if n<20 or ratio<.65:continue
            span=np.ptp(old[keep],axis=0)
            if span[0]<w*.2 or span[1]<h*.10:continue
            # Reject folds, poles and extreme scaling within the observed image.
            corners=np.array([[[0,0],[w,0],[w,h],[0,h]]],np.float32)
            denominators=np.c_[corners[0],np.ones(4)]@H[2]
            moved=cv2.perspectiveTransform(corners,H)[0]
            area=cv2.contourArea(moved,oriented=True)/(w*h)
            if np.any(denominators<=0) or not cv2.isContourConvex(moved) or not .3<area<4:continue
            proposed=H@ref[1];proposed/=proposed[2,2]
            try:points=project(proposed)
            except ValueError:continue
            p=np.asarray(points)
            if not np.isfinite(p).all() or p[:,1].min()<-h or p[:,1].max()>2*h:continue
            if self.pending is not None:
                xs=np.linspace(0,w-1,21)
                delta=np.max(np.abs(np.interp(xs,p[:,0],p[:,1])-np.interp(xs,self.pending[:,0],self.pending[:,1])))
                if delta<18:
                    self.pending=None
                    return proposed,dict(status='tracked',confidence=ratio,inliers=n,method='bezel_recovery',recovered=True)
            self.pending=p.copy()
            return None
        self.pending=None
        return None
