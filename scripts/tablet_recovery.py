"""Tablet optical flow with retained appearance for offscreen re-entry.

Missing frames remain flagged; only geometrically supported appearance matches
can replace a lost polygon. Explicit manual visibility remains user-controlled.
"""
import cv2
import numpy as np
from scripts.fast_aoi_tracker import (RegionTrack, prepare_gray, reseed_region,
                                      transform_polygon)


def visible_fraction(polygon, size):
    p=np.asarray(polygon,np.float32)
    if not np.isfinite(p).all() or not cv2.isContourConvex(p):return 0.
    w,h=size
    area=abs(cv2.contourArea(p))
    if area<4:return 0.
    frame=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    return max(0.,min(1.,cv2.intersectConvexConvex(p,frame)[0]/area))


def estimate_tablet_motion(previous, current, points, polygon, size):
    failed=(np.eye(3,dtype=np.float32),None,'lost',0.,0,0)
    if points is None or len(points)<12:return failed
    lk=dict(winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,good,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,**lk)
    if forward is None:return failed
    backward,back_good,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,**lk)
    if backward is None:return failed
    valid=(good.ravel()==1)&(back_good.ravel()==1)
    valid &= np.linalg.norm((backward-points).reshape(-1,2),axis=1)<1.5
    old,new=points[valid].reshape(-1,2),forward[valid].reshape(-1,2)
    if len(old)<12:return failed
    affine,mask=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1500,confidence=.995)
    if affine is None or mask is None or not np.isfinite(affine).all():return failed
    keep=mask.ravel().astype(bool);count=int(keep.sum());confidence=float(keep.mean())
    if count<12 or confidence<.55:return failed
    H=np.vstack([affine,[0,0,1]]).astype(np.float32)
    moved=transform_polygon(polygon,H)
    area=abs(cv2.contourArea(moved))/max(abs(cv2.contourArea(polygon)),1)
    support=cv2.contourArea(cv2.convexHull(old[keep]))/max(abs(cv2.contourArea(polygon)),1)
    if not np.isfinite(moved).all() or not cv2.isContourConvex(moved):return failed
    if not .65<area<1.5 or support<.08 or np.max(np.linalg.norm(moved-polygon,axis=1))>max(size)*.3:return failed
    return H,new[keep].reshape(-1,1,2),'tracked',confidence,count,len(old)


class TabletRecovery:
    def __init__(self, frame, polygon):
        self.size=(frame.shape[1],frame.shape[0])
        self.polygon=np.array(polygon,np.float32).copy()
        self.gray=prepare_gray(frame)
        self.detector=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.025)
        self.references=[];self.steps=0;self.pending=None
        self.parked=False
        self.allow_reference_update=True
        self.candidate_filter=lambda polygon:True
        self.quality={}
        self.searching=visible_fraction(self.polygon,self.size)<.4
        self.waiting=self.searching
        self.region=RegionTrack('tablet',0,self.polygon.copy())
        reseed_region(self.gray,self.region,self.size)
        self.add_reference(frame,self.polygon)

    def add_reference(self, frame, polygon):
        p=np.array(polygon,np.float32)
        if visible_fraction(p,self.size)<.85:return
        mask=np.zeros(frame.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],255)
        # Include the physical bezel, without a wide unrelated background ring.
        mask=cv2.dilate(mask,np.ones((13,13),np.uint8))
        gray=prepare_gray(frame)
        keypoints,descriptors=self.detector.detectAndCompute(gray,mask)
        if descriptors is None or len(keypoints)<14:return
        points=np.array([k.pt for k in keypoints],np.float32)
        self.references.append((points,descriptors,p.copy()))
        if len(self.references)>3:self.references.pop(1)  # Keep the original reference.

    def _locate(self, gray):
        keys,desc=self.detector.detectAndCompute(gray,None)
        if desc is None or len(keys)<14:return None
        current=np.array([k.pt for k in keys],np.float32)
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        for points,reference,polygon in reversed(self.references):
            pairs=matcher.knnMatch(reference,desc,k=2)
            matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.70*b.distance]
            unique={}
            for m in sorted(matches,key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
            matches=list(unique.values())
            if len(matches)<12:continue
            old=np.array([points[m.queryIdx] for m in matches]);new=np.array([current[m.trainIdx] for m in matches])
            H,mask=cv2.findHomography(old,new,cv2.RANSAC,2.5,maxIters=1500,confidence=.995)
            if H is None or mask is None or not np.isfinite(H).all():continue
            keep=mask.ravel().astype(bool)
            if keep.sum()<12 or keep.mean()<.65:continue
            support=cv2.contourArea(cv2.convexHull(old[keep]))/max(abs(cv2.contourArea(polygon)),1)
            if support<.12:continue
            moved=transform_polygon(polygon,H)
            if not np.isfinite(moved).all() or not cv2.isContourConvex(moved):continue
            ratio=abs(cv2.contourArea(moved))/max(abs(cv2.contourArea(polygon)),1)
            if not .4<ratio<2.5 or visible_fraction(moved,self.size)<.6:continue
            edges=np.linalg.norm(np.roll(moved,-1,axis=0)-moved,axis=1)
            if edges.min()<15 or edges.max()/edges.min()>6:continue
            if not self.candidate_filter(moved):continue
            return moved
        return None

    def update(self, frame):
        gray=prepare_gray(frame);self.steps+=1
        if self.searching and not self.parked and visible_fraction(self.polygon,self.size)<.85:self.waiting=True
        if not self.searching:
            H,points,status,confidence,inliers,tracked=estimate_tablet_motion(
                self.gray,gray,self.region.points,self.polygon,self.size)
            self.quality=dict(confidence=confidence,inliers=inliers,tracked_points=tracked)
            if status=='tracked' and self.candidate_filter(transform_polygon(self.polygon,H)):
                self.polygon=transform_polygon(self.polygon,H)
                self.region.polygon=self.polygon.copy();self.region.points=points
                self.gray=gray
                if visible_fraction(self.polygon,self.size)>=.4:
                    if points is None or len(points)<45 or self.steps%18==0:
                        reseed_region(gray,self.region,self.size)
                    if self.allow_reference_update and self.steps%50==0 and confidence>.75:self.add_reference(frame,self.polygon)
                    return self.polygon.copy(),'tracked'
            self.searching=True
            self.waiting=visible_fraction(self.polygon,self.size)<.85
        # Search at 1/5 frame rate while absent, but confirm a proposed match on
        # the very next frame before accepting a large re-entry displacement.
        if self.pending is not None or self.steps%5==1:
            found=self._locate(gray) if self.references else None
            if found is not None and self.pending is not None and np.max(np.linalg.norm(found-self.pending,axis=1))<20:
                self.polygon=found;self.region.polygon=found.copy();self.gray=gray
                reseed_region(gray,self.region,self.size)
                self.pending=None;self.searching=False;self.waiting=False;self.parked=False
                return found.copy(),'reacquired'
            self.pending=found
        status='waiting' if self.waiting else 'lost'
        # A failed estimate is not a visible AOI. Keep the appearance bank for
        # re-entry, but park the unused outline below the frame.
        self.polygon[:,1]+=max(0.,self.size[1]+8-float(self.polygon[:,1].min()))
        self.parked=True
        self.region.polygon=self.polygon.copy();self.region.points=None
        return self.polygon.copy(),status
