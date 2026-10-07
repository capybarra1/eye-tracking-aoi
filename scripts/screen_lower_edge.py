"""Re-detect physical lower borders from image edges and retained manual views.

Motion tracking supplies a search hint only. Long dark-bezel transitions verify
matches; changing game content and tablet pixels cannot be reference features.
"""
from __future__ import annotations
import cv2
import numpy as np


class LowerEdgeRecovery:
    def __init__(self, size, lock_edges):
        self.size=size;self.lock_edges=lock_edges
        self.detector=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.012)
        self.references=[];self.pending=None;self.steps=0
        self.flow_gray=None;self.flow_boundary=None

    @staticmethod
    def edges(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        lines=cv2.createLineSegmentDetector().detect(gray)[0]
        if lines is None:return []
        h,w=gray.shape;result=[]
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            if x1>x2:x1,y1,x2,y2=x2,y2,x1,y1
            dx=x2-x1
            if dx<max(45,w*.055) or abs(y2-y1)/dx>.8:continue
            xs=np.linspace(x1+3,x2-3,max(8,int(dx/3)));ys=y1+(xs-x1)*(y2-y1)/dx
            if ys.min()<7 or ys.max()>h-8:continue
            def sample(offset):return gray[np.rint(ys+offset).astype(int),np.rint(xs).astype(int)].astype(float)
            above=np.mean([sample(d) for d in (-2,-4,-6)],axis=0)
            below=np.mean([sample(d) for d in (2,4,6)],axis=0)
            support=float(np.mean((above<45)&(below-above>15)))
            if support>=.65:result.append((np.float32([[x1,y1],[x2,y2]]),support))
        return result

    def refine(self, frame, points, tablets=(), lines=None, radius=24, min_coverage=.10):
        """Snap each panel to an observed bezel, retaining hidden panel slopes."""
        p=np.float32(points);w,h=self.size
        lines=self.edges(frame) if lines is None else lines
        equations=[];targets=[];covered=0.;panels=[]
        for i,(a,b) in enumerate(zip(p[:-1],p[1:])):
            width=float(b[0]-a[0])
            if width<45:continue
            options=[]
            for edge,confidence in lines:
                lo=max(float(a[0]),float(edge[0,0]));hi=min(float(b[0]),float(edge[1,0]))
                if hi-lo<45:continue
                xs=np.linspace(lo,hi,12);ys=np.interp(xs,edge[:,0],edge[:,1]);hint=np.interp(xs,p[:,0],p[:,1])
                if any(cv2.pointPolygonTest(np.float32(t),(float(x),float(y)),False)>=0 for t in tablets for x,y in zip(xs,ys)):continue
                distance=float(np.median(np.abs(ys-hint)))
                slope=float((edge[1,1]-edge[0,1])/(edge[1,0]-edge[0,0]))
                if distance>radius or abs(slope-(b[1]-a[1])/width)>.16:continue
                options.append((float(hi-lo)*confidence/(1+distance*.08),lo,hi,edge,distance))
            if not options:continue
            _,lo,hi,edge,distance=max(options,key=lambda x:x[0])
            # A short exposed part may validate a panel but cannot determine the
            # slope of its whole width: require substantial horizontal support.
            if hi-lo<min(100,width*.28):continue
            panels.append(i);covered+=hi-lo
            for x in np.linspace(lo,hi,8):
                row=np.zeros(4);t=(x-a[0])/width;row[i]=1-t;row[i+1]=t
                equations.append(row);targets.append(np.interp(x,edge[:,0],edge[:,1]))
        if covered<w*min_coverage:return None
        # Keep invisible spans from following an unrelated line; shared endpoints
        # provide continuity between the three independently observed borders.
        A=list(equations);y=list(targets)
        for i in range(3):
            if i not in panels:
                row=np.zeros(4);row[i]=-1;row[i+1]=1;A.append(row);y.append(p[i+1,1]-p[i,1])
        for i in range(4):
            row=np.zeros(4);row[i]=.18;A.append(row);y.append(p[i,1]*.18)
        candidate=p.copy();candidate[:,1]=np.linalg.lstsq(np.asarray(A),np.asarray(y),rcond=None)[0]
        if np.max(np.abs(candidate[:,1]-p[:,1]))>radius*1.5:return None
        residual=float(np.median(np.abs(np.asarray(equations)@candidate[:,1]-targets)))
        if residual>5:return None
        return candidate,dict(edge_coverage=covered/w,edge_residual=residual,edge_panels=panels)

    def remember_flow(self, frame, points):
        self.flow_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.flow_boundary=np.float32(points).copy()

    def follow(self, frame, tablets=()):
        """Track the physical bezel band, even inside the generic wheel mask."""
        if self.flow_gray is None:return None
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);w,h=self.size;p=self.flow_boundary
        dy=np.arange(h)[:,None]-np.interp(np.arange(w),p[:,0],p[:,1])
        mask=np.uint8((dy>=-5)&(dy<=50))*255
        for t in tablets:cv2.fillPoly(mask,[np.int32(t)],0)
        points=cv2.goodFeaturesToTrack(self.flow_gray,400,.01,5,mask=mask,blockSize=5)
        if points is None or len(points)<16:return None
        options=dict(winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
        new,good,_=cv2.calcOpticalFlowPyrLK(self.flow_gray,gray,points,None,**options)
        if new is None:return None
        back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.flow_gray,new,None,**options)
        if back is None:return None
        keep=(good.ravel()==1)&(valid.ravel()==1)&(np.linalg.norm((back-points).reshape(-1,2),axis=1)<1.)
        a,b=points[keep].reshape(-1,2),new[keep].reshape(-1,2)
        if len(a)<16:return None
        A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=1000,confidence=.995)
        if A is None or inliers is None:return None
        keep=inliers.ravel().astype(bool);n=int(keep.sum());ratio=float(keep.mean())
        if n<16 or ratio<.65 or np.ptp(a[keep,0])<w*.18:return None
        H=np.vstack([A,[0,0,1]]);scale=float(np.linalg.det(A[:,:2]))
        if not .8<scale<1.25:return None
        moved=cv2.perspectiveTransform(p[None],H)[0]
        if np.max(np.linalg.norm(moved-p,axis=1))>w*.12:return None
        try:moved=self.lock_edges(moved.tolist(),self.size)
        except ValueError:return None
        refined=self.refine(frame,moved,tablets,radius=16,min_coverage=.06 if n>=40 and ratio>=.7 else .10)
        if refined is None:return None
        points,edge=refined
        return points,dict(status='tracked',method='bezel_flow',inliers=n,confidence=ratio,recovered=True,**edge)

    def make_reference(self, frame, points, tablets=()):
        p=np.float32(points);w,h=self.size
        verified=self.refine(frame,p,tablets)
        if verified is None:return None
        # Keep the user geometry, rather than silently rewriting a manual label.
        lower=np.interp(np.arange(w),p[:,0],p[:,1]);dy=np.arange(h)[:,None]-lower
        mask=np.uint8((dy>=-5)&(dy<=65))*255
        for tablet in tablets:cv2.fillPoly(mask,[np.int32(tablet)],0)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
        keys,desc=self.detector.detectAndCompute(gray,mask)
        if desc is None or len(keys)<20:return None
        return dict(points=p.copy(),xy=np.float32([k.pt for k in keys]),desc=desc)

    def add(self, reference):
        if reference is not None:self.references.append(reference)
        self.references=self.references[-8:]

    def locate(self, frame, tablets=()):
        self.steps+=1
        if not self.references or (self.pending is None and self.steps%4!=1):return None
        lines=self.edges(frame)
        if not lines:self.pending=None;return None
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
        keys,desc=self.detector.detectAndCompute(gray,None)
        if desc is None:self.pending=None;return None
        xy=np.float32([k.pt for k in keys]);candidates=[];w,h=self.size
        for ref in self.references:
            def ratios(a,b):return {m.queryIdx:m.trainIdx for pair in cv2.BFMatcher().knnMatch(a,b,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
            forward=ratios(ref['desc'],desc)
            # Reverse-check only proposed matches. This is exactly the same
            # mutual test as querying all current descriptors, with less work.
            targets=sorted(set(forward.values()))
            if len(targets)<20:continue
            local_reverse=ratios(desc[targets],ref['desc'])
            reverse={targets[i]:j for i,j in local_reverse.items()}
            pairs=[(a,b) for a,b in forward.items() if reverse.get(b)==a]
            if len(pairs)<20:continue
            old=np.float32([ref['xy'][a] for a,b in pairs]);new=np.float32([xy[b] for a,b in pairs])
            H,mask=cv2.findHomography(old,new,cv2.RANSAC,2.5,maxIters=1500,confidence=.995)
            if H is None or mask is None or not np.isfinite(H).all():continue
            keep=mask.ravel().astype(bool);n=int(keep.sum());ratio=float(keep.mean())
            if n<18 or ratio<.65:continue
            span=np.ptp(old[keep],axis=0)
            if span[0]<w*.25 or span[1]<h*.035:continue
            corners=np.float32([[[0,0],[w,0],[w,h],[0,h]]]);denom=np.c_[corners[0],np.ones(4)]@H[2]
            moved=cv2.perspectiveTransform(corners,H)[0];area=cv2.contourArea(moved,oriented=True)/(w*h)
            if np.any(denom<=0) or not cv2.isContourConvex(moved) or not .5<area<2:continue
            try:points=self.lock_edges(cv2.perspectiveTransform(ref['points'][None],H)[0].tolist(),self.size)
            except ValueError:continue
            refined=self.refine(frame,points,tablets,lines,min_coverage=.06 if n>=40 and ratio>=.70 else .10)
            if refined is None:continue
            points,edge=refined
            candidates.append((n*ratio,points,dict(status='reacquired',method='manual_bezel_edges',inliers=n,confidence=ratio,recovered=True,**edge)))
        if not candidates:self.pending=None;return None
        candidates.sort(key=lambda x:x[0],reverse=True);_,points,stats=candidates[0]
        xs=np.linspace(0,w-1,31);ys=np.interp(xs,points[:,0],points[:,1])
        # A second strong but incompatible explanation is an ambiguity, not a
        # reason to pick whichever reference happened to produce more points.
        for score,other,_ in candidates[1:]:
            if score>=candidates[0][0]*.8 and np.max(np.abs(np.interp(xs,other[:,0],other[:,1])-ys))>35:
                self.pending=None;return None
        previous=self.pending;self.pending=ys
        if previous is None or np.max(np.abs(previous-ys))>24:return None
        self.pending=None
        return points,stats
