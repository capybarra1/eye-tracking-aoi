"""Screen tracking using physical structures outside dynamic AOIs."""

import cv2
import numpy as np
from scripts.aoi_feature_cache import screen_sift


class StructureReferenceCache:
    """One retained SIFT reference; owned by a single video tracking state.

    Tracking gray arrays are replaced rather than modified in place. Retaining
    the actual object prevents identity reuse; the mask copy detects callers
    reusing and modifying their mask buffer between frames.
    """
    def __init__(self):
        self.previous=None
        self.mask=None
        self.features=None

    def get_or_compute(self, previous, mask, detector):
        same_mask=(mask is None and self.mask is None) or (
            mask is not None and self.mask is not None and np.array_equal(mask,self.mask))
        if self.previous is not previous or not same_mask or self.features is None:
            self.features=detector.detectAndCompute(previous,mask)
            self.previous=previous
            self.mask=None if mask is None else mask.copy()
        return self.features


def screen_coverage(polygons: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    """Extend the seeded physical lower edges to cover newly exposed side screens.

    This implements the user's coarse AOI rule: everything above the three
    lower borders. It deliberately includes ceiling pixels above the screens.
    """
    width, _ = size
    edges = []
    for polygon in polygons:
        bottom = polygon[np.argsort(polygon[:, 1])[-2:]]
        edges.append(bottom[np.argsort(bottom[:, 0])])
    edges.sort(key=lambda e: float(e[:,0].mean()))
    joins = [(edges[i][1] + edges[i+1][0])/2 for i in range(len(edges)-1)]
    def at_x(edge: np.ndarray, x: float) -> np.ndarray:
        dx = float(edge[1,0]-edge[0,0])
        if abs(dx)<1:
            raise ValueError('Screen lower edge is vertical; geometric coverage is unreliable')
        return np.array([x,edge[0,1]+(x-edge[0,0])*(edge[1,1]-edge[0,1])/dx],np.float32)
    boundaries = [at_x(edges[0],0),*joins,at_x(edges[-1],width-1)]
    if any(b[0] <= a[0] for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Screen lower edges have crossed or left the field of view')
    return [np.array([[a[0],0],[b[0],0],b,a],np.float32) for a,b in zip(boundaries,boundaries[1:])]


def structure_mask(shape: tuple[int, int], screens: list[np.ndarray], tablets: list[np.ndarray]) -> np.ndarray:
    height, width = shape
    excluded = np.zeros(shape, np.uint8)
    for polygon in screens + tablets:
        cv2.fillPoly(excluded, [np.rint(polygon).astype(np.int32)], 255)
    excluded = cv2.dilate(excluded, np.ones((13, 13), np.uint8))
    mask = cv2.bitwise_not(excluded)
    # The steering wheel and hands are not fixed reference structures.
    mask[int(height * .68):, :int(width * .56)] = 0
    mask[int(height * .92):] = 0
    mask[:4] = 0
    mask[:, :4] = 0
    mask[:, -4:] = 0
    return mask


def _fit_structure_motion(old, new, shape, method):
    if len(old)<16:return None
    h,w=shape
    corners=np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    H,mask=cv2.findHomography(old,new,cv2.RANSAC,2.,maxIters=1500,confidence=.995)
    candidates=[(H,mask,'homography')]
    # A constrained rotation/scale fit resists parallax when the general
    # projective fit is unstable on a small group of background structures.
    affine,am=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1500,confidence=.995)
    if affine is not None:candidates.append((np.vstack([affine,[0,0,1]]),am,'similarity'))
    for transform,inlier_mask,model in candidates:
        if transform is None or inlier_mask is None or not np.isfinite(transform).all():continue
        inliers=inlier_mask.ravel().astype(bool);count=int(inliers.sum());ratio=float(inliers.mean())
        support=cv2.contourArea(cv2.convexHull(old[inliers]))/(w*h) if count>=3 else 0
        moved=cv2.perspectiveTransform(corners[None],transform)[0]
        if not np.isfinite(moved).all() or not cv2.isContourConvex(moved):continue
        area=cv2.contourArea(moved,oriented=True)/(w*h)
        # Large displacements are accepted only with many spatially distributed
        # agreeing matches; frame rotation alone is not evidence of an error.
        strong=count>=30 and ratio>=.55 and support>=.03
        limit=max(w,h)*(.35 if strong else .1)
        if count<14 or ratio<(.55 if strong else .6) or support<.015:continue
        if not (.65 if strong else .8)<area<(1.5 if strong else 1.25):continue
        if np.max(np.linalg.norm(moved-corners,axis=1))>limit:continue
        return transform.astype(np.float32),dict(status='tracked',confidence=ratio,inliers=count,
                                                  tracked_points=len(old),method=method,model=model,support=support)
    return None


def estimate_structure_motion(previous: np.ndarray, current: np.ndarray, mask: np.ndarray,
                              cache: StructureReferenceCache|None=None) -> tuple[np.ndarray, dict]:
    failed=(np.eye(3,dtype=np.float32),dict(status='lost',confidence=0.,inliers=0,tracked_points=0))
    if mask is not None and not np.any(mask):return failed
    points=cv2.goodFeaturesToTrack(previous,650,.01,7,mask=mask,blockSize=7)
    if points is not None and len(points)>=16:
        lk=dict(winSize=(31,31),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        forward,good,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,**lk)
        if forward is not None:
            backward,back_good,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,**lk)
            if backward is not None:
                valid=(good.ravel()==1)&(back_good.ravel()==1)
                valid &= np.linalg.norm((backward-points).reshape(-1,2),axis=1)<1.2
                old,new=points[valid].reshape(-1,2),forward[valid].reshape(-1,2)
                result=_fit_structure_motion(old,new,previous.shape,'flow')
                if result is not None:return result
    # Match retained physical structures over the whole new image when local
    # optical flow cannot follow a fast turn. Never use moving screen interiors
    # as reference features.
    detector=screen_sift()
    keys1,desc1=(cache.get_or_compute(previous,mask,detector) if cache is not None
                 else detector.detectAndCompute(previous,mask))
    if desc1 is None:return failed
    keys2,desc2=detector.detectAndCompute(current,None)
    if desc2 is None or len(keys2)<16:return failed
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc1,desc2,k=2)
    matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    unique={}
    for match in sorted(matches,key=lambda m:m.distance):unique.setdefault(match.trainIdx,match)
    matches=list(unique.values())
    if len(matches)<20:return failed
    old=np.array([keys1[m.queryIdx].pt for m in matches],np.float32)
    new=np.array([keys2[m.trainIdx].pt for m in matches],np.float32)
    return _fit_structure_motion(old,new,previous.shape,'appearance') or failed
