"""Vector clipping for mutually exclusive on-image AOI regions."""
import numpy as np
import cv2


def halfplane(polygon, a, b, orientation, inside=True):
    p=np.asarray(polygon,dtype=float)
    if len(p)<3:return []
    a,b=np.asarray(a,float),np.asarray(b,float);edge=b-a
    def distance(v):return orientation*(edge[0]*(v[1]-a[1])-edge[1]*(v[0]-a[0]))*(1 if inside else -1)
    result=[];previous=p[-1];dp=distance(previous)
    for current in p:
        dc=distance(current)
        if (dc>=-1e-8)!=(dp>=-1e-8):
            result.append((previous+(current-previous)*dp/(dp-dc)).tolist())
        if dc>=-1e-8:result.append(current.tolist())
        previous,dp=current,dc
    return result if len(result)>=3 and abs(cv2.contourArea(np.float32(result)))>1e-5 else []


def convex_clip(polygon, boundary):
    p=np.float32(polygon);b=np.float32(boundary)
    if len(p)<3:return []
    area,q=cv2.intersectConvexConvex(p,b)
    if area<=1e-5 or q is None:return []
    if abs(area-abs(cv2.contourArea(p)))<1e-4:return p.tolist()
    return q.reshape(-1,2).tolist()


def convex_difference(polygon, cut):
    """Partition a convex polygon outside a convex cut, with no area overlap."""
    if cv2.intersectConvexConvex(np.float32(polygon),np.float32(cut))[0]<=1e-5:return [polygon]
    remaining=polygon;pieces=[]
    sign=1 if cv2.contourArea(np.float32(cut),oriented=True)>=0 else -1
    for a,b in zip(cut,np.roll(cut,-1,axis=0)):
        part=halfplane(remaining,a,b,sign,inside=False)
        if part:pieces.append(part)
        remaining=halfplane(remaining,a,b,sign)
        if not remaining:break
    return pieces
