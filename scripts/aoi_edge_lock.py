"""Cheap visual evidence for a screen bezel and a tablet's bright display panel."""
from __future__ import annotations
import cv2
import numpy as np


def strip(gray,y,slope,x=None):
    h,w=gray.shape;x=w/2 if x is None else x
    xs=np.linspace(x-46,x+46,64,dtype=np.float32)
    yy=y+slope*(xs-w/2)
    if yy.min()<8 or yy.max()>h-18 or xs.min()<0 or xs.max()>=w:return None
    xx=np.broadcast_to(xs,(24,64)).copy()
    ys=(yy[None,:]+np.linspace(-7,17,24,dtype=np.float32)[:,None]).astype(np.float32)
    p=cv2.remap(gray,xx,ys,cv2.INTER_LINEAR).astype(np.float32)
    p-=p.mean();norm=float(np.linalg.norm(p))
    return p/norm if norm>100 else None


def similarity(patch,bank):
    return max((float(np.sum(patch*p)) for p in bank),default=0.) if patch is not None else 0.


def tablet_panels(gray):
    """White half of this experiment's tablet UI; never a free-floating line."""
    h,w=gray.shape
    mask=np.uint8(gray>170)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not max(250,w*h*.002)<area<w*h*.13:continue
        hull=cv2.convexHull(contour)
        q=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True).reshape(-1,2).astype(np.float32)
        if len(q)!=4 or not cv2.isContourConvex(q):continue
        # A display has substantial height, straight sides, and a dark surround.
        rect=cv2.minAreaRect(q);short,long=sorted(rect[1])
        if short<12 or long<30 or not .22<short/long<.9 or area/(short*long)<.8:continue
        edges=[(a,b) for a,b in zip(q,np.roll(q,-1,axis=0)) if abs(b[0]-a[0])>12 and abs((b[1]-a[1])/(b[0]-a[0]))<.8]
        if not edges:continue
        a,b=min(edges,key=lambda ab:(ab[0][1]+ab[1][1])/2)
        if a[0]>b[0]:a,b=b,a
        midpoint=(a+b)/2;center=q.mean(axis=0)
        if center[1]-midpoint[1]<15:continue
        if not w*.25<center[0]<w*.9 or midpoint[1]<h*.28:continue
        slope=float((b[1]-a[1])/(b[0]-a[0]));y=float(midpoint[1]+(w/2-midpoint[0])*slope)
        xs=np.linspace(a[0]+2,b[0]-2,12);ys=a[1]+(xs-a[0])*slope
        if ys.min()<7 or ys.max()>h-8:continue
        xx=np.clip(np.rint(xs).astype(int),0,w-1);yy=np.rint(ys).astype(int)
        outside=gray[yy-4,xx].astype(float);inside=gray[yy+4,xx].astype(float)
        contrast=float(np.mean((outside<115)&(inside-outside>45)))
        if contrast<.65:continue
        result.append(dict(y=y,slope=slope,x=float(center[0]),width=float(b[0]-a[0]),height=float(long),quality=contrast))
    return result
