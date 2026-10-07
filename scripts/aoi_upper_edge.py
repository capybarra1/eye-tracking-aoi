"""Exposure assistance and calibrated upper display/frame references."""
from __future__ import annotations
import cv2
import numpy as np
from scripts.aoi_edge_lock import strip,similarity


def enhance_dark(gray):
    """Bounded contrast lift on a small working image, never the source video."""
    sample=gray[::4,::4]
    if float(np.median(sample))>=75 or np.ptp(sample)<3:return gray,False
    local=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(gray)
    # A gain cap avoids converting black sensor noise into bright 'edges'.
    lifted=np.maximum(gray,np.minimum(local.astype(np.float32)*1.25,gray.astype(np.float32)*2.8))
    return np.uint8(np.clip(lifted,0,255)),True


def lines(gray,detector):
    detected=detector.detect(gray)[0]
    return [] if detected is None else detected.reshape(-1,4).tolist()


def horizontal_edges(edges,width):
    result=[]
    for a,b,c,d in edges:
        if a>c:a,b,c,d=c,d,a,b
        if c-a<width*.12 or abs((d-b)/(c-a))>1.:continue
        slope=(d-b)/(c-a);y=b+(width/2-a)*slope
        result.append(dict(a=a,b=b,c=c,d=d,y=y,slope=slope,width=c-a,x=(a+c)/2))
    return result


def top_support(gray,top):
    h,w=gray.shape;xs=np.linspace(top['a']+2,top['c']-2,32);ys=top['y']+top['slope']*(xs-w/2)
    xx=np.clip(np.rint(xs).astype(int),0,w-1);yy=np.rint(ys).astype(int)
    if yy.min()<5 or yy.max()>=h-16:return False
    above=gray[yy-3,xx].astype(float);near=gray[yy+4,xx].astype(float);far=gray[yy+15,xx].astype(float)
    # A luminous screen continues below its bezel; a narrow bright metal bar does not.
    return bool(np.mean(above)<120 and np.mean(near-above>10)>.55 and np.mean(far-above>10)>.55
                and np.mean(far-above)>.5*np.mean(near-above))


def beam_support(gray,top):
    """A narrow bright bar may be a reference, but never the AOI boundary itself."""
    h,w=gray.shape;xs=np.linspace(top['a']+2,top['c']-2,32)
    xx=np.clip(np.rint(xs).astype(int),0,w-1)
    yy=np.rint(top['y']+top['slope']*(xs-w/2)).astype(int)
    if yy.min()<5 or yy.max()>=h-16:return False
    above=gray[yy-3,xx].astype(float);near=gray[yy+4,xx].astype(float);far=gray[yy+15,xx].astype(float)
    return bool(np.mean((near-above>18)&(near-far>18))>.7)


class UpperEdge:
    """Calibrate an upper edge/bar against a known lower boundary and frame sides."""
    def __init__(self):
        self.reference=None;self.detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

    def calibrate(self,gray,lower_y,lower_slope):
        h,w=gray.shape;edges=lines(gray,self.detector);options=[]
        gradient=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))
        for top in horizontal_edges(edges,w):
            is_screen=top_support(gray,top)
            if not is_screen and not beam_support(gray,top):continue
            height=lower_y-top['y']
            if not h*.13<height<h*.85 or abs(top['slope']-lower_slope)>.16:continue
            if not top['a']-12<w/2<top['c']+12:continue
            sides=[[],[]]
            for x1,y1,x2,y2 in edges:
                if y1>y2:x1,y1,x2,y2=x2,y2,x1,y1
                if y2-y1<max(18,height*.18) or abs(x2-x1)>(y2-y1)*.45:continue
                dx=(x2-x1)/(y2-y1)
                # Intersection with the top line, rather than its visible fragment's ends.
                den=1-dx*top['slope']
                if abs(den)<.3:continue
                x=(x1+dx*(top['y']-top['slope']*w/2-y1))/den
                if top['a']-height*.7<x<top['a']+10:side=0
                elif top['c']-10<x<top['c']+height*.7:side=1
                else:continue
                top_y=top['y']+top['slope']*(x-w/2)
                ys=np.linspace(top_y+10,lower_y-7,32);xs=x1+(ys-y1)*dx
                if ys.min()<1 or ys.max()>=h-1 or xs.min()<5 or xs.max()>=w-5:continue
                ix=np.rint(xs).astype(int);iy=np.rint(ys).astype(int)
                strength=np.max(np.stack([gradient[iy,ix+offset] for offset in (-3,0,3)]),axis=0)
                support=float(np.mean(strength>24))
                if support<.55:continue
                sides[side].append((x,support))
            for left,ls in sides[0]:
                for right,rs in sides[1]:
                    width=right-left;aspect=width/height
                    if not 1.2<aspect<2.45 or top['width']<width*.43:continue
                    center=(left+right)/2
                    if not w*.30<center<w*.70:continue
                    patch=strip(gray,top['y'],top['slope'],center)
                    if patch is None:continue
                    score=min(ls,rs)*top['width']/(1+abs(aspect-1.78)*4+abs(top['slope']-lower_slope)*8)
                    options.append((is_screen,score,{**top,'kind':'screen_top' if is_screen else 'frame_beam','x':center,'width':width,'visible_width':top['width'],'visible_x':top['x']},patch))
        if not options:return False
        _,_,top,patch=max(options,key=lambda x:(x[0],x[1]))
        self.reference=dict(kind=top['kind'],y=top['y'],slope=top['slope'],x=top['x'],width=top['width'],
            visible_width=top['visible_width'],visible_x=top['visible_x'],height=lower_y-top['y'],
            slope_offset=lower_slope-top['slope'],bank=[patch])
        return True

    def candidate(self,gray,lower_y,lower_slope,radius,recovery=0):
        ref=self.reference
        if ref is None:return None
        h,w=gray.shape;expected=lower_y-ref['height'];options=[]
        # Only a narrow strip around the expected top needs a second line search.
        margin=min(h,max(14,radius));tilt=abs(lower_slope)*ref['width']/2
        y0=max(0,int(expected-margin-tilt-8));y1=min(h,int(expected+margin+tilt+8))
        x0=max(0,int(ref['x']-ref['width']*.65));x1=min(w,int(ref['x']+ref['width']*.65))
        if y1-y0<12 or x1-x0<30:return None
        edges=[[a+x0,b+y0,c+x0,d+y0] for a,b,c,d in lines(gray[y0:y1,x0:x1],self.detector)]
        for top in horizontal_edges(edges,w):
            support=beam_support if ref.get('kind')=='frame_beam' else top_support
            if not support(gray,top):continue
            scale=top['width']/ref['width']
            if not .43<scale<1.25 or abs(top['x']-ref['x'])>w*.2:continue
            if abs(top['slope']+ref['slope_offset']-lower_slope)>(.35 if recovery else .18):continue
            if abs(top['y']-expected)>margin:continue
            match=similarity(strip(gray,top['y'],top['slope'],ref['x']),ref['bank'])
            if match<.60:continue
            # Width includes foreshortening from roll. Convert to scale first.
            scale=(top['width']/ref['visible_width'] if ref['visible_width']/ref['width']>.9 and scale>.8 else 1.)*np.sqrt(1+top['slope']**2)/np.sqrt(1+ref['slope']**2)
            height=ref['height']*scale*np.sqrt(1+top['slope']**2)/np.sqrt(1+ref['slope']**2)
            y=top['y']+height;slope=top['slope']+ref['slope_offset']
            if abs(y-lower_y)>radius+10:continue
            options.append(dict(y=y,slope=slope,source='screen_top',x=top['x'],top=top,
                quality=match,appearance=match,score=match/(1+abs(y-lower_y)*.1)))
        options.sort(key=lambda x:x['score'],reverse=True)
        if not options:return None
        if len(options)>1 and options[1]['score']>options[0]['score']*.85 and abs(options[1]['y']-options[0]['y'])>8:return None
        return options[0]
