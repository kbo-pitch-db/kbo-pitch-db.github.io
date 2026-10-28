"""Toe/heel support episodes with image-position coherence and air-pause rejection.

All decisions use unchanged 2D observations/background motion, never the
desired 3D skating score. A moving toe can release while a stationary heel
continues supporting the same foot. No landing event is required.
"""
from common import *
from events import smooth_xy

VERSION='toe-heel-position-coherence-v1'


def episodes(h,phase,bg):
    xy,sc=smooth_xy(h,bg);feet=xy[:,20:26];confidence=sc[:,20:26];n=len(feet)
    t=h['source_time'];v=np.linalg.norm(np.gradient(feet,t,axis=0),axis=-1)
    good=confidence>.4;ground=float(np.quantile(feet[...,1][good],.8))
    minimum=max(4,int(np.ceil(float(h['fps'])*.12)));weights=[];notes=[];seeds=[]
    contiguous=np.r_[True,(np.diff(t)>0)&(np.diff(t)<.1)&(np.diff(h['frame_index'])==1)]
    for marker in range(6):
        side=marker%2
        high=xy[:,13+side,1]-xy[:,11+side,1]<.08
        above=feet[:,marker,1]<feet[:,1-side,1]-.04
        stable=good[:,marker]&(v[:,marker]<.35)&(feet[:,marker,1]>ground-.10)&~(high&above)
        starts=np.flatnonzero(stable&(~np.r_[False,stable[:-1]]|~contiguous))
        ends=np.flatnonzero(stable&(~np.r_[stable[1:],False]|~np.r_[contiguous[1:],False]))
        for first,last in zip(starts,ends):
            # A persistent image displacement is a marker release, even if
            # its speed is low (rolling, dragging, walking, or camera-model uncertainty).
            start=int(first)
            while start<=last:
                end=start
                while end<last:
                    points=feet[start:end+2,marker];center=np.median(points,axis=0)
                    if np.linalg.norm(points-center,axis=-1).max()>.02:break
                    end+=1
                if end-start+1>=minimum:
                    seed=(start+end)//2;w=np.zeros(n)
                    w[start:end+1]=.6*confidence[start:end+1,marker]**2/(1+(v[start:end+1,marker]/.35)**2)
                    weights.append(w);seeds.append(seed);notes.append(dict(marker=marker,start=start,end=int(end),seed_frame=seed,
                        detector=VERSION,position_radius_body_height=.02,
                        semantics='Image-supported static toe/heel episode; rolling marker releases independently; not ground truth'))
                start=end+1
    return np.stack(weights,axis=1) if weights else np.zeros((n,0)),np.array(seeds,int),notes
