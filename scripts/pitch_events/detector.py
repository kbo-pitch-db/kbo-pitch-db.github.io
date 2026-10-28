"""Image-supported pitching events. No 3D contact pins or timing templates."""
import itertools
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

VERSION='image-pitch-events-20261029-v3'

def transform(points,matrices):
    return np.einsum('tij,tkj->tki',matrices,np.concatenate([points,np.ones((*points.shape[:-1],1))],axis=-1))[...,:2]

def smooth_observed(values,mask):
    out=np.full_like(values,np.nan,dtype=float)
    ids=np.flatnonzero(mask&np.isfinite(values))
    if len(ids):out=np.interp(np.arange(len(values)),ids,values[ids]);out=gaussian_filter1d(out,.65)
    return out

def associate_feet(xy,score,unique,times,height):
    """Maintain shoe identity through pose-estimator left/right label switches.

    Dynamic programming penalizes acceleration of the observed shoe positions,
    not distance to a 3D contact anchor. Low-confidence gaps retain their masks.
    """
    u=np.asarray(unique);pairs=[(13,14),(15,16),(20,21),(22,23),(24,25)];centers=np.stack([np.nanmean(xy[u][:,[20,22,24]],axis=1),np.nanmean(xy[u][:,[21,23,25]],axis=1)],axis=1)/height
    valid=np.isfinite(centers).all((1,2))
    for foot in range(2):
        for d in range(2):centers[:,foot,d]=smooth_observed(centers[:,foot,d],valid)
    if len(u)<3 or not np.isfinite(centers).all():return xy,score,[]
    costs={(0,0):0.};paths={(0,0):[0,0]}
    for at in range(2,len(u)):
        next_cost={};next_path={};dt=max(.01,times[u[at]]-times[u[at-1]]);dt0=max(.01,times[u[at-1]]-times[u[at-2]])
        for (a,b),cost in costs.items():
            for c in ([0] if at<4 else [0,1]):
                p0=centers[at-2,::(-1 if a else 1)];p1=centers[at-1,::(-1 if b else 1)];p2=centers[at,::(-1 if c else 1)];v0=(p1-p0)/dt0;v1=(p2-p1)/dt
                acceleration=np.linalg.norm(v1-v0,axis=1);step=np.linalg.norm(p2-p1,axis=1)
                candidate=cost+float(np.minimum(acceleration,8).sum()+.5*step.sum())+(.03 if b!=c else 0)
                state=(b,c)
                if candidate<next_cost.get(state,float('inf')):next_cost[state]=candidate;next_path[state]=paths[(a,b)]+[c]
        costs,paths=next_cost,next_path
    path=paths[min(costs,key=costs.get)];out=xy.copy();confidence=score.copy();swapped=[]
    for at,state in enumerate(path):
        if state:
            i=u[at];swapped.append(int(i))
            for left,right in pairs:out[i,[left,right]]=out[i,[right,left]];confidence[i,[left,right]]=confidence[i,[right,left]]
    return out,confidence,swapped

def release_tracks(detections,xy,score,unique,times,height,hand,matrices=None,video_origin=None):
    """Track learned ball detections and back-bracket their departure from the hand.

    The wrist only associates a visible ball trajectory with this pitcher. Its
    speed maximum is never used as a release timestamp.
    """
    arm=10 if hand=='R' else 9;u=np.asarray(unique);ts=times[u];wrist=xy[u,arm].copy()
    if matrices is not None:wrist=transform(wrist[:,None],matrices[u])[:,0]
    nodes={};origin=np.zeros(2) if video_origin is None else video_origin
    for at,i in enumerate(u):
        bs=[]
        for b in detections.get(str(i),[]):
            center=(np.asarray(b[:2])+b[2:4])/2+origin
            if b[4]<.12:continue
            if matrices is not None:center=(matrices[i]@np.r_[center,1])[:2]
            bs.append((center,float(b[4]),np.asarray(b[:4],float)))
        nodes[at]=bs
    tracks=[]
    # Extend only short smooth trajectories; no image interpolation or free
    # object motion inferred from the body skeleton.
    for start,bs in nodes.items():
        for point,conf,box in bs:
            beam=[[(start,point,conf,box)]]
            for step in range(8):
                grown=[]
                for chain in beam:
                    last,p,_,_=chain[-1];candidates=[]
                    for nxt in range(last+1,min(last+4,len(u))):
                        dt=ts[nxt]-ts[last]
                        if dt>.12 or ts[nxt]-ts[start]>.40:continue
                        for q,c,b in nodes[nxt]:
                            velocity=(q-p)/dt/height;speed=np.linalg.norm(velocity)
                            if not .15<speed<18:continue
                            if len(chain)>1:
                                a,p0,_,_=chain[-2];v0=(p-p0)/(ts[last]-ts[a])/height
                                if np.dot(velocity,v0)<0 or np.linalg.norm(velocity-v0)>max(3.,np.linalg.norm(v0)*1.2):continue
                            candidates.append(chain+[(nxt,q,c,b)])
                    if candidates:grown.extend(candidates)
                    elif len(chain)>=4:tracks.append(chain)
                if not grown:break
                beam=sorted(grown,key=lambda ch:len(ch)+np.mean([x[2] for x in ch]),reverse=True)[:5]
            tracks.extend(ch for ch in beam if len(ch)>=4)
    releases=[]
    for track in tracks:
        ix=np.array([x[0] for x in track]);tt=ts[ix];pts=np.array([x[1] for x in track]);conf=np.array([x[2] for x in track]);duration=tt[-1]-tt[0]
        if duration<.075 or conf.mean()<.25:continue
        z=tt-tt[0];fit=np.polynomial.polynomial.polyfit(z,pts,2);pred=np.polynomial.polynomial.polyval(z,fit).T
        residual=float(np.sqrt(np.mean(np.sum((pred-pts)**2,axis=1)))/height)
        if residual>.035:continue
        # A flying ball must separate from the throwing hand, rather than track
        # a held ball throughout preparation or a white patch on the uniform.
        distance=np.linalg.norm(pts-wrist[ix],axis=1)/height
        if distance[-1]<.35 or distance[-1]-distance[0]<.22:continue
        eligible=np.flatnonzero((ts>=tt[0]-.16)&(ts<=tt[0]+.04)&(score[u,arm]>=.4))
        if not len(eligible):continue
        projection=np.polynomial.polynomial.polyval(ts[eligible]-tt[0],fit).T
        d=np.linalg.norm(projection-wrist[eligible],axis=1)/height;near=int(eligible[np.argmin(d)]);minimum=float(min(d))
        if minimum>.20 or tt[0]-ts[near]>.105:continue
        after=[j for j,dist in zip(ix,distance) if j>near and dist>.14]
        if not after:continue
        first=after[0]
        if ts[first]-ts[near]>.14:continue
        lower=max(0,near-1);upper=max(near+1,min(first,near+2));upper=min(len(u)-1,upper)
        bracket=[int(u[lower]),int(u[upper])]
        value=float(len(track)*conf.mean()-residual*35-minimum*3-(ts[first]-ts[near])*2)
        releases.append(dict(interval=bracket,nearestFrame=int(u[near]),score=value,associationDistance=minimum,trajectoryResidual=residual,track=[dict(frame=int(u[j]),point=p.tolist(),confidence=c,box=b.tolist()) for j,p,c,b in track]))
    releases.sort(key=lambda r:r['score'],reverse=True);distinct=[]
    for r in releases:
        if any(abs(times[r['nearestFrame']]-times[v['nearestFrame']])<.12 for v in distinct):continue
        distinct.append(r)
    if not distinct:return None,[]
    if len(distinct)>1 and distinct[1]['score']>distinct[0]['score']*.85:return None,distinct[:3]
    return distinct[0],distinct[:3]

def knee_peak(xy,score,unique,times,height,hand,release):
    lead=13 if hand=='R' else 14;u=np.asarray(unique);end=release[0] if release else u[-1]
    lift=(xy[u,11:13,1].mean(1)-xy[u,lead,1])/height
    good=(score[u,lead]>=.4)&(score[u,11:13].min(1)>=.4)
    signal=smooth_observed(lift,good);eligible=np.flatnonzero((u<end)&good)
    if len(eligible)<7:return None
    at=int(eligible[np.nanargmax(signal[eligible])]);left=eligible[eligible<at];right=eligible[(eligible>at)&(times[u[eligible]]<=times[u[at]]+.45)]
    if len(left)<2 or len(right)<2:return None
    prominence=signal[at]-max(np.nanmin(signal[left]),np.nanmin(signal[right]))
    if prominence<.035:return None
    close=eligible[(signal[eligible]>=signal[at]-.015)&(abs(times[u[eligible]]-times[u[at]])<=.10)]
    return [int(u[max(0,min(close)-1)]),int(u[min(len(u)-1,max(close)+1)])]

def landing(xy,score,unique,times,height,hand,release,pkh=None,matrices=None):
    """Robust descending-to-ground change point, using individual foot markers.

    A two-stage trajectory fit estimates onset of contact, rather than selecting
    the first frame at which the whole foot finally stops moving.
    """
    if release is None:return None,{}
    lead=0 if hand=='R' else 1;foot_ids=[20+lead,22+lead,24+lead];u=np.asarray(unique);t=times[u];field=transform(xy,matrices) if matrices is not None else xy
    good=(score[u][:,foot_ids]>=.4).all(1)&np.isfinite(field[u][:,foot_ids]).all((1,2))
    start=times[pkh[0]] if pkh else max(times[u[0]],times[release[0]]-1.)
    ix=np.flatnonzero((t>=start)&(t<times[release[0]])&good)
    if len(ix)<8:return None,{'reason':'foot_observations_missing'}
    # The lowest supported toe/heel is the part that can make first contact.
    y=np.max(field[u][:,foot_ids,1],axis=1)/height;y=smooth_observed(y,good)
    end=ix[-1];tail=ix[max(0,len(ix)-5):];ground=float(np.median(y[tail]));candidates=[]
    for at in ix[2:-3]:
        left=ix[ix<=at];right=ix[ix>=at]
        left=left[t[left]>=t[at]-.30];right=right[t[right]<=t[at]+.20]
        if len(left)<3 or len(right)<4:continue
        tt=np.r_[left,right[1:]];z=t[tt]-t[at];design=np.column_stack([np.ones(len(z)),np.minimum(z,0),np.maximum(z,0)])
        fit=np.linalg.lstsq(design,y[tt],rcond=None)[0];error=np.sqrt(np.mean((design@fit-y[tt])**2))
        before,after=fit[1],fit[2]
        if before<.12 or abs(after)>.50*before or y[at]<ground-.065:continue
        # Favor a sharp change after substantial descent, not a low pause during
        # setup. Do not impose a normative landing-to-release duration.
        cost=error+.035*abs(after)/max(.1,before)+.2*abs(ground-y[at]);candidates.append((float(cost),int(at),float(before),float(after)))
    if not candidates:return None,{'reason':'ground_change_not_identified'}
    candidates.sort();best=candidates[0];near=[x for x in candidates if x[0]<=best[0]+.006 and abs(t[x[1]]-t[best[1]])<.10];low=min(x[1] for x in near);high=max(x[1] for x in near)
    bracket=[int(u[max(0,low-1)]),int(u[min(len(u)-1,high+1)])]
    if times[bracket[1]]>=times[release[0]]:return None,{'reason':'contact_release_intervals_overlap'}
    return bracket,dict(cost=best[0],beforeSlope=best[2],afterSlope=best[3],ground=ground,candidates=[list(x) for x in candidates[:8]])
