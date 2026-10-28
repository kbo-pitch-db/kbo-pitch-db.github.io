"""A small supervised image-contact head on frozen DINOv2 foot features."""
import numpy as np

def features(encoded):
    value=np.asarray(encoded,float);a,b,scene=value[:,0],value[:,1],value[:,2]
    # Symmetric in the shoe labels: raw pose left/right swaps cannot change it.
    return np.concatenate([(a+b)/2,abs(a-b),scene],axis=1)*10

def probabilities(encoded,model):
    x=features(encoded);z=x@np.asarray(model['coef'])+model['intercept'];return 1/(1+np.exp(-np.clip(z,-35,35)))

def decode_contact(frames,probability,times,release,pkh=None):
    if not release:return None,{'reason':'visible_ball_release_missing'}
    u=np.asarray(frames);t=times[u];p=np.clip(probability,.001,.999);start=max(0,times[release[0]]-1.2)
    ix=np.flatnonzero((t>=start)&(u<release[0]));candidates=[]
    for at in ix[2:-2]:
        left=ix[(t[ix]>=t[at]-.25)&(ix<at)];right=ix[(ix>=at)&(t[ix]<=t[at]+.17)]
        if len(left)<3 or len(right)<3:continue
        pre=float(np.median(p[left]));post=float(np.median(p[right]))
        if pre>.48 or post<.52 or post-pre<.18:continue
        loss=float(-np.log(1-p[left]).mean()-np.log(p[right]).mean());candidates.append((loss,int(at),pre,post))
    if not candidates:return None,{'reason':'visual_contact_transition_unresolved'}
    candidates.sort();best=candidates[0];near=[v[1] for v in candidates if v[0]<best[0]+.10 and abs(t[v[1]]-t[best[1]])<.09];low=min(near);high=max(near)
    bracket=[int(u[max(0,low-1)]),int(u[min(len(u)-1,high+1)])]
    if bracket[1]>=release[0]:return None,{'reason':'event_intervals_overlap'}
    return bracket,dict(loss=best[0],beforeProbability=best[2],afterProbability=best[3],method='DINOv2 frozen features and source-grouped contact classifier',probability=[[int(i),round(float(v),5)] for i,v in zip(u,p)])
