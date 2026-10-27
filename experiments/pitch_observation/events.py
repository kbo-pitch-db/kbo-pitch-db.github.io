"""Forward-only phase hypotheses. Fast arm is not ball release; unknown is kept."""
from common import *
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def smooth_xy(h,background=None):
 xy=h['keypoints'].copy();score=scores(h);n=len(xy)
 if background is not None:xy=np.einsum('tij,tkj->tki',background,np.concatenate([xy,np.ones((*xy.shape[:2],1))],-1))[...,:2]
 height=max(100,float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])))
 for j in range(26):
  good=(score[:,j]>.2)&np.isfinite(xy[:,j]).all(-1)
  if good.sum()<2:xy[:,j]=0;continue
  for k in range(2):xy[:,j,k]=np.interp(np.arange(n),np.flatnonzero(good),xy[good,j,k])
 return gaussian_filter1d(xy/height,max(.6,float(h['fps'])*.018),axis=0),score

def infer(h,background=None):
 xy,score=smooth_xy(h,background);n=len(xy);t=h['source_time']-h['source_time'][0];fps=float(h['fps'])
 rel=xy[:,9:11]-xy[:,5:7];v=np.linalg.norm(np.gradient(rel,t,axis=0),axis=-1)
 good=(score[:,9:11]>.45)&(score[:,5:7]>.45);v[~good]=0;peaks=[]
 for side in range(2):
  for p in find_peaks(v[:,side],distance=max(2,int(.15*fps)))[0]:peaks.append((float(v[p,side]),int(p),side))
 empty=dict(arm_fast_interval=None,lift_interval=None,plant_interval=None,lead_side=None,throwing_hand='unknown',release='unknown')
 if not peaks:return dict(empty,reason='no reliable fast-arm evidence')
 peak=max(peaks);strong=[p for p in peaks if p[0]>.8*peak[0]];amp,burst,side=min(strong,key=lambda p:p[1])
 if amp<.8:return dict(empty,reason='weak fast-arm evidence')
 arm=[max(0,burst-max(2,int(fps*.08))),min(n-1,burst+max(2,int(fps*.08)))]
 # An asymmetric knee lift must occur BEFORE the observed arm burst.
 asym=np.column_stack([xy[:,14,1]-xy[:,13,1],xy[:,13,1]-xy[:,14,1]])
 limit=max(1,burst-max(2,int(fps*.12)));lidx,lside=np.unravel_index(np.argmax(asym[:limit]),asym[:limit].shape)
 clear=asym[lidx,lside]>.10 and lidx>max(2,int(fps*.08)) and lidx<limit-1
 lift=[max(0,int(lidx)-max(1,int(fps*.05))),min(limit-1,int(lidx)+max(1,int(fps*.05)))] if clear else None
 lead=lside if clear else 1-side
 ids=[20+lead,22+lead,24+lead];foot=xy[:,ids].mean(1);speed=np.linalg.norm(np.gradient(foot,t,axis=0),axis=-1);conf=score[:,ids].mean(1)
 start=lift[1]+1 if lift else max(0,burst-int(.65*fps));end=min(n-1,burst+max(2,int(.12*fps)));plant=None;proposals=[];window=max(3,int(.12*fps))
 tail=np.arange(burst,min(n,burst+max(3,int(.18*fps))));tail=tail[conf[tail]>.5]
 plateau_reference=np.median(foot[tail],axis=0) if len(tail)>=2 else None
 for i in range(start,max(start,end-window+1)):
  before=speed[max(start,i-max(2,int(.1*fps))):i+1];after=speed[i:i+window]
  if len(before) and np.median(after)<.25 and np.max(before)>.3 and np.mean(conf[i:i+window])>.5:
   candidate=[max(start,i-max(1,int(fps*.05))),min(end,i+window-1)]
   if plateau_reference is not None and np.linalg.norm(np.median(foot[i:i+window],axis=0)-plateau_reference)<.05:proposals.append(candidate)
 if proposals:plant=[min(x[0] for x in proposals),max(x[1] for x in proposals)]
 ambiguity=bool(plant and plant[1]-plant[0]>.18*fps)
 return dict(arm_fast_interval=arm,arm_fast_peak=burst,arm_fast_side='left' if side==0 else 'right',arm_fast_speed_height_s=amp,
  throwing_hand='left_hypothesis' if side==0 else 'right_hypothesis',release='unknown',release_reason='no ball observations',
  lift_interval=lift,lift_status='observed_2d_hypothesis' if lift else 'outside_clip_or_unknown',lift_peak=int(lidx) if clear else None,
  lead_side='left' if lead==0 else 'right',lead_side_status='preburst_knee_evidence' if clear else 'opposite_fast_arm_hypothesis',
  plant_interval=plant,plant_status=('ambiguous_2d_interval' if ambiguity else 'observed_2d_interval') if plant else 'unknown',plant_proposal_intervals=proposals,contact_confidence=.2 if ambiguity else .6 if plant else .0,
  semantics='Image observations, not manual event ground truth. Unknown contact receives weak/zero constraint.')

def contact(h,event,background=None):
 xy,s=smooth_xy(h,background);xy=xy[:,20:26];s=s[:,20:26];v=np.linalg.norm(np.gradient(xy,h['source_time'],axis=0),axis=-1);w=np.exp(-(v/.3)**2)*s**2
 plant=event.get('plant_interval');burst=event.get('arm_fast_interval');time=np.arange(len(xy));lead=0 if event.get('lead_side')=='left' else 1;li=[lead,lead+2,lead+4];ri=[1-lead,3-lead,5-lead]
 if plant:
  ramp=np.clip((time-plant[0])/max(1,plant[1]-plant[0]),0,1);w[:,li]*=ramp[:,None];w[:,ri]*=(1-ramp[:,None])
 else:w*=.1
 w*=float(event.get('contact_confidence',.2))
 if burst:
  fade=np.clip((burst[1]+.35*float(h['fps'])-time)/max(1,.35*float(h['fps'])),0,1);w*=fade[:,None]
 return w.astype(np.float32)

def main():
 out=[]
 for rec in sample():
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];e=infer(h,bg);e.update(id=cid,legacy_lift=rec.get('lift'),legacy_plant=rec.get('plant'));out.append(e);print(cid,e.get('arm_fast_peak'),e.get('lift_interval'),e.get('plant_interval'),flush=True)
 save(OUT/'events_tune.json',out);csvwrite(OUT/'events_tune.csv',out)
if __name__=='__main__':main()
