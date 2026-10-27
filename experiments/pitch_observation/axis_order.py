"""Conditional source-direction ordering evidence, with held-out frames.

No target stride. A clear CF view and source throwing-side evidence are needed.
All admissible perspective direction hypotheses must agree before calling a
reversal. The interval ends BEFORE the fast-arm interval; follow-through is
excluded. This is a conditional ordinal observation, not 3D angle/position GT.
"""
from common import *
from events import infer
from candidates import CC,norm


def setup(cid,h):
 path=OUT/'camera_observations'/cid/'observation.json'
 if not path.exists():return None
 obs=json.loads(path.read_text());axes=load(path.with_name('axes.npz'))['pitch_axes']
 if obs['view']!='center_field_offset' or obs['throwing_hand']=='unknown' or len(axes)<30:return None
 bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];event=infer(h,bg);plant=event.get('plant_interval');arm=event.get('arm_fast_interval')
 if plant is None or arm is None:return None
 first,last=plant[0],min(plant[1],arm[0]-1)
 if last<first:return None
 lead=4 if obs['throwing_hand']=='right' else 5
 return dict(frames=np.arange(first,last+1),axes=axes,lead=lead,observation=obs,event=event,frame_interval=[first,last])

def metrics(points,evidence):
 if evidence is None:return dict(status='unknown',selection_frames=0,withheld_frames=0)
 p,_=norm(points);frames=evidence['frames'];v=p[frames,evidence['lead']];dot=v@evidence['axes'].T;worst=dot.min(1);best=dot.max(1);out=dict(status='conditional_directional_order',interval=evidence['frame_interval'],direction_hypotheses=len(evidence['axes']),lead='left' if evidence['lead']==4 else 'right',absolute_distance_validated=False)
 for label,keep in [('selection',frames%3!=2),('withheld',frames%3==2)]:
  valid=keep&np.isfinite(dot).all(1);out[label+'_frames']=int(valid.sum());out[label+'_robust_reverse_count']=int(((best<-.05)&valid).sum());out[label+'_robust_reverse_fraction']=float(np.mean(best[valid]<-.05)) if valid.any() else None;out[label+'_ambiguous_count']=int(((worst<-.05)&(best>=-.05)&valid).sum());out[label+'_robust_min_direction_leg']=float(best[valid].min()) if valid.any() else None
 return out
