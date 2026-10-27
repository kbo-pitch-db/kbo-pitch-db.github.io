"""Persistent soft marker positions, not velocity-only contact or end-position clamping.

Frozen articulation with a bounded sequence-wide focal update; no population stride target, monotone root,
framewise centering, end hold or copied legacy trajectory. Research results
remain separate until reprojection and source evidence are checked.
"""
from common import *
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg,phase_info
from events import smooth_xy
from pitch_study_fit import spline_basis,stable_rotation
from scipy.ndimage import gaussian_filter1d,binary_closing
import argparse

CONFIG=dict(version='rear-pivot-rigid-v6',steps=1200,root_acc_weight=.01,
 anchor_weights=[.10],pivot_weights=[1.,5.],plane_angles=[0.,.10,.25],
 source_gate_p50_px_increase=2.,wrist_gate_p95_px_increase=2.,
 release='Sustained marker displacement and speed after contact; no automatic fade at arm burst',
 persistence='Static per-marker anchors with robust tangential loss; separate floor-normal loss',
 articulation='Frozen local articulation and shape; bounded spline correction of root orientation; interankle3D invariant',arm_observation_weight=2.,camera='Bounded shared focal; no measured intrinsics',physical_timing='Broadcast only; slow-motion factor unknown')

def contact_state(h,phase,bg):
 xy,sc=smooth_xy(h,bg);xy=gaussian_filter1d(xy[:,20:26],max(1,float(h['fps'])*.06),axis=0);sc=sc[:,20:26];t=h['source_time'];n=len(t);plant=phase['plant_interval'];events=phase['events'];notes=[];columns=[];seeds=[]
 if not plant or phase['phase_side_conflict']:return np.zeros((n,0)),np.array([],int),[]
 lead=0 if phase['lead_side']=='left' else 1;speed=np.linalg.norm(np.gradient(xy,t,axis=0),axis=-1);window=max(3,round(float(h['fps'])*.10));confidence=float(events.get('contact_confidence',0))
 # This is a broad IMAGE support band, not a metric floor. It only rejects
 # obvious airborne pauses; all spatial/contact assignments remain hypotheses.
 ground=np.median([*xy[max(0,plant[0]-2):plant[1]+3,lead,1],*xy[:max(3,plant[0]//2),1-lead,1]])
 for j in range(6):
  stable=(speed[:,j]<.5)&(sc[:,j]>.35)&(xy[:,j,1]>ground-.12)
  if j%2==lead:stable[:plant[0]]=False
  stable=binary_closing(stable,structure=np.ones(max(2,round(float(h['fps'])*.06))),border_value=1)
  starts=np.flatnonzero(stable&~np.r_[False,stable[:-1]]);ends=np.flatnonzero(stable&~np.r_[stable[1:],False])
  for first,last in zip(starts,ends):
   if last-first+1<window:continue
   i0=max(plant[0] if j%2==lead else 0,first-1);i1=min(n-1,last+1);weight=np.zeros(n);weight[i0:i1+1]=confidence*sc[i0:i1+1,j]**2/(1+(speed[i0:i1+1,j]/.6)**2)
   notes.append(dict(marker=j,start=int(i0),end=int(i1),seed_frame=int((first+last)//2),semantics='Image-supported contact episode, not human contact GT'))
   columns.append(weight);seeds.append((first+last)//2)
 return np.stack(columns,1) if columns else np.zeros((n,0)),np.array(seeds,int),notes

def fit(rec,steps):
 import torch
 torch.set_num_threads(1);cid=rec['id'];folder=OUT/'rear_pivot_rigid_v6';dest=folder/f'{cid}.npz'
 if dest.exists() and dest.with_suffix('.json').exists():
  cached=json.loads(dest.with_suffix('.json').read_text())
  if cached['config']!=CONFIG or cached['source_fit_sha256']!=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz'):raise ValueError('Changed input/config requires a new versioned result directory')
  if 'root_orientation_hypotheses' in load(dest):return cached
 z=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz');_,_,_,h,_=init_inputs(cid);bg=z['background_transforms'];phase,_=phase_info(rec,h,bg);episode_weight,seeds,notes=contact_state(h,phase,bg);markers=np.array([q['marker'] for q in notes],int);cw=np.zeros((len(h['keypoints']),6));
 for e,j in enumerate(markers):cw[:,j]=np.maximum(cw[:,j],episode_weight[:,e])
 if not len(markers):return dict(id=cid,status='unknown_contact_no_fit',phase=phase)
 N=len(h['keypoints']);leg=native_leg(z['anatomical_keypoints']);T=lambda x:torch.tensor(x,dtype=torch.float64)
 hypotheses=[(w,a) for w in CONFIG['pivot_weights'] for a in CONFIG['plane_angles']];H=len(hypotheses);normal=T([[0,-np.cos(a),-np.sin(a)] for w,a in hypotheses]);pivot_strength=T([w for w,a in hypotheses]);basis=T(spline_basis(N,float(h['fps']),spacing=.08));initial=z['translation'].astype(float)/leg
 controls=torch.nn.Parameter(T(np.repeat(np.linalg.lstsq(basis.numpy(),initial,rcond=None)[0][None],H,axis=0)));points=T(np.concatenate([z['keypoints'],z['foot_keypoints']],1)/leg);feet=T(z['foot_keypoints'][:,markers]/leg)
 initfeet=z['foot_keypoints']/leg+initial[:,None];anchor0=np.array([np.median(initfeet[max(0,i-1):min(N,i+2),j],axis=0) for j,i in zip(markers,seeds)]);anchors=torch.nn.Parameter(T(np.repeat(anchor0[None],H,0)))
 contact=T(episode_weight);pivot_contact=np.zeros_like(episode_weight);rear=1 if phase['lead_side']=='left' else 0;pivot_notes=[]
 for side in [rear,1-rear]:
  eligible=[(i,q) for i,q in enumerate(notes) if q['marker']==side]
  if not eligible:continue
  e,q=eligible[0];start,end=q['start'],q['end'];t=h['source_time'];ramp=np.clip((t[end]-t)/.06,0,1)*np.clip((t-t[start])/.04,0,1);pivot_contact[:,e]=episode_weight[:,e]*ramp;pivot_notes.append(dict(episode=e,marker=side,start=start,end=end,heel_locked=False))
 grip_contact=T(pivot_contact);good=cw>.03;floor0=[]
 for n in normal.numpy():
  signed=(initfeet*n).sum(-1);floor0.append(-float(np.median(signed[good])) if good.any() else -float(np.quantile(signed,.1)))
 orientation=torch.nn.Parameter(torch.zeros((H,basis.shape[1],3),dtype=torch.float64));floor=torch.nn.Parameter(T(floor0));logf=torch.nn.Parameter(torch.zeros(H,dtype=torch.float64));opt=torch.optim.Adam([{'params':[controls,anchors,floor,orientation],'lr':.02},{'params':[logf],'lr':.003}])
 xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];target=T(np.nan_to_num(np.concatenate([xy[:,:17],xy[:,20:26]],1)));score=scores(h);weight=np.concatenate([score[:,:17]**2,score[:,20:26]**2*.35],1);weight[:,:5]*=.2;weight[:,7:11]*=2.;weight=T(weight);height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));dt=T(np.diff(h['source_time']));links=(dt>0)&(dt<.1)&T(np.diff(h['frame_index'])==1).bool();trip=links[:-1]&links[1:];mid=(dt[1:]+dt[:-1])/2
 def evaluate():
  tr=torch.einsum('nk,hkc->hnc',basis,controls);angles=.25*torch.tanh(torch.einsum('nk,hkc->hnc',basis,orientation)/.25);correction=stable_rotation(angles);posed=torch.einsum('hnij,nkj->hnki',correction,points);p=posed+tr[:,:,None];uv=p[...,:2]/p[...,2:].clamp_min(.2)*(float(z['focal'])*torch.exp(.3*torch.tanh(logf)))[:,None,None,None]+T(z['principal']);err=(uv-target)/height;image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1)
  v=(tr[:,1:]-tr[:,:-1])/dt[None,:,None];acc=(v[:,1:]-v[:,:-1])/mid[None,:,None];smooth=(acc[:,trip]/30).square().mean((1,2));absolute=torch.einsum('hnij,nkj->hnki',correction,feet)+tr[:,:,None];delta=absolute-anchors[:,None];dn=(delta*normal[:,None,None]).sum(-1);tangent=delta-dn[...,None]*normal[:,None,None]
  pin=((torch.sqrt(tangent.square().sum(-1)+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1);signed=(absolute*normal[:,None,None]).sum(-1)+floor[:,None,None];plane=((torch.sqrt(signed.square()+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1)
  prior=logf.square();gauge=((tr[:,0]-T(initial[0]))/max(1,abs(initial[0,2]))).square().mean(1)
  ori_prior=angles.square().mean((1,2));ori_temporal=(angles[:,2:]+angles[:,:-2]-2*angles[:,1:-1]).square().mean((1,2));total=image+.01*smooth+.00001*gauge+.0001*prior+.002*ori_prior+.001*ori_temporal
  if cw.sum()>0:total=total+.10*pin+.03*plane+.001*torch.relu(-signed-.03).square().mean((1,2));grip=(tangent.square().sum(-1)*grip_contact).sum((1,2))/grip_contact.sum().clamp_min(1);total=total+pivot_strength*grip
  return total,tr
 for i in range(steps):
  opt.param_groups[0]['lr']=.02*(.05+.95*(1-i/max(1,steps-1))**2);opt.param_groups[1]['lr']=.003*(.05+.95*(1-i/max(1,steps-1))**2);opt.zero_grad();loss,tr=evaluate()
  if not torch.isfinite(loss).all():raise ValueError('Nonfinite '+cid)
  loss.sum().backward();torch.nn.utils.clip_grad_norm_([controls,anchors,floor,orientation],10);opt.step()
 with torch.no_grad():loss,tr=evaluate()
 with torch.no_grad():angles=.25*torch.tanh(torch.einsum('nk,hkc->hnc',basis,orientation)/.25);corrections=stable_rotation(angles).numpy()
 base,_=metrics(z,h,bg,z['translation'].astype(float),cw);previous=load(OUT/'support_transfer_root_v1/root_support_transfer'/f'{cid}.npz');old,_=metrics(z,h,bg,previous['translation'],cw);records=[]
 for i,(w,a) in enumerate(hypotheses):
  translation=tr[i].numpy()*leg;candidate={**z,'focal':np.array(float(z['focal'])*float(torch.exp(.3*torch.tanh(logf[i]))))};
  for key in ['keypoints','anatomical_keypoints','foot_keypoints']:candidate[key]=np.einsum('nij,nkj->nki',corrections[i],z[key].astype(float))
  candidate['smpl_rotations']=z['smpl_rotations'].copy();candidate['smpl_rotations'][:,0]=corrections[i]@z['smpl_rotations'][:,0];m,_=metrics(candidate,h,bg,translation,cw);m.update(index=i,root_orientation_correction_p95_deg=float(np.percentile(np.linalg.norm(angles[i].numpy(),axis=-1)*180/np.pi,95)),pivot_weight=w,plane_angle=a,focal=float(candidate['focal']),objective=float(loss[i]),source_gate=m['source_p50_px']<=base['source_p50_px']+2 and m['wrist_p95_px']<=base['wrist_p95_px']+2);records.append(m)
 # Fixed selection: lowest objective among source-gated contact candidates;
 # keep all nine trajectories. Do not choose by the longest stride or endpoint.
 passed=[r for r in records if r['source_gate']];selected=min(passed or records,key=lambda r:r['objective'])['index'];translation=tr[selected].numpy()*leg;candidate={**z,'focal':np.array(records[selected]['focal'])};
 for key in ['keypoints','anatomical_keypoints','foot_keypoints']:candidate[key]=np.einsum('nij,nkj->nki',corrections[selected],z[key].astype(float))
 candidate['smpl_rotations']=z['smpl_rotations'].copy();candidate['smpl_rotations'][:,0]=corrections[selected]@z['smpl_rotations'][:,0];_,uv=metrics(candidate,h,bg,translation,cw);save_geometry(dest,candidate,translation,uv,cw,normal[selected].numpy(),float(floor[selected])*leg)
 with np.load(dest,allow_pickle=False) as q:payload={k:q[k] for k in q.files}
 atomic_npz(dest,**payload,root_hypotheses_translation=tr.numpy()*leg,persistent_marker_anchors=anchors[selected].detach().numpy()*leg,anchor_hypotheses=anchors.detach().numpy()*leg,contact_episode_weights=episode_weight,contact_episode_markers=markers,root_orientation_hypotheses=corrections)
 report=dict(id=cid,player=rec['player_name'],selected=selected,hypotheses=records,phase=phase,marker_states=notes,pivot_states=pivot_notes,contact_available=bool(cw.sum()>0),baseline=base,velocity_contact_v1=old,config=CONFIG,source_fit_sha256=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz'),human_review_complete=False)
 save(dest.with_suffix('.json'),report);print('PERSISTENT',rec['player_name'],selected,'source',round(records[selected]['source_p50_px'],2),'acc',round(records[selected]['root_acc_p95_leg_s2'],2),'gate',records[selected]['source_gate'],flush=True);return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample('tune',True)
 if a.id:records=[r for r in records if r['id']==a.id]
 if a.limit:records=records[:a.limit]
 folder=OUT/'rear_pivot_rigid_v6';folder.mkdir(exist_ok=True);save(folder/'protocol.json',dict(config=CONFIG,code_sha256=sha(__file__),test24_used=False));results=[fit(r,CONFIG['steps']) for r in records];save(folder/'results.json',results)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
