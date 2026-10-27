"""Persistent soft marker positions, not velocity-only contact or end-position clamping.

Frozen articulation and camera; no population stride target, monotone root,
framewise centering, end hold or copied legacy trajectory. Research results
remain separate until reprojection and source evidence are checked.
"""
from common import *
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg,phase_info
from events import smooth_xy
from pitch_study_fit import spline_basis
from scipy.ndimage import gaussian_filter1d
import argparse

CONFIG=dict(version='persistent-contact-v2',steps=650,root_acc_weight=.01,
 anchor_weights=[.01,.03,.10],plane_angles=[0.,.10,.25],
 source_gate_p50_px_increase=2.,wrist_gate_p95_px_increase=2.,
 release='Sustained marker displacement and speed after contact; no automatic fade at arm burst',
 persistence='Static per-marker anchors with robust tangential loss; separate floor-normal loss',
 articulation='Exact unchanged f4 pose/shape/rotation',physical_timing='Broadcast only; slow-motion factor unknown')

def contact_state(h,phase,bg):
 xy,sc=smooth_xy(h,bg);xy=gaussian_filter1d(xy[:,20:26],max(1,float(h['fps'])*.06),axis=0);sc=sc[:,20:26];t=h['source_time'];n=len(t);weights=np.zeros((n,6));seeds=np.zeros(6,int);plant=phase['plant_interval'];events=phase['events']
 if not plant or phase['phase_side_conflict']:return weights,seeds,[]
 lead=0 if phase['lead_side']=='left' else 1;rear=1-lead
 speed=np.linalg.norm(np.gradient(xy,t,axis=0),axis=-1);notes=[];window=max(3,round(float(h['fps'])*.12));confidence=float(events.get('contact_confidence',0))
 for j in range(6):
  is_lead=j%2==lead
  seed=plant[1] if is_lead else (events.get('lift_peak') or max(0,plant[0]//3))
  seeds[j]=seed;reference=np.median(xy[max(0,seed-1):min(n,seed+2),j],axis=0)
  start=plant[1] if is_lead else min(seed+1,n-1);departure=None
  for i in range(start,n-window+1):
   distance=np.linalg.norm(xy[i:i+window,j]-reference,axis=-1)
   if (distance>.06).all() and np.median(speed[i:i+window,j])>.4 and np.mean(sc[i:i+window,j]>.35)>.7:departure=i;break
  base=confidence*sc[:,j]**2/(1+(speed[:,j]/.6)**2)
  if is_lead:base*=np.clip((t-t[plant[0]])/max(.03,t[plant[1]]-t[plant[0]]),0,1)
  if departure is not None:base*=np.clip((t[departure]+.08-t)/.08,0,1)
  weights[:,j]=base;notes.append(dict(marker=j,seed_frame=int(seed),departure_frame=departure,available=True))
 return weights,seeds,notes

def fit(rec,steps):
 import torch
 torch.set_num_threads(1);cid=rec['id'];folder=OUT/'support_transfer_anchor_v2';dest=folder/f'{cid}.npz'
 if dest.exists() and dest.with_suffix('.json').exists():
  cached=json.loads(dest.with_suffix('.json').read_text())
  if cached['config']!=CONFIG or cached['source_fit_sha256']!=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz'):raise ValueError('Changed input/config requires a new versioned result directory')
  return cached
 z=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz');_,_,_,h,_=init_inputs(cid);bg=z['background_transforms'];phase,_=phase_info(rec,h,bg);cw,seeds,notes=contact_state(h,phase,bg);N=len(h['keypoints']);leg=native_leg(z['anatomical_keypoints']);T=lambda x:torch.tensor(x,dtype=torch.float64)
 hypotheses=[(w,a) for w in CONFIG['anchor_weights'] for a in CONFIG['plane_angles']];H=len(hypotheses);normal=T([[0,-np.cos(a),-np.sin(a)] for w,a in hypotheses]);strength=T([w for w,a in hypotheses]);basis=T(spline_basis(N,float(h['fps']),spacing=.08));initial=z['translation'].astype(float)/leg
 controls=torch.nn.Parameter(T(np.repeat(np.linalg.lstsq(basis.numpy(),initial,rcond=None)[0][None],H,axis=0)));points=T(np.concatenate([z['keypoints'],z['foot_keypoints']],1)/leg);feet=T(z['foot_keypoints']/leg)
 initfeet=z['foot_keypoints']/leg+initial[:,None];anchor0=np.array([np.median(initfeet[max(0,i-1):min(N,i+2),j],axis=0) for j,i in enumerate(seeds)]);anchors=torch.nn.Parameter(T(np.repeat(anchor0[None],H,0)))
 contact=T(cw);good=cw>.03;floor0=[]
 for n in normal.numpy():
  signed=(initfeet*n).sum(-1);floor0.append(-float(np.median(signed[good])) if good.any() else -float(np.quantile(signed,.1)))
 floor=torch.nn.Parameter(T(floor0));opt=torch.optim.Adam([controls,anchors,floor],lr=.02)
 xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];target=T(np.nan_to_num(np.concatenate([xy[:,:17],xy[:,20:26]],1)));score=scores(h);weight=np.concatenate([score[:,:17]**2,score[:,20:26]**2*.35],1);weight[:,:5]*=.2;weight=T(weight);height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));dt=T(np.diff(h['source_time']));links=(dt>0)&(dt<.1)&T(np.diff(h['frame_index'])==1).bool();trip=links[:-1]&links[1:];mid=(dt[1:]+dt[:-1])/2
 def evaluate():
  tr=torch.einsum('nk,hkc->hnc',basis,controls);p=points[None]+tr[:,:,None];uv=p[...,:2]/p[...,2:].clamp_min(.2)*float(z['focal'])+T(z['principal']);err=(uv-target)/height;image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1)
  v=(tr[:,1:]-tr[:,:-1])/dt[None,:,None];acc=(v[:,1:]-v[:,:-1])/mid[None,:,None];smooth=(acc[:,trip]/30).square().mean((1,2));absolute=feet[None]+tr[:,:,None];delta=absolute-anchors[:,None];dn=(delta*normal[:,None,None]).sum(-1);tangent=delta-dn[...,None]*normal[:,None,None]
  pin=((torch.sqrt(tangent.square().sum(-1)+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1);signed=(absolute*normal[:,None,None]).sum(-1)+floor[:,None,None];plane=((torch.sqrt(signed.square()+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1)
  prior=(anchors-T(anchor0)).square().mean((1,2));gauge=(tr[:,0]-T(initial[0])).square().mean(1)
  total=image+.01*smooth+.0001*gauge
  if cw.sum()>0:total=total+strength*pin+.03*plane+.005*prior+.001*torch.relu(-signed-.03).square().mean((1,2))
  return total,tr
 for i in range(steps):
  opt.param_groups[0]['lr']=.02*(.05+.95*(1-i/max(1,steps-1))**2);opt.zero_grad();loss,tr=evaluate()
  if not torch.isfinite(loss).all():raise ValueError('Nonfinite '+cid)
  loss.sum().backward();torch.nn.utils.clip_grad_norm_([controls,anchors,floor],10);opt.step()
 with torch.no_grad():loss,tr=evaluate()
 base,_=metrics(z,h,bg,z['translation'].astype(float),cw);previous=load(OUT/'support_transfer_root_v1/root_support_transfer'/f'{cid}.npz');old,_=metrics(z,h,bg,previous['translation'],cw);records=[]
 for i,(w,a) in enumerate(hypotheses):
  translation=tr[i].numpy()*leg;m,_=metrics(z,h,bg,translation,cw);m.update(index=i,anchor_weight=w,plane_angle=a,objective=float(loss[i]),source_gate=m['source_p50_px']<=base['source_p50_px']+2 and m['wrist_p95_px']<=base['wrist_p95_px']+2);records.append(m)
 # Fixed selection: lowest objective among source-gated contact candidates;
 # keep all nine trajectories. Do not choose by the longest stride or endpoint.
 passed=[r for r in records if r['source_gate']];selected=min(passed or records,key=lambda r:r['objective'])['index'];translation=tr[selected].numpy()*leg;_,uv=metrics(z,h,bg,translation,cw);save_geometry(dest,z,translation,uv,cw,normal[selected].numpy(),float(floor[selected])*leg)
 with np.load(dest,allow_pickle=False) as q:payload={k:q[k] for k in q.files}
 atomic_npz(dest,**payload,root_hypotheses_translation=tr.numpy()*leg,persistent_marker_anchors=anchors[selected].detach().numpy()*leg,anchor_hypotheses=anchors.detach().numpy()*leg)
 report=dict(id=cid,player=rec['player_name'],selected=selected,hypotheses=records,phase=phase,marker_states=notes,contact_available=bool(cw.sum()>0),baseline=base,velocity_contact_v1=old,config=CONFIG,source_fit_sha256=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz'),human_review_complete=False)
 save(dest.with_suffix('.json'),report);print('PERSISTENT',rec['player_name'],selected,'source',round(records[selected]['source_p50_px'],2),'acc',round(records[selected]['root_acc_p95_leg_s2'],2),'gate',records[selected]['source_gate'],flush=True);return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample('tune',True)
 if a.id:records=[r for r in records if r['id']==a.id]
 if a.limit:records=records[:a.limit]
 folder=OUT/'support_transfer_anchor_v2';folder.mkdir(exist_ok=True);save(folder/'protocol.json',dict(config=CONFIG,code_sha256=sha(__file__),test24_used=False));results=[fit(r,CONFIG['steps']) for r in records];save(folder/'results.json',results)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
