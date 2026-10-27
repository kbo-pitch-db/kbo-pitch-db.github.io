"""Joint lower-body/root/camera refinement; preserve the existing large-stride hypothesis.

No target average stride, render shifts or full-foot XYZ anchor. One toe per
initial support phase gets a normalized quadratic tangential closure. Heels
can rise and every contact releases according to observed episode boundaries.
F4 body shape and upper-limb local rotations remain fixed; small lower-body
and trunk corrections are evaluated against source observations, not GT.
"""
from common import *
from support_transfer_arm import contact_state
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg,phase_info
from pitch_study_fit import StudyFitter,stable_rotation,spline_basis
from scipy.spatial.transform import Rotation
import argparse

CONFIG=dict(version='rear-pivot-consistent-v8',steps=850,pivot_weights=[1.,5.],plane_angles=[0.,.1,.25],
 stride_vector_weight=4.,pose_correction_weight=.002,upper_relative_weight=.3,
 arm_observation_weight=2.,max_source_p50_increase_px=2.,max_wrist_p95_increase_px=2.,
 shape='fixed sequence shape',arm_local_rotations='unchanged; no smoothing',
 contact='single toe tangent closure per initial support, normal soft; release/heel lift allowed')

def rear_diagnostics(z,reference,meta):
 side=meta['phase']['lead_side'];rear=1 if side=='left' else 0 if side=='right' else None;episode=next((q for q in meta['marker_states'] if q['marker']==rear),None)
 if episode is None:return dict(rear_marker=rear,rear_interval=None,rear_backslide_from_start_leg=None,rear_max_retracement_leg=None,rear_tangent_extent_leg=None,rear_point_extent_3d_leg=None,semantics='Unknown/unobserved rear contact; no contact accuracy claim')
 n=reference['normal'];axis=np.array([0.,0.,1.]);axis-=axis@n*n;axis/=np.linalg.norm(axis);leg=native_leg(reference['anatomical_keypoints'])
 q=z['foot_keypoints'].astype(float)+z['translation'][:,None];start=episode['start'];end=episode['end'];d=q[start:end+1,rear]-q[start,rear];p=d@axis/leg
 tangent=d-(d@n)[:,None]*n
 return dict(rear_marker=rear,rear_interval=[start,end],rear_backslide_from_start_leg=float(max(0,-p.min())),rear_max_retracement_leg=float(np.max(np.maximum.accumulate(p)-p)),rear_tangent_extent_leg=float(np.linalg.norm(tangent,axis=-1).max()/leg),rear_point_extent_3d_leg=float(np.linalg.norm(d,axis=-1).max()/leg),semantics='Fixed v4 contact interval and plane, camera-depth tangent; not measured field-axis displacement')

OUTPUT_NAME='rear_pivot_consistent_v8'
SOURCE_FOLDER=OUT/'fits_strong/gvhmr_temporal_f4'
REFERENCE_FOLDER=OUT/'support_transfer_arm_v4'

def fit(rec,body,steps):
 import torch
 cid=rec['id'];folder=OUT/OUTPUT_NAME;dest=folder/f'{cid}.npz'
 source=SOURCE_FOLDER/f'{cid}.npz'
 if dest.exists() and dest.with_suffix('.json').exists():
  cached=json.loads(dest.with_suffix('.json').read_text())
  semantic=lambda c:{k:v for k,v in c.items() if k!='cuda_graph'}
  if semantic(cached['config'])!=semantic(CONFIG) or cached['source_fit_sha256']!=sha(source):raise ValueError('New version required for changed inputs/objective config')
  return cached
 z=load(source);_,_,_,h,_=init_inputs(cid);bg=z['background_transforms'];phase,_=phase_info(rec,h,bg);cw_e,seeds,notes=contact_state(h,phase,bg);markers=np.array([q['marker'] for q in notes],int);N=len(h['keypoints']);cw=np.zeros((N,6))
 rp=REFERENCE_FOLDER/f'{cid}.npz';mp=rp.with_suffix('.json')
 reference=load(rp) if rp.exists() and mp.exists() else z
 oldmeta=json.loads(mp.read_text()) if rp.exists() and mp.exists() else dict(phase=phase,marker_states=notes)
 for e,j in enumerate(markers):cw[:,j]=np.maximum(cw[:,j],cw_e[:,e])
 if not len(markers) and not CONFIG.get('allow_unknown_contact',False):return dict(id=cid,status='contact_unknown')
 T=lambda x:torch.as_tensor(x,dtype=torch.float32,device=body.device);beta=z['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;body.prepare_shape(beta)
 leg=native_leg(z['anatomical_keypoints']);r0=T(z['smpl_rotations']);hypotheses=[(w,a) for w in CONFIG['pivot_weights'] for a in CONFIG['plane_angles']];H=len(hypotheses)
 basis=T(spline_basis(N,float(h['fps']),spacing=.08));initial=z['translation']/leg;controls=torch.nn.Parameter(T(np.repeat(np.linalg.lstsq(basis.cpu().numpy(),initial,rcond=None)[0][None],H,0)));delta=torch.nn.Parameter(torch.zeros((H,N,24,3),device=body.device))
 betas=T(np.repeat(beta[None],H*N,0));normal=T([[0,-np.cos(a),-np.sin(a)] for w,a in hypotheses]);pivot_strength=T([w for w,a in hypotheses])
 initfeet=z['foot_keypoints']/leg+initial[:,None];anchor0=np.array([np.median(initfeet[max(0,i-1):min(N,i+2),j],0) for j,i in zip(markers,seeds)]).reshape(-1,3);anchors=torch.nn.Parameter(T(np.repeat(anchor0[None],H,0)))
 contact=T(cw_e);grip=np.zeros_like(cw_e);rear=1 if phase['lead_side']=='left' else 0 if phase['lead_side']=='right' else None;pivot_notes=[]
 sides=[] if rear is None else [rear] if CONFIG.get('rear_pivot_only',False) else [rear,1-rear]
 for side in sides:
  e=next((i for i,q in enumerate(notes) if q['marker']==side),None)
  if e is None:continue
  q=notes[e];a,b=q['start'],q['end'];times=h['source_time'];grip[:,e]=cw_e[:,e];pivot_notes.append(dict(episode=e,marker=side,start=a,end=b,heel_locked=False))
 grip=T(grip);good=cw>.03;floor0=[]
 for n in normal.cpu().numpy():
  signed=(initfeet*n).sum(-1);floor0.append(-float(np.median(signed[good])) if good.any() else -float(np.quantile(signed,.1)))
 knee_raw=Rotation.from_matrix(z['smpl_rotations'][:,[4,5]].reshape(-1,3,3)).as_rotvec().reshape(N,2,3);initflex=np.clip(knee_raw[:,:,0],.02,2.7);prob=np.clip((initflex+.08)/2.88,1e-4,1-1e-4);knee_flex=torch.nn.Parameter(T(np.repeat(np.log(prob/(1-prob))[None],H,0)));knee_side=torch.nn.Parameter(T(np.repeat(np.arctanh(np.clip(knee_raw[:,:,1:]/.18,-.95,.95))[None],H,0)))
 floor=torch.nn.Parameter(T(floor0));logf=torch.nn.Parameter(torch.zeros(H,device=body.device))
 # Local upper limbs stay exactly unchanged. Corrections are deliberately small
 # and knees get little off-hinge freedom; no target lower-limb configuration.
 limits=np.zeros((24,3),np.float32);limits[0]=.20;limits[[1,2]]=.40;limits[[3,6,9,12]]=.15;limits[[4,5]]=[.40,.06,.06];limits[[7,8,10,11]]=.40;limit=T(limits)
 graph_mode=CONFIG.get('cuda_graph',False)
 learning_rates=[torch.tensor(lr,device=body.device) if graph_mode else lr for lr in [.012,.012,.002]]
 optimizer=torch.optim.Adam([{'params':[controls,anchors,floor],'lr':learning_rates[0]},{'params':[delta,knee_flex,knee_side],'lr':learning_rates[1]},{'params':[logf],'lr':learning_rates[2]}],capturable=graph_mode,foreach=False if graph_mode else None)
 xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];target=T(np.nan_to_num(np.concatenate([xy[:,:17],xy[:,20:26]],1)));sc=scores(h);weight=np.concatenate([sc[:,:17]**2,sc[:,20:26]**2*CONFIG.get('foot_observation_weight',.35)],1);weight[:,:5]*=.2;weight[:,7:11]*=2;weight=T(weight)
 height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));dt=T(np.diff(h['source_time']));links=(dt>0)&(dt<.1)&T(np.diff(h['frame_index'])==1).bool();triple=links[1:]&links[:-1];mid=(dt[1:]+dt[:-1])/2
 base_p=T(z['keypoints']/leg);base_a=T(z['anatomical_keypoints']/leg);base_stride=base_a[:,7]-base_a[:,8]
 principal_tensor=T(z['principal']);knee_raw_tensor=T(knee_raw)
 triple_indices=torch.as_tensor(np.flatnonzero(triple.cpu().numpy()),dtype=torch.long,device=body.device);marker_indices=torch.as_tensor(markers,dtype=torch.long,device=body.device)
 upper_indices=torch.as_tensor([5,6,7,8,9,10],device=body.device);shoulder_indices=torch.as_tensor([5,6],device=body.device);base_upper=base_p[:,[5,6,7,8,9,10]]-base_p[:,[5,6]].mean(1,keepdim=True)
 def evaluate():
  angles=limit[None,None]*torch.tanh(delta);rot=r0[None]@stable_rotation(angles);
  if CONFIG.get('bounded_knees',False):
   knee_vec=torch.cat([(-.08+2.88*torch.sigmoid(knee_flex))[...,None],.18*torch.tanh(knee_side)],-1);rot=torch.cat([rot[:,:,:4],stable_rotation(knee_vec),rot[:,:,6:]],2)
  p,a,f=body.model(rot.reshape(-1,24,3,3),betas);p=p.reshape(H,N,17,3)/leg;a=a.reshape(H,N,24,3)/leg;f=f.reshape(H,N,6,3)/leg
  tr=torch.einsum('nk,hkc->hnc',basis,controls);focal=float(z['focal'])*torch.exp(.3*torch.tanh(logf));points=torch.cat([p,f],2)+tr[:,:,None];uv=points[...,:2]/points[...,2:].clamp_min(.2)*focal[:,None,None,None]+principal_tensor;error=(uv-target)/height;image=((torch.sqrt(error.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1)
  v=(tr[:,1:]-tr[:,:-1])/dt[None,:,None];acc=(v[:,1:]-v[:,:-1])/mid[None,:,None];smooth=(acc.index_select(1,triple_indices)/30).square().mean((1,2));absolute=f.index_select(2,marker_indices)+tr[:,:,None];d=absolute-anchors[:,None];dn=(d*normal[:,None,None]).sum(-1);tangent=d-dn[...,None]*normal[:,None,None]
  pin=((torch.sqrt(tangent.square().sum(-1)+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1);closure=(tangent.square().sum(-1)*grip).sum((1,2))/grip.sum().clamp_min(1);signed=(absolute*normal[:,None,None]).sum(-1)+floor[:,None,None];plane=((torch.sqrt(signed.square()+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1)
  stride=(a[:,:,7]-a[:,:,8]-base_stride).square().mean((1,2));stride_distance=(torch.linalg.vector_norm(a[:,:,7]-a[:,:,8],dim=-1)-torch.linalg.vector_norm(base_stride,dim=-1)).square().mean(1);upper=(p.index_select(2,upper_indices)-p.index_select(2,shoulder_indices).mean(2,keepdim=True)-base_upper).square().mean((1,2,3));pose_prior=angles.square().mean((1,2,3));temporal=(angles[:,2:]+angles[:,:-2]-2*angles[:,1:-1]).square().mean((1,2,3))
  loss=image+.01*smooth+.10*pin+.03*plane+pivot_strength*closure+CONFIG.get('pivot_normal_weight',0.)*pivot_strength*(signed.square()*grip).sum((1,2))/grip.sum().clamp_min(1)+CONFIG['stride_vector_weight']*stride+CONFIG.get('stride_distance_weight',0.)*stride_distance+CONFIG['upper_relative_weight']*upper+CONFIG['pose_correction_weight']*pose_prior+.002*temporal+.0001*logf.square()

  if CONFIG.get('bounded_knees',False):loss=loss+.002*(knee_vec-knee_raw_tensor).square().mean((1,2,3))+.002*(knee_vec[:,2:]+knee_vec[:,:-2]-2*knee_vec[:,1:-1]).square().mean((1,2,3))
  return loss,(p,a,f,tr,focal,rot)
 start=time.monotonic()
 parameters=[controls,anchors,floor,delta,knee_flex,knee_side,logf]
 def rates(step):
  decay=.05+.95*(step/max(1,steps-1)-1)**2
  for group,lr in zip(optimizer.param_groups,[.012,.012,.002]):
   if graph_mode:group['lr'].fill_(lr*decay)
   else:group['lr']=lr*decay
 def train_step():
  optimizer.zero_grad(set_to_none=not graph_mode);loss,_=evaluate();loss.sum().backward();torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step();return loss
 if graph_mode:
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for step in range(min(3,steps)):rates(step);loss=train_step()
  torch.cuda.current_stream().wait_stream(stream);rates(3)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):loss=train_step()
  # Warmup and the capture itself each count as real optimizer steps.
  for step in range(4,steps):
   rates(step);graph.replay()
   if step%100==0 and not torch.isfinite(loss).all():raise ValueError('Nonfinite '+cid)
 else:
  for step in range(steps):
   rates(step);loss=train_step()
   if not torch.isfinite(loss).all():raise ValueError('Nonfinite '+cid)
 with torch.no_grad():loss,data=evaluate()
 if not torch.isfinite(loss).all():raise ValueError('Nonfinite final '+cid)
 p,a,f,tr,focal,rot=[v.detach().cpu().numpy() for v in data];base,_=metrics(z,h,bg,z['translation'],cw);rows_=[];candidates=[]
 base_sep=np.linalg.norm(z['anatomical_keypoints'][:,7]-z['anatomical_keypoints'][:,8],axis=-1)
 for i,(w,angle) in enumerate(hypotheses):
  candidate={**z,'keypoints':p[i]*leg,'anatomical_keypoints':a[i]*leg,'foot_keypoints':f[i]*leg,'translation':tr[i]*leg,'focal':focal[i],'smpl_rotations':rot[i]};m,uv=metrics(candidate,h,bg,candidate['translation'],cw);m.update(rear_diagnostics(candidate,reference,oldmeta));sep=np.linalg.norm(candidate['anatomical_keypoints'][:,7]-candidate['anatomical_keypoints'][:,8],axis=-1);source_ok=all(m[key] is not None and base[key] is not None and m[key]<=base[key]+2 for key in ['source_p50_px','wrist_p95_px']);m.update(index=i,pivot_weight=w,plane_angle=angle,source_gate=source_ok,stride_change_p95_leg=float(np.percentile(abs(sep-base_sep)/leg,95)),objective=float(loss[i]),focal=float(focal[i]));
  m['feet_gate']=m['feet_p95_px'] is not None and base['feet_p95_px'] is not None and m['feet_p95_px']<=base['feet_p95_px']+CONFIG.get('max_feet_p95_increase_px',float('inf'));m['source_gate']=m['source_gate'] and m['feet_gate'];rows_.append(m);candidates.append((candidate,uv))
 eligible=[r for r in rows_ if r['source_gate'] and (r['rear_interval'] is None or (r['rear_tangent_extent_leg']<=.05 and r['rear_max_retracement_leg']<=.05 and r.get('rear_point_extent_3d_leg',0)<=CONFIG.get('max_rear_point_extent_3d_leg',float('inf')))) and r['stride_change_p95_leg']<=.05]
 selected=min(eligible,key=lambda r:r['source_p50_px'])['index'] if eligible else min(rows_,key=lambda r:r['objective'])['index'];candidate,uv=candidates[selected];save_geometry(dest,candidate,candidate['translation'],uv,cw,normal[selected].cpu().numpy(),float(floor[selected])*leg)
 with np.load(dest,allow_pickle=False) as saved:payload={k:saved[k] for k in saved.files}
 atomic_npz(dest,**payload,all_keypoints=p*leg,all_anatomical_keypoints=a*leg,all_foot_keypoints=f*leg,root_hypotheses_translation=tr*leg,all_smpl_rotations=rot,all_focal=focal,contact_episode_weights=cw_e,contact_episode_markers=markers,all_plane_normals=normal.cpu().numpy(),all_plane_offsets=floor.detach().cpu().numpy()*leg,all_persistent_anchors=anchors.detach().cpu().numpy()*leg)
 report=dict(id=cid,player=rec['player_name'],phase=phase,marker_states=notes,pivot_states=pivot_notes,baseline=base,reference_v4=rear_diagnostics(reference,reference,oldmeta),hypotheses=rows_,selected=selected,eligible_for_review=bool(eligible),contact_available=bool(len(markers)),seconds=time.monotonic()-start,config=CONFIG,source_fit_sha256=sha(source),human_review_complete=False)
 save(dest.with_suffix('.json'),report);print('JOINT',rec['player_name'],'eligible',len(eligible),'selected',selected,'sec',round(report['seconds'],1),flush=True);print(json.dumps(rows_,indent=2),flush=True);return report

def main():
 import torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15,0);ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();records=sample('tune',True)
 if args.id:records=[r for r in records if r['id']==args.id]
 if args.limit:records=records[:args.limit]
 folder=OUT/OUTPUT_NAME;folder.mkdir(exist_ok=True);save(folder/'protocol.json',dict(config=CONFIG,code_sha256=sha(__file__),test24_used=False));body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);results=[fit(r,body,CONFIG['steps']) for r in records];save(folder/'results.json',results)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
