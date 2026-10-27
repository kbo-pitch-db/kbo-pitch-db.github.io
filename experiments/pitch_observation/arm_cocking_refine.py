"""Post-release arm-only MotionBERT hypotheses; never modify the main release.

MotionBERT observes elbow/wrist positions, not wrist/palm flexion angles. Only
throwing shoulder and elbow rotations change; wrist/palm local rotations,
shape, lower-body articulation, camera and native-pelvis trajectory stay fixed.
"""
from common import *
from all813_release import ROOT
from hybrid_initializer import global_rotations
from pitch_study_fit import StudyFitter,stable_rotation
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg
from scipy.spatial.transform import Rotation
import argparse
from arm_cocking_axial import CONFIG as AXIAL_CONFIG

FOLDER=OUT/'arm_cocking_v1'
CONFIG=dict(version='arm-cocking-v1',axial_component=AXIAL_CONFIG,steps=350,depth_weights=[0.,.02,.10,.02,.10,0.,.02,.10,.02,0.,0.,.02,.02],
 sources=['single','single','single','flip','flip','single','single','single','flip','gv','gv','native_single','native_flip'],
 components=['full','full','full','full','full','forearm','forearm','forearm','forearm','full','forearm','full','full'],withhold_every_fifth=True,
 arm_only=True,shape_camera_root_lower_body_frozen=True,
 controls='GVHMR-only arm refit shares the MB source-refit objective and SO3 search space; no MB initialization/prior',
 wrist_local_rotations='Preserved: 17-joint lifting does not observe palm/wrist flexion',
 phase='Observed fast-arm proxy with soft pre-burst window; ball release unknown',
 meaning='Camera-depth hypotheses and original-source fit; not measured external-rotation or wrist-cocking accuracy')
LABELS=['mb_single_source_refit','mb_single_phase_002','mb_single_phase_010','mb_flip_phase_002','mb_flip_phase_010','mb_forearm_source_refit','mb_forearm_phase_002','mb_forearm_phase_010','mb_flip_forearm_phase_002','gv_source_refit','gv_forearm_source_refit','mb_native_phase_002','mb_native_flip_phase_002']

def deployment_guard():
 path=ROOT/'deployment.json'
 if not path.exists():raise RuntimeError('Main full813 release must be completed, verified and deployed BEFORE arm experiments')
 record=json.loads(path.read_text())
 if record.get('clips')!=813 or not record.get('verified_live'):raise RuntimeError('Verified full813 deployment record required')

def phase_window(h,phase):
 times=h['source_time'];event=phase['events'];interval=event.get('arm_fast_interval')
 if not interval:return np.zeros(len(times)),dict(status='unknown_no_arm_burst',release='unknown')
 a,b=interval;start=times[max(0,a)]-.28;end=times[min(len(times)-1,b)]+.08
 ramp=np.clip((times-start)/.10,0,1)*np.clip((end-times)/.10,0,1)
 return ramp.astype(np.float32),dict(status='2d_fast_arm_proxy',frame_interval=[max(0,int(np.searchsorted(times,start))),min(len(times)-1,int(np.searchsorted(times,end)))],release='unknown_no_ball_observation')

def prepare_graft(base,mb,bg,side,forearm_only=False):
 assert np.array_equal(base['frame_index'],mb['frame_index'])
 rotation=base['smpl_rotations'].copy();g=global_rotations(rotation);mg=global_rotations(mb['smpl_rotations'])
 # Original-image MB vectors -> stabilized camera basis. A similarity's
 # translation and zoom do not rotate a direction or establish metric scale.
 angle=np.arctan2(bg[:,1,0],bg[:,0,0]);roll=Rotation.from_euler('z',angle).as_matrix();shoulder=16+side;elbow=18+side;collar=13+side
 if forearm_only:rotation[:,elbow]=np.swapaxes(g[:,shoulder],-1,-2)@roll@mg[:,elbow]
 else:
  rotation[:,shoulder]=np.swapaxes(g[:,collar],-1,-2)@roll@mg[:,shoulder]
  rotation[:,elbow]=mb['smpl_rotations'][:,elbow]
 return rotation

def candidate_geometry(base,rot,p,a,f):
 # Canonical native-pelvis representation. This is a root-definition
 # conversion, not a fitted motion/centering correction: global native
 # pelvis is exactly the released trajectory for every candidate/frame.
 translation=base['translation'].astype(np.float64)+base['anatomical_keypoints'][:,0].astype(np.float64)-a[:,0].astype(np.float64)
 candidate={**base,'keypoints':p,'anatomical_keypoints':a,'foot_keypoints':f,'smpl_rotations':rot,'translation':translation}
 assert np.max(abs((a[:,:16]+translation[:,None])-(base['anatomical_keypoints'][:,:16].astype(np.float64)+base['translation'][:,None].astype(np.float64))))<.00002
 assert np.array_equal(rot[:,:16],base['smpl_rotations'][:,:16])
 assert np.array_equal(rot[:,20:],base['smpl_rotations'][:,20:])
 return candidate

def fit(cid,body):
 import torch
 source=ROOT/'selected'/f'{cid}.npz';base=load(source);meta=json.loads(source.with_suffix('.json').read_text());_,_,_,h,_=init_inputs(cid);bg=base['background_transforms'];phase=meta['phase'];hand=phase['throwing_hand'];side_name=hand if hand in ['left','right'] else phase['events'].get('arm_fast_side')
 if side_name not in ['left','right']:raise ValueError('Throwing side unknown; arm modification withheld')
 side=0 if side_name=='left' else 1
 hand_status='metadata' if hand in ['left','right'] else 'uncertain_2d_arm_burst_hypothesis'
 mbs={name:load(OUT/'adapters'/tag/f'{cid}.npz') for name,tag in [('single','mb_block_single_smpl'),('flip','mb_block_flip_smpl')]}
 native_mb={name:load(OUT/'raw'/tag/f'{cid}.npz') for name,tag in [('native_single','mb_block_single'),('native_flip','mb_block_flip_restored')]}
 mb_native_hash={name:sha(OUT/'raw'/tag/f'{cid}.npz') for name,tag in [('native_single','mb_block_single'),('native_flip','mb_block_flip_restored')]}
 mb_hash={name:sha(OUT/'adapters'/tag/f'{cid}.npz') for name,tag in [('single','mb_block_single_smpl'),('flip','mb_block_flip_smpl')]}
 solver_files_sha256={name:sha(Path(__file__).with_name(name)) for name in ['arm_cocking_refine.py','arm_cocking_axial.py','arm_cocking_native.py']}
 dest=FOLDER/'reports'/f'{cid}.json'
 if dest.exists():
  cached=json.loads(dest.read_text())
  if cached['config']!=CONFIG or cached['base_fit_sha256']!=sha(source) or cached['mb_adapter_sha256']!=mb_hash or cached['mb_native_sha256']!=mb_native_hash or cached['solver_files_sha256']!=solver_files_sha256:raise ValueError('Changed input/config requires a new experiment version')
  return cached
 N=len(h['keypoints']);leg=native_leg(base['anatomical_keypoints']);beta=base['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;body.prepare_shape(beta);H=len(LABELS);T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0');shoulder=16+side;elbow=18+side;wrist=20+side
 graft={name:prepare_graft(base,mb,bg,side) for name,mb in mbs.items()};forearm={name:prepare_graft(base,mb,bg,side,True) for name,mb in mbs.items()}
 from arm_cocking_native import native_graft
 native_diagnostics={}
 for name,native in native_mb.items():graft[name],native_diagnostics[name]=native_graft(base,native,bg,side)
 initial=np.stack([base['smpl_rotations'] if s=='gv' else (forearm if component=='forearm' else graft)[s] for s,component in zip(CONFIG['sources'],CONFIG['components'])]);r0=T(initial);betas=T(np.repeat(beta[None],H*N,0));delta=torch.nn.Parameter(torch.zeros((H,N,2,3),device='cuda:0'));optimizer=torch.optim.Adam([delta],lr=.015)
 mobility=np.ones((H,1,2,1),np.float32)
 for i,component in enumerate(CONFIG['components']):
  if component=='forearm':mobility[i,:,0]=0
 mobility=T(mobility)
 target_raw=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
 target=T(np.nan_to_num(target_raw[:,:17]));weight=scores(h)[:,:17]**2;mask=np.arange(N)%5==2;weight[mask]=0;weight=T(weight);height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));principal=T(base['principal']);pelvis_world=T(base['anatomical_keypoints'][:,0]+base['translation']);gate_array,window=phase_window(h,phase);gate=T(gate_array)
 confidence=T(.05+.95*np.minimum(scores(h)[:,7+side],scores(h)[:,9+side]));depth_strength=T(CONFIG['depth_weights']);gv_rot=T(base['smpl_rotations']);gv_arm=T(base['anatomical_keypoints'][:,[elbow,wrist]]-base['anatomical_keypoints'][:,shoulder,None]);gv_rot_arm=gv_rot[:,[shoulder,elbow]];relative=np.swapaxes(base['smpl_rotations'][:,[shoulder,elbow]],-1,-2)[None]@initial[:,:,[shoulder,elbow]];initial_relative=T(Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(H,N,2,3))
 links=(np.diff(h['frame_index'])==1)&(np.diff(h['source_time'])>0)&(np.diff(h['source_time'])<.1)
 if 'track_id' in h:links&=np.diff(h['track_id'])==0
 triple=links[1:]&links[:-1];temporal=T(triple.astype(float));dt=T(np.diff(h['source_time']));mid=(dt[1:]+dt[:-1])/2
 with torch.no_grad():
  _,a0,_=body.model(r0.reshape(-1,24,3,3),betas);a0=a0.reshape(H,N,24,3);target_depth=(a0[:,:,[elbow,wrist],2]-a0[:,:,shoulder,None,2])/leg
 def evaluate():
  correction=stable_rotation(initial_relative+delta*mobility);arm_rot=gv_rot_arm[None]@correction
  rot=torch.cat([r0[:,:,:shoulder],arm_rot[:,:,0:1],r0[:,:,shoulder+1:elbow],arm_rot[:,:,1:2],r0[:,:,elbow+1:]],2)
  p,a,f=body.model(rot.reshape(-1,24,3,3),betas);p=p.reshape(H,N,17,3);a=a.reshape(H,N,24,3);f=f.reshape(H,N,6,3);tr=pelvis_world[None]-a[:,:,0]
  points=p+tr[:,:,None];uv=points[...,:2]/points[...,2:].clamp_min(.2)*float(base['focal'])+principal;err=(uv-target)/height
  arm_ids=[7+side,9+side];image=((torch.sqrt(err[:,:,arm_ids].square().sum(-1)+.005**2)-.005)*weight[None,:,arm_ids]).sum((1,2))/weight[:,arm_ids].sum().clamp_min(1)
  depth=(a[:,:,[elbow,wrist],2]-a[:,:,shoulder,None,2])/leg;prior=((depth-target_depth).square()*gate[None,:,None]*confidence[None,:,None]).mean((1,2))
  current_arm=a[:,:,[elbow,wrist]]-a[:,:,shoulder,None];outside=((current_arm-gv_arm)/leg).square().mean(-1)*(1-gate)[None,:,None];outside=outside.mean((1,2))*depth_strength.gt(0)
  rotation_prior=(arm_rot-gv_rot_arm[None]).square().mean((1,2,3,4))
  v=(current_arm[:,1:]-current_arm[:,:-1])/dt[None,:,None,None]/leg;acc=(v[:,1:]-v[:,:-1])/mid[None,:,None,None]/30
  smooth=(acc.square().mean((2,3))*temporal[None]*(1-.95*gate[1:-1])[None]).mean(1)
  loss=image+depth_strength*prior+.025*outside+.0005*rotation_prior+.00005*smooth
  return loss,(p,a,f,rot)
 start=time.monotonic()
 for step in range(CONFIG['steps']):
  optimizer.param_groups[0]['lr']=.015*(.05+.95*(1-step/(CONFIG['steps']-1))**2);optimizer.zero_grad(set_to_none=True);loss,_=evaluate()
  if not torch.isfinite(loss).all():raise ValueError('Nonfinite arm loss '+cid)
  loss.sum().backward();torch.nn.utils.clip_grad_norm_([delta],10);optimizer.step()
 with torch.no_grad():loss,data=evaluate()
 p,a,f,rot=[v.cpu().numpy() for v in data];outputs={'current_GVHMR':base}
 for name,r in graft.items():
  with torch.no_grad():pp,aa,ff=body.model(T(r),T(np.repeat(beta[None],N,0)))
  outputs['raw_graft_'+name]=candidate_geometry(base,r,pp.cpu().numpy(),aa.cpu().numpy(),ff.cpu().numpy())
 for name,r in forearm.items():
  with torch.no_grad():pp,aa,ff=body.model(T(r),T(np.repeat(beta[None],N,0)))
  outputs['raw_forearm_'+name]=candidate_geometry(base,r,pp.cpu().numpy(),aa.cpu().numpy(),ff.cpu().numpy())
 for i,name in enumerate(LABELS):outputs[name]=candidate_geometry(base,rot[i],p[i],a[i],f[i])
 from arm_cocking_axial import fit_axial
 axial,axial_diagnostics=fit_axial(base,mbs,bg,side,body,h,gate_array);outputs.update(axial)
 for name,z in outputs.items():
  _,uv=metrics(z,h,bg,z['translation'],base['contact_weight']);save_geometry(FOLDER/'fits'/name/f'{cid}.npz',z,z['translation'],uv,base['contact_weight'],base['normal'],float(base['pilot_support_offset']))
 report=dict(id=cid,mb_native_sha256=mb_native_hash,native_direction_adapter=native_diagnostics,solver_files_sha256=solver_files_sha256,config=CONFIG,axial_diagnostics=axial_diagnostics,base_fit_sha256=sha(source),mb_adapter_sha256=mb_hash,throwing_side='left' if side==0 else 'right',hand_status=hand_status,window=window,source_observation_holdout_frames=np.flatnonzero(mask).tolist(),holdout_semantics='Withheld only from this arm refinement; initializers and released GVHMR saw full sequence',variants=list(outputs),seconds=time.monotonic()-start,solver_sha256=sha(__file__),wrist_rotation_status='Unobserved by MotionBERT17; original local rotations preserved',human_review_complete=False,main_release_changed=False)
 save(dest,report);print('ARM COCKING',cid,round(report['seconds'],1),'s',flush=True);return report

def main():
 deployment_guard()
 import torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30,0)
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['tune','confirmation','all'],default='tune');ap.add_argument('--id');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();protocol=json.loads((FOLDER/'protocol.json').read_text());ids=protocol['tune']+protocol['confirmation'] if a.cohort=='all' else protocol[a.cohort]
 if a.id:ids=[cid for cid in ids if cid==a.id]
 ids=ids[a.worker::a.workers]
 if any(cid in protocol['confirmation'] for cid in ids) and not (FOLDER/'selection_rule.json').exists():raise RuntimeError('Freeze tuning selection before confirmation execution')
 save(FOLDER/f'configuration.worker{a.worker}.json',CONFIG);body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False)
 for cid in ids:fit(cid,body)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
