"""Localized arm-marker fitting with connected temporal corrections.

Native SMPL arm FK is exact. Observation markers are baseline rigid offsets
attached to humerus/forearm; other markers and feet are explicitly frozen.
This is NOT full-mesh SMPL pose-corrective surface inference. No floor/root
fixup, arm-length stretching, depth averaging or per-frame display centering.
V2 was designed using V1 tuning only, before examining its confirmation.
"""
from common import *
from all813_release import ROOT
from arm_cocking_refine import deployment_guard,phase_window,prepare_graft
from arm_cocking_native import native_graft
from hybrid_initializer import global_rotations
from pitch_study_fit import stable_rotation,spline_basis
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg
from scipy.spatial.transform import Rotation
import argparse

FOLDER=OUT/'arm_cocking_v3'
SETTINGS=[
 ('gv_ray_perframe','ray_gv',False,0.),('mb_ray_perframe','ray_mb',False,0.),
 ('gv_ray_spline','ray_gv',True,0.),('mb_ray_spline','ray_mb',True,0.),('mb_flip_ray_spline','ray_flip',True,0.),
 ('mb_ray_spline_d002','ray_mb',True,.002),('mb_flip_ray_spline_d002','ray_flip',True,.002),('mb_ray_spline_d020','ray_mb',True,.02)]
CONFIG=dict(version='source-ray-depth-v3',steps=350,settings=SETTINGS,correction_spline_spacing_seconds=.06,withhold_every_fifth=True,
 marker_model='Fixed baseline marker offsets attached to current humerus/forearm; not full SMPL mesh correctives',
 lower_body='Native joints, other-arm points, feet, body/root/camera EXACTLY retained; this is structural scope, not independent accuracy evidence',
 temporal='Cubic spline of ROTATION CORRECTION, not smoothing the independent initial motion. Native frame/time retained.',
 diagnosis='V1/V2 tune only: transplanting MB image directions misaligns source; smooth corrections to a far-away pose are insufficient. Source-ray length constraints supply 4 depth candidates; MB selects depth branch, not XY',
 confirmation_not_examined_before_v3_seal=True)

LABELS={
 'current_GVHMR':'현재 메인 GVHMR',
 'gv_ray_perframe':'대조군 · GV 깊이 선택 · 프레임별 정합',
 'mb_ray_perframe':'MB 깊이 선택 · 프레임별 원본 정합',
 'gv_ray_spline':'대조군 · GV 깊이 선택 · 시간 연결',
 'mb_ray_spline':'MB 깊이 선택 · 시간 연결',
 'mb_flip_ray_spline':'MB flip 깊이 선택 · 시간 연결',
 'mb_ray_spline_d002':'MB 깊이 선택 · 약한 prior 0.002',
 'mb_flip_ray_spline_d002':'MB flip 깊이 선택 · 약한 prior 0.002',
 'mb_ray_spline_d020':'MB 깊이 선택 · prior 0.02',
 'raw_ray_gv':'원본 광선+길이 · GV 깊이 선택 초기',
 'raw_ray_mb':'원본 광선+길이 · MB 깊이 선택 초기',
 'raw_ray_flip':'원본 광선+길이 · MB flip 초기',
 'raw_ray_branch0':'깊이 가설 · 팔꿈치 near / 손목 near',
 'raw_ray_branch1':'깊이 가설 · 팔꿈치 near / 손목 far',
 'raw_ray_branch2':'깊이 가설 · 팔꿈치 far / 손목 near',
 'raw_ray_branch3':'깊이 가설 · 팔꿈치 far / 손목 far'}

def fit(cid):
 import torch
 source=ROOT/'selected'/f'{cid}.npz';base=load(source);meta=json.loads(source.with_suffix('.json').read_text());h=init_inputs(cid)[3];phase=meta['phase'];hand=phase['throwing_hand'];side=0 if hand=='left' else 1 if hand=='right' else None
 if side is None:raise ValueError('Verified hand required for V3')
 source_hashes=dict(base=sha(source),native=sha(OUT/'raw/mb_block_single'/f'{cid}.npz'),native_flip=sha(OUT/'raw/mb_block_flip_restored'/f'{cid}.npz'),solver=sha(__file__),ray_hypothesis_code=sha(Path(__file__).with_name('arm_depth_rays.py')),native_adapter_code=sha(Path(__file__).with_name('arm_cocking_native.py')),shared_phase_and_init_code=sha(Path(__file__).with_name('arm_cocking_refine.py')))
 dest=FOLDER/'reports'/f'{cid}.json'
 if dest.exists():
  cached=json.loads(dest.read_text());assert cached['source_hashes']==source_hashes and cached['config']==clean(CONFIG),'Changed V3 inputs/config require another version';return cached
 T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0');N=len(h['frame_index']);H=len(SETTINGS);s=16+side;e=18+side;w=20+side;palm=22+side;ce=7+side;cw=9+side;beta=base['smpl_betas'];leg=native_leg(base['anatomical_keypoints']);r=base['smpl_rotations'];g=global_rotations(r);bg=base['background_transforms'];bg_roll=Rotation.from_euler('z',np.arctan2(bg[:,1,0],bg[:,0,0])).as_matrix()
 from arm_depth_rays import generate
 natives={tag:load(OUT/'raw'/folder/f'{cid}.npz') for tag,folder in [('mb','mb_block_single'),('flip','mb_block_flip_restored')]}
 ray_rotations,ray_diagnostics=generate(base,h,phase,side,natives);initial={'ray_'+name:value for name,value in ray_rotations.items()}
 native_world=base['anatomical_keypoints'].astype(float)+base['translation'][:,None];coco_world=base['keypoints'].astype(float)+base['translation'][:,None];S=native_world[:,s]
 rest=lambda j,parent,R:np.einsum('tji,tj->ti',R,native_world[:,j]-native_world[:,parent])
 ve_all=rest(e,s,g[:,s]);vw_all=rest(w,e,g[:,e]);vh_all=rest(palm,w,g[:,w]);rest_error=max(float(abs(v-np.median(v,0)).max()) for v in [ve_all,vw_all,vh_all]);assert rest_error<.00001
 ve=np.median(ve_all,0);vw=np.median(vw_all,0);vh=np.median(vh_all,0);offe=np.einsum('tji,tj->ti',g[:,s],coco_world[:,ce]-native_world[:,e]);offw=np.einsum('tji,tj->ti',g[:,e],coco_world[:,cw]-native_world[:,w]);parent=T(g[:,13+side]);gs0=T(r[:,s]);ge0=T(r[:,e]);shoulder=T(S);VE=T(ve);VW=T(vw);OE=T(offe);OW=T(offw);principal=T(base['principal']);focal=float(base['focal'])
 init_r=np.stack([initial[tag] for name,tag,spline,depth in SETTINGS]);init_s=T(init_r[:,:,s]);init_e=T(init_r[:,:,e])
 basis=spline_basis(N,float(h['fps']),spacing=CONFIG['correction_spline_spacing_seconds']);padded=np.zeros((N,N),np.float32);padded[:,:basis.shape[1]]=basis;B=T(np.stack([padded if spline else np.eye(N,dtype=np.float32) for name,tag,spline,depth in SETTINGS]));controls=torch.nn.Parameter(torch.zeros((H,N,2,3),device='cuda:0'));opt=torch.optim.Adam([controls],lr=.015)
 xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];target=T(np.nan_to_num(xy[:,[ce,cw]]));sc=scores(h);weight=sc[:,[ce,cw]]**2;mask=np.arange(N)%5==2;weight[mask]=0;weight=T(weight);height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));gate,window=phase_window(h,phase);GATE=T(gate);confidence=T(np.minimum(sc[:,ce],sc[:,cw]));strength=T([x[3] for x in SETTINGS]);dt=T(np.diff(h['source_time']));links=(np.diff(h['frame_index'])==1)&(np.diff(h['source_time'])>0)&(np.diff(h['source_time'])<.1)
 if 'track_id' in h:links&=np.diff(h['track_id'])==0
 valid=T((links[1:]&links[:-1]).astype(float));mid=(dt[1:]+dt[:-1])/2;gv_arm=T(native_world[:,[e,w]]-S[:,None])
 def geometry():
  delta=torch.einsum('htk,hkjd->htjd',B,controls);rs=init_s@stable_rotation(delta[:,:,0]);re=init_e@stable_rotation(delta[:,:,1]);gs=parent[None]@rs;ge=gs@re;E=shoulder[None]+(gs@VE[...,None])[...,0];W=E+(ge@VW[...,None])[...,0];points=torch.stack([E+(gs@OE[None,...,None])[...,0],W+(ge@OW[None,...,None])[...,0]],2);arm=torch.stack([E-shoulder,W-shoulder],2)
  return rs,re,gs,ge,points,arm
 with torch.no_grad():target_depth=geometry()[-1][...,2]/leg
 def loss():
  rs,re,gs,ge,p,arm=geometry();uv=p[...,:2]/p[...,2:].clamp_min(.2)*focal+principal;err=(uv-target)/height;image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1);prior=(((arm[...,2]/leg-target_depth).square())*GATE[None,:,None]*confidence[None,:,None]).mean((1,2));outside=(((arm-gv_arm)/leg).square().mean(-1)*(1-GATE)[None,:,None]).mean((1,2))*strength.gt(0);rotation=((rs-gs0[None]).square().mean((1,2,3))+(re-ge0[None]).square().mean((1,2,3)))*.5;velocity=(arm[:,1:]-arm[:,:-1])/dt[None,:,None,None]/leg;acc=(velocity[:,1:]-velocity[:,:-1])/mid[None,:,None,None]/30;smooth=(acc.square().mean((2,3))*valid[None]*(1-.95*GATE[1:-1])[None]).mean(1)
  return image+strength*prior+.025*outside+.0005*rotation+.00005*smooth
 start=time.monotonic()
 for step in range(CONFIG['steps']):
  opt.param_groups[0]['lr']=.015*(.05+.95*(1-step/(CONFIG['steps']-1))**2);opt.zero_grad(set_to_none=True);v=loss()
  if not torch.isfinite(v).all():raise ValueError('Nonfinite V3 arm fit')
  v.sum().backward();torch.nn.utils.clip_grad_norm_([controls],10);opt.step()
 with torch.no_grad():rs,re,_,_,_,_=geometry()
 rs=rs.cpu().numpy();re=re.cpu().numpy();outputs={'current_GVHMR':base}
 output_rotations={name:value for name,value in [('raw_ray_'+key,value) for key,value in ray_rotations.items()]}
 for i,(name,tag,spline,depth) in enumerate(SETTINGS):
  rot=r.copy();rot[:,s]=rs[i];rot[:,e]=re[i];output_rotations[name]=rot
 for name,rot in output_rotations.items():
  GS=g[:,13+side].astype(float)@rot[:,s].astype(float);GE=GS@rot[:,e].astype(float);E=S+np.einsum('tij,j->ti',GS,ve);W=E+np.einsum('tij,j->ti',GE,vw);Hand=W+np.einsum('tij,j->ti',GE@r[:,w].astype(float),vh);a=base['anatomical_keypoints'].astype(float).copy();a[:,e]=E-base['translation'];a[:,w]=W-base['translation'];a[:,palm]=Hand-base['translation'];p=base['keypoints'].astype(float).copy();p[:,ce]=E+np.einsum('tij,tj->ti',GS,offe)-base['translation'];p[:,cw]=W+np.einsum('tij,tj->ti',GE,offw)-base['translation'];z={**base,'smpl_rotations':rot,'anatomical_keypoints':a,'keypoints':p,'foot_keypoints':base['foot_keypoints'].astype(float).copy()};assert np.array_equal(a[:,:16],base['anatomical_keypoints'][:,:16]);outputs[name]=z
 for name,z in outputs.items():
  _,uv=metrics(z,h,bg,z['translation'],base['contact_weight']);save_geometry(FOLDER/'fits'/name/f'{cid}.npz',z,z['translation'],uv,base['contact_weight'],base['normal'],float(base['pilot_support_offset']))
 report=dict(id=cid,ray_hypothesis_diagnostics=ray_diagnostics,config=CONFIG,source_hashes=source_hashes,variants=list(outputs),window=window,throwing_side=hand,hand_status='verified_metadata',source_observation_holdout_frames=np.flatnonzero(mask).tolist(),seconds=time.monotonic()-start,wrist_rotation_status='Preserved; MB17 cannot observe palm/wrist flexion',native_rest_vector_variation_max_model=rest_error,marker_scope='Frozen baseline surface points outside throwing elbow/wrist; local rigid arm offsets, not whole SMPL mesh',foot_preservation='Imposed structural scope, not a new contact-accuracy observation',human_review_complete=False,main_release_changed=False)
 save(dest,report);print('V3 ARM',cid,round(report['seconds'],1),'s',flush=True);return report

def main():
 deployment_guard()
 import torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.25,0);ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['tune','confirmation'],default='tune');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();FOLDER.mkdir(exist_ok=True);p=FOLDER/'protocol.json'
 if not p.exists():
  protocol=json.loads((OUT/'arm_cocking_v1/protocol.json').read_text());protocol.update(experiment_version=CONFIG['version'],design_data='V1/V2 tune6 only; confirmation outcomes not evaluated/visually inspected before V3 seal',prepared_v3_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(p,protocol)
 protocol=json.loads(p.read_text());ids=protocol[a.cohort][a.worker::a.workers]
 if a.cohort=='confirmation' and not (FOLDER/'selection_rule.json').exists():raise RuntimeError('Freeze V3 tuning rule before confirmation')
 save(FOLDER/f'configuration.worker{a.worker}.json',CONFIG)
 for cid in ids:fit(cid)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
