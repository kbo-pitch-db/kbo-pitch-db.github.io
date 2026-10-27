"""Critical real-data/geometry checks. Not a substitute for 3D validation."""
from common import *
import ast

def main():
 import torch
 from scipy.spatial.transform import Rotation
 from pitch_study_fit import StudyFitter,stable_rotation
 from scene_fit import SceneFitter
 from events import infer
 from candidates import CC,MB
 torch.set_num_threads(1);checks={};rng=np.random.default_rng(20261028)
 rv=rng.normal(size=(20,24,3)).astype(np.float32)*.4;ours=stable_rotation(torch.tensor(rv)).numpy();ref=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(ours.shape);error=float(abs(ours-ref).max());assert error<5e-7;checks['so3_exponential_max_error']=error
 # Run the official mapping function in isolation to avoid Python lib-package
 # collisions between WHAM and MotionBERT, using independent named-joint IDs.
 source=(WORKSPACE/'external/MotionBERT/lib/data/dataset_wild.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='halpe2h36m');namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<official mapping>','exec'),namespace);halpe=np.zeros((2,26,3),np.float32);halpe[...,:2]=np.arange(26)[None,:,None];halpe[...,2]=1;mapped=namespace['halpe2h36m'](halpe);assert np.array_equal(mapped[:,MB,:2],halpe[:,CC,:2]);checks['native_halpe_h36m_common12_left_right_mapping']='passed official converter'
 cid=sample()[0]['id'];r,c,s,h,w=init_inputs(cid);beta=np.nanmedian(w['smpl_betas'],0);R=np.concatenate([w['smpl_global_orient'],w['smpl_body_pose']],1);R=R[np.isfinite(R).all((1,2,3))][[0,5,10]];fitter=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);fitter.prepare_shape(beta);rt=torch.tensor(R,device='cuda:0');bt=torch.tensor(np.repeat(beta[None],len(R),0),device='cuda:0')
 with torch.no_grad():sparse=fitter.model(rt,bt);full=SceneFitter.model(fitter,rt,bt)
 dif=[float((a-b).abs().max()) for a,b in zip(sparse,full)];assert max(dif)<2e-6;checks['sparse_full_smpl_max_differences']=dif
 for key in ['fa1febd5056f_c0015','e72b3a782760_c0022']:
  hh=init_inputs(key)[3];e=infer(hh,load(PREVIOUS/'background'/f'{key}.npz')['full_transforms']);assert e['lift_interval'] is None or e['lift_interval'][1]<e['arm_fast_interval'][0];assert e['release']=='unknown';checks[key+'_no_followthrough_lift']=e
 hh=init_inputs('e72b3a782760_c0038')[3];e=infer(hh,load(PREVIOUS/'background/e72b3a782760_c0038.npz')['full_transforms']);assert e['plant_interval'][0]>140;checks['airborne_pause_not_selected_as_plant']=e['plant_interval']
 errors=[]
 for rec in sample('all',False):
  r,c,s,hh,w=init_inputs(rec['id']);time_error=np.max(abs((hh['source_time']-hh['source_time'][0])-(hh['frame_index']-hh['frame_index'][0])/float(hh['fps'])));assert time_error<1e-7;bg=load(PREVIOUS/'background'/f'{rec["id"]}.npz')['full_transforms'];uv=hh['keypoints'];hom=np.concatenate([np.nan_to_num(uv),np.ones((*uv.shape[:2],1))],-1);returned=np.einsum('tij,tkj->tki',np.linalg.inv(bg),np.einsum('tij,tkj->tki',bg,hom));errors.append(float(abs(returned-hom).max()))
 assert max(errors)<1e-3;checks['full_frame_stabilization_inverse_roundtrip_max_px']=max(errors);checks['all48_timestamp_frame_mapping']='passed, encoded-source time only'
 save(OUT/'validation.json',dict(status='passed',checks=checks,semantics='Software/representation/source-alignment checks only; not KBO 3D accuracy or manual review.'));print('CRITICAL VALIDATION PASSED',checks['sparse_full_smpl_max_differences'],flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
