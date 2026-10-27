"""Strong 3D inverse-kinematics adapter, separate from native MotionBERT output.

The complete native 3D candidate is fitted BEFORE any 2D/contact correction.
WHAM raw supplies only an IK solver seed and a fixed SMPL shape. The saved
initial candidate must actually differ from WHAM; conversion error is reported.
This is a custom adapter, never native MotionBERT mesh output or 3D GT.
"""
from common import *
from candidates import CC,MB,norm
from pitch_study_fit import StudyFitter,stable_rotation
import argparse

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--sources',nargs='+',default=['MotionBERT_native','mb_obp_adapter']);ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda:0');records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);beta=np.nanmedian(w['smpl_betas'],0);body.prepare_shape(beta);N=len(h['keypoints']);beta_t=tensor(np.repeat(beta[None],N,0));rot=np.concatenate([w['smpl_global_orient'],w['smpl_body_pose']],1);valid=np.isfinite(rot).all((1,2,3))
  for i in np.flatnonzero(~valid):
   good=np.flatnonzero(valid);rot[i]=rot[good[np.argmin(abs(good-i))]]
  r0=tensor(rot)
  with torch.no_grad():p0,a0,f0=body.model(r0,beta_t)
  leg=float(torch.linalg.vector_norm(a0[:,4]-a0[:,1],dim=-1).median()+torch.linalg.vector_norm(a0[:,7]-a0[:,4],dim=-1).median())
  for source in a.sources:
   tag={'MotionBERT_native':'mb_smpl','mb_obp_adapter':'mb_obp_smpl','mb_obp_single':'mb_obp_single_smpl','mb_obp_flip_restored':'mb_obp_flip_smpl','mb_single':'mb_single_smpl','mb_flip_restored':'mb_flip_smpl','mb_block_single':'mb_block_single_smpl','mb_block_flip_restored':'mb_block_flip_smpl'}[source]
   dest=OUT/'adapters'/tag/f'{cid}.npz'
   if dest.exists():continue
   path=PREVIOUS/'motionbert'/f'{cid}.npz' if source=='MotionBERT_native' else OUT/'raw'/source/f'{cid}.npz';z=load(path);target,_=norm(z['keypoints'][:,MB]);target=tensor(target*leg);delta=torch.nn.Parameter(torch.zeros((N,24,3),device='cuda:0'));opt=torch.optim.Adam([delta],lr=.04);start=time.monotonic()
   for i in range(240):
    opt.param_groups[0]['lr']=.04*(.05+.95*(1-i/239)**2);opt.zero_grad();angular=3.14*torch.tanh(delta/3.14);R=r0@stable_rotation(angular);p,anatomy,feet=body.model(R,beta_t);err=(p[:,CC]-target)/leg;loss=err.square().mean()+.00003*angular.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_([delta],10);opt.step()
   with torch.no_grad():R=r0@stable_rotation(3.14*torch.tanh(delta/3.14));p,anatomy,feet=body.model(R,beta_t)
   p=p.cpu().numpy();anatomy=anatomy.cpu().numpy();feet=feet.cpu().numpy();R=R.cpu().numpy();target=target.cpu().numpy();native_err=np.linalg.norm(p[:,CC]-target,axis=-1)/leg;initial_change=np.linalg.norm(p[:,CC]-p0.cpu().numpy()[:,CC],axis=-1)/leg
   atomic_npz(dest,smpl_rotations=R,smpl_betas=np.repeat(beta[None],N,0),keypoints=p,anatomical_keypoints=anatomy,foot_keypoints=feet,source_time=h['source_time'],frame_index=h['frame_index'],fps=h['fps'],coordinate_system='camera axes, hip relative; explicit MotionBERT-to-SMPL 3D IK approximation; sequence-fixed WHAM SMPL beta')
   save(dest.with_suffix('.json'),dict(id=cid,source_model=source,source_sha256=sha(path),seconds=time.monotonic()-start,source_to_adapter_p50_leg=float(np.median(native_err)),source_to_adapter_p95_leg=float(np.percentile(native_err,95)),raw_wham_to_new_initial_p50_leg=float(np.median(initial_change)),pose_seed='WHAM RAW only',target='Full independent native MotionBERT relative 3D candidate, leg-normalized once per sequence',loss='3D common12 coordinate loss + tiny rotation-change penalty; no image/contact/stride target',legacy_v2_used=False))
   print('STRONG MB INITIAL',cid,tag,round(np.median(initial_change),3),round(np.median(native_err),3),flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
