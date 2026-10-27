"""Explicit RGB-body / learned temporal-arm rotation candidate, never averaging.

NLF torso/legs/shape + independently adapted MotionBERT arm world rotations.
Shoulder local rotation is changed relative to NLF collar so the target arm
orientation is preserved under its new parent; downstream local rotations are
copied explicitly. This is a custom initializer, not an official model output.
"""
from common import *
from scipy.spatial.transform import Rotation
import argparse
PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

def global_rotations(r):
 out=r.copy()
 for j in range(1,24):out[:,j]=out[:,PARENTS[j]]@r[:,j]
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample(a.split,a.split=='tune')[a.offset:];records=records[:a.limit] if a.limit else records
 for rec in records:
  cid=rec['id'];n=load(OUT/'adapters/nlf_rgb'/f'{cid}.npz');m=load(OUT/'adapters/mb_block_single_smpl'/f'{cid}.npz');assert np.array_equal(n['frame_index'],m['frame_index']);r=n['smpl_rotations'].copy();ng=global_rotations(r);mg=global_rotations(m['smpl_rotations']);changes=[]
  for j in [16,17]:r[:,j]=ng[:,PARENTS[j]].transpose(0,2,1)@mg[:,j]
  r[:,18:24]=m['smpl_rotations'][:,18:24];hg=global_rotations(r);assert np.max(abs(hg[:,16:24]-mg[:,16:24]))<2e-5
  angle=np.rad2deg(Rotation.from_matrix((ng[:,16:24].transpose(0,1,3,2)@hg[:,16:24]).reshape(-1,3,3)).magnitude());atomic_npz(OUT/'adapters/nlf_mb_hybrid'/f'{cid}.npz',smpl_rotations=r,smpl_betas=n['smpl_betas'],source_time=n['source_time'],frame_index=n['frame_index']);save(OUT/'adapters/nlf_mb_hybrid'/f'{cid}.json',dict(id=cid,body='NLF original RGB torso/legs/global root rotation + fixed native NLF shape',arms='Public-3D-adapted MotionBERT single native depth hypothesis, 3D IK adapter, explicit global shoulder orientation transplantation',mean_arm_world_rotation_change_degrees=float(np.mean(angle)),max_arm_global_rotation_adapter_error=float(np.max(abs(hg[:,16:24]-mg[:,16:24]))),coordinate_averaging=False,wham_pose_not_used_as_body_initializer=True))
  print('HYBRID INITIAL',cid,float(np.mean(angle)),flush=True)
if __name__=='__main__':main()
