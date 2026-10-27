"""Sequence-fixed NLF body, preserving actual RGB rotations and camera candidate.

No WHAM/v2 pose/shape, no framewise recentering of saved scene, no inferred
floor or stride. Original nonparametric and own-SMPL outputs remain archived.
"""
from common import *
from pitch_study_fit import StudyFitter
import argparse

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample(a.split,a.split=='tune');records=records[a.offset:];records=records[:a.limit] if a.limit else records;body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);reg=body.body.J_regressor_wham[[11,12]].mean(0).cpu().numpy();keep=np.flatnonzero(abs(reg)>0)
 for rec in records:
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);z=load(OUT/'adapters/nlf_rgb'/f'{cid}.npz');beta=np.median(z['smpl_betas'],0);rot=z['smpl_rotations'];body.prepare_shape(beta);p=[];aa=[];feet=[]
  with torch.no_grad():
   for k in range(0,len(rot),64):
    out=body.model(torch.tensor(rot[k:k+64],device='cuda:0'),torch.tensor(np.repeat(beta[None],len(rot[k:k+64]),0),device='cuda:0'))
    for dest,v in zip([p,aa,feet],out):dest.append(v.cpu().numpy())
  p,aa,feet=map(np.concatenate,[p,aa,feet]);roots=[]
  for i in range(len(rot)):
   raw=load(OUT/'raw/nlf_frames'/cid/f'{i:05d}_f4.npz');roots.append(reg[keep]@(raw['vertices3d'][0,keep]/1000))
  tr=np.array(roots);f=float(raw['focal']);principal=raw['principal'];points=np.concatenate([p,feet],1)+tr[:,None];uv=points[...,:2]/points[...,2:]*f+principal;dest=OUT/'nlf_native_fixed'/f'{cid}.npz';atomic_npz(dest,keypoints=p,anatomical_keypoints=aa,foot_keypoints=feet,translation=tr,projected=uv,focal=f,principal=principal,smpl_rotations=rot,smpl_betas=beta,background_transforms=np.tile(np.eye(3),(len(p),1,1)),source_time=h['source_time'],frame_index=h['frame_index'])
  save(dest.with_suffix('.json'),dict(id=cid,model='Official independent NLF RGB rotations; sequence median native NLF SMPL betas',shape_fixed=True,wham_shape_or_pose_used=False,rotations_changed_by_fitting=False,camera='Uncalibrated full-frame f4 hypothesis; original inferred image-relative translation, not world motion',inference_root='raw/nlf_frames',native_outputs_untouched=True));print('NLF FIXED NATIVE',cid,flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
