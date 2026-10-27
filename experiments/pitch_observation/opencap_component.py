"""Execute official OpenCap OptimizeExtrinsics, without inventing height/intrinsics.

This is a camera-component experiment, NOT the complete official pipeline.
The existing camera-only per-frame root offsets are nuisance input; they are
not calibrated world motion. Missing OpenPose joints have zero confidence.
"""
from common import *
from candidates import CC
import argparse

def main():
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(WORKSPACE/'external/opencap-monocular'));sys.path.insert(0,str(WORKSPACE/'external/pytorch3d'))
 from utils.optimization_formulations import OptimizeExtrinsics
 from slahmr.slahmr.geometry.camera import reproject
 mapping=[12,9,13,10,14,11,5,2,6,3,7,4]
 records=[];tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda:0')
 for rec in sample():
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);z=load(OUT/'candidates/WHAM_raw'/f'{cid}.npz');N=len(z['keypoints']);valid=np.isfinite(z['keypoints']).all((1,2));p=np.nan_to_num(z['keypoints']);tr=z['translation'];center=np.nanmedian(tr,0)
  key3d=np.zeros((N,25,3),np.float32);key2d=np.zeros((N,25,3),np.float32);key3d[:,mapping]=p+tr[:,None]-center
  key2d[:,mapping,:2]=np.nan_to_num(h['keypoints'][:,CC]);key2d[:,mapping,2]=scores(h)[:,CC]*valid[:,None]
  intr=dict(fx=float(z['focal']),fy=float(z['focal']),cx=float(z['principal'][0]),cy=float(z['principal'][1]));R=np.tile(np.eye(3),(N,1,1));t=np.tile(center,(N,1));start=time.monotonic()
  opt=OptimizeExtrinsics(tensor(R),tensor(t),tensor(key2d),tensor(key3d),intr,height=None,weights={'reprojection':1.,'quat_1':1.},iterations=8,printer=False)
  # Official optimize() returns self.beta even when height optimization is off.
  # Supply a non-optimized placeholder only for that output field; no shape loss.
  opt.beta=tensor(np.zeros(10));initial=float(opt.loss_reprojection().detach());result=opt.optimize();final=float(opt.loss_reprojection().detach())
  save(OUT/'official_opencap'/f'{cid}.json',dict(id=cid,seconds=time.monotonic()-start,initial_square_reprojection=initial,final_square_reprojection=final,R=result['R'].cpu().numpy(),t=result['t'].cpu().numpy(),
   official_class='OptimizeExtrinsics',official_commit='e2d0b146533c8387ebfb823faaec0356c6175644',full_official_pipeline=False,known_height_used=False,intrinsics='focal=4*image diagonal hypothesis; not calibrated',
   modifications=['COCO common12 -> OpenPose25 adapter, other scores zero','Official return-beta bug handled with nonoptimized zero output placeholder','Known-height loss omitted, root offsets supplied as nuisance'],input_semantics='Camera-only alignment from WHAM raw, no v2 pose'))
  records.append(dict(id=cid,initial=initial,final=final,seconds=time.monotonic()-start));print('OFFICIAL OPENCAP CAMERA COMPONENT',cid,round(initial),round(final),flush=True)
 save(OUT/'official_opencap/summary.json',records)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
