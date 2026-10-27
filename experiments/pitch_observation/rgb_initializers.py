"""Actual RGB-only HMR2.0a predictions before WHAM and before any scene fitting.

Uses installed official WHAM copy of HMR2.0a, with its exact checkpoint coverage.
Features are also cached for a separate GVHMR network pass. No v2 fitted pose enters.
"""
from common import *
import argparse

def main():
 import cv2,torch
 from scipy.ndimage import median_filter
 torch.set_num_threads(1);cv2.setNumThreads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 root=WORKSPACE/'external/WHAM';sys.path.insert(0,str(root));os.chdir(root)
 from lib.models.preproc.backbone.hmr2 import hmr2
 from lib.models.preproc.backbone.utils import process_image
 from scene_fit import SceneFitter
 device='cuda:0';weight=root/'checkpoints/hmr2a.ckpt'
 weight_hash=sha(weight)
 model=hmr2(str(weight)).to(device).eval()
 state=torch.load(weight,map_location='cpu')['state_dict'];model.load_state_dict({k:v for k,v in state.items() if k in model.state_dict()},strict=True)
 del state
 body=SceneFitter(root,device);records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for record in records:
  cid=record['id'];path=OUT/'raw/hmr2a'/f'{cid}.npz';start=time.monotonic()
  if path.exists():print('CACHE HMR2A',cid,flush=True);continue
  r,c,s,h,prior=init_inputs(cid);count=len(h['keypoints']);boxes=h['bbox'].copy();valid=np.isfinite(boxes).all(-1)
  for j in range(4):boxes[:,j]=np.interp(np.arange(count),np.flatnonzero(valid),boxes[valid,j])
  center=(boxes[:,:2]+boxes[:,2:])*.5;size=(boxes[:,2:]-boxes[:,:2]).max(1)*1.2
  # Per-frame boxes are observed input, not fitted camera/body parameters.
  origin=np.array([float(c['crop_x1']),float(c['crop_y1'])]);bbox=np.c_[center,size]
  cap=cv2.VideoCapture(str(DATA/c['crop_video']));patches=[]
  for i in range(count):
   ok,image=cap.read()
   if not ok:raise ValueError('Crop frame mismatch '+cid)
   crop,_=process_image(image[:,:,::-1],center[i]-origin,size[i]/200.,256,256)
   patches.append(crop)
  cap.release();rot=[];betas=[];cams=[];features=[]
  with torch.no_grad():
   for start_index in range(0,count,8):
    batch=torch.from_numpy(np.stack(patches[start_index:start_index+8])).to(device)
    # Same HMR2.0a encode/decode functions as upstream, sharing the backbone pass.
    feature=model(batch,encode=True)
    parameters,cam,_=model.smpl_head(feature)
    rot.append(torch.cat([parameters['global_orient'],parameters['body_pose']],1).cpu().numpy());betas.append(parameters['betas'].cpu().numpy());cams.append(cam.cpu().numpy());features.append(feature.cpu().numpy())
  rotations=np.concatenate(rot);beta=np.concatenate(betas);cam=np.concatenate(cams);feat=np.concatenate(features)
  xyz=[];native=[];feet=[]
  with torch.no_grad():
   for i in range(0,count,32):
    b,j,f=body.model(torch.from_numpy(rotations[i:i+32]).to(device),torch.from_numpy(beta[i:i+32]).to(device))
    xyz.append(b.cpu().numpy());native.append(j.cpu().numpy());feet.append(f.cpu().numpy())
  atomic_npz(path,smpl_rotations=rotations,smpl_betas=beta,keypoints=np.concatenate(xyz),anatomical_keypoints=np.concatenate(native),foot_keypoints=np.concatenate(feet),
   weak_camera=cam,features=feat,bbx_xys=bbox.astype(np.float32),source_time=h['source_time'],frame_index=h['frame_index'],fps=h['fps'],
   coordinate_system='Camera x right, y down, z away; hip-midpoint-relative; raw SMPL beta per frame',
   model='HMR2.0a pretrained raw RGB, no WHAM temporal decoder and no scene fitting',model_sha256=weight_hash)
  save(path.with_suffix('.json'),dict(id=cid,seconds=time.monotonic()-start,model_sha256=weight_hash,input_video=c['crop_video'],
   source_2d_sha256=sha(DATA/r['pose_2d_refined']),native_body='SMPL',initial_pose_source='raw RGB HMR2.0a',v2_used=False,checkpoint_loaded_strict=True,
   independent_scope='Independent estimator of WHAM temporal decoder; shared HMR2.0a image backbone with WHAM/GVHMR, not independent sensor evidence',max_gpu_memory_bytes=torch.cuda.max_memory_allocated()))
  print('RAW HMR2A',cid,count,round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
