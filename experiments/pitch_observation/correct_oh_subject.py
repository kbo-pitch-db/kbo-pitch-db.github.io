"""Versioned recovery of one confirmed batter/pitcher selection error.

Original dataset/2D/3D files are read-only. This clip-specific foreground seed
is source-image evidence, not a general learned pitcher detector or 3D GT.
"""
from common import *
import argparse,copy,shutil

CID='4e09dd724a8c_c0011'
LEGACY=WORKSPACE/'research/pitch_observation_20261028'
FIX=OUT/'subject_correction_v1'
ORIGINAL_ROWS=rows
ORIGINAL_INPUTS=init_inputs

def infer_subject():
 import cv2,torch
 from ultralytics import YOLO
 from rtmlib import RTMPose
 from wham_pipeline import HALPE26,HALPE_EDGES
 from refine_motion import refine_trajectory
 torch.set_num_threads(1);cv2.setNumThreads(1);FIX.mkdir(parents=True,exist_ok=True)
 dest=FIX/'halpe26_refined.npz'
 if dest.exists():
  assert json.loads((FIX/'input_provenance.json').read_text())['code_sha256']==sha(__file__),'Use a new version for changed recovery code'
  print('CACHE corrected subject',flush=True);return
 r,c,s,old,w=ORIGINAL_INPUTS(CID);n=len(old['frame_index']);capture=cv2.VideoCapture(s['source_path']);capture.set(cv2.CAP_PROP_POS_FRAMES,int(c['source_frame_start']));frames=[]
 for i in range(n):
  ok,image=capture.read();assert ok;frames.append(image)
 capture.release();assert frames[0].shape[:2]==(1080,1920)
 video=FIX/'source_full_frames.mkv';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),float(old['fps']),(1920,1080));assert writer.isOpened()
 for image in frames:writer.write(image)
 writer.release()
 detector_path=WORKSPACE/'models/yolov8n.pt';detector=YOLO(str(detector_path));boxes=[];detections=[]
 # This shot's pitcher is the foreground person at the mound, lower than
 # batter/catcher/umpire throughout. Never substitute a nonmatching box.
 # Conditions are frozen from original-frame inspection, not 3D output.
 last=np.array([590.,580.,1000.,1020.]);supported=[]
 for i,image in enumerate(frames):
  result=detector.predict(image,classes=[0],conf=.20,imgsz=1280,device=0,verbose=False)[0]
  b=result.boxes.xyxy.cpu().numpy();conf=result.boxes.conf.cpu().numpy()
  pool=[j for j,p in enumerate(b) if 480<(p[0]+p[2])/2<1120 and p[3]>880 and p[1]>420 and p[3]-p[1]>180]
  detections.append(dict(frame=i,boxes=b,confidence=conf,eligible=pool))
  if not pool:boxes.append(np.full(4,np.nan));supported.append(False);continue
  center=lambda p:(p[:2]+p[2:])/2
  j=min(pool,key=lambda j:np.linalg.norm(center(b[j])-center(last)))
  last=b[j];boxes.append(last.copy());supported.append(True)
 boxes=np.asarray(boxes,np.float32);valid=np.array(supported);assert valid.mean()>.9,'Insufficient foreground-person evidence'
 for axis in range(4):boxes[:,axis]=np.interp(np.arange(n),np.flatnonzero(valid),boxes[valid,axis])
 pose_path=WORKSPACE/'models/rtmpose_halpe26_m.onnx';pose=RTMPose(str(pose_path),model_input_size=(192,256),backend='onnxruntime',device='cpu',to_openpose=False)
 points=[];confidence=[]
 for image,box in zip(frames,boxes):
  p,q=pose(image,bboxes=box[None]);assert p.shape==(1,26,2);points.append(p[0]);confidence.append(q[0])
 points=np.asarray(points,np.float32);confidence=np.asarray(confidence,np.float32);missing=~np.isfinite(points).all(-1)|(confidence<.2)
 values=dict(bbox=boxes,frame_index=old['frame_index'],pts=old['pts'],source_time=old['source_time'],track_id=np.zeros(n,np.int64),keypoints=points,scores=confidence,missing=missing,fps=old['fps'],joints=np.array(HALPE26),layout='halpe26',edges=np.asarray(HALPE_EDGES),model='RTMPose-m Halpe-26',model_sha256=sha(pose_path),coordinate_system='original source pixels',method='subject-correction-v1-same-halpe26',input_signature=sha(video)+sha(pose_path),source_pose_sha256=sha(DATA/r['pose_2d_refined']),detector_supported=valid)
 atomic_npz(FIX/'halpe26_raw.npz',**values)
 torso=np.linalg.norm((points[:,5]+points[:,6]-points[:,11]-points[:,12])/2,axis=-1);threshold=max(20,.35*np.nanmedian(torso))
 clean,filled,outlier=refine_trajectory(points,confidence,missing,old['source_time'],values['track_id'],old['frame_index'],.45,3,threshold)
 atomic_npz(dest,**dict(values,keypoints=clean,missing=~np.isfinite(clean).all(-1),interpolated=filled,outlier=outlier))
 save(FIX/'detections.json',detections)
 save(FIX/'input_provenance.json',dict(id=CID,code_sha256=sha(__file__),cause='Original track and Halpe26 bbox follow the batter, not pitcher',original_final_2d_sha256=sha(DATA/r['pose_2d_refined']),original_public_2d_sha256=sha(CODE/'data/assets'/CID/'pose_halpe26_refined.json'),corrected_2d_sha256=sha(dest),detector_weight_sha256=sha(detector_path),pose_weight_sha256=sha(pose_path),detector_supported_frames=int(valid.sum()),frames=n,frame_index_preserved=True,timestamp_preserved=True,source_video=Path(s['source_path']).name,source_video_sha256=sha(s['source_path']),source_start_frame=int(c['source_frame_start']),source_crop=[0,0,1920,1080],subject_seed='Clip-specific foreground mound region and temporal center continuity, source-inspected',original_files_overwritten=False,manual_human_review_complete=False,clip_starts_after_stride=True,complete_delivery=False,leg_lift='outside_observed_interval',stride_validated=False))
 diagnostic(FIX/'input_overlay.jpg',frames,load(dest),boxes)
 print('CORRECT SUBJECT',int(valid.sum()),'/',n,flush=True)

def diagnostic(path,frames,h,boxes):
 import cv2
 images=[]
 for i in [0,20,40,60,74]:
  image=frames[i].copy()
  for p in h['keypoints'][i]:
   if np.isfinite(p).all():cv2.circle(image,(int(p[0]),int(p[1])),5,(0,255,0),-1)
  b=boxes[i];cv2.rectangle(image,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,0,255),3)
  image=cv2.resize(image,(768,432));cv2.putText(image,f'corrected pitcher | F{i}',(5,25),0,.7,(0,255,255),2);images.append(image)
 cv2.imwrite(str(path),np.concatenate(images,0))

def corrected_inputs(cid):
 assert cid==CID
 r,c,s,h,w=ORIGINAL_INPUTS(cid);r=copy.deepcopy(r);c=copy.deepcopy(c)
 h=load(FIX/'halpe26_refined.npz');r['pose_2d']=str(FIX/'halpe26_raw.npz');r['pose_2d_refined']=str(FIX/'halpe26_refined.npz')
 c.update(crop_video=str(FIX/'source_full_frames.mkv'),crop_x1='0',crop_y1='0',crop_x2='1920',crop_y2='1080')
 feature=OUT/'raw/hmr2a'/f'{cid}.npz'
 # The original WHAM result belongs to the batter. None of its pose or shape
 # may initialize this recovery. HMR2a RGB supplies the fixed SMPL shape.
 beta=np.nanmedian(load(feature)['smpl_betas'],axis=0) if feature.exists() else np.zeros(10,np.float32)
 w=dict(smpl_betas=np.repeat(beta[None],len(h['frame_index']),0))
 return r,c,s,h,w

def install_inputs():
 import common
 common.init_inputs=corrected_inputs
 metadata=OUT/'metadata';metadata.mkdir(exist_ok=True);shutil.copy2(LEGACY/'metadata/player_hands.json',metadata/'player_hands.json')

def run(stage):
 import torch
 torch.set_num_threads(1);FIX.mkdir(parents=True,exist_ok=True)
 if stage=='observe':infer_subject();return
 install_inputs()
 import all813_release
 rec=next(r for r in all813_release.population() if r['id']==CID)
 if stage in ['features','network']:
  all813_release.run_upstream(stage,[rec],0)
 elif stage in ['base','contact']:
  import all813_fit
  all813_fit.ROOT.mkdir(exist_ok=True)
  # Reuse only the person/graphics-masked full-frame background transform.
  # Input crop is now the full frame, so its warp equals the full warp.
  dest=all813_fit.ROOT/'background'/f'{CID}.npz';dest.parent.mkdir(exist_ok=True)
  if not dest.exists():
   bg=load(LEGACY/'all813_gvhmr_v1/background'/f'{CID}.npz');bg['crop_transforms']=bg['full_transforms'].copy();atomic_npz(dest,**bg)
   save(dest.with_suffix('.json'),dict(source='Frozen person/graphics-masked full-frame image stabilization',full_frame_only=True,metric_camera=False))
  if stage=='base':all813_fit.base([rec])
  else:all813_fit.contact([rec],graph=True)
 elif stage=='select':
  import all813_select
  print(json.dumps(clean(all813_select.select(CID)),ensure_ascii=False),flush=True)
 else:raise ValueError(stage)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['observe','features','network','base','contact','select']);args=ap.parse_args();run(args.stage)
