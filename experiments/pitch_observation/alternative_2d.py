"""Separate 384px RTMW whole-body experiment; final Halpe-26 stays immutable."""
from common import *
import argparse

def main():
 import cv2,torch
 from rtmlib import RTMPose
 torch.set_num_threads(1);cv2.setNumThreads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 weight=WORKSPACE/'models/rtmw_384/end2end.onnx';whash=sha(weight)
 model=RTMPose(str(weight),model_input_size=(288,384),backend='onnxruntime',device='cuda:0')
 records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];dest=OUT/'alternative_2d/rtmw384'/f'{cid}.npz'
  if dest.exists():print('CACHE RTMW',cid,flush=True);continue
  r,c,s,h,w=init_inputs(cid);video=OUT/'original_clips'/f'{cid}.mp4';origin=np.zeros(2)
  if not video.exists():video=DATA/c['crop_video'];origin=np.array([float(c['crop_x1']),float(c['crop_y1'])])
  cap=cv2.VideoCapture(str(video));pts=[];conf=[];start=time.monotonic();boxes=h['bbox'].copy();good=np.isfinite(boxes).all(1)
  for j in range(4):boxes[:,j]=np.interp(np.arange(len(boxes)),np.flatnonzero(good),boxes[good,j])
  try:
   for i in range(len(h['keypoints'])):
    ok,image=cap.read()
    if not ok:raise RuntimeError('video frame mismatch '+cid)
    xy,sc=model(image,bboxes=(boxes[i]-np.r_[origin,origin])[None]);assert xy.shape==(1,133,2);pts.append(xy[0]+origin);conf.append(sc[0])
  finally:cap.release()
  pts=np.array(pts);conf=np.array(conf);native=dict(keypoints=pts,scores=conf,source_time=h['source_time'],frame_index=h['frame_index'],model_sha256=whash)
  atomic_npz(dest,**native)
  # This adapter preserves Halpe-only head/neck/hip placeholders with zero score.
  alternate={k:v.copy() for k,v in h.items()};alternate['keypoints']=h['keypoints'].copy();alternate['scores']=np.zeros_like(h['scores']);alternate['missing']=np.ones_like(h['missing']);alternate['interpolated']=np.zeros_like(h['missing']);alternate['outlier']=np.zeros_like(h['missing'])
  for target,source in list(zip(range(17),range(17)))+list(zip(range(20,26),[17,20,18,21,19,22])):
   alternate['keypoints'][:,target]=pts[:,source];alternate['scores'][:,target]=conf[:,source];alternate['missing'][:,target]=~np.isfinite(pts[:,source]).all(-1)
  alternate['input_signature']=np.array(whash);atomic_npz(OUT/'alternative_2d/rtmw_halpe_adapter'/f'{cid}.npz',**alternate)
  save(dest.with_suffix('.json'),dict(id=cid,model='Official RTMW DW-X-L cocktail14 384x288 ONNX',weight_sha256=whash,seconds=time.monotonic()-start,frames=len(pts),input_video=str(video),original_2d_sha256=sha(DATA/r['pose_2d_refined']),final_2d_modified=False,semantics='Separate detector observations, not corrected ground truth',mapping='COCO17 identity; native feet 17,20,18,21,19,22 -> Halpe20..25; Halpe-only head/neck/hip score zero'))
  print('RTMW',cid,len(pts),round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
