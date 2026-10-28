import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,csv,gzip,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from ultralytics import YOLO
R=Path(__file__).resolve().parents[1];O=R.parent/'research/event_detector_20261029';O.mkdir(exist_ok=True);(O/'ball').mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--ids');p.add_argument('--device',default='0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
C=json.load(open(R/'data/clips.json'))['clips'];rows={r['dataset_clip_id']:r for r in csv.DictReader(open(R.parent/'dataset/clips.csv'))};ids=set(json.load(open(a.ids))) if a.ids else None;C=[c for c in C if ids is None or c['id'] in ids][a.shard::a.shards]
model_path=R.parent/'models/baseballcv_ball_tracking_v4.pt';model=YOLO(str(model_path));sha=hashlib.sha256(model_path.read_bytes()).hexdigest();cv2.setNumThreads(1)
for at,c in enumerate(C):
 cid=c['id'];out=O/'ball'/f'{cid}.json'
 if out.exists():continue
 start=time.time();pose=json.load(open(R/'data'/c['pose']));analysis=json.load(gzip.open(R/'data/biomechanics/analysis'/f'{cid}.json.gz'));n=len(pose['frames']);xy=np.array([[v[:2] if v else [np.nan,np.nan] for v in f['keypoints']] for f in pose['frames']]);score=np.array([[v[2] if v else 0 for v in f['keypoints']] for f in pose['frames']]);u=np.array(analysis['cadence']['uniqueImageCandidates']);ts=np.array([f['time'] for f in pose['frames']]);arm=10 if analysis['throwingHand']=='R' else 9;shoulder=arm-4;h=max(80,np.nanmedian(np.maximum(xy[:,15,1],xy[:,16,1])-xy[:,0,1]));w=xy[u,arm]-xy[u,shoulder]
 for d in range(2):
  g=np.isfinite(w[:,d]);w[:,d]=np.interp(np.arange(len(u)),np.flatnonzero(g),w[g,d]) if g.any() else 0
 speed=np.linalg.norm(np.gradient(gaussian_filter1d(w,.65,axis=0),ts[u],axis=0),axis=1)/h;speed[score[u,arm]<.4]=0
 peaks=find_peaks(speed,distance=4)[0];peaks=sorted(peaks,key=lambda i:speed[i],reverse=True)[:4];eligible=np.zeros(len(u),bool)
 for q in peaks:eligible|=(ts[u]>=ts[u[q]]-.50)&(ts[u]<=ts[u[q]]+.35)
 selected=u[eligible];wanted=set(selected.tolist());cap=cv2.VideoCapture(str(R.parent/'dataset'/rows[cid]['crop_video']));images=[];frames=[]
 for f in range(n):
  ok,im=cap.read()
  if not ok:break
  if f in wanted:images.append(im);frames.append(f)
 cap.release();detections={}
 for offset in range(0,len(images),12):
  pred=model.predict(images[offset:offset+12],device=a.device,imgsz=640,conf=.08,iou=.4,verbose=False,half=True)
  for i,res in zip(frames[offset:offset+12],pred):detections[str(i)]=res.boxes.data.cpu().numpy().round(4).tolist()
 row=dict(clipId=cid,modelSha256=sha,model='BaseballCV ball_trackingv4 YOLO11x',fps=pose['fps'],height=float(h),cropShape=list(images[0].shape[:2]) if images else None,frames=frames,detections=detections,seconds=round(time.time()-start,2));out.write_text(json.dumps(row,separators=(',',':')));print(cid,len(images),'frames',row['seconds'],'seconds',flush=True)
