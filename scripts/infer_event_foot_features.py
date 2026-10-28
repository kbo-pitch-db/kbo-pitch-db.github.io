import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,csv,gzip,json,time
from pathlib import Path
import cv2,numpy as np,torch
R=Path(__file__).resolve().parents[1];O=R.parent/'research/event_detector_20261029/foot_features';O.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--ids');p.add_argument('--device',default='0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();device='cuda:'+a.device;sha='e1277af2ba9496fbadf7aec6eba56e8d882d1e35';m=torch.hub.load('facebookresearch/dinov2:'+sha,'dinov2_vits14',trust_repo=True,skip_validation=True).eval().to(device);torch.set_num_threads(2);cv2.setNumThreads(1)
C=json.load(open(R/'data/clips.json'))['clips'];wanted=set(json.load(open(a.ids))) if a.ids else None;C=[c for c in C if wanted is None or c['id'] in wanted][a.shard::a.shards];rows={r['dataset_clip_id']:r for r in csv.DictReader(open(R.parent/'dataset/clips.csv'))};mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
def crop(im,center,size):
 h,w=im.shape[:2];cx,cy=center;size=max(30,int(size));x1=int(cx-size/2);y1=int(cy-size/2);out=np.full((size,size,3),127,np.uint8);sx1=max(0,x1);sy1=max(0,y1);sx2=min(w,x1+size);sy2=min(h,y1+size)
 if sx2>sx1 and sy2>sy1:out[sy1-y1:sy2-y1,sx1-x1:sx2-x1]=im[sy1:sy2,sx1:sx2]
 return (cv2.resize(out,(224,224))[:,:,::-1].copy()/255.-mean)/std
for j,c in enumerate(C):
 cid=c['id'];dest=O/f'{cid}.npz'
 if dest.exists():continue
 st=time.time();p=json.load(open(R/'data'/c['pose']));d=json.load(gzip.open(R/'data/biomechanics/analysis'/f'{cid}.json.gz'));u=d['cadence']['uniqueImageCandidates'];xy=np.array([[v[:2] if v else [np.nan]*2 for v in f['keypoints']] for f in p['frames']]);row=rows[cid];origin=np.array([float(row['crop_x1']),float(row['crop_y1'])]);xy+=np.asarray(p.get('crop_box',origin.tolist())[:2])-origin;h=max(80,np.nanmedian(np.maximum(xy[:,15,1],xy[:,16,1])-xy[:,0,1]));cap=cv2.VideoCapture(str(R.parent/'dataset'/row['crop_video']));wants=set(u);patches=[];ids=[];outputs=[]
 for i in range(len(p['frames'])):
  ok,im=cap.read()
  if not ok:break
  if i not in wants:continue
  points=[]
  for foot in [0,1]:
   good=xy[i,[15+foot,20+foot,22+foot,24+foot]];center=np.nanmedian(good,axis=0)
   if not np.isfinite(center).all():center=np.array(im.shape[:2][::-1])/2
   center[1]+=.025*h;points.append(center);patches.append(crop(im,center,h*.30))
  points=np.array(points);center=points.mean(0);size=max(.40*h,np.max(np.ptp(points,axis=0))+.30*h);patches.append(crop(im,center,size));ids.append(i)
 cap.release()
 for begin in range(0,len(patches),96):
  x=torch.from_numpy(np.array(patches[begin:begin+96],np.float32).transpose(0,3,1,2)).to(device)
  with torch.inference_mode(),torch.autocast('cuda'):v=m(x).float();v=v/torch.linalg.norm(v,dim=1,keepdim=True)
  outputs.append(v.cpu().numpy())
 if outputs:np.savez_compressed(dest,frames=np.array(ids),features=np.concatenate(outputs).reshape(len(ids),3,-1).astype(np.float16),encoderRevision=sha)
 print(cid,len(ids),round(time.time()-st,2),flush=True)
