"""Source-based pitch-direction cones, NOT metric camera calibration.

The far player-support region is a loose home-direction observation, not a
surveyed home-plate point. Mound support and that region constrain DIRECTION
only under several focal/elevation/principal assumptions. Never uses 18.44m,
measured height, planar metric SLAM, or a population stride target.
Agent-confirmed camera view is stored separately from automatic proposals.
"""
from common import *
import argparse

# Reviewed from original full-frame tune montages, not predicted 3D. No human GT.
TUNE_HANDS={'e1cae96b4d83_c0027':'right','d30b22cfc0b0_c0036':'left','3fdcd2dbaf58_c0014':'right','e72b3a782760_c0038':'right','fa1febd5056f_c0015':'left','4e09dd724a8c_c0031':'right','d644482ee06d_c0015':'right','0b70417b94ab_c0013':'unknown','e56945a0fd80_c0028':'left','e72b3a782760_c0022':'left','99f1a527c09b_c0013':'right','e1cae96b4d83_c0013':'right'}

def cone(mound,home_region,width,height):
 axes=[];diagonal=np.hypot(width,height)
 for factor in [1.,4.,12.]:
  for elevation in [0.,4.,12.]:
   theta=np.deg2rad(elevation);normal=np.array([0.,-np.cos(theta),-np.sin(theta)])
   for offx,offy in [(0,0),(-.05,0),(.05,0),(0,-.05),(0,.05)]:
    principal=np.array([width*(.5+offx),height*(.5+offy)]);f=factor*diagonal;mp=np.r_[(mound-principal)/f,1.];den=normal@mp
    if den>=-.0001:continue
    m=-mp/den
    x1,y1,x2,y2=home_region
    for point in [[x1,y1],[x2,y1],[x1,y2],[x2,y2],[(x1+x2)/2,(y1+y2)/2]]:
     hp=np.r_[(np.array(point)-principal)/f,1.];d=normal@hp
     if d>=-.0001:continue
     q=-hp/d;v=q-m
     if v[2]<=0:continue
     for perturb_y in [-.03,0,.03]:
      direction=v/np.linalg.norm(v);direction[1]+=perturb_y;direction/=np.linalg.norm(direction);axes.append(direction)
 return np.array(axes,np.float32)

def main():
 import cv2,torch
 from ultralytics import YOLO
 from ultralytics.utils.ops import non_max_suppression
 torch.set_num_threads(1);cv2.setNumThreads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();model=YOLO(str(WORKSPACE/'models/yolov8n.pt')).model.cuda(0).eval();records=[]
 for rec in sample(a.split,not a.full_sample):
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);folder=OUT/'camera_observations'/cid;folder.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(OUT/'original_clips'/f'{cid}.mp4'));ok,frame=cap.read();cap.release()
  if not ok:raise RuntimeError('Cannot read original '+cid)
  H,W=frame.shape[:2];ratio=640/W;small=cv2.resize(frame,(640,int(round(H*ratio))));pad=(32-small.shape[0]%32)%32;small=cv2.copyMakeBorder(small,0,pad,0,0,cv2.BORDER_CONSTANT,value=(114,114,114));x=torch.from_numpy(np.ascontiguousarray(small[:,:,::-1].transpose(2,0,1)))[None].cuda(0).float()/255
  with torch.no_grad():pred=non_max_suppression(model(x),.18,.55,classes=[0])[0].cpu().numpy()
  boxes=pred[:,:4]/ratio;scores_person=pred[:,4];pitcher=h['bbox'][0];pitcher=h['bbox'][np.flatnonzero(np.isfinite(h['bbox']).all(1))[0]] if not np.isfinite(pitcher).all() else pitcher
  ankle=h['keypoints'][:min(5,len(h['keypoints'])),15:17];valid=np.isfinite(ankle).all(-1);foot=[]
  for i in range(len(ankle)):
   if valid[i].any():foot.append(ankle[i,valid[i]][np.argmax(ankle[i,valid[i],1])])
  mound=np.median(foot,axis=0) if foot else np.array([(pitcher[0]+pitcher[2])/2,pitcher[3]])
  candidates=[]
  for i,b in enumerate(boxes):
   lo=np.maximum(b[:2],pitcher[:2]);hi=np.minimum(b[2:],pitcher[2:]);inter=np.prod(np.maximum(hi-lo,0));area=np.prod(b[2:]-b[:2]);p_area=np.prod(pitcher[2:]-pitcher[:2]);iou=inter/max(area+p_area-inter,1)
   center=(b[:2]+b[2:])*.5
   if iou<.2 and .30*H<b[3]<mound[1]-.025*H and .06*H<b[3]-b[1]<.5*H and abs(center[0]-mound[0])<.5*W:candidates.append(i)
  # Select the best-supported nearby player cluster; retain a wide region.
  best=[]
  for i in candidates:
   center=(boxes[i,:2]+boxes[i,2:])*.5;group=[j for j in candidates if np.linalg.norm(((boxes[j,:2]+boxes[j,2:])*.5-center)/[W,H])<.20]
   if sum(scores_person[j] for j in group)>sum(scores_person[j] for j in best):best=group
  home_region=None;directions=np.empty((0,3));status='unknown'
  if len(best)>=2:
   points=np.c_[(boxes[best,0]+boxes[best,2])*.5,boxes[best,3]];home_region=[float(points[:,0].min()-20),float(points[:,1].min()-20),float(points[:,0].max()+20),float(points[:,1].max()+20)];directions=cone(mound,home_region,W,H);status='center_field_direction_proposal'
  confirmed=cid in TUNE_HANDS;hand=TUNE_HANDS.get(cid,'unknown');hand_evidence='Agent source-frame observation' if hand!='unknown' else 'unknown'
  metadata_path=OUT/'metadata/player_hands.json'
  if metadata_path.exists():
   m=json.loads(metadata_path.read_text()).get(rec['player'].split(':')[-1],{})
   if m.get('identity_matched') and m.get('throwing_hand')!='unknown':hand=m['throwing_hand'];hand_evidence=m['url']
  record=dict(id=cid,source_frame=0,view='center_field_offset' if confirmed else 'undetermined',view_hypotheses=['center_field_offset'] if confirmed else ['center_field_offset','side','other'],view_evidence='Codex inspection of original full-frame montage' if confirmed else 'Automatic player-support cluster only; requires source inspection',agent_review=confirmed,human_review=False,
   throwing_hand=hand,throwing_hand_evidence=hand_evidence,mound_support_pixel=mound,home_direction_region=home_region,home_region_person_indices=best,people_boxes=boxes,proposal_status=status,axis_hypotheses=len(directions),semantics='Directional perspective/plane hypotheses, not calibrated extrinsics, physical distance, or independently validated ground plane')
  save(folder/'observation.json',record);atomic_npz(folder/'axes.npz',pitch_axes=directions)
  draw=frame.copy()
  for i,b in enumerate(boxes):
   x1,y1,x2,y2=b.astype(int);color=(0,255,0) if i in best else (160,160,160);cv2.rectangle(draw,(x1,y1),(x2,y2),color,2);cv2.putText(draw,str(i),(x1,y1+20),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
  if home_region:
   x1,y1,x2,y2=np.round(home_region).astype(int);cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,255),3);cv2.arrowedLine(draw,tuple(np.round(mound).astype(int)),((x1+x2)//2,(y1+y2)//2),(0,255,255),3)
  cv2.imwrite(str(folder/'source.jpg'),draw);records.append(record);print('CAMERA DIRECTION',cid,len(best),len(directions),hand,flush=True)
 save(OUT/f'camera_observations_{a.split}.json',records)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
