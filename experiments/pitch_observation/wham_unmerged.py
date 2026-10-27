"""Retain WHAM flip hypotheses instead of immediately averaging rotations.

Uses the ORIGINAL Halpe input used by the archived WHAM raw for an exact TTA
control. The final refined Halpe files remain untouched and are still used for
subsequent fitting. This input distinction is explicit in provenance.
"""
from common import *
import argparse
from types import SimpleNamespace

def main():
 import cv2,torch
 torch.set_num_threads(1);cv2.setNumThreads(1)
 from wham_pipeline import load_wham,build_tracking,batched_features
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 args=SimpleNamespace(wham_root=WORKSPACE/'external/WHAM',device='cuda:0');cfg,network,extractor,model_hash,provenance=load_wham(args)
 from lib.data.datasets import CustomDataset
 from lib.utils.imutils import avg_preds,flip_pose
 from lib.utils.transforms import rotation_6d_to_matrix
 from scene_fit import SceneFitter
 body=SceneFitter(args.wham_root,'cuda:0');records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];dest=OUT/'raw/wham_single'/f'{cid}.npz'
  if dest.exists():print('CACHE WHAM SINGLE',cid,flush=True);continue
  start=time.monotonic();r,c,s,h,old=init_inputs(cid);original=load(DATA/r['pose_2d']);origin=np.array([float(c['crop_x1']),float(c['crop_y1'])]);video=str(DATA/c['crop_video']);N=len(h['keypoints']);cap=cv2.VideoCapture(video);W,H=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release();cache=OUT/'features_wham'/f'{cid}.pt';cache.parent.mkdir(exist_ok=True)
  if cache.exists():tracks=torch.load(cache,map_location='cpu')
  else:
   tracks=build_tracking(original,origin)
   with torch.no_grad():tracks=batched_features(extractor,video,tracks,8)
   torch.save(dict(tracks),cache)
  values={tag:dict(smpl_rotations=np.full((N,24,3,3),np.nan,np.float32),smpl_betas=np.full((N,10),np.nan,np.float32),keypoints=np.full((N,17,3),np.nan,np.float32),anatomical_keypoints=np.full((N,24,3),np.nan,np.float32),foot_keypoints=np.full((N,6,3),np.nan,np.float32)) for tag in ['wham_single','wham_flip_restored','wham_tta_reproduced']}
  with torch.no_grad():
   for track_id,track in tracks.items():
    ids=np.asarray(track['frame_id'],int);slam=np.zeros((len(ids),7));slam[:,3]=1;dataset=CustomDataset(cfg,{track_id:track},slam,W,H,float(h['fps']));preds=[]
    for flip in [False,True]:
     _,x,inits,features,mask,root,angvel,_,kwargs=dataset.load_data(0,flip);pred=network(x,inits,features,mask=mask,init_root=root,cam_angvel=angvel,return_y_up=True,**kwargs)
     preds.append((pred['pose'].squeeze(0).reshape(-1,24,6).clone(),pred['betas'].squeeze(0).clone()))
    raw_rot,raw_beta=preds[0];flipped_rot,flipped_beta=preds[1];merged_rot,merged_beta=avg_preds(raw_rot,raw_beta,flipped_rot,flipped_beta)
    for tag,pose,beta in [('wham_single',raw_rot,raw_beta),('wham_flip_restored',flip_pose(flipped_rot,representation='rotation_6d'),flipped_beta),('wham_tta_reproduced',merged_rot,merged_beta)]:
     rot=rotation_6d_to_matrix(pose);p,native,feet=body.model(rot,beta)
     for k,v in [('smpl_rotations',rot),('smpl_betas',beta),('keypoints',p),('anatomical_keypoints',native),('foot_keypoints',feet)]:values[tag][k][ids]=v.cpu().numpy()
  for tag,z in values.items():
   p=OUT/'raw'/tag/f'{cid}.npz';atomic_npz(p,**z,source_time=h['source_time'],frame_index=h['frame_index'],fps=h['fps'],coordinate_system='Camera x right y down z away; native hip-relative SMPL; uncalibrated model scale',model_sha256=model_hash)
   save(p.with_suffix('.json'),dict(id=cid,seconds_total=time.monotonic()-start,raw_2d_input= r['pose_2d'],raw_2d_sha256=sha(DATA/r['pose_2d']),final_2d_preserved=True,network=provenance,flip_hypothesis=tag,legacy_v2_pose_used=False))
  same=values['wham_tta_reproduced']['keypoints']-old['keypoints']*np.array([1,-1,-1]);difference=np.linalg.norm(values['wham_single']['keypoints']-values['wham_flip_restored']['keypoints'],axis=-1);save(OUT/'raw/wham_single'/f'{cid}_comparison.json',dict(id=cid,archived_merge_reproduction_max_model_m=float(np.nanmax(abs(same))),single_flip_disagreement_p50_model_m=float(np.nanmedian(difference)),single_flip_disagreement_p95_model_m=float(np.nanpercentile(difference,95))))
  print('UNMERGED WHAM',cid,round(time.monotonic()-start,1),'merge max',round(np.nanmax(abs(same)),6),'flip disagreement',round(np.nanmedian(difference),3),flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
