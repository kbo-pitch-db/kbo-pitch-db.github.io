"""Comparable raw alternatives; camera-only alignment never changes articulation."""
from common import *
import argparse
CC=[11,12,13,14,15,16,5,6,7,8,9,10]
MB=[4,1,5,2,6,3,11,14,12,15,13,16]
EDGES=[(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]

def norm(p):
 p=p-p[:,:2].mean(1,keepdims=True)
 leg=np.nanmedian(sum(np.linalg.norm(p[:,a]-p[:,b],axis=-1) for a,b in [(0,2),(2,4),(1,3),(3,5)])*.5)
 return p/max(float(leg),1e-5),float(leg)

def raw_candidates(cid):
 r,c,s,h,w=init_inputs(cid);result={}
 result['WHAM_raw']={'points':w['keypoints'][:,CC]*[1,-1,-1],'rotations':np.concatenate([w['smpl_global_orient'],w['smpl_body_pose']],1),'beta':np.nanmedian(w['smpl_betas'],0)}
 for tag in ['hmr2a','gvhmr_f1','gvhmr_f4','gvhmr_f12']:
  path=OUT/('raw' if tag=='hmr2a' else 'adapters')/tag/f'{cid}.npz'
  if not path.exists():continue
  z=load(path);result[tag]={'points':z['keypoints'][:,CC],'rotations':z['smpl_rotations'],'beta':np.nanmedian(z['smpl_betas'],0)}
 z=load(PREVIOUS/'motionbert'/f'{cid}.npz');result['MotionBERT_native']={'points':z['keypoints'][:,MB]}
 for tag in ['mb_obp_adapter','mb_obp_single','mb_obp_flip_restored','mb_block_single','mb_block_flip_restored','mb_single','mb_flip_restored']:
  path=OUT/'raw'/tag/f'{cid}.npz'
  if path.exists():result[tag]={'points':load(path)['keypoints'][:,MB]}
 final=rows('motion_wham_final_scene.csv')[cid];z=load(DATA/final['pose_3d_scene']);result['current']={'points':(z['keypoints']@z['scene_rotation'])[:,CC],
  'rotations':np.concatenate([z['smpl_global_orient'],z['smpl_body_pose']],1),'beta':z['smpl_betas']}
 return result,h,c,s

def align_camera(p,h,s,factor=4.):
 """Fit per-frame camera translation ONLY, to compare unchanged native poses.

 Focal is a hypothesis. No calibrated trajectory follows from this alignment.
 """
 import torch
 from scene_geometry import initial_camera
 tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
 xy=np.nan_to_num(h['keypoints'][:,CC]);score=scores(h)[:,CC];valid=np.isfinite(p).all((1,2));p=p.copy()
 for j in range(12):
  for k in range(3):
   good=np.isfinite(p[:,j,k]);p[:,j,k]=np.interp(np.arange(len(p)),np.flatnonzero(good),p[good,j,k])
 # Initializer utility expects COCO17; prepend five zeros so its face mask
 # does not accidentally downweight the hip/leg common-joint subset.
 padded=np.zeros((len(p),17,3),np.float32);padded[:,5:]=p
 x=np.zeros((len(p),17,2),np.float32);x[:,5:]=xy;q=np.zeros((len(p),17),np.float32);q[:,5:]=score
 focal=np.hypot(float(s['width']),float(s['height']))*factor;principal=np.array([float(s['width'])/2,float(s['height'])/2])
 _,_,t0=initial_camera(padded,x,q,focal,principal)
 pt=tensor(p);target=tensor(xy);weight=tensor(score**2*valid[:,None]);principal_t=tensor(principal)
 trans=torch.nn.Parameter(tensor(t0));opt=torch.optim.Adam([trans],lr=.04)
 height=max(float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])),100.)
 for i in range(60):
  opt.zero_grad();v=pt+trans[:,None];pred=v[...,:2]/v[...,2:].clamp_min(.5)*focal+principal_t
  d=(pred-target)/height;loss=((torch.sqrt(d.square().sum(-1)+.005**2)-.005)*weight).sum()/weight.sum().clamp_min(1);loss.backward();opt.step()
 with torch.no_grad():
  v=pt+trans[:,None];pred=v[...,:2]/v[...,2:]*focal+principal_t
 return pred.cpu().numpy(),trans.detach().cpu().numpy(),focal,principal

def main():
 import torch
 torch.set_num_threads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();records=[]
 for rec in sample(a.split,not a.full_sample):
  cid=rec['id'];cand,h,c,s=raw_candidates(cid)
  for tag,item in cand.items():
   p,leg=norm(item['points']);dest=OUT/'candidates'/tag/f'{cid}.npz'
   if dest.exists():z=load(dest);projection=z['projected'];trans=z['translation'];focal=float(z['focal']);principal=z['principal']
   else:
    projection,trans,focal,principal=align_camera(p,h,s)
    atomic_npz(dest,keypoints=p,projected=projection,translation=trans,focal=focal,principal=principal,source_time=h['source_time'],model_leg=leg,source_model=tag,coordinate_system='hip-relative, camera x right y down z away; one sequence-wide model-leg normalization; camera-only projection')
   error=np.linalg.norm(projection-h['keypoints'][:,CC],axis=-1);valid=(scores(h)[:,CC]>.5)&np.isfinite(error)
   m=dict(id=cid,split=rec['split'],candidate=tag,frames=len(p),native_leg=leg,valid_ratio=float(np.isfinite(p).all((1,2)).mean()))
   for name,ids in [('all',range(12)),('arm',[8,9,10,11]),('wrist',[10,11]),('leg',range(6))]:
    e=error[:,ids][valid[:,ids]]
    for q in [50,95]:m[f'{name}_reprojection_p{q}_px']=float(np.percentile(e,q)) if len(e) else None
   old=norm(cand['current']['points'])[0];diff=np.linalg.norm(p-old,axis=-1)
   m['difference_from_current_p50_leg']=float(np.nanmedian(diff));m['depth_difference_from_current_p95_leg']=float(np.nanpercentile(abs(p[...,2]-old[...,2]),95))
   records.append(m)
  print('CANDIDATES',cid,list(cand),flush=True)
 save(OUT/f'raw_candidates_{a.split}.json',records);csvwrite(OUT/f'raw_candidates_{a.split}.csv',records)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
