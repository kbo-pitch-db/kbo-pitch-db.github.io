"""Held-out RGB silhouette/surface-depth diagnostics; never 3D ground truth.

The depth reflection probe is deliberately adversarial, not another physical
body model. A good score cannot certify depth. Occlusion visibility, masks and
crop sensitivity are retained with original RGB evidence.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from common import *
from candidates import raw_candidates
import argparse

def geometry(fitter,rot,beta):
 import torch
 beta=np.asarray(beta);beta=beta[None] if beta.ndim==1 else beta
 with torch.no_grad():
  out=fitter.forward_smpl(fitter.body,body_pose=torch.tensor(rot[None,1:],device='cuda:0',dtype=torch.float32),global_orient=torch.tensor(rot[None,:1],device='cuda:0',dtype=torch.float32),betas=torch.tensor(beta,device='cuda:0',dtype=torch.float32),pose2rot=False,return_verts=True)
  vertices=out.vertices[0];joints=fitter.body.J_regressor_wham@vertices;root=joints[[11,12]].mean(0)
 return (vertices-root).cpu().numpy()

def raster(vertices,faces,colors,translation,focal,principal,origin,scale,renderer):
 import pyrender,trimesh
 v=vertices+translation;v=v*np.array([1,-1,-1])
 mesh=trimesh.Trimesh(v,faces,vertex_colors=colors,process=False)
 scene=pyrender.Scene(bg_color=[0,0,0,0]);scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
 camera=pyrender.IntrinsicsCamera(fx=focal*scale,fy=focal*scale,cx=(principal[0]-origin[0])*scale,cy=(principal[1]-origin[1])*scale,znear=.01,zfar=10000.)
 scene.add(camera,pose=np.eye(4));return renderer.render(scene,flags=pyrender.RenderFlags.FLAT|pyrender.RenderFlags.RGBA)

def main():
 import torch,cv2,pyrender
 from scipy.stats import spearmanr
 from scipy.ndimage import binary_erosion
 from scene_fit import SceneFitter
 torch.set_num_threads(1);cv2.setNumThreads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--fit-dir',default='fits_v2');ap.add_argument('--rgb',action='store_true');ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--fits',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();fitter=SceneFitter(WORKSPACE/'external/WHAM','cuda:0')
 faces=fitter.body.faces;skin=fitter.body.lbs_weights.argmax(-1).cpu().numpy();palette=np.tile([180,170,130,255],(24,1));palette[[1,4,7,10,13,16,18,20,22]]=[210,60,60,255];palette[[2,5,8,11,14,17,19,21,23]]=[70,110,230,255];colors=palette[skin].astype(np.uint8)
 renderer=pyrender.OffscreenRenderer(256,512);records=[];samples=sample(a.split,not a.full_sample)
 if a.limit:samples=samples[:a.limit]
 for rec in samples:
  cid=rec['id'];candidate,h,c,s=raw_candidates(cid);rawpath=OUT/'raw/hmr2a'/f'{cid}.npz';hmr=load(rawpath);w=init_inputs(cid)[4];old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);start=time.monotonic()
  entries={}
  for tag in ['current','WHAM_raw','hmr2a','gvhmr_f1','gvhmr_f4','gvhmr_f12']:
   z=load(OUT/'candidates'/tag/f'{cid}.npz');rot=candidate[tag]['rotations'];beta=hmr['smpl_betas'] if tag=='hmr2a' else w['smpl_betas'] if tag=='WHAM_raw' else candidate[tag]['beta']
   entries[tag]=(z,rot,beta,float(z['model_leg']))
  if a.fits:
   for tag in ['WHAM_raw','hmr2a','gvhmr','mb_smpl','mb_obp_smpl']:
    for stage in ['control','temporal','soft_confidence','soft_phase','combined','hard']:
     for fac in [4]:
      name=f'{tag}_{stage}_f{fac}';p=OUT/a.fit_dir/name/f'{cid}.npz'
      if p.exists():
       z=load(p);entries[name]=(z,z['smpl_rotations'],z['smpl_betas'],1.)
  if a.rgb:
   for tag in ['WHAM_raw','hmr2a','gvhmr']:
    for variant in ['control','depth','rtmw','rtmw_depth']:
     name=f'RGB_{tag}_{variant}_f4';p=OUT/'fits_rgb'/f'{tag}_{variant}_f4'/f'{cid}.npz'
     if p.exists():
      z=load(p);entries[name]=(z,z['smpl_rotations'],z['smpl_betas'],1.)
  allresults=[]
  for obs_path in sorted((OUT/'observations'/cid).glob('*.npz')):
   idx=int(obs_path.stem);obs=load(obs_path)
   if 'mask' not in obs:continue
   image=cv2.imread(str(obs_path.with_suffix('.jpg')));hh,ww=image.shape[:2];scale=min(1.,360/max(hh,ww));H,W=int(round(hh*scale)),int(round(ww*scale));renderer.viewport_width=W;renderer.viewport_height=H
   image=cv2.resize(image,(W,H));mask=cv2.resize(obs['mask'].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0;depths=[cv2.resize(obs[k].astype(np.float32),(W,H)) for k in ['full_image_inverse_depth','person_context_inverse_depth']]
   output=OUT/'mesh_evidence'/cid;output.mkdir(parents=True,exist_ok=True);panels=[]
   for name,(z,rot,beta,leg) in entries.items():
    if not np.isfinite(rot[idx]).all():continue
    bet=beta[idx] if np.asarray(beta).ndim==2 else beta
    vertices=geometry(fitter,rot[idx],bet)/leg;tr=z['translation'][idx];focal=float(z['focal']);principal=z['principal'];origin=obs['crop_origin']
    if 'background_transforms' in z:
     # Use equivalent full-image intrinsics/extrinsics for a similarity image warp.
     bg=z['background_transforms'][idx];inverse=np.linalg.inv(bg);A=inverse[:2,:2];sc=np.linalg.norm(A[0]);angle=np.arctan2(A[1,0],A[0,0]);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
     vertices=vertices@R.T;tr=tr@R.T;focal*=sc;principal=(inverse@np.r_[principal,1])[:2]
    color,render_depth=raster(vertices,faces,colors,tr,focal,principal,origin,scale,renderer);render_mask=render_depth>0;inter=(render_mask&mask);union=(render_mask|mask);iou=float(inter.sum()/max(union.sum(),1))
    m=dict(id=cid,split=rec['split'],frame=idx,candidate=name,silhouette_iou=iou,mask_pixels=int(mask.sum()),surface_overlap_pixels=int(inter.sum()),source='SAM box-prompt mask, not manual GT')
    overlap=binary_erosion(inter,iterations=2);inverse_depth=np.where(render_depth>0,1/np.maximum(render_depth,1e-6),np.nan)
    for label,zdepth in zip(['full','crop'],depths):
     good=overlap&np.isfinite(inverse_depth)
     rho=float(spearmanr(inverse_depth[good],zdepth[good]).statistic) if good.sum()>100 else None
     m[label+'_depth_spearman']=rho
    if name in ['current','hmr2a','gvhmr_f4','WHAM_raw']:
     blended=image.copy();blended[render_mask]=(image[render_mask]*.35+color[...,:3][render_mask][:,::-1]*.65).astype(np.uint8);cv2.putText(blended,name,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
     panels.append(blended)
    # Compare the same mesh with reversed camera-relative depth, without averaging.
    if name in ['WHAM_raw','hmr2a','gvhmr_f4','current']:
     reflection=vertices.copy();reflection[:,2]*=-1;_,rd=raster(reflection,faces[:,::-1],colors,tr,focal,principal,origin,scale,renderer);ri=np.where(rd>0,1/np.maximum(rd,1e-6),np.nan)
     for label,zdepth in zip(['full','crop'],depths):
      good=binary_erosion(mask&(rd>0),iterations=2)&np.isfinite(ri)
      rho=float(spearmanr(ri[good],zdepth[good]).statistic) if good.sum()>100 else None;m[label+'_reflection_spearman']=rho
    allresults.append(m)
   if panels:cv2.imwrite(str(output/f'{idx:05d}.jpg'),np.concatenate([image]+panels,1))
  save(OUT/'mesh_evidence'/cid/('final_metrics.json' if a.fits else 'raw_metrics.json'),allresults);records+=allresults;print('MESH EVIDENCE',cid,len(allresults),round(time.monotonic()-start,1),flush=True)
 renderer.delete();save(OUT/f'mesh_{a.split}_{a.fit_dir if a.fits else "raw"}.json',records);csvwrite(OUT/f'mesh_{a.split}_{a.fit_dir if a.fits else "raw"}.csv',records)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
