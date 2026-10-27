"""Independent RGB initializer before/after fitting on identical source keyframes."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from common import *
from scipy.spatial.transform import Rotation
from mesh_evidence import geometry,raster
from pool import load_pool
import pool,argparse

def main():
 import torch,cv2,pyrender
 from scene_fit import SceneFitter
 torch.set_num_threads(1);cv2.setNumThreads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample(a.split,a.split=='tune');records=records[a.offset:];records=records[:a.limit] if a.limit else records;body=SceneFitter(WORKSPACE/'external/WHAM','cuda:0');faces=body.body.faces;colors=np.tile([170,190,200,255],(len(body.body.v_template),1)).astype(np.uint8);renderer=pyrender.OffscreenRenderer(256,512);stats=[];pool.VERSION='strong'
 for rec in records:
  cid=rec['id'];directory='fits_nlf' if a.split=='tune' else 'fits_nlf_test';fitted=load(OUT/directory/'nlf_rgb_temporal_f4'/f'{cid}.npz');old=dict(load_pool(cid,a.split)[0])['current_actual'];folder=OUT/'nlf_comparison_rgb'/cid;folder.mkdir(parents=True,exist_ok=True)
  for order,path in enumerate(sorted((OUT/'observations'/cid).glob('*.npz'))):
   idx=int(path.stem);obs=load(path);p=OUT/'raw/nlf_frames'/cid/f'{idx:05d}_f4.npz';raw=load(p);hh,ww=obs['mask'].shape;scale=min(1.,360/max(hh,ww));H,W=int(round(hh*scale)),int(round(ww*scale));renderer.viewport_width=W;renderer.viewport_height=H;mask=cv2.resize(obs['mask'].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0;original=cv2.resize(cv2.imread(str(path.with_suffix('.jpg'))),(W,H));tiles=[original]
   native_shape_path=OUT/('fits_nlf_native_shape' if a.split=='tune' else 'fits_nlf_native_shape_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'
   entries=[('current_actual',old),('NLF_raw_own_SMPL_fit',None),('NLF_raw_fixed_shape','fixed_shape'),('NLF_controlled_WHAM_shape',fitted)]
   if native_shape_path.exists():entries.append(('NLF_controlled_native_shape',load(native_shape_path)))
   rgb_path=OUT/('fits_nlf_rgb' if a.split=='tune' else 'fits_nlf_rgb_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'
   if rgb_path.exists():entries.append(('NLF_RGB_contour_fitting',load(rgb_path)))
   for name,z in entries:
    if isinstance(z,str):
     beta=np.median(load(OUT/'adapters/nlf_rgb'/f'{cid}.npz')['smpl_betas'],0);rot=Rotation.from_rotvec(raw['pose'].reshape(24,3)).as_matrix();vertices=geometry(body,rot,beta);reg=body.body.J_regressor_wham[[11,12]].mean(0).cpu().numpy();tr=reg@(raw['vertices3d'][0]/1000);f=float(raw['focal']);principal=raw['principal']
    elif z is None:vertices=raw['vertices3d'][0]/1000;tr=np.zeros(3);f=float(raw['focal']);principal=raw['principal']
    else:
     beta=z['smpl_betas'];beta=beta[idx] if beta.ndim==2 else beta;vertices=geometry(body,z['smpl_rotations'][idx],beta);tr=z['translation'][idx];f=float(z['focal']);principal=z['principal'];iv=np.linalg.inv(z['background_transforms'][idx]);A=iv[:2,:2];sc=np.linalg.norm(A[0]);angle=np.arctan2(A[1,0],A[0,0]);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]);vertices=vertices@R.T;tr=tr@R.T;f*=sc;principal=(iv@np.r_[principal,1])[:2]
    color,depth=raster(vertices,faces,colors,tr,f,principal,obs['crop_origin'],scale,renderer);rend=depth>0;iou=float((rend&mask).sum()/max((rend|mask).sum(),1));stats.append(dict(id=cid,split=a.split,frame=idx,observation_split='withheld' if order%3==2 else 'selection',candidate=name,silhouette_iou=iou,mask_is_not_gt=True));panel=original.copy();panel[rend]=(panel[rend]*.4+color[...,:3][rend][:,::-1]*.6).astype(np.uint8);cv2.putText(panel,name,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1);tiles.append(panel)
   cv2.imwrite(str(folder/f'{idx:05d}.jpg'),np.concatenate(tiles,1))
  print('NLF MESH',cid,flush=True)
 renderer.delete();save(OUT/f'nlf_mesh_rgb_{a.split}_offset{a.offset}.json',stats);csvwrite(OUT/f'nlf_mesh_rgb_{a.split}_offset{a.offset}.csv',stats)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
