"""Compare native RGB output, its own SMPL fit, and controlled fitting separately."""
from common import *
from candidates import CC,norm
from scipy.spatial.transform import Rotation
from pitch_study_fit import StudyFitter
import argparse
NJ=[1,2,4,5,7,8,16,17,18,19,20,21]

def main():
 import torch
 from evaluate import evaluate_record
 from events import infer
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample(a.split,a.split=='tune');records=records[:a.limit] if a.limit else records;body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);rows_=[];source=[]
 for rec in records:
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];ev=infer(h,bg)
  for path in sorted((OUT/'raw/nlf_keyframes'/cid).glob('*.npz')):
   z=load(path);i=int(z['frame']);p=z['joints3d_nonparam'][0];fit=z['joints3d'][0];leg=float(np.linalg.norm(p[4]-p[1])+np.linalg.norm(p[7]-p[4]));q=dict(id=cid,frame=i,factor=path.stem.split('_f')[1],native_vs_own_smpl_fit_p50_leg=float(np.median(np.linalg.norm(p[NJ]-fit[NJ],axis=-1)))/leg,native_joint_uncertainty_p50_mm=float(np.median(z['joint_uncertainties'])),uncertainty_is_not_calibrated_accuracy=True)
   rot=Rotation.from_rotvec(z['pose'].reshape(24,3)).as_matrix().astype(np.float32)
   body.prepare_shape(z['betas'][0])
   with torch.no_grad():_,anatomy,_=body.model(torch.tensor(rot[None],device='cuda:0'),torch.tensor(z['betas'],device='cuda:0'))
   aa=anatomy[0].cpu().numpy();aa-=aa[0];ff=fit/1000;ff-=ff[0];q['smpl_parameter_adapter_max_joint_error_m']=float(np.max(np.linalg.norm(aa-ff,axis=-1)));rows_.append(q)
  directory='fits_nlf' if a.split=='tune' else 'fits_nlf_test'
  for path in (OUT/directory).glob(f'*/{cid}.npz'):
   z=load(path);p,leg=norm(z['keypoints'][:,CC]);record=dict(name=directory+'/'+path.parent.name,common=p,uv=z['projected'][:,CC],leg=leg,global_root=z['translation'],native=z['anatomical_keypoints'],z=z,kind='fit');v=evaluate_record(record,h,cid,a.split,[],ev);v['field_order']=__import__('axis_order').metrics(z['keypoints'][:,CC],__import__('axis_order').setup(cid,h));source.append(v)
  print('NLF COMPARE',cid,flush=True)
 save(OUT/f'nlf_native_diagnostics_{a.split}.json',rows_);csvwrite(OUT/f'nlf_native_diagnostics_{a.split}.csv',rows_);save(OUT/f'nlf_fitting_evaluation_{a.split}.json',source);csvwrite(OUT/f'nlf_fitting_evaluation_{a.split}.csv',source)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
