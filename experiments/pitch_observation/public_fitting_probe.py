"""Isolate fitting distortion using real public 3D + synthetic noisy 2D cameras.

This is an external-domain component experiment, NOT KBO 3D ground truth.
The camera is known in this probe so pose distortion can be separated from
camera error. Fixed generic SMPL shape; native MotionBERT prediction is an
independent 3D IK target, initialized without any WHAM/v2 pose.
"""
from common import *
from candidates import CC,norm
from public_adaptation import CMB,norm17,setup_model
from public_temporal_adaptation import suffix
from pitch_study_fit import StudyFitter,stable_rotation,spline_basis
import argparse


def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='val');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--coefficients',nargs='+',type=float,default=[.002,.02,.1,.5,2.]);a=ap.parse_args();model=setup_model();model.load_state_dict(torch.load(OUT/'temporal_adaptation/best.pt',map_location='cpu')['trainable'],strict=False);model.eval();cache=torch.load(OUT/'temporal_adaptation/prefix_features.pt',map_location='cpu');selected=[r for i,r in enumerate(cache) if r['split']==a.split and r['camera']['fps'] in [24,30]]
 # One 24Hz CF and one 24Hz broad-angle view per athlete; first two matching
 # distinct views are fixed by deterministic cache order, not their score.
 groups={}
 for r in selected:groups.setdefault(r['subject'],[]).append(r)
 selected=[r for records in groups.values() for r in records[:2]]
 if a.limit:selected=selected[:a.limit]
 # Load WHAM SMPL utilities only after MotionBERT Python modules are resolved.
 for key in list(sys.modules):
  if key=='lib' or key.startswith('lib.'):del sys.modules[key]
 body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda:0');body.prepare_shape(np.zeros(10,np.float32));results=[]
 for index,rec in enumerate(selected):
  case=rec['subject']+'_'+str(index);dest=OUT/'public_fitting_v2'/f'{a.split}_{case}.json'
  if dest.exists():results+=json.loads(dest.read_text());continue
  target=rec['target'].numpy();N=len(target);start=time.monotonic();beta=tensor(np.zeros((N,10),np.float32))
  with torch.no_grad():pred=model.head(suffix(model,rec['features'].float().cuda(0),N))[0].cpu().numpy()
  prediction=norm17(pred)[:,CMB];truth=target[:,CMB];r0=np.tile(np.eye(3),(N,24,1,1)).astype(np.float32)
  left=prediction[:,6]-prediction[:,7];left/=np.maximum(np.linalg.norm(left,axis=-1,keepdims=True),1e-6);up=prediction[:,6:8].mean(1)-prediction[:,:2].mean(1);up-=left*(up*left).sum(-1,keepdims=True);up/=np.maximum(np.linalg.norm(up,axis=-1,keepdims=True),1e-6);front=np.cross(left,up);r0[:,0]=np.stack([left,up,front],-1);r0=tensor(r0)
  with torch.no_grad():p0,native,feet=body.model(r0,beta)
  leg=float(torch.linalg.vector_norm(native[:,4]-native[:,1],dim=-1).median()+torch.linalg.vector_norm(native[:,7]-native[:,4],dim=-1).median());goal=tensor(prediction*leg);delta=torch.nn.Parameter(torch.zeros((N,24,3),device='cuda:0'));opt=torch.optim.Adam([delta],lr=.04)
  for step in range(240):
   opt.param_groups[0]['lr']=.04*(.05+.95*(1-step/239)**2);opt.zero_grad();R=r0@stable_rotation(3.14*torch.tanh(delta/3.14));p,native,feet=body.model(R,beta);loss=((p[:,CC]-goal)/leg).square().mean()+.00003*delta.square().mean();loss.backward();opt.step()
  with torch.no_grad():initial_R=(r0@stable_rotation(3.14*torch.tanh(delta/3.14))).detach();initial,_,_=body.model(initial_R,beta)
  initial_xyz=initial.detach();init_common=norm(initial_xyz.cpu().numpy()[:,CC])[0];_,image_leg=norm(initial_xyz.cpu().numpy()[:,CC]);D=rec['camera']['depth_leg']*image_leg;focal=800.;gt=tensor(truth*image_leg);truth_uv=gt[...,:2]/(gt[...,2:]+D)*focal+512.;height=float((truth_uv[...,1].max()-truth_uv[...,1].min()).cpu());rng=np.random.default_rng(20261028+index);noise=rng.normal(0,height*.015,(N,12,2));target_uv=truth_uv+tensor(noise);weight=np.ones((N,12),np.float32);missing=rng.random((N,12))<.05;weight[missing]=0;weight=tensor(weight);case_results=[]
  for coefficient in a.coefficients:
   angular=torch.nn.Parameter(torch.zeros((N,24,3),device='cuda:0'));basis=tensor(spline_basis(N,rec['camera']['fps'],.08));translation=torch.nn.Parameter(tensor(np.tile([0,0,D],(basis.shape[1],1))));opt=torch.optim.Adam([{'params':[angular],'lr':.008},{'params':[translation],'lr':.002}])
   for step in range(140):
    decay=.1+.9*(1-step/139)**2
    for group,rate in zip(opt.param_groups,[.008,.002]):group['lr']=rate*decay
    opt.zero_grad();R=initial_R@stable_rotation(angular);p,anatomy,feet=body.model(R,beta);v=p[:,CC]+(basis@translation)[:,None];uv=v[...,:2]/v[...,2:]*focal+512.;error=(uv-target_uv)/height;image=((torch.sqrt(error.square().sum(-1)+.005**2)-.005)*weight).sum()/weight.sum();depth=((p[...,2]-initial_xyz[...,2])/leg).square().mean();prior=angular.square().mean();smooth=(R[2:,:,:2]+R[:-2,:,:2]-2*R[1:-1,:,:2]).square().mean();loss=image+coefficient*depth+.0015*prior+.0004*smooth
    loss.backward();torch.nn.utils.clip_grad_norm_([angular,translation],10);opt.step()
   with torch.no_grad():p,_,_=body.model(initial_R@stable_rotation(angular),beta)
   fitted=norm(p.cpu().numpy()[:,CC])[0];row=dict(subject=rec['subject'],pitch=rec['pitch'],split=a.split,camera=rec['camera'],coefficient=coefficient,native_model_error_leg=float(np.linalg.norm(prediction-truth,axis=-1).mean()),smpl_initial_error_leg=float(np.linalg.norm(init_common-truth,axis=-1).mean()),fitted_error_leg=float(np.linalg.norm(fitted-truth,axis=-1).mean()),initial_depth_error_leg=float(abs(init_common[...,2]-truth[...,2]).mean()),fitted_depth_error_leg=float(abs(fitted[...,2]-truth[...,2]).mean()),noisy_2d_fit_objective=float(image.detach()),seconds=time.monotonic()-start,semantics='Actual public mocap joints with known synthetic camera and noisy 2D; sequence-leg normalized, not KBO accuracy');case_results.append(row)
  save(dest,case_results);results+=case_results;print('PUBLIC FIT PROBE',a.split,case,[(x['coefficient'],round(x['fitted_error_leg'],3)) for x in case_results],flush=True)
 save(OUT/f'public_fitting_v2_{a.split}.json',results);csvwrite(OUT/f'public_fitting_v2_{a.split}.csv',results)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
