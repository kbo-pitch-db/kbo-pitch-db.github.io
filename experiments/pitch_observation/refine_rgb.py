"""Cross RGB/2D evidence against identical raw-initialized pose checkpoints.

Depth affects only uncertain limb/torso ordinal inequalities, never a target
joint-center depth or target stride. Different initializers are not averaged.
Every branch starts from the same OBS-2 pose; this is a fitting ablation, not
another independent initializer. Native model outputs remain in raw/.
"""
from common import *
from pitch_study_fit import StudyFitter,stable_rotation,spline_basis
from events import infer
from depth_ordinals import evidence
import argparse

def run(fitter,cid,tag,factor,steps=100):
 import torch
 r,c,s,h,w=init_inputs(cid);basepath=OUT/'fits_v2'/f'{tag}_pose_f{factor}'/f'{cid}.npz';base=load(basepath);N=len(base['keypoints']);device='cuda:0';tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 fitter.prepare_shape(base['smpl_betas']);beta=tensor(np.repeat(base['smpl_betas'][None],N,0));r0=tensor(base['smpl_rotations']);bg=base['background_transforms'];principal=tensor(base['principal']);leg=float(base['model_leg']);basis=tensor(spline_basis(N,float(h['fps']),.08));initial_controls=np.linalg.lstsq(basis.cpu().numpy(),base['translation'],rcond=None)[0]
 ordinal,probes=evidence(cid);train=[q for q in ordinal if q['split_observation']=='fit'];qi=torch.tensor([q['frame'] for q in train],device=device);qj=torch.tensor([q['joint'] for q in train],device=device);fraction=tensor([q['torso_fraction'] for q in train]);sign=tensor([q['near_sign'] for q in train]);alternate=load(OUT/'alternative_2d/rtmw_halpe_adapter'/f'{cid}.npz');events=infer(h,bg)
 temporal=np.ones((N-2,24),np.float32);temporal[:,[16,17,18,19,20,21,22,23]]=.2
 if events.get('arm_fast_interval'):
  a,b=events['arm_fast_interval'];side=0 if events['arm_fast_side']=='left' else 1;temporal[max(0,a-3):min(N-2,b+3),[16+side,18+side,20+side,22+side]]=.02
 temporal=tensor(temporal);dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1)
 if 'track_id' in h:links&=np.diff(h['track_id'])==0
 triple=tensor(links[:-1]&links[1:]);dt=tensor(np.maximum(dt,1e-6));records=[]
 for variant in ['control','depth','rtmw','rtmw_depth']:
  dest=OUT/'fits_rgb'/f'{tag}_{variant}_f{factor}'/f'{cid}.npz'
  if dest.exists():continue
  start=time.monotonic();delta=torch.nn.Parameter(torch.zeros((N,24,3),device=device));controls=torch.nn.Parameter(tensor(initial_controls));lf=torch.nn.Parameter(tensor(0.));opt=torch.optim.Adam([{'params':[delta],'lr':.008},{'params':[controls],'lr':.004},{'params':[lf],'lr':.0004}])
  obs=alternate if variant.startswith('rtmw') else h;xy=np.einsum('tij,tkj->tki',bg,np.concatenate([obs['keypoints'],np.ones((*obs['keypoints'].shape[:2],1))],-1))[...,:2];score=scores(obs);height=max(100,float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])));target=tensor(np.nan_to_num(np.concatenate([xy[:,:17],xy[:,20:26]],1)));weights=np.concatenate([score[:,:17]**2,score[:,20:26]**2*.35],1);weights[:,:5]*=.2;weights[base['withheld_frames']]=0;weights=tensor(weights)
  def evaluate():
   angular=3.14*torch.tanh(delta/3.14);rot=r0@stable_rotation(angular);p,a,feet=fitter.model(rot,beta);tr=basis@controls;f=float(base['focal'])*torch.exp(.15*torch.tanh(lf));pts=torch.cat([p,feet],1)+tr[:,None];uv=pts[...,:2]/pts[...,2:].clamp_min(.5)*f+principal
   residual=(uv-target)/height;image=((torch.sqrt(residual.square().sum(-1)+.005**2)-.005)*weights).sum()/weights.sum().clamp_min(1);prior=angular.square().mean()
   smooth=((rot[2:,:,:2]+rot[:-2,:,:2]-2*rot[1:-1,:,:2]).square()*temporal[:,:,None,None]*triple[:,None,None,None]*(1/(30*dt[1:]))[:,None,None,None]**4).mean()
   depth=tensor(0.)
   if len(train):
    ref=p[qi][:,[5,6]].mean(1)*(1-fraction[:,None])+p[qi][:,[11,12]].mean(1)*fraction[:,None];gap=(p[qi,qj,2]-ref[:,2])/leg;depth=torch.relu(sign*gap-.25).square().mean()
   cam=((tr[:,2]-tensor(base['translation'][:,2]))/leg).square().mean();loss=image+.0015*prior+.0004*smooth+.00005*cam
   if variant in ['depth','rtmw_depth']:loss=loss+.005*depth
   return loss,(p,a,feet,tr,uv,f,rot),dict(image=image,prior=prior,depth_ordinal=depth,temporal=smooth)
  for i in range(steps):
   decay=.1+.9*(1-i/max(1,steps-1))**2
   for group,lr in zip(opt.param_groups,[.008,.004,.0004]):group['lr']=lr*decay
   opt.zero_grad();loss,data,losses=evaluate()
   if not torch.isfinite(loss):raise ValueError('Nonfinite RGB fit '+cid)
   loss.backward();torch.nn.utils.clip_grad_norm_([delta,controls,lf],10);opt.step()
  with torch.no_grad():loss,data,losses=evaluate()
  p,a,feet,tr,uv,f,rot=[x.cpu().numpy() for x in data];projected=np.einsum('tij,tkj->tki',np.linalg.inv(bg),np.concatenate([uv,np.ones((*uv.shape[:2],1))],-1))[...,:2]
  out=dict(base);out.update(keypoints=p,anatomical_keypoints=a,foot_keypoints=feet,translation=tr,projected=projected,projected_stabilized=uv,focal=f,smpl_rotations=rot);atomic_npz(dest,**out)
  stats=dict(id=cid,initializer=tag,variant=variant,focal_factor=factor,seconds=time.monotonic()-start,base_checkpoint=str(basepath.relative_to(OUT)),base_sha256=sha(basepath),source_2d='Separate RTMW384' if variant.startswith('rtmw') else 'Final Halpe26 read only',train_ordinal_count=len(train),withheld_ordinal_count=len(ordinal)-len(train),loss={k:float(v) for k,v in losses.items()},legacy_v2_pose_used=False,pseudo_depth_is_gt=False,path=str(dest.relative_to(OUT)));save(dest.with_suffix('.json'),stats);records.append(stats);print('RGB REFINE',cid,tag,factor,variant,round(stats['seconds'],1),flush=True)
 return records

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--initializers',nargs='+',default=['WHAM_raw','hmr2a','gvhmr']);ap.add_argument('--factors',nargs='+',type=int,default=[4]);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();fitter=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);recs=sample(a.split,not a.full_sample)
 if a.limit:recs=recs[:a.limit]
 records=[]
 for rec in recs:
  for tag in a.initializers:
   for factor in a.factors:records+=run(fitter,rec['id'],tag,factor)
 save(OUT/f'fits_rgb_{a.split}.json',records)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
