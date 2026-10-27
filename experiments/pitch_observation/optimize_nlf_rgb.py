"""Raw-initialized, staged camera/articulation fitting. No v2 pose initialization.

All camera hypotheses survive. Body shape is one fixed WHAM-derived SMPL beta
per sequence for controlled initializer comparisons, not a known human height.
The similarity stabilization is an image warp, not a metric camera solution.
"""
from common import *
from candidates import CC,norm
from events import infer,contact
from pitch_study_fit import StudyFitter,stable_rotation,spline_basis
from scene_geometry import initial_camera,fill_pose_gaps
import argparse,copy
from scipy.spatial.transform import Rotation

from rgb_mask_constraint import MaskFitter

class Fitter(MaskFitter):
 def fit_observed(self,cid,tag,steps=(60,160,80,80,80,80,80,80),mode='phase'):
  import torch
  self.configure_rgb(cid);start=time.monotonic();r,c,s,h,w=init_inputs(cid);N=len(h['keypoints']);device=self.device;tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
  H=3;factors=np.array([1.,4.,12.]);beta=np.nanmedian(load(OUT/'adapters/nlf_rgb'/f'{cid}.npz')['smpl_betas'],0);self.prepare_shape(beta)
  initial=[];source_paths=[]
  for i,f in enumerate(factors):
   if tag=='WHAM_raw':z=w;rot=np.concatenate([z['smpl_global_orient'],z['smpl_body_pose']],1);source_paths.append(DATA/r['prior_3d'])
   else:
    sub=tag if tag in ['hmr2a','mb_smpl','mb_obp_smpl','mb_obp_single_smpl','mb_obp_flip_smpl','wham_single','wham_flip_restored','mb_block_single_smpl','mb_block_flip_smpl','nlf_rgb'] else f'gvhmr_f{f:g}';p=OUT/('raw' if sub in ['hmr2a','wham_single','wham_flip_restored'] else 'adapters')/sub/f'{cid}.npz';z=load(p);rot=z['smpl_rotations'];source_paths.append(p)
   valid=np.isfinite(rot).all((1,2,3));rot=rot.copy()
   for k in np.flatnonzero(~valid):
    good=np.flatnonzero(valid);rot[k]=rot[good[np.argmin(abs(good-k))]]
   initial.append(rot)
  initial=np.array(initial);r0=tensor(initial);beta_t=tensor(np.repeat(beta[None],H*N,0))
  with torch.no_grad():
   p0,a0,f0=self.model(r0.reshape(-1,24,3,3),beta_t);p0=p0.reshape(H,N,17,3);a0=a0.reshape(H,N,24,3);f0=f0.reshape(H,N,6,3)
  leg=float(torch.linalg.vector_norm(a0[0,:,4]-a0[0,:,1],dim=-1).median()+torch.linalg.vector_norm(a0[0,:,7]-a0[0,:,4],dim=-1).median())
  bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];score=scores(h)
  height=max(100,float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])));principal=np.array([float(s['width'])/2,float(s['height'])/2]);focals=factors*np.hypot(float(s['width']),float(s['height']))
  withheld=(np.arange(N)%5==2) if self.withhold_2d else np.zeros(N,bool);camera_score=score.copy();camera_score[withheld]=0
  trans=[]
  for i,f in enumerate(focals):
   scale0,_,tr=initial_camera(p0[i].cpu().numpy(),xy[:,:17],camera_score[:,:17],f,principal)
   camera_valid=((camera_score[:,:17]>.3).sum(1)>=6)&(scale0>31)
   if camera_valid.sum()<4:raise ValueError('insufficient camera observations '+cid)
   for axis in range(3):tr[:,axis]=np.interp(np.arange(N),np.flatnonzero(camera_valid),tr[camera_valid,axis])
   tr[:,2]=np.nanmedian(tr[:,2]);trans.append(tr)
  trans=np.array(trans);basis=tensor(spline_basis(N,float(h['fps']),spacing=.08));controls0=np.array([np.linalg.lstsq(basis.cpu().numpy(),v,rcond=None)[0] for v in trans]);controls=torch.nn.Parameter(tensor(controls0));logf=torch.nn.Parameter(torch.zeros(H,device=device));delta=torch.nn.Parameter(torch.zeros((H,N,24,3),device=device))
  plane_angle=torch.nn.Parameter(torch.full((H,),.10,device=device));floor=torch.nn.Parameter(tensor(np.quantile(((f0.cpu().numpy()+trans[:,:,None])*np.array([0,np.cos(.10),np.sin(.10)])).sum(-1),.94,axis=(1,2))))
  optimizer=torch.optim.Adam([{'params':[controls],'lr':.012},{'params':[delta],'lr':.025},{'params':[logf,plane_angle,floor],'lr':.002}])
  target=tensor(np.concatenate([np.nan_to_num(xy[:,:17]),np.nan_to_num(xy[:,20:26])],1));weight=np.concatenate([score[:,:17]**2,score[:,20:26]**2*.35],1);weight[:,:5]*=.2
  # One fifth of frames is withheld from fitting, fixed before model selection.
  withheld=(np.arange(N)%5==2) if self.withhold_2d else np.zeros(N,bool);weight[withheld]=0;weight=tensor(weight)
  events=infer(h,bg);cw=tensor(contact(h,events,bg))
  from events import smooth_xy
  sx,ss=smooth_xy(h,bg);foot_speed=np.linalg.norm(np.gradient(sx[:,20:26],h['source_time'],axis=0),axis=-1);cw_conf=tensor(np.exp(-(foot_speed/.3)**2)*ss[:,20:26]**2)
  active_variant='camera'
  # A separate forefoot-midpoint hard-anchor ablation shares the same pose start.
  lead=0 if events.get('lead_side')=='left' else 1;rear=1-lead;plant=events.get('plant_interval');burst=events.get('arm_fast_interval');lift=events.get('lift_interval')
  phase_mix=np.zeros(N,np.float32);anchor_mask=np.zeros(N,np.float32)
  if plant and burst:
   phase_mix=np.clip((np.arange(N)-plant[0])/max(1,plant[1]-plant[0]),0,1).astype(np.float32);st=lift[0] if lift else 0;anchor_mask[st:burst[1]+1]=1.
  rear_mid=(f0[:,:,rear]+f0[:,:,rear+2])*.5;lead_mid=(f0[:,:,lead]+f0[:,:,lead+2])*.5
  rear_anchor=(rear_mid[:,max(0,(lift or [0])[0])]+tensor(trans)[:,max(0,(lift or [0])[0])]).detach()
  lead_anchor=torch.nn.Parameter((lead_mid[:,(plant or [0,0])[1]]+tensor(trans)[:,(plant or [0,0])[1]]).detach().clone());optimizer.add_param_group({'params':[lead_anchor],'lr':.012})
  dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1)
  # A tracker-id discontinuity prevents temporal priors across a possible identity gap.
  if 'track_id' in h:links&=np.diff(h['track_id'])==0
  link=tensor(links);triple=tensor(links[:-1]&links[1:]);rate=tensor(np.maximum(dt,1e-6));principal_t=tensor(principal)
  poseweights=tensor([1.,.6,.6,1.,.45,.45,1.,.3,.3,1.,.3,.3,1.,.6,.6,.8,.3,.3,.2,.2,.1,.1,.1,.1])
  temporal=poseweights[None].repeat(N-2,1)
  if events.get('arm_fast_interval'):
   aa,bb=events['arm_fast_interval'];side=0 if events['arm_fast_side']=='left' else 1
   temporal[max(0,aa-3):min(N-2,bb+3),[16+side,18+side,20+side,22+side]]*=.1
  prior_image=torch.tensor(p0.cpu().numpy(),device=device)
  def evaluate(stage):
   angular=3.14*torch.tanh(delta/3.14);rot=r0@stable_rotation(angular)
   p,anatomy,feet=self.model(rot.reshape(-1,24,3,3),beta_t);p=p.reshape(H,N,17,3);anatomy=anatomy.reshape(H,N,24,3);feet=feet.reshape(H,N,6,3)
   tr=torch.einsum('tk,hkc->htc',basis,controls)
   if active_variant=='hard':
    rm=(feet[:,:,rear]+feet[:,:,rear+2])*.5;lm=(feet[:,:,lead]+feet[:,:,lead+2])*.5;mix=tensor(phase_mix)[None,:,None];am=tensor(anchor_mask)[None,:,None]
    hard=(rear_anchor[:,None]-rm)*(1-mix)+(lead_anchor[:,None]-lm)*mix;tr=am*hard+(1-am)*tr
   f=tensor(focals)*torch.exp(.3*torch.tanh(logf))
   pts=torch.cat([p,feet],2)+tr[:,:,None];proj=pts[...,:2]/pts[...,2:].clamp_min(.5)*f[:,None,None,None]+principal_t
   err=(proj-target)/height;image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1)
   prior=(angular.square()*poseweights[None,None,:,None]).mean((1,2,3))
   # Uniform priors across candidates; no stride or mean-pitch target.
   cam_depth=((tr[:,:,2]-tensor(trans[:,:,2]))/leg).square().mean(1)
   depth_curve=((tr[:,2:,2]+tr[:,:-2,2]-2*tr[:,1:-1,2])/leg).square().mean(1)
   relative_depth=((p[...,2]-p0[...,2])/leg).square().mean((1,2))
   rgb=self.rgb_loss(tr,f,principal_t)
   total=self.rgb_weight*rgb+image+self.depth_weight*relative_depth+.0015*prior+.00001*cam_depth+.0002*depth_curve+.00001*logf.square()
   smooth=(((rot[:,2:,:,:2]+rot[:,:-2,:,:2]-2*rot[:,1:-1,:,:2])*(30*rate[1:])[None,:,None,None,None].reciprocal()**2).square()*temporal[None,:,:,None,None]*triple[None,:,None,None,None]).mean((1,2,3,4))
   root_smooth=(((tr[:,2:]-tr[:,1:-1])/rate[1:,None]-(tr[:,1:-1]-tr[:,:-2])/rate[:-1,None])/30/leg).square().mean((1,2))
   if active_variant in ['temporal','combined']:total=total+.0004*smooth+.00005*root_smooth
   normal=torch.stack([torch.zeros(H,device=device),-torch.cos(plane_angle),-torch.sin(plane_angle)],1)
   signed=((feet+tr[:,:,None])*normal[:,None,None]).sum(-1)+floor[:,None,None]
   support_weights=cw_conf if active_variant=='soft_confidence' else cw
   contact_height=(signed.square()*support_weights).sum((1,2))/support_weights.sum().clamp_min(1)/leg**2
   fv=(feet[:,1:]+tr[:,1:,None]-feet[:,:-1]-tr[:,:-1,None])/rate[None,:,None,None]/leg
   nv=(fv*normal[:,None,None]).sum(-1);tv=(fv-fv.new_tensor(1)*nv[...,None]*normal[:,None,None]).square().sum(-1)
   cl=torch.minimum(support_weights[1:],support_weights[:-1])*link[:,None];slide=((torch.sqrt(tv+.01)-.1)*cl).sum((1,2))/cl.sum().clamp_min(1)
   if active_variant in ['soft_phase','soft_confidence','combined']:
    # Independent heel/toe weights, scalar plane, weak normal/tangent losses.
    # No foot midpoint XYZ anchor, no post-fit floor lift, no forced support tail.
    if mode=='confidence':
     contact_height=contact_height # Explicit separate experiment can supply non-phase weights.
    total=total+.0006*contact_height+.00004*slide+.00001*(plane_angle-.10).square()
   return total,dict(image=image,rgb_contour=rgb,raw_relative_depth_prior=relative_depth,rotation_prior=prior,temporal=smooth,contact_normal=contact_height,contact_tangent=slide),(p,anatomy,feet,tr,proj,f,rot,normal,signed)
  phases=['camera','pose']+self.branches;steps=[60,160]+[80]*len(self.branches);allstats=[];base_parameters=None;base_optimizer=None
  parameters=[controls,delta,logf,plane_angle,floor,lead_anchor]
  for stage,nsteps in enumerate(steps):
   active_variant=phases[stage]
   if stage>=2:
    with torch.no_grad():
     for param,value in zip(parameters,base_parameters):param.copy_(value)
    optimizer.load_state_dict(copy.deepcopy(base_optimizer))
   for iteration in range(nsteps):
    delta.requires_grad_(stage>0)
    if stage==0:rates=[.012,.025,.002,.012]
    elif stage==1:rates=[.006,.012,.001,.006]
    else:rates=[.002,.003,.0003,.002]
    decay=.15+.85*(1-iteration/max(1,nsteps-1))**2
    for group,rate0 in zip(optimizer.param_groups,rates):group['lr']=rate0*decay
    optimizer.zero_grad(set_to_none=True);total,loss,data=evaluate(stage)
    if not torch.isfinite(total).all():raise ValueError('nonfinite objective '+cid)
    total.sum().backward();torch.nn.utils.clip_grad_norm_([controls,delta,logf,plane_angle,floor],10);optimizer.step()
   if stage==1:
    base_parameters=[p.detach().clone() for p in parameters];base_optimizer=copy.deepcopy(optimizer.state_dict())
   with torch.no_grad():total,loss,data=evaluate(stage)
   p,anatomy,feet,tr,proj,f,rot,normal,signed=[x.detach().cpu().numpy() for x in data]
   # Undo image stabilization when comparing to original 2D observations.
   uv=np.einsum('tij,htkj->htki',np.linalg.inv(bg),np.concatenate([proj,np.ones((*proj.shape[:3],1))],-1))[...,:2]
   for i,fac in enumerate(factors):
    dest=OUT/self.fit_dir/f'{tag}_{phases[stage]}_f{fac:g}'/f'{cid}.npz'
    atomic_npz(dest,keypoints=p[i],anatomical_keypoints=anatomy[i],foot_keypoints=feet[i],translation=tr[i],projected=uv[i],projected_stabilized=proj[i],focal=f[i],principal=principal,
     smpl_rotations=rot[i],smpl_betas=beta,source_time=h['source_time'],frame_index=h['frame_index'],normal=normal[i],foot_signed_height=signed[i],contact_weight=cw.cpu().numpy(),withheld_frames=withheld,
     background_transforms=bg,model_leg=leg,coordinate_system='camera relative articulation plus uncalibrated translation in image-stabilized reference; model scale, not calibrated world')
    err=np.linalg.norm(uv[i]-np.concatenate([h['keypoints'][:,:17],h['keypoints'][:,20:26]],1),axis=-1);sc=np.concatenate([score[:,:17],score[:,20:26]],1);ok=(sc>.5)&np.isfinite(err)
    m=dict(id=cid,phase_policy='conservative_v2' if 'contact_confidence' in events else 'initial_heuristic_v1',optimization_schedule=list(zip(phases,steps)),fit_version=self.fit_dir,shape_policy='tune_player_shared' if cid in self.shared_betas else 'per_sequence',raw_depth_prior_weight=self.depth_weight,all_2d_observations_used=not self.withhold_2d,withheld_semantics='Withheld only from final optimizer; pretrained initializers still saw full RGB/2D sequence',initializer=tag,stage=phases[stage],focal_factor=fac,seconds_elapsed=time.monotonic()-start,loss={k:float(v[i]) for k,v in loss.items()},events=events)
    for split,mask in [('fit',~withheld),('withheld',withheld)]:
     for name,ids in [('body',range(17)),('wrist',[9,10]),('elbow',[7,8]),('feet',range(17,23))]:
      e=err[:,ids][ok[:,ids]&mask[:,None]]
      for q in [50,95]:m[f'{split}_{name}_p{q}_px']=float(np.percentile(e,q)) if len(e) else None
    oldnorm=norm(p0[i].cpu().numpy()[:,CC])[0];newnorm=norm(p[i,:,CC].transpose(1,0,2) if p[i,:,CC].shape[0]==len(CC) else p[i,:,CC])[0]
    m['initial_to_fit_p50_leg']=float(np.nanmedian(np.linalg.norm(newnorm-oldnorm,axis=-1)));m['path']=str(dest.relative_to(OUT));m['v2_pose_used']=False
    allstats.append(m);save(dest.with_suffix('.json'),m)
   print('FIT STAGE',cid,tag,phases[stage],round(time.monotonic()-start,1),flush=True)
  return allstats

def main():
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261028)
 ap=argparse.ArgumentParser();ap.add_argument('--rgb-weight',type=float,default=.10);ap.add_argument('--branches',nargs='+',default=['control','temporal','soft_confidence','soft_phase','combined','hard']);ap.add_argument('--shared-shape');ap.add_argument('--full-2d',action='store_true');ap.add_argument('--depth-preserve',type=float,default=0.);ap.add_argument('--fit-dir',default='fits_v2');ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--initializers',nargs='+',default=['WHAM_raw','hmr2a','gvhmr']);ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--id');a=ap.parse_args()
 fitter=Fitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);fitter.rgb_weight=a.rgb_weight;fitter.branches=a.branches;fitter.shared_betas=json.loads(Path(a.shared_shape).read_text())['by_clip'] if a.shared_shape else {};fitter.fit_dir=a.fit_dir;fitter.withhold_2d=not a.full_2d;fitter.depth_weight=a.depth_preserve;records=sample(a.split,not a.full_sample)
 if a.id:records=[r for r in records if r['id']==a.id]
 records=records[a.offset:]
 if a.limit:records=records[:a.limit]
 allstats=[]
 for rec in records:
  for tag in a.initializers:
   dest=OUT/a.fit_dir/f'{tag}_{a.branches[-1]}_f12'/f'{rec["id"]}.json'
   if dest.exists():print('CACHE FIT',rec['id'],tag,flush=True);continue
   allstats+=fitter.fit_observed(rec['id'],tag)
 save(OUT/f'{a.fit_dir}_run_{a.split}.json',allstats)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
