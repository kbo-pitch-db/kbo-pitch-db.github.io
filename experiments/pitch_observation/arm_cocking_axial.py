"""Anatomy-limited MB forearm-plane hypotheses, distinct from free arm grafts.

Rotate the humerus about its current shoulder-elbow axis, retaining elbow
position and allowing only a small additional elbow-hinge correction. The
17-joint input does not observe palm/wrist flexion. This is a custom method,
not an official MotionBERT postprocessor or measured external rotation.
"""
from common import *
from hybrid_initializer import global_rotations
from scipy.spatial.transform import Rotation

CONFIG=dict(steps=350,variants=['mb_axial_source_refit','mb_axial_phase_002','mb_axial_phase_010','mb_flip_axial_phase_002','gv_axial_source_refit'],sources=['single','single','single','flip','gv'],depth_weights=[0.,.02,.10,.02,0.],maximum_additional_elbow_hinge_radians=.35,shoulder='Only axial rotation about released humeral direction',outside_phase='Released shoulder/elbow unchanged',wrist_local_rotation='Unchanged; not observed by MotionBERT17')
LABELS=CONFIG['variants']

def forearm_twist(upper,forearm,target):
 u=upper/np.maximum(np.linalg.norm(upper,axis=-1,keepdims=True),1e-8)
 f=forearm-np.sum(forearm*u,-1,keepdims=True)*u;t=target-np.sum(target*u,-1,keepdims=True)*u
 good=(np.linalg.norm(f,axis=-1)>.03*np.linalg.norm(forearm,axis=-1))&(np.linalg.norm(t,axis=-1)>.03*np.linalg.norm(target,axis=-1))
 angle=np.arctan2(np.sum(u*np.cross(f,t),-1),np.sum(f*t,-1));return np.where(good,angle,0.),good

def axial_graft(rot,upper,theta,side):
 g=global_rotations(rot);u=upper/np.maximum(np.linalg.norm(upper,axis=-1,keepdims=True),1e-8);r=rot.copy();r[:,16+side]=np.swapaxes(g[:,13+side],-1,-2)@Rotation.from_rotvec(u*theta[:,None]).as_matrix()@g[:,16+side];return r

def fit_axial(base,mbs,bg,side,body,h,gate):
 import torch
 from pitch_study_fit import stable_rotation
 from stride_review import native_leg
 from arm_cocking_refine import candidate_geometry
 T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0');r=base['smpl_rotations'];N=len(r);H=len(LABELS);beta=base['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;betas=T(np.repeat(beta[None],H*N,0));leg=native_leg(base['anatomical_keypoints']);shoulder=16+side;elbow=18+side;wrist=20+side;an=base['anatomical_keypoints'];upper=an[:,elbow]-an[:,shoulder];forearm=an[:,wrist]-an[:,elbow];u=upper/np.maximum(np.linalg.norm(upper,axis=-1,keepdims=True),1e-8);g=global_rotations(r);roll=Rotation.from_euler('z',np.arctan2(bg[:,1,0],bg[:,0,0])).as_matrix();angles={};validity={};raw={}
 with torch.no_grad():
  for source,m in mbs.items():
   _,ma,_=body.model(T(m['smpl_rotations']),T(np.repeat(beta[None],N,0)));ma=ma.cpu().numpy();target=np.einsum('tij,tj->ti',roll,ma[:,wrist]-ma[:,elbow]);angles[source],validity[source]=forearm_twist(upper,forearm,target);rr=axial_graft(r,upper,angles[source]*gate,side);p,a,f=body.model(T(rr),T(np.repeat(beta[None],N,0)));raw['raw_axial_'+source]=candidate_geometry(base,rr,p.cpu().numpy(),a.cpu().numpy(),f.cpu().numpy())
 angles['gv']=np.zeros(N);validity['gv']=np.ones(N,dtype=bool)
 start=T(np.arctanh(np.clip(np.stack([angles[s] for s in CONFIG['sources']])/np.pi,-.995,.995)));delta=torch.nn.Parameter(torch.zeros((H,N,2),device='cuda:0'));optimizer=torch.optim.Adam([delta],lr=.025);r0=T(np.repeat(r[None],H,0));gvshoulder=T(g[:,shoulder]);parent_inv=T(np.swapaxes(g[:,13+side],-1,-2));u=T(u);local_hinge=np.einsum('tij,tj->ti',np.swapaxes(g[:,shoulder],-1,-2),np.cross(upper,forearm));local_hinge/=np.maximum(np.linalg.norm(local_hinge,axis=-1,keepdims=True),1e-8);hinge=T(local_hinge)
 gate_t=T(gate);good=T(np.stack([validity[s] for s in CONFIG['sources']]));target_theta=T(np.stack([angles[s] for s in CONFIG['sources']]));weight=scores(h)[:,:17]**2;weight[np.arange(N)%5==2]=0;weights=T(weight[:,[7+side,9+side]]);xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];target=T(np.nan_to_num(xy[:,[7+side,9+side]]));height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));pelvis=T(an[:,0]+base['translation']);principal=T(base['principal']);strength=T(CONFIG['depth_weights']);conf=T(np.minimum(scores(h)[:,7+side],scores(h)[:,9+side]));dt=T(np.diff(h['source_time']));links=(np.diff(h['source_time'])>0)&(np.diff(h['source_time'])<.1)&(np.diff(h['frame_index'])==1);triples=T((links[1:]&links[:-1]).astype(float));mid=(dt[1:]+dt[:-1])/2
 def evaluate():
  theta=np.pi*torch.tanh(start+delta[...,0])*gate_t;phi=.35*torch.tanh(delta[...,1])*gate_t;sr=parent_inv[None]@stable_rotation(u[None]*theta[...,None])@gvshoulder[None];er=stable_rotation(hinge[None]*phi[...,None])@r0[:,:,elbow];rot=torch.cat([r0[:,:,:shoulder],sr[:,:,None],r0[:,:,shoulder+1:elbow],er[:,:,None],r0[:,:,elbow+1:]],2);p,a,f=body.model(rot.reshape(-1,24,3,3),betas);p=p.reshape(H,N,17,3);a=a.reshape(H,N,24,3);f=f.reshape(H,N,6,3);tr=pelvis[None]-a[:,:,0];world=p+tr[:,:,None];uv=world[...,:2]/world[...,2:].clamp_min(.2)*float(base['focal'])+principal;err=(uv[:,:,[7+side,9+side]]-target)/height;image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*weights).sum((1,2))/weights.sum().clamp_min(1)
  angular=1-torch.cos(theta-target_theta);prior=(angular*gate_t*good*conf).mean(1);forearm=(a[:,:,wrist]-a[:,:,elbow])/leg;v=(forearm[:,1:]-forearm[:,:-1])/dt[None,:,None];acc=(v[:,1:]-v[:,:-1])/mid[None,:,None]/30;smooth=(acc.square().mean(-1)*triples*(1-.95*gate_t[1:-1])).mean(1);loss=image+strength*prior+.0005*phi.square().mean(1)+.00001*smooth
  return loss,(p,a,f,rot,theta,phi)
 for step in range(CONFIG['steps']):
  optimizer.param_groups[0]['lr']=.025*(.05+.95*(1-step/(CONFIG['steps']-1))**2);optimizer.zero_grad(set_to_none=True);loss,_=evaluate()
  if not torch.isfinite(loss).all():raise ValueError('Nonfinite axial arm fit')
  loss.sum().backward();torch.nn.utils.clip_grad_norm_([delta],10);optimizer.step()
 with torch.no_grad():loss,data=evaluate()
 p,a,f,rr,theta,phi=[x.cpu().numpy() for x in data];outputs=dict(raw)
 for i,name in enumerate(LABELS):outputs[name]=candidate_geometry(base,rr[i],p[i],a[i],f[i])
 diagnostics=dict(config=CONFIG,degenerate_forearm_plane_frames={s:int((~v).sum()) for s,v in validity.items()},raw_twist_peak_radians={s:float(np.max(abs(v*gate))) for s,v in angles.items()},additional_hinge_peak_radians={name:float(abs(phi[i]).max()) for i,name in enumerate(LABELS)},native_elbow_position_change_max_model=max(float(abs((z['anatomical_keypoints'][:,elbow]+z['translation'])-(an[:,elbow]+base['translation'])).max()) for z in outputs.values()),interpretation='Humeral axial orientation hypothesis, not measured external rotation')
 return outputs,diagnostics
