"""Source-conditioned arm-depth hypotheses at fixed body/camera/bone lengths.

Elbow and wrist lie on image rays intersecting fixed-length limb spheres.
Keep both depth roots. MB supplies a weak relative-depth branch prior, never
its misaligned image coordinates. PCHIP interpolates SOURCE RESIDUALS to the
released prediction only at excluded/missing observations, not the motion.
"""
from common import *
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from arm_cocking_native import native_graft
from arm_cocking_refine import phase_window
from stride_review import native_leg

CONFIG=dict(version='source-ray-depth-v3',source_residual_interpolation='PCHIP of observed-minus-predicted 2D residual, not pose smoothing',withhold_every_fifth=True,both_depth_roots_retained=True,depth_prior_weight=.05,velocity_weight=1.,acceleration_weight=.05,root_shape_camera_frozen=True,bone_length='Released sequence-fixed native arm lengths, never stretched',native_mb_role='Relative depth inclination/branch only; not MB image location or metric trajectory',single_true_depth_identified=False)

def project(p,f,principal):return p[...,:2]/p[...,2:]*f+principal

def sphere_on_ray(center,uv,length,focal,principal):
 ray=np.r_[(np.asarray(uv)-principal)/focal,1.];q=ray/np.linalg.norm(ray);dist=np.linalg.norm(center);along=float(center@q);perp2=max(0.,float(center@center-along*along));clamped=perp2>length*length
 if clamped:
  # Closest feasible angular ray on this fixed-length sphere. No bone scale
  # change or arbitrary point displacement; expose the projection residual.
  axis=center/dist;perp=q-(q@axis)*axis;perp/=max(np.linalg.norm(perp),1e-12);theta=np.arcsin(np.clip(length/dist,0,1));q=np.cos(theta)*axis+np.sin(theta)*perp;along=float(center@q);perp2=max(0.,float(center@center-along*along))
 half=np.sqrt(max(0.,length*length-perp2));return np.stack([(along-half)*q,(along+half)*q]),clamped

def depth_path(Q,node,times,gate,leg):
 # Exact second-order dynamic program over four discrete hypotheses. This
 # selects whole poses; it never averages coordinates from different roots.
 n,k=Q.shape[:2];q=Q/leg;dt=np.diff(times)*30
 dp=np.empty((k,k));backs=[]
 for i in range(k):
  for j in range(k):dp[i,j]=node[0,i]+node[1,j]+CONFIG['velocity_weight']*np.mean(((q[1,j]-q[0,i])/dt[0])**2)*(1-.8*gate[1])
 for t in range(2,n):
  nxt=np.empty_like(dp);back=np.empty((k,k),int)
  for j in range(k):
   for kk in range(k):
    velocity=(q[t,kk]-q[t-1,j])/dt[t-1];last=(q[t-1,j]-q[t-2])/dt[t-2];acc=velocity[None]-last;cost=dp[:,j]+CONFIG['velocity_weight']*np.mean(velocity**2)*(1-.8*gate[t])+CONFIG['acceleration_weight']*np.mean(acc**2,axis=(1,2))*(1-.8*gate[t]);best=int(np.argmin(cost));nxt[j,kk]=cost[best]+node[t,kk];back[j,kk]=best
  dp=nxt;backs.append(back)
 path=np.zeros(n,int);path[-2],path[-1]=np.unravel_index(np.argmin(dp),dp.shape)
 for t in range(n-1,1,-1):path[t-2]=backs[t-2][path[t-1],path[t]]
 return path

def generate(base,h,phase,side,natives):
 s=16+side;e=18+side;w=20+side;ce=7+side;cw=9+side;N=len(h['source_time']);f=float(base['focal']);principal=base['principal'];a=base['anatomical_keypoints'].astype(float)+base['translation'][:,None];p=base['keypoints'].astype(float)+base['translation'][:,None];S=a[:,s];L1=float(np.median(np.linalg.norm(a[:,e]-S,axis=-1)));L2=float(np.median(np.linalg.norm(a[:,w]-a[:,e],axis=-1)));leg=native_leg(base['anatomical_keypoints']);bg=base['background_transforms'];obs=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2];sc=scores(h);mask=np.arange(N)%5==2;target=[];support=[];t=h['source_time']
 for j,cj in [(e,ce),(w,cw)]:
  uv0=project(p[:,cj],f,principal);native0=project(a[:,j],f,principal);valid=(sc[:,cj]>.5)&~mask&np.isfinite(obs[:,cj]).all(-1);ix=np.flatnonzero(valid);residual=np.zeros((N,2))
  if len(ix)>=2:
   raw=obs[ix,cj]-uv0[ix];query=np.clip(t,t[ix[0]],t[ix[-1]]);residual=PchipInterpolator(t[ix],raw,axis=0)(query);residual[ix]=raw
  target.append(native0+residual);support.append(int(len(ix)))
 target=np.stack(target,1);Q=np.empty((N,4,2,3));clamp=np.zeros((N,4,2),bool)
 for i in range(N):
  roots,ce_clamp=sphere_on_ray(S[i],target[i,0],L1,f,principal)
  for ei,E in enumerate(roots):
   wrists,cw_clamp=sphere_on_ray(E,target[i,1],L2,f,principal)
   for wi,W in enumerate(wrists):k=ei*2+wi;Q[i,k]=[E-S[i],W-S[i]];clamp[i,k]=[ce_clamp,cw_clamp]
 gate,_=phase_window(h,phase);gvu=(a[:,e]-S)/L1;gvf=(a[:,w]-a[:,e])/L2;depths=np.stack([Q[:,:,0,2]/L1,(Q[:,:,1,2]-Q[:,:,0,2])/L2],-1);gvtarget=np.stack([gvu[:,2],gvf[:,2]],-1);paths={'gv':depth_path(Q,CONFIG['depth_prior_weight']*((depths-gvtarget[:,None])**2).sum(-1),t,gate,leg)};roll=Rotation.from_euler('z',np.arctan2(bg[:,1,0],bg[:,0,0])).as_matrix();ms,me,mw=[11,12,13] if side==0 else [14,15,16]
 for name,native in natives.items():
  raw=native['keypoints'];u=np.einsum('tij,tj->ti',roll,raw[:,me]-raw[:,ms]);v=np.einsum('tij,tj->ti',roll,raw[:,mw]-raw[:,me]);u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-8);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8);d=np.stack([u[:,2],v[:,2]],-1);mbcost=((depths-d[:,None])**2).sum(-1);gvcost=((depths-gvtarget[:,None])**2).sum(-1);node=CONFIG['depth_prior_weight']*(gate[:,None]*mbcost+(1-gate[:,None])*gvcost);paths[name]=depth_path(Q,node,t,gate,leg)
 for i in range(4):paths['branch'+str(i)]=np.full(N,i,int)
 rotations={};details={};identity=np.tile(np.eye(3),(N,1,1))
 for name,path in paths.items():
  q=Q[np.arange(N),path];world=np.zeros((N,17,3));world[:,ms]=S;world[:,me]=S+q[:,0];world[:,mw]=S+q[:,1];r,adapter=native_graft(base,dict(keypoints=world,frame_index=base['frame_index'],source_time=base['source_time']),identity,side);rotations[name]=r;details[name]=dict(path=path.tolist(),ray_clamped_frames=int(clamp[np.arange(N),path].any(-1).sum()),adapter=adapter)
 diagnostics=dict(config=CONFIG,observations_used=support,excluded_direct_observation_frames=np.flatnonzero(mask).tolist(),candidate_count=4,paths=details,ray_clamped_frame_count=int(clamp.any((1,2)).sum()),bone_lengths=[L1,L2],bone_length_scope='Fixed released model, not measured person scale',not_gt=True)
 return rotations,diagnostics
