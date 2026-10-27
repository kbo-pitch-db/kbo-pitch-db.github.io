"""Official GVHMR learned raw parameters without its SMPL-X-dependent postprocessing.

The official network and checkpoint run unchanged. Official normalizer decoding is
reproduced without constructing a licensed SMPL-X mesh. RTMPose replaces ViTPose,
HMR2.0a features are shared, camera/focal are hypotheses, time is resampled to 30Hz.
SMPL-X raw parameter files and an APPROXIMATE SMPL rotation visualization are
saved separately. This is not the complete official GVHMR demo or native mesh.
"""
from common import *
import argparse,importlib.util
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation,Slerp

def r6d(x):
 import torch
 a,b=x[...,:3],x[...,3:];a=torch.nn.functional.normalize(a,dim=-1);b=b-(a*b).sum(-1,keepdim=True)*a;b=torch.nn.functional.normalize(b,dim=-1)
 return torch.stack([a,b,torch.linalg.cross(a,b,dim=-1)],dim=-2)

def main():
 import torch
 torch.set_num_threads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 root=WORKSPACE/'external/GVHMR';sys.path.insert(0,str(root))
 from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE
 from hmr4d.model.gvhmr.utils.stats_compose import MM_V1_AMASS_LOCAL_BEDLAM_CAM
 weight=WORKSPACE/'models/gvhmr_siga24_release.ckpt'
 assert sha(weight)=='4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91'
 checkpoint=torch.load(weight,map_location='cpu');state=checkpoint.get('state_dict',checkpoint)
 model=NetworkEncoderRoPE().cuda(0).eval();prefix='pipeline.denoiser3d.'
 network_state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
 model.load_state_dict(network_state,strict=True);del checkpoint,state,network_state
 mean=torch.tensor(MM_V1_AMASS_LOCAL_BEDLAM_CAM['mean'],device='cuda:0');std=torch.tensor(MM_V1_AMASS_LOCAL_BEDLAM_CAM['std'],device='cuda:0')
 from scene_fit import SceneFitter
 body=SceneFitter(WORKSPACE/'external/WHAM','cuda:0')
 records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];r,c,s,h,wham=init_inputs(cid);input_path=OUT/'raw/hmr2a'/f'{cid}.npz';image=load(input_path);t=h['source_time']-h['source_time'][0]
  newt=np.linspace(0,t[-1],max(2,int(round(t[-1]*30))+1));interp=lambda x:interp1d(t,x,axis=0,bounds_error=False,fill_value='extrapolate')(newt).astype(np.float32)
  box=interp(image['bbx_xys']);feat=interp(image['features']);xy=interp(np.nan_to_num(h['keypoints'][:,:17]));confidence=interp(scores(h)[:,:17]);outside=((xy<box[:,:2,None].transpose(0,2,1)-box[:,None,2:]*.5)|(xy>box[:,:2,None].transpose(0,2,1)+box[:,None,2:]*.5)).any(-1)
  confidence[outside]=0;obs=np.concatenate([2*(xy-box[:,None,:2])/box[:,None,2:],confidence[...,None]],-1)
  for factor in [1.,4.,12.]:
   tag=f'gvhmr_f{factor:g}';dest=OUT/'raw'/tag/f'{cid}.npz';start=time.monotonic()
   if dest.exists():print('CACHE',tag,cid,flush=True);continue
   focal=np.hypot(float(s['width']),float(s['height']))*factor
   cliff=box.copy();cliff[:,:2]-=[float(s['width'])/2,float(s['height'])/2];cliff/=focal
   with torch.no_grad():
    tensor=lambda x:torch.as_tensor(x,device='cuda:0',dtype=torch.float32)[None]
    pred=model(length=torch.tensor([len(newt)],device='cuda:0'),obs=tensor(obs),f_cliffcam=tensor(cliff),f_cam_angvel=tensor(np.zeros((len(newt),6),np.float32)),f_imgseq=tensor(feat))
    decoded=(pred['pred_x']*std+mean)[0]
    rotations=torch.cat([r6d(decoded[:,136:142])[:,None],r6d(decoded[:,:126].reshape(-1,21,6))],1).cpu().numpy()
    beta=decoded[:,126:136].cpu().numpy();gv_rotation=r6d(decoded[:,142:148]).cpu().numpy();velocity=decoded[:,148:151].cpu().numpy();cam=pred['pred_cam'][0].cpu().numpy();contact=torch.sigmoid(pred['static_conf_logits'][0]).cpu().numpy()
   original_rot=np.stack([Slerp(newt,Rotation.from_matrix(rotations[:,j]))(t).as_matrix() for j in range(22)],1).astype(np.float32)
   back=lambda x:interp1d(newt,x,axis=0,bounds_error=False,fill_value='extrapolate')(t).astype(np.float32)
   atomic_npz(dest,smplx_rotations=original_rot,smplx_betas=back(beta),smplx_global_gravity_rotation=Slerp(newt,Rotation.from_matrix(gv_rotation))(t).as_matrix().astype(np.float32),local_translation_velocity_30hz=velocity,
    weak_camera=back(cam),static_joint_hypotheses=back(contact),source_time=h['source_time'],frame_index=h['frame_index'],network_time=newt,
    model_sha256=sha(weight),representation='Official learned SMPL-X 22 shared body rotations; raw normalized-network decode; no native mesh/postprocessing')
   # Do not pass SMPL-X betas into SMPL. Use separate WHAM-derived SMPL shape only for a clearly labeled adapter.
   smpl_rot=np.concatenate([original_rot,np.tile(np.eye(3,dtype=np.float32),(len(t),2,1,1))],1)
   smpl_beta=np.nanmedian(wham['smpl_betas'],axis=0);xyz=[];native=[];feet=[]
   with torch.no_grad():
    for i in range(0,len(t),32):
     b,j,f=body.model(torch.tensor(smpl_rot[i:i+32],device='cuda:0'),torch.tensor(np.repeat(smpl_beta[None],min(32,len(t)-i),0),device='cuda:0'))
     xyz.append(b.cpu().numpy());native.append(j.cpu().numpy());feet.append(f.cpu().numpy())
   adapter=OUT/'adapters'/tag/f'{cid}.npz'
   atomic_npz(adapter,smpl_rotations=smpl_rot,smpl_betas=np.repeat(smpl_beta[None],len(t),0),keypoints=np.concatenate(xyz),anatomical_keypoints=np.concatenate(native),foot_keypoints=np.concatenate(feet),
    source_time=h['source_time'],frame_index=h['frame_index'],fps=h['fps'],focal_factor=factor,
    coordinate_system='Camera x right, y down, z away; hip-relative',representation='Approximate SMPL-X shared-rotation-to-SMPL adapter, not native GVHMR mesh')
   save(dest.with_suffix('.json'),dict(id=cid,seconds=time.monotonic()-start,strict_network_weights=True,raw_parameters=True,native_mesh_evaluated=False,licensed_smplx_available=False,postprocessing=False,
    modifications=['Final RTMPose input instead of ViTPose','HMR2.0a feature cache','30 Hz resampling preserving original output timestamps','Identity camera-angular-velocity hypothesis','No SMPL-X-dependent FK, IK or contact postprocess'],
    focal_factor=factor,weight_sha256=sha(weight),input_feature_sha256=sha(input_path),adapter=str(adapter.relative_to(OUT)),v2_used=False))
   print('RAW GVHMR NETWORK',cid,tag,round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
