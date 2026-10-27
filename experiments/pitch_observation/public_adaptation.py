"""Small supervised public-pitching adapter on frozen official MotionBERT features.

OBP joint centers are actual public mocap, NOT KBO GT. Athlete-disjoint splits,
synthetic perspective cameras, blur-like 2D noise/dropouts, no KBO pseudo-labels.
All evaluation errors are in ONE SEQUENCE-WIDE leg unit (oracle public scale),
not millimeters, and cannot be quoted as KBO 3D accuracy. No stride target loss.
The fitted adapter is a custom experiment, not PitcherNet reproduction.
"""
from common import *
import argparse,zipfile,copy
from scipy.spatial.transform import Rotation
CMB=[4,1,5,2,6,3,11,14,12,15,13,16]
NAMES=['lead_hip','rear_hip','lead_knee_jc','rear_knee_jc','lead_ankle_jc','rear_ankle_jc','glove_shoulder_jc','shoulder_jc','glove_elbow_jc','elbow_jc','glove_wrist_jc','wrist_jc']

def setup_model():
 import torch
 sys.path.insert(0,str(WORKSPACE/'external/MotionBERT'))
 from lib.model.DSTformer import DSTformer
 model=DSTformer(dim_in=3,dim_out=3,dim_feat=512,dim_rep=512,depth=5,num_heads=8,mlp_ratio=2,norm_layer=torch.nn.LayerNorm,maxlen=243,num_joints=17)
 checkpoint=torch.load(WORKSPACE/'models/motionbert_h36m.bin',map_location='cpu');model.load_state_dict({k.replace('module.','',1):v for k,v in checkpoint['model_pos'].items()},strict=True)
 return model.cuda(0).eval().requires_grad_(False)

def norm17(x):
 x=x-x[:,0:1];leg=np.nanmedian((np.linalg.norm(x[:,4]-x[:,5],axis=-1)+np.linalg.norm(x[:,5]-x[:,6],axis=-1)+np.linalg.norm(x[:,1]-x[:,2],axis=-1)+np.linalg.norm(x[:,2]-x[:,3],axis=-1))*.5)
 return x/max(float(leg),1e-6)

def prepare_public():
 import pandas as pd
 archive=OUT/'public_data/pitching_landmarks.zip.part';assert sha(archive)=='efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22'
 with zipfile.ZipFile(archive) as z:table=pd.read_csv(z.open('landmarks.csv'))
 metadata=pd.read_csv(WORKSPACE/'external/openbiomechanics/baseball_pitching/data/metadata.csv');meta=metadata.set_index('session_pitch');subjects=sorted(metadata.user.unique(),key=lambda v:hashlib.sha256(('20261028:'+str(v)).encode()).hexdigest());splits={str(v):'train' if i<24 else 'val' if i<32 else 'public_test' if i<48 else 'unused' for i,v in enumerate(subjects)}
 sequences=[];used=[]
 for pitch,group in table.groupby('session_pitch',sort=False):
  if pitch not in meta.index:continue
  row=meta.loc[pitch];subject=str(row['user']);split=splits[subject]
  if split=='unused' or subject in used:continue
  used.append(subject);p=np.stack([group[[name+'_'+axis for axis in 'xyz']].to_numpy() for name in NAMES],1).astype(np.float32);times=group.time.to_numpy();release=float(group.BR_time.iloc[0]);start=max(float(times[0]),release-1.35);end=min(float(times[-1]),release+.55);t=np.arange(start,end,1/30)
  p=np.stack([np.stack([np.interp(t,times,p[:,j,k]) for k in range(3)],1) for j in range(12)],1);p=p-p[:,:2].mean(1,keepdims=True);leg=np.median(sum(np.linalg.norm(p[:,aa]-p[:,bb],axis=-1) for aa,bb in [(0,2),(2,4),(1,3),(3,5)])*.5);p/=leg
  h36m=np.zeros((len(p),17,3),np.float32);h36m[:,CMB]=p;h36m[:,0]=p[:,:2].mean(1);h36m[:,8]=p[:,6:8].mean(1);h36m[:,7]=(h36m[:,0]+h36m[:,8])*.5
  # Virtual head is an INPUT placeholder only and never enters supervised losses.
  up=h36m[:,8]-h36m[:,0];up/=np.maximum(np.linalg.norm(up,axis=-1,keepdims=True),1e-6);h36m[:,9]=h36m[:,8]+up*.22;h36m[:,10]=h36m[:,8]+up*.32
  sequences.append(dict(subject=subject,pitch=pitch,split=split,positions=h36m,leg_m=float(leg),source_frames=len(group),release_source_time=release))
 save(OUT/'adaptation/public_split.json',dict(athletes=splits,used=[{k:v for k,v in x.items() if k!='positions'} for x in sequences],source_sha256=sha(archive),source_commit='44b98dae05cceb016f080ab39d105c85b8639084',mapping='Canonical lead=left, throwing=right; random anatomical reflection supplies opposite handedness; original handedness is not inferred',license='CC BY-NC-SA4.0 plus professional organization/financial firm exclusion; retained in public_data/LICENSE-DATA.md',virtual_joints='head/nose input placeholders, zero confidence and excluded from supervised/evaluation loss'))
 return sequences

def augment(sequence,rng,view,noise=True):
 from lib.utils.utils_data import crop_scale
 p=sequence['positions'].copy();yaw=float(rng.uniform(-18,18)) if view%4<3 else float(rng.uniform(-110,110));pitch=float(rng.uniform(1,15));R0=np.array([[0,-1,0],[0,0,-1],[1,0,0]])
 R=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@R0@Rotation.from_euler('z',yaw,degrees=True).as_matrix();p=p@R.T
 if view%2:
  p[...,0]*=-1;flip=[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13];p=p[:,flip]
 D=float(rng.uniform(12,100));xy=p[...,:2]/(p[...,2:]+D);conf=np.ones((*xy.shape[:2],1));conf[:,9:11]=0
 inp=np.concatenate([xy,conf],-1).astype(np.float32);inp=crop_scale(inp,[1,1]).astype(np.float32)
 if noise:
  inp[...,:2]+=rng.normal(0,.012,inp[...,:2].shape).astype(np.float32)
  # A coherent short wrist dropout approximates a low-confidence blur interval.
  if view%3==0:
   start=int(rng.integers(0,max(1,len(p)-5)));inp[start:start+5,[12,13,15,16],2]=0
 return inp,p.astype(np.float32),dict(yaw_degrees=yaw,pitch_degrees=pitch,depth_leg=D)

def cache_features(model):
 import torch
 root=OUT/'adaptation';root.mkdir(exist_ok=True);path=root/'public_features.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 rng=np.random.default_rng(20261028);sequences=prepare_public();records=[]
 for sequence in sequences:
  for view in range(8 if sequence['split']=='train' else 4):
   inp,target,camera=augment(sequence,rng,view)
   with torch.no_grad():rep=model(torch.tensor(inp[None],device='cuda:0'),return_rep=True)[0];base=model.head(rep).cpu().numpy();feature=rep.cpu().half()
   records.append(dict(features=feature,target=torch.tensor(target),baseline=torch.tensor(norm17(base)),subject=sequence['subject'],pitch=sequence['pitch'],split=sequence['split'],camera=camera))
  print('PUBLIC FEATURES',sequence['subject'],sequence['split'],len(sequence['positions']),flush=True)
 torch.save(records,path);return records

def make_adapter(model):
 import torch
 # Original pretrained head plus a zero-initialized low-rank residual, 18k params.
 class Adapter(torch.nn.Module):
  def __init__(self):
   super().__init__();self.head=copy.deepcopy(model.head);self.low=torch.nn.Sequential(torch.nn.Linear(512,32),torch.nn.GELU(),torch.nn.Linear(32,3));torch.nn.init.zeros_(self.low[-1].weight);torch.nn.init.zeros_(self.low[-1].bias)
  def forward(self,x):return self.head(x)+self.low(x)
 return Adapter().cuda(0)

def metrics(adapter,data):
 import torch
 out=[]
 with torch.no_grad():
  for rec in data:
   pred=adapter(rec['features'].float().cuda(0)).cpu().numpy();pred=norm17(pred);target=rec['target'].numpy();base=rec['baseline'].numpy()
   for label,v in [('pretrained',base),('adapted',pred)]:
    p=v[:,CMB];g=target[:,CMB];e=np.linalg.norm(p-g,axis=-1)
    # Bone-direction angle in public camera axes, without Procrustes alignment.
    starts=[0,2,1,3,6,8,7,9];ends=[2,4,3,5,8,10,9,11];a=p[:,ends]-p[:,starts];b=g[:,ends]-g[:,starts];a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6);b/=np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-6);ang=np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))
    out.append(dict(subject=rec['subject'],pitch=rec['pitch'],split=rec['split'],camera=rec['camera'],model=label,mean_joint_error_leg=float(e.mean()),depth_error_leg=float(abs(p[...,2]-g[...,2]).mean()),arm_error_leg=float(e[:,8:].mean()),leg_error_leg=float(e[:,:6].mean()),bone_direction_error_degrees=float(ang.mean())))
 return out

def train(args):
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261028);model=setup_model();data=cache_features(model);adapter=make_adapter(model);optimizer=torch.optim.Adam(adapter.parameters(),lr=.0005);train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='val'];best=float('inf');log=[];rng=np.random.default_rng(20261028);start=time.monotonic();root=OUT/'adaptation'
 # First adapt the shared output head and low-rank branch on public subjects only.
 for epoch in range(args.epochs):
  adapter.train();losses=[]
  for batch in np.array_split(rng.permutation(len(train)),max(1,len(train)//6)):
   optimizer.zero_grad();loss=0
   for index in batch:
    rec=train[int(index)];x=rec['features'].float().cuda(0);target=rec['target'].cuda(0);pred=adapter(x);pred=pred-pred[:,0:1]
    loss=loss+(pred[:,CMB]-target[:,CMB]).abs().mean()/len(batch)
   loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
  adapter.eval();v=metrics(adapter,val);score=np.mean([r['mean_joint_error_leg'] for r in v if r['model']=='adapted']);log.append(dict(epoch=epoch,train_l1=float(np.mean(losses)),val_error_leg=float(score),seconds=time.monotonic()-start))
  if score<best:
   best=score;torch.save(dict(state_dict=adapter.state_dict(),epoch=epoch,val_error_leg=score,base_sha256=sha(WORKSPACE/'models/motionbert_h36m.bin'),seed=20261028,train_athletes=sorted(set(r['subject'] for r in train))),root/'best_adapter.pt')
  torch.save(dict(state_dict=adapter.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch),root/'last_adapter.pt');save(root/'training_log.json',log)
  if epoch%5==0:print('PUBLIC ADAPTER EPOCH',epoch,round(float(np.mean(losses)),4),round(float(score),4),round(time.monotonic()-start,1),flush=True)
 adapter.load_state_dict(torch.load(root/'best_adapter.pt')['state_dict']);result=metrics(adapter,[r for r in data if r['split']!='train']);save(root/'public_evaluation.json',result);csvwrite(root/'public_evaluation.csv',result);save(root/'training_provenance.json',dict(epochs=args.epochs,seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in adapter.parameters()),frozen_backbone_parameters=sum(p.numel() for p in model.parameters()),public_subjects=len(set(r['subject'] for r in data)),kbo_labels_used=False,kbo_test_used_for_training=False,output='Custom frozen-MotionBERT-head/low-rank public-pitching adapter; no absolute scale/global motion',human_review=False))

def infer_kbo(args):
 import torch
 from lib.data.dataset_wild import halpe2h36m
 from lib.utils.utils_data import crop_scale,flip_data
 model=setup_model();adapter=make_adapter(model);checkpoint=torch.load(OUT/'adaptation/best_adapter.pt',map_location='cpu');adapter.load_state_dict(checkpoint['state_dict']);adapter.eval();weight_hash=sha(OUT/'adaptation/best_adapter.pt')
 for rec in sample(args.split,not args.full_sample):
  cid=rec['id'];dest=OUT/'raw/mb_obp_adapter'/f'{cid}.npz'
  if dest.exists():continue
  r,c,s,h,w=init_inputs(cid);sc=scores(h);inp=np.concatenate([h['keypoints'].copy(),sc[...,None]],-1)
  for j in range(26):
   good=np.isfinite(inp[:,j,:2]).all(-1)&(sc[:,j]>.0)
   if good.sum()<2:inp[:,j,:2]=0;continue
   for k in range(2):inp[:,j,k]=np.interp(np.arange(len(inp)),np.flatnonzero(good),inp[good,j,k])
  mapped=halpe2h36m(inp).astype(np.float32);normalized=crop_scale(mapped,[1,1]).astype(np.float32);outputs=[]
  edges=np.flatnonzero((np.diff(h['frame_index'])!=1)|(np.diff(h['track_id'])!=0))+1
  for segment in np.split(np.arange(len(inp)),edges):
   for start in range(0,len(segment),243):
    ids=segment[start:start+243];x=torch.tensor(normalized[ids][None],device='cuda:0')
    with torch.no_grad():
     p=adapter(model(x,return_rep=True));f=flip_data(adapter(model(flip_data(x),return_rep=True)));pred=((p+f)*.5)[0].cpu().numpy();pred-=pred[:,0:1]
    outputs.append(pred)
  xyz=np.concatenate(outputs);atomic_npz(dest,keypoints=xyz,source_time=h['source_time'],frame_index=h['frame_index'],confidence=mapped[...,2],model_sha256=weight_hash,coordinate_system='Native H36M17 root-relative; learned public-pitching leg-normalized scale; no global trajectory or absolute distance')
  print('KBO PUBLIC ADAPTER',cid,flush=True)

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','infer'],default='train');ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args()
 if a.mode=='train':train(a)
 else:
  sys.path.insert(0,str(WORKSPACE/'external/MotionBERT'));infer_kbo(a)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
