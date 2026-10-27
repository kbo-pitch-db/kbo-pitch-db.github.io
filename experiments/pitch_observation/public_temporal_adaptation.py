"""Small last-block MotionBERT fine-tuning on actual public OBP joint centers.

24 training athletes, 8 validation athletes. A NEW set of 16 public athletes
(split ranks 48..63) is reserved for confirmation, as the earlier 16-subject
benchmark has already been inspected. KBO final24 remains untouched.
Only the final spatial/temporal block pair, fusion, norm and output layers
are trained. Earlier four block pairs are frozen and their activations cached.
No KBO pseudo-labels, no target stride, no RGB/depth-fusion reproduction claim.
"""
from common import *
from public_adaptation import setup_model,make_adapter,CMB,NAMES,norm17
from scipy.spatial.transform import Rotation
import argparse,zipfile,copy

def sequences():
 import pandas as pd
 archive=OUT/'public_data/pitching_landmarks.zip.part';assert sha(archive)=='efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22'
 with zipfile.ZipFile(archive) as z:table=pd.read_csv(z.open('landmarks.csv'))
 meta=pd.read_csv(WORKSPACE/'external/openbiomechanics/baseball_pitching/data/metadata.csv').set_index('session_pitch');subjects=sorted(meta.user.unique(),key=lambda v:hashlib.sha256(('20261028:'+str(v)).encode()).hexdigest());assign={str(v):'train' if i<24 else 'val' if i<32 else 'previous_public_test' if i<48 else 'confirmation' if i<64 else 'unused' for i,v in enumerate(subjects)};records=[];used=set()
 for pitch,group in table.groupby('session_pitch',sort=False):
  if pitch not in meta.index:continue
  subject=str(int(meta.loc[pitch,'user']));split=assign[subject]
  if split in ['unused','previous_public_test'] or subject in used:continue
  used.add(subject);p=np.stack([group[[name+'_'+axis for axis in 'xyz']].to_numpy() for name in NAMES],1).astype(np.float32);times=group.time.to_numpy();release=float(group.BR_time.iloc[0]);start=max(float(times[0]),release-1.35);end=min(float(times[-1]),release+.55);p=p-p[:,:2].mean(1,keepdims=True);leg=np.median(sum(np.linalg.norm(p[:,aa]-p[:,bb],axis=-1) for aa,bb in [(0,2),(2,4),(1,3),(3,5)])*.5);p/=leg
  records.append(dict(subject=subject,pitch=pitch,split=split,positions=p,times=times,start=start,end=end))
 save(OUT/'temporal_adaptation/public_split.json',dict(assignments=assign,used=[{k:r[k] for k in ['subject','pitch','split']} for r in records],previous_test_excluded=True,kbo_labels_used=False))
 return records

def make_view(rec,view):
 from lib.utils.utils_data import crop_scale
 rng=np.random.default_rng(int(hashlib.sha256((rec['subject']+':'+str(view)+':20261028v2').encode()).hexdigest()[:8],16));fps=[24,30,60][view%3];times=np.arange(rec['start'],rec['end'],1/fps);p=np.stack([np.stack([np.interp(times,rec['times'],rec['positions'][:,j,k]) for k in range(3)],1) for j in range(12)],1).astype(np.float32);h=np.zeros((len(p),17,3),np.float32);h[:,CMB]=p;h[:,8]=p[:,6:8].mean(1);h[:,7]=h[:,8]*.5;up=h[:,8].copy();up/=np.maximum(np.linalg.norm(up,axis=-1,keepdims=True),1e-6);h[:,9]=h[:,8]+up*.22;h[:,10]=h[:,8]+up*.32
 yaw=float(rng.uniform(-18,18)) if view%4<3 else float(rng.uniform(-110,110));pitch=float(rng.uniform(1,15));R0=np.array([[0,-1,0],[0,0,-1],[1,0,0]]);R=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@R0@Rotation.from_euler('z',yaw,degrees=True).as_matrix();h=h@R.T
 if view%2:h[...,0]*=-1;h=h[:,[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]]
 D=float(rng.uniform(12,100));xy=h[...,:2]/(h[...,2:]+D);conf=np.ones((*xy.shape[:2],1));conf[:,9:11]=0;inp=crop_scale(np.concatenate([xy,conf],-1).astype(np.float32),[1,1]).astype(np.float32);inp[...,:2]+=rng.normal(0,.012,inp[...,:2].shape).astype(np.float32)
 if view%3==0:
  a=int(rng.integers(0,max(1,len(h)-5)));inp[a:a+5,[12,13,15,16],2]=0
 return inp,h.astype(np.float32),dict(fps=fps,yaw_degrees=yaw,pitch_degrees=pitch,depth_leg=D)

def prefix(model,x):
 B,F,J,C=x.shape;x=model.joints_embed(x.reshape(-1,J,C));x=x+model.pos_embed;x=(x.reshape(B,F,J,-1)+model.temp_embed[:,:F]).reshape(B*F,J,-1);x=model.pos_drop(x)
 for i in range(4):
  a=model.blocks_st[i](x,F);b=model.blocks_ts[i](x,F);alpha=model.ts_attn[i](torch_cat(a,b)).softmax(-1);x=a*alpha[:,:,0:1]+b*alpha[:,:,1:2]
 return x

def torch_cat(a,b):
 import torch
 return torch.cat([a,b],-1)

def suffix(model,x,F):
 a=model.blocks_st[4](x,F);b=model.blocks_ts[4](x,F);alpha=model.ts_attn[4](torch_cat(a,b)).softmax(-1);x=a*alpha[:,:,0:1]+b*alpha[:,:,1:2];return model.pre_logits(model.norm(x).reshape(1,F,17,-1))

def cache(model):
 import torch
 root=OUT/'temporal_adaptation';root.mkdir(exist_ok=True);path=root/'prefix_features.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 head_adapter=make_adapter(model);head_adapter.load_state_dict(torch.load(OUT/'adaptation/best_adapter.pt',map_location='cpu')['state_dict']);head_adapter.eval();records=[]
 for rec in sequences():
  for view in range(8 if rec['split']=='train' else 4):
   inp,target,camera=make_view(rec,view)
   with torch.no_grad():
    x=prefix(model,torch.tensor(inp[None],device='cuda:0'));rep=suffix(model,x,len(inp));base=model.head(rep)[0].cpu().numpy();head=head_adapter(rep)[0].cpu().numpy()
   records.append(dict(features=x.cpu().half(),target=torch.tensor(target),baseline=torch.tensor(norm17(base)),head_adapter=torch.tensor(norm17(head)),subject=rec['subject'],pitch=rec['pitch'],split=rec['split'],camera=camera))
  print('PREFIX CACHE',rec['subject'],rec['split'],flush=True)
 torch.save(records,path);return records

def evaluate(model,records):
 import torch
 out=[]
 with torch.no_grad():
  for rec in records:
   target=rec['target'].numpy();prediction=model.head(suffix(model,rec['features'].float().cuda(0),len(target)))[0].cpu().numpy();prediction=norm17(prediction)
   for label,p in [('pretrained',rec['baseline'].numpy()),('head_adapter',rec['head_adapter'].numpy()),('last_block',prediction)]:
    e=np.linalg.norm(p[:,CMB]-target[:,CMB],axis=-1)
    st=[0,2,1,3,6,8,7,9];en=[2,4,3,5,8,10,9,11];a=p[:,CMB][:,en]-p[:,CMB][:,st];b=target[:,CMB][:,en]-target[:,CMB][:,st];a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6);b/=np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-6);angle=np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))
    speeds=[];delays=[]
    for wrist,shoulder in [(13,11),(16,14)]:
     ov=np.linalg.norm(np.diff(target[:,wrist]-target[:,shoulder],axis=0),axis=-1);pv=np.linalg.norm(np.diff(p[:,wrist]-p[:,shoulder],axis=0),axis=-1);speeds.append(float(pv.max()/max(ov.max(),1e-6)));delays.append(float(abs(np.argmax(pv)-np.argmax(ov))/rec['camera']['fps']))
    out.append(dict(subject=rec['subject'],pitch=rec['pitch'],split=rec['split'],camera=rec['camera'],model=label,mean_joint_error_leg=float(e.mean()),depth_error_leg=float(abs(p[:,CMB,2]-target[:,CMB,2]).mean()),arm_error_leg=float(e[:,8:].mean()),leg_error_leg=float(e[:,:6].mean()),bone_direction_error_degrees=float(angle.mean()),arm_peak_speed_ratios=speeds,arm_peak_absolute_delay_s=delays))
 return out

def train(args):
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261028);model=setup_model();records=cache(model);train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];modules=[model.blocks_st[4],model.blocks_ts[4],model.ts_attn[4],model.norm,model.pre_logits,model.head]
 for module in modules:module.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.00002,weight_decay=.01);rng=np.random.default_rng(20261028);best=float('inf');root=OUT/'temporal_adaptation';log=[];start=time.monotonic();edges=[(4,5),(5,6),(1,2),(2,3),(11,12),(12,13),(14,15),(15,16)]
 for epoch in range(args.epochs):
  model.train();losses=[]
  for batch in np.array_split(rng.permutation(len(train)),max(1,len(train)//4)):
   opt.zero_grad();loss=0
   for index in batch:
    rec=train[int(index)];target=rec['target'].cuda(0);pred=model.head(suffix(model,rec['features'].float().cuda(0),len(target)))[0];pred=pred-pred[:,0:1];l=(pred[:,CMB]-target[:,CMB]).abs().mean();velocity=((pred[1:,CMB]-pred[:-1,CMB])-(target[1:,CMB]-target[:-1,CMB])).abs().mean()*rec['camera']['fps']/30;l=l+.05*velocity
    for a,b in edges:l=l+.005*(torch.linalg.vector_norm(pred[:,a]-pred[:,b],dim=-1)-torch.linalg.vector_norm(target[:,a]-target[:,b],dim=-1)).abs().mean()
    loss=loss+l/len(batch)
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach()))
  model.eval();scores=evaluate(model,val);score=float(np.mean([r['mean_joint_error_leg'] for r in scores if r['model']=='last_block']));log.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_leg_error=score,seconds=time.monotonic()-start));trainable={name:value.detach().cpu() for name,value in model.state_dict().items() if any(name.startswith(prefix) for prefix in ['blocks_st.4.','blocks_ts.4.','ts_attn.4.','norm.','pre_logits.','head.'])}
  torch.save(dict(trainable=trainable,optimizer=opt.state_dict(),epoch=epoch),root/'last.pt')
  if score<best:best=score;torch.save(dict(trainable=trainable,epoch=epoch,val_error_leg=score,base_sha256=sha(WORKSPACE/'models/motionbert_h36m.bin')),root/'best.pt')
  save(root/'training_log.json',log)
  if epoch%2==0:print('LAST BLOCK EPOCH',epoch,round(float(np.mean(losses)),4),round(score,4),round(time.monotonic()-start,1),flush=True)
 # Confirmation data are cached but not evaluated until the --confirm command.
 save(root/'provenance.json',dict(train_athletes=24,val_athletes=8,confirmation_athletes=16,trainable_parameters=sum(p.numel() for p in params),epochs=args.epochs,seconds=time.monotonic()-start,kbo_data_used=False,confirmation_evaluated=False,learning='Actual public joint-center supervision, random 24/30/60Hz camera views, 2D noise and dropout; not RGB fine-tuning or PitcherNet reproduction'))

def infer(args):
 import torch
 from lib.data.dataset_wild import halpe2h36m
 from lib.utils.utils_data import crop_scale,flip_data
 model=setup_model();model.load_state_dict(torch.load(OUT/'temporal_adaptation/best.pt',map_location='cpu')['trainable'],strict=False);model.eval()
 if args.mode=='confirm':
  records=cache(setup_model());results=evaluate(model,[r for r in records if r['split']=='confirmation']);save(OUT/'temporal_adaptation/confirmation.json',results);csvwrite(OUT/'temporal_adaptation/confirmation.csv',results);return
 for rec in sample(args.split,not args.full_sample):
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);sc=scores(h);inp=np.concatenate([h['keypoints'].copy(),sc[...,None]],-1)
  for j in range(26):
   good=np.isfinite(inp[:,j,:2]).all(-1)&(sc[:,j]>.0)
   if good.sum()<2:inp[:,j,:2]=0;continue
   for k in range(2):inp[:,j,k]=np.interp(np.arange(len(inp)),np.flatnonzero(good),inp[good,j,k])
  normalized=crop_scale(halpe2h36m(inp).astype(np.float32),[1,1]).astype(np.float32);outputs=[[],[]];edges=np.flatnonzero((np.diff(h['frame_index'])!=1)|(np.diff(h['track_id'])!=0))+1
  for segment in np.split(np.arange(len(inp)),edges):
   for k in range(0,len(segment),243):
    ids=segment[k:k+243];x=torch.tensor(normalized[ids][None],device='cuda:0')
    with torch.no_grad():preds=[model(x),flip_data(model(flip_data(x)))]
    for i,p in enumerate(preds):p=p[0].cpu().numpy();p-=p[:,0:1];outputs[i].append(p)
  for tag,v in zip(['mb_block_single','mb_block_flip_restored'],outputs):atomic_npz(OUT/'raw'/tag/f'{cid}.npz',keypoints=np.concatenate(v),source_time=h['source_time'],frame_index=h['frame_index'],coordinate_system='Native H36M17 root-relative camera axes; custom last-block public-mocap adaptation; no depth-hypothesis averaging, global trajectory or absolute scale')
  print('KBO LAST BLOCK',cid,flush=True)

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','infer','confirm'],default='train');ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');args=ap.parse_args()
 if args.mode=='train':train(args)
 else:
  sys.path.insert(0,str(WORKSPACE/'external/MotionBERT'));infer(args)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
