"""Actual separate original/flip-restored MotionBERT hypotheses; no depth averaging."""
from common import *
from public_adaptation import setup_model,make_adapter
import argparse

def main():
 import torch
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();model=setup_model();adapter=make_adapter(model);adapter.load_state_dict(torch.load(OUT/'adaptation/best_adapter.pt',map_location='cpu')['state_dict']);adapter.eval()
 from lib.data.dataset_wild import halpe2h36m
 from lib.utils.utils_data import crop_scale,flip_data
 tags=['mb_single','mb_flip_restored','mb_obp_single','mb_obp_flip_restored'];results=[]
 for rec in sample(a.split,not a.full_sample):
  cid=rec['id'];r,c,s,h,w=init_inputs(cid)
  if all((OUT/'raw'/tag/f'{cid}.npz').exists() for tag in tags):continue
  sc=scores(h);inp=np.concatenate([h['keypoints'].copy(),sc[...,None]],-1)
  for j in range(26):
   good=np.isfinite(inp[:,j,:2]).all(-1)&(sc[:,j]>.0)
   if good.sum()<2:inp[:,j,:2]=0;continue
   for k in range(2):inp[:,j,k]=np.interp(np.arange(len(inp)),np.flatnonzero(good),inp[good,j,k])
  mapped=halpe2h36m(inp).astype(np.float32);normalized=crop_scale(mapped,[1,1]).astype(np.float32);outputs={tag:[] for tag in tags};edges=np.flatnonzero((np.diff(h['frame_index'])!=1)|(np.diff(h['track_id'])!=0))+1;start=time.monotonic()
  for segment in np.split(np.arange(len(inp)),edges):
   for k in range(0,len(segment),243):
    ids=segment[k:k+243];x=torch.tensor(normalized[ids][None],device='cuda:0')
    with torch.no_grad():
     rep=model(x,return_rep=True);fliprep=model(flip_data(x),return_rep=True);preds=[model.head(rep),flip_data(model.head(fliprep)),adapter(rep),flip_data(adapter(fliprep))]
    for tag,p in zip(tags,preds):
     p=p[0].cpu().numpy();p-=p[:,0:1];outputs[tag].append(p)
  outputs={tag:np.concatenate(v) for tag,v in outputs.items()}
  for tag,v in outputs.items():atomic_npz(OUT/'raw'/tag/f'{cid}.npz',keypoints=v,confidence=mapped[...,2],source_time=h['source_time'],frame_index=h['frame_index'],coordinate_system='Native H36M17 hip-relative camera axes, no absolute scale/global motion; original/flip-restored hypotheses explicitly unmerged',model='Official MotionBERT backbone + custom public adapter' if 'obp' in tag else 'Official MotionBERT pretrained',flip_hypothesis=tag)
  previous=load(PREVIOUS/'motionbert'/f'{cid}.npz')['keypoints'];avg=(outputs['mb_single']+outputs['mb_flip_restored'])*.5;d=outputs['mb_single']-outputs['mb_flip_restored'];leg=np.median((np.linalg.norm(avg[:,4]-avg[:,5],axis=-1)+np.linalg.norm(avg[:,5]-avg[:,6],axis=-1)+np.linalg.norm(avg[:,1]-avg[:,2],axis=-1)+np.linalg.norm(avg[:,2]-avg[:,3],axis=-1))*.5)
  results.append(dict(id=cid,seconds=time.monotonic()-start,archived_default_reproduction_max=float(np.nanmax(abs(avg-previous))),single_flip_disagreement_p50_leg=float(np.median(np.linalg.norm(d,axis=-1))/leg),single_flip_depth_disagreement_p95_leg=float(np.percentile(abs(d[...,2]),95)/leg)))
  print('UNMERGED MB',cid,round(time.monotonic()-start,2),round(results[-1]['single_flip_depth_disagreement_p95_leg'],3),flush=True)
 save(OUT/f'motionbert_flip_{a.split}.json',results);csvwrite(OUT/f'motionbert_flip_{a.split}.csv',results)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
