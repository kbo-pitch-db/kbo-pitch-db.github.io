"""Resumable population execution without changing the fixed research split.

GPU workers are ordinary isolated processes, not agents. Existing final 2D,
WHAM results, frozen evaluations and six accepted previews are read-only.
"""
from common import *
import argparse,importlib,traceback,gc

ROOT=OUT/'all813_gvhmr_v1'

def population():
 old={r['id']:r for r in json.loads((PREVIOUS/'audit_all.json').read_text())}
 index=json.loads((CODE/'data/clips.json').read_text())
 records=[dict(old[c['id']],split='population',player_name=c['player']) for c in index['clips']]
 return sorted(records,key=lambda r:r['id'])

def feature_complete(cid):
 p=OUT/'raw/hmr2a'/f'{cid}.npz'
 return p.exists() and p.with_suffix('.json').exists()

def network_complete(cid):
 return all((OUT/'raw'/f'gvhmr_f{f}'/f'{cid}.npz').exists() and (OUT/'raw'/f'gvhmr_f{f}'/f'{cid}.json').exists() and (OUT/'adapters'/f'gvhmr_f{f}'/f'{cid}.npz').exists() for f in [1,4,12])

def run_upstream(stage,records,worker):
 import common,torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.38,0)
 complete=feature_complete if stage=='features' else network_complete
 module=importlib.import_module('rgb_initializers' if stage=='features' else 'gvhmr_network')
 failures={};progress=ROOT/f'{stage}_worker{worker}.json';current={}
 for attempt in range(2*len(records)+1):
  pending=[r for r in records if not complete(r['id']) and failures.get(r['id'],{}).get('attempts',0)<2]
  if stage=='network':pending=[r for r in pending if feature_complete(r['id'])]
  if not pending:break
  def selected(split='all',small=True):
   def iterator():
    for rec in pending:
     current.clear();current.update(rec)
     save(progress,dict(stage=stage,worker=worker,current=rec['id'],done=sum(complete(r['id']) for r in records),total=len(records),failures=failures,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
     yield rec
   return iterator()
  # These modules import sample from common when imported. Replace that name
  # locally; do not rewrite subset.json or sample48.json.
  module.sample=selected;sys.argv=[module.__file__]
  current.clear()
  try:module.main()
  except Exception as error:
   if not current:raise
   cid=current['id'];failures[cid]=dict(attempts=failures.get(cid,{}).get('attempts',0)+1,error=str(error),traceback=traceback.format_exc());print('POPULATION FAILURE',stage,cid,traceback.format_exc(),flush=True)
  gc.collect();torch.cuda.empty_cache()
 result=dict(stage=stage,worker=worker,done=sum(complete(r['id']) for r in records),total=len(records),failures=failures,completed_ids=[r['id'] for r in records if complete(r['id'])],unfinished_ids=[r['id'] for r in records if not complete(r['id'])])
 save(progress,result);print(json.dumps(result),flush=True)

def prepare():
 records=population();ROOT.mkdir(exist_ok=True)
 if not (ROOT/'input_manifest.json').exists():
  raw=rows('motion_wham.csv');clips=rows('clips.csv');manifest=[]
  for rec in records:
   cid=rec['id'];r=raw[cid];c=clips[cid]
   manifest.append(dict(id=cid,final_2d_path=r['pose_2d_refined'],final_2d_sha256=sha(DATA/r['pose_2d_refined']),wham_prior_sha256=sha(DATA/r['prior_3d']),crop_video=c['crop_video'],frames=int(c['frames'])))
  save(ROOT/'input_manifest.json',dict(population=records,inputs=manifest,canonical_index_sha256=sha(CODE/'data/clips.json'),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),original_pose_inputs_changed=False))
 print('PREPARED',len(records),'clips')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','features','network']);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--id');args=ap.parse_args();ROOT.mkdir(exist_ok=True)
 if args.stage=='prepare':prepare();return
 records=population()[args.worker::args.workers]
 if args.id:records=[r for r in population() if r['id']==args.id]
 if args.limit:records=records[:args.limit]
 run_upstream(args.stage,records,args.worker)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
