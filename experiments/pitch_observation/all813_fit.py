"""Population f4 fitting and the frozen v13 contact objective, checkpointed.

Optional CUDA graphs change execution overhead, not the objective or budget.
Unknown contacts stay zero-weight. No WHAM pose substitutes GVHMR output.
"""
from common import *
from all813_release import ROOT,population,network_complete
import argparse,shutil,traceback,gc

def background(cid):
 dest=ROOT/'background'/f'{cid}.npz'
 if dest.exists():return
 previous=PREVIOUS/'background'/f'{cid}.npz'
 if not previous.exists():
  previous=next((ROOT/f'background_worker{w}'/'background'/f'{cid}.npz' for w in [0,1] if (ROOT/f'background_worker{w}'/'background'/f'{cid}.json').exists()),previous)
 if previous.exists():
  dest.parent.mkdir(exist_ok=True);shutil.copy2(previous,dest)
  save(dest.with_suffix('.json'),dict(origin='Existing full-frame, person/graphics-masked research stabilization',source_sha256=sha(previous),metric_camera=False));return
 raise FileNotFoundError('Masked full-frame stabilization is not ready: '+cid)

def base_complete(cid):return all((ROOT/'base'/f'gvhmr_temporal_f{f}'/f'{cid}.npz').exists() and (ROOT/'base'/f'gvhmr_temporal_f{f}'/f'{cid}.json').exists() for f in [1,4,12])

def reuse_base(cid):
 for parent in ['fits_strong','fits_final']:
  source=OUT/parent
  if not all((source/f'gvhmr_temporal_f{f}'/f'{cid}.npz').exists() for f in [1,4,12]):continue
  for stage in ['camera','pose','temporal']:
   for f in [1,4,12]:
    for suffix in ['npz','json']:
     src=source/f'gvhmr_{stage}_f{f}'/f'{cid}.{suffix}';dest=ROOT/'base'/src.relative_to(source)
     if src.exists() and not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  return True
 return False

def base(records):
 import torch,optimize
 from pitch_study_fit import StudyFitter
 optimize.PREVIOUS=ROOT
 fitter=optimize.Fitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);fitter.branches=['temporal'];fitter.shared_betas={};fitter.fit_dir='all813_gvhmr_v1/base';fitter.withhold_2d=False;fitter.depth_weight=2.
 for rec in records:
  cid=rec['id']
  if base_complete(cid):continue
  if reuse_base(cid):print('BASE REUSE',cid,flush=True);continue
  if not network_complete(cid):continue
  try:background(cid)
  except FileNotFoundError:continue
  start=time.monotonic()
  try:
   fitter.fit_observed(cid,'gvhmr')
   save(ROOT/'base_status'/f'{cid}.json',dict(id=cid,status='complete',seconds=time.monotonic()-start,source_adapter_sha256=sha(OUT/'adapters/gvhmr_f4'/f'{cid}.npz')))
  except Exception as error:
   save(ROOT/'base_status'/f'{cid}.json',dict(id=cid,status='failed',error=str(error),traceback=traceback.format_exc()));print('BASE FAILED',cid,traceback.format_exc(),flush=True)
  torch.cuda.empty_cache()

def contact(records,smoke=False,graph=True):
 import torch,rear_pivot_bounded as bounded
 from pitch_study_fit import StudyFitter
 solver=bounded.solver;solver.CONFIG={**bounded.settings(),'version':'rear-pivot-population-v15','cuda_graph':graph,'allow_unknown_contact':True}
 solver.OUTPUT_NAME='all813_gvhmr_v1/graph_smoke' if smoke else 'all813_gvhmr_v1/contact';solver.SOURCE_FOLDER=ROOT/'base/gvhmr_temporal_f4';folder=OUT/solver.OUTPUT_NAME;folder.mkdir(parents=True,exist_ok=True)
 save(folder/f'protocol.gpu{os.environ.get("CUDA_VISIBLE_DEVICES","0")}.json',dict(config=solver.CONFIG,solver_sha256=sha(solver.__file__),main_source_sha256=sha(__file__),scope='Production population inference, not new held-out accuracy validation'))
 body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False)
 for rec in records:
  cid=rec['id']
  frozen=OUT/'rear_pivot_selected_v14'/f'{cid}.json'
  if not smoke and frozen.exists() and json.loads(frozen.read_text()).get('status')=='eligible_research_preview':
   print('CONTACT FROZEN V14 REUSE',cid,flush=True);continue
  if not base_complete(cid):continue
  try:solver.fit(rec,body,solver.CONFIG['steps'])
  except Exception as error:
   save(ROOT/('graph_smoke_failures' if smoke else 'contact_failures')/f'{cid}.json',dict(id=cid,error=str(error),traceback=traceback.format_exc()));print('CONTACT FAILED',cid,traceback.format_exc(),flush=True)
  gc.collect();torch.cuda.empty_cache()

def main():
 import torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30,0)
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['base','contact','smoke']);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--id');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--no-graph',action='store_true');a=ap.parse_args();records=population()[a.worker::a.workers]
 if a.id:records=[r for r in population() if r['id']==a.id]
 if a.limit:records=records[:a.limit]
 if a.stage=='base':base(records)
 else:
  for rec in records:
   if not base_complete(rec['id']):reuse_base(rec['id'])
  contact(records,a.stage=='smoke',not a.no_graph)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
