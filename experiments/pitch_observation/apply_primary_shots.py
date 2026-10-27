"""Activate source-confirmed shot boundaries and retain every replaced result.

Pause only this project's supervisor and 3D model workers during the pointer
change. RGB feature extraction, background workers and unrelated jobs continue.
"""
from common import *
from all813_release import ROOT
from supervise_all813 import running,PY,EXP
import argparse,signal,subprocess,shutil

def stop_workers():
 pid_path=ROOT/'supervisor.pid'
 if pid_path.exists():
  pid=int(pid_path.read_text())
  try:
   args=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
   if 'supervise_all813.py' in args:os.kill(pid,signal.SIGTERM)
  except FileNotFoundError:pass
 pids=[]
 for worker in [0,1]:
  for script,stage in [('all813_release.py','network'),('all813_fit.py','base'),('all813_fit.py','contact')]:
   for pid in running(script,stage,worker):os.kill(pid,signal.SIGTERM);pids.append(pid)
 for attempt in range(100):
  alive=[]
  for pid in pids:
   stat=Path(f'/proc/{pid}/stat')
   if stat.exists() and stat.read_text().split()[2]!='Z':alive.append(pid)
  if not alive:return pids
  time.sleep(.1)
 raise RuntimeError('Own 3D workers have not exited; no result files moved: '+str(alive))

def activate(ids):
 plans=[]
 for cid in ids:
  if (ROOT/'shot_validity'/f'{cid}.json').exists():continue
  source=next(ROOT/f'background_worker{w}/background'/f'{cid}.json' for w in [0,1] if (ROOT/f'background_worker{w}/background'/f'{cid}.json').exists());obs=json.loads(source.read_text());cuts=obs['cut_candidates'];r=rows('motion_wham.csv')[cid];h=load(DATA/r['pose_2d_refined']);count=len(h['frame_index']);n=int(min(cuts));assert 8<=n<count
  plans.append(dict(id=cid,valid_frames=n,source_clip_frames=count,cut_indices=cuts,confirmed_source_cut=True,review='Agent inspected original and original crop before/at cut; full/crop frame boundaries matched',human_review_complete=False,final_2d_sha256=sha(DATA/r['pose_2d_refined']),background_observation_sha256=sha(source)))
 if not plans:print('All requested policies already active');return
 pids=stop_workers();archives=[]
 try:
  for policy in plans:
   cid=policy['id'];folders=[OUT/'raw'/f'gvhmr_f{f}' for f in [1,4,12]]+[OUT/'adapters'/f'gvhmr_f{f}' for f in [1,4,12]]+[ROOT/'base'/f'gvhmr_{stage}_f{f}' for stage in ['camera','pose','temporal'] for f in [1,4,12]]+[ROOT/name for name in ['background','base_status','contact','selected']]
   moved=[]
   for folder in folders:
    for suffix in ['npz','json']:
     path=folder/f'{cid}.{suffix}'
     if not path.exists():continue
     target=ROOT/'pre_shot_correction_archive'/cid/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();digest=sha(path);shutil.move(str(path),target);moved.append(dict(original=str(path.relative_to(OUT)),archive=str(target.relative_to(OUT)),sha256=digest))
   save(ROOT/'shot_validity'/f'{cid}.json',policy);archives.append(dict(id=cid,policy=policy,archived_results=moved))
  save(ROOT/'shot_cut_review/activation.json',dict(stopped_own_3d_worker_pids=pids,policies=archives,original_inputs_modified=False,features_and_background_workers_interrupted=False))
 finally:
  log=(OUT/'logs/all813_supervisor.log').open('a');p=subprocess.Popen([str(PY),str(EXP/'supervise_all813.py')],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(ROOT/'supervisor.pid').write_text(str(p.pid));log.close()
 print('Activated shot policies',len(plans),'and resumed supervisor')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='+',required=True);args=ap.parse_args();activate(args.ids)
