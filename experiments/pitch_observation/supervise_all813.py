"""Durable stage supervisor; resumes own workers, never publishes anything."""
from common import *
from all813_release import ROOT,population,feature_complete,network_complete
from all813_fit import base_complete
import subprocess

PY=OUT/'env/bin/python';EXP=CODE/'experiments/pitch_observation'

def running(script,stage,worker):
 found=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
  except (OSError,UnicodeDecodeError):continue
  if len(args)<2 or not args[1].endswith('/'+script):continue
  if stage is not None and (len(args)<3 or args[2]!=stage):continue
  if '--worker' not in args or args[args.index('--worker')+1]!=str(worker):continue
  found.append(int(proc.name))
 return found

def start(script,stage,worker):
 log=OUT/'logs'/f'all813_supervised_{script[:-3]}_{stage or "background"}_gpu{worker}.log'
 environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(worker)}
 args=[str(PY),str(EXP/script)]+([stage] if stage else [])+['--worker',str(worker)]
 with log.open('a') as handle:p=subprocess.Popen(args,cwd=CODE,env=environment,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
 return p.pid

def frozen(cid):
 p=OUT/'rear_pivot_selected_v14'/f'{cid}.json'
 return p.exists() and json.loads(p.read_text()).get('status')=='eligible_research_preview'

def contact_complete(cid):
 p=ROOT/'contact'/f'{cid}.npz'
 return frozen(cid) or p.exists() and p.with_suffix('.json').exists()

def background_complete(cid):
 return (PREVIOUS/'background'/f'{cid}.npz').exists() or any((ROOT/f'background_worker{w}/background'/f'{cid}.json').exists() for w in [0,1])

def main():
 records=population();save(ROOT/'supervisor_provenance.json',dict(script_sha256=sha(__file__),auto_publish=False,models_and_final_2d_preserved=True,workers=2))
 while True:
  rows_=[];ready=True
  for worker in [0,1]:
   rr=records[worker::2];state=dict(worker=worker,total=len(rr),features=sum(feature_complete(r['id']) for r in rr),network=sum(network_complete(r['id']) for r in rr),background=sum(background_complete(r['id']) for r in rr),base=sum(base_complete(r['id']) for r in rr),contact=sum(contact_complete(r['id']) for r in rr),processes={})
   stages=[('all813_release.py','features',any(not feature_complete(r['id']) for r in rr)),('all813_background.py',None,any(not background_complete(r['id']) for r in rr)),('all813_release.py','network',any(feature_complete(r['id']) and not network_complete(r['id']) for r in rr)),('all813_fit.py','base',any(network_complete(r['id']) and background_complete(r['id']) and not base_complete(r['id']) for r in rr)),('all813_fit.py','contact',any(base_complete(r['id']) and not contact_complete(r['id']) for r in rr))]
   for script,stage,pending in stages:
    pids=running(script,stage,worker)
    if pending and not pids:pids=[start(script,stage,worker)]
    state['processes'][stage or 'background']=pids
   rows_.append(state);ready=ready and state['contact']==len(rr) and state['base']==len(rr) and state['network']==len(rr) and state['features']==len(rr)
  report=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ready_for_validation=ready,workers=rows_,auto_publish=False)
  save(ROOT/'pipeline_progress.json',report);print(json.dumps(report),flush=True)
  if ready:
   subprocess.run([str(PY),str(EXP/'all813_select.py')],cwd=CODE,check=True)
   save(ROOT/'pipeline_complete.json',report);return
  time.sleep(30)

if __name__=='__main__':main()
