"""Fresh four-clip confirmation of weak RGB contour fitting, no retuning."""
from common import *
import subprocess
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();folder=Path(__file__).parent;records=sample('test',False)[4:8];seal=OUT/'nlf_rgb_confirmation_protocol.json'
 if not seal.exists():save(seal,dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ids=[r['id'] for r in records],reason='Next four fixed test records, not the first four used to diagnose NLF fitting loss',rgb_contour_weight=.10,raw_depth_weight=2.,shape='Median native NLF shape, fixed per clip',mask_selection='Only keyframes with order % 3 != 2; every third withheld',rgb_constraint_sha256=sha(folder/'rgb_mask_constraint.py'),optimizer_sha256=sha(folder/'optimize_nlf_rgb.py'),no_change_to_main_frozen_pool=True))
 start=time.monotonic()
 while not all((OUT/'adapters/nlf_rgb'/f'{r["id"]}.npz').exists() for r in records):
  if time.monotonic()-start>1800:raise TimeoutError('Resume independent raw inference first')
  time.sleep(10)
 for script,args in [('optimize_nlf_shape.py',['--fit-dir','fits_nlf_native_shape_test','--split','test','--full-sample','--offset','4','--limit','4','--initializers','nlf_rgb','--branches','temporal','--full-2d','--depth-preserve','2']),('optimize_nlf_rgb.py',['--fit-dir','fits_nlf_rgb_test','--split','test','--full-sample','--offset','4','--limit','4','--initializers','nlf_rgb','--branches','temporal','--full-2d','--depth-preserve','2']),('nlf_native_fixed.py',['--split','test','--offset','4','--limit','4']),('nlf_mesh_rgb.py',['--split','test','--offset','4','--limit','4'])]:subprocess.run([sys.executable,str(folder/script),*args],check=True)
 print('FRESH RGB CONFIRMATION COMPLETE',flush=True)
