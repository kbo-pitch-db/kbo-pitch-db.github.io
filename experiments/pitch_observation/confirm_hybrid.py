"""Fixed explicit-component hypothesis on four untouched NLF test clips."""
from common import *
import subprocess
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();folder=Path(__file__).parent;records=sample('test',False)[12:16];dest=OUT/'hybrid_confirmation_protocol.json'
 if not dest.exists():save(dest,dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ids=[r['id'] for r in records],selection='Fixed test indices 12..15, no NLF reconstruction performance inspected before this seal',body='NLF torso/lower body/shape',arm='Learned temporal MotionBERT global arm orientation; explicit SO3 hierarchy adapter, no averaging',optimizer='Same native-shape staged fitter, depth coefficient 2, three camera hypotheses, 300 steps',initializer_sha256=sha(folder/'hybrid_initializer.py'),fitter_sha256=sha(folder/'optimize_nlf_hybrid.py'),main_frozen_protocol_unchanged=True))
 start=time.monotonic()
 while not all((OUT/'adapters/nlf_rgb'/f'{r["id"]}.npz').exists() for r in records):
  if time.monotonic()-start>1800:raise TimeoutError('Resume NLF inference')
  time.sleep(10)
 for script,args in [('hybrid_initializer.py',['--split','test','--offset','12','--limit','4']),('optimize_nlf_shape.py',['--fit-dir','fits_nlf_native_shape_test','--split','test','--full-sample','--offset','12','--limit','4','--initializers','nlf_rgb','--branches','temporal','--full-2d','--depth-preserve','2']),('optimize_nlf_hybrid.py',['--fit-dir','fits_nlf_hybrid_test','--split','test','--full-sample','--offset','12','--limit','4','--initializers','nlf_mb_hybrid','--branches','temporal','--full-2d','--depth-preserve','2']),('nlf_native_fixed.py',['--split','test','--offset','12','--limit','4'])]:subprocess.run([sys.executable,str(folder/script),*args],check=True)
 print('HYBRID CONFIRMATION FITS COMPLETE',flush=True)
