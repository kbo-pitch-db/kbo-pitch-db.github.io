"""Resume independent RGB initializer -> unchanged staged fitter, tune-only.

This is an exploratory extension after the main protocol was frozen. It cannot
enter the sealed test selector. Native NLF outputs remain separate from the
WHAM-shape-controlled SMPL fit.
"""
from common import *
from scipy.spatial.transform import Rotation
import subprocess,argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();records=sample(a.split,a.split=='tune');records=records[a.offset:];records=records[:a.limit] if a.limit else records
 script=Path(__file__).parent/'optimize.py';extra=Path(__file__).parent/'optimize_nlf.py';text=script.read_text().replace("'mb_block_flip_smpl']","'mb_block_flip_smpl','nlf_rgb']");extra.write_text(text)
 for rec in records:
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);start=time.monotonic();folder=OUT/'raw/nlf_frames'/cid
  while len(list(folder.glob('*_f4.npz')))<len(h['source_time']):
   if time.monotonic()-start>1800:raise TimeoutError('NLF RGB sequence budget; rerun after raw inference resumes')
   time.sleep(10)
  zs=[load(folder/f'{i:05d}_f4.npz') for i in range(len(h['source_time']))];rot=Rotation.from_rotvec(np.concatenate([z['pose'] for z in zs]).reshape(-1,3)).as_matrix().reshape(-1,24,3,3).astype(np.float32);betas=np.concatenate([z['betas'] for z in zs]);atomic_npz(OUT/'adapters/nlf_rgb'/f'{cid}.npz',smpl_rotations=rot,smpl_betas=betas,native_joint_centers_mm=np.concatenate([z['joints3d_nonparam'] for z in zs]),native_smpl_fit_centers_mm=np.concatenate([z['joints3d'] for z in zs]),source_time=h['source_time'],frame_index=h['frame_index'],representation='Official NLF RGB SMPL rotation output, no WHAM pose; fixed-WHAM-shape substitution occurs only in controlled fitting, not native output')
  subprocess.run([sys.executable,str(extra),'--fit-dir',('fits_nlf' if a.split=='tune' else 'fits_nlf_test'),'--split',a.split,'--full-sample','--id',cid,'--initializers','nlf_rgb','--branches','temporal','--full-2d','--depth-preserve','2'],check=True)
  print('INTEGRATED NLF',cid,flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
