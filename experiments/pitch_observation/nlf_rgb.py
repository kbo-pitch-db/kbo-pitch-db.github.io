"""Official NLF RGB inference, separate native nonparametric and SMPL-fit outputs.

Run with an isolated/read-only PyTorch>=2.4 interpreter (JIT requires newer ops).
No world motion, metric camera calibration, depth averaging or WHAM pose input.
"""
from common import *
import argparse

def main():
 import torch,torchvision,cv2
 torch.set_num_threads(1);cv2.setNumThreads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--offset',type=int,default=0);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--smoke',action='store_true');ap.add_argument('--all-frames',action='store_true');ap.add_argument('--factors',nargs='+',type=float,default=[1.,4.,12.]);a=ap.parse_args();weight=WORKSPACE/'models/nlf_s_multi_0.2.2.patch2.torchscript';m=torch.jit.load(str(weight),map_location='cuda:0').eval();records=sample(a.split,not a.full_sample)
 records=records[a.offset:]
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);folder=OUT/('raw/nlf_frames' if a.all_frames else 'raw/nlf_keyframes')/cid;folder.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(OUT/'original_clips'/f'{cid}.mp4'));start=time.monotonic();paths=sorted((OUT/'observations'/cid).glob('*.npz'))
  if a.all_frames:paths=[OUT/'observations'/cid/f'{i:05d}.npz' for i in range(len(h['source_time']))]
  if a.smoke:paths=paths[:1]
  for path in paths:
   idx=int(path.stem);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,image=cap.read()
   if not ok:raise ValueError('Missing full source '+cid)
   H,W=image.shape[:2];box=(load(path)['source_box'] if path.exists() else h['bbox'][idx]).copy();
   if not np.isfinite(box).all():
    good=np.flatnonzero(np.isfinite(h['bbox']).all(1));box=h['bbox'][good[np.argmin(abs(good-idx))]].copy()
   box[2:]-=box[:2];images=torch.from_numpy(image[:,:,::-1].copy()).permute(2,0,1)[None].cuda(0)
   for factor in a.factors:
    dest=folder/f'{idx:05d}_f{factor:g}.npz'
    if dest.exists():continue
    keycached=OUT/'raw/nlf_keyframes'/cid/dest.name
    if a.all_frames and keycached.exists():
     import shutil;shutil.copy2(keycached,dest);continue
    f=factor*np.hypot(W,H);K=torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1]]],dtype=torch.float32,device='cuda:0')
    with torch.no_grad():out=m.estimate_smpl_batched(images,[torch.tensor(box[None],device='cuda:0',dtype=torch.float32)],intrinsic_matrix=K,num_aug=1,internal_batch_size=1,model_name='smpl')
    values={k:v[0].detach().cpu().numpy() for k,v in out.items()};atomic_npz(dest,**values,focal=f,principal=np.array([W/2,H/2]),frame=idx,source_time=h['source_time'][idx]);print('NLF RAW',cid,idx,factor,{k:list(v.shape) for k,v in values.items()},flush=True)
   if a.smoke:break
  cap.release();save(folder/'provenance.json',dict(model='Official NLF S multi v0.2.2 patch2 TorchScript',sha256=sha(weight),torch=torch.__version__,seconds=time.monotonic()-start,body='Native nonparametric joint/vertex prediction and separately named SMPL fit, as returned by official method',input='Original full-resolution RGB + observed person box; supplied focal hypotheses, no pose prompt',camera_calibrated=False,global_trajectory=False,num_aug=1))
if __name__=='__main__':main()
