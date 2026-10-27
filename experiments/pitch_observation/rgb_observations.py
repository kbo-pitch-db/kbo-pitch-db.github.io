"""Cache independent *model observations* from original full-resolution RGB.

Depth is an RGB prior, never ground truth or joint-center depth. Both full-image
and person-context estimates are retained to expose crop/context sensitivity.
SAM masks are box-prompted, not manual segmentation. No final 2D is modified.
"""
from common import *
import argparse

def keyframes(h,n=13):
 return np.unique(np.linspace(0,len(h['keypoints'])-1,n).round().astype(int))

def main():
 import cv2,torch,subprocess,imageio_ffmpeg
 torch.set_num_threads(1);cv2.setNumThreads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--mask',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 sys.path.insert(0,str(WORKSPACE/'external/Depth-Anything-V2'))
 from depth_anything_v2.dpt import DepthAnythingV2
 weight=WORKSPACE/'models/depth_anything_v2_vits.pth';weight_hash=sha(weight)
 assert weight_hash=='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
 depth=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]);depth.load_state_dict(torch.load(weight,map_location='cpu'),strict=True);depth.cuda(0).eval()
 if a.mask:
  sys.path.insert(0,str(WORKSPACE/'external/segment-anything'))
  from segment_anything import sam_model_registry,SamPredictor
  mask_weight=WORKSPACE/'models/sam_vit_b_01ec64.pth'
  predictor=SamPredictor(sam_model_registry['vit_b'](checkpoint=str(mask_weight)).cuda(0).eval())
 records=sample(a.split,not a.full_sample)
 if a.limit:records=records[:a.limit]
 for rec in records:
  cid=rec['id'];r,c,s,h,raw=init_inputs(cid);dest=OUT/'observations'/cid;dest.mkdir(parents=True,exist_ok=True)
  # System OpenCV cannot decode some AV1 sources. FFmpeg uses the original
  # resolution, with accurate timestamp seeking; no crop or resize is applied.
  local=OUT/'original_clips'/f'{cid}.mp4';local.parent.mkdir(exist_ok=True)
  if not local.exists():
   tmp=local.with_name(local.stem+'.tmp.mp4')
   subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-threads','2','-ss',str(float(h['source_time'][0])),'-i',s['source_path'],'-frames:v',str(len(h['keypoints'])),'-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','17',str(tmp)],check=True)
   tmp.replace(local)
  cap=cv2.VideoCapture(str(local));start=time.monotonic();entries=[]
  try:
   for index in keyframes(h):
    p=dest/f'{index:05d}.npz'
    if p.exists() and (not a.mask or 'mask' in load(p)):
     entries.append(dict(frame=int(index),cached=True));continue
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
    if not ok:raise RuntimeError('Cannot read original '+cid)
    H,W=frame.shape[:2];box=h['bbox'][index].copy()
    if not np.isfinite(box).all():
     good=np.flatnonzero(np.isfinite(h['bbox']).all(1));box=h['bbox'][good[np.argmin(abs(good-index))]].copy()
    size=box[2:]-box[:2];lo=np.maximum(0,np.floor(box[:2]-.25*size)).astype(int);hi=np.minimum([W,H],np.ceil(box[2:]+.25*size)).astype(int);x1,y1=lo;x2,y2=hi
    crop=frame[y1:y2,x1:x2]
    with torch.no_grad():
     full_depth=depth.infer_image(frame,input_size=518)[y1:y2,x1:x2]
     crop_depth=depth.infer_image(crop,input_size=518)
    values=dict(full_image_inverse_depth=full_depth.astype(np.float16),person_context_inverse_depth=crop_depth.astype(np.float16),crop_origin=lo,source_box=box,frame_index=h['frame_index'][index],source_time=h['source_time'][index])
    if a.mask:
     predictor.set_image(frame[:,:,::-1]);m,quality,_=predictor.predict(box=box,multimask_output=True);best=int(quality.argmax());values.update(mask=m[best,y1:y2,x1:x2],sam_scores=quality,sam_chosen=best)
    atomic_npz(p,**values)
    cv2.imwrite(str(dest/f'{index:05d}.jpg'),crop)
    # Human-readable evidence: RGB and independently normalized inverse depths.
    panels=[crop]
    for z in [full_depth,crop_depth]:
     z=(z-np.percentile(z,2))/max(float(np.percentile(z,98)-np.percentile(z,2)),1e-6)
     panels.append(cv2.applyColorMap((np.clip(z,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO))
    if a.mask:
     overlay=crop.copy();overlay[values['mask']]=(overlay[values['mask']]*.5+np.array([0,180,0])*.5).astype(np.uint8);panels.append(overlay)
    cv2.imwrite(str(dest/f'{index:05d}_evidence.jpg'),np.concatenate(panels,1))
    entries.append(dict(frame=int(index),source_frame=int(c['source_frame_start'])+int(index),time=float(h['source_time'][index])))
   save(dest/'provenance.json',dict(id=cid,source=s['source_path'],source_size_bytes=Path(s['source_path']).stat().st_size,frames=entries,seconds=time.monotonic()-start,depth_weight_sha256=weight_hash,
    depth='Official Depth Anything V2 Small relative inverse depth; original full frame and separate 1.5x person crop',mask='Official SAM ViT-B box prompt' if a.mask else 'not computed',
    caveats=['Same RGB model prior, not sensor or GT','Surface depth cannot be assigned to joint centers','Separate crops have unrelated scale/shift','Mask prompt and pseudo-depth may fail at blur and occlusion'],human_review=False))
  finally:cap.release()
  print('RGB OBSERVATIONS',cid,len(entries),round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
