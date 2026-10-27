"""Original-to-crop frame alignment and coordinate checks, with explicit offsets."""
from common import *
import argparse

def main():
 import cv2
 from skimage.metrics import structural_similarity
 cv2.setNumThreads(1);ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();records=[]
 for rec in sample(a.split,not a.full_sample):
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);original=cv2.VideoCapture(str(OUT/'original_clips'/f'{cid}.mp4'));crop=cv2.VideoCapture(str(DATA/c['crop_video']));checks=[];x1,y1,x2,y2=[int(float(c[k])) for k in ['crop_x1','crop_y1','crop_x2','crop_y2']]
  for i in [3,len(h['keypoints'])//2,len(h['keypoints'])-4]:
   crop.set(cv2.CAP_PROP_POS_FRAMES,i);ok,b=crop.read()
   if not ok:continue
   candidates=[]
   for lag in [-2,-1,0,1,2]:
    original.set(cv2.CAP_PROP_POS_FRAMES,i+lag);ok,frame=original.read()
    if not ok:continue
    patch=frame[y1:y2,x1:x2]
    if patch.shape[:2]!=b.shape[:2]:raise ValueError('Unmodeled crop resize '+cid+' '+str((patch.shape,b.shape)))
    # Pixel comparison only, robust to the separately encoded source clip.
    aa=cv2.resize(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),(160,192));bb=cv2.resize(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),(160,192));score=float(structural_similarity(aa,bb,data_range=255));candidates.append((score,lag))
   best=max(candidates);zero=next(q[0] for q in candidates if q[1]==0);checks.append(dict(frame=i,best_lag=best[1],zero_lag_ssim=zero,best_lag_ssim=best[0]))
  original.release();crop.release();dt=np.diff(h['source_time']);record=dict(id=cid,checks=checks,all_zero_lag_best=all(q['best_lag']==0 for q in checks),positive_timestamps=bool((dt>0).all()),maximum_timestamp_frame_error_s=float(np.max(abs((h['source_time']-h['source_time'][0])-(h['frame_index']-h['frame_index'][0])/float(h['fps'])))),original_resolution=[int(s['width']),int(s['height'])],crop=[x1,y1,x2,y2],semantics='Encoded source frames, not confirmed real-time biomechanical timestamps; slow motion not rescaled')
  records.append(record);print('ALIGNMENT',cid,record['all_zero_lag_best'],[round(q['zero_lag_ssim'],3) for q in checks],flush=True)
 save(OUT/f'alignment_{a.split}.json',records)
if __name__=='__main__':main()
