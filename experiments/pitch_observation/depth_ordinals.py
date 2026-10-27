"""Conservative RGB-pseudo-depth order intervals; no surface-to-joint substitution.

Only visible limb pixels OUTSIDE the observed torso polygon and at torso height
are tested. Full-frame and context-crop depth must agree. Joint-center tests
allow a 0.25 modeled-leg ambiguity for surface/body thickness. Unknown stays
unknown. Every third keyframe is withheld from subsequent ordinal fitting.
"""
from common import *
import argparse

def evidence(cid):
 import cv2
 from scipy.ndimage import binary_erosion
 r,c,s,h,w=init_inputs(cid);confidence=scores(h);records=[];allprobes=[]
 paths=sorted((OUT/'observations'/cid).glob('*.npz'))
 for order,path in enumerate(paths):
  i=int(path.stem);obs=load(path)
  if 'mask' not in obs:continue
  xy=h['keypoints'][i]-obs['crop_origin'];mask=binary_erosion(obs['mask'],iterations=2);H,W=mask.shape
  torso=xy[[5,6,12,11]]
  if not np.isfinite(torso).all() or (confidence[i,[5,6,11,12]]<.6).any():continue
  shoulder=xy[[5,6]].mean(0);hip=xy[[11,12]].mean(0)
  if hip[1]-shoulder[1]<30:continue
  def patch(pos,z):
   x,y=np.round(pos).astype(int)
   if x<3 or y<3 or x>=W-3 or y>=H-3:return None
   m=mask[y-3:y+4,x-3:x+4];v=z[y-3:y+4,x-3:x+4][m]
   if len(v)<30:return None
   return float(np.median(v))
  for j in [7,8,9,10]:
   if confidence[i,j]<.65 or not np.isfinite(xy[j]).all():continue
   fraction=(xy[j,1]-shoulder[1])/max(hip[1]-shoulder[1],1)
   if fraction<.05 or fraction>.95:continue
   if cv2.pointPolygonTest(torso.astype(np.float32),tuple(xy[j].astype(float)),True)>-5:continue
   ref=shoulder*(1-fraction)+hip*fraction;deltas=[];spreads=[]
   for key in ['full_image_inverse_depth','person_context_inverse_depth']:
    z=obs[key].astype(np.float32);a=patch(xy[j],z);b=patch(ref,z)
    if a is None or b is None:break
    spread=max(float(np.percentile(z[mask],85)-np.percentile(z[mask],15)),1e-6);deltas.append((a-b)/spread);spreads.append(spread)
   if len(deltas)<2:continue
   accepted=np.sign(deltas[0])==np.sign(deltas[1]) and min(abs(np.array(deltas)))>.12
   q=dict(id=cid,frame=i,joint=j,torso_fraction=float(fraction),full_delta=float(deltas[0]),crop_delta=float(deltas[1]),near_sign=int(np.sign(deltas[0])) if accepted else None,
    status='pseudo_depth_order' if accepted else 'unknown_context_or_small_difference',split_observation='withheld' if order%3==2 else 'fit',
    center_depth_tolerance_leg=.25,limb_pixel=xy[j]+obs['crop_origin'],torso_pixel=ref+obs['crop_origin'],human_review=False)
   allprobes.append(q)
   if accepted:records.append(q)
 return records,allprobes

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();records=[];probes=[]
 for rec in sample(a.split,not a.full_sample):
  found,allprobes=evidence(rec['id']);records+=found;probes+=allprobes
 save(OUT/f'depth_ordinals_{a.split}.json',records);save(OUT/f'depth_ordinal_probes_{a.split}.json',probes);csvwrite(OUT/f'depth_ordinals_{a.split}.csv',records)
 print('Depth ordinal model observations',len(records),'unknown',len(probes)-len(records))
if __name__=='__main__':main()
