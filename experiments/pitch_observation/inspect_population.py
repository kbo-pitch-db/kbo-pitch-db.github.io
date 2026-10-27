"""Local-only original overlays and matched fixed-axis population diagnostics."""
from common import *
from all813_release import ROOT,enable_primary_shot_policy
from all813_select import select
from stride_review import native_leg
import argparse

def draw(cid):
 enable_primary_shot_policy()
 import cv2,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 meta=select(cid)
 if meta['status'] not in ['screened_consistency_preview','review_required_preview']:raise ValueError('Fit unavailable '+cid)
 old=load(ROOT/'base/gvhmr_temporal_f4'/f'{cid}.npz');new=load(ROOT/'selected'/f'{cid}.npz');r,c,s,h,w=init_inputs(cid);N=len(h['keypoints']);phase=meta['phase'];plant=phase['plant_interval'];burst=phase['events'].get('arm_fast_peak');frames=sorted(set([0,(plant or [N//3])[0],burst or N//2,N-1]));rear=1 if phase['lead_side']=='left' else 0
 normal=new['normal'].astype(float);normal/=np.linalg.norm(normal);x=np.array([1.,0.,0.]);x-=x@normal*normal;x/=np.linalg.norm(x);R=np.stack([x,np.cross(normal,x),normal]);assert np.linalg.det(R)>.99999;leg=native_leg(old['anatomical_keypoints']);models=[]
 origin=new['foot_keypoints'][0,rear]+new['translation'][0];origin=origin-(origin@normal+float(new['pilot_support_offset']))*normal
 for z in [old,new]:
  models.append(tuple((z[k]+z['translation'][:,None]-origin)@R.T/leg for k in ['anatomical_keypoints','foot_keypoints']))
 cloud=np.concatenate([a.reshape(-1,3) for pair in models for a in pair]);lo=cloud.min(0)-.1;hi=cloud.max(0)+.1;parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
 candidates=[ROOT/f'background_worker{i}/source_clips'/f'{cid}.mp4' for i in [0,1]]+[PREVIOUS/'source_clips'/f'{cid}.mp4',OUT/'original_clips'/f'{cid}.mp4'];video=next(p for p in candidates if p.exists());cap=cv2.VideoCapture(str(video));fig=plt.figure(figsize=(18,4*len(frames)))
 for row,i in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read();ax=fig.add_subplot(len(frames),5,row*5+1)
  if ok:
   ratio=image.shape[1]/float(s['width']);ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));obs=h['keypoints'][i];sc=scores(h)[i]>.5;ax.scatter(obs[sc,0]*ratio,obs[sc,1]*ratio,s=7,c='yellow')
   for z,color in [(old,'#62bbff'),(new,'#ff8159')]:
    uv=z['projected'][i,:17];ax.scatter(uv[:,0]*ratio,uv[:,1]*ratio,s=7,c=color,marker='+')
  ax.set_title(f'Local source | frame {i}');ax.axis('off')
  for col,(m,view) in enumerate([(0,(8,-5)),(1,(8,-5)),(0,(90,-90)),(1,(90,-90))],1):
   ax=fig.add_subplot(len(frames),5,row*5+col+1,projection='3d');a,f=models[m]
   for j,p in enumerate(parents[1:],1):ax.plot(*a[i,[p,j]].T,c='#28618d',lw=1.4)
   ax.scatter(*f[i].T,s=10,c='orange');ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(*view);ax.set_title(('F4 base' if m==0 else 'contact')+(' side' if col<3 else ' top'))
 cap.release();fig.suptitle(cid+' | one shared contact-frame origin and proper rotation for both candidates, same axes/scale; not calibrated field coordinates');fig.tight_layout();folder=ROOT/'local_source_qa';folder.mkdir(exist_ok=True);fig.savefig(folder/f'{cid}.png',dpi=110);plt.close(fig)
 save(folder/f'{cid}.json',dict(id=cid,source_frames=frames,source_video=str(video),human_review_complete=False,scope='Local automatic diagnostic and agent inspection only; not human review or 3D GT'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='+',required=True);a=ap.parse_args()
 for cid in a.ids:draw(cid)
