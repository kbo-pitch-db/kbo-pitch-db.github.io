"""Private original-pixel diagnostic sheets; never copy into Pages assets."""
from common import *
from arm_cocking_refine import FOLDER,deployment_guard
from stride_review import native_leg
import argparse,gzip

def draw(cid):
 import cv2,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 choice=json.loads((FOLDER/'selection_rule.json').read_text())['selected_variant'];report=json.loads((FOLDER/'reports'/f'{cid}.json').read_text());base=load(FOLDER/'fits/current_GVHMR'/f'{cid}.npz');new=load(FOLDER/'fits'/choice/f'{cid}.npz');h=init_inputs(cid)[3];source=init_inputs(cid)[2];clip=next(c for c in json.loads((CODE/'data/clips.json').read_text())['clips'] if c['id']==cid);pose=json.loads(gzip.decompress((CODE/'data'/clip['pose3d']).read_bytes()));R=np.array(pose['scene']['rotation_camera_to_scene']);Rplot=np.array([[1,0,0],[0,0,-1],[0,1,0]])@R;origin=np.array(pose['scene']['origin_camera']);leg=native_leg(base['anatomical_keypoints']);assert np.linalg.det(Rplot)>.9999
 coordinates=[(z['anatomical_keypoints']+z['translation'][:,None]-origin)@Rplot.T/leg for z in [base,new]];cloud=np.concatenate([p.reshape(-1,3) for p in coordinates]);lo=cloud.min(0)-.12;hi=cloud.max(0)+.12;parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
 window=report['window'].get('frame_interval') or [0,len(h['frame_index'])-1];frames=sorted(set([window[0],int(np.mean(window)),window[1],max(window[0],int(np.mean(window))-3)]));video=OUT/'original_clips'/f'{cid}.mp4'
 if not video.exists():video=next(p for p in [OUT/'all813_gvhmr_v1/background_worker0/source_clips'/f'{cid}.mp4',OUT/'all813_gvhmr_v1/background_worker1/source_clips'/f'{cid}.mp4',PREVIOUS/'source_clips'/f'{cid}.mp4'] if p.exists())
 cap=cv2.VideoCapture(str(video));side=0 if report['throwing_side']=='left' else 1;arm=[16+side,18+side,20+side];arms2=[5+side,7+side,9+side];fig=plt.figure(figsize=(18,len(frames)*3.7))
 for row,i in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read();ax=fig.add_subplot(len(frames),5,row*5+1)
  if ok:
   scale=image.shape[1]/float(source['width']);ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));obs=h['keypoints'][i,arms2]*scale;ax.plot(*obs.T,'o-',c='#c9f36c',lw=1,ms=3)
   alternate=OUT/'alternative_2d/rtmw384'/f'{cid}.npz'
   if alternate.exists():
    alt=load(alternate);ax.plot(*(alt['keypoints'][i,arms2]*scale).T,'x:',c='#e768dd',lw=1,ms=3)
   for z,color in [(base,'#a7bacb'),(new,'#ff9847')]:ax.plot(*(z['projected'][i,arms2]*scale).T,'+-',c=color,lw=1,ms=4)
  ax.set_title(f'Source + arm | F{i}');ax.axis('off')
  for col,(m,view) in enumerate([(0,(8,-5)),(1,(8,-5)),(0,(90,-90)),(1,(90,-90))],1):
   ax=fig.add_subplot(len(frames),5,row*5+col+1,projection='3d');a=coordinates[m]
   for j,p in enumerate(parents[1:],1):ax.plot(*a[i,[p,j]].T,c='#82938b',lw=.8)
   ax.plot(*a[i,arm].T,'o-',c='#4b6d91' if m==0 else '#d37927',lw=2,ms=3);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(*view);ax.set_title(('Released GVHMR' if m==0 else choice)+(' side' if col<3 else ' top'),fontsize=9)
 cap.release();fig.suptitle(cid+' | shared released origin, fixed axes/range; source green, GVHMR blue-gray, candidate orange');fig.tight_layout();folder=FOLDER/'local_source_qa';folder.mkdir(exist_ok=True);fig.savefig(folder/f'{cid}.png',dpi=110);plt.close(fig);save(folder/f'{cid}.json',dict(id=cid,source_frames=frames,variant=choice,source_video=str(video),agent_diagnostic=True,human_review_complete=False,public_broadcast_pixels=False))

if __name__=='__main__':
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2','v3'],default='v3');ap.add_argument('--ids',nargs='+',required=True);a=ap.parse_args()
 if a.version!='v1':FOLDER=__import__('arm_cocking_'+a.version).FOLDER
 for cid in a.ids:draw(cid)
