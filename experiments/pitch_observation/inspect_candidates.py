"""Fixed-axis raw candidate comparisons. Only one clip-wide scale per model."""
from common import *
from candidates import CC,EDGES
import argparse

def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import cv2
 ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--frames',nargs='*',type=int);a=ap.parse_args()
 for rec in sample():
  cid=rec['id']
  if a.id and cid!=a.id:continue
  r,c,s,h,w=init_inputs(cid);N=len(h['keypoints']);chosen=a.frames or [int(N*.3),int(N*.55),int(N*.7)]
  tags=['current','WHAM_raw','hmr2a','gvhmr_f4','MotionBERT_native','mb_obp_adapter'];data={tag:load(OUT/'candidates'/tag/f'{cid}.npz') for tag in tags}
  video=cv2.VideoCapture(str(OUT/'original_clips'/f'{cid}.mp4'))
  for frame in chosen:
   video.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=video.read()
   if not ok:continue
   boxes=h['bbox'];x1,y1=np.nanmin(boxes[:,:2],0)-30;x2,y2=np.nanmax(boxes[:,2:],0)+30;x1=max(0,int(x1));y1=max(0,int(y1));x2=min(image.shape[1],int(x2));y2=min(image.shape[0],int(y2))
   fig,axes=plt.subplots(4,len(tags),figsize=(18,13));fig.suptitle(f'{cid} source frame {frame} | red left / blue right | native raw (GVHMR approximate SMPL adapter)')
   for col,tag in enumerate(tags):
    z=data[tag];p=z['keypoints'][frame];proj=z['projected'][frame];ax=axes[0,col];ax.imshow(image[y1:y2,x1:x2,::-1]);ax.set_title(tag);ax.axis('off')
    for k,(i,j) in enumerate(EDGES):
     color='r' if i in [0,2,4,6,8,10] and j in [0,2,4,6,8,10] else 'dodgerblue' if i in [1,3,5,7,9,11] and j in [1,3,5,7,9,11] else 'lime'
     ax.plot(proj[[i,j],0]-x1,proj[[i,j],1]-y1,color=color,lw=1)
    for row,(a0,a1,label) in enumerate([(0,1,'camera x/y'),(2,1,'depth/y'),(0,2,'x/depth')],1):
     ax=axes[row,col]
     for i,j in EDGES:
      color='r' if i in [0,2,4,6,8,10] and j in [0,2,4,6,8,10] else 'dodgerblue' if i in [1,3,5,7,9,11] and j in [1,3,5,7,9,11] else 'gray'
      ax.plot(p[[i,j],a0],p[[i,j],a1],color=color,lw=2)
     ax.set_xlim(-1.6,1.6);ax.set_ylim(1.6,-1.6);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(label)
   folder=OUT/'inspection';folder.mkdir(exist_ok=True);fig.tight_layout();fig.savefig(folder/f'{cid}_{frame:05d}.jpg',dpi=105);plt.close(fig)
  video.release()

if __name__=='__main__':main()
