from common import *
from candidates import EDGES,CC

def main():
 import cv2
 records=[]
 for rec in sample():
  cid=rec['id'];h=init_inputs(cid)[3];z=load(OUT/'alternative_2d/rtmw384'/f'{cid}.npz');score=scores(h);height=float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1]));delta=np.linalg.norm(z['keypoints'][:,:17]-h['keypoints'][:,:17],axis=-1);valid=(score[:,:17]>.5)&(z['scores'][:,:17]>.5)
  records.append(dict(id=cid,wrist_disagreement_p95_px=float(np.percentile(delta[:,9:11][valid[:,9:11]],95)),body_disagreement_p95_px=float(np.percentile(delta[valid],95))))
  folder=OUT/'detector_evidence'/cid;folder.mkdir(parents=True,exist_ok=True)
  for path in sorted((OUT/'observations'/cid).glob('*.npz')):
   obs=load(path);i=int(path.stem);image=cv2.imread(str(path.with_suffix('.jpg')));panels=[]
   for tag,xy,conf in [('Halpe final',h['keypoints'][i,:17],score[i,:17]),('RTMW separate',z['keypoints'][i,:17],z['scores'][i,:17])]:
    panel=image.copy();p=xy-obs['crop_origin'];p=p[CC];confidence=conf[CC]
    for u,v in EDGES:
     if min(confidence[u],confidence[v])<.4:continue
     color=(0,0,255) if u in [0,2,4,6,8,10] and v in [0,2,4,6,8,10] else (255,150,0)
     cv2.line(panel,tuple(p[u].astype(int)),tuple(p[v].astype(int)),color,2)
    cv2.putText(panel,tag,(4,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);panels.append(panel)
   cv2.imwrite(str(folder/f'{i:05d}.jpg'),np.concatenate([image]+panels,1))
 save(OUT/'detector_comparison.json',records);csvwrite(OUT/'detector_comparison.csv',records)
if __name__=='__main__':main()
