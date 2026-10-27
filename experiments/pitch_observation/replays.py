"""Local context search only. A nearby replay is NOT automatically the same pitch."""
from common import *
import subprocess

def main():
 import cv2,imageio_ffmpeg
 out=OUT/'replay_search';out.mkdir(exist_ok=True);records=[]
 for rec in sample():
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);dest=out/f'{cid}.jpg';begin=max(0,float(h['source_time'][0])-8);end=min(float(s['duration_seconds']),float(h['source_time'][-1])+24)
  if dest.exists():continue
  folder=out/cid;folder.mkdir(exist_ok=True)
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-threads','2','-ss',str(begin),'-i',s['source_path'],'-t',str(end-begin),'-vf','fps=1/2,scale=480:-2','-threads','2',str(folder/'%03d.jpg')],check=True)
  tiles=[]
  for i,p in enumerate(sorted(folder.glob('*.jpg'))):
   image=cv2.imread(str(p));cv2.putText(image,f'{begin+2*i:.2f}s',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2);tiles.append(image)
  if not tiles:continue
  while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
  montage=np.concatenate([np.concatenate(tiles[i:i+4],1) for i in range(0,len(tiles),4)],0);cv2.imwrite(str(dest),montage)
  records.append(dict(id=cid,source=s['source_path'],search_interval_seconds=[begin,end],sample_spacing_seconds=2,same_pitch_confirmed=False,status='unreviewed_context_search',caveat='Need hitter, game state, ball path and temporal alignment; vicinity alone is insufficient'))
  print('REPLAY CONTEXT',cid,flush=True)
 save(out/'manifest.json',records)
if __name__=='__main__':main()
