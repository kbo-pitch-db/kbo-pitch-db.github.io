"""Browser-only VP9 proxies; never used for any inference or 2D fitting."""
from common import *
import subprocess,concurrent.futures,imageio_ffmpeg

def run(path):
 dest=path.with_suffix('.webm')
 if dest.exists():return
 temp=dest.with_name(dest.stem+'.tmp.webm');subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-vf','scale=min(960\\,iw):-2','-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','8','-crf','34','-b:v','0','-threads','2','-an','-y',str(temp)],check=True);temp.replace(dest);print('VP9',path.stem,flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,sorted((OUT/'original_clips').glob('*.mp4'))))
