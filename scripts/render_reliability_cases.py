#!/usr/bin/env python3
"""Render numerical skeleton comparisons only; no source broadcast pixels or audio."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np
import imageio_ffmpeg
from package_scene import records
from package_pitch_scene import checked_record,scene_records
from pitch_quality import motion_metrics


def render(dataset,output):
    previous=records(dataset/'motion_wham_scene.csv');current=scene_records(dataset);base=records(dataset/'motion_wham.csv')
    public={c['id']:c for c in json.loads((output/'clips.json').read_text())['clips']}
    folder=output/'reliability';folder.mkdir(parents=True,exist_ok=True);cases=[]
    for cid in ['7ef66f7a16b0_c0007','98e127654749_c0026','ae75625b241c_c0002','210747e6a7e9_c0034']:
        b,q=checked_record(dataset,cid,current[cid],base[cid],previous[cid])
        with np.load(dataset/previous[cid]['pose_3d_scene']) as z:a={k:z[k] for k in z.files}
        old,_=motion_metrics(a);new=q['screening']['metrics'];poses=[a['anatomical_keypoints'].copy(),b['anatomical_keypoints'].copy()]
        # Each method has its own inferred frame; horizontal alignment is constant per clip.
        for p in poses:
            anchor=np.nanmedian(p[:8,0],axis=0);p[:,:,0]-=anchor[0];p[:,:,2]-=anchor[2]
        def project(p):return np.stack([p[...,2]+p[...,0]*.32,p[...,1]],-1)
        clouds=np.concatenate([project(p).reshape(-1,2) for p in poses]);lo=np.nanmin(clouds,0);hi=np.nanmax(clouds,0);lo[1]=min(lo[1],-.10);hi[1]=max(hi[1],2.1)
        scale=min(445/max(.1,hi[0]-lo[0]),410/max(.1,hi[1]-lo[1]));center=(lo[0]+hi[0])/2
        path=folder/(cid+'.mp4');fps=float(b['fps'])
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1024x600','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE)
        for i in range(len(poses[0])):
            frame=np.full((600,1024,3),(30,46,38),np.uint8)
            cv2.putText(frame,cid+' | numerical skeletons | 0.5x',(20,30),0,.62,(225,235,225),1)
            for side,p in enumerate(poses):
                offset=side*512
                cv2.putText(frame,'V1: previous scene fit' if side==0 else (('V3: fixed-stance preview' if public.get(cid,{}).get('pose3d') else 'V3: diagnostic / withheld') if q['method']=='wham-pitch-anchor-v3' else 'V2: diagnostic / withheld'),(offset+18,65),0,.58,(190,220,190),1)
                def pixel(q):
                    x,y=project(q);return int(offset+256+(x-center)*scale),int(520-(y-lo[1])*scale)
                floor=int(520+lo[1]*scale);cv2.line(frame,(offset+10,floor),(offset+502,floor),(95,150,95),2)
                for j,parent in enumerate(a['anatomical_parents']):
                    if parent<0 or not np.isfinite(p[i,[j,parent]]).all():continue
                    color=(230,125,35) if j in {1,4,7,10,13,16,18,20,22} else (40,95,220) if j in {2,5,8,11,14,17,19,21,23} else (180,180,180)
                    cv2.line(frame,pixel(p[i,parent]),pixel(p[i,j]),color,3,cv2.LINE_AA);cv2.circle(frame,pixel(p[i,j]),3,(225,230,220),-1,cv2.LINE_AA)
            cv2.putText(frame,'Fixed bounds / floor within each clip. Depth, scale and stride are unvalidated.',(20,570),0,.50,(190,200,190),1)
            for _ in range(2):proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('Comparison encoding failed')
        cases.append(dict(id=cid,video=str(path.relative_to(output)),before=old,after=new,method=q['method'],fit_accepted=q['fit_accepted'],publication_accepted=bool(public.get(cid,{}).get('pose3d')),scene_signature=q['scene_signature'],phase_hints=q['screening']['phase_hints'],camera_sensitivity=q['camera_sensitivity'],
            interpretation='Selected known failure cases, not a held-out accuracy benchmark; no measured stride or calibrated common world frame'))
    (output/'scene_reliability_cases.json').write_text(json.dumps(cases,indent=2,allow_nan=False));print('Rendered numerical comparison cases:',len(cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('../dataset'));p.add_argument('--output',type=Path,default=Path('data'));a=p.parse_args();render(a.dataset,a.output)
