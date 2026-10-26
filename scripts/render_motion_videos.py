#!/usr/bin/env python3
"""Render optional Halpe-26 skeleton MP4 downloads from public numeric JSON only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from build_site_data import render_pose_media


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data'))
    parser.add_argument('--limit',type=int,default=48)
    args=parser.parse_args()
    index_path=args.output/'clips.json'
    index=json.loads(index_path.read_text())
    clips=[c for c in index['clips'] if c.get('pose3dBackend')=='wham'][:args.limit or None]
    for number,clip in enumerate(clips,1):
        source=args.output/clip['pose']
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        folder=source.parent
        signature_path=folder/'pose_halpe26_video.json'
        old=json.loads(signature_path.read_text()) if signature_path.exists() else {}
        if old.get('pose_sha256')!=digest or not (folder/'pose_halpe26.mp4').exists():
            pose=json.loads(source.read_text())
            if len(pose['joints'])!=26 or pose['dimension']!=2:
                raise ValueError('Expected actual Halpe-26 2D input')
            render_pose_media(pose,folder,'pose_halpe26.mp4')
            signature_path.write_text(json.dumps({'pose_sha256':digest,'model':pose['model'],
                'frames':len(pose['frames']),'fps':pose['fps'],'mediaPolicy':'pose-only'},indent=2))
        clip['poseVideo']=str(Path('assets')/clip['id']/'pose_halpe26.mp4')
        print('VIDEO',number,'/',len(clips),clip['id'],flush=True)
    size=sum((args.output/c['poseVideo']).stat().st_size for c in index['clips']
             if c.get('poseVideo') and (args.output/c['poseVideo']).is_file())
    index['packageStats'].update(videoBytes=size,videoMegabytes=round(size/1024**2,1))
    index['packageVersion']=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    temporary=index_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    temporary.replace(index_path)


if __name__=='__main__':
    main()
