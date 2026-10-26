#!/usr/bin/env python3
"""Refine cached WHAM SMPL parameters into a clip-fixed inferred support frame."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scene_geometry import VERSION,stabilize_crop
from refine_motion import atomic_npz


def read_rows(path):
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    workspace=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=workspace/'dataset')
    parser.add_argument('--wham-root',type=Path,default=workspace/'external/WHAM')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=90)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--ids',default='')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--no-background',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args();dataset=args.dataset.resolve()
    if not args.device.startswith('cuda') or args.steps<20:parser.error('CUDA and at least 20 steps required')
    import torch,cv2
    torch.set_num_threads(3);cv2.setNumThreads(1)
    from scene_fit import SceneFitter
    fitter=SceneFitter(args.wham_root,args.device)
    clips={r['dataset_clip_id']:r for r in read_rows(dataset/'clips.csv')}
    sources={r['source_id']:r for r in read_rows(dataset/'sources.csv')}
    records=read_rows(dataset/'motion_wham.csv')
    if args.ids:records=[r for r in records if r['dataset_clip_id'] in args.ids.split(',')]
    if args.limit:records=records[:args.limit]
    records=[r for i,r in enumerate(records) if i%args.shards==args.shard]
    index_name=f'motion_wham_scene.worker{args.shard}.csv' if args.shards>1 else 'motion_wham_scene.csv'
    entries={r['dataset_clip_id']:r for r in read_rows(dataset/index_name)}
    code_hash=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in
        ['scene_fit.py','scene_geometry.py','refine_wham_scene.py'])).hexdigest()
    failures=[]
    for number,record in enumerate(records,1):
        started=time.monotonic();cid=record['dataset_clip_id'];row=clips[cid]
        target=dataset/'poses/3d_scene/wham'/row['source_id']/(row['source_clip_id']+'.npz')
        quality=dataset/'quality/motion_wham_scene'/(cid+'.json')
        signature=hashlib.sha256((record['input_signature']+code_hash+str(args.steps)+str(args.no_background)).encode()).hexdigest()
        try:
            cached=False
            if target.exists() and quality.exists() and not args.force:
                with np.load(target,allow_pickle=False) as z:cached=str(z['scene_signature'])==signature
            if cached:
                diagnostics=json.loads(quality.read_text())
            else:
                with np.load(dataset/record['prior_3d'],allow_pickle=False) as z:prior={k:z[k] for k in z.files}
                with np.load(dataset/record['pose_2d_refined'],allow_pickle=False) as z:halpe={k:z[k] for k in z.files}
                if not np.array_equal(prior['frame_index'],halpe['frame_index']):raise ValueError('WHAM and Halpe frames differ')
                origin=np.array([float(row['crop_x1']),float(row['crop_y1'])]*2)
                if args.no_background:
                    background=np.repeat(np.eye(3,dtype=np.float32)[None],len(halpe['keypoints']),axis=0);ratio=0.
                else:
                    background,ratio=stabilize_crop(dataset/row['crop_video'],halpe['bbox']-origin,
                        dataset/'cache/scene_background'/(cid+'_'+str(halpe['input_signature'])[:12]+'.npz'))
                output,diagnostics=fitter.fit(prior,halpe,row,sources[row['source_id']],background,args.steps)
                diagnostics.update(dataset_clip_id=cid,method=VERSION,scene_signature=signature,
                    background_tracking_success_ratio=ratio,background_stabilization=not args.no_background,
                    inference_device=args.device,seconds=round(time.monotonic()-started,3),model_sha256=str(prior['model_sha256']),
                    review_status='unreviewed',benchmark_eligible=False)
                atomic_npz(target,**output,scores=prior['scores'],missing=~np.isfinite(output['keypoints']).all(-1),
                    source_time=halpe['source_time'],frame_index=halpe['frame_index'],fps=halpe['fps'],joints=prior['joints'],
                    model='WHAM + SMPL sequence fitting',model_sha256=prior['model_sha256'],
                    source_2d_signature=halpe['input_signature'],input_signature=prior['input_signature'],scene_signature=signature,
                    base_motion_signature=record['input_signature'],
                    method=VERSION,coordinate_system=diagnostics['coordinate_system'],fit_accepted=diagnostics['fit_accepted'],
                    background_transforms=background,provenance=prior['provenance'],inference_device=args.device,
                    score_semantics=prior['score_semantics'])
                quality.parent.mkdir(parents=True,exist_ok=True)
                temporary=quality.with_suffix('.tmp.json');temporary.write_text(json.dumps(diagnostics,indent=2,allow_nan=False));temporary.replace(quality)
            entries[cid]={'dataset_clip_id':cid,'pose_3d_scene':str(target.relative_to(dataset)),
                'scene_quality':str(quality.relative_to(dataset)),'scene_signature':signature,'method':VERSION,
                'fit_accepted':str(diagnostics['fit_accepted']).lower()}
            temporary=dataset/(index_name+'.tmp')
            with temporary.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(entries[cid]));writer.writeheader();writer.writerows(entries[k] for k in sorted(entries))
            temporary.replace(dataset/index_name)
            print(f'SCENE {number}/{len(records)} {cid} fit={diagnostics["fit_accepted"]} {time.monotonic()-started:.2f}s',flush=True)
        except Exception as error:
            import traceback
            failures.append({'dataset_clip_id':cid,'error':str(error),'traceback':traceback.format_exc()})
            print(f'SCENE FAILED {cid}: {error}',flush=True)
            (dataset/f'wham_scene_failures.worker{args.shard}.json').write_text(json.dumps(failures,indent=2))
            torch.cuda.empty_cache()
    return int(bool(failures))


if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime()
    raise SystemExit(main())
