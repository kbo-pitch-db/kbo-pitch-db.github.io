#!/usr/bin/env python3
"""Run fixed-stance rescue fits for phase-resolved, camera-stable v2 sequences."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
import numpy as np
from package_scene import records


def candidates(dataset):
    from publication_quality import screen_publication
    raw=records(dataset/'motion_wham.csv');seed=records(dataset/'motion_wham_pitch_scene.csv');planned=[]
    for cid,r in seed.items():
        q=json.loads((dataset/r['scene_quality']).read_text())
        # A short stride alone is precisely a reason to attempt this new fit.
        # A camera-ambiguous seed requires a wider camera refit and remains held.
        if set(q['screening']['reasons'])-{'short_stride_depth_review'}:continue
        with np.load(dataset/r['pose_3d_scene']) as z:motion={'anatomical_keypoints':z['anatomical_keypoints']}
        with np.load(dataset/raw[cid]['pose_2d_refined']) as z:halpe={k:z[k] for k in ['keypoints','scores','fps','bbox']}
        post=screen_publication(motion,q,halpe)
        if set(post['reasons'])-{'stride_direction_depth_review'}:continue
        planned.append(dict(id=cid,seed_stride_leg=q['screening']['phase_hints'].get('stride_at_plant_leg',0),seed_reasons=q['screening']['reasons']))
    return sorted(planned,key=lambda r:(-r['seed_stride_leg'],r['id']))


def main():
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=root/'dataset');p.add_argument('--devices',default='cuda:1,cuda:1,cuda:1,cuda:1,cpu,cpu')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--memory-fraction',type=float,default=.10);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();d=a.dataset.resolve();planned=candidates(d)
    (d/'wham_anchor_candidates.json').write_text(json.dumps(planned,indent=2));print('Anchor candidates:',len(planned),flush=True)
    if a.plan_only:return 0
    logs=d/'logs/pitch_anchor';logs.mkdir(parents=True,exist_ok=True);queue=logs/'queue.json';queue.write_text(json.dumps({'pending':[r['id'] for r in planned]}))
    # Archive a stale master before workers rewrite individual scenes. The final
    # collector only accepts current signatures from their private indexes.
    master=d/'motion_wham_pitch_anchor.csv'
    if master.exists():master.replace(logs/'previous_index.csv')
    jobs=[]
    for shard,device in enumerate(a.devices.split(',')):
        log=(logs/f'worker{shard}.log').open('w');environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        command=[os.sys.executable,str(Path(__file__).with_name('refine_anchor_scene.py')),'--dataset',str(d),'--device',device,'--steps',str(a.steps),
            '--memory-fraction',str(a.memory_fraction),'--queue',str(queue),'--shard',str(shard),'--shards',str(len(a.devices.split(',')))]
        job=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment);jobs.append((job,log));print('Started anchor worker',shard,device,job.pid,flush=True)
    while any(job.poll() is None for job,_ in jobs):
        done=sum((logs/f'worker{i}.log').read_text().count('ANCHOR ') for i in range(len(jobs)))
        print('Anchor completed:',done,'/',len(planned),flush=True);time.sleep(15)
    codes=[job.returncode for job,_ in jobs]
    for _,log in jobs:log.close()
    from package_pitch_scene import scene_records
    merged={cid:r for cid,r in scene_records(d).items() if r['method']=='wham-pitch-anchor-v3'}
    tmp=d/'motion_wham_pitch_anchor.tmp.csv'
    with tmp.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['dataset_clip_id','pose_3d_scene','scene_quality','scene_signature','method','fit_accepted']);w.writeheader();w.writerows(merged[k] for k in sorted(merged))
    tmp.replace(master)
    report=dict(worker_exit_codes=codes,planned_clips=len(planned),completed_clips=len(merged),numerical_passed=sum(r['fit_accepted']=='true' for r in merged.values()),
        interpretation='Constrained estimates and numerical screening, not measured 3D accuracy')
    (d/'wham_anchor_batch_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return int(any(codes) or set(r['id'] for r in planned)!=set(merged))

if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime();raise SystemExit(main())
