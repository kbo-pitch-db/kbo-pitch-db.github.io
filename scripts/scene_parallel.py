#!/usr/bin/env python3
"""Disjoint GPU scene-fit workers; one parent publishes completed inferred frames."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from package_scene import package_scene,records
import numpy as np


def write_index(dataset,merged):
    if not merged:return
    temporary=dataset/'motion_wham_scene.tmp.csv'
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(next(iter(merged.values()))))
        writer.writeheader();writer.writerows(merged[k] for k in sorted(merged))
    temporary.replace(dataset/'motion_wham_scene.csv')


def reusable(dataset,base,code_hash,min_steps):
    """Keep a successful, current fit even when it used more refinement iterations."""
    reusable={}
    paths=[dataset/'motion_wham_scene.csv',*sorted(dataset.glob('motion_wham_scene.worker*.csv'))]
    for path in paths:
        for cid,r in records(path).items():
            if cid not in base:continue
            try:
                q=json.loads((dataset/r['scene_quality']).read_text())
                expected=hashlib.sha256((base[cid]['input_signature']+code_hash+str(q['steps'])+'False').encode()).hexdigest()
                if not q['fit_accepted'] or q['steps']<min_steps or not q['background_stabilization']:continue
                if r['scene_signature']!=expected or q['scene_signature']!=expected:continue
                with np.load(dataset/r['pose_3d_scene'],allow_pickle=False) as z:
                    if str(z['scene_signature'])!=expected or not bool(z['fit_accepted']):continue
                reusable[cid]={**r,'fit_accepted':'true'}
            except (OSError,ValueError,KeyError):continue
    return reusable


def main():
    workspace=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=workspace/'dataset')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--steps',type=int,default=90)
    parser.add_argument('--publish-every',type=int,default=50)
    parser.add_argument('--retry-steps',type=int,default=360)
    args=parser.parse_args();dataset=args.dataset.resolve();gpus=args.gpus.split(',')
    active=set(records(dataset/'clips.csv'));workers=[];logs=[];published=-1;previous=-1
    code_hash=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in
        ['scene_fit.py','scene_geometry.py','refine_wham_scene.py'])).hexdigest()
    base=records(dataset/'motion_wham.csv')
    expected={cid:hashlib.sha256((r['input_signature']+code_hash+str(args.steps)+'False').encode()).hexdigest() for cid,r in base.items()}
    cached=reusable(dataset,base,code_hash,args.steps)
    cached={k:v for k,v in cached.items() if k in active}
    pending=sorted(active-set(cached))
    print(f'Reusing {len(cached)} successful fits; processing {len(pending)} remaining clips',flush=True)
    try:
        for shard,gpu in enumerate(gpus):
            if not pending:break
            log=(dataset/f'wham_scene_gpu{gpu}.log').open('w')
            workers.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('refine_wham_scene.py')),
                '--dataset',str(dataset),'--device','cuda:'+gpu,'--shards',str(len(gpus)),
                '--shard',str(shard),'--steps',str(args.steps),'--ids',','.join(pending),'--force'],stdout=log,stderr=subprocess.STDOUT))
            logs.append(log)
        while True:
            merged=dict(cached)
            for shard in range(len(gpus)):
                for k,v in records(dataset/f'motion_wham_scene.worker{shard}.csv').items():
                    if k in active and k not in cached and v['scene_signature']==expected[k]:merged[k]=v
            finished=all(worker.poll() is not None for worker in workers)
            write_index(dataset,merged)
            if merged and (len(merged)-published>=args.publish_every or finished):
                with (dataset/'wham_publication.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    package_scene(dataset,workspace/'kbo-pitch-db.github.io/data');published=len(merged)
            if len(merged)!=previous:
                print(f'COMPLETE SCENE {len(merged)}/{len(active)}',flush=True);previous=len(merged)
            if finished:break
            time.sleep(3)
        retry_ids=[cid for cid,r in merged.items() if r['fit_accepted']!='true']
        initially_limited=list(retry_ids);retry_history=[]
        retry_code=0
        for retry_steps in [args.retry_steps,args.retry_steps*2]:
            if not retry_ids or retry_steps<=args.steps:break
            print(f'Retrying {len(retry_ids)} limited fits at {retry_steps} steps',flush=True)
            with (dataset/'wham_scene_retry.log').open('a') as retry_log:
                retry_code=subprocess.run([sys.executable,str(Path(__file__).with_name('refine_wham_scene.py')),
                    '--dataset',str(dataset),'--device','cuda:'+gpus[0],'--steps',str(retry_steps),
                    '--ids',','.join(retry_ids),'--force'],stdout=retry_log,stderr=subprocess.STDOUT).returncode
            retry_history.append({'steps':retry_steps,'ids':list(retry_ids),'exit_code':retry_code})
            merged=records(dataset/'motion_wham_scene.csv')
            with (dataset/'wham_publication.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                package_scene(dataset,workspace/'kbo-pitch-db.github.io/data')
            retry_ids=[cid for cid,r in merged.items() if r['fit_accepted']!='true']
            if retry_code:break
        report={'completed':len(merged),'total':len(active),'worker_exit_codes':[w.returncode for w in workers],
            'uncompleted_ids':sorted(active-set(merged)),'joint_fit_accepted':sum(r['fit_accepted']=='true' for r in merged.values()),
            'reused_successful_fits':len(cached),'retried_ids':initially_limited,'retry_steps':args.retry_steps,
            'retry_exit_code':retry_code,'retry_history':retry_history}
        (dataset/'wham_scene_batch_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        return int(len(merged)!=len(active) or any(w.returncode for w in workers) or retry_code)
    finally:
        for worker in workers:
            if worker.poll() is None:worker.terminate()
        for log in logs:log.close()


if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime()
    raise SystemExit(main())
