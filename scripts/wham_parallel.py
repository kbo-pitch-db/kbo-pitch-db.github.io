#!/usr/bin/env python3
"""Run disjoint CUDA workers; one parent merges indexes and publishes results."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from package_motion import package


def rows(path):
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    workspace=Path(__file__).resolve().parents[2]
    parser.add_argument('--dataset',type=Path,default=workspace/'dataset')
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--publish-every',type=int,default=20)
    parser.add_argument('--feature-batch-size',type=int,default=16)
    args=parser.parse_args()
    dataset=args.dataset.resolve();output=workspace/'kbo-pitch-db.github.io/data'
    gpus=args.gpus.split(',');workers=[];logs=[]
    try:
        for shard,gpu in enumerate(gpus):
            log=(dataset/f'wham_gpu{gpu}.log').open('w')
            command=[sys.executable,str(Path(__file__).with_name('wham_pipeline.py')),
                '--dataset',str(dataset),'--device','cuda:'+gpu,'--num-shards',str(len(gpus)),
                '--shard-index',str(shard),'--continue-on-error',
                '--feature-batch-size',str(args.feature_batch_size)]
            workers.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
            logs.append(log)
        active={r['dataset_clip_id'] for r in rows(dataset/'clips.csv')}
        merged={r['dataset_clip_id']:r for r in rows(dataset/'motion_wham.csv')
                if r['dataset_clip_id'] in active}
        published=len(merged);previous=-1
        while True:
            for shard in range(len(gpus)):
                for record in rows(dataset/f'motion_wham.worker{shard}.csv'):
                    if record['dataset_clip_id'] in active:
                        merged[record['dataset_clip_id']]=record
            if merged:
                temporary=dataset/'motion_wham.tmp.csv'
                with temporary.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(next(iter(merged.values()))))
                    writer.writeheader();writer.writerows(merged[k] for k in sorted(merged))
                temporary.replace(dataset/'motion_wham.csv')
            finished=all(worker.poll() is not None for worker in workers)
            if merged and (len(merged)-published>=args.publish_every or finished):
                with (dataset/'wham_publication.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    package(dataset,output,backend='wham');published=len(merged)
            if len(merged)!=previous:
                print(f'COMPLETE {len(merged)}/{len(active)} WHAM clips',flush=True)
                previous=len(merged)
            if finished:break
            time.sleep(5)
        failures=[]
        for shard in range(len(gpus)):
            path=dataset/f'wham_failures.worker{shard}.json'
            if path.exists():
                failures.extend(r for r in json.loads(path.read_text()) if r['dataset_clip_id'] not in merged)
        report={'completed':len(merged),'total':len(active),
                'worker_exit_codes':[worker.returncode for worker in workers],
                'failures':failures,'uncompleted_ids':sorted(active-set(merged))}
        (dataset/'wham_batch_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2),flush=True)
        return int(len(merged)!=len(active))
    finally:
        for worker in workers:
            if worker.poll() is None:worker.terminate()
        for log in logs:log.close()


if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime()
    raise SystemExit(main())
