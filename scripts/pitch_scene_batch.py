#!/usr/bin/env python3
"""Run disjoint contact fits with private indexes; merge only after workers finish."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
from refine_wham_scene import read_rows


def main():
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=root/'dataset');p.add_argument('--devices',default='cpu,cpu,cpu,cpu')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--memory-fraction',type=float,default=.23)
    p.add_argument('--focal-factors',default='4,12');p.add_argument('--camera-starts',default='0,4,12');p.add_argument('--ids',default='');a=p.parse_args();d=a.dataset.resolve()
    devices=a.devices.split(',');jobs=[];logs=d/'logs/pitch_scene';logs.mkdir(parents=True,exist_ok=True)
    from package_scene import records
    queue_ids=list(records(d/'motion_wham.csv'))
    if a.ids:queue_ids=[cid for cid in queue_ids if cid in a.ids.split(',')]
    queue=logs/'queue.json';queue.write_text(json.dumps({'pending':queue_ids}))
    for shard,device in enumerate(devices):
        log=(logs/f'worker{shard}.log').open('w')
        command=[os.sys.executable,str(Path(__file__).with_name('refine_pitch_scene.py')),'--dataset',str(d),'--device',device,
            '--threads','1','--steps',str(a.steps),'--memory-fraction',str(a.memory_fraction),
            '--shard',str(shard),'--shards',str(len(devices)),'--camera-starts',a.camera_starts,'--queue',str(queue),'--focal-factors',a.focal_factors]
        if a.ids:command+=['--ids',a.ids]
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        job=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment);jobs.append((job,log))
        print(f'Started shard {shard}: {device}, pid={job.pid}',flush=True)
    while any(job.poll() is None for job,_ in jobs):
        done=sum(sum(line.startswith('PITCH ') for line in (logs/f'worker{i}.log').read_text().splitlines()) for i in range(len(jobs)))
        print(f'Completed fits this run: {done}',flush=True);time.sleep(15)
    codes=[job.returncode for job,_ in jobs]
    for _,log in jobs:log.close()
    from collect_pitch_scene import collect
    rows=collect(d)
    report=dict(worker_exit_codes=codes,completed_records=len(rows),screening_passed=sum(r['fit_accepted']=='true' for r in rows.values()),
        interpretation='Numerical/contact screening, not accuracy validation')
    (d/'wham_pitch_batch_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return int(any(codes) or not set(queue_ids)<=set(rows))


if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime();raise SystemExit(main())
