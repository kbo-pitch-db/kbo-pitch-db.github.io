#!/usr/bin/env python3
"""Refine screened WHAM pitch sequences with fixed forefoot stance anchors.

Current v2 candidates with stable camera initialization and observed pitch order are attempted; the full v2 index remains the
baseline, and a separate anchor index overlays it. No target stride is fitted.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from package_scene import records
from refine_motion import atomic_npz

CODE_FILES=['pitch_anchor_fit.py','anchor_quality.py','refine_anchor_scene.py','pitch_optimizer.py','pitch_quality.py','pitch_contacts.py','scene_fit.py','scene_geometry.py']

def signature(raw,seed,q,steps,starts):
    code=hashlib.sha256(b''.join(Path(__file__).with_name(f).read_bytes() for f in CODE_FILES)).hexdigest()
    return hashlib.sha256((raw['input_signature']+seed['scene_signature']+code+str(steps)+str(starts)).encode()).hexdigest()

def main():
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=root/'dataset');p.add_argument('--wham-root',type=Path,default=root/'external/WHAM')
    p.add_argument('--device',default='cuda:1');p.add_argument('--steps',type=int,default=300);p.add_argument('--memory-fraction',type=float,default=.10)
    p.add_argument('--queue',type=Path);p.add_argument('--ids',default='');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();d=a.dataset.resolve();starts=[None,1.2,1.8]
    import torch,cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(a.memory_fraction,torch.device(a.device))
    from pitch_anchor_fit import AnchorFitter,VERSION
    from anchor_quality import screen
    from package_pitch_scene import checked_record
    from publication_quality import screen_publication
    fitter=AnchorFitter(a.wham_root,a.device)
    seed=records(d/'motion_wham_pitch_scene.csv');raw=records(d/'motion_wham.csv');old=records(d/'motion_wham_scene.csv')
    clips=records(d/'clips.csv');sources={r['source_id']:r for r in csv.DictReader((d/'sources.csv').open())}
    name='motion_wham_pitch_anchor'+(f'.worker{a.shard}' if a.shards>1 else '')+'.csv'
    entries=records(d/name);ids=a.ids.split(',') if a.ids else list(seed)
    def load(path):
        with np.load(d/path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    if a.queue:
        def claims():
            import fcntl
            while True:
                with a.queue.open('r+') as stream:
                    fcntl.flock(stream,fcntl.LOCK_EX);state=json.load(stream)
                    if not state['pending']:return
                    cid=state['pending'].pop(0);stream.seek(0);json.dump(state,stream);stream.truncate();stream.flush()
                yield cid
        ids=claims()
    for i,cid in enumerate(ids):
        if not a.queue and i%a.shards!=a.shard:continue
        started=time.monotonic();r=raw[cid];s=seed[cid];row=clips[cid]
        initial,seedq=checked_record(d,cid,s,r,old[cid])
        if set(seedq['screening']['reasons'])-{'short_stride_depth_review'}:continue
        h=load(r['pose_2d_refined'])
        pre=screen_publication(initial,seedq,h)
        if set(pre['reasons'])-{'stride_direction_depth_review'}:continue
        sig=signature(r,s,seedq,a.steps,starts)
        target=d/'poses/3d_pitch_anchor/wham'/row['source_id']/(row['source_clip_id']+'.npz')
        quality=d/'quality/motion_wham_pitch_anchor'/(cid+'.json')
        if target.exists() and quality.exists() and json.loads(quality.read_text()).get('scene_signature')==sig:
            q=json.loads(quality.read_text())
        else:
            prior=load(r['prior_3d']);attempts=[];best=None
            for start in starts:
                o,q=fitter.fit(prior,h,row,sources[row['source_id']],initial['background_transforms'],a.steps,initial,anchor_start=start)
                o.update(source_time=h['source_time'],frame_index=h['frame_index'],fps=h['fps'])
                qc=screen(o,q);rank=(not qc['passed'],q['objective'])
                attempts.append(dict(anchor_start_leg=start,objective=q['objective'],screening=qc,
                    camera_pitch_degrees=q['camera_pitch_degrees'],focal_pixels=float(o['fitted_camera_focal']),median_reprojection_px=q['median_reprojection_px']))
                if best is None or rank<best[0]:best=(rank,o,q,len(attempts)-1)
            _,o,q,chosen=best
            strides=[c['screening']['phase_hints'].get('stride_at_plant_leg') for c in attempts if c['screening']['passed']]
            strides=[v for v in strides if v is not None]
            q.update(anchor_starts=starts,camera_candidates=attempts,chosen_camera_candidate=chosen,
                seed_camera_sensitivity=seedq['camera_sensitivity'],camera_sensitivity=dict(stride_at_plant_leg_range=[min(strides),max(strides)] if strides else None,
                interpretation='Spread among screened anchor initializations, not a confidence interval; camera spread also checked from the six-candidate seed fit'))
            qc=screen(o,q);q.update(screening=qc,fit_accepted=qc['passed'],dataset_clip_id=cid,scene_signature=sig,
                base_scene_signature=s['scene_signature'],inference_device=a.device,review_status='unreviewed',benchmark_eligible=False,seconds=round(time.monotonic()-started,3))
            atomic_npz(target,**o,scores=prior['scores'],missing=~np.isfinite(o['keypoints']).all(-1),joints=prior['joints'],
                model='WHAM + fixed-stance SMPL sequence fitting',model_sha256=prior['model_sha256'],
                source_2d_signature=h['input_signature'],input_signature=prior['input_signature'],scene_signature=sig,
                base_motion_signature=r['input_signature'],method=VERSION,coordinate_system=q['coordinate_system'],
                fit_accepted=qc['passed'],background_transforms=initial['background_transforms'],provenance=prior['provenance'],inference_device=a.device,score_semantics=prior['score_semantics'])
            quality.parent.mkdir(parents=True,exist_ok=True);tmp=quality.with_suffix('.tmp.json');tmp.write_text(json.dumps(q,indent=2,allow_nan=False));tmp.replace(quality)
        entries[cid]=dict(dataset_clip_id=cid,pose_3d_scene=str(target.relative_to(d)),scene_quality=str(quality.relative_to(d)),scene_signature=sig,method=VERSION,fit_accepted=str(q['fit_accepted']).lower())
        tmp=d/(name+'.tmp')
        with tmp.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(entries[cid]));w.writeheader();w.writerows(entries[k] for k in sorted(entries))
        tmp.replace(d/name)
        print('ANCHOR',cid,'pass='+str(q['fit_accepted']),round(time.monotonic()-started,2),q['screening']['reasons'],flush=True)
    return 0

if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime();raise SystemExit(main())
