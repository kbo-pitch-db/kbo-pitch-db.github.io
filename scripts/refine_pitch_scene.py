#!/usr/bin/env python3
"""Run the contact-constrained WHAM refinement, retaining v1 data for comparison."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from refine_wham_scene import read_rows
from refine_motion import atomic_npz


def main():
    workspace=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path);p.add_argument('--threads',type=int,default=1);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--output-tag',default='');p.add_argument('--dataset',type=Path,default=workspace/'dataset');p.add_argument('--wham-root',type=Path,default=workspace/'external/WHAM')
    p.add_argument('--device',default='cuda:1');p.add_argument('--memory-fraction',type=float,default=.23);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--camera-starts',default='0,4,12');p.add_argument('--ids',default='');p.add_argument('--limit',type=int,default=0);p.add_argument('--force',action='store_true');p.add_argument('--focal-factor',type=float);p.add_argument('--focal-factors',default='4,12')
    args=p.parse_args();d=args.dataset.resolve()
    starts=[float(v) for v in args.camera_starts.split(',')]
    focals=[args.focal_factor] if args.focal_factor is not None else [float(v) for v in args.focal_factors.split(',')]
    if not focals or not all(.5<=f<=30 for f in focals):p.error('Invalid focal hypotheses')
    if args.steps<20:p.error('At least 20 optimization steps are required')
    if not starts or not all(-3<=v<=28 for v in starts):p.error('Camera starts must be between -3 and 28 degrees')
    import torch,cv2
    torch.set_num_threads(args.threads);cv2.setNumThreads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,torch.device(args.device))
    from pitch_scene_fit import PitchFitter,VERSION
    from pitch_quality import screen
    from scene_geometry import stabilize_crop
    fitter=PitchFitter(args.wham_root,args.device)
    clips={r['dataset_clip_id']:r for r in read_rows(d/'clips.csv')};sources={r['source_id']:r for r in read_rows(d/'sources.csv')}
    old={r['dataset_clip_id']:r for r in read_rows(d/'motion_wham_scene.csv')};rows=read_rows(d/'motion_wham.csv')
    if args.ids:rows=[r for r in rows if r['dataset_clip_id'] in args.ids.split(',')]
    if args.limit:rows=rows[:args.limit]
    if not args.queue:rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
    total_records=len(rows)
    if args.queue:
        by_id={r['dataset_clip_id']:r for r in rows}
        def claims():
            import fcntl
            while True:
                with args.queue.open('r+') as stream:
                    fcntl.flock(stream,fcntl.LOCK_EX);state=json.load(stream)
                    if not state['pending']:return
                    cid=state['pending'].pop(0);stream.seek(0);json.dump(state,stream);stream.truncate();stream.flush()
                yield by_id[cid]
        rows=claims()
    if args.output_tag and not args.output_tag.replace('_','').isalnum():p.error('Invalid output tag')
    tag=('_'+args.output_tag) if args.output_tag else ''
    indexname='motion_wham_pitch_scene'+tag+(f'.worker{args.shard}' if args.shards>1 else '')+'.csv'
    entries={r['dataset_clip_id']:r for r in read_rows(d/indexname)}
    codehash=hashlib.sha256(b''.join(Path(__file__).with_name(f).read_bytes() for f in ['pitch_scene_fit.py','pitch_optimizer.py','pitch_quality.py','pitch_contacts.py','refine_pitch_scene.py','scene_fit.py','scene_geometry.py'])).hexdigest()
    failures=[]
    for n,r in enumerate(rows,1):
        cid=r['dataset_clip_id'];row=clips[cid];started=time.monotonic()
        signature=hashlib.sha256((r['input_signature']+old[cid]['scene_signature']+codehash+str(args.steps)+str(focals)+str(starts)).encode()).hexdigest()
        target=d/('poses/3d_pitch_scene'+tag+'/wham')/row['source_id']/(row['source_clip_id']+'.npz');quality=d/('quality/motion_wham_pitch_scene'+tag)/(cid+'.json')
        try:
            cached=False
            if target.exists() and quality.exists() and not args.force:
                with np.load(target) as z:cached=str(z['scene_signature'])==signature
                cached&=json.loads(quality.read_text()).get('scene_signature')==signature
            if cached:diagnostics=json.loads(quality.read_text())
            else:
                def load(path):
                    with np.load(d/path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
                prior=load(r['prior_3d']);halpe=load(r['pose_2d_refined']);initial=load(old[cid]['pose_3d_scene'])
                if not np.array_equal(prior['frame_index'],halpe['frame_index']):raise ValueError('Unaligned inputs')
                background=initial['background_transforms']
                attempts=[];selected=None;best_rank=None
                for focal_start,pitch_start in [(f,p) for f in focals for p in starts]:
                    candidate,fit=fitter.fit(prior,halpe,row,sources[row['source_id']],background,args.steps,initial,focal_start,pitch_start)
                    candidate.update(source_time=halpe['source_time'],frame_index=halpe['frame_index'],fps=halpe['fps'])
                    candidate_qc=screen(candidate,fit)
                    rank=(not candidate_qc['passed'],fit['objective'])
                    attempts.append(dict(focal_factor=focal_start,pitch_initial_degrees=pitch_start,camera_pitch_degrees=fit['camera_pitch_degrees'],
                        focal_pixels=float(candidate['fitted_camera_focal']),objective=fit['objective'],screening=candidate_qc,
                        median_reprojection_px=fit['median_reprojection_px']))
                    if best_rank is None or rank<best_rank:
                        selected=(candidate,fit,candidate_qc,len(attempts)-1);best_rank=rank
                output,diagnostics,qc,chosen=selected
                strides=[a['screening']['phase_hints'].get('stride_at_plant_leg') for a in attempts if a['screening']['passed']]
                strides=[v for v in strides if v is not None]
                diagnostics.update(focal_factors=focals,camera_starts=starts,camera_candidates=attempts,chosen_camera_candidate=chosen,
                    camera_sensitivity=dict(stride_at_plant_leg_range=[min(strides),max(strides)] if strides else None,
                        interpretation='Initialization sensitivity among numerically screened fits, not an uncertainty confidence interval or stride validation'))
                qc=screen(output,diagnostics)
                diagnostics.update(screening=qc,fit_accepted=qc['passed'],dataset_clip_id=cid,
                    scene_signature=signature,base_scene_signature=old[cid]['scene_signature'],inference_device=args.device,
                    review_status='unreviewed',benchmark_eligible=False,seconds=round(time.monotonic()-started,3))
                atomic_npz(target,**output,scores=prior['scores'],missing=~np.isfinite(output['keypoints']).all(-1),joints=prior['joints'],
                    model='WHAM + contact-constrained SMPL sequence fitting',model_sha256=prior['model_sha256'],
                    source_2d_signature=halpe['input_signature'],input_signature=prior['input_signature'],scene_signature=signature,
                    base_motion_signature=r['input_signature'],method=VERSION,coordinate_system=diagnostics['coordinate_system'],
                    fit_accepted=qc['passed'],background_transforms=background,provenance=prior['provenance'],inference_device=args.device,score_semantics=prior['score_semantics'])
                quality.parent.mkdir(parents=True,exist_ok=True);tmp=quality.with_suffix('.tmp.json');tmp.write_text(json.dumps(diagnostics,indent=2,allow_nan=False));tmp.replace(quality)
            entries[cid]=dict(dataset_clip_id=cid,pose_3d_scene=str(target.relative_to(d)),scene_quality=str(quality.relative_to(d)),scene_signature=signature,method=VERSION,fit_accepted=str(diagnostics['fit_accepted']).lower())
            tmp=d/(indexname+'.tmp')
            with tmp.open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=list(entries[cid]));writer.writeheader();writer.writerows(entries[k] for k in sorted(entries))
            tmp.replace(d/indexname)
            print(f'PITCH {n}/{total_records} {cid} pass={diagnostics["fit_accepted"]} {time.monotonic()-started:.2f}s '+','.join(diagnostics['screening']['reasons']),flush=True)
        except Exception as error:
            import traceback
            failures.append(dict(id=cid,error=str(error),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);torch.cuda.empty_cache()
            (d/('wham_pitch_scene_failures'+tag+f'.worker{args.shard}.json')).write_text(json.dumps(failures,indent=2))
    return int(bool(failures))


if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime();raise SystemExit(main())
