#!/usr/bin/env python3
"""Publish screened contact fits; never substitute an old failed 3D trajectory."""
import argparse,csv,hashlib,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from package_scene import records,export_scene
from pitch_quality import screen
from publication_quality import screen_record


def expected_signature(base,previous,quality):
    code=hashlib.sha256(b''.join(Path(__file__).with_name(f).read_bytes() for f in
        ['pitch_scene_fit.py','pitch_optimizer.py','pitch_quality.py','pitch_contacts.py','refine_pitch_scene.py','scene_fit.py','scene_geometry.py'])).hexdigest()
    return hashlib.sha256((base['input_signature']+previous['scene_signature']+code+str(quality['steps'])+str(quality['focal_factors'])+str(quality['camera_starts'])).encode()).hexdigest()


def checked_record(dataset,cid,record,base,previous):
    quality=json.loads((dataset/record['scene_quality']).read_text())
    anchor=quality.get('method')=='wham-pitch-anchor-v3'
    if anchor:
        from refine_anchor_scene import signature
        seed=records(dataset/'motion_wham_pitch_scene.csv')[cid]
        seedq=json.loads((dataset/seed['scene_quality']).read_text())
        expected=signature(base,seed,seedq,quality['steps'],quality['anchor_starts'])
        if quality['base_scene_signature']!=seed['scene_signature']:raise ValueError('Stale anchor seed: '+cid)
    else:expected=expected_signature(base,previous,quality)
    if record['scene_signature']!=expected or quality['scene_signature']!=expected:raise ValueError('Stale contact fit: '+cid)
    with np.load(dataset/record['pose_3d_scene'],allow_pickle=False) as z:motion={k:z[k] for k in z.files}
    if str(motion['scene_signature'])!=expected or str(motion['base_motion_signature'])!=base['input_signature']:raise ValueError('Incomplete contact fit: '+cid)
    if anchor:
        from anchor_quality import screen as screen_anchor
        qc=screen_anchor(motion,quality)
    else:qc=screen(motion,quality)
    if qc['passed']!=quality['fit_accepted'] or qc['reasons']!=quality['screening']['reasons'] or bool(motion['fit_accepted'])!=qc['passed']:raise ValueError('Screening disagrees with stored motion: '+cid)
    with np.load(dataset/base['pose_2d_refined'],allow_pickle=False) as z:
        np.testing.assert_array_equal(motion['frame_index'],z['frame_index']);np.testing.assert_array_equal(motion['source_time'],z['source_time'])
        if str(motion['source_2d_signature'])!=str(z['input_signature']):raise ValueError('Stale 2D input: '+cid)
    return motion,quality


def scene_records(dataset):
    """Overlay current anchored fits on the complete contact-fit diagnostic index."""
    motion=records(dataset/'motion_wham_pitch_scene.csv');base=records(dataset/'motion_wham.csv');previous=records(dataset/'motion_wham_scene.csv')
    for path in [dataset/'motion_wham_pitch_anchor.csv',*sorted(dataset.glob('motion_wham_pitch_anchor.worker*.csv'))]:
        for cid,r in records(path).items():
            checked_record(dataset,cid,r,base[cid],previous[cid]);motion[cid]=r
    return motion


def publication_review(dataset,base,motion,quality):
    post=screen_record(dataset,base,motion,quality)
    if quality['fit_accepted'] and quality.get('method')!='wham-pitch-anchor-v3':
        post={**post,'passed':False,'reasons':[*post['reasons'],'stance_anchor_refinement_required']}
    post={**post,'stride_proxy_definition':'screening.phase_hints.stride_at_plant_leg is native interankle spacing at inferred plant divided by modeled leg length, not a measured rubber-to-landing stride'}
    if quality.get('lead_anchor_xz') is not None:
        distance=float(np.linalg.norm(quality['lead_anchor_xz']))
        post['fitted_stance_anchor_separation_model_m']=distance
        post['fitted_stance_anchor_separation_per_modeled_stature']=distance/quality['modeled_stature_model_m']
        post['stance_anchor_separation_semantics']='Fitted rear/lead forefoot-anchor separation; model scale and support assumptions, not measured stride'
    return post


def publication_decision(quality,review=None,post=None):
    reasons=list(quality['screening']['reasons'])
    if post:reasons.extend(post.get('reasons',[]))
    current=review if review and review.get('scene_signature')==quality['scene_signature'] else None
    if current and current.get('decision')=='withhold':reasons.append('visual_review_failed')
    elif post and post.get('passed') and quality['fit_accepted'] and (not current or current.get('decision')!='preview_only'):
        reasons.append('visual_review_pending')
    return quality['fit_accepted'] and not reasons,reasons,current


def package_pitch_scene(dataset,output,require_complete=False):
    motion=scene_records(dataset);base=records(dataset/'motion_wham.csv');previous=records(dataset/'motion_wham_scene.csv')
    index_path=output/'clips.json';index=json.loads(index_path.read_text());qualities=[];published_qualities=[];publication_rows=[]
    review_path=dataset/'scene_review.json';reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    if require_complete and set(motion)!=set(base):raise ValueError('Incomplete contact-fit dataset')
    # Validate the complete batch before changing public files.
    for cid,record in motion.items():checked_record(dataset,cid,record,base[cid],previous[cid])
    for clip in index['clips']:
        cid=clip['id'];record=motion.get(cid);folder=output/'assets'/cid;folder.mkdir(parents=True,exist_ok=True)
        if record:
            z,q=checked_record(dataset,cid,record,base[cid],previous[cid]);qualities.append(q)
            post=publication_review(dataset,base[cid],z,q)
            publishable,reasons,review=publication_decision(q,reviews.get(cid),post)
            q={**q,'publication_screening':post,'publication_accepted':bool(publishable)}
            if review:q={**q,'manual_visual_review':review}
            if publishable:q={**q,'review_status':'visual_preview_only'}
            local_quality=dataset/'quality/motion_wham_publication'/(cid+'.json');local_quality.parent.mkdir(parents=True,exist_ok=True)
            local_payload=dict(dataset_clip_id=cid,scene_signature=record['scene_signature'],fit_accepted=q['fit_accepted'],
                publication_accepted=bool(publishable),reasons=reasons,publication_screening=post,manual_visual_review=review)
            local_tmp=local_quality.with_suffix('.tmp.json');local_tmp.write_text(json.dumps(local_payload,indent=2,allow_nan=False));local_tmp.replace(local_quality)
            publication_rows.append(dict(dataset_clip_id=cid,scene_signature=record['scene_signature'],publication_accepted=str(bool(publishable)).lower(),
                review_reasons=json.dumps(reasons,separators=(',',':')),publication_quality=str(local_quality.relative_to(dataset))))
            quality_path=folder/'motion_scene_quality.json';tmp=quality_path.with_suffix('.tmp.json');tmp.write_text(json.dumps(q,separators=(',',':'),allow_nan=False));tmp.replace(quality_path)
            clip.update(model3d=('WHAM + fixed-stance SMPL sequence fitting' if q['method']=='wham-pitch-anchor-v3' else 'WHAM + contact-constrained SMPL sequence fitting'),motionMethod=q['method'],
                motionQuality=str(quality_path.relative_to(output)),sceneSignature=record['scene_signature'],
                pose3dStatus='screened_estimate' if publishable else 'review_required',pose3dReasons=reasons,
                strideValidated=False)
            if publishable:
                published_qualities.append(q)
                payload=export_scene(dataset/record['pose_3d_scene'],q)
                payload['schema_version']=3
                payload['quality']={'status':'screened_estimate','screening':q['screening'],'publication_screening':post,'manual_visual_review':review,'camera_sensitivity':q['camera_sensitivity'],
                    'validation':'Visually reviewed preview; passing screening does not validate 3D accuracy or stride'}
                payload['scene']['contacts_semantics']='2D pitch-phase and marker-specific support hypotheses; assumed at least one supporting foot, not measured contact'
                destination=folder/'pose_wham_scene.json';tmp=destination.with_suffix('.tmp.json');tmp.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False));tmp.replace(destination)
                clip.update(pose3d=str(destination.relative_to(output)),sceneFrame='clip-fixed-inferred-support')
                continue
        else:clip.update(pose3dStatus='review_required',pose3dReasons=['contact_fit_pending'],strideValidated=False,motionQuality=None,sceneSignature=None)
        clip.update(pose3d=None,sceneFrame=None)
        (folder/'pose_wham_scene.json').unlink(missing_ok=True)
        (folder/'pose_wham.json').unlink(missing_ok=True)
    if publication_rows:
        local_index=dataset/'motion_wham_publication.csv';local_tmp=local_index.with_suffix('.tmp.csv')
        with local_tmp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(publication_rows[0]));writer.writeheader();writer.writerows(publication_rows)
        local_tmp.replace(local_index)
    final_index=dataset/'motion_wham_final_scene.csv';tmp=final_index.with_suffix('.tmp.csv')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset_clip_id','pose_3d_scene','scene_quality','scene_signature','method','fit_accepted']);writer.writeheader();writer.writerows(motion[k] for k in sorted(motion))
    tmp.replace(final_index)
    passed=published_qualities
    index['schemaVersion']=7;index['packageVersion']=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    index['packageStats'].update(clipsWithContactFit=len(qualities),clipsWithSceneFit=len(passed),clipsWithScreened3d=len(passed),clipsWith3d=len(passed),clipsWithRawWham=len(base))
    tmp=index_path.with_suffix('.tmp.json');tmp.write_text(json.dumps(index,ensure_ascii=False,indent=2));tmp.replace(index_path)
    summary=dict(completed_clips=len(qualities),total_clips=len(index['clips']),screening_passed=sum(q['fit_accepted'] for q in qualities),anchor_refined_clips=sum(q['method']=='wham-pitch-anchor-v3' for q in qualities),published_3d=len(passed),
        withheld_3d=len(index['clips'])-len(passed),pending_clips=len(index['clips'])-len(qualities),
        completed_review_required=len(qualities)-len(passed),status_counts=dict(Counter(c['pose3dStatus'] for c in index['clips'])),
        rejection_reasons=dict(Counter(reason for c in index['clips'] for reason in c.get('pose3dReasons',[]) if reason!='contact_fit_pending')),
        coordinate_system='One fixed inferred support frame per clip; one shared, fixed SMPL shape',
        constraints=['2D-observed pitch contacts','fixed forefoot XYZ during inferred stance for published previews','smooth planar root spline outside anchored stance','joint-specific temporal limits',
            'bounded knee hinge','lead foot / pelvis / rear foot ordering near plant','three tilt and two focal seed initializations','three fitted stance-anchor initializations'],
        median_reprojection_px_published=float(np.median([q['median_reprojection_px'] for q in passed])) if passed else None,
        interpretation='Conservative numerical, pitch-sequence and visual screening only. Support is a model assumption; depth, scale, measured stride and genuine airborne follow-through remain unvalidated.')
    for name in ['scene_fit_summary.json','scene_reliability_summary.json']:(output/name).write_text(json.dumps(summary,indent=2))
    motion_summary=dict(clips=len(index['clips']),frames=sum(c['frames'] for c in index['clips']),
        model_counts={'WHAM':len(passed),'withheld':len(index['clips'])-len(passed)},status_counts=summary['status_counts'],
        baseline_wham_inference_clips=len(base),published_final_3d=len(passed),
        interpretation='Final 3D availability after numerical, pitch-sequence and visual preview screening. All 2D remains available. No measured 3D/stride accuracy validation.')
    (output/'motion_summary.json').write_text(json.dumps(motion_summary,indent=2))
    progress_path=output/'wham_progress.json'
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());progress.update(scope='Baseline WHAM inference plus current final publication progress',contact_refinement_completed=len(qualities),public_final_3d=len(passed),contact_pending=summary['pending_clips'])
        progress_path.write_text(json.dumps(progress,indent=2))
    historical=output/'wham_scene_audit.json'
    if historical.exists():
        old=json.loads(historical.read_text());old.update(historical_scope='Superseded v1 structural checks; acceptance did not establish motion reliability',current_report='scene_reliability_summary.json')
        historical.write_text(json.dumps(old,indent=2))
    print(json.dumps(summary,indent=2),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('../dataset'));p.add_argument('--output',type=Path,default=Path('data'));p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();package_pitch_scene(a.dataset,a.output,a.require_complete)
