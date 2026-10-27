#!/usr/bin/env python3
"""Audit source identity, temporal screening, body invariants and publication gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from package_scene import records
from publication_quality import screen_record
from package_pitch_scene import checked_record,publication_decision,scene_records,publication_review


def audit(dataset,public=None,unchanged_2d=None):
    motion=scene_records(dataset);base=records(dataset/'motion_wham.csv');previous=records(dataset/'motion_wham_scene.csv')
    if set(motion)!=set(base):raise ValueError('Contact-fit index is incomplete')
    reviews=json.loads((dataset/'scene_review.json').read_text()) if (dataset/'scene_review.json').exists() else {}
    publication_index=records(dataset/'motion_wham_publication.csv')
    published={c['id']:c for c in json.loads((public/'clips.json').read_text())['clips']} if public else {}
    rows=[];count=0;max_bone_std=0.;maximum_knee_offaxis=0.
    for cid,r in motion.items():
        z,q=checked_record(dataset,cid,r,base[cid],previous[cid]);post=publication_review(dataset,base[cid],z,q);publishable,_,review=publication_decision(q,reviews.get(cid),post);count+=len(z['keypoints'])
        R=z['scene_rotation'];o=z['scene_origin'];valid=np.isfinite(z['anatomical_keypoints']).all((1,2))
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=2e-6);np.testing.assert_allclose(np.linalg.det(R),1,atol=2e-6)
        np.testing.assert_allclose((z['camera_translation']-o)@R.T,z['root_translation'],atol=6e-5,equal_nan=True)
        np.testing.assert_allclose(z['keypoints'][:,[11,12]].mean(1),z['root_translation'],atol=6e-5,equal_nan=True)
        if z['smpl_betas'].shape!=(10,):raise ValueError('Body shape varies by frame')
        from scipy.spatial.transform import Rotation
        knee=Rotation.from_matrix(z['smpl_body_pose'][valid][:,[3,4]].reshape(-1,3,3)).as_rotvec()
        if knee[:,0].min()<-.0801 or knee[:,0].max()>2.8001 or np.abs(knee[:,1:]).max()>.1801:raise ValueError('Knee constraints violated')
        maximum_knee_offaxis=max(maximum_knee_offaxis,float(np.abs(knee[:,1:]).max()))
        max_bone_std=max(max_bone_std,q['screening']['metrics']['native_bone_length_std_model_m'])
        if q['floor_reference_lower_envelope_shift_model_m']!=0:raise ValueError('Post-fit floor was shifted')
        if len(q['camera_candidates'])!=(3 if q['method']=='wham-pitch-anchor-v3' else 6):raise ValueError('Missing initialization sensitivity hypotheses')
        with np.load(dataset/base[cid]['prior_3d'],allow_pickle=False) as prior:
            if str(z['model_sha256'])!=str(prior['model_sha256']):raise ValueError('WHAM model provenance differs')
            np.testing.assert_array_equal(z['scores'],prior['scores'])
        if public:
            local=publication_index[cid]
            if local.get('preview_available')!='true' or local['scene_signature']!=r['scene_signature'] or (local['publication_accepted']=='true')!=publishable:raise ValueError('Stale local publication index')
            c=published[cid]
            public_quality=json.loads((public/c['motionQuality']).read_text())
            if not public_quality.get('preview_available') or public_quality['scene_signature']!=r['scene_signature'] or public_quality['publication_accepted']!=publishable:raise ValueError('Stale public publication decision')
            if public_quality['publication_screening']!=post:raise ValueError('Public pitch-sequence screening differs')
            if c['pose3dReasons']!=publication_decision(q,reviews.get(cid),post)[1]:raise ValueError('Public review reasons differ')
            status='screened_estimate' if publishable else 'review_required_preview'
            if c['pose3dStatus']!=status or not c.get('pose3d'):raise ValueError('Completed 3D preview not published')
            data=json.loads((public/c['pose3d']).read_text())
            if data['scene_signature']!=r['scene_signature'] or data['quality']['status']!=status:raise ValueError('Wrong public 3D version')
            if data['quality'].get('publication_accepted')!=publishable or data['quality'].get('review_reasons')!=c['pose3dReasons']:raise ValueError('Preview quality decision changed')
            if data.get('primary_representation',{}).get('frame_field')!='anatomy':raise ValueError('Native joint-center representation not declared')
            if 'pose-dependent' not in data.get('keypoints_semantics',''):raise ValueError('Surface landmark semantics missing')
            np.testing.assert_array_equal([f['source_frame'] for f in data['frames']],z['frame_index'])
            np.testing.assert_allclose([f['source_time'] for f in data['frames']],z['source_time'],atol=5.1e-7,rtol=0)
            for field,key in [('anatomy','anatomical_keypoints'),('feet','foot_keypoints')]:
                points=np.array([[p if p is not None else [np.nan]*3 for p in f[field]] for f in data['frames']])
                np.testing.assert_allclose(points,z[key],atol=5.1e-6,rtol=0,equal_nan=True)
            if c.get('strideValidated') is not False:raise ValueError('Uncalibrated stride labeled validated')
        rows.append(dict(id=cid,passed=q['fit_accepted'],published=True,quality_accepted=publishable,manual_review=review,publication_screening=post,reasons=q['screening']['reasons'],metrics=q['screening']['metrics'],
            median_reprojection_px=q['median_reprojection_px'],phase_hints=q['screening']['phase_hints'],camera_sensitivity=q['camera_sensitivity']))
    unchanged=0
    if unchanged_2d:
        for name,digest in json.loads(unchanged_2d.read_text()).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('2D data changed: '+name)
            unchanged+=1
    accepted=[r for r in rows if r['passed']]
    summary=dict(clips=len(rows),frames=count,screening_passed=len(accepted),anchor_refined_clips=sum(r['method']=='wham-pitch-anchor-v3' for r in motion.values()),published_3d=sum(r['published'] for r in rows),screened_previews=sum(r['quality_accepted'] for r in rows),review_required_previews=sum(not r['quality_accepted'] for r in rows),withheld_3d=sum(not r['published'] for r in rows),
        root_step_over_0_2_leg_all=sum(r['metrics']['root_step_max_leg']>.2 for r in rows),
        floating_over_10pct_frames_all=sum(r['metrics']['floating_ratio']>.1 for r in rows),
        max_native_bone_length_std_model_m=max_bone_std,max_knee_offaxis_radians=maximum_knee_offaxis,
        unchanged_2d_files=unchanged,public_checked=bool(public),
        interpretation='Internal consistency and stability screening only. Floor support is imposed by the model. No measured 3D or stride accuracy validation.')
    report=dict(summary=summary,records=rows)
    (dataset/'wham_pitch_scene_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(summary,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('../dataset'));p.add_argument('--public',type=Path);p.add_argument('--unchanged-2d',type=Path)
    a=p.parse_args();audit(a.dataset,a.public,a.unchanged_2d)
