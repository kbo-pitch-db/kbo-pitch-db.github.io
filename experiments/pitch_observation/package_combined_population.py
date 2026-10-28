"""Stage all813 actual combined fits, verify serialized contacts, then promote.

Old final 2D and model estimates are immutable. Historical WHAM scene bytes
are losslessly gzipped to keep Pages below its site-size budget; an archive
manifest records every original path, decoded SHA and Git blob identity.
"""
from common import *
from combined_population import ROOT,SITE,manifest,inputs
from stride_review import native_leg
import argparse,gzip,copy,shutil

VERSION='gvhmr-ground-motionbert-20261028-v1'
FOLDER='combined-v1'
METHOD='gvhmr-ground-motionbert-v1'


def encode(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(clean(payload),ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0))


def pack(rec,clip,chosen,stage):
    cid=rec['id'];fit=ROOT/chosen['fit_path'];audit=json.loads((ROOT/chosen['audit_path']).read_text())
    assert audit['geometry_pass'] and sha(fit)==audit['hashes']['combined']
    z=load(fit);arm=json.loads(fit.with_suffix('.json').read_text());folder=fit.parents[1]
    groundpath=folder/'ground'/fit.name;ground=json.loads(groundpath.with_suffix('.json').read_text())
    assert arm['motionbert_applied'] and arm['arm_local_rotations_preserved_after_motionbert']
    assert sha(DATA/rec['pose_2d_refined'])==rec['pose_2d_sha256']
    assert sha(SITE/'data'/rec['public_2d'])==rec['public_2d_sha256']
    previous=json.loads(gzip.decompress((SITE/'data'/rec['public_pose']).read_bytes()))
    two=json.loads((SITE/'data'/clip['poseRefined']).read_text());h=inputs(rec)
    n=len(z['frame_index']);assert n==rec['valid_frames'] and len(two['frames'])==rec['source_frames']
    normal=z['normal'].astype(float);normal/=np.linalg.norm(normal)
    x=np.array([1.,0.,0.]);x-=x@normal*normal;x/=np.linalg.norm(x)
    R=np.stack([x,normal,np.cross(x,normal)]);assert np.linalg.det(R)>.99999
    primary=load(groundpath)['primary_support_episode'];first=int(np.flatnonzero(primary>=0)[0])
    marker=ground['contact_episodes'][int(primary[first])]['marker']
    fworld=z['foot_keypoints'].astype(float)+z['translation'][:,None]
    point=fworld[first,marker];origin=point-(point@normal+float(z['pilot_support_offset']))*normal
    transformed={key:(z[key].astype(float)+z['translation'][:,None]-origin)@R.T for key in ['keypoints','anatomical_keypoints','foot_keypoints']}
    point6=lambda p:np.round(p,6).tolist()
    score=scores(h);frames=[]
    for i,twoframe in enumerate(two['frames']):
        frame={key:twoframe[key] for key in ['frame','source_frame','time','source_time']}
        if i<n:
            assert twoframe['source_frame']==int(z['frame_index'][i]) and abs(twoframe['source_time']-float(z['source_time'][i]))<1e-6
            frame.update(keypoints=[point6(p)+[round(float(score[i,j]),4)] for j,p in enumerate(transformed['keypoints'][i])],
                anatomy=point6(transformed['anatomical_keypoints'][i]),feet=point6(transformed['foot_keypoints'][i]),available=True,
                interpolated=h.get('interpolated',np.zeros_like(h['scores'],bool))[i,:17].tolist(),
                outlier=h.get('outlier',np.zeros_like(h['scores'],bool))[i,:17].tolist())
        else:
            assert previous.get('shot_validity',{}).get('confirmed_source_cut')
            frame.update(keypoints=[None]*17,anatomy=[None]*24,feet=[None]*6,available=False,unavailable_reason='source_camera_change')
        frames.append(frame)
    # Recheck the serialized geometry, including rounding, rather than trusting
    # an optimizer's intended objective or an availability label.
    leg=native_leg(z['anatomical_keypoints']);maximum=0.
    for e in ground['contact_episodes']:
        p=np.array([f['feet'][e['marker']] for f in frames[e['start']:e['end']+1]])
        diameter=float(np.linalg.norm(p[:,None]-p[None],axis=-1).max()/leg)
        assert diameter<=audit['rule']['contact_diameter_max_leg']+1e-5,cid
        maximum=max(maximum,diameter)
    assert transformed['foot_keypoints'][:,:,1].min(-1).max()/leg<=audit['rule']['minimum_foot_height_max_leg']+1e-5,cid
    assert np.maximum(-transformed['anatomical_keypoints'][:,:,1],0).max()/leg<=audit['rule']['floor_penetration_max_leg']+1e-5,cid
    inherited=[r for r in clip.get('pose3dReasons',[]) if 'background' in r or r=='possible_camera_cut']
    source_flags=[key+'_regression' for key,value in audit['source_checks'].items() if not value]
    keys=['joints','primary_representation','keypoints_semantics','anatomy_joints','anatomy_parents','foot_joints']
    pose={key:copy.deepcopy(previous[key]) for key in keys}
    pose.update(schema_version=4,dimension=3,fps=two['fps'],model='GVHMR + ground/contact IK + MotionBERT arm refinement',
        method=METHOD,model_sha256=previous['model_sha256'],source_fit_sha256=sha(fit),scene_signature=sha(fit),
        model_components=dict(gvhmr_checkpoint_sha256=previous['model_sha256'],
            motionbert_pretrained_checkpoint_sha256=manifest()['base_motionbert_checkpoint_sha256'],
            motionbert_adapted_checkpoint_sha256=manifest()['motionbert_checkpoint_sha256'],
            actual_motionbert_prediction_sha256=arm['hashes']['motionbert'],
            ground_solver_sha256=ground['config']['solver_sha256'],arm_solver_sha256=arm['hashes']['solver'],
            contact_ik_solver_sha256=arm['contact_ik_hashes']['solver'],
            rigid_shoe_solver_sha256=arm['rigid_sole_refit']['solver_sha256']),
        coordinate_system='One inferred support frame per clip; Y up; uncalibrated model meters',
        score_semantics='Original 2D observation scores only, not calibrated 3D confidence',
        scene=dict(frame='clip-fixed-inferred-support',show_floor=True,floor=dict(height=0.,estimated=True),
            rotation_camera_to_scene=R.tolist(),origin_camera=origin.tolist(),origin_source_frame=int(z['frame_index'][first]),
            per_frame_recentering=False,anchor_policy='One constant origin/rotation; native FK contact reconstruction before packaging',
            calibration='Estimated monocular camera/plane; no measured scale, mound geometry or absolute stride',
            focal_pixels_estimate=float(z['focal'])),
        quality=dict(status='estimate_unvalidated',numeric_finite=True,contact_consistency_passed=True,
            serialized_contact_diameter_max_leg=maximum,native_fk_error_max_model=audit['native_fk_error_max_model'],
            native_bone_variation_max_model=audit['native_bone_variation_max_model'],
            full_support_check_passed=True,minimum_foot_height_max_leg=audit['minimum_foot_height_max_leg'],
            native_floor_penetration_max_leg=audit['native_floor_penetration_max_leg'],
            prior_quality_reasons=clip.get('pose3dReasons',[]),inherited_observation_warnings=inherited,
            source_baseline_semantics='New GVHMR fit to corrected pitcher observations' if rec.get('subject_recovery_version') else 'Frozen pre-combination GVHMR fit on the same final 2D observations',
            constant_shoe_shape=True,sole_length_variation_max_model=audit['sole_length_variation_max_model'],
            sole_fk_error_max_model=audit['sole_fk_error_max_model'],
            shoe_rotation_step_at30fps_max_radians=audit['shoe_rotation_step_at30fps_max_radians'],
            ankle_velocity_change_at30fps_max_leg=audit['ankle_velocity_change_at30fps_max_leg'],human_review_complete=False,stride_validated=False,contact_validated=False,global_trajectory_validated=False,
            benchmark_eligible=False,strict_source_pass=audit['source_pass'],source_regression_flags=source_flags,
            valid_3d_frames=n,unavailable_3d_frames=len(frames)-n),
        fit_provenance=dict(version=VERSION,selected_fit_sha256=sha(fit),ground_fit_sha256=arm['hashes']['ground'],
            motionbert_applied=True,both_arms_processed=True,
            arm_choices=[dict(side=r['side'],hypothesis=r['selected']['name'],continuity_strength=r['selected']['strength'],
                             ray_clamped_frames=r['selected']['ray_clamped_frames'],unsupported_returns=r['selected']['unsupported_returns']) for r in arm['arms']],
            contact_episodes=ground['contact_episodes'],ground_config=ground['config'],contact_ik=arm['contact_ik'],
            full_smpl_surface_pose_correctives=False,
            additional_refinements=dict(root_depth=arm.get('root_depth_refit'),arm_refit=arm.get('arm_refit'),plane=arm.get('plane_refit'),rigid_sole_refit=arm['rigid_sole_refit'],root_trajectory=arm.get('root_trajectory_refit')),
            observation_marker_model='Local limb observation markers; sequence-fixed rigid shoes in native ankle frames',
            shoe_local_offsets=point6(z['shoe_local_offsets']),shoe_ankle_joints=z['shoe_ankle_joints'].tolist()),
        limitations=['Contact checks enforce the inferred support model; they are not independent measured KBO 3D accuracy',
            'Single-camera depth, camera, player scale, plane and absolute stride remain unvalidated',
            'All reliable source 2D observations used; no new held-out 3D accuracy claim',
            'MotionBERT conditions depth branches; wrist/palm flexion and ball release are not observed',
            'Source reprojection regressions remain explicitly recorded; broadcast slow-motion factor unknown',
            'Pitching support prior keeps one shoe near the estimated floor; it is not measured contact or flight detection'],frames=frames)
    if clip.get('subjectCorrection'):
        pose.update(subject='pitcher',subject_correction=rec.get('subject_recovery_version',previous.get('subject_correction')))
        pose['fit_provenance'].update(phase=rec['phase'],subject_input_sha256=rec['pose_2d_sha256'],
            subject_correction=clip['subjectCorrection'],shape_source=rec.get('shape_source',previous.get('fit_provenance',{}).get('shape_source')))
        if not rec.get('subject_recovery_version'):
            pose['quality']['absolute_stride']='not_observed'
        if not rec.get('subject_recovery_version'):pose['limitations'].append('This corrected pitcher clip starts after stride; leg lift and landing are outside the observed interval')
    if n<len(frames):pose['shot_validity']=copy.deepcopy(previous['shot_validity'])
    relative=f'reconstructions/{FOLDER}/{cid}.json.gz';encode(stage/'data'/relative,pose)
    if rec.get('subject_recovery_version'):
        report=json.loads((SITE/'data'/rec['subject_correction']).read_text());report['new_3d_sha256']=sha(stage/'data'/relative)
        save(stage/'data'/rec['subject_correction'],report)
    quality=f'quality/{FOLDER}/{cid}.json';save(stage/'data'/quality,dict(id=cid,version=VERSION,
        **pose['quality'],physical_checks=audit['checks'],physical_rule=audit['rule'],before=audit['before'],after=audit['after'],
        source_checks=audit['source_checks'],contact_episodes=audit['contact_episodes'],
        root_step_at_30fps_max_leg=audit['root_step_at_30fps_max_leg'],
        root_velocity_change_at30fps_max_leg=audit['root_velocity_change_at30fps_max_leg'],
        root_step_diagnostic_semantics=audit['root_step_diagnostic_semantics'],normalization_leg_model=leg,
        floor_penetration_p95_leg=audit['floor_penetration_p95_leg'],floor_penetration_max_leg=audit['floor_penetration_max_leg'],
        contact_height_max_leg=audit['contact_height_max_leg'],
        selected_fit_sha256=sha(fit)))
    clip=copy.deepcopy(clip)
    for key in ['supportAudit','postprocessingVersion']:clip.pop(key,None)
    clip.update(pose3d=relative,pose3dBackend='gvhmr-motionbert',model3d=pose['model'],motionMethod=METHOD,
        motionQuality=quality,sceneSignature=sha(fit),sceneFrame=pose['scene']['frame'],reconstructionVersion=VERSION,
        pose3dStatus='estimate_unvalidated',pose3dReasons=['monocular_3d_unvalidated']+source_flags+inherited,
        pose3dPrimaryLayout='smpl24',pose3dSupplementaryLayout='coco17')
    return clip,dict(id=cid,frames=len(frames),valid_3d_frames=n,selected_fit_sha256=sha(fit),
        gzip_sha256=sha(stage/'data'/relative),final_2d_public_sha256=rec['public_2d_sha256'],
        contact_diameter_max_leg=maximum,strict_source_pass=audit['source_pass'],source_regression_flags=source_flags,
        before=audit['before'],after=audit['after'],motionbert_prediction_sha256=arm['hashes']['motionbert'],
        native_fk_error_max_model=audit['native_fk_error_max_model'],native_bone_variation_max_model=audit['native_bone_variation_max_model'],
        sole_length_variation_max_model=audit['sole_length_variation_max_model'],sole_fk_error_max_model=audit['sole_fk_error_max_model'])


def stage_all():
    selection=json.loads((ROOT/'selection.json').read_text());assert selection['summary']['selected_geometry_pass']==813
    index=json.loads((ROOT/'original_index.json').read_text());records={r['id']:r for r in manifest()['records']}
    stage=ROOT/'site_stage';clips=[];audit=[]
    for clip in index['clips']:
        clip={**clip,**records[clip['id']].get('clip_update',{})}
        new,r=pack(records[clip['id']],clip,selection['selected'][clip['id']],stage);clips.append(new);audit.append(r)
    index.update(schemaVersion=10,packageVersion=VERSION,clips=clips)
    index['packageStats'].update(clipsWith3d=813,clipsWithGvhmr=813,clipsWithMotionBert=813,
        clipsWithContactFit=813,clipsWithSceneFit=813,clipsWithWham=0,clipsWithScreened3d=0)
    if (ROOT/'tail_recovery_v1/subject_corrections.json').exists():
        shutil.copy2(ROOT/'tail_recovery_v1/subject_corrections.json',stage/'data/subject_corrections.json')
    save(stage/'data/clips.json',index)
    summary=dict(release_id=VERSION,clips=813,players=len({c['playerKey'] for c in clips}),
        frames=sum(r['frames'] for r in audit),valid_3d_frames=sum(r['valid_3d_frames'] for r in audit),
        motionbert_clips=813,gvhmr_clips=813,contact_consistency_passed=813,contact_consistency_failed=0,
        contact_diameter_max_leg=max(r['contact_diameter_max_leg'] for r in audit),
        strict_source_pass=sum(r['strict_source_pass'] for r in audit),strict_source_regression_clips=sum(not r['strict_source_pass'] for r in audit),
        constant_shoe_shape_clips=813,sole_length_variation_max_model=max(r['sole_length_variation_max_model'] for r in audit),
        sole_fk_error_max_model=max(r['sole_fk_error_max_model'] for r in audit),
        source_regression_counts={key:sum(key+'_regression' in r['source_regression_flags'] for r in audit) for key in ['source_p50_px','wrist_p95_px','feet_p95_px']},
        original_final_2d_changed=bool(manifest().get('recovered_final_2d_clips')),
        active_final_2d_recovered_clips=manifest().get('recovered_final_2d_clips',[]),original_final_2d_assets_overwritten=False,
        human_manually_reviewed=0,absolute_3d_accuracy_validated=False,
        interpretation='All actual combined estimates pass saved-coordinate support/FK/bone/continuity screens. Source regression flags are retained; no claim of measured 3D accuracy.')
    summary['unavailable_3d_frames']=summary['frames']-summary['valid_3d_frames']
    save(stage/'data/combined_release_summary.json',summary)
    save(stage/'data/motion_summary.json',dict(summary,model_counts={'GVHMR + ground/contact IK + MotionBERT arm refinement':813},published_final_3d=813))
    save(ROOT/'package_records.json',audit);csvwrite(stage/'data/combined_release_audit.csv',audit)
    save(ROOT/'staged_release.json',summary);print(json.dumps(summary,indent=2))


def archive_wham_scenes():
    records=[]
    old=json.loads(gzip.decompress((SITE/'data/releases/wham-20261028/original_index.json.gz').read_bytes()))
    for clip in old['clips']:
        path=SITE/'data'/clip['pose3d'];dest=path.with_suffix('.json.gz')
        raw=path.read_bytes() if path.exists() else gzip.decompress(dest.read_bytes())
        compressed=gzip.compress(raw,compresslevel=9,mtime=0)
        assert gzip.decompress(compressed)==raw;dest.write_bytes(compressed)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        records.append(dict(original_path=str(path.relative_to(SITE/'data')),gzip_path=str(dest.relative_to(SITE/'data')),
            decoded_sha256=hashlib.sha256(raw).hexdigest(),original_git_blob_sha1=blob))
    assert len(records)==813
    save(SITE/'data/releases/wham-20261028/scene_pose_archive.json',records)
    for record in records:
        path=SITE/'data'/record['original_path']
        if path.exists():path.unlink()


def promote():
    stage=ROOT/'site_stage';index=json.loads((stage/'data/clips.json').read_text());summary=json.loads((stage/'data/combined_release_summary.json').read_text())
    assert len(index['clips'])==813 and summary['contact_consistency_failed']==0
    records={r['id']:r for r in json.loads((ROOT/'package_records.json').read_text())}
    for c in index['clips']:
        r=records[c['id']];assert sha(stage/'data'/c['pose3d'])==r['gzip_sha256']
        assert sha(SITE/'data'/c['poseRefined'])==r['final_2d_public_sha256']
    archive=SITE/'data/releases/support-20261028/original_index.json.gz';archive.parent.mkdir(exist_ok=True)
    old=(SITE/'data/clips.json').read_bytes();assert hashlib.sha256(old).hexdigest()==manifest()['canonical_index_sha256']
    archive.write_bytes(gzip.compress(old,compresslevel=9,mtime=0))
    for relative in [f'data/reconstructions/{FOLDER}',f'data/quality/{FOLDER}']:
        shutil.copytree(stage/relative,SITE/relative,dirs_exist_ok=True)
    for name in ['combined_release_summary.json','combined_release_audit.csv','motion_summary.json']:
        shutil.copy2(stage/'data'/name,SITE/'data'/name)
    if (stage/'data/subject_corrections').exists():shutil.copytree(stage/'data/subject_corrections',SITE/'data/subject_corrections',dirs_exist_ok=True)
    if (stage/'data/subject_corrections.json').exists():shutil.copy2(stage/'data/subject_corrections.json',SITE/'data/subject_corrections.json')
    archive_wham_scenes()
    save(SITE/'data/clips.json',index)
    save(ROOT/'promotion.json',dict(release_id=VERSION,clips=813,index_sha256=sha(SITE/'data/clips.json'),original_final_2d_assets_overwritten=False,versioned_final_2d_recoveries=manifest().get('recovered_final_2d_clips',[])))
    print('PROMOTED LOCAL',VERSION,'813 actual combined estimates; deployment still required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--promote',action='store_true');a=p.parse_args()
    if a.promote:promote()
    else:stage_all()
