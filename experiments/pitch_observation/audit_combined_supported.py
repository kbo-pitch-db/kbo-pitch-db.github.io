"""Audit saved combined coordinates, every contact and every native bone.

These are integrity/contact-model/source-observation checks, not measured
KBO 3D accuracy. Contact reduction is objective-linked. Timing stays broadcast
timing, and camera-cut frames are never filled with an unrelated 3D estimate.
"""
from common import *
from combined_population import ROOT as PRODUCTION_ROOT,manifest,inputs
ROOT=PRODUCTION_ROOT/'foot_pose_v4'
from support_transfer_root import metrics
from independent_support import episode_metrics
from stride_review import native_leg
from pitch_study_fit import StudyFitter
from scipy.spatial.transform import Rotation
import argparse,time,traceback

RULE=dict(version='combined-population-full-support-qc-v5',minimum_foot_height_max_leg=.08,shoe_rotation_step_at30fps_max_radians=1.1,ankle_velocity_change_at30fps_max_leg=.20,contact_diameter_max_leg=.01,
    primary_contact_error_max_leg=.0001,floor_penetration_p95_leg=.02,
    floor_penetration_max_leg=.04,contact_height_max_leg=.08,
    root_step_at_30fps_max_leg=.20,root_velocity_change_at30fps_max_leg=.12,native_bone_variation_max_model=.00002,
    native_fk_error_max_model=.00002,source_p50_increase_px=2.,
    wrist_p95_increase_px=2.,feet_p95_increase_px=2.,
    new_source_unsupported_forearm_returns=0,
    interpretation='Operational source/geometry screens, not ground-truth accuracy bounds')
PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])


def audit(rec,body):
    import torch
    cid=rec['id'];path=ROOT/'combined'/f'{cid}.npz'
    if not path.exists():return None
    arm=json.loads(path.with_suffix('.json').read_text());ground_path=ROOT/'ground'/path.name
    ground=json.loads(ground_path.with_suffix('.json').read_text());dest=ROOT/'audit'/f'{cid}.json'
    hashes=dict(combined=sha(path),ground=sha(ground_path),auditor=sha(__file__))
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['hashes']==hashes and old['rule']==RULE:return old
    z=load(path);base=load(OUT/rec['source']);g=load(ground_path);h=inputs(rec);leg=native_leg(z['anatomical_keypoints']);n=len(h['frame_index'])
    assert sha(DATA/rec['pose_2d_refined'])==rec['pose_2d_sha256']
    assert np.array_equal(z['frame_index'],h['frame_index']) and np.array_equal(z['source_time'],h['source_time'])
    assert np.array_equal(z['smpl_betas'],base['smpl_betas'])
    assert arm['contact_ik']['native_segment_lengths_preserved']
    assert arm['arm_local_rotations_preserved_after_motionbert']
    assert arm['rigid_sole_refit']['constant_shape']
    assert np.isfinite(z['anatomical_keypoints']).all() and np.isfinite(z['foot_keypoints']).all()
    a=z['anatomical_keypoints'];bones=np.linalg.norm(a[:,1:]-a[:,PARENTS[1:]],axis=-1)
    bone_error=float(abs(bones-np.median(bones,axis=0)).max())
    foot=z['foot_keypoints']+z['translation'][:,None];normal=z['normal'].astype(float);normal/=np.linalg.norm(normal)
    signed=(foot*normal).sum(-1)+float(z['pilot_support_offset']);penetration=np.maximum(-signed,0)/leg
    min_foot_height=float((signed.min(-1)/leg).max())
    anatomy_signed=((a+z['translation'][:,None])*normal).sum(-1)+float(z['pilot_support_offset'])
    native_penetration=float(np.maximum(-anatomy_signed,0).max()/leg)
    contacts=episode_metrics(z,ground['contact_episodes']);diameter=max(e['diameter_leg'] for e in contacts)
    from hybrid_initializer import global_rotations
    global_joint=global_rotations(z['smpl_rotations']);ankles=z['shoe_ankle_joints']
    expected=z['anatomical_keypoints'][:,ankles]+np.einsum('tkij,kj->tki',global_joint[:,ankles],z['shoe_local_offsets'])
    sole_fk=float(abs(expected-z['foot_keypoints']).max())
    sole_lengths=np.linalg.norm(z['foot_keypoints'][:,[0,1]]-z['foot_keypoints'][:,[4,5]],axis=-1)
    sole_variation=float(np.ptp(sole_lengths,axis=0).max())
    primary=g['primary_support_episode'];ids=np.flatnonzero(primary>=0);markers=g['support_episode_markers'][primary[ids]]
    anchors=g['support_episode_anchors'][primary[ids]];primary_error=float(np.linalg.norm(foot[ids,markers]-anchors,axis=-1).max()/leg)
    height=float(signed[ids,markers].max()/leg)
    root=a[:,0]+z['translation'];step=np.linalg.norm(np.diff(root,axis=0),axis=-1)/leg
    scaled_step=float((step/(np.diff(h['source_time'])*30)).max())
    dt30=np.diff(h['source_time'])*30
    velocity=np.diff(root,axis=0)/dt30[:,None]/leg
    change=np.diff(velocity,axis=0)/((dt30[1:]+dt30[:-1])/2)[:,None]
    root_velocity_change=float(np.linalg.norm(change,axis=-1).max())
    shoe=global_joint[:,[7,8]]
    angular=Rotation.from_matrix((shoe[1:]@shoe[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).magnitude().reshape(n-1,2)/dt30[:,None]
    ankle=a[:,[7,8]]+z['translation'][:,None]
    ankle_velocity=np.diff(ankle,axis=0)/dt30[:,None,None]/leg
    ankle_change=np.diff(ankle_velocity,axis=0)/((dt30[1:]+dt30[:-1])/2)[:,None,None]
    shoe_step=float(angular.max());ankle_continuity=float(np.linalg.norm(ankle_change,axis=-1).max())
    knee=Rotation.from_matrix(z['smpl_rotations'][:,[4,5]].reshape(-1,3,3)).as_rotvec()
    before,_=metrics(base,h,base['background_transforms'],base['translation'],z['contact_weight'])
    after,_=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight'])
    body.prepare_shape(z['smpl_betas'])
    with torch.no_grad():
        _,native,_=body.model(torch.tensor(z['smpl_rotations'],dtype=torch.float32),
            torch.tensor(np.repeat(z['smpl_betas'][None],n,0),dtype=torch.float32))
    native=native.numpy();fk=float(abs((native-native[:,:1])-(a-a[:,:1])).max())
    checks=dict(non_flight=min_foot_height<=RULE['minimum_foot_height_max_leg'],native_floor=native_penetration<=RULE['floor_penetration_max_leg'],shoe_rotation_continuity=shoe_step<=RULE['shoe_rotation_step_at30fps_max_radians'],
        ankle_continuity=ankle_continuity<=RULE['ankle_velocity_change_at30fps_max_leg'],sole_rigidity=sole_variation<=.00002,sole_fk=sole_fk<=.00002,contact_diameter=diameter<=RULE['contact_diameter_max_leg'],
        primary_contact=primary_error<=RULE['primary_contact_error_max_leg'],
        floor_p95=np.percentile(penetration,95)<=RULE['floor_penetration_p95_leg'],
        floor_max=penetration.max()<=RULE['floor_penetration_max_leg'],
        contact_height=height<=RULE['contact_height_max_leg'],root_step_diagnostic=scaled_step<=RULE['root_step_at_30fps_max_leg'],
        root_continuity=root_velocity_change<=RULE['root_velocity_change_at30fps_max_leg'],
        bones=bone_error<=RULE['native_bone_variation_max_model'],fk=fk<=RULE['native_fk_error_max_model'],
        knees=bool(knee[:,0].min()>=-.0801 and knee[:,0].max()<=2.8001 and abs(knee[:,1:]).max()<=.1801),
        no_new_forearm_returns=not any(r['selected']['unsupported_returns'] for r in arm['arms']))
    source_checks={key:after[key] is not None and before[key] is not None and after[key]<=before[key]+limit for key,limit in [('source_p50_px',2.),('wrist_p95_px',2.),('feet_p95_px',2.)]}
    report=dict(id=cid,frames=n,hashes=hashes,rule=RULE,checks=checks,source_checks=source_checks,
        geometry_pass=bool(all(v for k,v in checks.items() if k!='root_step_diagnostic')),source_pass=bool(all(source_checks.values())),
        contact_diameter_max_leg=diameter,primary_contact_error_max_leg=primary_error,
        minimum_foot_height_max_leg=min_foot_height,native_floor_penetration_max_leg=native_penetration,
        root_step_diagnostic_semantics='Absolute modeled velocity is not calibrated physical speed; continuous fast motion is allowed. Root continuity checks normalized velocity changes.',
        root_velocity_change_at30fps_max_leg=root_velocity_change,contact_height_max_leg=height,floor_penetration_p95_leg=float(np.percentile(penetration,95)),
        floor_penetration_max_leg=float(penetration.max()),root_step_at_30fps_max_leg=scaled_step,
        native_bone_variation_max_model=bone_error,native_fk_error_max_model=fk,
        sole_length_variation_max_model=sole_variation,sole_fk_error_max_model=sole_fk,
        shoe_rotation_step_at30fps_max_radians=shoe_step,ankle_velocity_change_at30fps_max_leg=ankle_continuity,
        root_refit_delta_max_leg=arm['contact_ik']['root_delta_max_leg'],normalization_leg_model=leg,contact_episodes=contacts,before=before,after=after,
        motionbert_applied=arm['motionbert_applied'],body_and_feet_unchanged=arm['body_and_feet_unchanged'],
        actual_3d_accuracy_validated=False,human_review_complete=False,
        confidence_semantics='Original 2D detector scores only, never calibrated 3D confidence')
    save(dest,report);print('AUDIT',cid,'geometry',report['geometry_pass'],'source',report['source_pass'],flush=True)
    return report


def run(once=False):
    import torch
    torch.set_num_threads(1);body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    records=manifest()['records'];failures={}
    while True:
        results=[];pending=[]
        for rec in records:
            try:
                r=audit(rec,body)
                if r is None:pending.append(rec['id'])
                else:results.append(r)
            except Exception as e:
                failures[rec['id']]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        summary=dict(total=len(records),audited=len(results),pending=pending,failures=failures,
            geometry_pass=sum(r['geometry_pass'] for r in results),source_pass=sum(r['source_pass'] for r in results),
            geometry_failed_ids=[r['id'] for r in results if not r['geometry_pass']],
            source_failed_ids=[r['id'] for r in results if not r['source_pass']],
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save(ROOT/'audit_summary.json',summary)
        if not pending or once:return
        time.sleep(20)


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();run(a.once)
