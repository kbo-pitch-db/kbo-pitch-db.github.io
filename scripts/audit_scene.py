#!/usr/bin/env python3
"""Audit inferred clip frames, SMPL integrity, provenance and published bone motion."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as stream:return {r['dataset_clip_id']:r for r in csv.DictReader(stream)}


def audit(dataset,public=None,require_complete=False):
    clips=read(dataset/'clips.csv');base=read(dataset/'motion_wham.csv');scene=read(dataset/'motion_wham_scene.csv')
    if not set(scene)<=set(clips):raise ValueError('Inactive scene clip IDs')
    if require_complete and set(scene)!=set(clips):raise ValueError('Incomplete scene migration')
    code_hash=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in
        ['scene_fit.py','scene_geometry.py','refine_wham_scene.py'])).hexdigest()
    metrics=[];frames=0;bone_errors=[];interpolated=0;scene_floor_shifts=[]
    published={c['id']:c for c in json.loads((public/'clips.json').read_text())['clips']} if public else {}
    public_count=0
    for cid,record in scene.items():
        quality=json.loads((dataset/record['scene_quality']).read_text())
        expected=hashlib.sha256((base[cid]['input_signature']+code_hash+str(quality['steps'])+str(not quality['background_stabilization'])).encode()).hexdigest()
        if expected!=record['scene_signature']:raise ValueError('Stale scene implementation/inputs: '+cid)
        with np.load(dataset/record['pose_3d_scene'],allow_pickle=False) as archive:
            z={k:archive[k] for k in archive.files}
        count=len(z['keypoints']);frames+=count
        np.testing.assert_array_equal(z['scene_signature'],record['scene_signature'])
        if quality['scene_signature']!=record['scene_signature'] or quality['fit_accepted']!=bool(z['fit_accepted']):raise ValueError('Scene/quality provenance differs')
        np.testing.assert_array_equal(z['base_motion_signature'],base[cid]['input_signature'])
        with np.load(dataset/clips[cid]['pose_2d'],allow_pickle=False) as raw:
            np.testing.assert_array_equal(z['frame_index'],raw['frame_index'])
            np.testing.assert_array_equal(z['source_time'],raw['source_time'])
        raw_hash=hashlib.sha256((dataset/clips[cid]['pose_2d']).read_bytes()).hexdigest()
        with np.load(dataset/base[cid]['pose_2d'],allow_pickle=False) as halpe:
            if str(halpe['source_pose_sha256'])!=raw_hash or str(halpe['input_signature'])!=str(z['source_2d_signature']):raise ValueError('Stale detector source')
        rotation=z['scene_rotation'];origin=z['scene_origin']
        if rotation.shape!=(3,3) or origin.shape!=(3,):raise ValueError('Frame transform varies by frame')
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(rotation),1,atol=2e-6)
        expected_root=(z['camera_translation']-origin)@rotation.T
        np.testing.assert_allclose(expected_root,z['root_translation'],atol=5e-5,equal_nan=True)
        np.testing.assert_allclose(z['keypoints'][:,[11,12]].mean(1),z['root_translation'],atol=5e-5,equal_nan=True)
        if z['smpl_betas'].shape!=(10,):raise ValueError('Body shape is not shared by the clip')
        if z['keypoints'].shape!=(count,17,3) or z['anatomical_keypoints'].shape!=(count,24,3) or z['foot_keypoints'].shape!=(count,6,3):raise ValueError('Wrong model layouts')
        valid=np.isfinite(z['anatomical_keypoints']).all((1,2))
        if not valid.any():raise ValueError('No supported anatomy')
        for name in ['keypoints','anatomical_keypoints','foot_keypoints','root_translation']:
            if not np.isfinite(z[name][valid]).all():raise ValueError('Non-finite supported scene frame')
        bones=z['anatomical_keypoints'][valid,1:]-z['anatomical_keypoints'][valid][:,z['anatomical_parents'][1:]]
        error=float(np.std(np.linalg.norm(bones,axis=-1),axis=0).max())
        if error>5e-5:raise ValueError(f'Anatomical lengths vary: {cid}: {error}')
        bone_errors.append(error)
        if np.nanmin(z['foot_keypoints'][:,:,1])<-.0001:raise ValueError('Reference floor is above modeled feet')
        interp=z['interpolated_frames'];interpolated+=int(interp.sum())
        for group in np.split(np.flatnonzero(interp),np.flatnonzero(np.diff(np.flatnonzero(interp))>1)+1):
            if len(group)>3:raise ValueError('Long pose gap was filled')
        chosen=quality['candidate_diagnostics'][quality['chosen_camera_candidate']]
        if quality['fit_accepted']:
            metrics.append({'image_ratio':chosen['image']/max(chosen['initial']['image'],1e-12),
                'contact_velocity_ratio':chosen['contact_velocity']/max(chosen['initial']['contact_velocity'],1e-12),
                'reprojection_px':chosen['median_reprojection_px']})
        scene_floor_shifts.append(quality['floor_reference_lower_envelope_shift_model_m'])
        clip=published.get(cid)
        if clip and clip.get('sceneSignature')==record['scene_signature']:
            data=json.loads((public/clip['pose3d']).read_text());public_count+=1
            if data['scene']['per_frame_recentering'] or data['scene']['floor']['height']!=0:raise ValueError('Public frame is not fixed')
            if len(data['frames'])!=count or len(data['anatomy_joints'])!=24:raise ValueError('Public anatomy mismatch')
            np.testing.assert_allclose(data['scene']['rotation_camera_to_scene'],rotation,atol=1e-7)
            np.testing.assert_array_equal([f['source_frame'] for f in data['frames']],z['frame_index'])
            np.testing.assert_allclose([f['source_time'] for f in data['frames']],z['source_time'],atol=5.1e-7,rtol=0)
            for field,name,n in [('anatomy','anatomical_keypoints',24),('feet','foot_keypoints',6)]:
                if any(len(f[field])!=n for f in data['frames']):raise ValueError('Wrong published bone layout')
                p=np.asarray([[v if v is not None else [np.nan]*3 for v in f[field]] for f in data['frames']])
                np.testing.assert_allclose(p,z[name],atol=5.1e-6,rtol=0,equal_nan=True)
            p=np.asarray([[v if v is not None else [np.nan]*4 for v in f['keypoints']] for f in data['frames']])
            visible=np.isfinite(p[:,:,:3]).all(-1)
            expected_visible=np.isfinite(z['keypoints']).all(-1)&((z['scores']>=.2)|z['interpolated_frames'][:,None])
            np.testing.assert_array_equal(visible,expected_visible)
            np.testing.assert_allclose(p[:,:,:3][visible],z['keypoints'][visible],atol=5.1e-6,rtol=0)
            np.testing.assert_allclose(p[:,:,3][visible],z['scores'][visible],atol=5.1e-5,rtol=0)
            np.testing.assert_array_equal([f['interpolated'] for f in data['frames']],np.repeat(z['interpolated_frames'][:,None],17,axis=1))
            if not (public/clip['motionQuality']).is_file():raise ValueError('Missing scene quality link')
        elif require_complete and public:raise ValueError('Scene is not current in public data: '+cid)
    def median(key):return float(np.median([r[key] for r in metrics])) if metrics else None
    report={'completed_clips':len(scene),'total_clips':len(clips),'frames':frames,'public_scene_clips':public_count,
        'joint_fit_accepted':len(metrics),'initialization_fallback':len(scene)-len(metrics),
        'interpolated_pose_frames':interpolated,'maximum_anatomical_bone_length_std_model_m':max(bone_errors,default=0),
        'median_image_objective_ratio_after_to_initial':median('image_ratio'),
        'median_soft_contact_velocity_squared_ratio_after_to_initial':median('contact_velocity_ratio'),
        'median_reprojection_px_accepted':median('reprojection_px'),
        'median_floor_reference_shift_model_m':float(np.median(scene_floor_shifts)) if scene_floor_shifts else None,
        'checks':['current implementation and input signatures','source frame/time identity','one rigid frame per clip',
            'camera-to-scene root transform','one shared SMPL shape per clip','constant anatomical lengths',
            'bounded pose interpolation','fixed floor below modeled feet','published coordinates match NPZ'],
        'interpretation':'Numerical/fit diagnostics only. Floor, camera and scale are inferred; no measured stride accuracy, surveyed world coordinates or OpenSim IK.'}
    path=dataset/'wham_scene_audit.json';temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(path)
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('../dataset'))
    parser.add_argument('--public',type=Path);parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();audit(args.dataset,args.public,args.require_complete)
