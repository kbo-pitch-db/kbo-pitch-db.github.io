#!/usr/bin/env python3
"""Add completed numeric motion assets to an existing pose-only site bundle."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

JOINTS = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
          'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
          'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']


def export_pose(path, row, dimension, fps, raw3d=False):
    with np.load(path, allow_pickle=False) as z:
        points, scores = z['keypoints'].copy(), z['scores']
        if dimension == 2:
            points -= np.array([float(row['crop_x1']), float(row['crop_y1'])])
        elif raw3d:
            points -= ((points[:,11] + points[:,12]) / 2)[:,None,:]
        missing = z['missing'] if 'missing' in z else np.zeros(scores.shape, bool)
        if missing.ndim == 1:
            missing = np.broadcast_to(missing[:,None], scores.shape)
        interpolated = z['interpolated'] if 'interpolated' in z else np.zeros(scores.shape, bool)
        outlier = z['outlier'] if 'outlier' in z else np.zeros(scores.shape, bool)
        frames = []
        for i, values in enumerate(points):
            joints = []
            for j, point in enumerate(values):
                if missing[i,j] or not np.isfinite(point).all() or (scores[i,j] < .2 and not interpolated[i,j]):
                    joints.append(None)
                else:
                    joints.append([*np.round(point, 5 if dimension == 3 else 2).tolist(),
                                   round(float(scores[i,j]), 4) if np.isfinite(scores[i,j]) else None])
            frames.append({'frame': i, 'source_frame': int(z['frame_index'][i]),
                'time': round(i / fps, 6), 'source_time': round(float(z['source_time'][i]), 6),
                'keypoints': joints, 'interpolated': interpolated[i].tolist(),
                'outlier': outlier[i].tolist()})
        names = z['joints'].tolist() if 'joints' in z else JOINTS
        if len(names) != points.shape[1]:
            raise ValueError('Joint names do not match the actual model output')
        data = {'schema_version': 1, 'dimension': dimension, 'fps': fps, 'joints': names,
            'coordinate_system': 'crop pixels' if dimension == 2 else 'hip-relative model meters; x right, y up, z toward camera',
            'model': str(z['model']) if 'model' in z else 'RTMPose',
            'method': str(z['method']) if 'method' in z else 'raw learned monocular prior',
            'frames': frames}
        if dimension == 3:
            data['model_sha256'] = str(z['model_sha256'])
            data['limitations'] = 'Learned monocular depth and scale; no camera calibration, global translation, OpenSim IK, kinetics or pitching validation.'
        if 'edges' in z:
            data['edges'] = z['edges'].tolist()
        if 'score_semantics' in z:
            data['score_semantics'] = str(z['score_semantics'])
        return data


def package_wham(dataset, output):
    """Switch only completed, genuinely WHAM-derived clips; never relabel a baseline."""
    index_path = output/'clips.json'
    index = json.loads(index_path.read_text())
    motion_path = dataset/'motion_wham.csv'
    if not motion_path.is_file():
        raise ValueError('No completed WHAM results: run wham_pipeline.py on the GPU server first')
    with motion_path.open() as stream:
        motion = {r['dataset_clip_id']:r for r in csv.DictReader(stream)}
    if not motion:
        raise ValueError('No completed WHAM results')
    with (dataset/'clips.csv').open() as stream:
        rows = {r['dataset_clip_id']:r for r in csv.DictReader(stream)}
    cleanup = []
    for clip in index['clips']:
        item = motion.get(clip['id'])
        if item is None:
            continue
        row = rows[clip['id']]
        if row.get('pose_2d'):
            import hashlib
            raw_hash = hashlib.sha256((dataset/row['pose_2d']).read_bytes()).hexdigest()
            with np.load(dataset/item['pose_2d'],allow_pickle=False) as halpe:
                if str(halpe['source_pose_sha256'])!=raw_hash:
                    raise ValueError('Stale Halpe inputs; rerun WHAM pipeline: '+clip['id'])
        expected_pose = str(Path('assets')/clip['id']/'pose_halpe26.json')
        clip['poseFile'] = 'pose_halpe26.json'
        if (clip.get('whamSignature') == item['input_signature'] and clip.get('pose')==expected_pose
                and clip.get('pose3dBackend')=='wham'
                and all((output/clip[field]).is_file() for field in ['pose','poseRefined','pose3d','pose3dRaw'])):
            continue
        fields = [('pose',item['pose_2d'],'pose_halpe26.json',2,False),
            ('poseRefined',item['pose_2d_refined'],'pose_halpe26_refined.json',2,False),
            ('pose3d',item['pose_3d'],'pose_wham.json',3,False),
            ('pose3dRaw',item['prior_3d'],'pose_wham_raw.json',3,True)]
        # Validate all numerical payloads before switching this clip in the public index.
        clip.pop('sceneFrame',None)
        clip.pop('sceneSignature',None)
        payloads = []
        for field,source,name,dimension,raw in fields:
            data = export_pose(dataset/source,row,dimension,clip['fps'],raw)
            expected = 26 if dimension==2 else 17
            if len(data['frames']) != clip['frames'] or len(data['joints']) != expected:
                raise ValueError('WHAM/Halpe public alignment mismatch: '+clip['id'])
            if dimension==3 and not data['model'].startswith('WHAM'):
                raise ValueError('Refusing to publish a non-WHAM result as WHAM')
            if dimension==2 and data['joints'][:17]!=JOINTS:
                raise ValueError('Halpe-26 to COCO-17 mapping differs')
            payloads.append((field,name,data))
        folder = output/'assets'/clip['id']
        folder.mkdir(parents=True,exist_ok=True)
        for field,name,data in payloads:
            (folder/name).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False))
            clip[field] = str(Path('assets')/clip['id']/name)
        quality = json.loads((dataset/item['motion_quality']).read_text())
        (folder/'motion_wham_quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False))
        clip.update(pose3dBackend='wham',pose2dLayout='halpe26',pose3dLayout='coco17',
            model='RTMPose-m Halpe-26',model3d='WHAM',coordinate='Halpe-26; crop pixels',
            pose3dStatus=item['pose_3d_status'],motionMethod=item['method'],
            motionQuality=str(Path('assets')/clip['id']/'motion_wham_quality.json'),
            whamSignature=item['input_signature'])
        # Reuse only a video rendered from exactly these 26-joint numerical inputs.
        clip['poseVideo'] = ''
        video_signature=folder/'pose_halpe26_video.json'
        if video_signature.exists() and (folder/'pose_halpe26.mp4').is_file():
            import hashlib
            signature=json.loads(video_signature.read_text())
            if signature.get('pose_sha256')==hashlib.sha256((folder/'pose_halpe26.json').read_bytes()).hexdigest():
                clip['poseVideo']=str(Path('assets')/clip['id']/'pose_halpe26.mp4')
        from build_site_data import render_thumbnail
        render_thumbnail(payloads[0][2],folder)
        cleanup.extend(folder/name for name in ['pose.json','pose_refined.json','pose_3d.json',
            'pose_3d_raw.json','motion_quality.json','pose.mp4'])
    index['packageStats']['clipsWithWham'] = sum(c.get('pose3dBackend')=='wham' for c in index['clips'])
    video_bytes = sum((output/c['poseVideo']).stat().st_size for c in index['clips']
                      if c.get('poseVideo') and (output/c['poseVideo']).is_file())
    index['packageStats'].update(videoBytes=video_bytes,videoMegabytes=round(video_bytes/1024**2,1))
    index['schemaVersion'] = 5
    index['packageVersion'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    temporary = index_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    temporary.replace(index_path)
    # Dataset baselines remain intact. Remove generated, superseded public copies
    # only after switching the index so the deployment tree does not double in size.
    for path in cleanup:
        path.unlink(missing_ok=True)
    atomic_progress = output/'wham_progress.tmp.json'
    atomic_progress.write_text(json.dumps({'completed_wham':index['packageStats']['clipsWithWham'],
        'total_clips':len(index['clips']),
        'remaining_baseline':sum(c.get('pose3dBackend')!='wham' for c in index['clips']),
        'target_2d':'RTMPose Halpe-26','target_3d':'WHAM'},indent=2))
    atomic_progress.replace(output/'wham_progress.json')
    from collections import Counter
    summary_path = output/'motion_summary.json'
    if summary_path.exists() and not (output/'baseline_motion_summary.json').exists():
        (output/'baseline_motion_summary.json').write_bytes(summary_path.read_bytes())
    summary_path.write_text(json.dumps({'clips':len(index['clips']),
        'frames':sum(c['frames'] for c in index['clips']),
        'model_counts':{'WHAM':index['packageStats']['clipsWithWham'],
                        'legacy_baseline':len(index['clips'])-index['packageStats']['clipsWithWham']},
        'status_counts':dict(Counter(c.get('pose3dStatus','unknown') for c in index['clips'])),
        'interpretation':'Active public clip models. All are unreviewed monocular estimates; no pitching ground-truth accuracy validation.'},indent=2))
    print('Packaged WHAM:',index['packageStats']['clipsWithWham'],'/',len(index['clips']))
    if (dataset/'motion_wham_scene.csv').exists():
        from package_scene import package_scene
        package_scene(dataset,output)


def package(dataset, output, backend='auto'):
    if backend=='wham':
        return package_wham(dataset,output)
    index_path = output / 'clips.json'
    index = json.loads(index_path.read_text())
    clips = {r['dataset_clip_id']: r for r in csv.DictReader((dataset/'clips.csv').open())}
    motion = {r['dataset_clip_id']: r for r in csv.DictReader((dataset/'motion.csv').open())}
    wham_ids = set()
    if backend=='auto' and (dataset/'motion_wham.csv').is_file():
        with (dataset/'motion_wham.csv').open() as stream:
            wham_ids = {r['dataset_clip_id'] for r in csv.DictReader(stream)}
    completed = 0
    for clip in index['clips']:
        if clip['id'] in wham_ids:
            completed += bool(clip.get('pose3d'))
            continue
        row, item = clips[clip['id']], motion.get(clip['id'])
        if not item:
            continue
        folder = output / 'assets' / clip['id']
        folder.mkdir(parents=True, exist_ok=True)
        fields = [('poseRefined', item['pose_2d_refined'], 'pose_refined.json', 2, False),
                  ('pose3d', item['pose_3d'], 'pose_3d.json', 3, False)]
        if item['pose_3d']:
            fields.append(('pose3dRaw', str(Path('poses/3d_prior')/row['source_id']/(row['source_clip_id']+'.npz')),
                           'pose_3d_raw.json', 3, True))
        for key, source, name, dimension, raw3d in fields:
            clip.pop(key, None)
            if not source or not (dataset/source).is_file():
                continue
            data = export_pose(dataset/source, row, dimension, clip['fps'], raw3d)
            if len(data['frames']) != clip['frames']:
                raise ValueError('site/pose frame mismatch: '+clip['id'])
            (folder/name).write_text(json.dumps(data, ensure_ascii=False, separators=(',',':'), allow_nan=False))
            clip[key] = str(Path('assets')/clip['id']/name)
        quality = json.loads((dataset/item['motion_quality']).read_text())
        (folder/'motion_quality.json').write_text(json.dumps(quality, indent=2, allow_nan=False))
        clip['motionQuality'] = str(Path('assets')/clip['id']/'motion_quality.json')
        clip['pose3dStatus'] = item['pose_3d_status']
        clip['motionMethod'] = item['method']
        completed += bool(clip.get('pose3d'))
    index['schemaVersion'] = 4
    index['packageVersion'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    index['packageStats']['clipsWith3d'] = completed
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print('Packaged', completed, '3D clips /', len(index['clips']))
    if backend=='auto' and (dataset/'motion_wham.csv').is_file():
        package_wham(dataset,output)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path('../dataset'))
    ap.add_argument('--output', type=Path, default=Path('data'))
    ap.add_argument('--backend',choices=['auto','wham'],default='auto')
    args = ap.parse_args()
    package(args.dataset, args.output, args.backend)
