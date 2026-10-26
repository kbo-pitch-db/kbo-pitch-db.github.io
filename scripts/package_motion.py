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
        data = {'schema_version': 1, 'dimension': dimension, 'fps': fps, 'joints': JOINTS,
            'coordinate_system': 'crop pixels' if dimension == 2 else 'hip-relative model meters; x right, y up, z toward camera',
            'model': 'RTMPose' if dimension == 2 else str(z['model']),
            'method': str(z['method']) if 'method' in z else 'raw learned monocular prior',
            'frames': frames}
        if dimension == 3:
            data['model_sha256'] = str(z['model_sha256'])
            data['limitations'] = 'Learned monocular depth and scale; no camera calibration, global translation, OpenSim IK, kinetics or pitching validation.'
        return data


def package(dataset, output):
    index_path = output / 'clips.json'
    index = json.loads(index_path.read_text())
    clips = {r['dataset_clip_id']: r for r in csv.DictReader((dataset/'clips.csv').open())}
    motion = {r['dataset_clip_id']: r for r in csv.DictReader((dataset/'motion.csv').open())}
    completed = 0
    for clip in index['clips']:
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path('../dataset'))
    ap.add_argument('--output', type=Path, default=Path('data'))
    args = ap.parse_args()
    package(args.dataset, args.output)
