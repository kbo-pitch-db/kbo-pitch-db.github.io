#!/usr/bin/env python3
"""Check completed motion assets and write aggregate stability diagnostics."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(dataset, output):
    clips = {r['dataset_clip_id']:r for r in csv.DictReader((dataset/'clips.csv').open())}
    rows = list(csv.DictReader((dataset/'motion.csv').open()))
    if {r['dataset_clip_id'] for r in rows} != set(clips):
        raise ValueError('motion index must cover exactly the active dataset')
    quality, frames = [], 0
    statuses = Counter()
    for item in rows:
        row = clips[item['dataset_clip_id']]
        raw_path = dataset/row['pose_2d']
        raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        with np.load(raw_path,allow_pickle=False) as raw:
            count = len(raw['keypoints'])
            frames += count
            if count != int(row['frames']):
                raise ValueError('frame count: '+item['dataset_clip_id'])
            for key,dim in [('pose_2d_refined',2),('pose_3d',3)]:
                if not item[key]:
                    raise ValueError('unfinished asset: '+item['dataset_clip_id']+' '+key)
                with np.load(dataset/item[key],allow_pickle=False) as data:
                    if data['keypoints'].shape != (count,17,dim):
                        raise ValueError('pose shape: '+item[key])
                    if str(data['input_sha256']) != raw_hash:
                        raise ValueError('raw input changed: '+item[key])
                    np.testing.assert_array_equal(raw['frame_index'],data['frame_index'])
                    np.testing.assert_array_equal(raw['source_time'],data['source_time'])
                    supported = ~data['missing']
                    if not np.isfinite(data['keypoints'][supported]).all():
                        raise ValueError('nonfinite supported joints: '+item[key])
                    if dim == 3:
                        root = (data['keypoints'][:,11]+data['keypoints'][:,12])/2
                        finite = np.isfinite(root).all(-1)
                        np.testing.assert_allclose(root[finite],0,atol=1e-6)
        quality.append(json.loads((dataset/item['motion_quality']).read_text()))
        statuses[item['pose_3d_status']] += 1
    def median(values):
        clean = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(clean)) if clean else None
    ratios2 = [q['refined_acceleration_px_s2']/q['raw_acceleration_px_s2']
               for q in quality if q['raw_acceleration_px_s2'] and q['refined_acceleration_px_s2'] is not None]
    ratios3 = [q['3d']['refined_acceleration_model_m_s2']/q['3d']['raw_acceleration_model_m_s2']
               for q in quality if q['3d']['raw_acceleration_model_m_s2'] and q['3d']['refined_acceleration_model_m_s2'] is not None]
    summary = {'clips':len(rows),'frames':frames,'status_counts':dict(statuses),
        'median_2d_displacement_px':median([q['median_displacement_px'] for q in quality]),
        'median_2d_acceleration_ratio_refined_to_raw':median(ratios2),
        'median_3d_acceleration_ratio_refined_to_raw':median(ratios3),
        'median_3d_supported_joint_ratio':median([q['3d']['valid_joint_ratio'] for q in quality]),
        'association_rejected_frames':sum(q['3d']['association_rejected_frames'] for q in quality),
        'model':'MediaPipe BlazePose GHUM full',
        'interpretation':'Stability diagnostics only. Acceleration reductions are per-clip medians on common supported joints, then aggregated across clips. No 3D ground truth, calibrated scale, world translation, OpenSim IK, kinetics or pitching validation.',
        'checks':['all active clip IDs indexed','raw input hashes match','source frame/time align exactly',
                  'expected COCO-17 shapes','supported coordinates finite','3D hip center at origin']}
    output.write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('../dataset'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    audit(args.dataset,args.output or args.dataset/'motion_summary.json')
