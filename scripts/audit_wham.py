#!/usr/bin/env python3
"""Verify WHAM/Halpe assets against original inputs and aggregate diagnostics."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def audit_public(output, dataset, clips, ids, require_complete):
    index=json.loads((output/'clips.json').read_text())
    packaged={c['id']:c for c in index['clips']}
    if set(packaged)!=set(clips):
        raise ValueError('Public and dataset clip IDs differ')
    published={c['id'] for c in index['clips'] if c.get('pose3dBackend')=='wham'}
    if not published<=ids or (require_complete and published!=ids):
        raise ValueError('Public WHAM migration differs from dataset')
    payloads=0;videos=0
    for cid in sorted(published):
        clip=packaged[cid]
        if clip['mediaPolicy']!='pose-only' or clip['model3d']!='WHAM':
            raise ValueError('Incorrect public model/media metadata: '+cid)
        for field in ('poseThumbnail','motionQuality','poseVideo'):
            if clip.get(field) and not (output/clip[field]).is_file():
                raise ValueError('Missing public file: '+clip[field])
        videos+=bool(clip.get('poseVideo'))
        with np.load(dataset/clips[cid]['pose_2d'],allow_pickle=False) as raw:
            source_frames=raw['frame_index'].copy();source_times=raw['source_time'].copy()
        if clip['frames']!=len(source_frames):
            raise ValueError('Public clip frame count differs: '+cid)
        for field,dimension,joints in [('pose',2,26),('poseRefined',2,26),('pose3dRaw',3,17),('pose3d',3,17)]:
            data=json.loads((output/clip[field]).read_text())
            if data['dimension']!=dimension or len(data['joints'])!=joints or len(data['frames'])!=clip['frames']:
                raise ValueError('Public payload layout differs: '+cid)
            np.testing.assert_array_equal([f['source_frame'] for f in data['frames']],source_frames)
            np.testing.assert_allclose([f['source_time'] for f in data['frames']],source_times,rtol=0,atol=5.1e-7)
            for f in data['frames']:
                if len(f['keypoints'])!=joints or len(f['interpolated'])!=joints or len(f['outlier'])!=joints:
                    raise ValueError('Public frame layout differs: '+cid)
                if any(p is not None and (len(p)!=dimension+1 or not all(v is not None and np.isfinite(v) for v in p))
                       for p in f['keypoints']):
                    raise ValueError('Invalid supported public joint: '+cid)
            expected='WHAM' if dimension==3 else 'RTMPose-m Halpe-26'
            if not data['model'].startswith(expected):
                raise ValueError('Incorrect public model: '+cid)
            payloads+=1
    progress=json.loads((output/'wham_progress.json').read_text())
    if progress['completed_wham']!=len(published) or progress['remaining_baseline']!=len(packaged)-len(published):
        raise ValueError('Public progress counts differ')
    return {'clips':len(published),'numerical_payloads':payloads,'optional_mp4s':videos,
            'checks':['asset links','model/layout metadata','source frame/time equality','finite supported JSON joints','progress counts']}


def audit(dataset, require_complete=False, public=None):
    with (dataset/'clips.csv').open() as stream:
        clips={r['dataset_clip_id']:r for r in csv.DictReader(stream)}
    with (dataset/'motion_wham.csv').open() as stream:
        records=list(csv.DictReader(stream))
    ids={r['dataset_clip_id'] for r in records}
    if len(ids)!=len(records) or not ids<=set(clips):
        raise ValueError('Duplicate or inactive WHAM clip IDs')
    if require_complete and ids!=set(clips):
        raise ValueError(f'WHAM migration incomplete: {len(ids)}/{len(clips)}')
    qualities=[];frames=0;devices=Counter()
    for record in records:
        row=clips[record['dataset_clip_id']]
        raw_path=dataset/row['pose_2d']
        raw_hash=hashlib.sha256(raw_path.read_bytes()).hexdigest()
        with np.load(raw_path,allow_pickle=False) as raw, \
             np.load(dataset/record['pose_2d'],allow_pickle=False) as halpe, \
             np.load(dataset/record['pose_2d_refined'],allow_pickle=False) as halpe_ref, \
             np.load(dataset/record['prior_3d'],allow_pickle=False) as prior, \
             np.load(dataset/record['pose_3d'],allow_pickle=False) as refined:
            count=len(raw['keypoints']);frames+=count
            if str(halpe['source_pose_sha256'])!=raw_hash:
                raise ValueError('Stale original input: '+record['dataset_clip_id'])
            if not str(prior['model']).startswith('WHAM'):
                raise ValueError('3D result is not WHAM')
            if str(prior['source_2d_signature'])!=str(halpe['input_signature']):
                raise ValueError('WHAM does not match these Halpe inputs')
            if len(str(prior['model_sha256']))!=64:
                raise ValueError('Missing model digest')
            for data,dimension,joints in [(halpe,2,26),(halpe_ref,2,26),(prior,3,17),(refined,3,17)]:
                if data['keypoints'].shape!=(count,joints,dimension):
                    raise ValueError('Unexpected pose shape')
                np.testing.assert_array_equal(raw['frame_index'],data['frame_index'])
                np.testing.assert_array_equal(raw['source_time'],data['source_time'])
                supported=~data['missing']
                if not np.isfinite(data['keypoints'][supported]).all():
                    raise ValueError('Non-finite supported pose')
            np.testing.assert_array_equal(halpe['scores'],halpe_ref['scores'])
            center=(refined['keypoints'][:,11]+refined['keypoints'][:,12])/2
            finite=np.isfinite(center).all(-1)
            np.testing.assert_allclose(center[finite],0,atol=1e-6)
            device=str(prior['inference_device']) if 'inference_device' in prior else 'GPU smoke/early run'
            devices[device]+=1
        qualities.append(json.loads((dataset/record['motion_quality']).read_text()))
    def median(values):
        values=[v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None
    ratios=[q['refined_acceleration_model_m_s2']/q['raw_acceleration_model_m_s2']
            for q in qualities if q['raw_acceleration_model_m_s2'] and q['refined_acceleration_model_m_s2'] is not None]
    report={'completed_clips':len(ids),'total_clips':len(clips),'frames':frames,
        'status_counts':dict(Counter(q['status'] for q in qualities)),
        'inference_devices':dict(devices),
        'median_supported_3d_joint_ratio':median([q['valid_joint_ratio'] for q in qualities]),
        'median_3d_acceleration_ratio_refined_to_raw':median(ratios),
        'median_root_relative_reprojection_px':median([q['median_root_relative_reprojection_px'] for q in qualities]),
        'checks':['raw input hashes','26-joint 2D / 17-joint 3D shapes','source frame/time equality',
                  'finite supported joints','unchanged detector confidence','hip-centered 3D','WHAM provenance'],
        'interpretation':'Tracking/stability diagnostics only; no pitching ground-truth accuracy validation.'}
    if public is not None:
        report['public']=audit_public(public,dataset,clips,ids,require_complete)
    temporary=dataset/'wham_audit.tmp.json'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
    temporary.replace(dataset/'wham_audit.json')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('../dataset'))
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--public',type=Path,help='Also audit the generated public data directory')
    args=parser.parse_args()
    audit(args.dataset,args.require_complete,args.public)
