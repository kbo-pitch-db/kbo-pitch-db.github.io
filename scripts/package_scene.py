#!/usr/bin/env python3
"""Publish completed inferred scene motion without relabeling legacy coordinates."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np


def records(path):
    if not path.exists():return {}
    with path.open() as stream:return {r['dataset_clip_id']:r for r in csv.DictReader(stream)}


def export_scene(path,quality):
    with np.load(path,allow_pickle=False) as archive:
        z={k:archive[k] for k in archive.files}
        rotation=z['scene_rotation'];origin=z['scene_origin']
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=2e-6)
        if not np.isclose(np.linalg.det(rotation),1,atol=2e-6):raise ValueError('Scene transform is not a proper rotation')
        def point(p,precision=5):
            return [round(float(v),precision) for v in p] if np.isfinite(p).all() else None
        frames=[]
        for i,points in enumerate(z['keypoints']):
            keypoints=[]
            for j,p in enumerate(points):
                value=point(p)
                keypoints.append([*value,round(float(z['scores'][i,j]),4)] if value is not None and (z['scores'][i,j]>=.2 or z['interpolated_frames'][i]) else None)
            frames.append({'frame':i,'source_frame':int(z['frame_index'][i]),'time':round(i/float(z['fps']),6),
                'source_time':round(float(z['source_time'][i]),6),'keypoints':keypoints,
                'anatomy':[point(p) for p in z['anatomical_keypoints'][i]],
                'feet':[point(p) for p in z['foot_keypoints'][i]],
                'contacts':[round(float(v),3) for v in z['contact_weights'][i]],
                'interpolated':[bool(z['interpolated_frames'][i])]*17,'outlier':[False]*17})
        return {'schema_version':2,'dimension':3,'fps':float(z['fps']),'joints':z['joints'].tolist(),
            'anatomy_joints':z['anatomical_joints'].tolist(),'anatomy_parents':z['anatomical_parents'].tolist(),
            'foot_joints':z['foot_joints'].tolist(),'coordinate_system':str(z['coordinate_system']),
            'model':str(z['model']),'model_sha256':str(z['model_sha256']),'method':str(z['method']),
            'score_semantics':str(z['score_semantics']),'scene_signature':str(z['scene_signature']),
            'scene':{'frame':'clip-fixed-inferred-support','floor':{'height':0,'normal':[0,1,0],'type':'inferred local support plane'},
                'rotation_camera_to_scene':rotation.tolist(),'origin_camera':origin.tolist(),
                'anchor_policy':quality['anchor_policy'],'calibration':'uncalibrated monocular; model-estimated scale',
                'per_frame_recentering':False,'focal_pixels_estimate':float(z['fitted_camera_focal']),
                'fit_accepted':bool(z['fit_accepted']),
                'contacts_semantics':'Soft marker-specific support hypotheses, not calibrated contact probabilities',
                'anatomy_semantics':'SMPL joint centers; procedural bone graphics are illustrative'},
            'limitations':quality['limitations'],'frames':frames}


def package_scene(dataset,output):
    motion=records(dataset/'motion_wham_scene.csv')
    if not motion:return
    base=records(dataset/'motion_wham.csv')
    index_path=output/'clips.json';index=json.loads(index_path.read_text())
    cleanup=[];qualities=[]
    for clip in index['clips']:
        record=motion.get(clip['id'])
        if record is None:continue
        path=dataset/record['pose_3d_scene']
        quality=json.loads((dataset/record['scene_quality']).read_text())
        with np.load(path,allow_pickle=False) as z:
            if str(z['scene_signature'])!=record['scene_signature']:raise ValueError('Scene index signature differs')
            if quality['scene_signature']!=record['scene_signature'] or bool(z['fit_accepted'])!=quality['fit_accepted']:
                raise ValueError('Incomplete scene/quality commit; rerun this clip with --force: '+clip['id'])
            if str(z['base_motion_signature'])!=base[clip['id']]['input_signature']:
                continue  # New source results need a new scene fit; retain the actual base label.
        expected=str(Path('assets')/clip['id']/'pose_wham_scene.json')
        qualities.append(quality)
        if clip.get('sceneSignature')==record['scene_signature'] and clip.get('pose3d')==expected and (output/expected).is_file():continue
        payload=export_scene(path,quality)
        if len(payload['frames'])!=clip['frames'] or not payload['model'].startswith('WHAM'):raise ValueError('Invalid scene provenance/alignment')
        folder=output/'assets'/clip['id'];folder.mkdir(parents=True,exist_ok=True)
        for name,data in [('pose_wham_scene.json',payload),('motion_scene_quality.json',quality)]:
            destination=folder/name;temporary=destination.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False));temporary.replace(destination)
        clip.update(pose3d=expected,model3d='WHAM + SMPL sequence fitting',motionMethod=quality['method'],
            motionQuality=str(Path('assets')/clip['id']/'motion_scene_quality.json'),sceneFrame='clip-fixed-inferred-support',
            sceneSignature=record['scene_signature'],pose3dStatus='available' if quality['fit_accepted'] else 'limited')
        cleanup.append(folder/'pose_wham.json')
    count=sum(c.get('sceneFrame')=='clip-fixed-inferred-support' for c in index['clips'])
    index['schemaVersion']=6;index['packageStats']['clipsWithSceneFit']=count
    index['packageVersion']=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    temporary=index_path.with_suffix('.tmp.json');temporary.write_text(json.dumps(index,ensure_ascii=False,indent=2));temporary.replace(index_path)
    for path in cleanup:path.unlink(missing_ok=True)
    summary={'completed_clips':count,'total_clips':len(index['clips']),
        'joint_fit_accepted':sum(q['fit_accepted'] for q in qualities),
        'status_counts':dict(Counter(c.get('pose3dStatus','unknown') for c in index['clips'])),
        'coordinate_system':'One inferred support frame per clip; no per-frame recentering',
        'interpretation':'Camera, ground and contacts are inferred. No measured stride accuracy or surveyed world coordinates.'}
    (output/'scene_fit_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Packaged scene fits: {count}/{len(index["clips"])}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('../dataset'))
    parser.add_argument('--output',type=Path,default=Path('data'));args=parser.parse_args()
    package_scene(args.dataset,args.output)
