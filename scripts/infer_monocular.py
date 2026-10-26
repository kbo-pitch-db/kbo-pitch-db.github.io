#!/usr/bin/env python3
"""CPU monocular 3D priors. Run in an environment with mediapipe 0.10.x solutions.

The output is learned hip-relative depth, not triangulation or calibrated world motion.
No source images, model binaries or SMPL assets are exported.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kbo-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
MAP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def fingerprint(root, row, model_hash):
    pose = root / row['pose_2d']
    video = root / row['crop_video']
    return hashlib.sha256((hashlib.sha256(pose.read_bytes()).hexdigest() +
        str(video.stat().st_size) + str(video.stat().st_mtime_ns) +
        model_hash + ':mp-full-v1').encode()).hexdigest()


def infer(job):
    import cv2
    import numpy as np
    import mediapipe as mp
    cv2.setNumThreads(1)
    root, row, model_hash = job
    root = Path(root)
    output = root / 'poses/3d_prior' / row['source_id'] / (row['source_clip_id'] + '.npz')
    signature = fingerprint(root, row, model_hash)
    if output.exists():
        with np.load(output, allow_pickle=False) as z:
            if str(z['input_signature']) == signature:
                return row['dataset_clip_id'], 'cached'
    with np.load(root / row['pose_2d'], allow_pickle=False) as z:
        count = len(z['keypoints'])
        source_time, frame_index = z['source_time'].copy(), z['frame_index'].copy()
    xyz = np.full((count, 17, 3), np.nan, dtype=np.float32)
    xy = np.full((count, 17, 2), np.nan, dtype=np.float32)
    scores = np.zeros((count, 17), dtype=np.float32)
    capture = cv2.VideoCapture(str(root / row['crop_video']))
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        # Each clip owns a tracker; state never crosses clips/camera cuts.
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                smooth_landmarks=False, enable_segmentation=False,
                min_detection_confidence=.5, min_tracking_confidence=.5) as model:
            for i in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('crop/pose frame mismatch at %d of %d' % (i, count))
                result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.pose_world_landmarks or not result.pose_landmarks:
                    continue
                for j, landmark in enumerate(MAP):
                    w = result.pose_world_landmarks.landmark[landmark]
                    p = result.pose_landmarks.landmark[landmark]
                    xyz[i, j] = [w.x, -w.y, -w.z]  # x right, y up, z toward camera
                    xy[i, j] = [p.x * frame.shape[1] + float(row['crop_x1']),
                                p.y * frame.shape[0] + float(row['crop_y1'])]
                    scores[i, j] = min(w.visibility, p.visibility)
    finally:
        capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, keypoints=xyz, keypoints_2d=xy, scores=scores,
        source_time=source_time, frame_index=frame_index, fps=fps,
        input_signature=signature, model='MediaPipe BlazePose GHUM full',
        model_sha256=model_hash, mediapipe_version=mp.__version__,
        coordinate_system='hip-relative model meters; x right, y up, z toward camera')
    temporary.replace(output)
    return row['dataset_clip_id'], 'inferred'


def main():
    import mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--priority-index', type=Path)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    model_path = Path(mp.__file__).parent / 'modules/pose_landmark/pose_landmark_full.tflite'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    rows = list(csv.DictReader((args.dataset / 'clips.csv').open()))
    priority = set()
    if args.priority_index:
        priority = {r['id'] for r in json.loads(args.priority_index.read_text())['clips']}
    rows.sort(key=lambda r: (r['dataset_clip_id'] not in priority, r['dataset_clip_id']))
    if args.limit:
        rows = rows[:args.limit]
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(infer, (str(args.dataset.resolve()), row, model_hash)): row for row in rows}
        for i, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                cid, status = future.result()
                print('%d/%d %s %s' % (i, len(rows), cid, status), flush=True)
            except Exception as error:
                failures.append({'id': jobs[future]['dataset_clip_id'], 'error': str(error)})
                print('FAILED', failures[-1], flush=True)
    (args.dataset / 'monocular_inference.json').write_text(json.dumps({
        'model': 'MediaPipe BlazePose GHUM full', 'model_sha256': model_hash,
        'mediapipe_version': mp.__version__, 'clips': len(rows), 'failures': failures}, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
