"""Confidence-aware temporal refinement of measured 2D and learned monocular 3D.

OpenCap-inspired separation of image evidence, learned depth and body constraints.
This module does not implement WHAM, camera calibration, OpenSim IK or kinetics.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

VERSION = 'monocular-refine-v1'
JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle']
# The torso is constrained as well as the limbs. Face landmarks retain learned shape.
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def segments(times, tracks=None, frame_index=None):
    """Never interpolate across identity changes or source-frame discontinuities."""
    times = np.asarray(times)
    delta = np.diff(times)
    typical = np.median(delta[delta > 0]) if np.any(delta > 0) else 1 / 30
    cuts = (delta <= 0) | (delta > typical * 1.8)
    if tracks is not None:
        cuts |= np.diff(tracks) != 0
    if frame_index is not None:
        cuts |= np.diff(frame_index) != 1
    return np.split(np.arange(len(times)), np.flatnonzero(cuts) + 1)


def smooth_run(values, weights, strength=.45):
    """Solve weighted observations + acceleration penalty with a pentadiagonal solve."""
    n = len(values)
    if n < 3 or strength == 0:
        return values.copy()
    # D = second difference; build D.T @ diag(lambda) @ D without a dense matrix.
    penalty = np.full(n - 2, strength)
    base = np.zeros((5, n))
    for offset, coefficients in [(0, (1, 4, 1)), (1, (-2, -2)), (2, (1,))]:
        diagonal = np.zeros(n - offset)
        for shift, coefficient in enumerate(coefficients):
            diagonal[shift:shift + n - 2] += coefficient * penalty
        base[2 - offset, offset:] = diagonal
        base[2 + offset, :n - offset] = diagonal
    system = base.copy()
    system[2] += weights + 1e-8
    return solve_banded((2, 2), system, weights[:, None] * values,
                        check_finite=True)


def refine_trajectory(points, scores, missing, times, tracks=None,
                      frame_index=None, strength=.45, max_gap=3, threshold=None):
    """Reject only isolated spikes, fill bounded short gaps, preserve long gaps.

    Interpolation and outlier masks are separate from detector confidence; scores
    are never relabeled as accuracy probabilities. Source arrays are untouched.
    """
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    missing = np.broadcast_to(np.asarray(missing, bool), scores.shape)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(scores) & (scores >= .2) & ~missing
    output = np.full_like(points, np.nan)
    interpolated = np.zeros_like(valid)
    outlier = np.zeros_like(valid)
    groups = segments(times, tracks, frame_index)
    for group in groups:
        if not len(group):
            continue
        dt = np.diff(np.asarray(times)[group])
        fps = 1 / np.median(dt) if len(dt) else 30
        # Same physical acceleration cost at different sampling frequencies.
        penalty = strength * (fps / 30) ** 4
        for j in range(points.shape[1]):
            p, good = points[group, j].copy(), valid[group, j].copy()
            if threshold is not None and len(group) >= 3:
                middle_error = np.linalg.norm(p[1:-1] - .5 * (p[:-2] + p[2:]), axis=-1)
                neighbor_distance = np.linalg.norm(p[2:] - p[:-2], axis=-1)
                spike = (good[:-2] & good[1:-1] & good[2:] &
                         (middle_error > threshold) & (neighbor_distance < threshold * .65))
                # Consecutive candidates are motion, not an isolated bad observation.
                isolated = spike & ~np.r_[False, spike[:-1]] & ~np.r_[spike[1:], False]
                bad = np.flatnonzero(isolated) + 1
                good[bad] = False
                outlier[group[bad], j] = True
            gap = np.flatnonzero(~good)
            for run in np.split(gap, np.flatnonzero(np.diff(gap) > 1) + 1):
                if not len(run):
                    continue
                a, b = int(run[0]) - 1, int(run[-1]) + 1
                if len(run) <= max_gap and a >= 0 and b < len(group) and good[a] and good[b]:
                    alpha = ((np.asarray(times)[group[run]] - times[group[a]]) /
                             (times[group[b]] - times[group[a]]))
                    p[run] = p[a] * (1 - alpha[:, None]) + p[b] * alpha[:, None]
                    interpolated[group[run], j] = True
            supported = good | interpolated[group, j]
            support = np.flatnonzero(supported)
            for run in np.split(support, np.flatnonzero(np.diff(support) > 1) + 1):
                if not len(run):
                    continue
                weight = np.clip(scores[group[run], j], .2, 1) ** 2
                weight[interpolated[group[run], j]] = .08
                output[group[run], j] = smooth_run(p[run], weight, penalty)
    return output.astype(np.float32), interpolated, outlier


def jitter(points, times, valid=None):
    """Median acceleration magnitude; a stability diagnostic, not pose accuracy."""
    if len(points) < 3:
        return None
    dt = np.diff(times)
    velocity = np.diff(points, axis=0) / dt[:, None, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:])[:, None, None] / 2)
    value = np.linalg.norm(acceleration, axis=-1)
    if valid is not None:
        value[~(valid[:-2] & valid[1:-1] & valid[2:])] = np.nan
    finite = value[np.isfinite(value)]
    return float(np.median(finite)) if finite.size else None


def constrain_bones(points, scores, strength=1.0, iterations=20):
    result = points.astype(float).copy()
    targets = []
    for a, b in BONES:
        length = np.linalg.norm(points[:, b] - points[:, a], axis=-1)
        reliable = (scores[:, a] > .5) & (scores[:, b] > .5) & np.isfinite(length)
        pool = length[reliable] if np.any(reliable) else length[np.isfinite(length)]
        targets.append(float(np.median(pool)) if len(pool) else np.nan)
    # Symmetric limbs share a clip-specific length estimate (no forced population size).
    for left, right in [(1, 3), (2, 4), (5, 6), (8, 10), (9, 11)]:
        if np.isfinite(targets[left]) and np.isfinite(targets[right]):
            targets[left] = targets[right] = (targets[left] + targets[right]) / 2
    for _ in range(iterations):
        for (a, b), target in zip(BONES, targets):
            if not np.isfinite(target):
                continue
            vector = result[:, b] - result[:, a]
            length = np.linalg.norm(vector, axis=-1)
            good = np.isfinite(length) & (length > 1e-6)
            correction = vector[good] * ((length[good] - target) / length[good])[:, None] * strength
            wa = np.clip(scores[good, a], .2, 1)
            wb = np.clip(scores[good, b], .2, 1)
            result[good, a] += correction * (wb / (wa + wb))[:, None]
            result[good, b] -= correction * (wa / (wa + wb))[:, None]
    # Keep pelvis at the origin in every supported frame.
    root = (result[:, 11] + result[:, 12]) / 2
    result -= root[:, None, :]
    return result.astype(np.float32), np.asarray(targets, np.float32)


def refine_3d(prior, prior_scores, xy, scores, times, tracks, frame_index,
              strength=.45, max_gap=3):
    root = .5 * (prior[:, 11] + prior[:, 12])
    centered = prior - root[:, None, :]
    smooth, filled, spikes = refine_trajectory(centered, prior_scores,
        ~np.isfinite(centered).all(-1), times, tracks, frame_index,
        strength, max_gap, threshold=.25)
    root2d = .5 * (xy[:, 11] + xy[:, 12])
    observed = xy - root2d[:, None, :]
    observed[:, :, 1] *= -1
    # Robust per-clip weak-perspective scale. Depth comes only from the learned model.
    scales = []
    for a, b in [(5, 11), (6, 12), (5, 6), (11, 12)]:
        pixel_length = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)
        model_length = np.linalg.norm(smooth[:, a, :2] - smooth[:, b, :2], axis=-1)
        good = (scores[:, a] > .5) & (scores[:, b] > .5) & (model_length > .06)
        ratios = pixel_length[good] / model_length[good]
        scales.extend(ratios[np.isfinite(ratios)].tolist())
    scale = float(np.median(scales)) if scales else None
    fitted = smooth.copy()
    if scale is not None and scale > 1:
        image_target = observed / scale
        discrepancy = np.linalg.norm(image_target - smooth[:, :, :2], axis=-1)
        # Ignore gross disagreement (possible wrong-person association).
        weight = .35 * np.clip(scores, 0, 1) ** 2 * np.clip(1 - discrepancy / .4, 0, 1)
        weight[~np.isfinite(weight)] = 0
        good = np.isfinite(image_target).all(-1) & np.isfinite(smooth).all(-1)
        fitted[:, :, :2][good] += weight[good, None] * (image_target[good] - smooth[:, :, :2][good])
    result, lengths = constrain_bones(fitted, prior_scores)
    valid = np.isfinite(result).all(-1)
    reprojection = None
    if scale is not None:
        error = np.linalg.norm(result[:, :, :2] * scale - observed, axis=-1)
        supported = valid & (scores > .5)
        if supported.any():
            reprojection = float(np.nanmedian(error[supported]))
    diagnostics = {
        'valid_joint_ratio': float(valid.mean()),
        'interpolated_joints': int(filled.sum()), 'outlier_joints': int(spikes.sum()),
        'weak_perspective_pixels_per_model_meter': scale,
        'median_root_relative_reprojection_px': reprojection,
        'raw_acceleration_model_m_s2': jitter(centered, times, valid),
        'refined_acceleration_model_m_s2': jitter(result, times, valid),
        'status': 'available' if valid.mean() >= .8 else 'limited',
    }
    return result, filled, spikes, lengths, diagnostics


def process_dataset(dataset, strength=.45, max_gap=3, clip_ids=None):
    dataset = Path(dataset)
    rows = list(csv.DictReader((dataset / 'clips.csv').open()))
    prior_index = dataset / 'motion.csv'
    entries = {r['dataset_clip_id']: r for r in csv.DictReader(prior_index.open())} if prior_index.exists() else {}
    for number, row in enumerate(rows, 1):
        cid = row['dataset_clip_id']
        if clip_ids and cid not in clip_ids:
            continue
        path = dataset / row['pose_2d']
        with np.load(path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        times = raw['source_time']
        xy = raw['keypoints']
        torso = np.linalg.norm(.5 * (xy[:, 5] + xy[:, 6]) - .5 * (xy[:, 11] + xy[:, 12]), axis=-1)
        finite_torso = torso[np.isfinite(torso) & (torso > 1)]
        threshold = max(20, .35 * np.median(finite_torso)) if len(finite_torso) else 50
        missing = raw.get('missing', np.zeros(xy.shape[:2], bool))
        if missing.ndim == 1:
            missing = missing[:, None]
        tracks, frames = raw.get('track_id'), raw.get('frame_index')
        clean, filled, spikes = refine_trajectory(xy, raw['scores'], missing, times,
            tracks, frames, strength, max_gap, threshold)
        rel2 = Path('poses/2d_refined') / row['source_id'] / (row['source_clip_id'] + '.npz')
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        # Normalization fields describe raw coordinates and must not masquerade as refined.
        refined_fields = {k:v for k,v in raw.items() if k not in ('normalized','normalized_valid')}
        atomic_npz(dataset / rel2, **{**refined_fields, 'keypoints': clean,
            'missing': ~np.isfinite(clean).all(-1), 'interpolated': filled,
            'outlier': spikes, 'input_sha256': signature, 'method': VERSION,
            'strength': strength, 'max_gap': max_gap})
        common = np.isfinite(clean).all(-1) & np.isfinite(xy).all(-1) & ~np.broadcast_to(missing, xy.shape[:2])
        diagnostic = {'method': VERSION, 'strength': strength, 'max_gap': max_gap,
            'raw_2d_sha256': signature, 'interpolated_2d_joints': int(filled.sum()),
            'outlier_2d_joints': int(spikes.sum()),
            'raw_acceleration_px_s2': jitter(xy, times, common),
            'refined_acceleration_px_s2': jitter(clean, times, common),
            'median_displacement_px': float(np.nanmedian(np.linalg.norm(clean[common] - xy[common], axis=-1))) if common.any() else None,
            'interpretation': 'Stability diagnostics only; no 3D ground truth or pitching validation.'}
        prior_path = dataset / 'poses/3d_prior' / row['source_id'] / (row['source_clip_id'] + '.npz')
        rel3 = ''
        status = 'pending_prior'
        if prior_path.exists():
            with np.load(prior_path, allow_pickle=False) as prior:
                video = dataset / row['crop_video']
                expected = hashlib.sha256((signature + str(video.stat().st_size) +
                    str(video.stat().st_mtime_ns) + str(prior['model_sha256']) + ':mp-full-v1').encode()).hexdigest()
                if str(prior['input_signature']) != expected:
                    raise ValueError('stale 3D prior; rerun inference: ' + cid)
                if len(prior['keypoints']) != len(xy) or not np.array_equal(prior['frame_index'], frames):
                    raise ValueError('3D prior is not aligned: ' + cid)
                if not np.allclose(prior['source_time'], times, atol=1e-6):
                    raise ValueError('3D prior source times differ: ' + cid)
                prior_scores = prior['scores'].copy()
                disagreement = np.linalg.norm(prior['keypoints_2d'][:,5:] - xy[:,5:], axis=-1)
                comparable = (prior_scores[:,5:] > .5) & (raw['scores'][:,5:] > .5)
                disagreement[~comparable] = np.nan
                frame_error = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).sum() >= 4 else np.nan for v in disagreement])
                wrong_person = frame_error > np.maximum(torso * .65, 35)
                prior_scores[wrong_person] = 0
                result, fill3, spike3, lengths, quality = refine_3d(prior['keypoints'],
                    prior_scores, clean, raw['scores'], times, tracks, frames, strength, max_gap)
                quality['association_rejected_frames'] = int(wrong_person.sum())
                torso_scale = float(np.median(finite_torso)) if len(finite_torso) else None
                relative_error = quality['median_root_relative_reprojection_px']
                quality['median_reprojection_torso_ratio'] = relative_error / torso_scale if relative_error is not None and torso_scale else None
                if quality['median_reprojection_torso_ratio'] is None or quality['median_reprojection_torso_ratio'] > .35:
                    quality['status'] = 'limited'
                rel3 = Path('poses/3d') / row['source_id'] / (row['source_clip_id'] + '.npz')
                atomic_npz(dataset / rel3, keypoints=result, scores=prior_scores,
                    missing=~np.isfinite(result).all(-1), interpolated=fill3, outlier=spike3,
                    frame_index=frames, source_time=times, fps=prior['fps'],
                    model=prior['model'], model_sha256=prior['model_sha256'],
                    input_sha256=signature, method=VERSION, strength=strength, max_gap=max_gap,
                    coordinate_system='hip-relative model meters; x right, y up, z toward camera',
                    bone_edges=np.asarray(BONES), bone_lengths=lengths)
                diagnostic['3d'] = {**quality, 'model': str(prior['model']),
                                     'model_sha256': str(prior['model_sha256'])}
                status = quality['status']
        quality_rel = Path('quality/motion') / (cid + '.json')
        (dataset / quality_rel).parent.mkdir(parents=True, exist_ok=True)
        temporary_quality = (dataset / quality_rel).with_suffix('.tmp.json')
        temporary_quality.write_text(json.dumps(diagnostic, indent=2, allow_nan=False))
        temporary_quality.replace(dataset / quality_rel)
        entries[cid] = {'dataset_clip_id': cid, 'pose_2d_refined': str(rel2),
            'pose_3d': str(rel3), 'motion_quality': str(quality_rel),
            'pose_3d_status': status, 'method': VERSION,
            'review_status': 'unreviewed', 'benchmark_eligible': 'false'}
        if number % 50 == 0:
            print('REFINED', number, '/', len(rows), flush=True)
    active = {r['dataset_clip_id'] for r in rows}
    entries = {k: v for k, v in entries.items() if k in active}
    if entries:
        temp = prior_index.with_suffix('.tmp.csv')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(next(iter(entries.values()))))
            writer.writeheader()
            writer.writerows(entries[k] for k in sorted(entries))
        temp.replace(prior_index)
    print('MOTION', len(entries), 'clips;', sum(bool(r['pose_3d']) for r in entries.values()), 'with 3D', flush=True)
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=.45)
    parser.add_argument('--max-gap', type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.strength <= 10 or not 0 <= args.max_gap <= 10:
        parser.error('strength must be 0..10 and max-gap 0..10')
    raise SystemExit(process_dataset(args.dataset, args.strength, args.max_gap))
