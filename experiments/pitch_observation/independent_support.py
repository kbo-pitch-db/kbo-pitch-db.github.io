"""Image-supported toe episodes independent of the lead-plant event detector.

Low image speed is evidence for a contact hypothesis, not proof of 3D contact.
Heels are not locked, airborne pauses are rejected by the observed support band,
and missing observations/time discontinuities always release a hypothesis.
"""
from common import *
from events import smooth_xy

VERSION = 'independent-toe-episodes-v1'


def contact_state(h, phase, background):
    xy, confidence = smooth_xy(h, background)
    xy, confidence = xy[:, 20:22], confidence[:, 20:22]
    times = h['source_time']
    n = len(times)
    if n < 4 or phase.get('phase_side_conflict'):
        return np.zeros((n, 0)), np.array([], int), []
    reliable = confidence > .4
    if not reliable.any():
        return np.zeros((n, 0)), np.array([], int), []
    # An image band only: no measured ground plane or guessed mound dimensions.
    ground = float(np.quantile(xy[..., 1][reliable], .8))
    speed = np.linalg.norm(np.gradient(xy, times, axis=0), axis=-1)
    continuous = np.r_[True, (np.diff(times) > 0) & (np.diff(times) < .1)
                        & (np.diff(h['frame_index']) == 1)]
    minimum = max(4, int(np.ceil(float(h['fps']) * .12)))
    lead = {'left': 0, 'right': 1}.get(phase.get('lead_side'))
    lift = phase.get('events', {}).get('lift_interval')
    weights, seeds, notes = [], [], []
    for marker in range(2):
        stable = (speed[:, marker] < .35) & reliable[:, marker]
        stable &= xy[:, marker, 1] > ground - .10
        if lead == marker and lift:
            stable[:lift[1] + 1] = False
        starts = np.flatnonzero(stable & (~np.r_[False, stable[:-1]] | ~continuous))
        ends = np.flatnonzero(stable & (~np.r_[stable[1:], False]
                                      | ~np.r_[continuous[1:], False]))
        for start, end in zip(starts, ends):
            if end - start + 1 < minimum:
                continue
            weight = np.zeros(n)
            weight[start:end + 1] = (.6 * confidence[start:end + 1, marker] ** 2
                / (1 + (speed[start:end + 1, marker] / .35) ** 2))
            seed = int((start + end) // 2)
            weights.append(weight)
            seeds.append(seed)
            notes.append(dict(marker=marker, start=int(start), end=int(end),
                seed_frame=seed, detector=VERSION, independent_of_plant=True,
                semantics='Image-supported toe contact hypothesis; not manual contact ground truth'))
    return (np.stack(weights, 1) if weights else np.zeros((n, 0)),
            np.array(seeds, int), notes)


def episode_metrics(z, notes):
    feet = z['foot_keypoints'].astype(float) + z['translation'][:, None]
    leg = float(z['model_leg'])
    result = []
    for note in notes:
        p = feet[note['start']:note['end'] + 1, note['marker']]
        result.append({**note, 'extent_from_start_leg': float(
            np.linalg.norm(p - p[0], axis=-1).max() / leg),
            'diameter_leg': float(np.linalg.norm(p[:, None] - p[None], axis=-1).max() / leg)})
    return result
