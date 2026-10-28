"""Reproducible, conservative lead-knee image-pattern pilot. Never edits poses.

Only audited event intervals enter the report. Angles are 2D projections, not
anatomical flexion. Repeated source frames are not counted as observations.
Run with --audit-video to refresh optional cached-image cadence diagnostics
(requires OpenCV + NumPy and the local, unpublished source crops).
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
AUDIT = DATA / 'mechanics/event_audit.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def flexion(frame, hand, threshold=.4):
    ids = [11, 13, 15] if hand == 'R' else [12, 14, 16]
    points = [frame['keypoints'][i] for i in ids]
    if any(not p or len(p) < 3 or p[2] < threshold or not all(math.isfinite(v) for v in p[:3]) for p in points):
        return None
    if any(any(frame[k][i] for i in ids) if isinstance(frame.get(k), list) else frame.get(k, False) for k in ['interpolated', 'outlier']):
        return None
    a, b, c = points
    u, v = [[p[i] - b[i] for i in range(2)] for p in [a, c]]
    denominator = math.hypot(*u) * math.hypot(*v)
    if denominator < 1e-9:
        return None
    return 180 - math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(u, v))/denominator))))


def smooth(values, radius):
    # Only called on a fully observed interval; never bridges missing frames.
    return [sum(values[max(0, i-radius):i+radius+1])/len(values[max(0, i-radius):i+radius+1]) for i in range(len(values))]


def at(values, frame):
    low = int(frame)
    high = min(len(values)-1, low+1)
    return values[low] + (values[high]-values[low])*(frame-low)


def summarize(values, landing, release):
    waveform = [at(values, landing+(release-landing)*i/40) for i in range(41)]
    observed = [(landing, at(values, landing))] + [(i, values[i]) for i in range(math.ceil(landing), math.floor(release)+1)] + [(release, at(values, release))]
    maximum = max(v for _, v in observed)
    peak_frames = [f for f, v in observed if abs(v-maximum) < 1e-8]
    return dict(waveform=waveform, landingFlexion=waveform[0], releaseFlexion=waveform[-1],
                extraFlexion=max(0, maximum-waveform[0]), extension=maximum-waveform[-1],
                netExtension=waveform[0]-waveform[-1], maximumFlexionFrame=peak_frames[0],
                maximumFlexionProgress=100*(peak_frames[0]-landing)/(release-landing))


def video_audit(clips, audit):
    import cv2
    import numpy as np
    result = {}
    for event in audit['samples']:
        if event['decision'] != 'include':
            continue
        clip = clips[event['clipId']]
        source, local = clip['id'].rsplit('_', 1)
        path = ROOT.parent / 'dataset/videos/crop' / source / (local+'.mp4')
        capture = cv2.VideoCapture(str(path))
        previous, differences, frames = None, [], []
        begin, end = event['landing'][0], event['release'][1]
        for index in range(end+1):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'Missing source frame: {path}:{index}')
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 160)).astype(float)
            if begin <= index <= end and previous is not None:
                difference = float(np.mean(np.abs(image-previous)))
                differences.append(difference)
                if difference >= 1.0 or index == begin:
                    frames.append(index)
            previous = image
        capture.release()
        result[clip['id']] = dict(sourceSha256=digest(path), range=[begin, end], independentFrameCandidates=frames,
            adjacentComparisons=len(differences), nearDuplicatePairs=sum(d < 1 for d in differences),
            method='160x160 grayscale mean absolute adjacent difference <1/255; image-cadence diagnostic, not a physical-time calibration',
            playbackRateVerified=False)
    (DATA/'mechanics/cadence_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


def build():
    clips = {c['id']: c for c in read(DATA/'clips.json')['clips']}
    audit = read(AUDIT)
    detail = read(DATA/f"portal/players/{audit['playerCode']}.json")
    cadence = read(DATA/'mechanics/cadence_audit.json')
    profile = detail['profile']
    samples, excluded = [], []
    events = {s['clipId']: s for s in audit['samples']}
    for cid, link in detail['clips'].items():
        event, clip = events.get(cid), clips[cid]
        reason = ('event_not_audited' if not event else event.get('reason'))
        if not reason and link['status'] != 'auto_accepted':
            reason = 'pitch_link_unresolved'
        if reason:
            excluded.append(dict(clipId=cid, reason=reason))
            continue
        if profile['hand'] not in ['R', 'L']:
            excluded.append(dict(clipId=cid, reason='throwing_side_unknown'))
            continue
        pose_path = DATA/clip['pose']
        pose = read(pose_path)
        start, end = event['landing'][0], event['release'][1]
        values = [flexion(f, profile['hand']) for f in pose['frames']]
        if any(v is None for v in values[start:end+1]):
            excluded.append(dict(clipId=cid, reason='knee_observation_missing'))
            continue
        # Restrict filtering to the audited, fully supported interval.
        raw = values[start:end+1]
        landing = sum(event['landing'])/2-start
        release = sum(event['release'])/2-start
        base = summarize(smooth(raw, 1), landing, release)
        variants = []
        for radius in [0, 1, 2]:
            filtered = smooth(raw, radius)
            for land in [event['landing'][0], sum(event['landing'])/2, event['landing'][1]]:
                for rel in [event['release'][0], sum(event['release'])/2, event['release'][1]]:
                    variants.append(summarize(filtered, land-start, rel-start))
        sensitivity = {key: [min(v[key] for v in variants), max(v[key] for v in variants)]
                       for key in ['extension', 'extraFlexion', 'landingFlexion', 'releaseFlexion', 'maximumFlexionProgress']}
        # Current 3D is retained only as an inspection diagnostic, never pooled
        # with 2D angles or used as a validated player mechanics measurement.
        model_path = DATA/clip['pose3d']
        with gzip.open(model_path, 'rt') as stream:
            model = json.load(stream)
        model_values = []
        ids = [1, 4, 7] if profile['hand'] == 'R' else [2, 5, 8]
        for frame in [round(landing+start), round(release+start)]:
            f = model['frames'][frame]
            if f.get('available') is False:
                model_values.append(None)
                continue
            a, b, c = [f['anatomy'][i] for i in ids]
            u, v = [[p[i]-b[i] for i in range(3)] for p in [a, c]]
            den = math.sqrt(sum(x*x for x in u)*sum(x*x for x in v))
            model_values.append(180-math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(u, v))/den)))))
        record = next(r for r in detail['records'] if r['id'] == link['pitchId'])
        sample = dict(clipId=cid, pitchId=link['pitchId'], pitch=record['pitch'], speed=record['speed'],
            strictObservationEligible=all(flexion(f, profile['hand'], .5) is not None for f in pose['frames'][start:end+1]),
            minimumJointScore=min(f['keypoints'][i][2] for f in pose['frames'][start:end+1] for i in ([11,13,15] if profile['hand']=='R' else [12,14,16])),
            date=record['date'], gameId=record['gameId'], cameraGroup=audit['cameraGroup'],
            throwingHand=profile['hand'], leadSide='left' if profile['hand']=='R' else 'right',
            landingInterval=event['landing'], releaseInterval=event['release'],
            landingFrame=landing+start, releaseFrame=release+start,
            sourceLandingInterval=[pose['frames'][i]['source_frame'] for i in event['landing']],
            sourceReleaseInterval=[pose['frames'][i]['source_frame'] for i in event['release']],
            playbackIntervalSeconds=[(event['release'][0]-event['landing'][1])/pose['fps'], (event['release'][1]-event['landing'][0])/pose['fps']],
            fps=pose['fps'], rawFrames=list(range(start, end+1)), rawFlexion=raw,
            settings={str(2*r+1): summarize(smooth(raw, r), landing, release) for r in [0,1,2]},
            observationFrameCandidates=cadence[cid]['independentFrameCandidates'], cadence=cadence[cid],
            waveform=base.pop('waveform'), metrics=base, sensitivity=sensitivity,
            modelInspection=dict(landingFlexion=model_values[0], releaseFlexion=model_values[1], eligibleForAggregation=False),
            sourceUrl=clip['sourceUrl'], recordSourceUrl=record['sourceUrl'],
            poseSha256=digest(pose_path), modelSha256=digest(model_path), pitchLinkStatus=link['status'])
        sample['metrics']['maximumFlexionFrame'] += start
        samples.append(sample)
    report = dict(schemaVersion=1, version=audit['version'], playerCode=profile['code'],
        modelVersion=clips[samples[0]['clipId']]['reconstructionVersion'],
        coordinate='Halpe-26 original crop pixels; projected hip-knee-ankle flexion = 180 - included angle; 0 = projected straight leg',
        aggregationUnit='One unique pitch, within one player × one game × center-field live view × pitch type',
        eventAuditSha256=digest(AUDIT), eventReview=audit['review'], humanReviewed=False,
        physicalTimeValidated=False, modelSensitivityValidated=False, scopeClipCount=len(detail['clips']),
        samples=samples, excluded=excluded,
        definitions=dict(extension='Maximum projected flexion between landing and release minus release projected flexion; positive = straightening from maximum bend',
            observationScreen='Exploratory screen: original hip/knee/ankle joint scores >=0.4 throughout both event intervals and the intervening frames; no interpolated/outlier joints. Strict screen >=0.5 is available separately. Scores are model scores, not accuracy probabilities. No threshold tuning per clip.',
            netExtension='Landing projected flexion minus release projected flexion; positive = net straightening',
            extraFlexion='Maximum projected flexion minus landing projected flexion',
            maximumFlexionProgress='First maximum projected flexion, normalized landing=0%, release=100%; not peak extension velocity',
            distribution='Across unique pitches; IQR is not model confidence or event uncertainty',
            sensitivity='27 combinations: 3 landing endpoints × 3 release endpoints × 1/3/5-frame centered box means. Min/max is an operational sensitivity envelope, not a confidence interval. No gap filling.',
            physicalTime='Frame timestamps describe broadcast playback only. No absolute angular velocity, physical elapsed milliseconds, or peak-speed estimates.'),
        references=['https://pmc.ncbi.nlm.nih.gov/articles/PMC11329978/', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11544755/', 'https://www.opencap.ai/best-practices?variant=monocular'])
    def rounded(value):
        if isinstance(value, float): return round(value, 4)
        if isinstance(value, list): return [rounded(v) for v in value]
        if isinstance(value, dict): return {k: rounded(v) for k, v in value.items()}
        return value
    output = DATA/'mechanics'/f"{profile['code']}.json"
    output.write_text(json.dumps(rounded(report), ensure_ascii=False, separators=(',', ':'))+'\n')
    print(f'{output.relative_to(ROOT)}: {len(samples)} included, {len(excluded)} excluded')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-video', action='store_true')
    args = parser.parse_args()
    if args.audit_video:
        video_audit({c['id']: c for c in read(DATA/'clips.json')['clips']}, read(AUDIT))
    build()
