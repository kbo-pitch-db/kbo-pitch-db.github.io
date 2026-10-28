"""Build the small visitor-facing index from existing records and quality files.

No new motion or ball-flight measurements are calculated here. Keep unaccepted
candidate events and sensitive values out of the player headline statistics.
"""
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('thorax_pelvis_axial_fc', 'lead_knee_net_extension', 'pelvis_trunk_peak_gap_ms')


def metric_summary(rows, key):
    samples = []
    seen = set()
    for row in rows:
        value = row.get('metrics', {}).get(key, {})
        identity = row.get('pitchId') or row['id']
        if identity in seen or value.get('status') not in ('provisional_3d', 'sensitive_3d'):
            continue
        number = value.get('value')
        if not isinstance(number, (float, int)) or not math.isfinite(number):
            continue
        seen.add(identity)
        samples.append(dict(id=identity, clipId=row['primaryClipId'], pitch=row['pitch'],
                            value=number, status=value['status'], date=row['date']))
    stable = [s['value'] for s in samples if s['status'] == 'provisional_3d']
    # Three independent pitches is a display minimum, not an accuracy threshold.
    status = 'provisional' if len(stable) >= 3 else 'single' if len(samples) == 1 else 'insufficient' if stable else 'sensitive' if len(samples) >= 2 else 'missing'
    return dict(value=statistics.median(stable) if len(stable) >= 3 else None,
                n=len(stable), observedN=len(samples), status=status, samples=samples)


def candidate(clip, info, analysis):
    """Prefer supported complete deliveries, then source agreement; deterministic ties."""
    frames = len(analysis.get('frameTimes', [])) or clip['frames']
    events = analysis.get('events', {})
    pkh, fc, br = [events.get(k, {}).get('interval') for k in ('pkh', 'fc', 'br')]
    # Require preparation, landing, release and a visible follow-through margin.
    complete = bool(pkh and fc and br and
                    0 <= pkh[0] <= pkh[-1] < fc[0] <= fc[-1] < br[0] <= br[-1] < frames - 1 and
                    frames - br[-1] - 1 >= clip['fps'] * .2)
    qpath = ROOT / 'data' / clip['motionQuality']
    quality = json.loads(qpath.read_text())
    coverage = quality.get('valid_3d_frames', 0) / max(1, frames)
    joints = info.get('metrics', {})
    image_coverage = min((m['validFrames'] / max(1, m['totalFrames']) for m in joints.values()), default=0)
    accepted = info.get('status') == 'auto_accepted'
    padding = min(fc[0], frames - br[-1] - 1) / frames if complete else 0
    rank = (int(complete), coverage, image_coverage, int(quality.get('strict_source_pass', False)),
            -quality.get('after', {}).get('source_p50_px', 1e6), padding, int(accepted))
    return dict(clipId=clip['id'], pitch=clip['pitch'] if accepted else None,
                complete=complete, rank=rank)


def main():
    index_bytes = (ROOT / 'data/clips.json').read_bytes()
    index = json.loads(index_bytes)
    portal = json.loads((ROOT / 'data/portal/summary.json').read_text())
    database = json.loads((ROOT / 'data/metrics/database.json').read_text())
    rows_by_player = defaultdict(list)
    for row in database['rows']:
        rows_by_player[row['playerCode']].append(row)
    clips_by_player = defaultdict(list)
    for clip in index['clips']:
        clips_by_player[clip['playerCode']].append(clip)
    players = {}
    for profile in portal['players']:
        code = profile['code']
        detail = json.loads((ROOT / 'data' / profile['detailPath']).read_text())
        candidates = []
        for clip in clips_by_player[code]:
            path = ROOT / 'data/biomechanics/analysis' / (clip['id'] + '.json.gz')
            analysis = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
            candidates.append(candidate(clip, detail['clips'].get(clip['id'], {}), analysis))
        candidates.sort(key=lambda c: (c['rank'], c['clipId']), reverse=True)
        by_pitch = {}
        for c in candidates:
            if c['pitch'] and c['pitch'] not in by_pitch:
                by_pitch[c['pitch']] = c['clipId']
        players[code] = dict(representative=next((c['clipId'] for c in candidates if c['pitch']), candidates[0]['clipId'] if candidates else None),
                             byPitch=by_pitch,
                             metrics={k: metric_summary(rows_by_player[code], k) for k in METRICS})
    out = dict(schemaVersion=1, poseVersion=index['packageVersion'],
               clipIndexSha256=hashlib.sha256(index_bytes).hexdigest(),
               databaseVersion=database['version'], players=players)
    path = ROOT / 'data/portal/player_overview.json'
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'{len(players)} players; {path.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
