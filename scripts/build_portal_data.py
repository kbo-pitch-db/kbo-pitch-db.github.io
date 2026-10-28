#!/usr/bin/env python3
"""Build the research portal from frozen public clips and cached Naver records.

Does not change poses, the canonical clip index, or the link audit. All joins
require accepted audit evidence AND the same game/player/pitch identifier.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
METRICS = {
    'leftKnee': [11, 13, 15], 'rightKnee': [12, 14, 16],
    'leftElbow': [5, 7, 9], 'rightElbow': [6, 8, 10],
}


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def joint_angle(frame, joints):
    points = frame.get('keypoints', [])
    for i in joints:
        if i >= len(points) or points[i] is None or len(points[i]) < 3:
            return None
        if any(number(x) is None for x in points[i][:3]) or points[i][2] < .5:
            return None
        for field in ['interpolated', 'outlier']:
            flags = frame.get(field, [])
            if i < len(flags) and flags[i]:
                return None
    a, b, c = [points[i] for i in joints]
    u, v = [a[i]-b[i] for i in range(2)], [c[i]-b[i] for i in range(2)]
    denominator = math.hypot(*u)*math.hypot(*v)
    if denominator < 1e-8:
        return None
    return round(math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(u, v))/denominator)))), 2)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')


def build(dataset, output):
    index_path = SITE/'data/clips.json'
    index = json.loads(index_path.read_text())
    clips = index['clips']
    source_rows = rows(dataset/'sources.csv')
    sources = {r['source_id']: r for r in source_rows}
    pitches = rows(dataset/'naver_sports/pitches.csv')
    # Repeated source files must never change the denominators.
    pitch_map = {}
    for p in pitches:
        if p['naver_pitch_id'] in pitch_map and p != pitch_map[p['naver_pitch_id']]:
            raise ValueError('Conflicting pitch record: '+p['naver_pitch_id'])
        pitch_map[p['naver_pitch_id']] = p
    pitches = list(pitch_map.values())
    accepted = {r['dataset_clip_id']: r for r in rows(dataset/'naver_sports/auto_accepted.csv')}
    audit = {r['dataset_clip_id']: r for r in rows(dataset/'naver_sports/link_audit.csv')}
    games = {}
    for path in sorted((dataset/'naver_sports/raw/schedules').glob('*.json.gz')):
        for game in json.load(gzip.open(path)).get('result', {}).get('games', []):
            games[game['gameId']] = game
    # Handedness is from the historical game roster, never today's team page.
    hands, batter_hands = defaultdict(set), defaultdict(set)
    for path in sorted((dataset/'naver_sports/raw/games').glob('*/inning_*.json.gz')):
        relay = json.load(gzip.open(path)).get('result', {}).get('textRelayData', {})
        gid = relay.get('gameId')
        for side in ['home', 'away']:
            for section in ['Entry', 'Lineup']:
                for role in ['pitcher', 'batter']:
                    for person in relay.get(side+section, {}).get(role, []):
                        hit = person.get('hitType') or person.get('hittype') or ''
                        key = (gid, str(person.get('pcode', '')), person.get('name'))
                        if role == 'pitcher' and hit[:2] in ['좌투', '우투', '우언', '좌언']:
                            hands[key].add('L' if hit.startswith('좌') else 'R')
                        if role == 'batter' and hit[-2:] in ['좌타', '우타', '양타']:
                            batter_hands[key].add({'좌타': 'L', '우타': 'R', '양타': 'S'}[hit[-2:]])
    by_player, by_clip, link_map = defaultdict(list), defaultdict(list), {}
    links_by_pitch = defaultdict(list)
    statuses = Counter()
    for c in clips:
        by_clip[c['playerCode']].append(c)
        a, review = accepted.get(c['id'], {}), audit.get(c['id'], {})
        p = pitch_map.get(a.get('naver_pitch_id'), {})
        same = bool(p and review.get('auto_decision') == 'accepted'
            and review.get('naver_pitch_id') == p['naver_pitch_id']
            and p['pitcher_code'] == c['playerCode'] and p['pitcher_name'] == c['player']
            and p['naver_game_id'] == sources.get(c.get('sourceId'), {}).get('naver_game_id')
            and c.get('playerEvidence', {}).get('pitchId') == p['naver_pitch_id'])
        status = 'auto_accepted' if same else ('held' if review.get('auto_decision') == 'hold' else 'unresolved')
        statuses[status] += 1
        link_map[c['id']] = {'status': status, 'pitchId': p['naver_pitch_id'] if same else None,
                            'auditDecision': review.get('auto_decision', 'not_proposed')}
        if same:
            links_by_pitch[p['naver_pitch_id']].append(c['id'])
    for p in pitches:
        if p['pitcher_code'] in by_clip:
            by_player[p['pitcher_code']].append(p)
    profiles = []
    for code, player_clips in sorted(by_clip.items()):
        records, teams, player_hands = [], set(), set()
        for p in sorted(by_player[code], key=lambda r: (r['naver_game_id'], int(r['relay_seqno']))):
            gid = p['naver_game_id']
            team = games.get(gid, {}).get('homeTeamName' if p['half_inning'] == 'top' else 'awayTeamName')
            if team:
                teams.add(team)
            player_hands.update(hands.get((gid, code, p['pitcher_name']), set()))
            bh = batter_hands.get((gid, p['batter_code'], p['batter_name']), set())
            valid_pts = p['tracking_status'] == 'derived_validated_x'
            records.append({
                'id': p['naver_pitch_id'], 'gameId': gid, 'date': p['game_date'],
                'inning': int(p['inning']), 'half': p['half_inning'], 'atBat': p['at_bat_id'],
                'sequence': int(p['pitcher_pitch_number']), 'atBatPitch': int(p['pitch_number']),
                'balls': int(p['balls_before']), 'strikes': int(p['strikes_before']),
                'outs': int(p['outs_before']), 'pitch': p['pitch_type_raw'] or None,
                'speed': number(p['speed_kmh']), 'result': p['pitch_result'],
                'batter': p['batter_name'], 'batterHand': next(iter(bh)) if len(bh) == 1 else None,
                'plateX': number(p['cross_plate_x_ft']) if valid_pts else None,
                'plateZ': number(p['plate_z_ft_derived']) if valid_pts else None,
                'zoneTop': number(p['top_sz_ft']) if valid_pts else None,
                'zoneBottom': number(p['bottom_sz_ft']) if valid_pts else None,
                'trackingStatus': p['tracking_status'], 'sourceUrl': p['source_url'],
                'clipIds': sorted(links_by_pitch.get(p['naver_pitch_id'], [])),
            })
        dates = sorted({c['date'] for c in player_clips})
        clip_games = {sources.get(c.get('sourceId'), {}).get('naver_game_id') for c in player_clips}
        metric_counts = Counter()
        clip_info = {}
        for c in player_clips:
            pose = json.loads((SITE/'data'/(c.get('poseRefined') or c['pose'])).read_text())
            quality = {}
            for name, joints in METRICS.items():
                valid = [v for frame in pose['frames'] if (v := joint_angle(frame, joints)) is not None]
                quality[name] = {'validFrames': len(valid), 'totalFrames': len(pose['frames'])}
                if len(valid) >= 10:
                    metric_counts[name] += 1
            clip_info[c['id']] = {**link_map[c['id']], 'metrics': quality}
        profile = {
            'code': code, 'name': player_clips[0]['player'], 'season': '2024',
            'teams': sorted(teams), 'hand': next(iter(player_hands)) if len(player_hands) == 1 else None,
            'clipCount': len(player_clips), 'linkedClips': sum(link_map[c['id']]['status'] == 'auto_accepted' for c in player_clips),
            'clipGames': len(clip_games - {None}), 'dateStart': dates[0], 'dateEnd': dates[-1],
            'recordCount': len(records), 'recordGames': len({p['gameId'] for p in records}),
            'recordStart': min((p['date'] for p in records), default=None),
            'recordEnd': max((p['date'] for p in records), default=None),
            'pitchCounts': dict(Counter(c['pitch'] or 'unknown' for c in player_clips)),
            'metricEligibleClips': {k: metric_counts[k] for k in METRICS},
            'detailPath': f'portal/players/{code}.json',
        }
        profiles.append(profile)
        write(output/'players'/f'{code}.json', {'profile': profile, 'records': records, 'clips': clip_info})
    clip_dates = sorted(c['date'] for c in clips)
    summary = {
        'schemaVersion': 1, 'version': 'portal-20261028-v1', 'updated': '2026-10-28',
        'poseVersion': index['packageVersion'], 'season': '2024',
        'clipCount': len(clips), 'playerCount': len(profiles),
        'gameCount': len({sources[c['sourceId']]['naver_game_id'] for c in clips}),
        'frameCount': sum(c['frames'] for c in clips),
        'dateStart': clip_dates[0], 'dateEnd': clip_dates[-1],
        'recordCount': len(pitches), 'recordGames': len({p['naver_game_id'] for p in pitches}),
        'profileRecordCount': sum(p['recordCount'] for p in profiles),
        'recordsWithPts': sum(p['tracking_status'] == 'derived_validated_x' for p in pitches),
        'links': dict(statuses), 'uniqueLinkedPitches': len(links_by_pitch),
        'duplicateReplayClips': sum(max(0, len(ids)-1) for ids in links_by_pitch.values()),
        'humanReviewedLinks': 0, 'players': profiles,
        'provenance': {
            str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [dataset/'naver_sports/pitches.csv', dataset/'naver_sports/auto_accepted.csv',
                      dataset/'naver_sports/link_audit.csv', dataset/'sources.csv']
        },
        'clipIndexSha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'interpretation': 'Highlight clips and records in 32 acquired games are distinct samples, neither is a complete season. Automatic pitch links are not human validation. PTS plate x is provider-reported; z is derived at the provider crossing plane from its quadratic trajectory. Pose angles are image projections, not calibrated joint rotations.',
    }
    # Deterministic archive includes every acquired record, including the 806
    # pitches by players outside the motion sample. Never package broadcast media.
    (output/'pitch_records.csv.gz').write_bytes(gzip.compress(
        (dataset/'naver_sports/pitches.csv').read_bytes(), mtime=0))
    write(output/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['players', 'provenance', 'interpretation']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=SITE.parent/'dataset')
    parser.add_argument('--output', type=Path, default=SITE/'data/portal')
    args = parser.parse_args()
    build(args.dataset.resolve(), args.output.resolve())
