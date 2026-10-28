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
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
METRICS = {
    'leftKnee': [11, 13, 15], 'rightKnee': [12, 14, 16],
    'leftElbow': [5, 7, 9], 'rightElbow': [6, 8, 10],
}
SWING_RESULTS = {'S', 'F', 'H'}  # Naver text: swing-and-miss, foul, ball put in play.
ORDINARY_RESULTS = SWING_RESULTS | {'B', 'T'}  # Bunt codes V/W are excluded.
BENCHMARKS = {
    'fastballVelocity': ('fastballs', 15), 'whiffRate': ('swings', 20),
    'cswRate': ('ordinaryPitches', 50), 'zoneRate': ('locations', 50),
    'breakingShare': ('pitches', 50),
}


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*fraction
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)]-values[low])*(position-low)


def pitch_statistics(records):
    """Pitch-based statistics; never require a corresponding video clip."""
    swings = [r for r in records if r['result'] in SWING_RESULTS]
    ordinary = [r for r in records if r['result'] in ORDINARY_RESULTS]
    locations = [r for r in records if all(number(r.get(k)) is not None
        for k in ['plateX', 'plateZ', 'zoneBottom', 'zoneTop']) and r['zoneTop'] > r['zoneBottom']]
    fastballs = [r['speed'] for r in records if r['pitch'] == '직구' and r['speed'] is not None]
    ratio = lambda count, total: round(100*count/total, 3) if total else None
    pitch_counts = Counter(r['pitch'] or 'unknown' for r in records)
    pitch_stats = {}
    for pitch, count in pitch_counts.items():
        group = [r for r in records if (r['pitch'] or 'unknown') == pitch]
        velocities = [r['speed'] for r in group if r['speed'] is not None]
        attempts = [r for r in group if r['result'] in SWING_RESULTS]
        pitch_stats[pitch] = {
            'count': count, 'share': ratio(count, len(records)), 'speedN': len(velocities),
            'mean': round(statistics.mean(velocities), 2) if velocities else None,
            **{key: round(quantile(velocities, q), 2) if velocities else None
               for key, q in [('min', 0), ('p10', .1), ('p25', .25), ('median', .5), ('p75', .75), ('p90', .9), ('max', 1)]},
            'swings': len(attempts), 'whiffs': sum(r['result'] == 'S' for r in attempts),
            'whiffRate': ratio(sum(r['result'] == 'S' for r in attempts), len(attempts)),
        }
    fastball_family = sum(pitch_counts[p] for p in ['직구', '투심', '커터'])
    breaking = sum(pitch_counts[p] for p in ['슬라이더', '커브'])
    offspeed = sum(pitch_counts[p] for p in ['체인지업', '포크'])
    main = pitch_counts.most_common(1)[0][0] if pitch_counts else None
    style = ('small_sample' if len(records) < 30 else 'fastball' if fastball_family/len(records) >= .55
             else 'breaking' if breaking/len(records) >= .45 else 'offspeed' if offspeed/len(records) >= .30 else 'mixed')
    return {
        'pitches': len(records), 'swings': len(swings), 'ordinaryPitches': len(ordinary),
        'locations': len(locations), 'fastballs': len(fastballs),
        'whiffs': sum(r['result'] == 'S' for r in swings),
        'calledOrSwingStrikes': sum(r['result'] in {'T', 'S'} for r in ordinary),
        'inZone': sum(abs(r['plateX']) <= 17/24 and r['zoneBottom'] <= r['plateZ'] <= r['zoneTop'] for r in locations),
        'fastballVelocity': round(statistics.mean(fastballs), 3) if fastballs else None,
        'whiffRate': ratio(sum(r['result'] == 'S' for r in swings), len(swings)),
        'cswRate': ratio(sum(r['result'] in {'T', 'S'} for r in ordinary), len(ordinary)),
        'zoneRate': ratio(sum(abs(r['plateX']) <= 17/24 and r['zoneBottom'] <= r['plateZ'] <= r['zoneTop'] for r in locations), len(locations)),
        'breakingShare': ratio(breaking, len(records)), 'fastballShare': ratio(fastball_family, len(records)),
        'offspeedShare': ratio(offspeed, len(records)), 'primaryPitch': main, 'archetype': style,
        'pitchCounts': dict(pitch_counts), 'byPitch': pitch_stats,
    }


def add_benchmarks(profiles):
    cohorts = {}
    for metric, (denominator, minimum) in BENCHMARKS.items():
        eligible = [p for p in profiles if p['recordCount'] >= 50
                    and p['statistics'][denominator] >= minimum and p['statistics'][metric] is not None]
        values = [p['statistics'][metric] for p in eligible]
        cohorts[metric] = {'players': len(values), 'minimumRecords': 50,
                          'denominator': denominator, 'minimumObservations': minimum,
                          'median': quantile(values, .5)}
        for p in profiles:
            value = p['statistics'][metric]
            qualified = p in eligible and len(values) >= 5
            percentile = (100*(sum(v < value for v in values)+.5*sum(v == value for v in values))/len(values)) if qualified else None
            p.setdefault('benchmarks', {})[metric] = {
                'value': value, 'percentile': round(percentile, 1) if percentile is not None else None,
                'observations': p['statistics'][denominator], 'cohortPlayers': len(values), 'qualified': qualified,
            }
    return cohorts


def representative_clip(player_clips, vectors, main_pitch):
    candidates = [c for c in player_clips if c['pitch'] == main_pitch]
    if len(candidates) < 2:
        labeled = Counter(c['pitch'] for c in player_clips if c['pitch'])
        selected_pitch = labeled.most_common(1)[0][0] if labeled else None
        candidates = [c for c in player_clips if c['pitch'] == selected_pitch] if selected_pitch else player_clips
    eligible = [c for c in candidates if sum(v is not None for v in vectors[c['id']]) >= 42]
    if not eligible:
        return {'clipId': player_clips[0]['id'], 'pitch': player_clips[0]['pitch'] or None,
                'candidateCount': len(candidates), 'selection': 'example_only'} if player_clips else None
    centers = [statistics.median(values) if (values := [vectors[c['id']][i] for c in eligible
               if vectors[c['id']][i] is not None]) else None for i in range(84)]
    def distance(clip):
        pairs = [(v, center) for v, center in zip(vectors[clip['id']], centers) if v is not None and center is not None]
        return statistics.mean(abs(v-center)/180 for v, center in pairs) + .05*(1-len(pairs)/84), clip['id']
    chosen = min(eligible, key=distance)
    return {'clipId': chosen['id'], 'pitch': chosen['pitch'] or None, 'candidateCount': len(eligible),
            'selection': 'median_projected_angle_curve', 'alignment': 'normalized_clip_progress_not_events'}


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
        by_player[p['pitcher_code']].append(p)
    profiles, details = [], {}
    for code in sorted(set(by_clip) | set(by_player)):
        player_clips = by_clip[code]
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
        metric_counts, vectors = Counter(), {}
        clip_info = {}
        for c in player_clips:
            pose = json.loads((SITE/'data'/(c.get('poseRefined') or c['pose'])).read_text())
            quality = {}
            vectors[c['id']] = []
            for name, joints in METRICS.items():
                values = [joint_angle(frame, joints) for frame in pose['frames']]
                valid = [v for v in values if v is not None]
                vectors[c['id']].extend(values[round(i*(len(values)-1)/20)] for i in range(21))
                quality[name] = {'validFrames': len(valid), 'totalFrames': len(pose['frames'])}
                if len(valid) >= 10:
                    metric_counts[name] += 1
            clip_info[c['id']] = {**link_map[c['id']], 'metrics': quality}
        profile = {
            'code': code, 'name': player_clips[0]['player'] if player_clips else by_player[code][0]['pitcher_name'], 'season': '2024',
            'teams': sorted(teams), 'hand': next(iter(player_hands)) if len(player_hands) == 1 else None,
            'clipCount': len(player_clips), 'linkedClips': sum(link_map[c['id']]['status'] == 'auto_accepted' for c in player_clips),
            'clipGames': len(clip_games - {None}), 'dateStart': dates[0] if dates else None, 'dateEnd': dates[-1] if dates else None,
            'recordCount': len(records), 'recordGames': len({p['gameId'] for p in records}),
            'recordStart': min((p['date'] for p in records), default=None),
            'recordEnd': max((p['date'] for p in records), default=None),
            'pitchCounts': dict(Counter(c['pitch'] or 'unknown' for c in player_clips)),
            'metricEligibleClips': {k: metric_counts[k] for k in METRICS},
            'detailPath': f'portal/players/{code}.json',
            'statistics': pitch_statistics(records),
        }
        profile['representative'] = representative_clip(player_clips, vectors, profile['statistics']['primaryPitch'])
        profiles.append(profile)
        details[code] = {'schemaVersion': 2, 'version': 'portal-20261028-v2', 'profile': profile, 'records': records, 'clips': clip_info}
    cohorts = add_benchmarks(profiles)
    for code, detail in details.items():
        write(output/'players'/f'{code}.json', detail)
    clip_dates = sorted(c['date'] for c in clips)
    summary = {
        'schemaVersion': 2, 'version': 'portal-20261028-v2', 'updated': '2026-10-28',
        'poseVersion': index['packageVersion'], 'season': '2024',
        'clipCount': len(clips), 'playerCount': len(profiles),
        'motionPlayerCount': len(by_clip) - sum(not cs for cs in by_clip.values()),
        'gameCount': len({sources[c['sourceId']]['naver_game_id'] for c in clips}),
        'frameCount': sum(c['frames'] for c in clips),
        'dateStart': clip_dates[0], 'dateEnd': clip_dates[-1],
        'recordCount': len(pitches), 'recordGames': len({p['naver_game_id'] for p in pitches}),
        'profileRecordCount': sum(p['recordCount'] for p in profiles),
        'recordsWithPts': sum(p['tracking_status'] == 'derived_validated_x' for p in pitches),
        'links': dict(statuses), 'uniqueLinkedPitches': len(links_by_pitch),
        'duplicateReplayClips': sum(max(0, len(ids)-1) for ids in links_by_pitch.values()),
        'humanReviewedLinks': 0, 'players': profiles,
        'benchmarks': cohorts,
        'benchmarkDefinition': 'Midrank percentile among acquired-game pitchers with at least 50 records and metric-specific observation thresholds; not league/season rankings. More is not always better. No composite grade.',
        'resultCodes': {'B': 'ball', 'T': 'called_strike', 'S': 'swinging_strike', 'F': 'foul', 'H': 'in_play', 'V': 'bunt_miss', 'W': 'bunt_foul'},
        'provenance': {
            str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [dataset/'naver_sports/pitches.csv', dataset/'naver_sports/auto_accepted.csv',
                      dataset/'naver_sports/link_audit.csv', dataset/'sources.csv']
        },
        'clipIndexSha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'interpretation': 'Highlight clips and records in 32 acquired games are distinct samples, neither is a complete season. Automatic pitch links are not human validation. PTS plate x is provider-reported; z is derived at the provider crossing plane from its quadratic trajectory. Pose angles are image projections, not calibrated joint rotations.',
    }
    # Deterministic archive includes every acquired record, including the 806
    # pitches by players without a collected motion clip. Never package broadcast media.
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
