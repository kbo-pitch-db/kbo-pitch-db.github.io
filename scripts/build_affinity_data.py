"""Build observed-pitch outcome profiles, independently of the motion cohort.

This is a KBO adaptation of the Affinity interaction, not Statcast's batted-ball
model. Never interpret the provider's `Hit` (in play) as a base hit, or infer PA
outcomes from this CSV's incomplete at-bat/order metadata.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from import_published_pts import SOURCE_SHA256, SOURCE_URL

OUTCOMES = ['Ball', 'Strike', 'Swing', 'Foul', 'Hit']
BUNTS = {'BuntFoul', 'BuntSwing'}
MIN_PITCHES = 500


def aggregate(rows, minimum=MIN_PITCHES):
    groups = {'hitters': {}, 'pitchers': {}}
    seen, dates, games = set(), set(), set()
    excluded = Counter()
    totals = Counter()
    for row in rows:
        identity = (row['game_id'], row['pitchId'])
        if identity in seen:
            raise ValueError('Duplicate pitch identity: ' + repr(identity))
        seen.add(identity)
        if row['result'] in BUNTS:
            excluded[row['result']] += 1
            continue
        if row['result'] not in OUTCOMES:
            raise ValueError('Unknown result: ' + row['result'])
        if row['stance'] not in ['L', 'R']:
            raise ValueError('Unknown batting side')
        date = row['game_id'][:8]
        dates.add(date)
        games.add(row['game_id'])
        totals[row['result']] += 1
        for role, code, name, side in [
            ('hitters', row['batter'], row['batter_name'], row['stance']),
            ('pitchers', row['pitcher'], row['picher_name'], ''),
        ]:
            key = code + ('-' + side if side else '')
            entry = groups[role].setdefault(key, dict(
                key=key, code=code, name=name, side=side,
                counts=Counter(), games=set(), dates=set()))
            if entry['name'] != name:
                raise ValueError('Conflicting player names: ' + code)
            entry['counts'][row['result']] += 1
            entry['games'].add(row['game_id'])
            entry['dates'].add(date)
    result = {}
    for role, entries in groups.items():
        players = []
        for entry in entries.values():
            counts = [entry['counts'][key] for key in OUTCOMES]
            count = sum(counts)
            if count < minimum:
                continue
            players.append(dict(
                key=entry['key'], code=entry['code'], name=entry['name'],
                side=entry['side'], counts=counts, pitches=count,
                games=len(entry['games']), dateStart=min(entry['dates']),
                dateEnd=max(entry['dates'])))
        result[role] = sorted(players, key=lambda p: p['key'])
    if not dates:
        raise ValueError('No usable pitches')
    return dict(
        schemaVersion=1, version='kbo-outcomes-20261030-v1', season=2024,
        dateStart=min(dates), dateEnd=max(dates), games=len(games),
        sourcePitches=len(seen), usablePitches=sum(totals.values()),
        excludedBunts=dict(excluded), outcomeOrder=OUTCOMES,
        outcomeCounts=[totals[key] for key in OUTCOMES],
        minimumPitches=minimum,
        similarity=dict(id='jensen-shannon-distance-base2-v1',
                        formula='100 * (1 - sqrt(JS_base2(p, q)))',
                        smoothing='none',
                        scope='Same role; hitter batting sides kept separate'),
        source=dict(url=SOURCE_URL, sha256=SOURCE_SHA256,
                    coverage='Available tracked 2024 pitches; not all KBO pitches',
                    comparison='Pitch outcomes, not Statcast batted-ball types',
                    inPlay='Hit means in play, not a base hit'),
        groups=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/visuals/affinity.json')
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source SHA256 mismatch')
    with args.source.open(encoding='cp949', newline='') as stream:
        result = aggregate(csv.DictReader(stream))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({key: result[key] for key in ['sourcePitches', 'usablePitches', 'games']}))
    print({role: len(players) for role, players in result['groups'].items()})


if __name__ == '__main__':
    main()
