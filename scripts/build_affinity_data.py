"""Build pitcher repertoire and joint speed/HB/IVB distributions from PTS.

No plate-appearance or batter profiles are constructed. Motion-linked records
remain a separate cohort. Source pitches with incomplete fits are not imputed.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from build_pitch_flight import calculate, time_at_y, FIELDS, VERSION as FLIGHT_VERSION
from import_published_pts import SOURCE_SHA256, SOURCE_URL

MIN_PITCHES = 500
MIN_COVERAGE = .9
QUANTILES = 65
SCALES = [10., 10., 10.]  # km/h, cm, cm; fixed before comparing players.
TYPE_INFO = [
    ('4-Seam', '직구', 'Four-seam', '#df704b'),
    ('2-Seam', '투심', 'Two-seam', '#579583'),
    ('Cutter', '커터', 'Cutter', '#c879a5'),
    ('Slider', '슬라이더', 'Slider', '#5889b2'),
    ('Curve', '커브', 'Curveball', '#8972aa'),
    ('Change-up', '체인지업', 'Changeup', '#c09c36'),
    ('Fork', '포크', 'Forkball', '#6e85a1'),
    ('Sinker', '싱커', 'Sinker', '#53745d'),
    ('Knuckle', '너클볼', 'Knuckleball', '#72786f'),
    ('슬러브', '슬러브', 'Slurve', '#8a6d9b'),
]
TYPE_IDS = [p[0] for p in TYPE_INFO]
AXES = [dict(id='speed', unit='km/h', step=1.),
        dict(id='hb', unit='cm', step=2.), dict(id='ivb', unit='cm', step=2.)]


def directions():
    # 13 unoriented directions: 3 axes, 6 plane diagonals, 4 space diagonals.
    vectors = []
    for v in itertools.product([-1, 0, 1], repeat=3):
        if not any(v) or next(x for x in v if x) < 0:
            continue
        norm = math.sqrt(sum(x*x for x in v))
        vectors.append([x/norm for x in v])
    return sorted(vectors, key=lambda v: (sum(x != 0 for x in v), v))


def observation(row, with_vaa=False):
    """Same crossing-x gate and movement implementation as existing PTS release."""
    try:
        speed = float(row['speed'])
        raw = [float(row[k]) for k in ['x0','y0','z0','vx0','vy0','vz0','ax','ay','az']]
        cross_x, cross_y = float(row['x']), float(row['crossPlateY'])
        if not all(math.isfinite(v) for v in [speed, *raw, cross_x, cross_y]):
            return None, 'nonfinite'
        if not 0 < speed < 250:
            return None, 'invalid_speed'
        x0,y0,z0,vx,vy,vz,ax,ay,az = raw
        time = time_at_y(y0,vy,ay,cross_y)
        if time is None or time <= 0 or abs(x0+vx*time+.5*ax*time*time-cross_x) > .05:
            return None, 'inconsistent_crossing'
        fit = dict(zip(FIELDS,raw), tracking_status='derived_validated_x')
        values = calculate(fit)
        if values['status'] != 'derived_pts':
            return None, values['status']
        return [speed, values['hb_cm'], values['ivb_cm'], *([values['vaa_deg']] if with_vaa else [])], None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, 'missing_or_invalid_fields'


def summarize(points, vectors):
    values = np.asarray(points, dtype=np.float64)
    # Midpoint quadrature approximates the full empirical distributions; never
    # replace a distribution with its mean or generate synthetic pitches.
    levels = (np.arange(QUANTILES)+.5)/QUANTILES
    projected = (values / np.asarray(SCALES)) @ np.asarray(vectors).T
    projections = np.quantile(projected,levels,axis=0).T
    histograms, ranges = [], []
    for i, axis in enumerate(AXES):
        bins = Counter(math.floor(float(x)/axis['step']) for x in values[:,i])
        histograms.append([[index, count] for index, count in sorted(bins.items())])
        ranges.append([round(float(v),5) for v in np.quantile(values[:,i],[.05,.25,.5,.75,.95])])
    return dict(n=len(points), projections=np.round(projections,6).tolist(),
                histograms=histograms, quantiles=ranges)


def pack_projections(data):
    data['similarity']['projectionEncoding']='delta-int-1e4'
    for player in data['players']:
        for t in player['types']:
            if t:
                packed=[]
                for q in t['projections']:
                    values=[round(v*10000) for v in q]
                    packed.append([values[0]]+[b-a for a,b in zip(values,values[1:])])
                t['projections']=packed
    return data


def aggregate(rows, minimum=MIN_PITCHES, minimum_coverage=MIN_COVERAGE, season=2024):
    groups, seen, dates, games = {}, set(), set(), set()
    excluded, totals = Counter(), Counter()
    for row in rows:
        key = (row['game_id'],row['pitchId'])
        if row['game_id'][:4] != str(season):
            raise ValueError('Mixed seasons are not a comparison cohort')
        if key in seen:
            raise ValueError('Duplicate pitch identity: '+repr(key))
        seen.add(key)
        code, name = row['pitcher'],row['picher_name']
        if not code or not name:
            raise ValueError('Missing pitcher identity')
        p = groups.setdefault(code,dict(code=code,name=name,observed=0,
            points=[[] for _ in TYPE_IDS],games=set(),dates=set(),excluded=Counter()))
        if name != p['name']:
            raise ValueError('Conflicting pitcher names')
        p['observed'] += 1
        points, reason = observation(row)
        if row['stuff'] not in TYPE_IDS:
            reason = 'unknown_pitch_type'
        if reason:
            p['excluded'][reason] += 1; excluded[reason] += 1
            continue
        index = TYPE_IDS.index(row['stuff'])
        p['points'][index].append(points)
        p['games'].add(row['game_id']);p['dates'].add(row['game_id'][:8])
        totals[row['stuff']] += 1
        games.add(row['game_id']);dates.add(row['game_id'][:8])
    if not dates:
        raise ValueError('No usable pitches')
    vectors = directions()
    players = []
    eligible_pitches = 0
    for p in groups.values():
        count = sum(map(len,p['points']))
        if count < minimum or count/p['observed'] < minimum_coverage:
            continue
        types = [summarize(values,vectors) if values else None for values in p['points']]
        players.append(dict(key=p['code'],code=p['code'],name=p['name'],
            counts=[len(v) for v in p['points']],types=types,pitches=count,
            observedPitches=p['observed'],excluded=dict(p['excluded']),games=len(p['games']),
            dateStart=min(p['dates']),dateEnd=max(p['dates'])))
        eligible_pitches += count
    return dict(schemaVersion=2,version='kbo-arsenal-20261030-v1',season=season,
        dateStart=min(dates),dateEnd=max(dates),games=len(games),sourcePitches=len(seen),
        usablePitches=sum(totals.values()),eligiblePitches=eligible_pitches,
        excluded=dict(excluded),minimumPitches=minimum,minimumCoverage=minimum_coverage,
        pitchTypes=[dict(id=i,name=dict(ko=ko,en=en),color=color) for i,ko,en,color in TYPE_INFO],
        axes=AXES,similarity=dict(id='usage-overlap-sliced-wasserstein-v1',
            formula='100 * sum_k min(p_k,q_k) * exp(-SW1_k)',scales=SCALES,
            directions=vectors,quantileCount=QUANTILES,quantileRule='(i+0.5)/Q; linear empirical quantiles',
            movement=FLIGHT_VERSION,horizontalSign='Catcher right; signed; no handedness reflection',
            unknowns='Excluded; no imputation; all three features required per pitch'),
        source=dict(url=SOURCE_URL,sha256=SOURCE_SHA256,
            coverage=f'Available tracked {season} pitches, not complete season coverage',
            movement='HB/IVB derived from PTS fits; 40 ft reference; gravity/drag corrected'),
        players=sorted(players,key=lambda p:p['key']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--season',type=int,default=2024)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'scripts/pts_sources.json').read_text())
    spec=config['seasons'][str(args.season)]
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Source SHA256 mismatch')
    with args.source.open(encoding='cp949',newline='') as stream:
        result=aggregate(csv.DictReader(stream),season=args.season)
    result['source'].update(sha256=spec['sha256'],url=f"https://github.com/{config['repository']}/blob/{config['commit']}/data/{args.season}.csv")
    output=args.output or root/f'data/visuals/affinity-{args.season}.json.gz'
    index_path=root/'data/visuals/affinity.json'
    if output.resolve()==index_path.resolve():raise ValueError('affinity.json is the season index; choose a cohort output')
    output.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(pack_projections(result),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    payload=gzip.compress(encoded,compresslevel=9,mtime=0)if output.suffix=='.gz'else encoded
    output.write_bytes(payload)
    if args.output is None:
        index=json.loads(index_path.read_text())if index_path.exists()else dict(schemaVersion=3,version='kbo-pts-seasons-20261030-v1',seasons={})
        if index.get('schemaVersion')!=3:raise ValueError('Unexpected season index schema')
        index['seasons'][str(args.season)]=dict(season=args.season,dateStart=result['dateStart'],dateEnd=result['dateEnd'],sourcePitches=result['sourcePitches'],pitchers=len(result['players']),path=str(output.relative_to(root/'data')),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        index['latestSeason']=max(map(int,index['seasons']))
        index_path.write_text(json.dumps(index,separators=(',',':'))+'\n')
    print(json.dumps({key:result[key] for key in ['sourcePitches','usablePitches','eligiblePitches','games','excluded']}))
    print('Pitchers:',len(result['players']),'bytes:',output.stat().st_size)


if __name__=='__main__':main()
