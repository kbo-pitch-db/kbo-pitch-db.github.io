"""Import observed PTS rows when the relay API is unavailable.

The published CSV omits some pitches and relay metadata. Never reconstruct
cumulative pitch counts, half innings, at-bat IDs or before-pitch counts from
its row order. Such rows cannot be automatically linked using those fields.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_BLOB = 'c96f7e24fb50f08381b6c8060342f7a3d4ccd754'
SOURCE_SHA256 = 'b4fc2497f579cc60c714b54c1fe91e9046c905cccc4dfae6eae6f85a2a6d21f4'
SOURCE_URL = 'https://github.com/leekichang/where-do-the-robot-umpires-see/blob/37cf62be0604aa8a7f077759eebf7a6eda002edb/data/2024.csv'
RESULTS = {'Ball': 'B', 'Strike': 'T', 'Swing': 'S', 'Hit': 'H',
           'Foul': 'F', 'BuntFoul': 'W', 'BuntSwing': 'V'}
PITCHES = {'4-Seam': '직구', '2-Seam': '투심', 'Cutter': '커터',
           'Slider': '슬라이더', 'Fork': '포크', 'Curve': '커브',
           'Change-up': '체인지업', 'Knuckle': '너클볼', 'Sinker': '싱커'}
TEAMS = {'KIA': 'HT', 'NC': 'NC', '롯데': 'LT', 'KT': 'KT', '삼성': 'SS',
         '키움': 'WO', 'LG': 'LG', 'SSG': 'SK', '한화': 'HH', '두산': 'OB'}


def read(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.csv')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def convert(row, fields, trajectory, raw_file):
    result = dict.fromkeys(fields, '')
    game = row['game_id']
    date = '-'.join([game[:4], game[4:6], game[6:8]])
    result.update(naver_pitch_id=game+':'+row['pitchId'], naver_game_id=game,
                  season=game[:4], game_date=date, inning=row['inning'],
                  pitch_number=row['pitch_num'], pts_pitch_id=row['pitchId'],
                  pitcher_code=row['pitcher'], pitcher_name=row['picher_name'],
                  batter_code=row['batter'], batter_name=row['batter_name'],
                  pitch_result=RESULTS[row['result']], pitch_type_raw=PITCHES.get(row['stuff'], ''),
                  speed_kmh=row['speed'], source_url=SOURCE_URL, raw_file=raw_file,
                  extraction_method='published_naver_pts_csv', review_status='source_record_unreviewed')
    numeric = {key: row[key] for key in ['x0','y0','z0','vx0','vy0','vz0','ax','ay','az']}
    numeric.update(crossPlateX=row['x'], crossPlateY=row['crossPlateY'])
    track, status = trajectory(numeric)
    result['tracking_status'] = status
    for source, target in [('x0','x0_ft'),('y0','y0_ft'),('z0','z0_ft'),
                           ('vx0','vx0_ft_s'),('vy0','vy0_ft_s'),('vz0','vz0_ft_s'),
                           ('ax','ax_ft_s2'),('ay','ay_ft_s2'),('az','az_ft_s2'),
                           ('x','cross_plate_x_ft'),('crossPlateY','cross_plate_y_ft'),
                           ('topSz','top_sz_ft'),('bottomSz','bottom_sz_ft')]:
        result[target] = row[source]
    if track:
        result.update(trajectory_points_json=json.dumps(track['points'], separators=(',', ':')),
                      trajectory_coordinate_system='Naver PTS local x/y/z in feet; quadratic PITCHf/x model, y0 to crossPlateY',
                      trajectory_time_unit='seconds from y0 reference',
                      trajectory_duration_s=track['duration'], plate_z_ft_derived=track['plate_z'],
                      trajectory_x_residual_ft=track['x_residual'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    payload = args.csv.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError('Published CSV changed; validate a new source version before importing')
    if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest() != SOURCE_BLOB:
        raise ValueError('Git source identity mismatch')
    sys.path.insert(0, str(WORKSPACE/'kbo_pitch_motion_pilot'))
    from kbo_pitch_pilot.naver import trajectory
    from kbo_pitch_pilot.dataset import PITCH_FIELDS
    external = list(csv.DictReader(args.csv.open(encoding='cp949')))
    by_game = {}
    for row in external:
        by_game.setdefault(row['game_id'], []).append(row)
    sources = read(args.dataset/'sources.csv')
    pitches = {r['naver_pitch_id']: r for r in read(args.dataset/'naver_sports/pitches.csv')}
    source_games, imported = {}, 0
    raw_relative = f'naver_sports/raw/published_csv/2024-{SOURCE_BLOB}.csv'
    for source in sources:
        teams = source['teams_text'].split(' vs ')
        expected = {TEAMS[t] for t in teams}
        candidates = [g for g in by_game if g[:8] == source['game_date'].replace('-', '')
                      and {g[8:10], g[10:12]} == expected]
        if len(candidates) != 1:
            raise ValueError('Game identity is missing or ambiguous: '+source['source_id'])
        game = candidates[0]
        source['naver_game_id'] = game
        source['naver_game_match_status'] = 'published_pts_date_teams_unique'
        source_games[source['source_id']] = game
        for row in by_game[game]:
            key = game+':'+row['pitchId']
            if key in pitches:
                continue
            pitches[key] = convert(row, PITCH_FIELDS, trajectory, raw_relative)
            imported += 1
    clips = read(args.dataset/'clips.csv')
    for clip in clips:
        clip['naver_game_id'] = source_games[clip['source_id']]
    target = args.dataset/raw_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(args.csv, target)
    write(args.dataset/'naver_sports/pitches.csv', list(pitches.values()), PITCH_FIELDS)
    write(args.dataset/'sources.csv', sources, list(sources[0]))
    write(args.dataset/'clips.csv', clips, list(clips[0]))
    report = dict(sourceUrl=SOURCE_URL, sourceGitBlob=SOURCE_BLOB, sourceSha256=SOURCE_SHA256,
                  games=source_games, recordCount=len(pitches), importedRowsThisRun=imported, rowScope='Available published tracked pitches; not all game pitches',
                  missingFields=['half_inning','pitcher_pitch_number','relay_seqno','at_bat_id',
                                 'balls_before','strikes_before','outs_before'],
                  automaticPitchIdentityLinksCreated=0)
    (args.dataset/'naver_sports/published_csv_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
