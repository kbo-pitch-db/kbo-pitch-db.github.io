import csv,json,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'kbo_pitch_motion_pilot'))
from kbo_pitch_pilot.match_naver import observe_clip,OBS_FIELDS
from kbo_pitch_pilot.dataset import atomic_csv

dataset=Path(sys.argv[1])
sources={r['source_id']:r for r in csv.DictReader((dataset/'sources.csv').open())}
clips=list(csv.DictReader((dataset/'clips.csv').open()))
pitches=defaultdict(list)
for row in csv.DictReader((dataset/'naver_sports/pitches.csv').open()):pitches[row['naver_game_id']].append(row)
def observe(clip):
    return observe_clip(dataset,clip,sources[clip['source_id']],pitches[clip['naver_game_id']])
with ThreadPoolExecutor(max_workers=3) as pool:observations=list(pool.map(observe,clips))
atomic_csv(dataset/'naver_sports/scoreboard_observations.csv',observations,OBS_FIELDS)
print('Scoreboards',len(observations),flush=True)
