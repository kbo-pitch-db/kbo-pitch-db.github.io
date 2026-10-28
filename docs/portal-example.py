"""Read a player’s pitch records and follow an accepted link to final 2D.
Python 3 standard library only. Run: python3 portal-example.py
"""
import json
from urllib.request import urlopen

BASE = 'https://kbo-pitch-db.github.io/data/'

def read(path):
    with urlopen(BASE + path) as response:
        return json.load(response)

index = read('clips.json')
summary = read('portal/summary.json')
assert summary['poseVersion'] == index['packageVersion'], 'Release mismatch'
player = max(summary['players'], key=lambda p: p['linkedClips'])
detail = read(player['detailPath'])
clips = {clip['id']: clip for clip in index['clips']}
for pitch in detail['records']:
    for clip_id in pitch['clipIds']:
        clip = clips[clip_id]
        link = detail['clips'][clip_id]
        assert link['status'] == 'auto_accepted'
        assert link['pitchId'] == pitch['id']
        pose = read(clip.get('poseRefined') or clip['pose'])
        print(player['name'], pitch['id'], pitch['pitch'], pitch['speed'],
              len(pose['frames']), 'frames', pitch['sourceUrl'])
        # Automatic audit acceptance is not manual or biomechanical validation.
        raise SystemExit(0)
