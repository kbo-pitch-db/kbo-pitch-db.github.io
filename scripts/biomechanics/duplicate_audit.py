"""Exact reused-image/source-overlap audit. Hashes are not redistributed RGB.

Failure to find an exact match does not establish distinct physical pitches;
recompressed, recropped or slow-motion replays remain unresolved until linked.
"""
import collections,csv,hashlib,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
REPO=Path(__file__).resolve().parents[2]


def fingerprint(job):
 c,row=job;cv2.setNumThreads(1);cap=cv2.VideoCapture(str(REPO.parent/'dataset'/row['crop_video']));hashes=set();n=0
 while True:
  ok,image=cap.read()
  if not ok:break
  # Require exact downsampled pixel agreement, not a weak form-similarity score.
  gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(96,96));hashes.add(hashlib.sha256(gray.tobytes()).hexdigest());n+=1
 cap.release();return c['id'],hashes,n


def main():
 clips=json.loads((REPO/'data/clips.json').read_text())['clips'];catalog={r['dataset_clip_id']:r for r in csv.DictReader(open(REPO.parent/'dataset/clips.csv'))};rows=json.loads((REPO/'data/metrics/database.json').read_text())['rows'];row_by_clip={r['primaryClipId']:r for r in rows if r['primaryClipId']};cache=REPO.parent/'research/biomechanics_20261029/image_fingerprints.json'
 if cache.exists():data=json.loads(cache.read_text())
 else:
  with ProcessPoolExecutor(max_workers=4) as pool:data={cid:dict(hashes=sorted(h),frames=n) for cid,h,n in pool.map(fingerprint,[(c,catalog[c['id']]) for c in clips])}
  cache.write_text(json.dumps(data))
 matches=[]
 for i,a in enumerate(clips):
  for b in clips[i+1:]:
   ra,rb=row_by_clip[a['id']],row_by_clip[b['id']]
   if ra['playerCode']!=rb['playerCode'] or ra['gameId']!=rb['gameId']:continue
   shared=len(set(data[a['id']]['hashes'])&set(data[b['id']]['hashes']));overlap=0
   if a['sourceId']==b['sourceId']:
    overlap=max(0,min(a['sourceEnd'],b['sourceEnd'])-max(a['sourceStart'],b['sourceStart']))/max(.001,min(a['duration'],b['duration']))
   if shared>=3 or overlap>=.8:
    matches.append(dict(clips=[a['id'],b['id']],sharedExactImageHashes=shared,sourceIntervalOverlapFraction=overlap,pitchIds=[ra['pitchId'],rb['pitchId']],acceptedIdentity=False,disposition='requires source identity adjudication; do not merge automatically'))
 accepted=[r for r in rows if r['unit']=='pitch_record' and r['primaryClipId']]
 output=dict(clips=len(clips),framesDecoded=sum(v['frames'] for v in data.values()),method='Same player/game, >=3 exact 96x96 grayscale image hashes or >=80% shared source interval. Not a perceptual biomechanics classifier.',candidates=matches,duplicateAcceptedPitchIds=len(accepted)-len({r['pitchId'] for r in accepted}),unresolvedClipUnitPolicy='Unlinked clips remain clip observations and never inflate unique-pitch aggregates.',limitations='Recompressed, recropped, mirrored or speed-edited replays can evade exact hashes; zero matches does not establish physical uniqueness.')
 (REPO/'data/metrics/duplicate_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k not in ['candidates','method','limitations','unresolvedClipUnitPolicy']}),len(matches),'candidates',flush=True)

if __name__=='__main__':main()
