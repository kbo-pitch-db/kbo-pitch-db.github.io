"""Compare preserved automatic proposals with provisional source-image brackets.

This is a discrepancy audit, not independent human-ground-truth validation.
No source images or private paths are written to the public report.
"""
import gzip,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def separation(a,b):return max(0,b[0]-a[1],a[0]-b[1])
def ball_proposal(tracks,unique):
    if not tracks:return None
    best=tracks[0]
    if len(tracks)>1 and tracks[1]['score']>.85*best['score'] and abs(tracks[1]['frames'][0]-best['frames'][0])>2:return None
    at=unique.index(best['frames'][0]);return [unique[max(0,at-1)],best['frames'][1]]
def main():
    reviews=json.load(open(ROOT/'data/metrics/source_reviews_20261029.json'))['clips'];clips={c['id']:c for c in json.load(open(ROOT/'data/clips.json'))['clips']};rows=[]
    for review in reviews:
        cid=review['clipId'];a=json.load(gzip.open(ROOT/'data/biomechanics/analysis'/f'{cid}.json.gz'));auto=a['automaticEventProposals'];u=a['cadence']['uniqueImageCandidates'];raw={k:auto['image'].get(k) for k in ['pkh','fc']};raw['br']=ball_proposal(auto['ballTracks'],u)
        rows.append(dict(clipId=cid,player=clips[cid]['player'],sourceUrl=clips[cid]['sourceUrl'],fps=a['cadence']['fileFps'],reviewed={k:review.get(k) for k in ['pkh','fc','br']},automatic=raw))
    summary={}
    for key in ['pkh','fc','br']:
        reference=[r for r in rows if r['reviewed'][key]];paired=[r for r in reference if r['automatic'][key]]
        summary[key]=dict(reviewed=len(reference),automaticAvailable=len(paired),automaticMissing=len(reference)-len(paired),overlappingBrackets=sum(separation(r['reviewed'][key],r['automatic'][key])==0 for r in paired),medianMidpointDiscrepancyPlaybackMs=statistics.median([abs(statistics.mean(r['automatic'][key])-statistics.mean(r['reviewed'][key]))/r['fps']*1000 for r in paired]) if paired else None)
    result=dict(version='event-discrepancy-20261029-v1',independentGroundTruth=False,scope='19 source-reviewed provisional brackets selected in the prior release; selected sample, not held-out ground truth. Missing detections counted separately. Timing is playback time.',summary=summary,clips=rows)
    (ROOT/'data/kinematics/event_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
