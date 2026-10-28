"""Publish source-reviewed events, separate automatic candidates, and frozen-sample evaluation."""
import gzip,hashlib,json,statistics
from pathlib import Path
import numpy as np
from audit_kinematic_events import ball_proposal,separation
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT.parent/'research/event_detector_20261029'
DEST=ROOT/'data/kinematics'

def auto_eligible(candidate,policy):
    ev=candidate['events'];r=candidate['eventEvidence'].get('br');f=candidate['eventEvidence'].get('fc',{});fps=candidate['fps']
    if not r or not ev['br']:return False
    p=policy['release'];track=r['track']
    release_ok=len(track)>=p['minimumTrackImages'] and statistics.mean(v['confidence'] for v in track)>=p['minimumMeanDetectionConfidence'] and r['trajectoryResidual']<=p['maximumTrajectoryResidualBodyHeights'] and r['associationDistance']<=p['maximumAssociationDistanceBodyHeights'] and (ev['br'][1]-ev['br'][0])/fps<=p['maximumIntervalPlaybackSeconds']
    p=policy['landing'];contact_ok=ev['fc'] is not None and f.get('beforeProbability',1)<=p['maximumBeforeContactProbability'] and f.get('afterProbability',0)>=p['minimumAfterContactProbability'] and (ev['fc'][1]-ev['fc'][0])/fps<=p['maximumIntervalPlaybackSeconds']
    return bool(release_ok and contact_ok and ev['fc'][1]<ev['br'][0])

def evaluate(refs,candidates):
    rows=[]
    for ref in refs:
        cid=ref['clipId'];a=json.load(gzip.open(ROOT/'data/biomechanics/analysis'/f'{cid}.json.gz'));auto=a['automaticEventProposals'];old=dict(fc=auto['image'].get('fc'),br=ball_proposal(auto['ballTracks'],a['cadence']['uniqueImageCandidates']));rows.append(dict(clipId=cid,reference=ref,old=old,new=candidates[cid]['events'],fps=candidates[cid]['fps']))
    summary={}
    for event in ['fc','br']:
        summary[event]={}
        for version in ['old','new']:
            eligible=[c for c in rows if c['reference'].get(event)];paired=[c for c in eligible if c[version].get(event)];delta=[abs(statistics.mean(c[version][event])-statistics.mean(c['reference'][event]))/c['fps']*1000 for c in paired];gaps=[separation(c[version][event],c['reference'][event])/c['fps']*1000 for c in paired]
            summary[event][version]=dict(referenceN=len(eligible),detectedN=len(paired),missingN=len(eligible)-len(paired),medianMidpointDifferenceMs=statistics.median(delta) if delta else None,p90MidpointDifferenceMs=float(np.percentile(delta,90)) if delta else None,overlappingBrackets=sum(v==0 for v in gaps),nonoverlapOver100ms=sum(v>100 for v in gaps))
        common=[c for c in rows if c['reference'].get(event) and c['old'].get(event) and c['new'].get(event)];summary[event]['sameDetectedClips']={'n':len(common),**{version:statistics.median([abs(statistics.mean(c[version][event])-statistics.mean(c['reference'][event]))/c['fps']*1000 for c in common]) if common else None for version in ['old','new']}}
    return dict(summary=summary,clips=rows)

def main():
    frozen=json.load(open(PRIVATE/'validation_freeze.json'))
    if hashlib.sha256((PRIVATE/'contact_model.json').read_bytes()).hexdigest()!=frozen['modelSha256']:raise ValueError('Contact head differs from the frozen evaluation model')
    index=json.load(open(ROOT/'data/clips.json'));old={r['clipId']:r for r in json.load(open(ROOT/'data/metrics/source_reviews_20261029.json'))['clips']};new=json.load(open(PRIVATE/'new_source_reviews.json'));reviews={**old,**{r['clipId']:r for r in new['clips']}};freeze=json.load(open(PRIVATE/'validation_freeze.json'));excluded=json.load(open(DEST/'source_review.json'))['clips'];candidates={c['id']:json.load(open(PRIVATE/'candidates'/f"{c['id']}.json")) for c in index['clips']};out={}
    for clip in index['clips']:
        cid=clip['id'];c=candidates[cid];r=reviews.get(cid);source={key:dict(interval=r.get(key),source='agent_visual_source_bracket',state='provisional' if r.get(key) else 'missing') for key in ['pkh','fc','br']} if r else {};evidence=c['eventEvidence'];ball=evidence.get('br');foot=evidence.get('fc',{})
        out[cid]=dict(sourceEvents=source,automaticEvents=c['events'],previewEligible=auto_eligible(c,freeze['publicationPolicy']),reviewed=r is not None,modelSha256=c['modelSha256'],releaseEvidence={k:ball[k] for k in ['associationDistance','trajectoryResidual','track']} if ball else None,landingEvidence={k:foot[k] for k in ['method','beforeProbability','afterProbability','loss','reason'] if k in foot},footIdentitySwitchCount=len(evidence.get('footLabelSwapFrames',[])),cameraStabilized=c['cameraStabilized'],fps=c['fps'],sourceUrl=clip['sourceUrl'])
        if excluded.get(cid,{}).get('excludeSequence'):out[cid].update(automaticEvents=dict(pkh=None,fc=None,br=None),previewEligible=False,excluded=True)
    data=dict(schemaVersion=1,version='image-pitch-events-20261029-v3',poseVersion=index['packageVersion'],policy='Source-reviewed provisional events and automatic image-supported previews are separate. No wrist-speed substitution. Automatic previews never enter player headline statistics.',clips=out)
    encoded=(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n').encode();(DEST/'events_v3.json').write_bytes(encoded);(DEST/'events_v3.json.gz').write_bytes(gzip.compress(encoded,mtime=0));(DEST/'source_event_reviews_v3.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');(DEST/'models').mkdir(exist_ok=True);(DEST/'models/foot_contact_linear.json').write_bytes((PRIVATE/'contact_model.json').read_bytes())
    report=evaluate(new['clips'],candidates);report.update(version='event-validation-20261029-v3',independentHumanGroundTruth=False,referenceScope='12 new clip-disjoint provisional image reviews; 10 visible contact references, 12 release brackets. Shared source videos with development exist. Detector core and head frozen before native event-bracket annotation. Preliminary coarse sheets only located review windows; no new labels used for fitting or tuning. No population accuracy claim.',freeze=freeze,developmentClipIds=list(old),modelSha256=hashlib.sha256((PRIVATE/'contact_model.json').read_bytes()).hexdigest(),upstream=dict(ballModel='https://github.com/BaseballCV/BaseballCV',ballWeightSha256=next(iter(candidates.values()))['modelSha256'],footEncoder='https://github.com/facebookresearch/dinov2',encoderRevision='e1277af2ba9496fbadf7aec6eba56e8d882d1e35'),coverage=dict(totalClips=len(out),releaseCandidates=sum(bool(c['automaticEvents']['br']) for c in out.values()),landingCandidates=sum(bool(c['automaticEvents']['fc']) for c in out.values()),automaticPreviewEligible=sum(c['previewEligible'] and not c['reviewed'] for c in out.values()),reviewedClips=sum(c['reviewed'] for c in out.values())))
    (DEST/'event_validation_v3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'coverage':report['coverage'],'evaluation':report['summary']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
