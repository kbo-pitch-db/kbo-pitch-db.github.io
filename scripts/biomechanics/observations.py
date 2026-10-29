"""Independent image-event hypotheses and decoded-image cadence diagnostics.

No model contact pins, MER, normative knee angles or physical-clock guesses
enter event detection. A wrist burst only narrows a ball search window; it is
never published as ball release. Unknown events remain unknown.
"""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

REPO=Path(__file__).resolve().parents[2]
STUDY=REPO.parent/'research/pitch_observation_20261028'
OUT=REPO.parent/'research/biomechanics_20261029/observations'


def valid_point(frame,index,threshold=.4):
    p=frame['keypoints'][index]
    if not p or len(p)<3 or not np.isfinite(p[:3]).all() or p[2]<threshold:return False
    for key in ['interpolated','outlier']:
        flag=frame.get(key,False)
        if flag[index] if isinstance(flag,list) else flag:return False
    return True


def detect_events(xy,score,unique,fps,height,hand,stabilized=None):
    n=len(xy);lead=0 if hand=='R' else 1;arm=1 if hand=='R' else 0
    if len(unique)<8:return dict(pkh=None,fc=None,armWindow=None,reason='insufficient_independent_images')
    points=xy[unique];confidence=score[unique];t=np.asarray(unique)
    wrist=points[:,9+arm]-points[:,5+arm];wgood=(confidence[:,9+arm]>=.4)&(confidence[:,5+arm]>=.4)
    velocity=np.linalg.norm(np.gradient(wrist,axis=0),axis=-1)/height;velocity[~wgood]=0
    peaks=find_peaks(velocity,distance=3)[0];peaks=[i for i in peaks if i<len(t)*.9]
    if not peaks:return dict(pkh=None,fc=None,armWindow=None,reason='delivery_window_not_identified')
    amplitude=max(velocity[p] for p in peaks);strong=[p for p in peaks if velocity[p]>=.65*amplitude]
    burst=int(min(strong));window=[int(t[max(0,burst-4)]),int(t[min(len(t)-1,burst+4)])]
    lift=(points[:,11:13,1].mean(1)-points[:,13+lead,1])/height
    lift_good=(confidence[:,13+lead]>=.4)&(confidence[:,11:13].min(1)>=.4)
    lift[~lift_good]=-np.inf;limit=max(1,burst-2);pk=int(np.argmax(lift[:limit]))
    pkh=[int(t[max(0,pk-1)]),int(t[min(len(t)-1,pk+1)])] if pk>1 and pk<limit-1 and lift[pk]>.04 else None
    # Require genuinely observed toe/heel trajectories, not a model-pinned foot.
    field=stabilized if stabilized is not None else xy
    foot_ids=[20+lead,22+lead,24+lead]
    observed=(score[:,foot_ids]>=.4).all(1)
    toes=field[:,foot_ids].mean(1)[unique]/height
    speed=np.linalg.norm(np.gradient(median_filter(toes,size=(3,1),mode='nearest'),axis=0),axis=-1)
    start=max(1,pk+1) if pkh else max(1,burst-20);end=min(len(t)-4,burst+1);proposals=[]
    for ratio in [.25,.35,.45]:
        found=[]
        for k in range(start,end):
            before=speed[max(start,k-4):k+1];after=speed[k+1:k+4]
            if len(before)<2 or len(after)<3 or not observed[t[max(0,k-1):min(len(t),k+4)]].all():continue
            moving=float(np.max(before));stable=float(np.median(after))
            spread=np.ptp(toes[k+1:k+4],axis=0).max()
            lowered=(lift[k] if np.isfinite(lift[k]) else 0)<max(.06,lift[pk]*.65 if pkh else .06)
            if moving>.015 and stable<moving*ratio and spread<.025 and lowered:found.append(k)
        if found:
            # First stable image interval after the observed stride movement.
            k=found[0];proposals.append([int(t[max(start,k-1)]),int(t[min(len(t)-1,k+2)])])
    fc=[min(p[0] for p in proposals),max(p[1] for p in proposals)] if proposals else None
    if fc and fc[1]>=window[1]:fc=None
    return dict(pkh=pkh,fc=fc,armWindow=window,armBurstFrame=int(t[burst]),
                source='image_foot_deceleration_hypothesis',plantProposals=proposals,
                reason=None if fc else 'foot_contact_not_identified')


def white_components(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,2]>215)&(hsv[:,:,1]<65)).astype('uint8')
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    return [p for stat,p in zip(stats[1:],centroids[1:]) if 2<=stat[4]<=90 and .3<=stat[2]/max(1,stat[3])<=3]


def ball_release(frames,xy,score,unique,window,hand,height,crop_origin):
    if window is None:return None,[]
    arm=10 if hand=='R' else 9;eligible=[i for i in unique if window[0]<=i<=window[1]];components={}
    for i in eligible:
        if score[i,arm]<.4:continue
        wrist=xy[i,arm]-crop_origin
        components[i]=[p for p in white_components(frames[i]) if np.linalg.norm(p-wrist)<.6*height]
    tracks=[]
    for at,i in enumerate(eligible[:-2]):
        j,k=eligible[at+1:at+3];w=xy[i,arm]-crop_origin
        for a in components.get(i,[]):
            if np.linalg.norm(a-w)>.075*height:continue
            for b in components.get(j,[]):
                v=b-a;distance=np.linalg.norm(v)
                if not .008*height<distance<.25*height:continue
                for c in components.get(k,[]):
                    u=c-b;length=np.linalg.norm(u)
                    if not .008*height<length<.25*height or np.dot(u,v)/(length*distance)<.65:continue
                    d=[np.linalg.norm(p-(xy[f,arm]-crop_origin)) for f,p in zip([i,j,k],[a,b,c])]
                    if d[2]<.07*height or d[2]<d[0]+.04*height:continue
                    score_track=(distance+length)/height-d[0]/height
                    tracks.append(dict(frames=[int(i),int(j),int(k)],points=[p.round(3).tolist() for p in [a,b,c]],score=float(score_track)))
    tracks.sort(key=lambda t:t['score'],reverse=True)
    if not tracks:return None,[]
    best=tracks[0]
    if len(tracks)>1 and tracks[1]['score']>.85*best['score'] and abs(tracks[1]['frames'][0]-best['frames'][0])>2:return None,tracks[:3]
    p=unique.index(best['frames'][0]);interval=[unique[max(0,p-1)],best['frames'][1]]
    return interval,tracks[:3]


def process(job):
    cv2.setNumThreads(1);clip,row,hand,review=job;cid=clip['id'];dest=OUT/(cid+'.json')
    pose_path=REPO/'data'/clip['pose'];pose=json.loads(pose_path.read_text());frames=pose['frames'];n=len(frames)
    sha=hashlib.sha256(pose_path.read_bytes()).hexdigest()
    video_path=REPO.parent/'dataset'/row['crop_video'];video_stat=video_path.stat();signature=hashlib.sha256(json.dumps(dict(script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pose=sha,review=review,hand=hand,catalog=row,videoBytes=video_stat.st_size,videoMtimeNs=video_stat.st_mtime_ns),sort_keys=True).encode()).hexdigest()
    if dest.exists():
        old=json.loads(dest.read_text())
        if old.get('version')=='image-events-20261029-v1' and old.get('poseSha256')==sha and old.get('inputSignature')==signature:return old
    original_origin=np.array([float(row['crop_x1']),float(row['crop_y1'])]);origin=np.asarray(pose.get('crop_box',original_origin.tolist())[:2],float)
    xy=np.zeros((n,26,2));scores=np.zeros((n,26))
    for i,f in enumerate(frames):
        for j,p in enumerate(f['keypoints']):
            if valid_point(f,j,.2):xy[i,j]=np.asarray(p[:2])+origin;scores[i,j]=p[2]
    height=float(np.nanmedian(np.maximum(xy[:,15,1],xy[:,16,1])-xy[:,0,1]));height=max(120,min(1200,height))
    cap=cv2.VideoCapture(str(REPO.parent/'dataset'/row['crop_video']));images=[];unique=[];differences=[];previous=None
    for i in range(n):
        ok,image=cap.read()
        if not ok:break
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(160,160)).astype(float)
        difference=float(np.mean(abs(gray-previous))) if previous is not None else None
        if i==0 or difference>=1.:unique.append(i)
        if difference is not None:differences.append(difference)
        images.append(image);previous=gray
    cap.release()
    if len(images)!=n:
        result=dict(clipId=cid,version='image-events-20261029-v1',poseSha256=sha,inputSignature=signature,events={},missingReason='source_rgb_unavailable',cadence={'physicalTimeStatus':'unknown'})
        dest.write_text(json.dumps(result));return result
    # Camera motion correction is image-derived; never inferred from a pinned 3D foot.
    bg=STUDY/'all813_gvhmr_v1/background'/(cid+'.npz');stabilized=None
    if bg.exists():
        with np.load(bg) as data:
            transforms=data['full_transforms']
            if len(transforms)==n:stabilized=np.einsum('tij,tkj->tki',transforms,np.concatenate([xy,np.ones((n,26,1))],-1))[...,:2]
    proposal=detect_events(xy,scores,unique,float(pose['fps']),height,hand,stabilized)
    br,tracks=ball_release(images,xy,scores,unique,proposal.get('armWindow'),hand,height,original_origin)
    events={}
    for key in ['pkh','fc']:
        interval=proposal.get(key)
        events[key]=dict(interval=interval,source='image_keypoint_hypothesis',state='provisional' if interval else 'missing',reason=None if interval else ('preparation_outside_clip_or_unresolved' if key=='pkh' else proposal.get('reason')))
    events['br']=dict(interval=br,source='rgb_ball_departure_hypothesis',state='provisional' if br else 'missing',reason=None if br else 'ball_release_not_observed')
    events['mer']=dict(interval=None,source=None,state='missing',reason='humeral_axial_rotation_unidentifiable')
    if review:
        for key in ['pkh','fc','br']:
            interval=review.get(key);events[key]=dict(interval=interval,source='agent_visual_source_bracket',state='provisional' if interval else 'missing',reason=None if interval else 'event_outside_clip_or_unresolved')
    # Reject a failed chronology; never sort event times to conceal it.
    fc=events['fc']['interval'];release=events['br']['interval']
    if fc and release and fc[1]>=release[0]:
        events['br'].update(interval=None,state='missing',reason='event_order_ambiguous')
    gaps=np.diff(unique);ratio=sum(x<1 for x in differences)/max(1,len(differences));fps=float(pose['fps'])
    result=dict(clipId=cid,version='image-events-20261029-v1',poseSha256=sha,inputSignature=signature,events=events,
        imageProposal=proposal,ballTracks=tracks,cropOrigin=origin.tolist(),legacyCropOrigin=original_origin.tolist(),
        coordinateOriginCorrected=bool(np.any(origin!=original_origin)),
        cadence=dict(fileFps=fps,uniqueImageCandidates=unique,nearDuplicatePairs=sum(x<1 for x in differences),decodedFrames=n,
            inferredIndependentImageRate=round(fps*(1-ratio),3),physicalTimeStatus='unverified_playback_rate',physicalSecondsPerPlaybackSecond=None,
            effectiveResolutionClaim='Image-cadence diagnostic only; no certified independent physical sampling rate',
            uniformCadenceFraction=float(np.mean(gaps==np.median(gaps))) if len(gaps) else None),
        frameTimes=[f['source_time'] for f in frames],sourceFrames=[f['source_frame'] for f in frames],
        inspected=bool(review),view=review.get('view') if review else None,pitchAxisCamera=review.get('pitchAxisCamera') if review else None,
        axisUncertaintyDegrees=review.get('axisUncertaintyDegrees') if review else None,
        originalImagesPublished=False,externalEventAccuracyValidated=False)
    dest.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n');return result


def main():
    global STUDY
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--ids',type=Path);parser.add_argument('--study-root',type=Path);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.study_root:STUDY=args.study_root.resolve()
    clips=json.loads((REPO/'data/clips.json').read_text())['clips'];profiles={p['code']:p for p in json.loads((REPO/'data/portal/summary.json').read_text())['players']}
    if args.ids:
        wanted=set(json.loads(args.ids.read_text()));clips=[c for c in clips if c['id'] in wanted]
    rows={r['dataset_clip_id']:r for r in csv.DictReader(open(REPO.parent/'dataset/clips.csv'))};reviews={r['clipId']:r for r in json.loads((REPO/'data/metrics/source_reviews_20261029.json').read_text())['clips']}
    if args.limit:clips=clips[:args.limit]
    jobs=[(c,rows[c['id']],profiles[c['playerCode']]['hand'],reviews.get(c['id'])) for c in clips]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(process,jobs)):
            results.append(result)
            if (i+1)%100==0:print('Image events / cadence',i+1,'/',len(jobs),flush=True)
    counts={event:sum(bool(r.get('events',{}).get(event,{}).get('interval')) for r in results) for event in ['pkh','fc','br','mer']}
    print(json.dumps(dict(clips=len(results),eventCounts=counts,coordinateOriginCorrections=sum(r.get('coordinateOriginCorrected',False) for r in results)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
