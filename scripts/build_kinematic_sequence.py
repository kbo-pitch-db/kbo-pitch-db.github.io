"""Event-aligned sequence estimates, using the user's normal-speed default.

This release is separate from independently verified physical-time metrics.
Yaw-rate order does not require a plate heading: changing that heading adds
only a constant yaw offset. Clip length and pre-pitch pauses are not a clock.
"""
import argparse,csv,gzip,hashlib,io,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from biomechanics.calculation import support_masks

ROOT=Path(__file__).resolve().parents[1]
VERSION='pitch-sequence-20261029-v3'
KEYS=['pelvis','trunk','elbow']


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return round(float(x),5) if math.isfinite(x) else None
    return x


def local_velocity(values,times,mask,windows=(3,5,7),max_gap=.12):
    """Fit derivatives on observed timestamps only, never resample missing frames."""
    values=np.asarray(values,float);times=np.asarray(times,float);good=np.flatnonzero(np.asarray(mask)&np.isfinite(values))
    results={w:np.full(len(values),np.nan) for w in windows}
    if len(good)<3:return results
    spacing=float(np.median(np.diff(times[good])))
    runs=np.split(good,np.where((np.diff(times[good])>min(max_gap,spacing*2.6))|(np.diff(good)>2))[0]+1)
    for run in runs:
        if len(run)<3:continue
        v=np.degrees(np.unwrap(np.radians(values[run])))
        for width,out in results.items():
            if len(run)<width:continue
            starts=np.clip(np.arange(len(run))-width//2,0,len(run)-width)
            ids=starts[:,None]+np.arange(width)[None,:]
            scale=max(1e-6,float(np.median(np.diff(times[run]))))
            z=(times[run[ids]]-times[run,None])/scale
            design=np.stack([np.ones_like(z),z,z*z],axis=-1)
            transpose=np.swapaxes(design,1,2)
            fit=np.linalg.solve(transpose@design,(transpose@v[ids][...,None]))[...,0]
            out[run]=fit[:,1]/scale
    return results


def phase(time,pkh,fc,br):
    time=np.asarray(time,float)
    after=100*(time-fc)/(br-fc)
    if pkh is None or pkh>=fc:return after
    return np.where(time<fc,-100*(fc-time)/(fc-pkh),after)


def relation(a,b,resolution):
    if a[1]+resolution*.5<b[0]:return 'before'
    if b[1]+resolution*.5<a[0]:return 'after'
    return 'overlap'


def reviewed_events(events,unique):
    """Never promote a motion proxy or an unreviewed RGB track to an event."""
    points={};intervals={}
    for key in ['pkh','fc','br']:
        event=events.get(key,{})
        bracket=event.get('interval')
        if event.get('source')!='agent_visual_source_bracket' or not bracket:
            points[key]=None;continue
        candidates=unique[(unique>=bracket[0])&(unique<=bracket[1])]
        if not len(candidates):points[key]=None;continue
        midpoint=float(np.mean(bracket));points[key]=int(min(candidates,key=lambda i:abs(i-midpoint)))
        intervals[key]=list(bracket)
    return points,intervals


def classify_stance(xy,score,u,times,h,pk,lead):
    initial=u[times[u]-times[0]<=min(.25,max(.05,(times[pk]-times[0])*.3))]
    if len(initial)<1 or times[pk]-times[0]<.25:return 'unresolved',{'reason':'preparation_missing'}
    ids=[5,6,11,12,13+lead,15+lead,16-lead]
    if np.mean((score[initial][:,ids]>=.4).all(1))<.7:return 'unresolved',{'reason':'preparation_occluded'}
    lift=(xy[:,11:13,1].mean(1)-xy[:,13+lead,1])/h
    ground=float(np.nanmedian(lift[initial]));ankles=float(np.nanmedian(abs(xy[initial,15,1]-xy[initial,16,1]))/h)
    if ground>-.10 or ankles>.12:return 'unresolved',{'reason':'leg_lift_already_started'}
    before=u[(u<=pk)&(times[u]-times[0]<=min(.65,(times[pk]-times[0])*.8))]
    hip=np.linalg.norm(xy[:,11]-xy[:,12],axis=-1)/h;shoulder=np.linalg.norm(xy[:,5]-xy[:,6],axis=-1)/h
    hi=float(np.nanmedian(hip[initial]));sh=float(np.nanmedian(shoulder[initial]));late=u[(u<=pk)&(times[u]>=times[pk]-.2)]
    hs=float(np.nanmedian(hip[late]));ss=float(np.nanmedian(shoulder[late]))
    foot=xy[before,16-lead]/h;move=float(np.max(np.linalg.norm(foot-foot[0],axis=-1))) if len(foot) else 0
    diagnostics=dict(hipStart=hi,shoulderStart=sh,hipAtLift=hs,shoulderAtLift=ss,trailStep=move,initialKneeHeight=ground,reason='image_stance_geometry')
    if hi>.085 and sh>.115 and (hs<hi*.82 or ss<sh*.78) and move>.012:return 'windup',diagnostics
    if hi<.105 and sh<.13 and move<.045:return 'set',diagnostics
    if move>.055 and sh>.10 and hi>.08:return 'windup',diagnostics
    return 'unresolved',{**diagnostics,'reason':'stance_ambiguous'}


def analyse(clip,pose,analysis,review=None,event_data=None):
    n=len(pose['frames']);fps=pose['fps'];times=np.array([f['time'] for f in pose['frames']],float);hand=analysis['throwingHand'];factor=1 if hand=='R' else -1;lead=0 if hand=='R' else 1;arm=1 if hand=='R' else 0
    xy=np.array([[p[:2] if p else [np.nan,np.nan] for p in f['keypoints']] for f in pose['frames']]);score=np.array([[p[2] if p else 0 for p in f['keypoints']] for f in pose['frames']]);u=np.array(analysis['cadence']['uniqueImageCandidates'],int);u=u[u<n]
    h=max(50,float(np.nanmedian(np.maximum(xy[:,15,1],xy[:,16,1])-xy[:,0,1])))
    base=dict(clipId=clip['id'],playerCode=clip['playerCode'],player=clip['player'],pitchId=clip.get('playerEvidence',{}).get('pitchId'),pitch=clip['pitch'] or None,date=clip['date'],sourceUrl=clip['sourceUrl'],timeBasis='normal_speed_assumed_by_user',slowMotion='not_flagged',stance='unresolved',state='insufficient_observation',fps=fps,frameCount=n)
    if review and review.get('excludeSequence'):
        return {**base,'state':'not_pitch_delivery','reason':review['reason'],'reviewedSource':True}
    if len(u)<8:return {**base,'reason':'insufficient_independent_images'}
    ts=times[u];w=xy[u,9+arm]-xy[u,5+arm];wgood=(score[u,9+arm]>=.4)&(score[u,5+arm]>=.4)
    for d in range(2):
        good=np.isfinite(w[:,d]);w[:,d]=np.interp(np.arange(len(u)),np.flatnonzero(good),w[good,d]) if good.any() else 0
    speed=np.linalg.norm(np.gradient(gaussian_filter1d(w,.65,axis=0),ts,axis=0),axis=1)/h;speed[~wgood]=0
    ev=analysis['events'];pk_ev=ev.get('pkh',{}).get('interval');br_ev=ev.get('br',{}).get('interval');source_review=ev.get('br',{}).get('source')=='agent_visual_source_bracket'
    accepted,brackets=reviewed_events(event_data['sourceEvents'] if event_data else ev,u)
    lift=(xy[:,11:13,1].mean(1)-xy[:,13+lead,1])/h;lift[score[:,13+lead]<.4]=np.nan
    native_lift=lift[u].copy();okay=np.isfinite(native_lift)
    native_lift=np.interp(np.arange(len(u)),np.flatnonzero(okay),native_lift[okay]) if okay.any() else np.zeros(len(u))
    smoothed_lift=gaussian_filter1d(native_lift,.65)
    leg_peaks=find_peaks(smoothed_lift,prominence=.04,distance=max(2,int(.18/max(.01,np.median(np.diff(ts))))))[0]
    leg_peaks=[i for i in leg_peaks if i<len(u)*.8 and ts[i]-ts[0]>.08]
    if source_review and pk_ev:pk=int(round(np.mean(pk_ev)))
    elif leg_peaks:pk=int(u[leg_peaks[0]])
    elif pk_ev:pk=int(round(np.mean(pk_ev)))
    else:pk=int(u[np.argmax(smoothed_lift[:max(2,int(len(u)*.6))])])
    # Use the first supported acceleration after the first leg lift. A later
    # follow-through, camera change or second person cannot extend the pitch.
    search=(u>pk)&(ts<=times[min(pk,n-1)]+2.5)&(ts>=times[min(pk,n-1)]+.08)
    peaks=find_peaks(speed,distance=2)[0];available=[i for i in peaks if search[i]]
    top=max([speed[i] for i in available],default=0);strong=[i for i in available if speed[i]>=max(1.,top*.55)]
    burst=int(u[strong[0]]) if strong else int(u[np.flatnonzero(search)[np.argmax(speed[search])]]) if search.any() else min(n-1,pk+1)
    br=int(round(np.mean(br_ev))) if br_ev and np.mean(br_ev)>pk else int(burst)
    br=min(n-1,max(0,br));br_kind='source_reviewed_release' if source_review else 'image_release_candidate' if br_ev and np.mean(br_ev)>pk else 'arm_acceleration_proxy'
    if not (0<=pk<br):return {**base,'reason':'delivery_interval_unresolved'}
    fc_ev=ev.get('fc',{}).get('interval');fc=int(round(np.mean(fc_ev))) if fc_ev else None;fc_kind='source_reviewed_landing' if ev.get('fc',{}).get('source')=='agent_visual_source_bracket' else 'image_landing_candidate'
    if fc is None or not pk<fc<br:
        candidates=u[(u>pk)&(u<br)];feet=[20+lead,22+lead,24+lead];foot=xy[:,feet].mean(1);observed=(score[:,feet]>=.4).all(1)
        if len(candidates)>=4:
            ground=np.nanpercentile(foot[candidates,1],85);v=np.linalg.norm(np.gradient(gaussian_filter1d(foot[u],.6,axis=0),ts,axis=0),axis=1)/h
            eligible=[int(i) for at,i in enumerate(u) if i in candidates and at>=2 and at+2<len(u) and observed[u[at-1:at+3]].all() and foot[i,1]>=ground-.035*h and np.median(v[at+1:at+3])<max(.12,np.max(v[at-2:at+1])*.5)]
            fc=eligible[0] if eligible else None;fc_kind='image_foot_stability_candidate'
    stance,diagnostic=classify_stance(xy,score,u,times,h,pk,lead)
    if review and review.get('stance') in ['set','windup','unresolved']:stance=review['stance'];diagnostic['reason']='source_image_review';diagnostic['reviewNote']=review.get('reason')
    reviewed_aligned=accepted['fc'] is not None and accepted['br'] is not None and brackets['fc'][1]<brackets['br'][0]
    if accepted['pkh'] is not None and reviewed_aligned and brackets['pkh'][1]>=brackets['fc'][0]:accepted['pkh']=None;brackets.pop('pkh',None)
    preview=bool(event_data and event_data['previewEligible'] and not event_data['reviewed'])
    alignment_intervals=dict(brackets);alignment=dict(accepted)
    if preview:
        alignment_intervals={k:v for k,v in event_data['automaticEvents'].items() if k!='pkh' and v}
        alignment={k:int(min(u,key=lambda i:abs(i-np.mean(alignment_intervals[k])))) if k in alignment_intervals else None for k in ['pkh','fc','br']}
    aligned=reviewed_aligned or preview
    if aligned:
        fc,br=alignment['fc'],alignment['br']
        if alignment['pkh'] is not None:pk=alignment['pkh']
        else:pk=int(u[np.argmin(abs(times[u]-max(0,2*times[fc]-times[br])))])
    event_candidates=event_data['automaticEvents'] if event_data else {k:ev.get(k,{}).get('interval') for k in ['pkh','fc','br']}
    base.update(stance=stance,stanceEvidence=diagnostic,events=accepted,eventIntervals=brackets,alignmentEvents=alignment if aligned else None,alignmentIntervals=alignment_intervals if aligned else {},eventAcceptance='source_reviewed_provisional' if reviewed_aligned else 'automatic_preview' if preview else 'events_unreviewed_or_incomplete',eventCandidates=event_candidates,diagnosticWindow=dict(start=pk,end=br,motionBurst=burst,releaseProxyUsed=False),eventBasis={k:'source_reviewed_'+k if accepted[k] is not None else 'unconfirmed' for k in accepted},independentImageCount=len(u),independentMedianDt=float(np.median(np.diff(ts))),activeDuration=float(times[br]-times[pk]),peakImageWristSpeed=float(max(speed)))
    if fc is not None and 0<times[br]-times[fc]:base['landingReleasePlaybackSeconds']=float(times[br]-times[fc])
    # A long clip or duplicate images alone never indicate slow motion.
    active=times[br]-times[pk];suspect=(active>1.25 and max(speed)<3.8) or (base.get('landingReleasePlaybackSeconds',0)>.68 and active>1.1 and max(speed)<4.2)
    if review and review.get('slowMotion') is not None:suspect=review['slowMotion']=='suspected'
    base['slowMotion']='suspected' if suspect else 'not_flagged';base['timeBasis']='playback_only_slow_suspected' if suspect else 'normal_speed_assumed_by_user'
    segment=analysis['coordinate']['segment'];curves={};masks=support_masks(pose,hand)
    for key,field in [('pelvis','pelvisFramesScene'),('trunk','thoraxFramesScene')]:
        matrix=np.array(segment[field],float);a=matrix[:,:,0];yaw=factor*np.degrees(np.arctan2(-a[:,2],a[:,0]));yaw[np.hypot(a[:,0],a[:,2])<.2]=np.nan
        curves[key]=yaw
    curves['elbow']=-np.array([v if v is not None else np.nan for v in analysis['series']['elbow_flexion']],float)
    source_masks=dict(pelvis=masks['pelvis_rotation'],trunk=masks['trunk_rotation'],elbow=masks['elbow_flexion'])
    peaks={};velocities={};window=(u>=pk)&(u<=br) if aligned else np.ones(len(u),bool)
    for key,values in curves.items():
        estimates=local_velocity(values[u],ts,source_masks[key][u]);native=[]
        for width,v in estimates.items():
            ids=np.flatnonzero(window&np.isfinite(v))
            if len(ids)<3:continue
            at=int(ids[np.argmax(v[ids])]);native.append(dict(width=width,frame=int(u[at]),time=float(ts[at]),speed=float(v[at]),boundary=at in [ids[0],ids[-1]]))
        if not native:continue
        nominal=next((x for x in native if x['width']==5),native[0])
        if nominal['speed']<=0:continue
        if aligned:
            starts=alignment_intervals.get('pkh') or [max(0,2*alignment_intervals['fc'][0]-alignment_intervals['br'][1]),max(0,2*alignment_intervals['fc'][1]-alignment_intervals['br'][0])]
            for start in starts:
                for end in alignment_intervals['br']:
                    for width,v in estimates.items():
                        ids=np.flatnonzero((u>=start)&(u<=end)&np.isfinite(v))
                        if len(ids)<3:continue
                        at=int(ids[np.argmax(v[ids])]);native.append(dict(width=width,eventWindow=[start,end],frame=int(u[at]),time=float(ts[at]),speed=float(v[at]),boundary=at in [ids[0],ids[-1]]))
        resolution=float(np.median(np.diff(ts)))
        low=min(v['time'] for v in native)-resolution/2;high=max(v['time'] for v in native)+resolution/2
        peaks[key]=dict(frame=nominal['frame'],time=nominal['time'],range=[low,high],boundary=any(v['boundary'] for v in native),speedAssumedDegS=None if suspect else nominal['speed'],filterPeaks=native)
        primary=estimates[5] if np.isfinite(estimates[5][window]).any() else estimates[3];valid=np.isfinite(primary)&window;scale=max(1,max([float(x) for x in primary[valid]],default=1))
        velocities[key]=[[int(i),float(t),None if not np.isfinite(v) else float(v/scale)] for i,t,v in zip(u[window],ts[window],primary[window])]
    base['peaks']=peaks;base['curves']=velocities
    if 'pelvis' not in peaks or 'trunk' not in peaks:return {**base,'reason':'segment_peak_unavailable'}
    p,t=peaks['pelvis'],peaks['trunk'];resolution=float(np.median(np.diff(ts)));order=relation(p['range'],t['range'],resolution)
    if p['boundary'] or t['boundary']:order='overlap'
    base['pelvisTrunkOrder']=order;base['gapPlaybackMs']=1000*(t['time']-p['time']);base['gapRangePlaybackMs']=[1000*(t['range'][0]-p['range'][1]),1000*(t['range'][1]-p['range'][0])]
    base['gapAssumedMs']=None if not reviewed_aligned or suspect or p['boundary'] or t['boundary'] or base['gapRangePlaybackMs'][1]-base['gapRangePlaybackMs'][0]>160 else base['gapPlaybackMs']
    base['state']='estimated' if reviewed_aligned else 'automatic_preview' if preview else 'release_aligned_only' if accepted['br'] is not None else 'events_unverified';base['reason']=None if reviewed_aligned else 'automatic_events_not_in_headline_statistics' if preview else 'source_event_review_required'
    if aligned:
        pk_time=times[alignment['pkh']] if alignment['pkh'] is not None else None
        base['peakPhase']={k:dict(value=float(phase(v['time'],pk_time,times[fc],times[br])),range=phase(v['range'],pk_time,times[fc],times[br]).tolist()) for k,v in peaks.items()}
    return base


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    index=json.load(open(ROOT/'data/clips.json'));clips=index['clips'][:args.limit or None];review_path=ROOT/'data/kinematics/source_review.json';review=json.load(open(review_path)).get('clips',{}) if review_path.exists() else {}
    events=json.load(open(ROOT/'data/kinematics/events_v3.json'));result={}
    for i,c in enumerate(clips):
        pose=json.load(open(ROOT/'data'/c['pose']));analysis=json.load(gzip.open(ROOT/'data/biomechanics/analysis'/f"{c['id']}.json.gz"));result[c['id']]=analyse(c,pose,analysis,review.get(c['id']),events['clips'][c['id']])
        if (i+1)%100==0:print('Sequence',i+1,flush=True)
    out=dict(schemaVersion=3,version=VERSION,poseVersion=index['packageVersion'],eventVersion=events['version'],normalSpeedPolicy='User instructed default 1x playback unless unusually slow motion is suspected; not an independently verified physical clock.',alignment='FC=0%, BR=100%; PKH=-100% only when reviewed and observed. Otherwise pre-contact window spans one FC-to-BR duration and is not a knee-lift event. Strong automatic image events align previews only; headline statistics require source-reviewed events. Missing event pairs remain in clip time. No wrist-speed substitution.',segmentDefinitions={'pelvis':'Signed angular velocity about inferred vertical of calibrated model pelvis frame','trunk':'Signed angular velocity about inferred vertical of calibrated model thorax frame','elbow':'Elbow extension angular velocity, not humeral internal rotation'},clips=result)
    out['coverage']={k:dict(Counter(v[k] for v in result.values())) for k in ['stance','slowMotion','state']}
    dest=ROOT/'data/kinematics';dest.mkdir(exist_ok=True);encoded=(json.dumps(clean(out),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode();(dest/'sequences.json').write_bytes(encoded);(dest/'sequences.json.gz').write_bytes(gzip.compress(encoded,mtime=0))
    rows=io.StringIO();w=csv.writer(rows);w.writerow(['clip_id','player_code','player','date','pitch','stance','slow_motion','state','time_basis','pkh_frame','landing_frame','release_frame','release_basis','pelvis_peak_frame','trunk_peak_frame','elbow_peak_frame','pelvis_trunk_order','gap_assumed_ms','gap_playback_ms','gap_range_min_playback_ms','gap_range_max_playback_ms','stance_basis','native_median_dt_seconds','event_acceptance','sequence_version','auto_fc_min','auto_fc_max','auto_br_min','auto_br_max','source_url'])
    for c in result.values():
        ev=c.get('events',{});pk=c.get('peaks',{});w.writerow([c['clipId'],c['playerCode'],c['player'],c['date'],c['pitch'],c['stance'],c['slowMotion'],c['state'],c['timeBasis'],ev.get('pkh'),ev.get('fc'),ev.get('br'),c.get('eventBasis',{}).get('br'),pk.get('pelvis',{}).get('frame'),pk.get('trunk',{}).get('frame'),pk.get('elbow',{}).get('frame'),c.get('pelvisTrunkOrder'),c.get('gapAssumedMs'),c.get('gapPlaybackMs'),*c.get('gapRangePlaybackMs',[None,None]),c.get('stanceEvidence',{}).get('reason'),c.get('independentMedianDt'),c.get('eventAcceptance'),VERSION,*(c.get('eventCandidates',{}).get('fc') or [None,None]),*(c.get('eventCandidates',{}).get('br') or [None,None]),c['sourceUrl']])
    (dest/'sequences.csv.gz').write_bytes(gzip.compress(rows.getvalue().encode('utf-8-sig'),mtime=0));print(out['coverage']);print('bytes',len(encoded))


if __name__=='__main__':main()
