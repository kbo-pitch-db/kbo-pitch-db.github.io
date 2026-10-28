"""Event-aligned 3D measurements with explicit per-metric observability.

Spatial computation never substitutes a 2D projection. Unknown physical time
blocks physical derivatives only, not spatial measurements. All sensitivity
thresholds are operational diagnostics, not normal/abnormal athlete cutoffs.
"""
import itertools
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .geometry import segment_series,pitch_frame,signed_horizontal,unit


def supported(frame,index):
    p=frame['keypoints'][index]
    if not p or len(p)<3 or not np.isfinite(p[:3]).all() or p[2]<.4:return False
    for key in ['interpolated','outlier']:
        flag=frame.get(key,False)
        if flag[index] if isinstance(flag,list) else flag:return False
    return True


def support_masks(pose,hand):
    lead=[11,13,15] if hand=='R' else [12,14,16];arm=[6,8,10] if hand=='R' else [5,7,9]
    pelvis=[11,12];thorax=[5,6,11,12];arm_upper=arm[:2];feet=[20,22,24] if hand=='R' else [21,23,25]
    ids=dict(lead_knee_flexion=lead,elbow_flexion=arm,thorax_pelvis_relative_axial=thorax,
       shoulder_line_separation=thorax,upper_arm_elevation=thorax+arm_upper,shoulder_horizontal_abduction=thorax+arm_upper,
       pelvis_rotation=pelvis,trunk_rotation=thorax,trunk_forward_lean=thorax,trunk_lateral_lean=thorax,
       foot_lateral=[15,16],foot_direction=feet,stride_length_cm=[15,16],stride_height_pct=[15,16])
    return {key:np.asarray([all(supported(f,j) for j in joints) for f in pose['frames']]) for key,joints in ids.items()}


def unwrap_local(values):return np.degrees(np.unwrap(np.radians(values)))


def evaluate(metric,series,events,unique,times,masks,override=None):
    event=metric['referenceEvent'];needed=['fc','br'] if event in ['fc_br','pkh_br'] else [event]
    if event=='pkh_br':needed=['pkh','br']
    if any(not events.get(e,{}).get('interval') for e in needed):return None,[], 'event_not_identified'
    bounds={e:events[e]['interval'] for e in needed}
    points={e:int(np.floor(np.mean(bounds[e])+.5)) for e in needed}
    if override:points.update(override)
    unique=np.asarray(unique,dtype=int)
    if not len(unique):return None,[],'source_pose_unavailable'
    points={e:int(unique[np.argmin(abs(unique-i))]) for e,i in points.items()}
    key=metric.get('series') or metric['operation'];v=series.get(key);mask=masks.get(key)
    if v is None:return None,[],'pitch_axis_unavailable'
    mask=np.asarray(mask) if mask is not None else np.ones(len(v),bool)
    mask=mask[:len(v)]&np.isfinite(v)
    if any(i>=len(v) or not np.isfinite(v[i]) for i in points.values()):
        absent=any(i>=len(v) or not masks.get('source_pose_available',np.ones(len(v),bool))[i] for i in points.values())
        return None,list(points.values()),'source_pose_unavailable' if absent else 'degenerate_coordinate_geometry'
    if any(not mask[i] for i in points.values()):return None,list(points.values()),'required_joints_occluded'
    op=metric['operation']
    if op in ['event','foot_lateral','foot_direction']:
        i=points[needed[0]];return float(v[i]),[i],None
    a,b=points[needed[0]],points[needed[-1]]
    if b<=a:return None,[a,b],'event_order_ambiguous'
    frames=unique[(unique>=a)&(unique<=b)];good=frames[mask[frames]]
    if len(good)<3 or len(good)<.8*len(frames):return None,good.tolist(),'required_joints_occluded'
    w=v[good]
    if key in ['thorax_pelvis_relative_axial','shoulder_line_separation','pelvis_rotation','trunk_rotation']:w=unwrap_local(w)
    if op=='net_extension':out=v[a]-v[b]
    elif op=='change':out=unwrap_local(v[[a,b]])[1]-v[a] if 'rotation' in key else v[b]-v[a]
    elif op=='maximum':out=np.max(w)
    elif op=='extra_flexion':out=max(0,np.max(w)-v[a])
    elif op=='argmax':out=100*(times[good[np.argmax(w)]]-times[a])/(times[b]-times[a])
    elif op=='transition':
        peak=int(np.argmax(w));out=None
        for k in range(peak,len(w)-2):
            if w[k+1]<w[k] and w[k+2]<w[k+1] and w[k]-w[k+2]>=1:
                out=100*(times[good[k]]-times[a])/(times[b]-times[a]);break
        if out is None:return None,good.tolist(),'extension_transition_unresolved'
    else:raise ValueError('Unimplemented operation '+op)
    return float(out),good.tolist(),None


def measurement(metric,series,observation,masks,alternatives=None,axis_variants=None):
    events=observation['events'];unique=observation['cadence'].get('uniqueImageCandidates',[]);times=np.asarray(observation['frameTimes'])
    event=metric['referenceEvent'];needed=['fc','br'] if event=='fc_br' else ['pkh','br'] if event=='pkh_br' else [event]
    sources={e:events.get(e,{}).get('source') for e in needed}
    out=dict(value=None,status=None,calculationVersion=metric['calculationVersion'],definitionCorrespondence=metric['definitionCorrespondence'],
        externalAccuracyValidation='not_validated_against_external_GT',eventSource=sources,timeStatus=observation['cadence']['physicalTimeStatus'],
        usedFrames=[],validFrames=0,totalFrames=len(times),stability='not_evaluable',missingReason=None)
    reason=metric.get('blockedReason')
    requirements=[]
    if any(not events.get(e,{}).get('interval') for e in needed):requirements.append('event_not_identified')
    axis_series=['pelvis_rotation','trunk_rotation','trunk_forward_lean','trunk_lateral_lean']
    if (metric.get('series') in axis_series or metric['operation'] in ['foot_lateral','foot_direction'] or metric['id'].startswith(('pelvis_peak','trunk_peak')) or metric['id']=='pelvis_trunk_peak_gap_ms') and observation.get('pitchAxisCamera') is None:requirements.append('pitch_axis_unavailable')
    if reason and (reason!='physical_time_unverified' or observation['cadence'].get('physicalTimeStatus')!='verified' or not observation['cadence'].get('physicalClockEvidence')):requirements.append(reason)
    out['missingRequirements']=sorted(set(requirements))
    if reason in ['metric_scale_unverified','measured_stature_unavailable'] and metric['id'] in series:reason=None;out['missingRequirements']=[r for r in requirements if r not in ['metric_scale_unverified','measured_stature_unavailable']]
    if reason=='physical_time_unverified':
        from .physical import compute
        value,frames,error,diagnostic=compute(metric,series,observation,masks)
        out.update(value=None if value is None else round(value,5),usedFrames=frames,validFrames=len(frames),sensitivity=diagnostic)
        if error:
            out.update(status=error,missingReason=error);out['missingRequirements']=sorted(set(out['missingRequirements']+[error]));return out
        auto=any(s!='agent_visual_source_bracket' for s in sources.values());span=diagnostic.get('range')
        sensitive=(span is not None and np.ptp(span)>max(abs(value)*.15,10)) or ('peak' in metric['id'] and diagnostic.get('physicalImageMedianIntervalMs',0)>1000/120)
        out.update(status='event_candidate' if auto else 'sensitive_3d' if sensitive else 'provisional_3d',stability='candidate_event_unaccepted' if auto else 'sensitive' if sensitive else 'within_tested_settings',stabilityFlags=['filter_or_native_cadence_sensitive'] if sensitive else [],eventAcceptance='unaccepted_automatic_candidate' if auto else 'source_reviewed_provisional_not_human_adjudicated')
        return out
    if reason:
        out.update(status=reason,missingReason=reason);out['missingRequirements']=sorted(set(out['missingRequirements']+[reason]));return out
    value,frames,reason=evaluate(metric,series,events,unique,times,masks)
    out.update(value=None if value is None else round(value,5),usedFrames=frames,validFrames=len(frames),frameRange=[min(frames),max(frames)] if frames else None)
    if reason:out.update(status=reason,missingReason=reason);out['missingRequirements']=sorted(set(out['missingRequirements']+[reason]));return out
    # Event uncertainty: actual native images at both interval endpoints and midpoint.
    choices={e:sorted(set([events[e]['interval'][0],int(np.floor(np.mean(events[e]['interval'])+.5)),events[e]['interval'][1]])) for e in needed}
    sensitivity=[]
    for point in itertools.product(*choices.values()):
        result,_,_=evaluate(metric,series,events,unique,times,masks,dict(zip(choices,point)))
        if result is not None:sensitivity.append(result)
    smoothing=[]
    # Diagnostic only: smooth at original independent-image index, never fill observation gaps.
    for sigma in [.5,1.]:
        trial={k:v.copy() for k,v in series.items()}
        for k,v in trial.items():
            mask=masks.get(k,np.ones(len(v),bool));indices=np.asarray([i for i in unique if i<len(v) and mask[i] and np.isfinite(v[i])])
            # Split gaps so smoothing cannot bridge unobserved intervals.
            for segment in np.split(indices,np.where(np.diff(indices)>2)[0]+1):
                if len(segment)>=3:
                    data=v[segment];data=unwrap_local(data) if k in ['thorax_pelvis_relative_axial','shoulder_line_separation','pelvis_rotation','trunk_rotation'] else data
                    v[segment]=gaussian_filter1d(data,sigma,mode='nearest')
        result,_,_=evaluate(metric,trial,events,unique,times,masks)
        if result is not None:smoothing.append(result)
    alternative_values={}
    for label,trial in (alternatives or {}).items():
        result,_,_=evaluate(metric,trial,events,unique,times,masks)
        if result is not None:alternative_values[label]=round(result,5)
    axis_values=[]
    for trial in axis_variants or []:
        result,_,_=evaluate(metric,trial,events,unique,times,masks)
        if result is not None:axis_values.append(result)
    def span(vals):return [round(min(vals),5),round(max(vals),5)] if vals else None
    out['sensitivity']=dict(eventFrames=span(sensitivity),smoothing=span(smoothing),alternativeModels=alternative_values,pitchAxis=span(axis_values))
    event_span=np.ptp(sensitivity) if sensitivity else 0;filter_delta=max([abs(x-value) for x in smoothing] or [0]);model_delta=max([abs(x-value) for x in alternative_values.values()] or [0]);axis_delta=max([abs(x-value) for x in axis_values] or [0])
    limit=15 if metric['unit']=='°' else 20
    flags=[]
    if event_span>limit:flags.append('event_sensitive')
    if filter_delta>limit:flags.append('filter_sensitive')
    if model_delta>limit:flags.append('model_sensitive')
    if axis_delta>limit:flags.append('axis_sensitive')
    auto=any(s!='agent_visual_source_bracket' for s in sources.values())
    out.update(status='event_candidate' if auto else 'sensitive_3d' if flags else 'provisional_3d',
               stability='candidate_event_unaccepted' if auto else 'sensitive' if flags else 'within_tested_settings',stabilityFlags=flags,
               eventAcceptance='unaccepted_automatic_candidate' if auto else 'source_reviewed_provisional_not_human_adjudicated')
    return out


def make_series(a,rot,camera,R,hand,observation,feet=None):
    axis=observation.get('pitchAxisCamera');axis=R@np.asarray(axis) if axis is not None else None
    s,cal=segment_series(a,rot,camera,R,hand,axis)
    calibration=observation.get('metricCalibration',{})
    scale=calibration.get('modelUnitToMeters')
    if calibration.get('metricScaleVerified') and calibration.get('scaleEvidence') and scale and scale>0:
        s['stride_length_cm']=100*scale*np.linalg.norm(a[:,7]-a[:,8],axis=-1)
        height=calibration.get('measuredStatureCm')
        if height and height>0 and calibration.get('statureEvidence'):s['stride_height_pct']=100*s['stride_length_cm']/height
    if axis is not None:
        B=pitch_frame(axis);lead,trail=(7,8) if hand=='R' else (8,7);lh,lk=(1,4) if hand=='R' else (2,5)
        leg=np.nanmedian(np.linalg.norm(a[:,lh]-a[:,lk],axis=-1)+np.linalg.norm(a[:,lk]-a[:,lead],axis=-1));factor=1 if hand=='R' else -1
        s['foot_lateral']=100*factor*((a[:,lead]-a[:,trail])@B[:,1])/leg
        if feet is not None:
            big,small,heel=(0,2,4) if hand=='R' else (1,3,5)
            v=(feet[:,big]+feet[:,small])*.5-feet[:,heel]
            s['foot_direction']=factor*signed_horizontal(np.broadcast_to(B[:,0],v.shape),v)
    return s,cal


def sequence_diagnostic(series,observation,masks):
    """Peak-order sensitivity in playback image index, not physical angular velocity."""
    fc=observation['events'].get('fc',{}).get('interval');br=observation['events'].get('br',{}).get('interval')
    if not fc or not br:return dict(state='event_not_identified',physicalSequencePublished=False)
    unique=np.asarray(observation['cadence']['uniqueImageCandidates']);u=unique[(unique>=fc[0])&(unique<=br[1])]
    if len(u)<5:return dict(state='insufficient_independent_images',physicalSequencePublished=False)
    runs=[]
    keys=[k for k in ['lead_knee_flexion','pelvis_rotation','trunk_rotation','elbow_flexion'] if k in series]
    for sigma in [0,.5,1.]:
        peaks={}
        for k in keys:
            if not masks.get(k,np.ones(len(series[k]),bool))[u].all():continue
            vals=unwrap_local(series[k][u]);vals=gaussian_filter1d(vals,sigma,mode='nearest') if sigma else vals
            vel=np.gradient(vals,u)*(-1 if 'flexion' in k else 1);peaks[k]=int(u[np.argmax(vel)])
        runs.append(dict(sigmaIndependentImages=sigma,peakFrames=peaks))
    ranges={k:[min(x['peakFrames'][k] for x in runs),max(x['peakFrames'][k] for x in runs)] for k in keys if all(k in x['peakFrames'] for x in runs)}
    pairs={}
    for a,b in itertools.combinations(ranges,2):
        x,y=ranges[a],ranges[b];pairs[a+' → '+b]='before' if x[1]+1<y[0] else 'after' if y[1]+1<x[0] else 'simultaneous_or_uncertain'
    return dict(state='diagnostic_only',timeBasis='playback original image indices; no physical angular velocity or efficiency claim',runs=runs,peakFrameRanges=ranges,pairOrder=pairs,physicalSequencePublished=False)
