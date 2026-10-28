"""Physical-time calculations run only with an explicit independently verified clock.

Native independent images are retained. Local polynomial filtering/derivatives
never upsample frames. 5/7/9-image sensitivity is reported; this differs from
13.4/20 Hz laboratory Butterworth pipelines and cannot recover missed peaks.
"""
import numpy as np
from scipy.signal import savgol_filter

SERIES={'lead_knee':'lead_knee_flexion','pelvis':'pelvis_rotation','trunk':'trunk_rotation','elbow':'elbow_flexion'}


def compute(metric,series,observation,masks):
    cadence=observation['cadence'];factor=cadence.get('physicalSecondsPerPlaybackSecond')
    if cadence.get('physicalTimeStatus')!='verified' or not factor or factor<=0:return None,[], 'physical_time_unverified',None
    if not cadence.get('physicalClockEvidence'):return None,[], 'physical_time_unverified',None
    events=observation['events'];needed=['fc','br']
    if any(not events.get(e,{}).get('interval') for e in needed):return None,[],'event_not_identified',None
    times=np.asarray(observation['frameTimes'],float)*factor;unique=np.asarray(cadence['uniqueImageCandidates'],int)
    fc=int(unique[np.argmin(abs(unique-int(np.floor(np.mean(events['fc']['interval'])+.5))))]);br=int(unique[np.argmin(abs(unique-int(np.floor(np.mean(events['br']['interval'])+.5))))])
    if br<=fc or times[br]<=times[fc]:return None,[],'event_order_ambiguous',None
    id=metric['id']
    if id=='fc_br_time_ms':
        ranges=[1000*(times[b]-times[a]) for a in events['fc']['interval'] for b in events['br']['interval'] if b>a]
        return 1000*(times[br]-times[fc]),[fc,br],None,dict(eventTimeRangeMs=[min(ranges),max(ranges)],physicalClockEvidence=cadence['physicalClockEvidence'])
    required=['pelvis','trunk'] if id=='pelvis_trunk_peak_gap_ms' else [next((p for p in SERIES if id.startswith(p+'_')),None)]
    if required==[None]:return None,[],'implementation_incomplete',None
    estimates={};used=set();sampling=[]
    for segment in required:
        key=SERIES[segment]
        if key not in series:return None,[],'pitch_axis_unavailable',None
        good=np.asarray(masks.get(key,np.ones(len(times),bool)))&np.isfinite(series[key]);indices=unique[good[unique]]
        # A complete observed segment containing both events is mandatory.
        runs=np.split(indices,np.where(np.diff(indices)>2)[0]+1);indices=next((r for r in runs if len(r)>=7 and fc in r and br in r),None)
        if indices is None:return None,[],'insufficient_native_samples',None
        dt=np.diff(times[indices]);med=float(np.median(dt));sampling.append(med)
        if med<=0 or np.max(abs(dt-med))>.1*med:return None,indices.tolist(),'irregular_physical_sampling',None
        vals=np.asarray(series[key])[indices];vals=np.degrees(np.unwrap(np.radians(vals))) if segment in ['pelvis','trunk'] else vals;sign=-1 if 'flexion' in key else 1
        window=(indices>=fc)&(indices<=br)
        if window.sum()<3:return None,indices.tolist(),'insufficient_native_samples',None
        peak=[]
        for width in [5,7,9]:
            if width>len(indices):continue
            smoothed=savgol_filter(vals,width,3,mode='interp');velocity=sign*np.gradient(smoothed,times[indices],edge_order=2);at=np.where(window)[0][np.argmax(velocity[window])]
            peak.append(dict(width=width,velocity=float(velocity[at]),timeMs=float(1000*(times[indices[at]]-times[fc])),frame=int(indices[at])))
        estimates[segment]=peak;used.update(indices[window].tolist())
    if len(required)==2:
        pairs=[(x,y) for x in estimates['pelvis'] for y in estimates['trunk']];results=[y['timeMs']-x['timeMs'] for x,y in pairs];nominal=estimates['trunk'][len(estimates['trunk'])//2]['timeMs']-estimates['pelvis'][len(estimates['pelvis'])//2]['timeMs']
    else:
        k='velocity' if id.endswith('peak_velocity') else 'timeMs';a=estimates[required[0]];results=[x[k] for x in a];nominal=a[len(a)//2][k]
    diagnostic=dict(filter='Savitzky–Golay polynomial 3; native-image windows 5/7/9; np.gradient physical seconds',filterEstimates=estimates,range=[min(results),max(results)],physicalImageMedianIntervalMs=1000*max(sampling),physicalClockEvidence=cadence['physicalClockEvidence'],peakOrder='simultaneous_or_uncertain' if len(required)==2 and min(results)<=0<=max(results) else 'resolved_within_tested_settings',temporalLimitation='Native sampling can underestimate fast peaks; interpolation cannot restore them.')
    return float(nominal),sorted(used),None,diagnostic
