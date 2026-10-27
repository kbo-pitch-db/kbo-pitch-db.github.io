"""Conservative pitch phase hypotheses from stabilized Halpe-26 observations only."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def infer_pitch_contacts(xy,scores,height,fps):
    count=len(xy);confidence=np.nanmean(scores[:,[13,14,15,16]],1)
    difference=gaussian_filter1d(np.nan_to_num(xy[:,14,1]-xy[:,13,1])/height,max(.7,fps*.025))
    magnitude=np.abs(difference)
    peaks,_=find_peaks(magnitude,prominence=.035,distance=max(3,int(fps*.15)))
    candidates=[int(i) for i in peaks if magnitude[i]>.10 and i<count*.75 and confidence[i]>.4]
    if magnitude[0]>.10 and confidence[0]>.4:candidates.insert(0,0)
    support=np.zeros((count,6),np.float32)
    if not candidates:return support,{'status':'ambiguous','reason':'No clearly observed initial leg lift'}
    lift=candidates[0];side=0 if difference[lift]>0 else 1;peak=magnitude[lift]
    toeids=[20,22] if side==0 else [21,23]
    markers=xy[:,toeids].copy()/height
    marker_valid=(scores[:,toeids]>.2)&np.isfinite(markers).all(-1)
    observed=marker_valid.any(1)
    if observed.sum()<4:return support,{'status':'ambiguous','reason':'Insufficient observed landing-foot markers'}
    # Keep the same toe midpoint when confidences change; confidence-weighted
    # marker locations would themselves create apparent foot motion.
    for marker in range(2):
        good=marker_valid[:,marker]
        if good.sum()<2:
            other=1-marker;good=marker_valid[:,other]
            if good.sum()<2:return support,{'status':'ambiguous','reason':'Insufficient toe trajectory'}
            values=markers[:,other]
        else:values=markers[:,marker]
        for axis in range(2):markers[:,marker,axis]=np.interp(np.arange(count),np.flatnonzero(good),values[good,axis])
    toes=gaussian_filter1d(markers.mean(1),max(1.,fps*.04),axis=0)
    speed=np.linalg.norm(np.gradient(toes,axis=0),axis=-1)*fps
    plant=None;stable=None
    window=max(3,int(round(fps*.12)))
    for i in range(lift+max(2,int(fps*.06)),count-window):
        lowered=(difference[i] if side==0 else -difference[i])<max(.05,peak*.3)
        # A velocity minimum can be the airborne foot reversing direction. Require
        # sustained observed stability, then locate the preceding landing deceleration.
        segment=toes[i:i+window]
        steady=np.median(speed[i:i+window])<.20 and np.ptp(segment,axis=0).max()<.018
        if not (lowered and steady and observed[i:i+window].mean()>.8 and np.nanmean(scores[i:i+window,toeids])>.55):continue
        a=max(lift+1,i-max(4,int(fps*.50)))
        peaks2,_=find_peaks(speed[a:i+1],height=.50,prominence=.15)
        peaks2=np.array([k for k in peaks2 if observed[max(0,a+k-1):min(count,a+k+2)].all()],int)
        if not len(peaks2):continue
        last_peak=a+int(peaks2[-1])
        if speed[last_peak]<.35 or i<=last_peak+1:continue
        deceleration=np.gradient(speed)
        plant=last_peak+1+int(np.argmin(deceleration[last_peak+1:i+1]))
        if not observed[max(0,plant-1):min(count,plant+2)].all():
            plant=None;continue
        stable=i;break
    if plant is None:return support,{'status':'ambiguous','reason':'No confidently observed landing after first leg lift','lead_side_hint':['left','right'][side],'leg_lift_frame_hint':lift}
    lead=[0,2] if side==0 else [1,3];rear=[1,3] if side==0 else [0,2]
    # Rear forefoot through drive; lead forefoot after plant. Never pin both heels.
    support[:plant+1,rear]=1.
    ramp=np.minimum(1.,(np.arange(count-plant)+1)/max(2,int(fps*.05)))
    support[plant:,lead]=(np.exp(-(speed[plant:]/.65)**2)*ramp)[:,None]
    support*=np.clip(scores[:,20:26],0,1)**2
    return support,dict(status='observed phase hypothesis',lead_side_hint=['left','right'][side],leg_lift_frame_hint=lift,foot_plant_frame_hint=plant,stable_landing_evidence_frame=stable,velocity_interpolated_frames=int((~observed).sum()),foot_plant_frame_interval=[max(lift+1,plant-max(1,int(fps*.035))),min(stable,plant+max(2,int(fps*.05)))],
        interpretation='Automatic 2D event/contact hypotheses; not manual pitch labels or contact probabilities')
