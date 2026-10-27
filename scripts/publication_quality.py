"""Independent, conservative publication review of pitch-event order and depth.

This is a post-fit veto. It never changes coordinates or treats a wrist-speed proxy
as an annotated ball-release event.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def screen_publication(motion,quality,halpe):
    reasons=[];phase=quality['screening']['phase_hints'];plant=phase.get('foot_plant_frame_hint');lift=phase.get('leg_lift_frame_hint')
    xy=halpe['keypoints'];scores=halpe['scores'];fps=float(halpe['fps']);count=len(xy)
    height=max(100.,float(np.nanmedian(halpe['bbox'][:,3]-halpe['bbox'][:,1])))
    arm_peaks=[];strengths=[]
    for side in [0,1]:
        wrist,shoulder=9+side,5+side
        good=(scores[:,wrist]>.4)&(scores[:,shoulder]>.4)&np.isfinite(xy[:,[wrist,shoulder]]).all((1,2))
        peaks=[];speed=np.zeros(count)
        if good.sum()>=4:
            relative=(xy[:,wrist]-xy[:,shoulder])/height
            for axis in range(2):relative[:,axis]=np.interp(np.arange(count),np.flatnonzero(good),relative[good,axis])
            relative=gaussian_filter1d(relative,max(1.,fps*.035),axis=0)
            speed=np.linalg.norm(np.gradient(relative,axis=0),axis=-1)*fps
            indices,_=find_peaks(speed,height=1.5,prominence=.5,distance=max(2,int(fps*.1)))
            peaks=[int(i) for i in indices if good[max(0,i-1):min(count,i+2)].all()]
            if speed[0]>1.5 and speed[0]>=speed[1] and good[:2].all():peaks.insert(0,0)
        arm_peaks.append([dict(frame=i,speed_height_per_second=float(speed[i])) for i in peaks])
        strengths.append(max([speed[i] for i in peaks],default=0.))
    dominant=int(np.argmax(strengths));event=arm_peaks[dominant][0]['frame'] if arm_peaks[dominant] else None
    # Require several observed preparation frames; a follow-through-only clip can
    # otherwise have its returning rear leg mislabeled as a fresh leg lift.
    hip=xy[:,[11,12]].mean(1);shoulder=xy[:,[5,6]].mean(1)
    prepared=(hip[:,1]-shoulder[:,1]>.1*height)&(xy[:,[9,10],1].max(1)<hip[:,1]+.06*height)
    prepared&=(scores[:,[5,6,9,10,11,12]].min(1)>.35)&np.isfinite(xy[:,[5,6,9,10,11,12]]).all((1,2))
    preparation_count=0
    if lift is not None:
        stop=min(count,max(int(lift)+1,max(3,int(fps*.12))))
        run=0
        for flag in prepared[:stop]:run=run+1 if flag else 0;preparation_count=max(preparation_count,run)
    if preparation_count<max(2,int(round(fps*.10))):reasons.append('pitch_preparation_unresolved')
    if event is None:reasons.append('arm_phase_unresolved')
    elif lift is None or plant is None or lift>event or plant>event+max(2,int(fps*.15)):
        reasons.append('pitch_event_order_review')
    forward=None
    if plant is not None:
        p=motion['anatomical_keypoints'];lead=7 if phase.get('lead_side_hint')=='left' else 8;rear=15-lead
        vector=p[plant,lead,[0,2]]-p[plant,rear,[0,2]];length=float(np.linalg.norm(vector))
        forward=float(-vector[1]/max(length,1e-8))
        if forward<.5:reasons.append('stride_direction_depth_review')
    return dict(rules_version='pitch-publication-review-v1',rules_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),passed=not reasons,reasons=reasons,arm_motion_side_hint=['left','right'][dominant],
        first_reliable_fast_arm_peak_frame=event,arm_peaks=arm_peaks,preparation_consecutive_frames=preparation_count,
        forward_stride_fraction=forward,
        interpretation='Conservative review only. Fast wrist motion is an image-space proxy, not measured ball release. Approximate home direction is away from the broadcast camera; uncertain preparation, phase order or mostly lateral stride is withheld.')


def screen_record(dataset,base,motion,quality):
    if not quality['fit_accepted']:
        return dict(passed=False,reasons=[],status='not_run_after_numerical_rejection')
    with np.load(dataset/base['pose_2d_refined'],allow_pickle=False) as z:halpe={k:z[k] for k in z.files}
    return screen_publication(motion,quality,halpe)
