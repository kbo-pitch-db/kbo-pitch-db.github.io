"""Additional tests of an imposed stance anchor; these do not measure 3D accuracy."""
import numpy as np
from pitch_quality import screen as screen_base


def stance_metrics(z,fit):
    phase=fit.get('observed_phase_hints',{});plant=phase.get('foot_plant_frame_hint');lift=phase.get('leg_lift_frame_hint')
    if plant is None or lift is None:return dict(stance_links=0,stance_midpoint_speed_max_model_m_s=None,stance_midpoint_height_max_model_m=None)
    feet=z['foot_keypoints'];times=z['source_time'];fps=float(z['fps']);count=len(feet)
    lead=[0,2] if phase.get('lead_side_hint')=='left' else [1,3];rear=[1,3] if lead==[0,2] else [0,2]
    margin=max(3,int(np.ceil(fps*.06)));end=min(count,int(fit.get('anchor_end_frame',count-1)))
    speeds=[];heights=[]
    for indices,a,b in [(rear,int(lift)+margin,int(plant)-margin),(lead,int(plant)+margin,end)]:
        toe=feet[a:b,indices].mean(1);dt=np.diff(times[a:b]);frames=z['frame_index'][a:b]
        valid=np.isfinite(toe).all(1);links=valid[1:]&valid[:-1]&(np.diff(frames)==1)&(dt>0)&(dt<.1)
        speeds.extend((np.linalg.norm(np.diff(toe[:,[0,2]],axis=0),axis=1)/np.maximum(dt,1e-6))[links].tolist())
        heights.extend(np.abs(toe[valid,1]-.005).tolist())
    return dict(stance_links=len(speeds),stance_midpoint_speed_max_model_m_s=max(speeds) if speeds else None,
        stance_midpoint_height_max_model_m=max(heights) if heights else None)


def screen(z,fit):
    q=screen_base(z,fit);m=stance_metrics(z,fit);q['metrics'].update(m)
    if m['stance_links']<4:q['reasons'].append('insufficient_anchored_stance')
    elif m['stance_midpoint_speed_max_model_m_s']>.02 or m['stance_midpoint_height_max_model_m']>.01:
        q['reasons'].append('stance_anchor_inconsistent')
    upstream=fit.get('seed_camera_sensitivity',{}).get('stride_at_plant_leg_range')
    if upstream is not None and upstream[1]-upstream[0]>.35 and 'depth_initialization_sensitivity' not in q['reasons']:
        q['reasons'].append('depth_initialization_sensitivity')
    q['passed']=not q['reasons']
    q['interpretation']+=' Exact forefoot stance is imposed, not observed 3D contact. Anchor and camera initialization sensitivity can veto publication.'
    return q
