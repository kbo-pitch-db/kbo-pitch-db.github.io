"""Conservative numerical screening of inferred pitching motion, not accuracy validation."""
import numpy as np


def motion_metrics(z):
    p=z['anatomical_keypoints'];foot=z['foot_keypoints'];root=z['root_translation'];t=z['source_time'];fps=float(z['fps'])
    valid=np.isfinite(p).all((1,2));dt=np.diff(t);links=valid[1:]&valid[:-1]&(np.diff(z['frame_index'])==1)&(dt>0)&(dt<.1)
    lengths=np.linalg.norm(p[:,[4,5,7,8]]-p[:,[1,2,4,5]],axis=-1)
    leg=float(np.nanmedian(lengths.sum(-1)/2))
    def pct(v,q=95):
        v=np.asarray(v);v=v[np.isfinite(v)];return float(np.percentile(v,q)) if len(v) else None
    velocity=np.diff(root,axis=0)/np.maximum(dt[:,None],1e-6)/leg
    relative=p-root[:,None];jv=np.linalg.norm(np.diff(relative,axis=0),axis=-1)/np.maximum(dt[:,None],1e-6)/leg
    floor=np.min(np.nan_to_num(foot[:,:,1],nan=np.inf),axis=1)/leg
    bones=np.linalg.norm(p[:,1:]-p[:,z['anatomical_parents'][1:]],axis=-1)
    contact=z['contact_weights'];cl=np.minimum(contact[:-1],contact[1:])>.25;cl&=links[:,None]
    fv=np.linalg.norm(np.diff(foot,axis=0),axis=-1)/np.maximum(dt[:,None],1e-6)/leg
    lower=[0,1,2,3,4,5,6,7,8,9,10,11]
    metrics=dict(leg_model_m=leg,supported_ratio=float(valid.mean()),root_speed_max_leg_s=pct(np.linalg.norm(velocity[links],axis=-1),100),
        root_speed_p95_leg_s=pct(np.linalg.norm(velocity[links],axis=-1)),root_step_max_leg=pct(np.linalg.norm(np.diff(root,axis=0)[links],axis=-1)/leg,100),
        lower_body_relative_speed_max_leg_s=pct(jv[links][:,lower],100),joint_relative_speed_max_leg_s=pct(jv[links],100),
        foot_clearance_p95_leg=pct(floor[valid]),floating_ratio=float(np.mean(floor[valid]>.15)),
        minimum_foot_height_leg=pct(foot[:,:,1]/leg,0),contact_slide_p95_leg_s=pct(fv[cl]),contact_link_count=int(cl.sum()),
        native_bone_length_std_model_m=float(np.nanstd(bones,axis=0).max()),
        peak_interankle_separation_leg=pct(np.linalg.norm(p[:,7]-p[:,8],axis=-1)/leg,100))
    phase={}
    return metrics,phase


def screen(z,fit):
    metrics,phase=motion_metrics(z);reasons=[]
    phase=fit.get('observed_phase_hints',{'status':'ambiguous'})
    plant=phase.get('foot_plant_frame_hint')
    phase=dict(phase)
    if plant is not None:
        p=z['anatomical_keypoints'];phase['stride_at_plant_leg']=float(np.linalg.norm(p[plant,7,[0,2]]-p[plant,8,[0,2]])/metrics['leg_model_m'])
        lead=7 if phase.get('lead_side_hint')=='left' else 8;rear=15-lead
        hip=float(p[plant,[1,2],2].mean());leg=metrics['leg_model_m']
        phase['plant_order_violation_leg']=float(max(p[plant,lead,2]-hip,hip-p[plant,rear,2],0)/leg)
        if phase['plant_order_violation_leg']>.15:reasons.append('implausible_plant_order')
    phase['stride_validation']='Uncalibrated depth; not a measured stride'
    def exceeds(name,limit):
        value=metrics[name]
        if value is None or value>limit:reasons.append(name)
    if metrics['supported_ratio']<.95:reasons.append('insufficient_supported_frames')
    exceeds('root_speed_max_leg_s',7.)
    exceeds('lower_body_relative_speed_max_leg_s',23.)
    exceeds('joint_relative_speed_max_leg_s',60.)
    exceeds('floating_ratio',.03)
    exceeds('native_bone_length_std_model_m',5e-5)
    if metrics['minimum_foot_height_leg']<-.03:reasons.append('floor_penetration')
    if metrics['contact_slide_p95_leg_s'] is None or metrics['contact_slide_p95_leg_s']>2.5:reasons.append('unstable_support')
    if fit.get('median_reprojection_height',1)>.04 or fit.get('p95_reprojection_height',1)>.12 or fit.get('lower_body_reprojection_height_p95',1)>.10 or fit.get('feet_reprojection_height_p95',1)>.10:reasons.append('poor_2d_alignment')
    if plant is None:reasons.append('pitch_contact_phase_ambiguous')
    if 'camera_candidates' in fit and plant is not None and phase.get('stride_at_plant_leg',0)<.8:
        reasons.append('short_stride_depth_review')
    depth_range=fit.get('camera_sensitivity',{}).get('stride_at_plant_leg_range')
    if depth_range is not None and depth_range[1]-depth_range[0]>.35:reasons.append('depth_initialization_sensitivity')
    return dict(passed=not reasons,reasons=reasons,metrics=metrics,phase_hints=phase,
        interpretation='Screening only; passing does not validate 3D accuracy, depth, player height or stride. Support is a modeling constraint, not independent evidence. Plant separation below 0.8 modeled leg lengths triggers conservative depth review; it is not a universal pitching limit.')
