"""Contact-constrained SMPL fitting initialized by WHAM, with one fixed camera/floor.

The stance model assumes at least one supporting foot. Genuine airborne follow-through,
ambiguous contact and camera cuts require review. No target stride length is imposed.
"""
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation,Slerp
from scipy.ndimage import gaussian_filter1d
from scene_fit import SceneFitter
from pitch_contacts import infer_pitch_contacts
from scene_geometry import support_normal,initial_camera,fill_pose_gaps,SMPL_JOINTS,SMPL_PARENTS,FOOT_NAMES,transform_scene

VERSION='wham-pitch-anchor-v3'


def spline_basis(count,fps,spacing=.10):
    n=max(4,int(np.ceil((count-1)/fps/spacing))+1)
    knots=np.linspace(0,count-1,n)
    return make_interp_spline(knots,np.eye(n),k=3)(np.arange(count)).astype(np.float32)


def repair_rotation_spikes(rotations,valid,fps):
    """Repair only isolated return spikes, on SO(3); fast sustained motion is retained."""
    out=rotations.copy();changed=np.zeros(rotations.shape[:2],bool)
    # An isolated reversal in the pelvis/legs is distinct from a fast throwing arm.
    for j in range(24):
        limit=.50 if j<16 else .95
        for i in range(1,len(out)-1):
            if not valid[i-1:i+2].all():continue
            a,b,c=Rotation.from_matrix(out[[i-1,i,i+1],j])
            ab=(a.inv()*b).magnitude();bc=(b.inv()*c).magnitude();ac=(a.inv()*c).magnitude()
            if min(ab,bc)>limit*(30/fps)**.35 and ac<min(ab,bc)*.4:
                out[i,j]=Slerp([0,1],Rotation.from_matrix(out[[i-1,i+1],j]))([.5]).as_matrix()[0];changed[i,j]=True
    return out,changed


def stable_rotation(vector):
    import torch
    theta=torch.linalg.vector_norm(vector,dim=-1,keepdim=True)
    q=torch.cat([torch.cos(theta*.5),vector*.5*torch.sinc(theta/(2*np.pi))],-1)
    r,i,j,k=q.unbind(-1);two=2./q.square().sum(-1)
    return torch.stack([1-two*(j*j+k*k),two*(i*j-k*r),two*(i*k+j*r),
        two*(i*j+k*r),1-two*(i*i+k*k),two*(j*k-i*r),
        two*(i*k-j*r),two*(j*k+i*r),1-two*(i*i+j*j)],-1).reshape(*vector.shape[:-1],3,3)


def static_rigid_transform(rot,joints,parents):
    import torch
    from torch.nn import functional as F
    from smplx.lbs import transform_mat
    joints=joints.unsqueeze(-1);relative=joints.clone();relative[:,1:]=relative[:,1:]-joints[:,parents[1:]]
    local=transform_mat(rot.reshape(-1,3,3),relative.reshape(-1,3,1)).reshape(-1,24,4,4)
    chain=[local[:,0]]
    for i in range(1,24):chain.append(chain[SMPL_PARENTS[i]]@local[:,i])
    world=torch.stack(chain,1)
    A=world-F.pad(world@F.pad(joints,[0,0,0,1]),[3,0,0,0,0,0,0,0])
    return world[:,:,:3,3],A


class AnchorFitter(SceneFitter):
    def __init__(self,wham_root,device,use_cuda_graph=True):
        super().__init__(wham_root,device)
        self.rotation=stable_rotation
        self.use_cuda_graph=use_cuda_graph and str(device).startswith('cuda')
        torch=self.torch
        keep=(self.body.J_regressor_wham[:17].abs().sum(0)>0).nonzero().flatten()
        keep=torch.unique(torch.cat([keep,torch.as_tensor(self.foot_vertices,device=device)]),sorted=True)
        self.selected_vertices=keep
        self.selected_regressor=self.body.J_regressor_wham[:17,keep]
        self.selected_weights=self.body.lbs_weights[keep]
        self.selected_posedirs=self.body.posedirs.reshape(207,-1,3)[:,keep].reshape(207,-1)
        self.selected_feet=torch.as_tensor([int((keep==i).nonzero()[0]) for i in self.foot_vertices],device=device)

    def prepare_shape(self,beta):
        torch=self.torch
        with torch.no_grad():
            shaped=self.body.v_template+torch.einsum('vck,k->vc',self.body.shapedirs,torch.as_tensor(beta,dtype=torch.float32,device=self.device))
            self.rest_joints=self.body.J_regressor@shaped
            self.rest_vertices=shaped[self.selected_vertices]
            self.rest_stature=float(shaped[:,1].max()-shaped[:,1].min())

    def model(self,rotations,betas):
        # Exact SMPL LBS at the union of nonzero WHAM-regressor/foot vertices.
        # No mesh approximation: unobserved mesh vertices have zero contribution.
        torch=self.torch;count=len(rotations)
        feature=(rotations[:,1:]-torch.eye(3,device=self.device)).reshape(count,207)
        posed=self.rest_vertices[None]+(feature@self.selected_posedirs).reshape(count,-1,3)
        joints,A=static_rigid_transform(rotations,self.rest_joints[None].expand(count,-1,-1),self.body.parents)
        transforms=(self.selected_weights[None]@A.reshape(count,24,16)).reshape(count,-1,4,4)
        homogeneous=torch.cat([posed,torch.ones((*posed.shape[:2],1),device=self.device)],-1)
        vertices=(transforms@homogeneous[:,:,:,None])[:,:,:3,0]
        body=self.selected_regressor[None]@vertices;root=(body[:,11:12]+body[:,12:13])*.5
        return body-root,joints-root,vertices[:,self.selected_feet]-root

    def fit(self,prior,halpe,row,source,background,steps=180,initial=None,focal_factor=4.,pitch_degrees=4.,anchor_start=None):
        torch=self.torch;device=self.device
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        count=len(prior['keypoints']);fps=float(halpe['fps'])
        rotations=np.concatenate([prior['smpl_global_orient'],prior['smpl_body_pose']],1).copy()
        valid=np.isfinite(rotations).all((1,2,3))&np.isfinite(prior['keypoints']).all((1,2))
        observed=valid.copy()
        rotations,interpolated=fill_pose_gaps(rotations,valid,halpe['frame_index'],halpe.get('track_id'))
        valid|=interpolated
        if valid.sum()<8:raise ValueError('Insufficient supported WHAM frames')
        if initial is not None:
            initrot=np.concatenate([initial['smpl_global_orient'],initial['smpl_body_pose']],1)
            use=valid&np.isfinite(initrot).all((1,2,3));rotations[use]=initrot[use]
        rotations,repaired=repair_rotation_spikes(rotations,valid,fps)
        good=np.flatnonzero(valid)
        for i in np.flatnonzero(~valid):rotations[i]=rotations[good[np.argmin(abs(good-i))]]
        beta=np.nanmedian(prior['smpl_betas'][observed],axis=0)
        r0=tensor(rotations)
        self.prepare_shape(beta)
        with torch.no_grad():body0,anatomy0,feet0=self.model(r0,tensor(np.repeat(beta[None],count,axis=0)))
        leg=float(torch.linalg.norm(anatomy0[0,4]-anatomy0[0,1])+torch.linalg.norm(anatomy0[0,7]-anatomy0[0,4]))
        origin=np.array([float(row['crop_x1']),float(row['crop_y1'])])
        local=halpe['keypoints']-origin
        xy=np.einsum('tij,tkj->tki',background,np.concatenate([local,np.ones((*local.shape[:2],1))],-1))[:,:,:2]+origin
        confidence=halpe['scores'].copy();confidence[halpe['missing']|~np.isfinite(xy).all(-1)]=0;confidence[~observed]=0
        xy=np.nan_to_num(xy)
        height=max(100.,float(np.nanmedian(halpe['bbox'][:,3]-halpe['bbox'][:,1])))
        principal=tensor([float(source['width'])/2,float(source['height'])/2]);diagonal=np.hypot(float(source['width']),float(source['height']))
        focal=float(diagonal*focal_factor)
        scale,offset,trans=initial_camera(body0.cpu().numpy(),xy[:,:17],confidence[:,:17],focal,principal.cpu().numpy())
        n0=support_normal(body0.cpu().numpy())
        pitch0=float(np.arctan2(-initial['scene_rotation'][1,2],-initial['scene_rotation'][1,1])) if initial is not None else float(np.deg2rad(pitch_degrees))
        pitch=torch.nn.Parameter(tensor(pitch0));roll0=float(initial['scene_rotation'][1,0]) if initial is not None else 0.;roll=torch.nn.Parameter(tensor(roll0))
        unit_x=tensor([1.,0,0])
        def frame():
            # The up vector is global, with an anatomically plausible broadcast view range.
            p=pitch.clamp(np.deg2rad(-3),np.deg2rad(28));r=roll.clamp(-.10,.10)
            n=torch.stack([r,-torch.cos(p),-torch.sin(p)]);n=n/n.norm()
            x=unit_x;x=x-n*x.dot(n);x=x/x.norm();z=torch.linalg.cross(x,n)
            return torch.stack([x,n,z])
        R=frame().detach();fw=feet0@R.T
        minheight=(-.012*torch.logsumexp(-fw[:,:,1]/.012,dim=1)).cpu().numpy()
        median_t=np.nanmedian(trans[observed],axis=0)
        # Depth initialization is global. Per-frame weak-perspective depth is never reused.
        h0=-minheight+.005
        camera0=median_t-(np.c_[np.zeros(count),h0,np.zeros(count)]@R.cpu().numpy()).mean(0)
        if initial is not None:
            focal=float(initial['fitted_camera_focal'])
            camera0=initial['scene_origin'].copy()
        camera=torch.nn.Parameter(tensor(camera0));logf=torch.nn.Parameter(tensor(0.))
        basis=tensor(spline_basis(count,fps));knots=basis.shape[1]
        initial_planar=initial['root_translation'][:,[0,2]].copy() if initial is not None else np.c_[np.zeros(count),np.zeros(count)]
        for axis in range(2):
            finite=np.isfinite(initial_planar[:,axis]);initial_planar[:,axis]=np.interp(np.arange(count),np.flatnonzero(finite),initial_planar[finite,axis])
        controls=torch.nn.Parameter(tensor(np.linalg.lstsq(basis.cpu().numpy(),initial_planar,rcond=None)[0]))
        phase_contacts,phase=infer_pitch_contacts(xy,confidence,height,fps)
        plant=phase.get('foot_plant_frame_hint');lift=phase.get('leg_lift_frame_hint')
        lead_side=0 if phase.get('lead_side_hint')=='left' else 1
        lead_indices=[0,2,4] if lead_side==0 else [1,3,5]
        rear_indices=[1,3,5] if lead_side==0 else [0,2,4]
        if plant is not None and initial is not None:
            a=max(0,lift-int(fps*.15));b=max(a+1,plant)
            rear_initial=initial['foot_keypoints'][a:b][:,rear_indices[:2]].mean(1)
            rear_anchor=np.nanmedian(rear_initial,axis=0);rear_anchor[1]=0.
            lead_initial=np.nanmedian(initial['foot_keypoints'][plant:plant+3][:,lead_indices[:2]].mean(1),axis=0)
            anchor_initial=lead_initial-rear_anchor;anchor_initial[1]=0.
            if anchor_start is not None:anchor_initial[2]=-float(anchor_start)*leg
        else:
            rear_anchor=np.zeros(3,np.float32);anchor_initial=np.array([0.,0.,-leg])
        lead_anchor=torch.nn.Parameter(tensor(anchor_initial[[0,2]]))
        rear_anchor_t=tensor(rear_anchor)
        anchor_weight=np.zeros(count,np.float32);lead_weight=np.zeros(count,np.float32);transition_weight=np.zeros(count,np.float32)
        anchor_end=count-1
        if plant is not None:
            start=max(0,lift-int(fps*.15));ramp=max(2,int(fps*.08));time=np.arange(count)
            anchor_weight=np.clip((time-start+1)/ramp,0,1)
            lead_weight=np.clip((time-plant+max(1,fps*.025))/max(2,fps*.05),0,1)
            transition_weight=np.exp(-((time-plant)/max(1.,fps*.045))**2)
            ids=[20,22] if lead_side==0 else [21,23]
            toe=xy[:,ids].mean(1)/height
            stable=np.median(toe[plant+max(1,int(fps*.1)):min(count,plant+max(3,int(fps*.35)))],axis=0)
            velocity=np.linalg.norm(np.gradient(toe,axis=0),axis=-1)*fps
            for k in range(plant+max(5,int(fps*.5)),count-max(3,int(fps*.12))):
                stop=k+max(3,int(fps*.12))
                if (np.linalg.norm(toe[k:stop]-stable,axis=1)>.05).all() and np.median(velocity[k:stop])>.35 and confidence[k:stop,ids].mean()>.55:
                    anchor_end=k;break
            anchor_weight*=np.clip((anchor_end-time+ramp)/ramp,0,1)
        anchor_mask=tensor(anchor_weight);lead_mix=tensor(lead_weight);double_contact=tensor(transition_weight)
        delta=torch.nn.Parameter(torch.zeros((count,24,3),device=device))
        knee0=Rotation.from_matrix(rotations[:,[4,5]].reshape(-1,3,3)).as_rotvec().reshape(count,2,3)
        fraction=np.clip((knee0[:,:,0]+.08)/2.88,.01,.99)
        knee_flex=torch.nn.Parameter(tensor(np.log(fraction/(1-fraction))))
        knee_side=torch.nn.Parameter(tensor(np.arctanh(np.clip(knee0[:,:,1:]/.18,-.95,.95))))
        optimizer=torch.optim.Adam([{'params':[controls,camera,lead_anchor],'lr':.014},{'params':[delta,knee_flex,knee_side],'lr':.018},
            {'params':[pitch,roll,logf],'lr':.002}],capturable=str(device).startswith('cuda'))
        target=tensor(np.concatenate([xy[:,:17],xy[:,20:26]],1));weights=tensor(np.concatenate([confidence[:,:17]**2,confidence[:,20:26]**2*.8],1));weights[:,:5]*=.35
        observedfeet=tensor(confidence[:,20:26]**2)
        phase_support=tensor(phase_contacts)
        order_window=np.zeros(count,np.float32);drive_window=np.zeros(count-1,np.float32)
        plant=phase.get('foot_plant_frame_hint');lift=phase.get('leg_lift_frame_hint')
        lead_side=0 if phase.get('lead_side_hint')=='left' else 1
        if plant is not None:
            order_window[max(lift,plant-max(2,int(fps*.12))):min(count,plant+max(2,int(fps*.08)))]=1.
            drive_window[lift:plant]=1.
        order_mask=tensor(order_window);drive_mask=tensor(drive_window)
        vel2=np.linalg.norm(np.gradient(xy[:,20:26],axis=0),axis=-1)*fps/height
        stationary=tensor(np.exp(-(vel2/.4)**2))
        links=valid[:-1]&valid[1:]&(np.diff(halpe['frame_index'])==1)
        dt=np.diff(halpe['source_time']);links&=(dt>0)&(dt<.1)
        link=tensor(links);triple=tensor(links[:-1]&links[1:]);mask=tensor(valid)
        rate=tensor(np.maximum(dt,1/fps))
        vmax=tensor([13.,13.,13.,12.,22.,22.,12.,24.,24.,12.,24.,24.,15.,20.,20.,20.,40.,40.,45.,45.,50.,50.,50.,50.])
        beta_t=tensor(np.repeat(beta[None],count,axis=0))
        poseweight=tensor([.7,.5,.5,1.,.4,.4,1.,.3,.3,1.,.3,.3,1.,.7,.7,1.,.5,.5,.4,.4,.3,.3,.3,.3])

        def evaluate():
            rot=r0@self.rotation(1.5*torch.tanh(delta))
            # SMPL's local knee flexion axis is x. Keep a small axial/valgus allowance,
            # with a broad -4.6 to 160 degree flexion range; no target pitch angle.
            knee_vector=torch.cat([(-.08+2.88*torch.sigmoid(knee_flex))[:,:,None],.18*torch.tanh(knee_side)],-1)
            rot=torch.cat([rot[:,:4],self.rotation(knee_vector),rot[:,6:]],1)
            body,anatomy,feet=self.model(rot,beta_t)
            rotation=frame();feet_world=feet@rotation.T
            # A kinematic support equation couples root height to the articulated feet.
            # It is part of the fitted model, never a renderer shift or post-hoc floor move.
            root_height=.005+.012*torch.logsumexp(-feet_world[:,:,1]/.012,dim=1)
            planar=basis@controls;root_free=torch.stack([planar[:,0],root_height,planar[:,1]],-1)
            # Fixed forefoot-midpoint X/Z and support height are kinematic equations.
            # They are solved together with pose and camera, not applied to the render.
            rear_mid=(feet_world[:,rear_indices[0]]+feet_world[:,rear_indices[1]])*.5
            lead_mid=(feet_world[:,lead_indices[0]]+feet_world[:,lead_indices[1]])*.5
            rear_y=rear_mid[:,1]
            lead_y=lead_mid[:,1]
            root_rear=torch.stack([rear_anchor_t[0]-rear_mid[:,0],.005-rear_y,rear_anchor_t[2]-rear_mid[:,2]],-1)
            root_lead=torch.stack([rear_anchor_t[0]+lead_anchor[0]-lead_mid[:,0],.005-lead_y,rear_anchor_t[2]+lead_anchor[1]-lead_mid[:,2]],-1)
            root_contact=root_rear*(1-lead_mix[:,None])+root_lead*lead_mix[:,None]
            root_world=root_contact*anchor_mask[:,None]+root_free*(1-anchor_mask[:,None])
            closure=((root_rear-root_lead).square().sum(-1)/leg**2*double_contact).sum()/double_contact.sum().clamp_min(1.)
            translation=root_world@rotation+camera
            pts=torch.cat([body,feet],1)+translation[:,None]
            f=focal*torch.exp(logf.clamp(-.65,.65))
            projected=pts[:,:,:2]/pts[:,:,2:].clamp_min(1.)*f+principal
            residual=(projected-target)/height
            robust=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=.04)
            image=(robust.sum(-1)*weights).sum()/weights.sum().clamp_min(1.)/.025
            absolute_feet=feet_world+root_world[:,None]
            clearance=absolute_feet[:,:,1]
            penetration=(torch.relu(-clearance-.01).square()*mask[:,None]).sum()/(mask.sum()*6).clamp_min(1.)/leg**2
            # Soft contact is marker-specific: toes can support while heels lift.
            support=torch.exp(-((clearance-.015).clamp_min(0)/(.08*leg))**2)*observedfeet
            contact=torch.maximum(support*stationary,phase_support*support.clamp_min(1e-8).sqrt())
            phase_height=(clearance.square()*phase_support).sum()/phase_support.sum().clamp_min(1.)/leg**2
            cv=torch.minimum(contact[:-1],contact[1:])*link[:,None]
            foot_velocity=(absolute_feet[1:]-absolute_feet[:-1])/rate[:,None,None]/leg
            slide=(foot_velocity.square().sum(-1)*cv).sum()/cv.sum().clamp_min(1.)
            rv=(root_world[1:]-root_world[:-1])/rate[:,None]/leg
            root_speed=(torch.relu(rv.norm(dim=-1)-5.).square()*link).sum()/link.sum().clamp_min(1.)
            acceleration=((root_world[2:]+root_world[:-2]-2*root_world[1:-1])/leg)*(fps/30)**2
            root_acc=(acceleration.square().sum(-1)*triple).sum()/triple.sum().clamp_min(1.)
            # Rotation matrices avoid Euler/axis-angle wrap jumps. Joint-specific limits
            # retain much faster arm motion than pelvis/torso motion.
            speed=(rot[1:,:,:,:2]-rot[:-1,:,:,:2]).square().sum((-1,-2)).clamp_min(1e-12).sqrt()/rate[:,None]/1.41421356
            angular_excess=(torch.relu(speed-vmax).square()*link[:,None]).mean()
            jvel=(anatomy[1:]-anatomy[:-1])/rate[:,None,None]/leg
            lower_excess=(torch.relu(jvel[:,:12].norm(dim=-1)-14.).square()*link[:,None]).mean()
            angular_acc=(rot[2:,:,:,:2]+rot[:-2,:,:,:2]-2*rot[1:-1,:,:,:2])*(fps/30)**2
            pose_acc=(angular_acc.square().sum((-1,-2))*poseweight*triple[:,None]).mean()
            anatomy_world=anatomy@rotation.T
            lead_z=anatomy_world[:,7+lead_side,2];rear_z=anatomy_world[:,8-lead_side,2]
            pelvis_z=(anatomy_world[:,1,2]+anatomy_world[:,2,2])*.5
            ordering=(torch.relu(lead_z-pelvis_z).square()+torch.relu(pelvis_z-rear_z).square())/leg**2
            pitch_order=(ordering*order_mask).sum()/order_mask.sum().clamp_min(1.)
            reverse_drive=(torch.relu(rv[:,2]-.2).square()*drive_mask*link).sum()/(drive_mask*link).sum().clamp_min(1.)
            prior_loss=((rot-r0).square().sum((-1,-2))*.5*poseweight*mask[:,None]).sum()/(mask.sum()*24).clamp_min(1.)
            normal_loss=((pitch-pitch0)/.12)**2+((roll-roll0)/.06)**2
            total=image+.05*slide+10.*phase_height+.15*root_speed+12.*root_acc+.12*angular_excess+.2*lower_excess+2.*pose_acc+.06*prior_loss+.005*normal_loss+.01*logf.square()+3.*pitch_order+.08*reverse_drive+100.*closure+30.*penetration
            return total,dict(image=image,slide=slide,root_speed_excess=root_speed,root_acceleration=root_acc,angular_excess=angular_excess,pose_acceleration=pose_acc,pose_prior=prior_loss,phase_height=phase_height,lower_velocity_excess=lower_excess,pitch_order=pitch_order,reverse_drive=reverse_drive,anchor_closure=closure,penetration=penetration), (body,anatomy,feet,translation,rotation,projected,f,rot,contact)
        from pitch_optimizer import optimize
        parameters=[controls,camera,lead_anchor,delta,knee_flex,knee_side,pitch,roll,logf]
        bestloss,best=optimize(parameters,optimizer,evaluate,steps,device,self.use_cuda_graph)
        with torch.no_grad():
            for p,b in zip([controls,camera,lead_anchor,delta,knee_flex,knee_side,pitch,roll,logf],best):p.copy_(b)
            total,losses,data=evaluate()
        body,anatomy,feet,translation,rotation,projected,f,rot,contact=[v.detach().cpu().numpy() for v in data]
        # Choose one horizontal origin using the first supported foot. Camera/floor remain fixed.
        camera_origin=camera.detach().cpu().numpy();scene_feet=(feet+translation[:,None]-camera_origin)@rotation.T
        anchor=np.nanmedian(scene_feet[good[:min(8,len(good))]].reshape(-1,3),axis=0)
        anchor[1]=0.;scene_origin=camera_origin+anchor@rotation
        output={k:transform_scene(v,rotation,scene_origin) for k,v in [('keypoints',body+translation[:,None]),('anatomical_keypoints',anatomy+translation[:,None]),('foot_keypoints',feet+translation[:,None]),('root_translation',translation)]}
        for value in output.values():value[~valid]=np.nan
        rot[~valid]=np.nan;translation[~valid]=np.nan
        output.update(scene_rotation=rotation,scene_origin=scene_origin.astype(np.float32),contact_weights=contact.astype(np.float32),interpolated_frames=interpolated,
            smpl_global_orient=rot[:,:1],smpl_body_pose=rot[:,1:],camera_translation=translation,camera_principal=principal.cpu().numpy(),
            smpl_betas=beta,fitted_camera_focal=np.array(f),anatomical_joints=np.array(SMPL_JOINTS),anatomical_parents=np.array(SMPL_PARENTS),foot_joints=np.array(FOOT_NAMES))
        error=np.linalg.norm(projected-target.cpu().numpy(),axis=-1)/height
        high=confidence[:,:17]>.5
        diagnostics=dict(anchor_end_frame=int(anchor_end),anchor_start_leg=anchor_start,lead_anchor_xz=lead_anchor.detach().cpu().tolist(),modeled_stature_model_m=self.rest_stature,optimizer_backend='cuda_graph' if self.use_cuda_graph else 'eager',steps=steps,fit_accepted=False,method=VERSION,objective=bestloss,
            losses={k:float(v.detach()) for k,v in losses.items()},focal_factor=focal_factor,pitch_initial_degrees=pitch_degrees,observed_phase_hints=phase,
            median_reprojection_height=float(np.median(error[:,:17][high])),p95_reprojection_height=float(np.percentile(error[:,:17][high],95)),
            median_reprojection_px=float(np.median(error[:,:17][high])*height),person_height_px=height,
            lower_body_reprojection_height_p95=float(np.percentile(error[:,11:17][high[:,11:17]],95)) if high[:,11:17].any() else 1.,
            feet_reprojection_height_p95=float(np.percentile(error[:,17:][confidence[:,20:26]>.5],95)) if (confidence[:,20:26]>.5).any() else 1.,
            anchor_policy='fixed rear/lead forefoot midpoint XZ during observed stance, with fitted transition; explicit tail release',
            coordinate_system='clip-fixed inferred support frame; model-estimated meters; y normal to inferred local floor',
            supported_frames=int(valid.sum()),total_frames=count,interpolated_pose_frames=int(interpolated.sum()),isolated_rotation_spikes_repaired=int(repaired.sum()),
            camera_pitch_degrees=float(pitch.detach().cpu())*180/np.pi,
            limitations='Uncalibrated WHAM-initialized monocular SMPL fit. Constant within-clip shape; assumed local support, including follow-through. Genuine airborne motion, camera cuts and ambiguous depth require review. No measured player height or stride ground truth; no OpenSim IK.',
            floor_reference_lower_envelope_shift_model_m=0.,scene_rotation=rotation.tolist(),scene_origin_camera=scene_origin.tolist())
        return output,diagnostics
