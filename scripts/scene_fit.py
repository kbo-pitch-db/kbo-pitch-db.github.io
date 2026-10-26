"""Fit WHAM SMPL initialization to monocular observations and soft foot support.

Camera/floor estimates are nuisance variables, not calibrated measurements.
No target stride length or population pitching pose is imposed.
"""
import numpy as np
from scene_geometry import initial_camera,support_normal,contact_weights,clip_frame,transform_scene,fill_pose_gaps,SMPL_JOINTS,SMPL_PARENTS,FOOT_NAMES


class SceneFitter:
    def __init__(self,wham_root,device):
        import os,sys,torch
        from pathlib import Path
        wham_root=Path(wham_root).resolve();sys.path.insert(0,str(wham_root));before=Path.cwd()
        try:
            os.chdir(wham_root)
            from lib.models import build_body_model
            self.body=build_body_model(device,1)
        finally:os.chdir(before)
        self.body.requires_grad_(False)
        from smplx import SMPL
        from smplx.vertex_ids import vertex_ids
        from lib.utils.transforms import axis_angle_to_matrix
        self.forward_smpl=SMPL.forward;self.rotation=axis_angle_to_matrix
        names=['LBigToe','RBigToe','LSmallToe','RSmallToe','LHeel','RHeel']
        self.foot_vertices=[vertex_ids['smplh'][n] for n in names]
        self.device=device;self.torch=torch

    def model(self,rotations,betas):
        torch=self.torch
        output=self.forward_smpl(self.body,body_pose=rotations[:,1:],global_orient=rotations[:,:1],
            betas=betas,pose2rot=False,return_verts=True)
        joints=torch.einsum('jv,tvk->tjk',self.body.J_regressor_wham,output.vertices)[:,:17]
        root=joints[:,[11,12]].mean(1,keepdim=True)
        return joints-root,output.joints[:,:24]-root,output.vertices[:,self.foot_vertices]-root

    def fit(self,prior,halpe,row,source,background,steps=90,candidates=(1.,4.,12.)):
        torch=self.torch;device=self.device
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        count=len(prior['keypoints']);hypotheses=len(candidates)
        rotations=np.concatenate([prior['smpl_global_orient'],prior['smpl_body_pose']],axis=1).copy()
        supported=np.isfinite(rotations).all((1,2,3))&np.isfinite(prior['keypoints']).all((1,2))
        if supported.sum()<5:raise ValueError('Too few supported SMPL frames for scene fitting')
        observed=supported.copy()
        rotations,interpolated=fill_pose_gaps(rotations,supported,halpe['frame_index'],halpe.get('track_id'))
        supported|=interpolated
        indices=np.flatnonzero(supported)
        for i in np.flatnonzero(~supported):rotations[i]=rotations[indices[np.argmin(abs(indices-i))]]
        beta=np.nanmedian(prior['smpl_betas'][supported],axis=0)
        origin=np.array([float(row['crop_x1']),float(row['crop_y1'])])
        local=halpe['keypoints']-origin
        homogeneous=np.concatenate([local,np.ones((*local.shape[:2],1))],axis=-1)
        xy=np.einsum('tij,tkj->tki',background,homogeneous)[:,:,:2]+origin
        confidence=halpe['scores'].copy()
        confidence[halpe['missing']|~np.isfinite(xy).all(-1)]=0
        confidence[~observed]=0
        xy=np.nan_to_num(xy)
        times=halpe['source_time'];fps=float(halpe['fps'])
        linked=supported[1:]&supported[:-1]&(np.diff(halpe['frame_index'])==1)
        linked&=(np.diff(times)>0)&(np.diff(times)<1.8/fps)
        if 'track_id' in halpe:linked&=(np.diff(halpe['track_id'])==0)|(interpolated[1:]|interpolated[:-1])
        height=float(np.nanmedian(halpe['bbox'][:,3]-halpe['bbox'][:,1]))
        height=max(height,100.)
        principal=np.array([float(source['width'])/2,float(source['height'])/2],np.float32)
        diagonal=np.hypot(float(source['width']),float(source['height']))
        rotation0=tensor(rotations)
        with torch.no_grad():
            body0,anatomy0,feet0=self.model(rotation0,tensor(np.repeat(beta[None],count,axis=0)))
        p0=body0.cpu().numpy();foot0=feet0.cpu().numpy();n0=support_normal(p0)
        scales=[];offsets=[];contacts=[];levels=[]
        for focal in np.asarray(candidates)*diagonal:
            scale,offset,translation=initial_camera(p0,xy[:,:17],confidence[:,:17],focal,principal)
            for k in np.flatnonzero(interpolated):
                neighbors=np.flatnonzero(observed)
                scale[k]=np.interp(k,neighbors,scale[neighbors])
                for axis in range(2):offset[k,axis]=np.interp(k,neighbors,offset[neighbors,axis])
            translation=np.c_[(offset-principal)/scale[:,None],focal/scale]
            support,level=contact_weights(xy[:,20:26]/height,confidence[:,20:26],foot0+translation[:,None],n0,fps,linked)
            scales.append(scale);offsets.append(offset);contacts.append(support);levels.append(level)
        focal0=tensor(np.asarray(candidates)*diagonal)
        logf=torch.nn.Parameter(torch.zeros(hypotheses,device=device))
        logscale=torch.nn.Parameter(tensor(np.log(np.asarray(scales))))
        offset=torch.nn.Parameter(tensor(np.array(offsets)/height))
        delta=torch.nn.Parameter(torch.zeros((hypotheses,count,24,3),device=device))
        shape=torch.nn.Parameter(tensor(np.repeat(beta[None],hypotheses,axis=0)))
        normal_delta=torch.nn.Parameter(torch.zeros((hypotheses,3),device=device))
        plane_offset=torch.nn.Parameter(tensor(-np.array(levels)))
        target=tensor(np.concatenate([xy[:,:17],xy[:,20:26]],axis=1))[None]
        weights=tensor(np.concatenate([confidence[:,:17]**2,confidence[:,20:26]**2*.35],axis=1))[None]
        weights[:,:,:5]*=.4
        contact=tensor(np.asarray(contacts))
        links=tensor(linked)[None,:,None]
        triple=tensor(linked[:-1]&linked[1:])[None,:,None,None]
        mask=tensor(supported)[None]
        normal0=tensor(n0)[None]
        pose_weights=tensor([1.,.6,.6,1.5,.5,.5,1.5,.7,.7,1.5,.8,.8,1.5,1.5,1.5,1.5,1.,1.,1.,1.,1.,1.,1.,1.])[None,None,:,None]
        principal_t=tensor(principal)
        optimizer=torch.optim.Adam([
            {'params':[logscale,offset,plane_offset],'lr':.008},
            {'params':[delta,shape,normal_delta,logf],'lr':.009}])

        def evaluate(optimize_body=True):
            angular=.65*torch.tanh(delta)
            rot=rotation0[None]@self.rotation(angular) if optimize_body else rotation0[None].expand(hypotheses,-1,-1,-1,-1)
            if optimize_body:
                body,anatomy,feet=self.model(rot.reshape(-1,24,3,3),shape[:,None].expand(-1,count,-1).reshape(-1,10))
                body=body.reshape(hypotheses,count,17,3)
                anatomy=anatomy.reshape(hypotheses,count,24,3);feet=feet.reshape(hypotheses,count,6,3)
            else:
                body=body0[None].expand(hypotheses,-1,-1,-1)
                anatomy=anatomy0[None].expand(hypotheses,-1,-1,-1);feet=feet0[None].expand(hypotheses,-1,-1,-1)
            focal=focal0*torch.exp(.6*torch.tanh(logf))
            scale=logscale.exp().clamp(10,4000)
            tz=focal[:,None]/scale
            translation=torch.cat([(offset*height-principal_t)/scale[:,:,None],tz[:,:,None]],dim=-1)
            points=torch.cat([body,feet],dim=2)+translation[:,:,None]
            projected=points[...,:2]/points[...,2:].clamp_min(1.)*focal[:,None,None,None]+principal_t
            residual=(projected-target)/height
            # Robust Huber loss in image-height units, with observed landmark confidence.
            absolute=residual.abs();robust=torch.where(absolute<.04,.5*residual**2,.04*(absolute-.02))
            reprojection=(robust.sum(-1)*weights).sum((1,2))/weights.sum().clamp_min(1.)
            normal=normal0+.25*torch.tanh(normal_delta)
            normal=normal/normal.norm(dim=-1,keepdim=True)
            feet_world=feet+translation[:,:,None]
            signed=(feet_world*normal[:,None,None]).sum(-1)+plane_offset[:,None,None]
            support=(signed.square()*contact).sum((1,2))/contact.sum((1,2)).clamp_min(1.)
            penetration=(torch.relu(-signed-.015).square()*mask[:,:,None]).sum((1,2))/(mask.sum()*6).clamp_min(1.)
            velocity=(feet_world[:,1:]-feet_world[:,:-1])*fps
            contact_link=torch.minimum(contact[:,1:],contact[:,:-1])*links
            slide=(velocity.square().sum(-1)*contact_link).sum((1,2))/contact_link.sum((1,2)).clamp_min(1.)
            pose_prior=(angular.square()*pose_weights*mask[:,:,None,None]).sum((1,2,3))/(mask.sum()*72).clamp_min(1.)
            pose_smooth=((rot[:,2:,:,:2]+rot[:,:-2,:,:2]-2*rot[:,1:-1,:,:2]).square()*triple[:,:,:,:,None]).mean((1,2,3,4))*(fps/30)**4
            trans_smooth=((translation[:,2:]+translation[:,:-2]-2*translation[:,1:-1]).square()*triple[:,:,:,0]).mean((1,2))*(fps/30)**4
            shape_prior=(shape-tensor(beta)[None]).square().mean(-1)
            normal_prior=(normal-normal0).square().sum(-1)
            losses={'image':reprojection,'contact_height':support,'penetration':penetration,'contact_velocity':slide,
                'pose_prior':pose_prior,'pose_acceleration':pose_smooth,'translation_acceleration':trans_smooth,
                'shape_prior':shape_prior,'normal_prior':normal_prior,'focal_prior':logf.square()}
            total=reprojection+.03*support+.15*penetration+.0008*slide+.012*pose_prior+.002*pose_smooth+.004*trans_smooth+.0001*shape_prior+.008*normal_prior+.00002*logf.square()
            return total,losses,(body,anatomy,feet,translation,normal,projected,focal,angular)

        with torch.no_grad():initial_total,initial_losses,initial_data=evaluate(False)
        initial={k:v.detach().cpu().numpy().copy() for k,v in initial_losses.items()}
        # Start by fitting the camera/support variables before changing joint rotations.
        for iteration in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total,losses,data=evaluate(iteration>=min(15,steps//4))
            if not torch.isfinite(total).all():raise ValueError('Non-finite scene fitting objective')
            total.sum().backward()
            torch.nn.utils.clip_grad_norm_([logf,logscale,offset,delta,shape,normal_delta,plane_offset],10.)
            optimizer.step()
        with torch.no_grad():total,losses,data=evaluate(True)
        values=[v.detach().cpu().numpy() for v in data]
        body,anatomy,feet,translation,normals,projected,focals,angular=values
        totals=total.detach().cpu().numpy();chosen=int(np.argmin(totals))
        reprojection=np.linalg.norm(projected-target.cpu().numpy(),axis=-1)
        all_diagnostics=[]
        for c in range(hypotheses):
            loss={k:float(v[c].detach().cpu()) for k,v in losses.items()}
            loss['initial']={k:float(v[c]) for k,v in initial.items()}
            loss['focal_pixels']=float(focals[c]);loss['objective']=float(totals[c])
            loss['median_reprojection_px']=float(np.median(reprojection[c,:,:17][confidence[:,:17]>.5]))
            all_diagnostics.append(loss)
        c=chosen;candidate_ok=all_diagnostics[c]['image']<=initial['image'][c]*1.1+1e-7
        candidate_ok&=all_diagnostics[c]['penetration']<.02
        candidate_ok&=bool(np.isfinite(translation[c,supported]).all() and np.ptp(translation[c,supported],axis=0).max()<8.)
        hypothesis_stride=[float(np.nanmax(np.linalg.norm(body[h,:,15]-body[h,:,16],axis=-1)[supported])) for h in range(hypotheses)]
        if not candidate_ok:
            # Preserve an explicit fitted-camera initialization when joint fitting regresses.
            with torch.no_grad():
                body=body0[None].expand(hypotheses,-1,-1,-1).cpu().numpy()
                anatomy=anatomy0[None].expand(hypotheses,-1,-1,-1).cpu().numpy()
                feet=feet0[None].expand(hypotheses,-1,-1,-1).cpu().numpy()
            translation=initial_data[3].cpu().numpy();normals=initial_data[4].cpu().numpy()
            offsets_plane=-np.array(levels);selected_shape=beta;focals=focal0.detach().cpu().numpy()
        else:
            offsets_plane=plane_offset.detach().cpu().numpy();selected_shape=shape[c].detach().cpu().numpy()
        camera_body=body[c]+translation[c,:,None]
        camera_anatomy=anatomy[c]+translation[c,:,None]
        camera_feet=feet[c]+translation[c,:,None]
        # Choose the visible reference floor once from the clip's lower foot envelope.
        # This is one origin change, never a frame-wise lift or foot/stride adjustment.
        optimizer_plane_offset=float(offsets_plane[c])
        minimum=float(np.nanmin(camera_feet[supported]@normals[c]+optimizer_plane_offset))
        floor_reference_shift=max(0.,.005-minimum)
        reference_offset=optimizer_plane_offset+floor_reference_shift
        rotation,anchor,policy=clip_frame(normals[c],reference_offset,camera_feet,np.array(contacts[c]))
        output={key:transform_scene(v,rotation,anchor) for key,v in
            [('keypoints',camera_body),('anatomical_keypoints',camera_anatomy),('foot_keypoints',camera_feet),('root_translation',translation[c])]}
        for key in ['keypoints','anatomical_keypoints','foot_keypoints','root_translation']:output[key][~supported]=np.nan
        with torch.no_grad():
            fitted_rotations=(rotation0@self.rotation(tensor(angular[c]))).cpu().numpy() if candidate_ok else rotations.copy()
        fitted_rotations[~supported]=np.nan
        camera_translation=translation[c].copy();camera_translation[~supported]=np.nan
        output.update(scene_rotation=rotation,scene_origin=anchor,contact_weights=contacts[c],
            interpolated_frames=interpolated,
            smpl_global_orient=fitted_rotations[:,:1],smpl_body_pose=fitted_rotations[:,1:],
            camera_translation=camera_translation,camera_principal=principal,
            smpl_betas=selected_shape,fitted_camera_focal=np.array(focals[c]),
            anatomical_joints=np.array(SMPL_JOINTS),anatomical_parents=np.array(SMPL_PARENTS),foot_joints=np.array(FOOT_NAMES))
        diagnostics={'fit_accepted':bool(candidate_ok),'chosen_camera_candidate':c,
            'output_stage':'joint pose/shape/camera fit' if candidate_ok else 'SMPL and weak-projection camera initialization; joint-fit candidate rejected',
            'candidate_diagnostics':all_diagnostics,'steps':steps,
            'anchor_policy':policy,'scene_rotation':rotation.tolist(),'scene_origin_camera':anchor.tolist(),
            'floor_normal_camera':normals[c].tolist(),'floor_offset_camera':reference_offset,
            'optimizer_floor_offset_camera':optimizer_plane_offset,
            'floor_reference_lower_envelope_shift_model_m':floor_reference_shift,
            'supported_frames':int(supported.sum()),'total_frames':count,
            'interpolated_pose_frames':int(interpolated.sum()),
            'contacts_above_0_3':int((contacts[c]>.3).sum()),
            'camera_hypothesis_peak_interankle_model_m':hypothesis_stride,
            'camera_hypothesis_interpretation':'Candidate spread is a sensitivity diagnostic, not a measured-stride confidence interval.',
            'coordinate_system':'clip-fixed inferred support frame; model-estimated meters; y normal to inferred local floor',
            'limitations':'Uncalibrated monocular inference. Camera/floor/contact are estimated; no measured stride ground truth, surveyed mound plane or OpenSim IK.'}
        return output,diagnostics
