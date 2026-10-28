"""Audit and publish only the versioned correction; preserve all prior assets."""
from common import *
import argparse,copy,gzip,shutil
from correct_oh_subject import CID,FIX,install_inputs

def package(promote=False):
 import torch
 torch.set_num_threads(1)
 install_inputs()
 from correct_oh_subject import corrected_inputs
 from package_explorer_3d import build_pose
 from package_motion import export_pose
 from build_site_data import render_pose_media
 from pitch_study_fit import StudyFitter
 public=CODE/'data';index=json.loads((public/'clips.json').read_text());old=next(c for c in index['clips'] if c['id']==CID);assert 'subjectCorrection' not in old,'Already promoted'
 before=copy.deepcopy(index);old_two=json.loads((public/old['poseRefined']).read_text());old_pose=json.loads(gzip.decompress((public/old['pose3d']).read_bytes()))
 path=OUT/'all813_gvhmr_v1/selected'/f'{CID}.npz';z=load(path);meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['selected_pose_sha256'];r,c,s,h,w=corrected_inputs(CID)
 assert np.array_equal(z['frame_index'],h['frame_index']);assert np.array_equal(z['source_time'],h['source_time']);assert len(h['frame_index'])==75
 assert meta['phase']['events']['lift_interval'] is None and meta['phase']['plant_interval'] is None
 assert not meta['contact_hypotheses_available'];assert not z['contact_weight'].any()
 for k in ['keypoints','anatomical_keypoints','foot_keypoints','translation','smpl_rotations']:assert np.isfinite(z[k]).all()
 # Check genuine native FK and fixed shape, not just plausible rendered bones.
 beta=z['smpl_betas'];beta=np.median(beta,axis=0) if beta.ndim==2 else beta
 body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False);body.prepare_shape(beta)
 with torch.no_grad():p,a,f=body.model(torch.tensor(z['smpl_rotations']),torch.tensor(np.repeat(beta[None],75,axis=0)))
 fk=max(float(abs(v.numpy()-z[k]).max()) for k,v in [('keypoints',p),('anatomical_keypoints',a),('foot_keypoints',f)]);assert fk<1e-5
 stage=FIX/'site_stage';asset=Path('assets')/CID/'subject-corrected-v1';folder=stage/'data'/asset;folder.mkdir(parents=True,exist_ok=True)
 raw=export_pose(FIX/'halpe26_raw.npz',c,2,old['fps']);two=export_pose(FIX/'halpe26_refined.npz',c,2,old['fps'])
 for payload in [raw,two]:
  payload.update(subject='pitcher',subject_correction='oh-delivery3-v1',source_image_size=[1920,1080],crop_box=[0,0,1920,1080])
  assert [(f['source_frame'],f['source_time'],f['time']) for f in payload['frames']]==[(f['source_frame'],f['source_time'],f['time']) for f in old_two['frames']]
 for name,payload in [('pose_halpe26.json',raw),('pose_halpe26_refined.json',two)]:
  (folder/name).write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False))
 render_pose_media(two,folder,'pose.mp4')
 pose,_,_=build_pose(CID,'GVHMR + phase-aware contact fitting','gvhmr_f4',path,old_pose,two)
 normal=z['normal'].astype(float);factor=np.linalg.norm(normal);normal/=factor;offset=float(z['pilot_support_offset'])/factor
 x=np.array([1.,0.,0.]);x-=(x@normal)*normal;x/=np.linalg.norm(x);rotation=np.stack([x,normal,np.cross(x,normal)]);assert np.linalg.det(rotation)>.99999
 feet=z['foot_keypoints'].astype(float)+z['translation'][:,None];point=feet[0,np.argmin(feet[0]@normal+offset)];origin=point-(point@normal+offset)*normal
 values={k:(z[k].astype(float)+z['translation'][:,None]-origin)@rotation.T for k in ['keypoints','anatomical_keypoints','foot_keypoints']}
 for i,frame in enumerate(pose['frames']):
  frame['keypoints']=[[*p,float(q[3])] for p,q in zip(np.round(values['keypoints'][i],6).tolist(),frame['keypoints'])];frame['anatomy']=np.round(values['anatomical_keypoints'][i],6).tolist();frame['feet']=np.round(values['foot_keypoints'][i],6).tolist();frame['available']=True
 cloud=np.concatenate([values['anatomical_keypoints'].reshape(-1,3),values['foot_keypoints'].reshape(-1,3)])
 reasons=list(dict.fromkeys(meta['quality_reasons']+['clip_starts_after_stride','full_stride_not_observed']))
 pose.update(method='gvhmr-phase-aware-contact',release_id='gvhmr-contact-20261028-v1-subject-corrected1',subject='pitcher',subject_correction='oh-delivery3-v1',coordinate_system='Clip-fixed inferred support frame; uncalibrated model meters')
 pose['scene']=dict(frame='clip-fixed-inferred-support',show_floor=True,floor=dict(height=0.,estimated=True,normal_camera=normal.tolist(),offset_camera=offset,contact_available=False,rear_contact_evaluable=False),rotation_camera_to_scene=rotation.tolist(),origin_camera=origin.tolist(),comparison_bounds=dict(min=cloud.min(0).tolist(),max=cloud.max(0).tolist()),per_frame_recentering=False,anchor_policy='One constant inferred plane frame; no per-frame recentering or endpoint hold',calibration='Unknown contact: reference floor, not measured mound',focal_pixels_estimate=float(z['focal']))
 pose['quality'].update(status='review_required_preview',numeric_finite=True,numerical_checks='native_fk_checked_all_75_frames_and_bounded_knees',consistency_screen_passed=False,source_visual_comparison='Codex inspected original pitcher overlays; not human annotation',human_review_complete=False,root_relative_articulation='estimated_unvalidated',global_trajectory='estimated_unvalidated',contact='unknown',rear_contact_evaluable=False,absolute_stride='not_observed',contact_validated=False,benchmark_eligible=False,reasons=reasons,valid_3d_frames=75,unavailable_3d_frames=0)
 pose['fit_provenance']=dict(selected_fit_sha256=sha(path),base_fit_sha256=meta['source_fit_sha256'],phase=meta['phase'],contact_episodes=[],metrics=meta['selected_metrics'],subject_input_sha256=sha(FIX/'halpe26_refined.npz'),shape_source='Median HMR2a RGB shape of corrected pitcher; no original batter WHAM pose or shape',background_assessment=meta['background_assessment'])
 pose['limitations']=['Versioned correction of a confirmed wrong-person input, not measured KBO 3D ground truth','Clip starts after stride; leg lift and landing are unobserved, release not ball-verified','No contact anchor is applied when contact is unknown','GVHMR network with RTMPose/HMR2a and approximate shared-rotation SMPL adapter','Scale, camera, depth, ground and biomechanical accuracy remain unvalidated']
 relative=Path('reconstructions/gvhmr-release-v1/subject-corrected-v1')/f'{CID}.json.gz';dest=stage/'data'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(json.dumps(clean(pose),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
 quality=Path('quality/gvhmr-release-v1/subject-corrected-v1')/f'{CID}.json';save(stage/'data'/quality,dict(id=CID,**pose['quality'],metrics=meta['selected_metrics'],selected_fit_sha256=sha(path)))
 report=dict(id=CID,version='oh-delivery3-v1',cause='Original bbox/2D tracks the batter, not the pitcher',old_clip=old,original_2d_sha256=sha(public/old['poseRefined']),original_raw_2d_sha256=sha(public/old['pose']),original_3d_sha256=sha(public/old['pose3d']),new_2d=str(asset/'pose_halpe26_refined.json'),new_2d_sha256=sha(folder/'pose_halpe26_refined.json'),new_raw_2d=str(asset/'pose_halpe26.json'),new_raw_2d_sha256=sha(folder/'pose_halpe26.json'),new_3d=str(relative),new_3d_sha256=sha(dest),new_quality=str(quality),selected_fit_sha256=sha(path),native_fk_max_error_model=fk,frames=75,frame_and_timestamp_preserved=True,original_assets_preserved=True,other_clips_unchanged=True,original_batter_shape_or_pose_used=False,original_pixels_published=False,contact_available=False,complete_delivery=False,stride_validated=False,human_review_complete=False,input_provenance=json.loads((FIX/'input_provenance.json').read_text()),selection=meta,code_sha256=sha(__file__))
 report_rel=Path('subject_corrections')/f'{CID}-v1.json';save(stage/'data'/report_rel,report)
 replacement=copy.deepcopy(old);replacement.update(pose=str(asset/'pose_halpe26.json'),poseRefined=str(asset/'pose_halpe26_refined.json'),poseFile='pose_halpe26.json',poseThumbnail=str(asset/'pose_thumbnail.jpg'),poseVideo='',pose3d=str(relative),motionQuality=str(quality),sceneSignature=sha(path),reconstructionVersion=pose['release_id'],pose3dReasons=reasons,subjectCorrection=str(report_rel),poseStatus='available',deliveryCoverage='post-stride arm motion and follow-through; leg lift and landing outside interval')
 index['clips']=[replacement if c['id']==CID else c for c in index['clips']]
 assert [c for c in before['clips'] if c['id']!=CID]==[c for c in index['clips'] if c['id']!=CID]
 summary=json.loads((public/'gvhmr_release_summary.json').read_text());summary['subject_corrected_clips']=[CID];summary['original_final_2d_assets_preserved']=True;summary['subject_correction_semantics']='Versioned replacement of a confirmed batter input; original 2D/3D assets preserved, no 3D accuracy claim'
 registry_path=public/'subject_corrections.json';registry=json.loads(registry_path.read_text()) if registry_path.exists() else dict(schema_version=1,clips={});registry['clips'][CID]=dict(manifest=str(report_rel),old_2d=old['poseRefined'],new_2d=replacement['poseRefined'],old_3d=old['pose3d'],new_3d=replacement['pose3d'])
 save(stage/'data/subject_corrections.json',registry);save(stage/'data/gvhmr_release_summary.json',summary);save(stage/'data/clips.json',index)
 if promote:
  assert sha(public/old['poseRefined'])==report['original_2d_sha256'];assert sha(public/old['pose3d'])==report['original_3d_sha256']
  shutil.copytree(stage/'data',public,dirs_exist_ok=True)
 save(FIX/'publication_audit.json',dict(id=CID,staged=True,promoted_local=promote,frames=75,fk_max_error_model=fk,original_assets_preserved=True,other_812_clips_unchanged=True,source_timing_unchanged=True,original_pixels_published=False,main_index_sha256=sha(stage/'data/clips.json')))
 print(json.dumps(json.loads((FIX/'publication_audit.json').read_text()),indent=2))

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();ap=argparse.ArgumentParser();ap.add_argument('--promote',action='store_true');a=ap.parse_args();package(a.promote)
