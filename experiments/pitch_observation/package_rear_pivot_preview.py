"""Package selected FK-consistent fits behind an explicit Explorer URL only.

Original 2D and published 813-clip release are read-only. The display transform
is one proper rigid transform per clip, never a sliding-foot correction.
"""
from common import *
from package_explorer_3d import build_pose
import gzip

def main():
 index=json.loads((CODE/'data/clips.json').read_text());by={r['id']:r for r in index['clips']}
 folder=OUT/'rear_pivot_selected_v14'
 registry=dict(schema_version=1,version='rear-pivot-v14-20261028',clips={},role='unlisted_research_preview',published_release_changed=False)
 for rec in sample('tune',True):
  cid=rec['id'];path=folder/f'{cid}.npz';meta_path=path.with_suffix('.json')
  if not meta_path.exists():continue
  meta=json.loads(meta_path.read_text())
  if meta['status']!='eligible_research_preview':continue
  selected=meta['selected_metrics'];assert selected['consistency_gate'] and selected['joint_gate']
  assert sha(path)==meta['selected_pose_sha256']
  z=load(path);base=by[cid];old=json.loads((CODE/'data'/base['pose3d']).read_text());two=json.loads((CODE/'data'/base['poseRefined']).read_text())
  pose,_,_=build_pose(cid,'GVHMR + pivot-aware joint fitting','gvhmr_f4',path,old,two)
  normal=z['normal'].astype(float);normal/=np.linalg.norm(normal)
  x=np.array([1.,0.,0.]);x-=np.dot(x,normal)*normal;x/=np.linalg.norm(x)
  rotation=np.stack([x,normal,np.cross(x,normal)]);assert np.linalg.det(rotation)>.999999
  absolute=z['foot_keypoints'].astype(float)+z['translation'][:,None]
  first=np.argwhere(z['contact_weight']>.03)[0];contact=absolute[first[0],first[1]];offset=float(z['pilot_support_offset'])
  origin=contact-(contact@normal+offset)*normal
  transformed={k:(z[k].astype(float)+z['translation'][:,None]-origin)@rotation.T for k in ['keypoints','anatomical_keypoints','foot_keypoints']}
  for i,frame in enumerate(pose['frames']):
   for j,value in enumerate(transformed['keypoints'][i]):frame['keypoints'][j]=[*np.round(value,6).tolist(),frame['keypoints'][j][3]]
   frame['anatomy']=np.round(transformed['anatomical_keypoints'][i],6).tolist();frame['feet']=np.round(transformed['foot_keypoints'][i],6).tolist()
  cloud=np.concatenate([transformed['anatomical_keypoints'].reshape(-1,3),transformed['foot_keypoints'].reshape(-1,3)])
  pose.update(method='gvhmr-rear-pivot-v14',coordinate_system='Clip-fixed inferred support frame; Y normal to one estimated plane; uncalibrated model meters')
  pose['scene']=dict(frame='clip-fixed-inferred-support',show_floor=True,floor=dict(height=0.,estimated=True,normal_camera=normal.tolist(),offset_camera=offset),rotation_camera_to_scene=rotation.tolist(),origin_camera=origin.tolist(),comparison_bounds=dict(min=cloud.min(0).tolist(),max=cloud.max(0).tolist()),per_frame_recentering=False,anchor_policy='One constant projection of initial supported marker onto one fitted plane; no time-dependent floor, camera following, endpoint hold or root recentering',calibration='Estimated contact plane, not surveyed mound or calibrated world',focal_pixels_estimate=float(z['focal']))
  pose['limitations']=['Research preview; public 813-clip dataset release unchanged','GVHMR approximate SMPL adapter; fixed shape and local upper-limb rotations; lower-body/trunk/root/shared camera jointly fitted','Initial rear toe gets finite confidence-weighted closure; heel lift and subsequent contact release allowed','Contact, camera, absolute stride and model scale unvalidated; no measured 3D GT','Broadcast playback timestamps; physical slow-motion factor unknown','Original-source worst-frame foot reprojection may worsen despite improved p95; no human manual-review claim']
  pose['quality'].update(status='research_candidate_unvalidated',contact_validated=False,consistency_screen_passed=True)
  pose['fit_diagnostics']=dict(selected_metrics=selected,selection_rule=meta['rule'],phase=meta['phase'],contact_episodes=meta['marker_states'],human_review_complete=False,scope='Six previously used tuning clips; frozen test24 untouched; contact residual is objective-linked, not independent accuracy evidence')
  relative=f'reconstructions/rear-pivot-v14/{cid}.json.gz';dest=CODE/'data'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(gzip.compress(json.dumps(clean(pose),ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0))
  variant=dict(id='rear_pivot_v14',label=pose['model'],pose3d=relative,model3d=pose['model'],method=pose['method'],sceneFrame=pose['scene']['frame'],sourceFitSha256=sha(path),qualityUrl=None,role='preview',sha256_gzip=sha(dest),coordinates=pose['coordinate_system'])
  registry['clips'][cid]=dict(preferred='rear_pivot_v14',variants=dict(rear_pivot_v14=variant))
 save(CODE/'data/preview_rear_pivot_registry.json',registry)
 save(folder/'preview_export.json',dict(clips=len(registry['clips']),registry_sha256=sha(CODE/'data/preview_rear_pivot_registry.json'),canonical_index_sha256=sha(CODE/'data/clips.json'),source_2d_assets_changed=False,published_release_changed=False,packager_sha256=sha(__file__)))
 print('REAR PIVOT EXPLICIT-LINK PREVIEWS',len(registry['clips']))

if __name__=='__main__':main()
