"""Explicit-link Explorer previews; never replace the complete published release."""
from common import *
from package_explorer_3d import build_pose
import gzip

def main():
 index=json.loads((CODE/'data/clips.json').read_text());by={r['id']:r for r in index['clips']};folder=OUT/'support_transfer_arm_v4';registry=dict(version='support-v4-ground-20261028',clips={},role='unlisted_research_preview',published_release_changed=False);report=[]
 for rec in sample('tune',True):
  cid=rec['id'];path=folder/f'{cid}.npz';meta_path=path.with_suffix('.json')
  if not meta_path.exists():continue
  meta=json.loads(meta_path.read_text());chosen=meta['hypotheses'][meta['selected']]
  if not chosen['source_gate'] or not meta['contact_available']:
   report.append(dict(id=cid,status='not_exported',reason='source_gate_or_contact_unavailable'));continue
  z=load(path);base=by[cid];old=json.loads((CODE/'data'/base['pose3d']).read_text());two=json.loads((CODE/'data'/base['poseRefined']).read_text());pose,_,_=build_pose(cid,'GVHMR + support trajectory','gvhmr_f4',path,old,two)
  normal=z['normal'].astype(float);normal/=np.linalg.norm(normal);x=np.array([1.,0.,0.]);x-=np.dot(x,normal)*normal;x/=np.linalg.norm(x);rotation=np.stack([x,normal,np.cross(x,normal)]);assert np.linalg.det(rotation)>.999999
  absolute=z['foot_keypoints'].astype(float)+z['translation'][:,None];valid=np.argwhere(z['contact_weight']>.03);first=valid[0];contact=absolute[first[0],first[1]];offset=float(z['pilot_support_offset']);origin=contact-(contact@normal+offset)*normal
  transformed={k:(z[k].astype(float)+z['translation'][:,None]-origin)@rotation.T for k in ['keypoints','anatomical_keypoints','foot_keypoints']}
  for i,frame in enumerate(pose['frames']):
   for j,value in enumerate(transformed['keypoints'][i]):frame['keypoints'][j]=[*np.round(value,6).tolist(),frame['keypoints'][j][3]]
   frame['anatomy']=np.round(transformed['anatomical_keypoints'][i],6).tolist();frame['feet']=np.round(transformed['foot_keypoints'][i],6).tolist()
  pose.update(method='gvhmr-support-v4',coordinate_system='Clip-fixed inferred support frame; Y normal to one estimated plane; uncalibrated model meters')
  cloud=np.concatenate([transformed['anatomical_keypoints'].reshape(-1,3),transformed['foot_keypoints'].reshape(-1,3)])
  pose['scene']=dict(frame='clip-fixed-inferred-support',show_floor=True,floor=dict(height=0.,estimated=True,normal_camera=normal.tolist(),offset_camera=offset),rotation_camera_to_scene=rotation.tolist(),origin_camera=origin.tolist(),comparison_bounds=dict(min=cloud.min(0).tolist(),max=cloud.max(0).tolist()),per_frame_recentering=False,anchor_policy='One constant projection of initial supported marker onto one fitted plane; no time-dependent floor, camera following, endpoint hold or root recentering',calibration='Estimated contact plane, not surveyed mound or calibrated world',focal_pixels_estimate=float(z['focal']))
  pose['limitations']=['Research preview only; complete public dataset release unchanged','Frozen GVHMR SMPL adapter articulation; root and shared camera optimized using marker contact episodes','Ground/contact/absolute stride remain unvalidated; no measured 3D GT','Physical slow-motion timing unknown']
  relative=f'reconstructions/support-v4/{cid}.json.gz';dest=CODE/'data'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(json.dumps(clean(pose),ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0))
  variant=dict(id='support_v4',label='GVHMR + support trajectory',pose3d=relative,model3d=pose['model'],method=pose['method'],sceneFrame=pose['scene']['frame'],sourceFitSha256=sha(path),qualityUrl=None,role='preview',sha256_gzip=sha(dest),coordinates=pose['coordinate_system'])
  registry['clips'][cid]=dict(preferred='support_v4',variants=dict(support_v4=variant));report.append(dict(id=cid,status='preview_only',source_gate=True,source_p50_px=chosen['source_p50_px'],wrist_p95_px=chosen['wrist_p95_px']))
 save(CODE/'data/preview_registry.json',registry);save(OUT/'root_return_diagnosis/preview_export.json',dict(clips=len(registry['clips']),results=report,published_release_changed=False))
 print('EXPLICIT-LINK PREVIEWS',len(registry['clips']))
if __name__=='__main__':main()
