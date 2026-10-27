"""Stage the full GVHMR release, then atomically promote its canonical index.

Promotion requires all 813 new estimates, unchanged final 2D and explicit
quality states. Archived WHAM raw JSON is losslessly gzipped to stay inside
the static Pages size budget. Original bytes and index remain recoverable.
"""
from common import *
from all813_release import ROOT,population,enable_primary_shot_policy
from package_explorer_3d import build_pose
import argparse,gzip,copy,shutil

RELEASE='gvhmr-contact-20261028-v1'
FOLDER='gvhmr-release-v1'

def pack(cid,clip,site):
 path=ROOT/'selected'/f'{cid}.npz';meta=json.loads(path.with_suffix('.json').read_text());z=load(path)
 assert sha(path)==meta['selected_pose_sha256']
 old=json.loads((CODE/'data'/clip['pose3d']).read_text());two=json.loads((CODE/'data'/clip['poseRefined']).read_text());count=len(z['frame_index']);assert count<=len(two['frames']);short_two={**two,'frames':two['frames'][:count]};pose,_,_=build_pose(cid,'GVHMR + phase-aware contact fitting','gvhmr_f4',path,old,short_two)
 normal=z['normal'].astype(float);norm=np.linalg.norm(normal);normal/=norm;offset=float(z['pilot_support_offset'])/norm
 x=np.array([1.,0.,0.]);x-=x@normal*normal;x/=np.linalg.norm(x);rotation=np.stack([x,normal,np.cross(x,normal)]);assert np.linalg.det(rotation)>.99999
 feet=z['foot_keypoints'].astype(float)+z['translation'][:,None];valid=np.argwhere(z['contact_weight']>.03)
 first=valid[0] if len(valid) else np.array([0,int(np.argmin(feet[0]@normal+offset))]);point=feet[first[0],first[1]];origin=point-(point@normal+offset)*normal
 transformed={k:(z[k].astype(float)+z['translation'][:,None]-origin)@rotation.T for k in ['keypoints','anatomical_keypoints','foot_keypoints']}
 assert all(np.isfinite(v).all() for v in transformed.values())
 for i,frame in enumerate(pose['frames']):
  for j,p in enumerate(transformed['keypoints'][i]):frame['keypoints'][j]=[*np.round(p,6).tolist(),frame['keypoints'][j][3]]
  frame['anatomy']=np.round(transformed['anatomical_keypoints'][i],6).tolist();frame['feet']=np.round(transformed['foot_keypoints'][i],6).tolist()
  frame['available']=True
 policy_path=ROOT/'shot_validity'/f'{cid}.json'
 if count<len(two['frames']):
  policy=json.loads(policy_path.read_text());assert policy['confirmed_source_cut'] and policy['valid_frames']==count
  pose['shot_validity']={**policy,'valid_frame_range_exclusive':[0,count]}
  for f in two['frames'][count:]:pose['frames'].append(dict(frame=f['frame'],source_frame=f['source_frame'],time=f['time'],source_time=f['source_time'],available=False,unavailable_reason='source_camera_change',keypoints=[None]*17,anatomy=[None]*24,feet=[None]*6,interpolated=[False]*17,outlier=[False]*17))
 cloud=np.concatenate([transformed['anatomical_keypoints'].reshape(-1,3),transformed['foot_keypoints'].reshape(-1,3)])
 pose.update(method='gvhmr-phase-aware-contact',release_id=RELEASE,coordinate_system='Clip-fixed inferred support frame; Y along one estimated plane normal; uncalibrated model meters')
 pose['scene']=dict(frame='clip-fixed-inferred-support',show_floor=True,floor=dict(height=0.,estimated=True,normal_camera=normal.tolist(),offset_camera=offset,contact_available=meta.get('contact_hypotheses_available',meta['contact_available']),rear_contact_evaluable=meta['contact_available']),rotation_camera_to_scene=rotation.tolist(),origin_camera=origin.tolist(),comparison_bounds=dict(min=cloud.min(0).tolist(),max=cloud.max(0).tolist()),per_frame_recentering=False,anchor_policy='One constant projection onto one fitted plane; no framewise floor, camera following, endpoint hold or recentering',calibration='Estimated contact plane, not surveyed mound or calibrated world',focal_pixels_estimate=float(z['focal']))
 m=meta['selected_metrics'];screen=bool(m.get('consistency_gate'))
 pose['quality'].update(status='review_required_preview',numeric_finite=True,consistency_screen_passed=screen,numerical_checks='finite_fk_and_bounded_knees',source_visual_comparison='not_manually_reviewed',human_review_complete=False,root_relative_articulation='estimated_unvalidated',global_trajectory='estimated_unvalidated',contact='hypothesis_unvalidated' if meta.get('contact_hypotheses_available',meta['contact_available']) else 'unknown',rear_contact_evaluable=meta['contact_available'],absolute_stride='unvalidated_model_scale',contact_validated=False,benchmark_eligible=False,reasons=meta['quality_reasons'])
 pose['quality'].update(valid_3d_frames=count,unavailable_3d_frames=len(two['frames'])-count)
 pose['limitations']=['Monocular estimate, not KBO measured 3D ground truth','Official GVHMR learned network with final RTMPose/HMR2a inputs and approximate shared-rotation SMPL adapter, not native SMPL-X postprocessing','Fixed sequence SMPL shape and local upper-arm chain; lower body, trunk, root and shared focal fitted','Unknown/unobserved contact has zero constraint; heels can rise and contact episodes release','Model scale, camera, depth, ground and absolute stride unvalidated; broadcast slow-motion physical timing unknown','Consistency/source screens do not constitute manual visual review or measured accuracy']
 pose['fit_provenance']=dict(selected_fit_sha256=sha(path),base_fit_sha256=meta['source_fit_sha256'],reused_frozen_v14=meta.get('reused_frozen_v14',False),selection_status=meta['status'],contact_episodes=meta['marker_states'],phase=meta['phase'],background_assessment=meta.get('background_assessment'),metrics=m,estimator_sha256=pose['model_sha256'])
 relative=f'reconstructions/{FOLDER}/{cid}.json.gz';dest=site/'data'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(json.dumps(clean(pose),ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0))
 quality_relative=f'quality/{FOLDER}/{cid}.json';save(site/'data'/quality_relative,dict(id=cid,release_id=RELEASE,**pose['quality'],metrics=m,selected_fit_sha256=sha(path)))
 result=copy.deepcopy(clip)
 for key in ['pose3dRaw','whamSignature','legacyPose3d']:result.pop(key,None)
 result.update(pose3d=relative,pose3dBackend='gvhmr',pose3dPrimaryLayout='smpl24',pose3dSupplementaryLayout='coco17',motionMethod=pose['method'],model3d=pose['model'],pose3dStatus='review_required_preview',pose3dReasons=meta['quality_reasons'],sceneFrame=pose['scene']['frame'],sceneSignature=sha(path),motionQuality=quality_relative,strideValidated=False,reconstructionVersion=RELEASE)
 record=dict(id=cid,player=clip['player'],old_method=clip.get('motionMethod'),new_method=pose['method'],selected_fit_sha256=sha(path),gzip_sha256=sha(dest),final_2d_public_sha256=sha(CODE/'data'/clip['poseRefined']),frames=len(pose['frames']),valid_3d_frames=count,unavailable_3d_frames=len(two['frames'])-count,contact_available=meta['contact_available'],contact_hypotheses_available=meta.get('contact_hypotheses_available',meta['contact_available']),consistency_screen_passed=screen,human_review_complete=False,reused_frozen_v14=meta.get('reused_frozen_v14',False),quality_reasons=meta['quality_reasons'],metrics=m)
 return result,record

def stage(limit=0):
 enable_primary_shot_policy()
 archive=ROOT/'original_public_index.json'
 if not archive.exists():
  released_archive=CODE/'data/releases/wham-20261028/original_index.json.gz'
  if released_archive.exists():archive.write_bytes(gzip.decompress(released_archive.read_bytes()))
  else:shutil.copy2(CODE/'data/clips.json',archive)
 index=json.loads(archive.read_text());records=json.loads((ROOT/'input_manifest.json').read_text())['inputs'];by={r['id']:r for r in records}
 site=ROOT/'site_stage';site.mkdir(exist_ok=True);clips=[];report=[]
 selected_clips=index['clips']
 if limit:selected_clips=[c for c in selected_clips if (ROOT/'selected'/f'{c["id"]}.npz').exists()][:limit]
 for clip in selected_clips:
  cid=clip['id'];input_record=by[cid]
  assert sha(DATA/input_record['final_2d_path'])==input_record['final_2d_sha256'],'Original final 2D changed: '+cid
  new,r=pack(cid,clip,site);clips.append(new);report.append(r)
 result={**index,'schemaVersion':9,'packageVersion':RELEASE,'clips':clips,'packageStats':{**{k:v for k,v in index.get('packageStats',{}).items() if not k.startswith('clipsWith')},'clipsWith3d':len(clips),'clipsWithWham':0,'clipsWithRawWham':0,'clipsWithGvhmr':len(clips),'clipsWithSceneFit':len(clips),'clipsWithContactFit':sum(r['contact_hypotheses_available'] for r in report),'clipsWithScreened3d':0,'screenedPreviews':0,'reviewRequiredPreviews':len(clips)}}
 save(site/'data/clips.json',result)
 summary=dict(release_id=RELEASE,clips=len(clips),players=len({c['playerKey'] for c in clips}),frames=sum(r['frames'] for r in report),gvhmr_clips=len(clips),legacy_fallback_clips=0,contact_available=sum(r['contact_available'] for r in report),contact_unknown=sum(not r['contact_available'] for r in report),consistency_screen_passed=sum(r['consistency_screen_passed'] for r in report),human_manually_reviewed=0,absolute_stride_validated=0,reused_frozen_v14=sum(r['reused_frozen_v14'] for r in report),original_index_sha256=sha(archive),packager_sha256=sha(__file__),quality_semantics='Availability and objective-linked consistency; not measured 3D accuracy or manual source review')
 summary.update(contact_hypotheses_available=sum(r['contact_hypotheses_available'] for r in report),rear_contact_evaluable=sum(r['contact_available'] for r in report),contact_hypotheses_unknown=sum(not r['contact_hypotheses_available'] for r in report),contact_count_semantics='contact_available is legacy alias for evaluable initial rear toe; contact_hypotheses_available includes other observed marker episodes')
 summary.update(valid_3d_frames=sum(r['valid_3d_frames'] for r in report),unavailable_3d_frames=sum(r['unavailable_3d_frames'] for r in report),clips_with_camera_cut_exclusion=sum(r['unavailable_3d_frames']>0 for r in report))
 save(site/'data/gvhmr_release_summary.json',summary)
 save(site/'data/motion_summary.json',dict(clips=len(clips),frames=summary['frames'],valid_3d_frames=summary['valid_3d_frames'],unavailable_3d_frames=summary['unavailable_3d_frames'],model_counts={'GVHMR + phase-aware contact fitting':len(clips)},status_counts={'review_required_preview':len(clips)},release_id=RELEASE,published_final_3d=len(clips),screened_previews=0,review_required_previews=len(clips),consistency_screen_passed=summary['consistency_screen_passed'],interpretation='Actual new GVHMR reconstruction for every clip, with explicit failed/unknown checks. Availability and objective-linked screening are not measured 3D accuracy or manual review.'))
 save(ROOT/'release_records.json',report);csvwrite(ROOT/'release_records.csv',report);save(ROOT/'staged_release.json',summary);print(json.dumps(summary,indent=2))

def promote():
 site=ROOT/'site_stage';summary=json.loads((site/'data/gvhmr_release_summary.json').read_text());assert summary['clips']==813 and summary['legacy_fallback_clips']==0
 records=json.loads((ROOT/'release_records.json').read_text());index=json.loads((site/'data/clips.json').read_text());assert len(index['clips'])==813
 audit=json.loads((ROOT/'population_audit.json').read_text());assert audit['summary']['clips']==813
 assert summary['valid_3d_frames']==audit['summary']['frames']
 assert summary['frames']==sum(r['frames'] for r in json.loads((ROOT/'input_manifest.json').read_text())['inputs'])
 audited={r['id']:r['selected_fit_sha256'] for r in audit['results']}
 for c,r in zip(index['clips'],records):
  assert c['id']==r['id'] and sha(site/'data'/c['pose3d'])==r['gzip_sha256']
  assert r['selected_fit_sha256']==audited[c['id']]
  assert sha(CODE/'data'/c['poseRefined'])==r['final_2d_public_sha256']
 archive_dir=CODE/'data/releases/wham-20261028';archive_dir.mkdir(parents=True,exist_ok=True);original=(ROOT/'original_public_index.json').read_bytes();(archive_dir/'original_index.json.gz').write_bytes(gzip.compress(original,compresslevel=9,mtime=0))
 old_summary=CODE/'data/motion_summary.json';archived_summary=archive_dir/'motion_summary.json'
 if not archived_summary.exists():shutil.copy2(old_summary,archived_summary)
 old_index=json.loads(original);archive_records=[]
 for c in old_index['clips']:
  relative=c.get('pose3dRaw')
  if not relative:continue
  source=CODE/'data'/relative;compressed=source.with_suffix(source.suffix+'.gz')
  if source.exists():
   raw=source.read_bytes();compressed.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(compressed.read_bytes())==raw;source.unlink()
  assert compressed.exists();decoded=gzip.decompress(compressed.read_bytes());blob=hashlib.sha1(b'blob '+str(len(decoded)).encode()+b'\0'+decoded).hexdigest();archive_records.append(dict(id=c['id'],original_path=relative,gzip_path=relative+'.gz',decoded_sha256=hashlib.sha256(decoded).hexdigest(),original_git_blob_sha1=blob))
 save(archive_dir/'raw_pose_archive.json',archive_records)
 for relative in [f'data/reconstructions/{FOLDER}',f'data/quality/{FOLDER}']:
  shutil.copytree(site/relative,CODE/relative,dirs_exist_ok=True)
 shutil.copy2(site/'data/gvhmr_release_summary.json',CODE/'data/gvhmr_release_summary.json')
 shutil.copy2(site/'data/motion_summary.json',CODE/'data/motion_summary.json')
 # Commit the pointer switch only after every immutable asset/2D check passes.
 save(CODE/'data/clips.json',index);save(ROOT/'promotion.json',dict(release_id=RELEASE,clips=813,canonical_index_sha256=sha(CODE/'data/clips.json'),raw_archive_clips=len(archive_records),original_2d_changed=False,old_wham_values_losslessly_preserved=True))
 print('PROMOTED LOCAL WORKTREE',813,'GVHMR clips; push/deploy is a separate reviewed step')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--promote',action='store_true');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 if a.promote:promote()
 else:stage(a.limit)
