"""Select actual new GVHMR candidates; distinguish availability from screening.

Every finite published preview identifies its actual failed/unknown checks.
No threshold relaxation, WHAM fallback, manual-review transfer, or floor lift.
"""
from common import *
from all813_release import ROOT,population,enable_primary_shot_policy
from select_rear_pivot import RULE as ORIGINAL_RULE
from support_transfer_root import save_geometry,metrics
from scipy.spatial.transform import Rotation
import argparse,shutil

RULE={**ORIGINAL_RULE,'version':'all813-gvhmr-selection-v1','bounded_knees':True,
 'review_preview_selection':'Prefer full frozen screens, then source-and-joint screens, then numerically valid bounded-joint candidates. Failed checks remain explicit.'}

def background_assessment(cid):
 paths=[PREVIOUS/'background'/f'{cid}.json']+[ROOT/f'background_worker{w}/background'/f'{cid}.json' for w in [0,1]]
 for path in paths:
  if path.exists():
   r=json.loads(path.read_text());result={k:r.get(k) for k in ['status','success_ratio','cut_candidates','semantics']}
   if (ROOT/'shot_validity'/f'{cid}.json').exists() and (r.get('success_ratio') or 0)>.7:result.update(source_full_clip_status=r['status'],status='usable_primary_shot_image_stabilization')
   return result
 return dict(status='unknown',success_ratio=None,cut_candidates=[],semantics='No metric camera calibration')

def select(cid):
 enable_primary_shot_policy()
 folder=ROOT/'selected';dest=folder/f'{cid}.npz';report_path=dest.with_suffix('.json');folder.mkdir(exist_ok=True)
 preserved=OUT/'rear_pivot_selected_v14'/f'{cid}.npz';preserved_meta=preserved.with_suffix('.json')
 if preserved.exists() and preserved_meta.exists():
  old=json.loads(preserved_meta.read_text())
  if old.get('status')=='eligible_research_preview':
   if not dest.exists():shutil.copy2(preserved,dest)
   assert sha(dest)==sha(preserved)
   report={**old,'status':'screened_consistency_preview','reused_frozen_v14':True,'release_selector_version':RULE['version'],'selection_code_sha256':sha(__file__),'human_review_complete':False,'selected_pose_sha256':sha(dest),'quality_reasons':['monocular_global_depth_and_absolute_stride_unvalidated'],'contact_available':True,'contact_hypotheses_available':bool(old['marker_states']),'background_assessment':background_assessment(cid)}
   save(report_path,report);return report
 path=ROOT/'contact'/f'{cid}.npz';meta_path=path.with_suffix('.json')
 if not path.exists() or not meta_path.exists():return dict(id=cid,status='contact_fit_not_ready')
 parent_sha=sha(path)
 if dest.exists() and report_path.exists():
  cached=json.loads(report_path.read_text())
  if cached.get('fit_parent_sha256')==parent_sha and cached.get('release_selector_version')==RULE['version'] and cached.get('selection_code_sha256')==sha(__file__):return cached
 z=load(path);meta=json.loads(meta_path.read_text());source=load(ROOT/'base/gvhmr_temporal_f4'/f'{cid}.npz');_,_,_,h,_=init_inputs(cid);leg=float(source['model_leg']);candidates=[]
 for row in meta['hypotheses']:
  i=row['index'];rot=z['all_smpl_rotations'][i];a=z['all_anatomical_keypoints'][i];p=z['all_keypoints'][i];f=z['all_foot_keypoints'][i];tr=z['root_hypotheses_translation'][i]
  finite=all(np.isfinite(v).all() for v in [rot,a,p,f,tr,z['all_focal'][i],z['all_plane_normals'][i],z['all_plane_offsets'][i]])
  if not finite:continue
  knee=Rotation.from_matrix(rot[:,[4,5]].reshape(-1,3,3)).as_rotvec();joint=bool(knee[:,0].min()>=-.0801 and knee[:,0].max()<=2.8001 and abs(knee[:,1:]).max()<=.1801)
  signed=((f+tr[:,None])*z['all_plane_normals'][i]).sum(-1)+z['all_plane_offsets'][i];penetration=np.maximum(-signed,0)/leg
  reference='archived v4 reference plane' if (OUT/'support_transfer_arm_v4'/f'{cid}.npz').exists() else 'F4 base reference plane'
  r={**row,'semantics':'Fixed image-supported contact interval and '+reference+'; camera-depth tangent, not measured field-axis displacement','numeric_finite':finite,'joint_gate':joint,'knee_x_min_rad':float(knee[:,0].min()),'knee_x_max_rad':float(knee[:,0].max()),'floor_penetration_p95_leg':float(np.percentile(penetration,95)),'floor_penetration_max_leg':float(penetration.max())}
  keys=['stride_change_p95_leg','floor_penetration_p95_leg','floor_penetration_max_leg'];contact=r['rear_interval'] is not None
  if contact:keys+=['rear_point_extent_3d_leg','rear_max_retracement_leg']
  r['consistency_gate']=bool(r['source_gate'] and joint and contact and all(r[k]<=RULE[k] for k in keys));r['failed_checks']=[k for k in keys if r[k]>RULE[k]]
  if not r['source_gate']:r['failed_checks'].append('source_reprojection')
  if not joint:r['failed_checks'].append('knee_bounds')
  if not contact:r['failed_checks'].append('rear_contact_unknown')
  candidates.append(r)
 eligible=[r for r in candidates if r['consistency_gate']]
 source_ok=[r for r in candidates if r['source_gate'] and r['joint_gate']]
 bounded=[r for r in candidates if r['joint_gate']]
 if not bounded:return dict(id=cid,status='numerical_or_joint_failure',candidates=candidates)
 if eligible:chosen=min(eligible,key=lambda r:r['source_p50_px'])
 else:
  def score(r):
   source=100 if r.get('source_p50_px') is None else r['source_p50_px'];wrist=100 if r.get('wrist_p95_px') is None else r['wrist_p95_px']
   return r['floor_penetration_p95_leg']*4+(r.get('rear_point_extent_3d_leg') or 0)*2+r['stride_change_p95_leg']*2+source/200+wrist/400
  chosen=min(source_ok or bounded,key=score)
 i=chosen['index'];candidate={**source,'keypoints':z['all_keypoints'][i],'anatomical_keypoints':z['all_anatomical_keypoints'][i],'foot_keypoints':z['all_foot_keypoints'][i],'smpl_rotations':z['all_smpl_rotations'][i],'translation':z['root_hypotheses_translation'][i],'focal':z['all_focal'][i]}
 assert np.array_equal(candidate['smpl_rotations'][:,16:],source['smpl_rotations'][:,16:])
 assert np.array_equal(candidate['frame_index'],h['frame_index']) and np.array_equal(candidate['source_time'],h['source_time'])
 _,uv=metrics(candidate,h,source['background_transforms'],candidate['translation'],z['contact_weight']);save_geometry(dest,candidate,candidate['translation'],uv,z['contact_weight'],z['all_plane_normals'][i],float(z['all_plane_offsets'][i]))
 bg=background_assessment(cid);reasons=chosen['failed_checks']+['monocular_global_depth_and_absolute_stride_unvalidated']
 if bg['status'] not in ['usable_image_stabilization','usable_primary_shot_image_stabilization']:reasons.append('background_stabilization_uncertain')
 if bg.get('cut_candidates'):reasons.append('source_camera_cut_excluded' if (ROOT/'shot_validity'/f'{cid}.json').exists() else 'possible_camera_cut')
 if meta['phase']['phase_side_conflict']:reasons.append('pitch_hand_contact_side_conflict')
 report=dict(id=cid,player=meta['player'],status='screened_consistency_preview' if eligible else 'review_required_preview',selected=i,selected_metrics=chosen,candidates=candidates,rule=RULE,release_selector_version=RULE['version'],selection_code_sha256=sha(__file__),fit_parent_sha256=parent_sha,source_fit_sha256=meta['source_fit_sha256'],selected_pose_sha256=sha(dest),phase=meta['phase'],marker_states=meta['marker_states'],contact_available=bool(chosen['rear_interval'] is not None),contact_hypotheses_available=bool(meta['marker_states']),background_assessment=bg,human_review_complete=False,quality_reasons=reasons,reused_frozen_v14=False)
 save(report_path,report);return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id');a=ap.parse_args();records=[r for r in population() if not a.id or r['id']==a.id];results=[select(r['id']) for r in records];save(ROOT/'selection_results.json',results);csvwrite(ROOT/'selection_summary.csv',[{k:r.get(k) for k in ['id','player','status','selected','quality_reasons','contact_available','reused_frozen_v14']} for r in results]);print('SELECTED',sum(r['status'] in ['screened_consistency_preview','review_required_preview'] for r in results),'/',len(records))

if __name__=='__main__':main()
