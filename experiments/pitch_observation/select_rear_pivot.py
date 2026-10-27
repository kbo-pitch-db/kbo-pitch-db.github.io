"""Select among saved candidates using source, pivot, stride and floor checks.

No refitting, floor lowering, stride scaling or coordinate correction. This
separates a raw-source-only selector from a grounded consistency selector.
Thresholds are research consistency screens, never measured accuracy bounds.
"""
from common import *
from support_transfer_root import save_geometry,metrics
import argparse
from scipy.spatial.transform import Rotation

RULE=dict(version='rear-pivot-selection-v12',rear_point_extent_3d_leg=.03,
 rear_max_retracement_leg=.03,stride_change_p95_leg=.05,
 floor_penetration_p95_leg=.04,floor_penetration_max_leg=.08,
 ranking='lowest raw source p50 among candidates passing every screen',
 source_gates='original source/wrist and raw feet p95 each no more than +2 px',
 meaning='Objective-linked consistency and source agreement; not KBO 3D accuracy')

PARENT_NAME='rear_pivot_v11'
SELECTED_NAME='rear_pivot_selected_v12'

def select(cid):
 path=OUT/PARENT_NAME/f'{cid}.npz';meta_path=path.with_suffix('.json')
 if not meta_path.exists():return dict(id=cid,status='input_not_available')
 z=load(path);meta=json.loads(meta_path.read_text());source=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz');_,_,_,h,_=init_inputs(cid);leg=float(source['model_leg']);rr=[]
 for row in meta['hypotheses']:
  i=row['index'];foot=z['all_foot_keypoints'][i]+z['root_hypotheses_translation'][i,:,None];signed=(foot*z['all_plane_normals'][i]).sum(-1)+z['all_plane_offsets'][i];penetration=np.maximum(-signed,0)/leg
  knee=Rotation.from_matrix(z['all_smpl_rotations'][i][:,[4,5]].reshape(-1,3,3)).as_rotvec();joint_ok=bool(knee[:,0].min()>=-.0801 and knee[:,0].max()<=2.8001 and np.max(abs(knee[:,1:]))<=.1801)
  r={**row,'joint_gate':joint_ok,'knee_x_min_rad':float(knee[:,0].min()),'knee_x_max_rad':float(knee[:,0].max()),'floor_penetration_p95_leg':float(np.percentile(penetration,95)),'floor_penetration_max_leg':float(np.max(penetration))}
  r['consistency_gate']=bool(r['source_gate'] and (joint_ok or not RULE.get('bounded_knees',False)) and all(r[key]<=RULE[key] for key in ['rear_point_extent_3d_leg','rear_max_retracement_leg','stride_change_p95_leg','floor_penetration_p95_leg','floor_penetration_max_leg']));rr.append(r)
 eligible=[r for r in rr if r['consistency_gate']];report=dict(id=cid,player=meta['player'],status='eligible_research_preview' if eligible else 'withheld_source_or_consistency',candidates=rr,rule=RULE,fit_parent_sha256=sha(path),source_fit_sha256=meta['source_fit_sha256'],phase=meta['phase'],marker_states=meta['marker_states'],human_review_complete=False)
 folder=OUT/SELECTED_NAME;folder.mkdir(exist_ok=True)
 if eligible:
  chosen=min(eligible,key=lambda r:r['source_p50_px']);i=chosen['index'];candidate={**source,'keypoints':z['all_keypoints'][i],'anatomical_keypoints':z['all_anatomical_keypoints'][i],'foot_keypoints':z['all_foot_keypoints'][i],'smpl_rotations':z['all_smpl_rotations'][i],'translation':z['root_hypotheses_translation'][i],'focal':z['all_focal'][i]};_,uv=metrics(candidate,h,source['background_transforms'],candidate['translation'],z['contact_weight']);dest=folder/f'{cid}.npz';save_geometry(dest,candidate,candidate['translation'],uv,z['contact_weight'],z['all_plane_normals'][i],float(z['all_plane_offsets'][i]));report.update(selected=i,selected_metrics=chosen,selected_pose_sha256=sha(dest),contact_available=True)
 save(folder/f'{cid}.json',report);return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id');args=ap.parse_args();ids=[args.id] if args.id else [r['id'] for r in sample('tune',True)];folder=OUT/SELECTED_NAME;folder.mkdir(exist_ok=True);save(folder/'protocol.json',dict(rule=RULE,selector_sha256=sha(__file__),test24_used=False));rows_=[select(cid) for cid in ids];save(folder/'results.json',rows_)
 for r in rows_:print(r['id'],r['status'],r.get('selected'))

if __name__=='__main__':main()
