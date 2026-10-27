"""Population integrity, raw-source tail errors and sampled exact FK audit.

No fabricated ground truth, physical-time conversion or manual review. Runs
after selection; CPU FK does not compete for inference GPU memory.
"""
from common import *
from all813_release import ROOT,population
from audit_rear_pivot import source_detail
from stride_review import native_leg
from scipy.spatial.transform import Rotation
import argparse

def main():
 import torch
 from pitch_study_fit import StudyFitter
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
 inputs={r['id']:r for r in json.loads((ROOT/'input_manifest.json').read_text())['inputs']};records=population()[:a.limit or None];results=[];parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
 for rec in records:
  cid=rec['id'];path=ROOT/'selected'/f'{cid}.npz'
  if not path.exists():raise FileNotFoundError(cid)
  meta=json.loads(path.with_suffix('.json').read_text());z=load(path);base=load(ROOT/'base/gvhmr_temporal_f4'/f'{cid}.npz');r,c,s,h,w=init_inputs(cid)
  assert sha(DATA/inputs[cid]['final_2d_path'])==inputs[cid]['final_2d_sha256']
  assert np.array_equal(z['frame_index'],h['frame_index']) and np.array_equal(z['source_time'],h['source_time'])
  assert np.array_equal(z['smpl_betas'],base['smpl_betas']);assert np.array_equal(z['smpl_rotations'][:,16:],base['smpl_rotations'][:,16:])
  assert all(np.isfinite(z[k]).all() for k in ['keypoints','anatomical_keypoints','foot_keypoints','translation','smpl_rotations','focal','normal'])
  rot=z['smpl_rotations'];rotation_error=float(abs(np.swapaxes(rot,-1,-2)@rot-np.eye(3)).max());assert rotation_error<.0001
  knee=Rotation.from_matrix(rot[:,[4,5]].reshape(-1,3,3)).as_rotvec();assert knee[:,0].min()>=-.0801 and knee[:,0].max()<=2.8001 and abs(knee[:,1:]).max()<=.1801
  lengths=np.linalg.norm(z['anatomical_keypoints'][:,1:]-z['anatomical_keypoints'][:,parents[1:]],axis=-1);bone_error=float(abs(lengths-np.median(lengths,0)).max());assert bone_error<.00005
  beta=z['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;body.prepare_shape(beta);frames=np.unique([0,len(rot)//2,len(rot)-1])
  with torch.no_grad():p,an,f=body.model(torch.tensor(rot[frames],dtype=torch.float32),torch.tensor(np.repeat(beta[None],len(frames),0),dtype=torch.float32))
  fk=max(float(abs(actual.numpy()-z[k][frames]).max()) for k,actual in [('keypoints',p),('anatomical_keypoints',an),('foot_keypoints',f)]);assert fk<.00001
  current,uv=source_detail(z,h,z['contact_weight']);previous,_=source_detail(base,h,z['contact_weight'])
  report=dict(id=cid,player=rec['player_name'],status=meta['status'],contact_available=meta['contact_available'],consistency_screen_passed=bool(meta['selected_metrics'].get('consistency_gate')),selected_fit_sha256=sha(path),source_2d_sha256=inputs[cid]['final_2d_sha256'],frames=len(rot),fk_sample_frames=frames.tolist(),fk_max_error_model=fk,rotation_orthogonality_error=rotation_error,bone_length_variation_max_model=bone_error,knee_x_min_rad=float(knee[:,0].min()),knee_x_max_rad=float(knee[:,0].max()),before=previous,after=current,quality_reasons=meta['quality_reasons'],human_review_complete=False,absolute_stride_validated=False)
  results.append(report)
  if len(results)%50==0:print('AUDIT',len(results),'/',len(records),flush=True)
 summary=dict(clips=len(results),frames=sum(r['frames'] for r in results),original_final_2d_hashes_unchanged=True,sampled_fk_all_clips=True,fk_frames=sum(len(r['fk_sample_frames']) for r in results),fk_error_max_model=max(r['fk_max_error_model'] for r in results),contact_available=sum(r['contact_available'] for r in results),consistency_screen_passed=sum(r['consistency_screen_passed'] for r in results),source_feet_p95_worse_vs_base=sum(r['after']['feet_p95_px'] is not None and r['before']['feet_p95_px'] is not None and r['after']['feet_p95_px']>r['before']['feet_p95_px'] for r in results),source_feet_max_worse_vs_base=sum(r['after']['feet_p100_px'] is not None and r['before']['feet_p100_px'] is not None and r['after']['feet_p100_px']>r['before']['feet_p100_px'] for r in results),human_manually_reviewed=0,actual_3d_accuracy_claim=False,semantics='Source observations and model consistency; exact FK is numerical integrity, not KBO accuracy')
 save(ROOT/'population_audit.json',dict(summary=summary,results=results));csvwrite(ROOT/'population_audit.csv',[dict(id=r['id'],player=r['player'],status=r['status'],contact_available=r['contact_available'],fk_max_error_model=r['fk_max_error_model'],**{'after_'+k:v for k,v in r['after'].items()}) for r in results]);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
