"""Check actual saved combined coordinates, not just model labels/availability."""
import gzip,hashlib,json,unittest
from pathlib import Path
import numpy as np
from release_fixture import support_index_bytes

SITE=Path(__file__).resolve().parents[1]
PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])

class CombinedPopulationReleaseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.index=json.loads((SITE/'data/clips.json').read_text())
  cls.old={c['id']:c for c in json.loads(support_index_bytes())['clips']}
  cls.summary=json.loads((SITE/'data/combined_release_summary.json').read_text())
  added=SITE/'data/releases/combined-20261029-increment-v1/quality_records.json'
  cls.added={r['id']:r for r in json.loads(added.read_text())} if added.exists() else {}

 def test_all_actual_combined_coordinates_and_serialized_support_geometry(self):
  self.assertEqual(len(self.index['clips']),self.summary['clips'])
  self.assertTrue(set(self.old)<={c['id'] for c in self.index['clips']})
  self.assertEqual(len({c['playerKey'] for c in self.index['clips']}),self.summary['players'])
  frames=valid=source_pass=0;flags={}
  for c in self.index['clips']:
   with self.subTest(clip=c['id']):
    old=self.old.get(c['id'])
    if old and c['poseRefined']!=old['poseRefined']:
     recovery=json.loads((SITE/'data'/c['subjectCorrection']).read_text())
     self.assertEqual(recovery['version'],'subject-tail-v1')
     self.assertEqual(recovery['old_clip']['poseRefined'],old['poseRefined'])
     self.assertEqual(recovery['new_2d'],c['poseRefined'])
     self.assertEqual(hashlib.sha256((SITE/'data'/old['poseRefined']).read_bytes()).hexdigest(),recovery['original_2d_sha256'])
     before=json.loads((SITE/'data'/old['poseRefined']).read_text());after=json.loads((SITE/'data'/c['poseRefined']).read_text())
     self.assertEqual(before['frames'][:recovery['recovery_start_frame']],after['frames'][:recovery['recovery_start_frame']])
     self.assertEqual(hashlib.sha256((SITE/'data'/c['pose3d']).read_bytes()).hexdigest(),recovery['new_3d_sha256'])
    elif old:
     for key in ['pose','poseRefined']:self.assertEqual(c.get(key),old.get(key))
    else:
     self.assertEqual(hashlib.sha256((SITE/'data'/c['poseRefined']).read_bytes()).hexdigest(),self.added[c['id']]['final_2d_public_sha256'])
    self.assertEqual(c['pose3dBackend'],'gvhmr-motionbert')
    self.assertEqual(c['motionMethod'],'gvhmr-ground-motionbert-v1')
    if old:self.assertNotEqual(c['pose3d'],old['pose3d'])
    pose=json.loads(gzip.decompress((SITE/'data'/c['pose3d']).read_bytes()))
    q=json.loads((SITE/'data'/c['motionQuality']).read_text())
    two=json.loads((SITE/'data'/c['poseRefined']).read_text())
    previous=json.loads(gzip.decompress((SITE/'data'/old['pose3d']).read_bytes())) if old else None
    self.assertEqual(pose['source_fit_sha256'],c['sceneSignature'])
    if previous:self.assertNotEqual(pose['source_fit_sha256'],previous['source_fit_sha256'])
    self.assertEqual(pose['method'],c['motionMethod'])
    self.assertTrue(pose['fit_provenance']['motionbert_applied'])
    self.assertTrue(pose['fit_provenance']['both_arms_processed'])
    self.assertTrue(pose['fit_provenance']['additional_refinements']['rigid_sole_refit']['constant_shape'])
    for key in ['motionbert_pretrained_checkpoint_sha256','motionbert_adapted_checkpoint_sha256','actual_motionbert_prediction_sha256','rigid_shoe_solver_sha256']:
     self.assertRegex(pose['model_components'][key],r'^[0-9a-f]{64}$')
    self.assertFalse(pose['scene']['per_frame_recentering']);self.assertTrue(pose['scene']['show_floor'])
    self.assertEqual(pose['scene']['floor']['height'],0.)
    for key in ['human_review_complete','stride_validated','contact_validated','global_trajectory_validated','benchmark_eligible']:
     self.assertFalse(pose['quality'][key])
    self.assertTrue(all(v for k,v in q['physical_checks'].items() if k!='root_step_diagnostic'))
    self.assertEqual(q['strict_source_pass'],all(q['source_checks'].values()))
    self.assertEqual(len(pose['frames']),len(two['frames']))
    self.assertEqual(pose['fps'],two['fps']);n=pose['quality']['valid_3d_frames'];valid+=n;frames+=len(two['frames'])
    for i,(a,b) in enumerate(zip(pose['frames'],two['frames'])):
     self.assertEqual([a[k] for k in ['source_frame','source_time','time']],[b[k] for k in ['source_frame','source_time','time']])
     if i>=n:
      self.assertFalse(a['available']);self.assertEqual(a['unavailable_reason'],'source_camera_change')
      self.assertTrue(all(p is None for p in a['anatomy']+a['feet']))
      self.assertTrue(pose['shot_validity']['confirmed_source_cut'])
    anatomy=np.array([f['anatomy'] for f in pose['frames'][:n]])
    feet=np.array([f['feet'] for f in pose['frames'][:n]])
    self.assertTrue(np.isfinite(anatomy).all() and np.isfinite(feet).all())
    lengths=np.linalg.norm(anatomy[:,1:]-anatomy[:,PARENTS[1:]],axis=-1)
    self.assertLess(np.ptp(lengths,axis=0).max(),.000025)
    for side in range(2):
     for u,v in [(side,side+2),(side,side+4),(side+2,side+4)]:
      self.assertLess(np.ptp(np.linalg.norm(feet[:,u]-feet[:,v],axis=-1)),.000025)
    leg=q['normalization_leg_model'];diameter=0.
    for e in pose['fit_provenance']['contact_episodes']:
     p=feet[e['start']:e['end']+1,e['marker']]
     diameter=max(diameter,np.linalg.norm(p[:,None]-p[None],axis=-1).max()/leg)
    self.assertLessEqual(diameter,q['physical_rule']['contact_diameter_max_leg']+.00001)
    self.assertLessEqual(feet[:,:,1].min(-1).max()/leg,q['physical_rule']['minimum_foot_height_max_leg']+.00001)
    self.assertLessEqual(np.maximum(-anatomy[:,:,1],0).max()/leg,q['physical_rule']['floor_penetration_max_leg']+.00001)
    penetration=np.maximum(-feet[...,1],0)/leg
    self.assertLessEqual(np.percentile(penetration,95),q['physical_rule']['floor_penetration_p95_leg']+.00001)
    self.assertLessEqual(penetration.max(),q['physical_rule']['floor_penetration_max_leg']+.00001)
    # A rigid triangle determines shoe orientation; recompute rotation changes
    # from published foot points instead of trusting a renderer or label.
    from scipy.spatial.transform import Rotation
    dt30=np.diff([f['source_time'] for f in pose['frames'][:n]])*30
    ankle_velocity=np.diff(anatomy[:,[7,8]],axis=0)/dt30[:,None,None]/leg
    ankle_change=np.diff(ankle_velocity,axis=0)/((dt30[1:]+dt30[:-1])/2)[:,None,None]
    self.assertLessEqual(np.linalg.norm(ankle_change,axis=-1).max(),q['physical_rule']['ankle_velocity_change_at30fps_max_leg']+.00005)
    for side in range(2):
     triangle=feet[:,[side,side+2,side+4]];triangle-=triangle.mean(1,keepdims=True)
     U,S,Vt=np.linalg.svd(np.einsum('tki,tkj->tij',triangle[:-1],triangle[1:]))
     R=Vt.swapaxes(-1,-2)@U.swapaxes(-1,-2)
     negative=np.linalg.det(R)<0;Vt[negative,-1]*=-1;R=Vt.swapaxes(-1,-2)@U.swapaxes(-1,-2)
     angular=Rotation.from_matrix(R).magnitude()/dt30
     self.assertLessEqual(angular.max(),q['physical_rule']['shoe_rotation_step_at30fps_max_radians']+.002)
    velocity=np.diff(anatomy[:,0],axis=0)/dt30[:,None]/leg
    change=np.diff(velocity,axis=0)/((dt30[1:]+dt30[:-1])/2)[:,None]
    self.assertLessEqual(np.linalg.norm(change,axis=-1).max(),q['physical_rule']['root_velocity_change_at30fps_max_leg']+.00005)
    source_pass+=q['strict_source_pass']
    for flag in q['source_regression_flags']:flags[flag]=flags.get(flag,0)+1
  self.assertEqual(frames,self.summary['frames']);self.assertEqual(valid,self.summary['valid_3d_frames'])
  self.assertEqual(source_pass,self.summary['strict_source_pass'])
  self.assertEqual(self.summary['contact_consistency_failed'],0)
  self.assertEqual(self.summary['constant_shoe_shape_clips'],len(self.index['clips']))
  self.assertEqual(len(self.summary['active_final_2d_recovered_clips']),8)
  self.assertFalse(self.summary['original_final_2d_assets_overwritten'])
  for key,count in self.summary['source_regression_counts'].items():self.assertEqual(count,flags.get(key+'_regression',0))

 def test_historical_wham_scene_bytes_are_losslessly_archived(self):
  archive=json.loads((SITE/'data/releases/wham-20261028/scene_pose_archive.json').read_text())
  self.assertEqual(len(archive),813)
  for row in archive:
   data=gzip.decompress((SITE/'data'/row['gzip_path']).read_bytes())
   self.assertEqual(hashlib.sha256(data).hexdigest(),row['decoded_sha256'])
   self.assertEqual(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),row['original_git_blob_sha1'])

if __name__=='__main__':unittest.main()
