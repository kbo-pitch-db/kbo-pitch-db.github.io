"""FK-audited toe support previews remain explicit, aligned and unscaled."""
import gzip,hashlib,json,math,unittest
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]

class RearPivotPublicationTests(unittest.TestCase):
 def test_selected_geometry_source_and_public_boundaries(self):
  registry=json.loads((SITE/'data/preview_rear_pivot_registry.json').read_text())
  audit=json.loads((SITE/'studies/rear-pivot-20261028/results.json').read_text())
  reports={r['id']:r for r in audit['results']};clips={r['id']:r for r in json.loads((SITE/'data/clips.json').read_text())['clips']}
  self.assertEqual(len(registry['clips']),6);self.assertEqual(set(registry['clips']),set(reports))
  self.assertFalse(registry['published_release_changed']);self.assertFalse(audit['test24_used'])
  self.assertEqual(hashlib.sha256((SITE/'data/clips.json').read_bytes()).hexdigest(),'8617ba492c8ba06e08595bc0351afb0d55ed601d7fd215821beece325aa45b16')
  for cid,entry in registry['clips'].items():
   v=entry['variants'][entry['preferred']];raw=(SITE/'data'/v['pose3d']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),v['sha256_gzip']);pose=json.loads(gzip.decompress(raw));r=reports[cid];m=r['selected_metrics'];scene=pose['scene']
   self.assertEqual(pose['method'],'gvhmr-rear-pivot-v14');self.assertTrue(scene['show_floor']);self.assertEqual(scene['floor']['height'],0);self.assertFalse(scene['per_frame_recentering']);self.assertTrue(m['joint_gate']);self.assertTrue(m['consistency_gate'])
   for k in ['human_review_complete','stride_validated','global_trajectory_validated','contact_validated']:self.assertFalse(pose['quality'][k])
   self.assertLess(max(r['fk_max_error_model'].values()),2e-5);self.assertLess(r['native_bone_length_max_change_model'],2e-5)
   for metric in ['source_p50_px','wrist_p95_px','feet_p95_px']:self.assertLessEqual(r['v14'][metric],r['f4'][metric]+2)
   self.assertLessEqual(m['stride_change_p95_leg'],.05);self.assertGreaterEqual(m['knee_x_min_rad'],-.0801);self.assertLessEqual(m['knee_x_max_rad'],2.8001)
   two=json.loads((SITE/'data'/clips[cid]['poseRefined']).read_text());self.assertEqual(pose['fps'],two['fps']);self.assertEqual(len(pose['frames']),len(two['frames']))
   for a,b in zip(pose['frames'],two['frames']):self.assertEqual([a[k] for k in ['time','source_time','source_frame']],[b[k] for k in ['time','source_time','source_frame']])
   # The renderer receives an already stationary toe. A rigid scene transform
   # cannot conceal skating; check directly in the serialized geometry.
   start,end=r['v14']['rear_interval'];j=r['v14']['rear_marker'];foot0=pose['frames'][start]['feet'][j]
   a=pose['frames'][0]['anatomy'];leg=(math.dist(a[1],a[4])+math.dist(a[4],a[7])+math.dist(a[2],a[5])+math.dist(a[5],a[8]))/2
   extent=max(math.dist(f['feet'][j],foot0) for f in pose['frames'][start:end+1])/leg
   self.assertLessEqual(extent,.03001);self.assertAlmostEqual(extent,r['v14']['rear_point_extent_3d_leg'],delta=.00002)
   self.assertLess(extent,r['v4']['rear_point_extent_3d_leg'])
   if cid in ['3fdcd2dbaf58_c0014','d30b22cfc0b0_c0036']:
    delta=[pose['frames'][-1]['anatomy'][0][k]-pose['frames'][0]['anatomy'][0][k] for k in range(3)]
    camera_depth=sum(delta[k]*scene['rotation_camera_to_scene'][k][2] for k in range(3));self.assertGreater(camera_depth,.4)

 def test_tail_regressions_are_disclosed(self):
  audit=json.loads((SITE/'studies/rear-pivot-20261028/results.json').read_text())
  self.assertTrue(audit['source_foot_p100_worse_clips'])
  self.assertIn('noindex',(SITE/'studies/rear-pivot-20261028/index.html').read_text())
  self.assertIn('Disallow: /data/reconstructions/rear-pivot-v14/',(SITE/'robots.txt').read_text())

if __name__=='__main__':unittest.main()
