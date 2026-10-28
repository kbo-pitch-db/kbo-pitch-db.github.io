"""Published preview invariants: exact main body, timing and shared view."""
import gzip,hashlib,json,unittest
from pathlib import Path
import numpy as np
from release_fixture import population_index_bytes
SITE=Path(__file__).resolve().parents[1];PAGE=SITE/'studies/arm-cocking-20261028'

class ArmCockingPreviewTests(unittest.TestCase):
 def test_actual_payloads_preserve_main_body_feet_timing_and_camera_scope(self):
  catalog=json.loads((PAGE/'catalog.json').read_text());self.assertEqual(len(catalog),12);main={r['id']:r for r in json.loads((SITE/'data/clips.json').read_text())['clips']};p=json.loads((PAGE/'provenance.json').read_text());self.assertEqual(p['main_index_sha256'],hashlib.sha256(population_index_bytes()).hexdigest());self.assertFalse(p['main_release_changed']);self.assertFalse(p['full_smpl_surface_pose_correctives']);self.assertTrue(p['foot_markers_frozen_by_design']);self.assertTrue(p['wrist_flexion_angle_not_observed'])
  for rec in catalog:
   raw=(PAGE/rec['file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),rec['sha256_gzip']);d=json.loads(gzip.decompress(raw));self.assertEqual(len(d['models']),16);self.assertFalse(d['human_review_complete']);self.assertFalse(d['display_transform']['per_frame_recentering']);two=json.loads((SITE/'data'/main[rec['id']]['poseRefined']).read_text());self.assertEqual(d['source_frames'],[f['source_frame'] for f in two['frames']]);self.assertEqual(d['source_times'],[f['source_time'] for f in two['frames']]);self.assertEqual(d['times'],[f['time'] for f in two['frames']]);b=d['models']['current_GVHMR'];an=np.array([f['anatomy'] for f in b['frames']]);feet=np.array([f['feet'] for f in b['frames']]);published=json.loads(gzip.decompress((SITE/'data'/main[rec['id']]['pose3d']).read_bytes()));expected=np.array([f['anatomy'] for f in published['frames']])/d['display_transform']['leg_divisor'];self.assertLess(float(abs(an-expected).max()),.00003)
   for model in d['models'].values():
    a=np.array([f['anatomy'] for f in model['frames']]);f=np.array([f['feet'] for f in model['frames']]);self.assertTrue(np.isfinite(a).all());self.assertTrue(np.isfinite(f).all());np.testing.assert_array_equal(a[:,:18],an[:,:18]);np.testing.assert_array_equal(f,feet);self.assertEqual(model['scene']['comparison_bounds'],b['scene']['comparison_bounds']);self.assertEqual(model['scene']['floor']['height'],0);self.assertFalse(model['scene']['per_frame_recentering'])
 def test_independent_player_source_split_and_unlisted_boundary(self):
  p=json.loads((PAGE/'protocol.json').read_text());main={r['id']:r for r in json.loads((SITE/'data/clips.json').read_text())['clips']};self.assertEqual(len(p['tune']),6);self.assertEqual(len(p['confirmation']),6);self.assertFalse({main[x]['playerKey'] for x in p['tune']}&{main[x]['playerKey'] for x in p['confirmation']});self.assertFalse({x.split('_c')[0] for x in p['tune']}&{x.split('_c')[0] for x in p['confirmation']});selection=json.loads((PAGE/'selection.json').read_text());self.assertTrue(selection['confirmation_not_used_for_selection']);self.assertFalse(selection['unique_3d_solution_identified']);self.assertFalse(selection['adopted_for_main']);self.assertNotIn('arm-cocking',(SITE/'index.html').read_text());self.assertIn('noindex,nofollow,noarchive',(PAGE/'index.html').read_text());self.assertFalse(any(p.suffix in ['.png','.jpg','.jpeg','.mp4','.npz','.pt','.pkl'] for p in PAGE.rglob('*')))

if __name__=='__main__':unittest.main()
