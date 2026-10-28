"""A wrong-person correction must version its inputs and preserve source time."""
import gzip,hashlib,json,unittest
from pathlib import Path
import numpy as np
SITE=Path(__file__).resolve().parents[1];DATA=SITE/'data';CID='4e09dd724a8c_c0011'

class SubjectCorrectionTests(unittest.TestCase):
 def test_actual_recovery_preserves_old_assets_and_tracks_a_distinct_foreground_person(self):
  c=next(c for c in json.loads((DATA/'clips.json').read_text())['clips'] if c['id']==CID);r=json.loads((DATA/c['subjectCorrection']).read_text());old=r['old_clip']
  for path,key in [(old['poseRefined'],'original_2d_sha256'),(old['pose'],'original_raw_2d_sha256'),(old['pose3d'],'original_3d_sha256'),(c['poseRefined'],'new_2d_sha256'),(c['pose3d'],'new_3d_sha256')]:self.assertEqual(hashlib.sha256((DATA/path).read_bytes()).hexdigest(),r[key])
  before=json.loads((DATA/old['poseRefined']).read_text());after=json.loads((DATA/c['poseRefined']).read_text());pose=json.loads(gzip.decompress((DATA/c['pose3d']).read_bytes()));self.assertEqual(len(after['frames']),75)
  for a,b,d in zip(before['frames'],after['frames'],pose['frames']):
   for k in ['source_frame','source_time','time']:self.assertEqual(a[k],b[k]);self.assertEqual(b[k],d[k])
  # Original 456x836 crop starts at (576,84); corrected input is full-frame.
  old_shoulder=np.array([[(f['keypoints'][j][1]+84) for j in [5,6] if f['keypoints'][j]] for f in before['frames']],dtype=object)
  new_shoulder=[np.mean([f['keypoints'][j][1] for j in [5,6] if f['keypoints'][j]]) for f in after['frames']]
  separation=[b-np.mean(list(a)) for a,b in zip(old_shoulder,new_shoulder) if len(a)]
  self.assertGreater(np.median(separation),200)
  self.assertEqual(r['input_provenance']['detector_supported_frames'],75);self.assertFalse(r['original_batter_shape_or_pose_used']);self.assertLess(r['native_fk_max_error_model'],1e-5)
  self.assertFalse(r['complete_delivery']);self.assertFalse(r['contact_available']);self.assertFalse(r['human_review_complete']);self.assertFalse(pose['scene']['per_frame_recentering']);self.assertEqual(pose['scene']['floor']['height'],0)
  self.assertIsNone(pose['fit_provenance']['phase']['events']['lift_interval']);self.assertIsNone(pose['fit_provenance']['phase']['plant_interval']);self.assertFalse(pose['fit_provenance']['contact_episodes'])
  registry=json.loads((DATA/'subject_corrections.json').read_text());self.assertEqual(registry['clips'][CID]['new_3d'],c['pose3d'])

if __name__=='__main__':unittest.main()
