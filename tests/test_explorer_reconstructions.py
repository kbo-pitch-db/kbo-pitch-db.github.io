"""Explorer assets must match the selected fits and immutable final 2D timing."""
import gzip,json,math,statistics,unittest
from pathlib import Path
SITE=Path(__file__).resolve().parents[1];DATA=SITE/'data'
class ExplorerReconstructionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.registry=json.loads((DATA/'reconstruction_registry.json').read_text());cls.clips={c['id']:c for c in json.loads((DATA/'clips.json').read_text())['clips']}
 def test_coverage_and_explicit_provenance(self):
  self.assertEqual(len(self.registry['clips']),36);self.assertEqual(self.registry['coverage'],dict(total=813,latest=36,nlf=24,legacy_only=777));self.assertNotIn('1e24233962b6_c0018',self.registry['clips'])
  for cid,entry in self.registry['clips'].items():
   self.assertEqual(entry['preferred'],'gvhmr_f4');self.assertEqual(entry['legacy_pose3d'],self.clips[cid]['pose3d']);self.assertNotEqual(entry['variants']['gvhmr_f4']['pose3d'],self.clips[cid]['pose3d'])
 def test_actual_pose_timing_and_stride_match_the_reviewed_results(self):
  checked=0
  for cid,entry in self.registry['clips'].items():
   clip=self.clips[cid];two=json.loads((DATA/clip['poseRefined']).read_text());study=json.loads((SITE/'studies/stride-20261028/viewer'/f'{cid}.json').read_text())
   for method,v in entry['variants'].items():
    pose=json.loads(gzip.decompress((DATA/v['pose3d']).read_bytes()));self.assertEqual(pose['method'],method);self.assertEqual(pose['source_fit_sha256'],v['sourceFitSha256']);self.assertEqual(pose['fps'],two['fps']);self.assertEqual(len(pose['frames']),len(two['frames']));self.assertFalse(pose['scene']['show_floor']);self.assertIsNone(pose['scene']['floor']);self.assertFalse(pose['scene']['per_frame_recentering']);self.assertFalse(pose['quality']['human_review_complete']);self.assertFalse(pose['quality']['stride_validated'])
    curve=study['models']['gvhmr_f4' if method=='gvhmr_f4' else 'NLF native shape + fitting']['stride']['native_interankle_3d_leg'];leg=[]
    for f,g in zip(pose['frames'],two['frames']):
     self.assertEqual([f[k] for k in ['frame','source_frame','time','source_time']],[g[k] for k in ['frame','source_frame','time','source_time']]);self.assertEqual(len(f['anatomy']),24);self.assertEqual(len(f['feet']),6)
     p=f['anatomy'];d=lambda a,b:math.dist(p[a],p[b]);leg.append((d(1,4)+d(4,7)+d(2,5)+d(5,8))/2)
    scale=statistics.median(leg)
    for f,expected in zip(pose['frames'],curve):self.assertAlmostEqual(math.dist(f['anatomy'][7],f['anatomy'][8])/scale,expected,delta=.0002)
    self.assertLess(statistics.pstdev(leg)/scale,.00002);checked+=1
  self.assertEqual(checked,60)
if __name__=='__main__':unittest.main()
