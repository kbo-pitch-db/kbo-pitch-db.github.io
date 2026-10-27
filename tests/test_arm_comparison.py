"""Comparison packaging must preserve timing and actual latest GVHMR motion."""
import gzip,hashlib,json,math,unittest
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]
PAGE=SITE/'studies/arm-comparison-20261028'

class ArmComparisonTests(unittest.TestCase):
 def test_exact_timing_shared_bounds_and_constant_display_transforms(self):
  catalog=json.loads((PAGE/'catalog.json').read_text());registry=json.loads((SITE/'data/preview_rear_pivot_registry.json').read_text());clips={r['id']:r for r in json.loads((SITE/'data/clips.json').read_text())['clips']}
  self.assertEqual(len(catalog),6)
  for r in catalog:
   raw=(PAGE/r['file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),r['sha256_gzip']);data=json.loads(gzip.decompress(raw));self.assertEqual(set(data['models']),{'nlf_mb','gvhmr'})
   two=json.loads((SITE/'data'/clips[r['id']]['poseRefined']).read_text());self.assertEqual(data['times'],[f['time'] for f in two['frames']])
   entry=registry['clips'][r['id']];ref=json.loads(gzip.decompress((SITE/'data'/entry['variants'][entry['preferred']]['pose3d']).read_bytes()))
   self.assertFalse(data['human_review_complete']);self.assertFalse(data['absolute_stride_validated'])
   a=ref['frames'][0]['anatomy'];leg=(math.dist(a[1],a[4])+math.dist(a[4],a[7])+math.dist(a[2],a[5])+math.dist(a[5],a[8]))/2
   self.assertEqual(data['models']['nlf_mb']['method'],'nlf_mb_hybrid_temporal_f4')
   for m in data['models'].values():
    self.assertEqual(m['scene']['comparison_bounds'],data['comparison_bounds']);self.assertFalse(m['display_transform']['per_frame_recentering']);self.assertEqual(m['scene']['floor']['height'],0)
    self.assertEqual([f['source_frame'] for f in m['frames']],[f['source_frame'] for f in two['frames']]);self.assertEqual([f['source_time'] for f in m['frames']],[f['source_time'] for f in two['frames']])
   # Only a single constant origin/scale change is allowed. Verify complete
   # articulation and root trajectories against the existing v14 release.
   new=data['models']['gvhmr'];rear=data['rear_marker'];start=data['rear_interval'][0];anchor=ref['frames'][start]['feet'][rear]
   for f,g in zip(new['frames'],ref['frames']):
    for field in ['anatomy','feet']:
     for p,q in zip(f[field],g[field]):
      for k in range(3):self.assertAlmostEqual(p[k],(q[k]-anchor[k])/leg,delta=.00002)

 def test_unlisted_and_no_broadcast_media(self):
  html=(PAGE/'index.html').read_text();self.assertIn('noindex,nofollow,noarchive',html)
  self.assertFalse(any(p.suffix in ['.mp4','.jpg','.png','.npz','.pkl'] for p in PAGE.rglob('*')))
  provenance=json.loads((PAGE/'provenance.json').read_text());self.assertFalse(provenance['new_inference_or_fitting']);self.assertFalse(provenance['published_default_changed'])
  self.assertNotIn('arm-comparison',(SITE/'index.html').read_text())

if __name__=='__main__':unittest.main()
