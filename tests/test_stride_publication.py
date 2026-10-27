"""Validate the actual published estimates, source policy and stride semantics."""
import hashlib,json,math,subprocess,unittest
from pathlib import Path
from urllib.parse import urlparse,parse_qs
SITE=Path(__file__).resolve().parents[1];STUDY=SITE/'studies/stride-20261028'
class StridePublicationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.catalog=json.loads((STUDY/'viewer/catalog.json').read_text());cls.clips=[json.loads((STUDY/'viewer'/f'{r["id"]}.json').read_text()) for r in cls.catalog]
 def test_numeric_only_bundle_and_integrity(self):
  p=json.loads((STUDY/'publication.json').read_text());self.assertFalse(p['original_broadcast_media_included']);self.assertEqual(p['previous_production_poses_replaced'],0)
  actual={str(f.relative_to(STUDY)) for f in STUDY.rglob('*') if f.is_file()};self.assertEqual(actual,set(p['files'])|{'publication.json'})
  for name,digest in p['files'].items():
   file=STUDY/name;self.assertIn(file.suffix,['.json','.csv','.html','.js','.txt']);self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),digest)
  html=(STUDY/'comparison.html').read_text().lower();self.assertNotIn('<video',html);self.assertNotIn('<img',html)
 def test_sample_and_defaults_preserved(self):
  self.assertEqual(len(self.clips),36);self.assertEqual(sum(r['split']=='test' for r in self.catalog),24)
  old=json.loads((SITE/'studies/pitch-20261027/sample.json').read_text());new=json.loads((STUDY/'sample48.json').read_text());self.assertEqual(old,new)
  for z in self.clips:
   self.assertEqual(z['default_model'],'gvhmr_f4');self.assertFalse(z['absolute_stride_validated']);self.assertFalse(z['human_review_complete']);self.assertIn('gvhmr_f4',z['models']);self.assertIn('current_actual',z['models']);self.assertTrue(all(a<b for a,b in zip(z['times'],z['times'][1:])))
   u=urlparse(z['sourceUrl']);self.assertEqual(u.hostname,'www.youtube.com');self.assertEqual(len(parse_qs(u.query)['v'][0]),11)
 def test_stride_is_computed_from_published_native_ankles(self):
  checked=0
  for z in self.clips:
   for name,m in z['models'].items():
    points=m['native_lower_hip_relative_leg'];curve=m['stride']['native_interankle_3d_leg'];self.assertEqual(len(points),len(z['times']));self.assertEqual(len(curve),len(points))
    for p,d in zip(points,curve):
     if d is None or any(v is None for j in [4,5] for v in p[j]):continue
     actual=math.sqrt(sum((a-b)**2 for a,b in zip(p[4],p[5])));self.assertAlmostEqual(actual,d,delta=.0004);checked+=1
    self.assertEqual(m['stride']['plant_interval'],z['phase']['plant_interval'])
    if z['phase']['phase_side_conflict']:self.assertIsNone(m['stride']['plant_interankle_3d_leg_range'])
  self.assertGreater(checked,10000)
 def test_camera_only_preserves_raw_articulation(self):
  for z in self.clips:
   a=z['models']['GVHMR raw f4 SMPL adapter'];b=z['models']['GVHMR f4 camera only'];self.assertIsNone(a['global_fixed'])
   for x,y in zip(a['relative'],b['relative']):
    for xx,yy in zip(x,y):
     for aaa,bbb in zip(xx,yy):
      if aaa is not None and bbb is not None:self.assertAlmostEqual(aaa,bbb,delta=.0004)
 def test_original_813_public_index_unchanged(self):
  old=subprocess.check_output(['git','show','d5ecec895426ac277cde22e71517f69ec4a965b0:data/clips.json'],cwd=SITE);self.assertEqual(old,(SITE/'data/clips.json').read_bytes())
if __name__=='__main__':unittest.main()
