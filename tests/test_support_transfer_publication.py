"""The separate public pilot must identify real aligned candidates and preserve poses."""
import gzip,hashlib,json,unittest
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]
STUDY=SITE/'studies/support-transfer-20261028'

class SupportTransferPublicationTests(unittest.TestCase):
 def test_manifest_and_coverage(self):
  manifest=json.loads((STUDY/'publication.json').read_text());protocol=json.loads((STUDY/'protocol.json').read_text());catalog=json.loads((STUDY/'viewer/catalog.json').read_text())
  self.assertEqual(manifest['clips'],12);self.assertFalse(manifest['explorer_default_changed']);self.assertFalse(manifest['original_broadcast_media_included']);self.assertFalse(manifest['human_review_complete'])
  self.assertEqual({r['id'] for r in catalog},set(protocol['pilot']+protocol['confirmation']))
  self.assertEqual(sum(r['split']=='pilot6' for r in catalog),6)
  for name,expected in manifest['files'].items():self.assertEqual(hashlib.sha256((STUDY/name).read_bytes()).hexdigest(),expected,name)
  self.assertLess(sum(p.stat().st_size for p in STUDY.rglob('*') if p.is_file()),5_000_000)

 def test_gzip_candidates_align_and_keep_articulation(self):
  catalog=json.loads((STUDY/'viewer/catalog.json').read_text());prior=SITE/'studies/stride-20261028/viewer'
  for clip in catalog:
   data=json.loads(gzip.decompress((STUDY/'viewer'/f'{clip["id"]}.json.gz').read_bytes()));old=json.loads((prior/f'{clip["id"]}.json').read_text());models=data['models'];base=models['gvhmr_f4']
   self.assertEqual(set(models),{'gvhmr_f4','root_temporal_only','root_support_transfer'});self.assertEqual(data['times'],old['times']);self.assertEqual(data['source_times'],old['source_times']);self.assertEqual(data['observations'],old['observations']);self.assertEqual(data['scores'],old['scores']);self.assertFalse(data['human_review_complete']);self.assertFalse(data['absolute_stride_validated'])
   for model in models.values():
    self.assertEqual(model['relative'],base['relative']);self.assertEqual(model['stride']['native_interankle_3d_leg'],base['stride']['native_interankle_3d_leg']);self.assertEqual(len(model['global_fixed']),len(data['times']));self.assertEqual(len(model['projected']),len(data['times']))
   self.assertTrue(data['sourceUrl'].startswith('https://www.youtube.com/watch?v='))
   self.assertNotIn('video',data)

if __name__=='__main__':unittest.main()
