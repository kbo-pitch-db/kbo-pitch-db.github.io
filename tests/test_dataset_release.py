"""Public dataset boundaries and explicitly requested, rigid-scene previews."""
import gzip,hashlib,json,math,unittest
from pathlib import Path
from html.parser import HTMLParser

SITE=Path(__file__).resolve().parents[1]
class Page(HTMLParser):
 def __init__(self,text):super().__init__();self.tags=[];self.feed(text)
 def handle_starttag(self,tag,attrs):self.tags.append((tag,dict(attrs)))

class DatasetReleaseTests(unittest.TestCase):
 def test_public_page_has_no_experiment_controls_or_links(self):
  tags=Page((SITE/'index.html').read_text()).tags
  self.assertFalse(any(a.get('id') in ['only-current-3d','only-3d','reconstruction-method'] for _,a in tags))
  self.assertFalse(any(('studies/' in a.get('href','') or a.get('href')=='reliability.html') for _,a in tags))
  for path in (SITE/'studies').rglob('*.html'):
   self.assertTrue(any(t=='meta' and a.get('name')=='robots' and 'noindex' in a.get('content','') for t,a in Page(path.read_text()).tags),str(path))
  self.assertIn('Disallow: /studies/',(SITE/'robots.txt').read_text())
  self.assertEqual(hashlib.sha256((SITE/'data/clips.json').read_bytes()).hexdigest(),'8617ba492c8ba06e08595bc0351afb0d55ed601d7fd215821beece325aa45b16')

 def test_preview_ground_is_one_rigid_transform_and_articulation_is_preserved(self):
  registry=json.loads((SITE/'data/preview_registry.json').read_text());clips={c['id']:c for c in json.loads((SITE/'data/clips.json').read_text())['clips']}
  self.assertFalse(registry['published_release_changed']);self.assertEqual(len(registry['clips']),10)
  for cid,entry in registry['clips'].items():
   self.assertEqual(entry['preferred'],'support_v4');v=entry['variants']['support_v4'];raw=(SITE/'data'/v['pose3d']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),v['sha256_gzip']);pose=json.loads(gzip.decompress(raw));two=json.loads((SITE/'data'/clips[cid]['poseRefined']).read_text());scene=pose['scene'];r=scene['rotation_camera_to_scene']
   self.assertTrue(scene['show_floor']);self.assertEqual(scene['floor']['height'],0);self.assertFalse(scene['per_frame_recentering']);self.assertFalse(pose['quality']['human_review_complete']);self.assertFalse(pose['quality']['stride_validated'])
   for a in range(3):
    for b in range(3):self.assertAlmostEqual(sum(r[a][k]*r[b][k] for k in range(3)),float(a==b),places=6)
   ref=json.loads((SITE/'studies/stride-20261028/viewer'/f'{cid}.json').read_text())['models']['gvhmr_f4']['stride'];leg=ref['modeled_leg_length']
   self.assertEqual(len(pose['frames']),len(two['frames']));self.assertEqual(pose['fps'],two['fps'])
   for i,(f,g) in enumerate(zip(pose['frames'],two['frames'])):
    self.assertEqual([f[k] for k in ['source_frame','source_time','time']],[g[k] for k in ['source_frame','source_time','time']]);self.assertAlmostEqual(math.dist(f['anatomy'][7],f['anatomy'][8])/leg,ref['native_interankle_3d_leg'][i],delta=.00025)
   if cid=='3fdcd2dbaf58_c0014':
    # Regression for the inspected example: the stored endpoint is distinct
    # and forward in camera depth. This is not a generic monotonic-root rule.
    delta=[pose['frames'][-1]['anatomy'][0][k]-pose['frames'][0]['anatomy'][0][k] for k in range(3)]
    camera_depth=sum(delta[k]*r[k][2] for k in range(3));self.assertGreater(camera_depth,.4)

if __name__=='__main__':unittest.main()
