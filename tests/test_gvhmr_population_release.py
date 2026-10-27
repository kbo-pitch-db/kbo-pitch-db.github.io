"""The requested main migration must include every actual GVHMR pose and 2D."""
import gzip,hashlib,json,math,subprocess,unittest
from pathlib import Path
from release_fixture import original_clips,original_index_bytes
SITE=Path(__file__).resolve().parents[1]
BASELINE='7d24094ad6aec0d3518864bdf820c7c87c4427ef'

class GvhmrPopulationReleaseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.index=json.loads((SITE/'data/clips.json').read_text());cls.old={c['id']:c for c in original_clips()}
  cls.summary=json.loads((SITE/'data/gvhmr_release_summary.json').read_text())
  tree=subprocess.check_output(['git','ls-tree','-r',BASELINE,'data/assets'],cwd=SITE,text=True)
  cls.blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}

 def test_every_current_clip_is_new_actual_gvhmr_and_final_2d_is_immutable(self):
  self.assertEqual(len(self.index['clips']),813);self.assertEqual(self.summary['gvhmr_clips'],813);self.assertEqual(self.summary['legacy_fallback_clips'],0);self.assertEqual(self.summary['human_manually_reviewed'],0)
  count=0
  for c in self.index['clips']:
   old=self.old[c['id']];self.assertEqual(c['playerKey'],old['playerKey']);self.assertEqual(c['poseRefined'],old['poseRefined']);self.assertEqual(c['pose3dBackend'],'gvhmr');self.assertEqual(c['motionMethod'],'gvhmr-phase-aware-contact');self.assertNotEqual(c['pose3d'],old['pose3d'])
   raw=(SITE/'data'/c['poseRefined']).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();self.assertEqual(blob,self.blobs['data/'+c['poseRefined']]);two=json.loads(raw)
   pose=json.loads(gzip.decompress((SITE/'data'/c['pose3d']).read_bytes()));self.assertEqual(pose['method'],c['motionMethod']);self.assertEqual(pose['source_fit_sha256'],c['sceneSignature']);self.assertEqual(len(pose['frames']),len(two['frames']));self.assertEqual(pose['fps'],two['fps']);self.assertFalse(pose['scene']['per_frame_recentering']);self.assertTrue(pose['scene']['show_floor']);self.assertEqual(pose['scene']['floor']['height'],0)
   for field in ['human_review_complete','stride_validated','global_trajectory_validated','contact_validated']:self.assertFalse(pose['quality'][field])
   for a,b in zip(pose['frames'],two['frames']):
    self.assertEqual([a[k] for k in ['source_frame','source_time','time']],[b[k] for k in ['source_frame','source_time','time']]);self.assertEqual(len(a['anatomy']),24);self.assertEqual(len(a['feet']),6)
    self.assertTrue(all(math.isfinite(v) for p in a['anatomy']+a['feet'] for v in p))
   count+=len(pose['frames'])
  self.assertEqual(count,self.summary['frames'])

 def test_old_raw_values_are_losslessly_preserved_and_six_previews_do_not_regress(self):
  self.assertEqual(hashlib.sha256(original_index_bytes()).hexdigest(),'8617ba492c8ba06e08595bc0351afb0d55ed601d7fd215821beece325aa45b16')
  archive=json.loads((SITE/'data/releases/wham-20261028/raw_pose_archive.json').read_text());self.assertEqual(len(archive),813)
  for r in archive:
   raw=gzip.decompress((SITE/'data'/r['gzip_path']).read_bytes());blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();self.assertEqual(blob,self.blobs['data/'+r['original_path']]);self.assertEqual(blob,r['original_git_blob_sha1'])
  clips={c['id']:c for c in self.index['clips']};registry=json.loads((SITE/'data/preview_rear_pivot_registry.json').read_text())
  for cid,entry in registry['clips'].items():
   a=json.loads(gzip.decompress((SITE/'data'/clips[cid]['pose3d']).read_bytes()));v=entry['variants'][entry['preferred']];b=json.loads(gzip.decompress((SITE/'data'/v['pose3d']).read_bytes()));self.assertEqual(a['source_fit_sha256'],b['source_fit_sha256'])
   for x,y in zip(a['frames'],b['frames']):
    for field in ['anatomy','feet']:
     for p,q in zip(x[field],y[field]):
      for u,v in zip(p,q):self.assertAlmostEqual(u,v,delta=.000005)

if __name__=='__main__':unittest.main()
