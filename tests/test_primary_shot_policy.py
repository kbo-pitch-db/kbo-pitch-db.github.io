"""Confirmed camera cuts truncate 3D views without changing final 2D bytes."""
import csv,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/pitch_observation'))
import common

class PrimaryShotPolicyTests(unittest.TestCase):
 def test_source_timestamp_prefix_and_original_files_are_preserved(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);data=root/'dataset';data.mkdir();policies=root/'run/shot_validity';policies.mkdir(parents=True)
   np.savez(data/'two.npz',keypoints=np.ones((10,26,2)),scores=np.ones((10,26)),frame_index=np.arange(100,110),source_time=np.arange(10)/30,bbox=np.ones((10,4)),joints=np.array(['j']*26))
   np.savez(data/'prior.npz',keypoints=np.ones((10,17,3)),frame_index=np.arange(100,110),source_time=np.arange(10)/30,smpl_betas=np.ones((10,10)),joints=np.array(['j']*17))
   for name,row in [('motion_wham.csv',dict(dataset_clip_id='S_c1',pose_2d_refined='two.npz',prior_3d='prior.npz')),('clips.csv',dict(dataset_clip_id='S_c1',source_id='S')),('sources.csv',dict(source_id='S',width='1920',height='1080'))]:
    with (data/name).open('w') as f:w=csv.DictWriter(f,row.keys());w.writeheader();w.writerow(row)
   original=(data/'two.npz').read_bytes();digest=hashlib.sha256(original).hexdigest();(policies/'S_c1.json').write_text(json.dumps(dict(valid_frames=8,source_clip_frames=10,confirmed_source_cut=True,final_2d_sha256=digest)))
   with patch.object(common,'DATA',data),patch.dict(os.environ,{'KBO_GVHMR_PRIMARY_SHOT_ROOT':str(root/'run')}):
    _,_,_,h,w=common.init_inputs('S_c1');np.testing.assert_array_equal(h['frame_index'],np.arange(100,108));np.testing.assert_allclose(h['source_time'],np.arange(8)/30);self.assertEqual(len(w['keypoints']),8);self.assertEqual(len(h['joints']),26);self.assertEqual(int(h['source_clip_frames']),10)
   self.assertEqual((data/'two.npz').read_bytes(),original)
   with patch.object(common,'DATA',data),patch.dict(os.environ,{'KBO_GVHMR_PRIMARY_SHOT_ROOT':''}):self.assertEqual(len(common.init_inputs('S_c1')[3]['frame_index']),10)

if __name__=='__main__':unittest.main()
