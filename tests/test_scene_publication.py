import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_scene import export_scene
from package_pitch_scene import publication_decision
from scene_parallel import reusable
from scene_geometry import SMPL_JOINTS,SMPL_PARENTS,FOOT_NAMES


class ScenePublicationTests(unittest.TestCase):
    def test_public_export_preserves_fixed_coordinates_and_low_confidence_interpolation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.npz';count=5
            points=np.zeros((count,17,3),np.float32);points[:,:,0]=np.arange(count)[:,None]*.1+2;points[:,:,1]=1
            scores=np.ones((count,17));scores[2]=0
            np.savez_compressed(path,keypoints=points,scores=scores,
                anatomical_keypoints=np.ones((count,24,3)),foot_keypoints=np.zeros((count,6,3)),
                scene_rotation=np.eye(3),scene_origin=np.zeros(3),frame_index=np.arange(count)+100,
                source_time=np.arange(count)/30+10,fps=30,joints=np.array([str(i) for i in range(17)]),
                anatomical_joints=SMPL_JOINTS,anatomical_parents=SMPL_PARENTS,foot_joints=FOOT_NAMES,
                coordinate_system='clip-fixed inferred support',model='WHAM synthetic fixture',model_sha256='a'*64,
                method='synthetic fixture',score_semantics='detector confidence',scene_signature='fixture',
                fitted_camera_focal=2000,fit_accepted=True,contact_weights=np.zeros((count,6)),
                interpolated_frames=np.array([False,False,True,False,False]))
            data=export_scene(path,{'anchor_policy':'synthetic fixed frame','limitations':'fixture only'})
            self.assertFalse(data['scene']['per_frame_recentering'])
            self.assertEqual(data['frames'][0]['keypoints'][11][:3],[2,1,0])
            self.assertEqual(data['frames'][4]['keypoints'][11][:3],[2.4,1,0])
            self.assertEqual(data['frames'][2]['keypoints'][0][3],0)
            self.assertTrue(data['frames'][2]['interpolated'][0])
            self.assertEqual(len(data['frames'][0]['anatomy']),24)
            self.assertEqual(data['primary_representation']['frame_field'],'anatomy')
            self.assertIn('pose-dependent',data['keypoints_semantics'])

    def test_visual_veto_is_bound_to_the_reviewed_scene_signature(self):
        q={'fit_accepted':True,'scene_signature':'current','screening':{'reasons':[]}}
        review={'scene_signature':'current','decision':'withhold','reason':'visible contact failure'}
        allowed,reasons,_=publication_decision(q,review)
        self.assertFalse(allowed);self.assertIn('visual_review_failed',reasons)
        review['scene_signature']='older'
        self.assertTrue(publication_decision(q,review)[0])
        q['fit_accepted']=False;q['screening']['reasons']=['depth_initialization_sensitivity']
        self.assertFalse(publication_decision(q,{'scene_signature':'current','decision':'approve'})[0])
        q['fit_accepted']=True;q['screening']['reasons']=[]
        self.assertFalse(publication_decision(q,post={'passed':True,'reasons':[]})[0])
        self.assertTrue(publication_decision(q,{'scene_signature':'current','decision':'preview_only'},post={'passed':True,'reasons':[]})[0])

    def test_legacy_scene_packager_routes_through_current_publication_gate(self):
        from unittest.mock import patch
        from package_scene import package_scene
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'motion_wham_pitch_scene.csv').write_text('dataset_clip_id\n')
            with patch('package_pitch_scene.package_pitch_scene',return_value='current gate') as current:
                self.assertEqual(package_scene(root,root/'public'),'current gate')
                current.assert_called_once_with(root,root/'public')

    def test_successful_higher_iteration_cache_is_reused_and_stale_quality_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);expected=hashlib.sha256(b'basecode360False').hexdigest()
            np.savez(root/'scene.npz',scene_signature=expected,fit_accepted=True)
            quality={'steps':360,'fit_accepted':True,'background_stabilization':True,'scene_signature':expected}
            (root/'quality.json').write_text(json.dumps(quality))
            row={'dataset_clip_id':'clip','pose_3d_scene':'scene.npz','scene_quality':'quality.json','scene_signature':expected,'fit_accepted':'true'}
            with (root/'motion_wham_scene.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            cached=reusable(root,{'clip':{'input_signature':'base'}},'code',90)
            self.assertEqual(set(cached),{'clip'})
            quality['scene_signature']='stale';(root/'quality.json').write_text(json.dumps(quality))
            self.assertEqual(reusable(root,{'clip':{'input_signature':'base'}},'code',90),{})


if __name__=='__main__':unittest.main()
