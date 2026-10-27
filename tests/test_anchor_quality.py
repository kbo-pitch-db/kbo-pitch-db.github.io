import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from anchor_quality import stance_metrics


class StanceTests(unittest.TestCase):
    def fixture(self):
        count=90;feet=np.zeros((count,6,3));feet[:,:,1]=.005
        z=dict(foot_keypoints=feet,source_time=np.arange(count)/30.,frame_index=np.arange(count),fps=30.)
        q=dict(observed_phase_hints=dict(foot_plant_frame_hint=40,leg_lift_frame_hint=10,lead_side_hint='left'),anchor_end_frame=80)
        return z,q

    def test_fixed_support_and_actual_skating_are_distinguished(self):
        z,q=self.fixture();m=stance_metrics(z,q)
        self.assertGreater(m['stance_links'],20);self.assertEqual(m['stance_midpoint_speed_max_model_m_s'],0)
        z['foot_keypoints'][43:80,[0,2],2]=np.arange(37)[:,None]*.03
        self.assertAlmostEqual(stance_metrics(z,q)['stance_midpoint_speed_max_model_m_s'],.9)

    def test_swing_and_released_foot_motion_are_excluded(self):
        z,q=self.fixture();z['foot_keypoints'][15:37,[0,2],2]=np.arange(22)[:,None]*.04
        z['foot_keypoints'][45:80,[1,3],1]=.4
        z['foot_keypoints'][81:,[0,2],2]=1.
        m=stance_metrics(z,q)
        self.assertEqual(m['stance_midpoint_speed_max_model_m_s'],0)
        self.assertAlmostEqual(m['stance_midpoint_height_max_model_m'],0)


class RescuePlanningTests(unittest.TestCase):
    def test_short_stride_is_refitted_but_ambiguous_camera_and_late_phase_are_held(self):
        import csv,json,tempfile
        from unittest.mock import patch
        from pitch_anchor_batch import candidates
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);base=[];seed=[]
            for cid,reasons in [('short',['short_stride_depth_review']),('camera',['depth_initialization_sensitivity']),('late',[])]:
                q=dict(screening=dict(reasons=reasons,phase_hints={'stride_at_plant_leg':.6}))
                (d/(cid+'.json')).write_text(json.dumps(q))
                np.savez(d/(cid+'.npz'),anatomical_keypoints=np.zeros((5,24,3)))
                np.savez(d/(cid+'2.npz'),keypoints=np.zeros((5,26,2)),scores=np.ones((5,26)),fps=30,bbox=np.ones((5,4)))
                base.append(dict(dataset_clip_id=cid,pose_2d_refined=cid+'2.npz'))
                seed.append(dict(dataset_clip_id=cid,scene_quality=cid+'.json',pose_3d_scene=cid+'.npz'))
            for name,rows in [('motion_wham.csv',base),('motion_wham_pitch_scene.csv',seed)]:
                with (d/name).open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with patch('publication_quality.screen_publication',side_effect=[dict(reasons=['stride_direction_depth_review']),dict(reasons=['pitch_event_order_review'])]):
                planned=candidates(d)
            self.assertEqual([r['id'] for r in planned],['short'])

if __name__=='__main__':unittest.main()
