import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publication_quality import screen_publication


class PublicationQualityTests(unittest.TestCase):
    def fixture(self):
        n=70;xy=np.full((n,26,2),250.,dtype=float);xy[:,:,1]=200
        xy[:,[11,12],1]=350;xy[:,[5,6],1]=200
        xy[:,9,0]=250+250*np.tanh((np.arange(n)-30)/2)
        h={'keypoints':xy,'scores':np.ones((n,26)),'fps':30,'bbox':np.tile([0,0,500,600],(n,1))}
        p=np.zeros((n,24,3));p[:,7,2]=-.6;p[:,8,2]=.6
        q={'screening':{'phase_hints':{'lead_side_hint':'left','leg_lift_frame_hint':5,'foot_plant_frame_hint':20}}}
        return {'anatomical_keypoints':p},q,h

    def test_complete_order_is_screened_without_modifying_2d(self):
        m,q,h=self.fixture();before=h['keypoints'].copy();r=screen_publication(m,q,h)
        self.assertTrue(r['passed'],r);np.testing.assert_array_equal(before,h['keypoints'])

    def test_followthrough_landing_and_incomplete_preparation_are_reviewed(self):
        m,q,h=self.fixture();q['screening']['phase_hints']['foot_plant_frame_hint']=45
        self.assertIn('pitch_event_order_review',screen_publication(m,q,h)['reasons'])
        q['screening']['phase_hints']['foot_plant_frame_hint']=20;h['keypoints'][:8,[5,6],1]=340
        self.assertIn('pitch_preparation_unresolved',screen_publication(m,q,h)['reasons'])

    def test_lateral_depth_collapse_and_unobserved_arm_are_reviewed(self):
        m,q,h=self.fixture();m['anatomical_keypoints'][:,7,[0,2]]=[1,0];m['anatomical_keypoints'][:,8,[0,2]]=[-1,0]
        self.assertIn('stride_direction_depth_review',screen_publication(m,q,h)['reasons'])
        h['scores'][:,[9,10]]=0
        self.assertIn('arm_phase_unresolved',screen_publication(m,q,h)['reasons'])


if __name__=='__main__':unittest.main()
