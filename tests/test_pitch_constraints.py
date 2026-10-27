import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pitch_scene_fit import repair_rotation_spikes,spline_basis
from pitch_contacts import infer_pitch_contacts
from pitch_quality import screen
from scene_geometry import SMPL_PARENTS


class PitchConstraintTests(unittest.TestCase):
    def test_first_lift_is_used_instead_of_later_followthrough_kick(self):
        n=100;xy=np.zeros((n,26,2));xy[:,:,1]=500
        lift=np.maximum(0,1-np.abs(np.arange(n)-20)/12);kick=np.maximum(0,1-np.abs(np.arange(n)-70)/15)
        xy[:,13,1]-=lift*160;xy[:,14,1]-=kick*220
        xy[:,[20,22],1]-=lift[:,None]*200;xy[:,[21,23],1]-=kick[:,None]*250
        w,hint=infer_pitch_contacts(xy,np.ones((n,26)),500,30)
        self.assertEqual(hint['lead_side_hint'],'left');self.assertLess(hint['leg_lift_frame_hint'],30)
        self.assertTrue(25<hint['foot_plant_frame_hint']<45)
        self.assertGreater(w[20,1],.9);self.assertEqual(w[20,0],0)
        self.assertGreater(w[45,0],.9)

    def test_airborne_velocity_pause_is_not_landing(self):
        n=70;xy=np.zeros((n,26,2));xy[:,:,1]=500;time=np.arange(n)
        lift=np.maximum(0,1-np.abs(time-8)/10)
        xy[:,13,1]-=lift*160
        x=np.interp(time,[0,15,19,21,28,35,69],[0,0,0,0,80,100,100])
        y=np.interp(time,[0,8,15,19,21,28,35,69],[300,250,420,480,480,485,490,490])
        for joint in [20,22]:xy[:,joint,0]=x;xy[:,joint,1]=y
        _,hint=infer_pitch_contacts(xy,np.ones((n,26)),500,60)
        self.assertTrue(26<=hint['foot_plant_frame_hint']<=37,hint)
        self.assertGreater(hint['stable_landing_evidence_frame'],hint['foot_plant_frame_hint'])

    def test_missing_foot_pixels_do_not_become_a_false_impact(self):
        n=70;xy=np.zeros((n,26,2));xy[:,:,1]=500;time=np.arange(n)
        lift=np.maximum(0,1-np.abs(time-8)/12);xy[:,13,1]-=lift*160
        foot=np.interp(time,[0,8,25,69],[350,250,490,490])
        for j in [20,22]:xy[:,j,1]=foot
        scores=np.ones((n,26));scores[:41,[20,22]]=.4
        xy[35:38,[20,22]]=0;scores[35:38,[20,22]]=0
        _,hint=infer_pitch_contacts(xy,scores,500,30)
        self.assertEqual(hint['status'],'ambiguous',hint)

    def test_so3_spike_repair_keeps_a_sustained_fast_rotation(self):
        r=np.broadcast_to(np.eye(3),(12,24,3,3)).copy();r[5,4]=Rotation.from_rotvec([1.4,0,0]).as_matrix()
        r[:,18]=Rotation.from_rotvec(np.c_[np.arange(12)*.7,np.zeros((12,2))]).as_matrix()
        out,changed=repair_rotation_spikes(r,np.ones(12,bool),30)
        self.assertTrue(changed[5,4]);self.assertFalse(changed[:,18].any())
        np.testing.assert_allclose(out[5,4],np.eye(3),atol=1e-7)
        np.testing.assert_allclose(out[:,18],r[:,18])

    def test_planar_spline_is_a_smooth_partition_of_unity(self):
        basis=spline_basis(180,60);np.testing.assert_allclose(basis.sum(1),1,atol=1e-6)
        # A constant depth remains constant; no per-frame depth reset is introduced.
        np.testing.assert_allclose(basis@np.ones(basis.shape[1])*12,12,atol=1e-5)

    def fixture(self):
        p=np.zeros((40,24,3));p[:,:,1]=.9
        p[:,1]=[-.1,.8,0];p[:,2]=[.1,.8,0];p[:,4]=[-.1,.4,-.12];p[:,5]=[.1,.4,.12]
        p[:,7]=[-.1,.05,-.25];p[:,8]=[.1,.05,.25]
        z=dict(anatomical_keypoints=p,foot_keypoints=np.zeros((40,6,3))+.005,root_translation=np.tile([0,.9,0],(40,1)),
            source_time=np.arange(40)/30,frame_index=np.arange(40),fps=30,anatomical_parents=np.array(SMPL_PARENTS),contact_weights=np.ones((40,6)))
        fit=dict(median_reprojection_height=.01,p95_reprojection_height=.03,lower_body_reprojection_height_p95=.03,feet_reprojection_height_p95=.03,
            observed_phase_hints=dict(lead_side_hint='left',leg_lift_frame_hint=5,foot_plant_frame_hint=12))
        return z,fit

    def test_screen_rejects_teleport_and_floating_even_with_constant_bones(self):
        z,fit=self.fixture();self.assertTrue(screen(z,fit)['passed'])
        for field in ['anatomical_keypoints','foot_keypoints','root_translation']:z[field][20:,...,0]+=2
        q=screen(z,fit);self.assertIn('root_speed_max_leg_s',q['reasons']);self.assertLess(q['metrics']['native_bone_length_std_model_m'],1e-6)
        z,fit=self.fixture()
        for field in ['anatomical_keypoints','foot_keypoints','root_translation']:z[field][...,1]+=.4
        self.assertIn('floating_ratio',screen(z,fit)['reasons'])

    def test_short_stride_is_reviewed_without_changing_coordinates(self):
        z,fit=self.fixture();fit['camera_candidates']=[{}]*6
        z['anatomical_keypoints'][:,7,2]=-.05;z['anatomical_keypoints'][:,8,2]=.05
        before=z['anatomical_keypoints'].copy();q=screen(z,fit)
        self.assertIn('short_stride_depth_review',q['reasons']);np.testing.assert_array_equal(before,z['anatomical_keypoints'])

    def test_unsupported_phase_or_bad_feet_cannot_pass_on_good_median_2d_fit(self):
        z,fit=self.fixture();fit['feet_reprojection_height_p95']=.2
        self.assertIn('poor_2d_alignment',screen(z,fit)['reasons'])
        fit['feet_reprojection_height_p95']=.02;fit['observed_phase_hints']={'status':'ambiguous'}
        self.assertIn('pitch_contact_phase_ambiguous',screen(z,fit)['reasons'])


if __name__=='__main__':unittest.main()
