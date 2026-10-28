"""Native FK contacts must stay fixed without stretching bones or pinning flight."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/pitch_observation'))
from combined_contact_ik import inverse_legs
from hybrid_initializer import global_rotations

P=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])


def observation(double=True):
    n=20;rest=np.zeros((24,3))
    for j in range(1,24):rest[j]=[0,.1,0]
    rest[1]=[-.1,0,0];rest[2]=[.1,0,0]
    rest[[4,5,7,8]]=[0,.45,0];rest[[10,11]]=[0,.03,-.15]
    r=np.broadcast_to(np.eye(3),(n,24,3,3)).copy()
    r[:,[4,5]]=Rotation.from_rotvec([.5,0,0]).as_matrix()
    G=global_rotations(r);a=np.zeros((n,24,3))
    for j in range(1,24):a[:,j]=a[:,P[j]]+np.einsum('tij,j->ti',G[:,P[j]],rest[j])
    tr=np.tile([0.,1.,10.],(n,1));tr[:,2]+=np.linspace(0,.02,n)
    f=np.zeros((n,6,3))
    for j in range(6):f[:,j]=a[:,7+j%2]+np.einsum('tij,j->ti',G[:,7+j%2],[.015*(j//2),.03,-.13])
    p=a[:,:17].copy();base=dict(anatomical_keypoints=a,keypoints=p,foot_keypoints=f,translation=tr,smpl_rotations=r,model_leg=np.array(.9))
    count=2 if double else 1;notes=[dict(marker=j,start=0,end=n-1) for j in range(count)]
    anchors=np.array([f[0,j]+tr[0] for j in range(count)])
    ground=dict(primary_support_episode=np.zeros(n,int),support_episode_weights=np.ones((n,count)),support_episode_anchors=anchors)
    return base,ground,notes


class CombinedContactIKTests(unittest.TestCase):
    def assert_bones(self,a,b):
        before=np.linalg.norm(a[:,1:]-a[:,P[1:]],axis=-1)
        after=np.linalg.norm(b[:,1:]-b[:,P[1:]],axis=-1)
        np.testing.assert_allclose(after,before,atol=1e-6,rtol=0)

    def test_both_supporting_feet_are_fixed_with_bounded_knees(self):
        base,ground,notes=observation();result,report=inverse_legs(base,ground,notes)
        self.assert_bones(base['anatomical_keypoints'],result['anatomical_keypoints'])
        for j in range(2):
            p=result['foot_keypoints'][:,j]+result['translation']
            np.testing.assert_allclose(p,np.tile(ground['support_episode_anchors'][j],(len(p),1)),atol=1e-6,rtol=0)
        knee=Rotation.from_matrix(result['smpl_rotations'][:,[4,5]].reshape(-1,3,3)).as_rotvec()
        self.assertGreaterEqual(knee[:,0].min(),-.080001);self.assertLessEqual(knee[:,0].max(),2.800001)
        self.assertFalse(report['unreachable']);self.assertFalse(report['render_time_clamp'])

    def test_unobserved_other_foot_remains_free(self):
        base,ground,notes=observation(False);result,report=inverse_legs(base,ground,notes)
        np.testing.assert_array_equal(result['anatomical_keypoints'][:,[2,5,8,11]],base['anatomical_keypoints'][:,[2,5,8,11]])
        np.testing.assert_array_equal(result['foot_keypoints'][:,[1,3,5]],base['foot_keypoints'][:,[1,3,5]])
        p=result['foot_keypoints'][:,1]+result['translation']
        self.assertGreater(np.linalg.norm(p[-1]-p[0]),.01)

    def test_explicit_flight_ankle_target_uses_native_fk_without_stretch(self):
        base,ground,notes=observation(False)
        target=base['anatomical_keypoints'][:,[7,8]]+base['translation'][:,None]
        target[:,1,0]+=np.linspace(0,.04,len(target))
        ground['desired_ankle_world']=target
        ground['desired_ankle_valid']=np.ones((len(target),2),bool)
        result,report=inverse_legs(base,ground,notes)
        self.assert_bones(base['anatomical_keypoints'],result['anatomical_keypoints'])
        actual=result['anatomical_keypoints'][:,[7,8]]+result['translation'][:,None]
        np.testing.assert_allclose(actual,target,atol=1e-6,rtol=0)
        self.assertFalse(report['unreachable'])

    def test_unreachable_contacts_are_reported_without_stretching(self):
        base,ground,notes=observation();ground['support_episode_anchors'][0,0]+=10
        result,report=inverse_legs(base,ground,notes)
        self.assertTrue(report['unreachable'])
        self.assert_bones(base['anatomical_keypoints'],result['anatomical_keypoints'])


if __name__=='__main__':unittest.main()
