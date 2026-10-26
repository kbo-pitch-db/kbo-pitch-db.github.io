import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scene_geometry import initial_camera,clip_frame,transform_scene,contact_weights,fill_pose_gaps


class SceneGeometryTests(unittest.TestCase):
    def test_short_pose_gaps_remain_rotations_and_preserve_long_gaps(self):
        rotations=np.broadcast_to(np.eye(3),(15,24,3,3)).copy()
        valid=np.ones(15,bool);valid[3]=False;valid[7:11]=False
        rotations[~valid]=np.nan
        result,filled=fill_pose_gaps(rotations,valid,np.arange(15),np.ones(15))
        self.assertTrue(filled[3]);self.assertFalse(filled[7:11].any())
        np.testing.assert_allclose(result[3]@result[3].transpose(0,2,1),np.broadcast_to(np.eye(3),(24,3,3)),atol=1e-7)
        tracks=np.ones(15);tracks[4:]=2
        _,filled=fill_pose_gaps(rotations,valid,np.arange(15),tracks)
        self.assertFalse(filled[3])
    def test_one_rigid_transform_preserves_distances_and_motion(self):
        normal=np.array([0.,-.97,-.243]);normal/=np.linalg.norm(normal)
        feet=np.zeros((20,6,3));feet[:,:,1]=1.;feet[:,:,2]=5.
        offset=-np.dot(normal,feet[0,0]);contacts=np.ones((20,6))
        rotation,origin,_=clip_frame(normal,offset,feet,contacts)
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(rotation)),1.,places=6)
        points=np.random.default_rng(4).normal(size=(20,17,3));points[10:]+=np.array([.5,0,0])
        result=transform_scene(points,rotation,origin)
        np.testing.assert_allclose(np.linalg.norm(result[:,15]-result[:,16],axis=-1),np.linalg.norm(points[:,15]-points[:,16],axis=-1),rtol=1e-6)
        expected=(points[10]-points[9])@rotation.T
        np.testing.assert_allclose(result[10]-result[9],expected,atol=1e-6)
        np.testing.assert_allclose(transform_scene(feet,rotation,origin)[:,:,1],0,atol=1e-6)

    def test_camera_initialization_recovers_known_weak_projection(self):
        points=np.random.default_rng(2).normal(size=(10,17,3)).astype(np.float32)*.2
        scale=350.;center=np.array([960.,540.]);translation=np.array([.2,.4,2000/scale])
        xy=points[:,:,:2]*scale+translation[:2]*scale+center
        scales,offset,recovered=initial_camera(points,xy,np.ones((10,17)),2000,center)
        np.testing.assert_allclose(scales,scale,atol=1e-3)
        np.testing.assert_allclose(recovered,np.broadcast_to(translation,recovered.shape),atol=1e-5)

    def test_contact_does_not_lock_raised_foot_or_link_a_gap(self):
        xy=np.zeros((20,6,2));feet=np.zeros((20,6,3));feet[:,1,1]=-.7
        weights,_=contact_weights(xy,np.ones((20,6)),feet,np.array([0,-1,0]),30,np.ones(19,bool))
        self.assertGreater(weights[:,0].mean(),.9)
        self.assertLess(weights[:,1].mean(),.01)
        linked=np.ones(19,bool);linked[8]=False
        weights,_=contact_weights(xy,np.ones((20,6)),feet,np.array([0,-1,0]),30,linked)
        self.assertTrue((weights[9]==0).all())


if __name__=='__main__':unittest.main()
