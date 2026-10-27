"""Analytic synthetic geometry checks, not measured KBO accuracy."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/pitch_observation'))
from arm_depth_rays import sphere_on_ray,project,depth_path

class ArmRayGeometryTests(unittest.TestCase):
 def test_two_exact_depth_roots_have_same_image_and_fixed_bone_length(self):
  center=np.array([0.,0.,5.]);uv=np.array([50.,0.]);roots,clamped=sphere_on_ray(center,uv,.5,1000.,np.zeros(2));self.assertFalse(clamped)
  np.testing.assert_allclose(np.linalg.norm(roots-center,axis=-1),.5,atol=1e-11);np.testing.assert_allclose(project(roots,1000.,np.zeros(2)),np.tile(uv,(2,1)),atol=1e-10);self.assertGreater(np.linalg.norm(roots[0]-roots[1]),.5)
 def test_infeasible_observation_is_flagged_without_stretching_arm(self):
  roots,clamped=sphere_on_ray(np.array([0.,0.,5.]),np.array([200.,0.]),.5,1000.,np.zeros(2));self.assertTrue(clamped);np.testing.assert_allclose(np.linalg.norm(roots-np.array([0.,0.,5.]),axis=-1),.5,atol=1e-10);self.assertTrue(np.all(project(roots,1000.,np.zeros(2))[:,0]<200))
 def test_temporal_selector_chooses_existing_whole_hypotheses(self):
  q=np.zeros((6,4,2,3));q[:,1,:,2]=.5;q[:,2,:,2]=-.5;q[:,3,:,2]=.9;node=np.ones((6,4));node[:,1]=0;path=depth_path(q,node,np.arange(6)/30,np.zeros(6),1.);np.testing.assert_array_equal(path,1);selected=q[np.arange(6),path];np.testing.assert_array_equal(selected,np.repeat(q[:1,1],6,axis=0))

if __name__=='__main__':unittest.main()
