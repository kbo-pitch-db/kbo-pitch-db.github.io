"""Synthetic coordinate-adapter invariants; no pre-release experiment execution."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/pitch_observation'))
from arm_cocking_refine import prepare_graft,candidate_geometry
from hybrid_initializer import global_rotations
from arm_cocking_axial import forearm_twist,axial_graft
from arm_cocking_native import native_graft,unit

class ArmCockingCoordinateTests(unittest.TestCase):
 def test_arm_transplant_uses_stabilized_camera_and_preserves_other_chains(self):
  rng=np.random.default_rng(42);base_r=Rotation.from_rotvec(rng.normal(0,.2,(48,3))).as_matrix().reshape(2,24,3,3).astype(np.float32);mb_r=Rotation.from_rotvec(rng.normal(0,.4,(48,3))).as_matrix().reshape(2,24,3,3).astype(np.float32);base=dict(smpl_rotations=base_r,frame_index=np.array([4,5]));mb=dict(smpl_rotations=mb_r,frame_index=np.array([4,5]));roll=Rotation.from_euler('z',[.13,-.17]).as_matrix();bg=roll.copy();bg[:,:2,:2]*=1.1;bg[:,:2,2]=[[12,4],[-7,8]]
  target=global_rotations(mb_r)
  for side in [0,1]:
   actual=prepare_graft(base,mb,bg,side);glob=global_rotations(actual)
   for j in [16+side,18+side]:np.testing.assert_allclose(glob[:,j],roll@target[:,j],atol=5e-7)
   np.testing.assert_array_equal(actual[:,:16],base_r[:,:16]);np.testing.assert_array_equal(actual[:,20:],base_r[:,20:])
   other=[17-side,19-side];np.testing.assert_array_equal(actual[:,other],base_r[:,other])

 def test_forearm_only_keeps_shoulder_orientation_and_hits_forearm_hypothesis(self):
  rot=np.tile(np.eye(3),(2,24,1,1));mb=rot.copy();mb[:,18]=Rotation.from_euler('y',[.5,-.7]).as_matrix();mb[:,19]=Rotation.from_euler('y',[-.4,.6]).as_matrix();base=dict(smpl_rotations=rot,frame_index=np.array([0,1]));prior=dict(smpl_rotations=mb,frame_index=np.array([0,1]));bg=np.tile(np.eye(3),(2,1,1))
  for side in [0,1]:
   actual=prepare_graft(base,prior,bg,side,True);g=global_rotations(actual);expected=global_rotations(mb)
   np.testing.assert_array_equal(actual[:,16+side],rot[:,16+side]);np.testing.assert_allclose(g[:,18+side],expected[:,18+side],atol=1e-10)

 def test_axial_rotation_changes_forearm_plane_without_moving_upper_arm(self):
  r=np.tile(np.eye(3),(2,24,1,1));upper=np.array([[1.,0.,0.],[1.,0.,0.]]);forearm=np.array([[0.,1.,0.],[0.,1.,0.]]);target=np.array([[0.,0.,1.],[0.,0.,-1.]])
  theta,valid=forearm_twist(upper,forearm,target);self.assertTrue(valid.all())
  for side in [0,1]:
   actual=axial_graft(r,upper,theta,side);g=global_rotations(actual)
   np.testing.assert_allclose(np.einsum('tij,tj->ti',g[:,16+side],upper),upper,atol=1e-10)
   np.testing.assert_allclose(np.einsum('tij,tj->ti',g[:,18+side],forearm),target,atol=1e-10)
   np.testing.assert_array_equal(actual[:,18+side],r[:,18+side]);np.testing.assert_array_equal(actual[:,20:],r[:,20:])
  angle,valid=forearm_twist(upper,upper,target);self.assertFalse(valid.any());np.testing.assert_array_equal(angle,0)

 def test_native_motionbert_directions_are_transferred_without_whole_body_ik(self):
  r=np.tile(np.eye(3),(2,24,1,1));a=np.zeros((2,24,3));upper=np.tile([1.,0.,0.],(2,1));forearm=np.tile([0.,1.,0.],(2,1));target_u=np.array([[0.,1.,0.],[0.,0.,1.]]);target_f=unit(np.array([[0.,0.,1.],[1.,0.,.3]]));mb=np.zeros((2,17,3));bg=np.tile(np.eye(3),(2,1,1));times=np.array([0.,.03]);frames=np.array([4,5])
  for side in [0,1]:
   ss,ee,ww=[11,12,13] if side==0 else [14,15,16];mb[:,ee]=target_u;mb[:,ww]=target_u+target_f;a[:,18+side]=upper;a[:,20+side]=upper+forearm
   base=dict(smpl_rotations=r,anatomical_keypoints=a,source_time=times,frame_index=frames);prior=dict(keypoints=mb,source_time=times,frame_index=frames)
   actual,diagnostics=native_graft(base,prior,bg,side);g=global_rotations(actual)
   np.testing.assert_allclose(np.einsum('tij,tj->ti',g[:,16+side],upper),target_u,atol=1e-7)
   np.testing.assert_allclose(np.einsum('tij,tj->ti',g[:,18+side],forearm),target_f,atol=1e-7)
   np.testing.assert_array_equal(actual[:,:16],r[:,:16]);np.testing.assert_array_equal(actual[:,20:],r[:,20:]);self.assertEqual(diagnostics['invalid_direction_frames'],0)

 def test_changing_root_definition_never_changes_native_world_trajectory(self):
  anatomy=np.arange(144,dtype=float).reshape(2,24,3)/100;tr=np.array([[0.,0.,8.],[.2,.1,8.5]]);rot=np.tile(np.eye(3),(2,24,1,1));shift=np.array([[.01,-.02,.03],[-.02,.01,.02]])
  base=dict(anatomical_keypoints=anatomy,translation=tr,smpl_rotations=rot)
  changed=candidate_geometry(base,rot,np.zeros((2,17,3)),anatomy+shift[:,None],np.zeros((2,6,3)))
  np.testing.assert_allclose(changed['anatomical_keypoints']+changed['translation'][:,None],anatomy+tr[:,None],atol=1e-12)

if __name__=='__main__':unittest.main()
