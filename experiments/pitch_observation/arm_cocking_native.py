"""Direct H36M17 arm-direction adapter; no full-body SMPL IK intermediary.

Maps native MotionBERT upper-arm/forearm camera directions to the released
SMPL arm lengths/shoulder location. Humeral twist and elbow-plane rotation
are explicit hypotheses; palm/wrist local rotations stay unchanged.
"""
from common import *
from hybrid_initializer import global_rotations
from scipy.spatial.transform import Rotation
from arm_cocking_axial import forearm_twist

def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)

def align_direction(u,v,fallback):
 u=unit(u);v=unit(v);cross=np.cross(u,v);size=np.linalg.norm(cross,axis=-1);dot=np.clip((u*v).sum(-1),-1,1);axis=cross/np.maximum(size[:,None],1e-9);opposite=(size<1e-7)&(dot<0)
 axis[opposite]=unit(fallback[opposite]);bad=opposite&(np.linalg.norm(axis,axis=-1)<.5)
 if bad.any():
  reference=np.eye(3)[np.argmin(abs(u[bad]),axis=-1)];axis[bad]=unit(np.cross(u[bad],reference))
 return Rotation.from_rotvec(axis*np.arctan2(size,dot)[:,None]).as_matrix(),opposite

def native_graft(base,native,bg,side):
 r=base['smpl_rotations'].copy();g=global_rotations(r);a=base['anatomical_keypoints'];s=16+side;e=18+side;w=20+side;ns,ne,nw=[11,12,13] if side==0 else [14,15,16];p=native['keypoints'];assert np.array_equal(base['frame_index'],native['frame_index']) and np.array_equal(base['source_time'],native['source_time'])
 roll=Rotation.from_euler('z',np.arctan2(bg[:,1,0],bg[:,0,0])).as_matrix();u=a[:,e]-a[:,s];f=a[:,w]-a[:,e];target_u=np.einsum('tij,tj->ti',roll,p[:,ne]-p[:,ns]);target_f=np.einsum('tij,tj->ti',roll,p[:,nw]-p[:,ne]);valid=np.isfinite(target_u).all(-1)&np.isfinite(target_f).all(-1)&(np.linalg.norm(target_u,axis=-1)>1e-6)&(np.linalg.norm(target_f,axis=-1)>1e-6)
 target_u[~valid]=u[~valid];target_f[~valid]=f[~valid];plane=np.cross(u,f);hinge=unit(plane);bend0=np.arctan2(np.linalg.norm(plane,axis=-1),np.sum(u*f,axis=-1));bend1=np.arctan2(np.linalg.norm(np.cross(target_u,target_f),axis=-1),np.sum(target_u*target_f,axis=-1));bend=bend1-bend0
 hinge_degenerate=np.linalg.norm(plane,axis=-1)<1e-7
 if hinge_degenerate.any():
  reference=np.eye(3)[np.argmin(abs(unit(u[hinge_degenerate])),axis=-1)];hinge[hinge_degenerate]=unit(np.cross(u[hinge_degenerate],reference))
 alignment,opposite=align_direction(u,target_u,hinge);bent=np.einsum('tij,tj->ti',Rotation.from_rotvec(hinge*bend[:,None]).as_matrix(),f);aligned=np.einsum('tij,tj->ti',alignment,bent);twist,plane_valid=forearm_twist(target_u,aligned,target_f);shoulder=Rotation.from_rotvec(unit(target_u)*twist[:,None]).as_matrix()@alignment@g[:,s];r[:,s]=np.swapaxes(g[:,13+side],-1,-2)@shoulder;local_hinge=np.einsum('tij,tj->ti',np.swapaxes(g[:,s],-1,-2),hinge);r[:,e]=Rotation.from_rotvec(local_hinge*bend[:,None]).as_matrix()@r[:,e];r[~valid]=base['smpl_rotations'][~valid]
 diagnostic=dict(source='Native H36M17 MB upper-arm/forearm directions, no full-body SMPL IK',invalid_direction_frames=int((~valid).sum()),nearly_opposite_upperarm_frames=int(opposite.sum()),degenerate_forearm_plane_frames=int((~plane_valid).sum()),elbow_plane_rotation_radians_p95=float(np.percentile(abs(bend),95)),humeral_twist_radians_p95=float(np.percentile(abs(twist),95)),absolute_or_wrist_flexion_accuracy_claim=False)
 return r,diagnostic
