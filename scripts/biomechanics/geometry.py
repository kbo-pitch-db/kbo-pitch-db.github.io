"""Explicit right-handed frames and signed angles. All angles in degrees.

Analysis frame: X toward plate, Y = up × X, Z = inferred up. Segment frames:
X anterior, Y anatomical left, Z superior. Euler decomposition is intrinsic ZYX
(R=Rz(yaw) Ry(pitch) Rx(roll)); not an undocumented Visual3D default.
"""
import numpy as np
from scipy.spatial.transform import Rotation

PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])


def unit(v):
    v=np.asarray(v,float);norm=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.divide(v,norm,out=np.full_like(v,np.nan),where=norm>1e-8)


def included(a,b,c):
    u=unit(np.asarray(a)-b);v=unit(np.asarray(c)-b)
    return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1),-1,1)))


def flexion(a,b,c):return 180-included(a,b,c)


def wrap(degrees):return (np.asarray(degrees)+180)%360-180


def global_rotations(local):
    out=np.empty_like(local)
    for j,p in enumerate(PARENTS):out[:,j]=local[:,j] if p<0 else out[:,p]@local[:,j]
    return out


def anatomical_frame(left,up):
    z=unit(up);y=unit(left-z*np.sum(left*z,axis=-1,keepdims=True));x=unit(np.cross(y,z));y=unit(np.cross(z,x))
    return np.stack([x,y,z],axis=-1)


def rest_calibration(camera_joints,global_R):
    """Recover constant native rest offsets through FK, not posed shoulder lines.

    This constructs model anatomical frames, not ASIS/PSIS/C7 marker frames.
    Collar motion does not define the thorax segment rotation.
    """
    offsets=np.zeros((24,3))
    for j,p in enumerate(PARENTS):
        if p>=0:offsets[j]=np.nanmedian(np.einsum('tji,tj->ti',global_R[:,p],camera_joints[:,j]-camera_joints[:,p]),axis=0)
    rest=np.zeros((24,3))
    for j,p in enumerate(PARENTS):
        if p>=0:rest[j]=rest[p]+offsets[j]
    pelvis=anatomical_frame(rest[1]-rest[2],rest[9]-rest[0])
    thorax=anatomical_frame(rest[16]-rest[17],rest[12]-rest[9])
    return pelvis,thorax,rest


def euler_zyx(matrix):
    matrix=np.asarray(matrix)
    yaw=np.degrees(np.arctan2(matrix[...,1,0],matrix[...,0,0]))
    pitch=np.degrees(np.arctan2(-matrix[...,2,0],np.hypot(matrix[...,0,0],matrix[...,1,0])))
    roll=np.degrees(np.arctan2(matrix[...,2,1],matrix[...,2,2]))
    singular=np.hypot(matrix[...,0,0],matrix[...,1,0])<1e-3
    yaw=np.where(singular,np.nan,yaw);roll=np.where(singular,np.nan,roll)
    return np.stack([yaw,pitch,roll],axis=-1)


def pitch_frame(direction,up=np.array([0.,1.,0.])):
    z=unit(up);d=np.asarray(direction,float);x=unit(d-z*np.dot(d,z));y=unit(np.cross(z,x))
    return np.column_stack([x,y,z])


def signed_horizontal(first,second,up=np.array([0.,1.,0.])):
    first=np.asarray(first);second=np.asarray(second)
    ah=first-up*np.sum(first*up,axis=-1,keepdims=True);bh=second-up*np.sum(second*up,axis=-1,keepdims=True)
    a=unit(ah);b=unit(bh)
    out=np.degrees(np.arctan2(np.sum(np.cross(a,b)*up,axis=-1),np.sum(a*b,axis=-1)))
    good=(np.linalg.norm(ah,axis=-1)>.2*np.linalg.norm(first,axis=-1))&(np.linalg.norm(bh,axis=-1)>.2*np.linalg.norm(second,axis=-1))
    return np.where(good,out,np.nan)


def plane_inclinations(superior,pitch_basis,hand):
    """Two distinct signed global-plane tilts, not one total vertical angle.

    These are not substituted for a laboratory Cardan/joint-coordinate angle.
    They remain interpretable when the thorax rotates about its superior axis.
    """
    local=np.asarray(superior)@pitch_basis;factor=1 if hand=='R' else -1
    return np.degrees(np.arctan2(local[...,0],local[...,2])),factor*np.degrees(np.arctan2(local[...,1],local[...,2]))


def segment_series(scene_joints,local_rotations,camera_joints,camera_to_scene,hand,pitch_axis=None):
    """Return explicit spatial series; no timestamp conversion or event inference."""
    R=np.asarray(camera_to_scene,float);G=global_rotations(local_rotations)
    Cp,Ct,rest=rest_calibration(camera_joints,G)
    pelvis=R[None]@G[:,0]@Cp;thorax=R[None]@G[:,9]@Ct
    relative=np.swapaxes(pelvis,1,2)@thorax;rel=euler_zyx(relative)
    factor=1 if hand=='R' else -1
    lead=(1,4,7) if hand=='R' else (2,5,8);arm=(17,19,21) if hand=='R' else (16,18,20)
    a=scene_joints;hipline=a[:,2]-a[:,1];shoulderline=a[:,17]-a[:,16]
    upper=unit(a[:,arm[1]]-a[:,arm[0]])
    upper_local=np.einsum('tji,tj->ti',thorax,upper)
    elevation=np.degrees(np.arccos(np.clip(-upper_local[:,2],-1,1)))
    # The outward anatomical direction is right (-Y) for R, left (+Y) for L.
    lateral=-factor*upper_local[:,1]
    horizontal=np.degrees(np.arctan2(-upper_local[:,0],lateral))
    horiz_len=np.hypot(upper_local[:,0],upper_local[:,1]);horizontal[horiz_len<.2]=np.nan
    series=dict(lead_knee_flexion=flexion(a[:,lead[0]],a[:,lead[1]],a[:,lead[2]]),
        elbow_flexion=flexion(a[:,arm[0]],a[:,arm[1]],a[:,arm[2]]),
        thorax_pelvis_relative_axial=-factor*rel[:,0],
        shoulder_line_separation=-factor*signed_horizontal(hipline,shoulderline),
        upper_arm_elevation=elevation,shoulder_horizontal_abduction=horizontal)
    if pitch_axis is not None:
        B=pitch_frame(pitch_axis);p=euler_zyx(B.T[None]@pelvis);t=euler_zyx(B.T[None]@thorax)
        forward,lateral=plane_inclinations(thorax[:,:,2],B,hand)
        series.update(pelvis_rotation=factor*p[:,0],trunk_rotation=factor*t[:,0],
            trunk_forward_lean=forward,trunk_lateral_lean=lateral)
    return series,dict(pelvisCalibration=Cp.tolist(),thoraxCalibration=Ct.tolist(),
        restJoints=rest.tolist(),pelvisFramesScene=pelvis.tolist(),thoraxFramesScene=thorax.tolist(),thoraxSegment='SMPL spine3 joint 9 cumulative rotation; native-rest calibration',
        rotationOrder='intrinsic ZYX',anatomicalMarkerCalibration=False)
