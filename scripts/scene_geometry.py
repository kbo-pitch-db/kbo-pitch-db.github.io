"""Geometry for an inferred, clip-fixed support frame (not surveyed world coordinates)."""
import numpy as np

SMPL_JOINTS = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee',
    'spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
    'left_collar','right_collar','head','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
FOOT_NAMES = ['left_big_toe','right_big_toe','left_small_toe','right_small_toe','left_heel','right_heel']
VERSION = 'wham-scene-fit-v1'


def fill_pose_gaps(rotations,valid,frame_index,tracks,max_gap=3):
    """Interpolate short internal SMPL gaps on SO(3), preserving identity boundaries."""
    from scipy.spatial.transform import Rotation,Slerp
    result=rotations.copy();filled=np.zeros(len(valid),bool)
    missing=np.flatnonzero(~valid)
    for gap in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
        if not len(gap) or len(gap)>max_gap:continue
        a,b=int(gap[0])-1,int(gap[-1])+1
        if a<0 or b>=len(valid) or not valid[a] or not valid[b]:continue
        if np.any(np.diff(frame_index[a:b+1])!=1):continue
        if tracks is not None and (tracks[a]<0 or tracks[a]!=tracks[b]):continue
        alpha=(gap-a)/(b-a)
        for joint in range(24):
            slerp=Slerp([0,1],Rotation.from_matrix(result[[a,b],joint]))
            result[gap,joint]=slerp(alpha).as_matrix()
        filled[gap]=True
    return result,filled


def initial_camera(points, xy, scores, focal, principal):
    """Weighted weak-perspective initialization for subsequent perspective fitting."""
    weight=np.where(np.isfinite(xy).all(-1)&np.isfinite(points).all(-1),np.clip(scores,0,1)**2,0)
    weight[:, :5]*=.4
    weight/=np.maximum(weight.sum(-1,keepdims=True),1e-8)
    p=np.nan_to_num(points[:,:,:2]);q=np.nan_to_num(xy)
    pc=(p*weight[:,:,None]).sum(1);qc=(q*weight[:,:,None]).sum(1)
    numerator=((p-pc[:,None])*(q-qc[:,None])*weight[:,:,None]).sum((1,2))
    denominator=(((p-pc[:,None])**2)*weight[:,:,None]).sum((1,2))
    scale=np.clip(numerator/np.maximum(denominator,1e-8),30,2500)
    offset=qc-scale[:,None]*pc
    z=focal/scale
    translation=np.c_[(offset-principal)/scale[:,None],z]
    return scale.astype(np.float32),offset.astype(np.float32),translation.astype(np.float32)


def support_normal(joints):
    """One initial up direction from the early torso; it remains a fitted assumption."""
    v=(joints[:,5]+joints[:,6]-joints[:,11]-joints[:,12])/2
    v=v[:max(3,len(v)//6)]
    n=np.nanmedian(v,axis=0)
    if not np.isfinite(n).all() or np.linalg.norm(n)<.05:n=np.array([0.,-1.,-.15])
    n=n/np.linalg.norm(n)
    # Camera axes: x right, y down, z away. A support normal must point upward.
    if n[1]>0:n=-n
    return n.astype(np.float32)


def clip_frame(normal, offset, feet, contacts):
    """Return a single rigid camera-to-scene transform, anchored once at the start."""
    n=np.asarray(normal,dtype=float);n/=np.linalg.norm(n)
    x=np.array([1.,0,0]);x-=n*np.dot(x,n);x/=np.linalg.norm(x)
    z=np.cross(x,n);rotation=np.stack([x,n,z])
    early=max(3,min(15,len(feet)//5))
    w=np.asarray(contacts[:early],float)
    good=np.isfinite(feet[:early]).all(-1)
    w=np.where(good,w,0)
    if w.sum()>.1:
        origin=(np.nan_to_num(feet[:early])*w[:,:,None]).sum((0,1))/w.sum()
        policy='early supported foot observations projected onto inferred local plane'
    else:
        origin=np.nanmedian(feet[:early].reshape(-1,3),axis=0)
        policy='early foot median projected onto inferred local plane; contact uncertain'
    origin=origin-n*(np.dot(n,origin)+float(offset))
    return rotation.astype(np.float32),origin.astype(np.float32),policy


def transform_scene(points, rotation, origin):
    return ((points-origin)@rotation.T).astype(np.float32)


def contact_weights(xy, confidence, feet_camera, normal, fps, linked):
    """Soft, marker-specific support hypotheses; heels and toes can lift independently."""
    distance=feet_camera@normal
    finite=distance[np.isfinite(distance)]
    level=float(np.percentile(finite,12)) if len(finite) else 0.
    height=np.maximum(distance-level,0)
    velocity=np.zeros(xy.shape[:2],np.float32)
    if len(xy)>1:
        delta=np.linalg.norm(np.diff(xy,axis=0),axis=-1)*fps
        delta[~linked]=np.inf
        velocity[1:]=delta;velocity[0]=velocity[1]
    # Pixel speed is normalized before this call by the image-space person height.
    weight=np.exp(-(velocity/.28)**2)*np.exp(-(height/.12)**2)
    weight*=np.clip(confidence,0,1)**2
    weight[~np.isfinite(weight)]=0
    return weight.astype(np.float32),level


def stabilize_crop(video, boxes, cache=None):
    """Robust background affine tracking. Never align using the pitcher silhouette."""
    import cv2
    if cache is not None and cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            if len(z['transforms'])==len(boxes):return z['transforms'],float(z['success_ratio'])
    capture=cv2.VideoCapture(str(video));transforms=[np.eye(3,dtype=np.float32)]
    previous=None;points=None;successes=0;scale=.5
    try:
        for i in range(len(boxes)):
            ok,image=capture.read()
            if not ok:raise ValueError('Video length differs from pose during background tracking')
            gray=cv2.cvtColor(cv2.resize(image,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY)
            mask=np.full(gray.shape,255,np.uint8)
            box=np.asarray(boxes[i])*scale
            if np.isfinite(box).all():
                a,b,c,d=box.astype(int);pad=12
                mask[max(0,b-pad):max(0,d+pad),max(0,a-pad):max(0,c+pad)]=0
            if previous is not None:
                transform=transforms[-1].copy()
                if points is not None and len(points)>=16:
                    target,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None)
                    back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous,target,None)
                    keep=(status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back-points,axis=(1,2))<1.5)
                    source=points[keep,0];destination=target[keep,0]
                    if len(source)>=16:
                        matrix,inliers=cv2.estimateAffinePartial2D(source,destination,method=cv2.RANSAC,ransacReprojThreshold=1.5)
                        if matrix is not None and inliers.sum()>=12:
                            factor=np.linalg.norm(matrix[0,:2]);shift=np.linalg.norm(matrix[:,2])
                            spread=np.ptp(source[inliers[:,0]>0],axis=0)
                            if .97<factor<1.03 and shift<30 and np.prod(spread)>gray.size*.03:
                                full=np.eye(3);full[:2]=matrix;full[:2,2]/=scale
                                transform=transforms[-1]@np.linalg.inv(full)
                                successes+=1
                transforms.append(transform.astype(np.float32))
            previous=gray
            points=cv2.goodFeaturesToTrack(gray,maxCorners=250,qualityLevel=.02,minDistance=8,mask=mask)
    finally:capture.release()
    transforms=np.asarray(transforms,dtype=np.float32)
    ratio=successes/max(1,len(boxes)-1)
    if cache is not None:
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,transforms=transforms,success_ratio=ratio)
    return transforms,ratio
