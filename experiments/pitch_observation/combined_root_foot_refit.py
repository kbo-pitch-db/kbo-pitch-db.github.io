"""Refine root depth while preserving its image ray, then solve contact IK.

One fixed camera/scene is retained. This changes reconstructed motion, never
the renderer's origin. Start/end positions receive strong observations; no
endpoint reset or population-stride target is introduced.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from combined_contact_ik import inverse_legs
from support_transfer_root import metrics,save_geometry
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from hybrid_initializer import global_rotations
from arm_cocking_native import align_direction
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import argparse


def refit(rec,source):
    cid=rec['id'];dest=ROOT/'retry_rootfoot/combined'/f'{cid}.npz'
    if dest.exists():return
    base=load(source);groundpath=source.parents[1]/'ground'/source.name;ground=load(groundpath)
    notes=json.loads(groundpath.with_suffix('.json').read_text())['contact_episodes']
    original=base
    G=global_rotations(base['smpl_rotations']);rot=base['smpl_rotations'].astype(float).copy()
    anatomy=base['anatomical_keypoints'].astype(float).copy();feet=base['foot_keypoints'].astype(float).copy()
    normal=base['normal'].astype(float);normal/=np.linalg.norm(normal)
    for side in range(2):
        ankle,knee,toe=7+side,4+side,10+side;markers=np.array([side,side+2,side+4])
        q=Rotation.from_matrix(G[:,ankle]).as_quat()
        for t in range(1,len(q)):
            if q[t]@q[t-1]<0:q[t]*=-1
        q=gaussian_filter1d(q,1.5,axis=0);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        R=Rotation.from_quat(q).as_matrix()
        local=np.einsum('tji,tkj->tki',G[:,ankle],feet[:,markers]-anatomy[:,ankle,None])
        supported=base['contact_weight'][:,markers].sum(1)>.1
        offsets=np.median(local[supported] if supported.any() else local,axis=0)
        sole=np.cross(offsets[1]-offsets[0],offsets[2]-offsets[0]);sole/=max(np.linalg.norm(sole),1e-9)
        if np.median((G[:,ankle]@sole[:,None])[:,:,0]@normal)<0:sole*=-1
        active=np.zeros(len(R),bool)
        for episode in notes:
            if episode['marker']%2==side:active[episode['start']:episode['end']+1]=True
        for t in np.flatnonzero(active):
            candidates=[(i,e) for i,e in enumerate(notes) if e['marker']%2==side and e['start']<=t<=e['end']]
            primary=int(ground['primary_support_episode'][t]);chosen=next(((i,e) for i,e in candidates if i==primary),candidates[0])
            ei,e=chosen;anchor=ground['support_episode_anchors'][ei];j=int(np.flatnonzero(markers==e['marker'])[0])
            implied=anchor+(R[t]@(offsets-offsets[j]).T).T
            if (implied@normal+float(base['pilot_support_offset'])).min()<-.005:
                align,_=align_direction((R[t]@sole)[None],normal[None],np.array([[1.,0.,0.]]));R[t]=align[0]@R[t]
        rot[:,ankle]=np.swapaxes(G[:,knee],-1,-2)@R
        rest=np.median(np.einsum('tji,tj->ti',G[:,ankle],anatomy[:,toe]-anatomy[:,ankle]),axis=0)
        anatomy[:,toe]=anatomy[:,ankle]+np.einsum('tij,j->ti',R,rest)
        feet[:,markers]=anatomy[:,ankle,None]+np.einsum('tij,kj->tki',R,offsets)
    base={**base,'anatomical_keypoints':anatomy,'foot_keypoints':feet,'smpl_rotations':rot}
    root=base['anatomical_keypoints'][:,0]+base['translation'];n=len(root)
    D=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)).tocsr()
    weights=np.ones(n);weights[[0,-1]]=5
    desired=np.column_stack([spsolve(diags(weights)+8*(D.T@D),weights*root[:,j]) for j in range(3)])
    depth=desired[:,2]
    z={**base,'translation':desired-base['anatomical_keypoints'][:,0]}
    z,report=inverse_legs(z,ground,notes);h=inputs(rec)
    m,uv=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight'])
    save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    meta=json.loads(source.with_suffix('.json').read_text());meta.update(output_sha256=sha(dest),contact_ik=report,
        contact_ik_hashes=dict(source=sha(source),solver=sha(Path(__file__).with_name('combined_contact_ik.py'))),
        root_depth_refit=dict(solver_sha256=sha(__file__),source_sha256=sha(source),regularization=8,ankle_rotation_sigma_frames=1.5,sole_nonpenetration_rotation_constraint=True,sequence_fixed_rigid_foot_markers=True,
            image_ray_preserved_before_contact_ik=False,render_transform_changed=False,
            root_depth_change_max_leg=float(abs(depth-root[:,2]).max()/float(base['model_leg']))),metrics=m)
    save(dest.with_suffix('.json'),meta)
    folder=dest.parents[1];(folder/'ground').mkdir(exist_ok=True)
    for suffix in ['npz','json']:
        link=folder/'ground'/f'{cid}.{suffix}'
        if not link.exists():link.symlink_to((source.parents[1]/'ground'/f'{cid}.{suffix}').resolve())
    if not (folder/'motionbert').exists():(folder/'motionbert').symlink_to(ROOT/'motionbert',target_is_directory=True)
    print('ROOT DEPTH',cid,meta['root_depth_refit']['root_depth_change_max_leg'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    refit(next(r for r in manifest()['records'] if r['id']==a.id),a.source)
