"""Rigid per-sequence shoe markers plus native contact IK on the selected body.

SMPL skin/toe pose correctives must not stretch an illustrated baseball shoe.
Native ankle-to-foot bones and every other native segment remain fixed length.
Support positions are reconstruction constraints, never renderer edits.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from combined_contact_ik import inverse_legs
from hybrid_initializer import global_rotations
from support_transfer_root import metrics,save_geometry
from arm_cocking_native import align_direction
from scipy.spatial.transform import Rotation
import argparse


def process(rec,choice):
    cid=rec['id'];source=ROOT/choice['fit_path'];folder=ROOT/'rigid_soles_v1';dest=folder/'combined'/f'{cid}.npz'
    if dest.exists() and dest.with_suffix('.json').exists():return
    base=load(source);groundpath=source.parents[1]/'ground'/source.name;ground=load(groundpath)
    notes=json.loads(groundpath.with_suffix('.json').read_text())['contact_episodes']
    G=global_rotations(base['smpl_rotations']);a=base['anatomical_keypoints'].astype(float).copy()
    rot=base['smpl_rotations'].astype(float).copy();feet=base['foot_keypoints'].astype(float).copy();leg=float(base['model_leg'])
    normal=base['normal'].astype(float);normal/=np.linalg.norm(normal);offsets=np.zeros((6,3))
    ankle_ids=np.array([7,8,7,8,7,8])
    for side in range(2):
        ankle,knee=7+side,4+side;markers=np.array([side,side+2,side+4]);R=G[:,ankle].copy()
        local=np.einsum('tji,tkj->tki',R,feet[:,markers]-a[:,ankle,None])
        good=base['contact_weight'][:,markers].sum(1)>.1
        offsets[markers]=np.median(local[good] if good.any() else local,axis=0)
        sole=np.cross(offsets[markers[1]]-offsets[markers[0]],offsets[markers[2]]-offsets[markers[0]])
        sole/=max(np.linalg.norm(sole),1e-9)
        if np.median((R@sole[:,None])[:,:,0]@normal)<0:sole*=-1
        for t,episode in enumerate(ground['primary_support_episode']):
            active=[(i,e) for i,e in enumerate(notes) if e['marker']%2==side and e['start']<=t<=e['end']]
            if not active:continue
            i,e=next(((i,e) for i,e in active if i==episode),active[0]);marker=e['marker'];anchor=ground['support_episode_anchors'][i]
            predicted=anchor+(R[t]@(offsets[markers]-offsets[marker]).T).T
            if (predicted@normal+float(base['pilot_support_offset'])).min()<-.008*leg:
                align,_=align_direction((R[t]@sole)[None],normal[None],np.array([[1.,0.,0.]]))
                adjustment=Rotation.from_matrix(align[0]).as_rotvec();left,right=0.,1.
                for unused in range(20):
                    middle=(left+right)/2;candidate=Rotation.from_rotvec(adjustment*middle).as_matrix()@R[t]
                    p=anchor+(candidate@(offsets[markers]-offsets[marker]).T).T
                    if (p@normal+float(base['pilot_support_offset'])).min()>=-.008*leg:right=middle
                    else:left=middle
                R[t]=Rotation.from_rotvec(adjustment*right).as_matrix()@R[t]
        rot[:,ankle]=np.swapaxes(G[:,knee],-1,-2)@R
        rest=np.median(np.einsum('tji,tj->ti',G[:,ankle],a[:,10+side]-a[:,ankle]),axis=0)
        a[:,10+side]=a[:,ankle]+np.einsum('tij,j->ti',R,rest)
        feet[:,markers]=a[:,ankle,None]+np.einsum('tij,kj->tki',R,offsets[markers])
    z={**base,'smpl_rotations':rot,'anatomical_keypoints':a,'foot_keypoints':feet}
    z,report=inverse_legs(z,ground,notes);h=inputs(rec)
    m,uv=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight'])
    z.update(shoe_local_offsets=offsets,shoe_ankle_joints=ankle_ids,
        shoe_marker_model=np.array('Sequence-fixed ankle-frame shoe markers; no skin pose-corrective stretching'))
    save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    meta=json.loads(source.with_suffix('.json').read_text());meta.update(output_sha256=sha(dest),contact_ik=report,
        contact_ik_hashes=dict(source=sha(source),solver=sha(Path(__file__).with_name('combined_contact_ik.py'))),
        rigid_sole_refit=dict(solver_sha256=sha(__file__),source_sha256=sha(source),constant_shape=True,
            sole_nonpenetration_rotation_constraint=True,native_bone_lengths_preserved=True),metrics=m)
    save(dest.with_suffix('.json'),meta)
    (folder/'ground').mkdir(exist_ok=True)
    for suffix in ['npz','json']:
        link=folder/'ground'/f'{cid}.{suffix}'
        if not link.exists():link.symlink_to((source.parents[1]/'ground'/f'{cid}.{suffix}').resolve())
    print('RIGID SOLE',cid,report['root_delta_max_leg'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--id');a=p.parse_args()
    choices=json.loads((ROOT/'selection_frozen.json').read_text())['selected']
    for rec in manifest()['records'][a.worker::a.workers]:
        if a.id and rec['id']!=a.id:continue
        process(rec,choices[rec['id']])
