"""Refine root depth while preserving its image ray, then solve contact IK.

One fixed camera/scene is retained. This changes reconstructed motion, never
the renderer's origin. Start/end positions receive strong observations; no
endpoint reset or population-stride target is introduced.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from combined_contact_ik import inverse_legs
from support_transfer_root import metrics,save_geometry
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import argparse


def refit(rec,source):
    cid=rec['id'];dest=ROOT/'retry_rootdepth/combined'/f'{cid}.npz'
    if dest.exists():return
    base=load(source);groundpath=source.parents[1]/'ground'/source.name;ground=load(groundpath)
    notes=json.loads(groundpath.with_suffix('.json').read_text())['contact_episodes']
    root=base['anatomical_keypoints'][:,0]+base['translation'];n=len(root)
    D=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)).tocsr()
    weights=np.ones(n);weights[[0,-1]]=10000
    depth=spsolve(diags(weights)+20*(D.T@D),weights*root[:,2])
    desired=root*(depth/root[:,2])[:,None]
    z={**base,'translation':desired-base['anatomical_keypoints'][:,0]}
    z,report=inverse_legs(z,ground,notes);h=inputs(rec)
    m,uv=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight'])
    save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    meta=json.loads(source.with_suffix('.json').read_text());meta.update(output_sha256=sha(dest),contact_ik=report,
        contact_ik_hashes=dict(source=sha(source),solver=sha(Path(__file__).with_name('combined_contact_ik.py'))),
        root_depth_refit=dict(solver_sha256=sha(__file__),source_sha256=sha(source),regularization=20,
            image_ray_preserved_before_contact_ik=True,render_transform_changed=False,
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
