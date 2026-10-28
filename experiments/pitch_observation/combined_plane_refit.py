"""Bounded single-plane calibration from saved contact and sole observations.

No pose/root/foot coordinates change. The one clip-wide inferred plane may
rotate by at most a few degrees and move at most 2% of modeled leg length at
the contact centroid. This is not a measured mound or per-frame floor lift.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from support_transfer_root import metrics,save_geometry
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import argparse


def refit(rec,source):
    cid=rec['id'];dest=ROOT/'retry_plane/combined'/f'{cid}.npz'
    if dest.exists():return
    z=load(source);groundpath=source.parents[1]/'ground'/source.name
    notes=json.loads(groundpath.with_suffix('.json').read_text())['contact_episodes'];leg=float(z['model_leg'])
    feet=z['foot_keypoints'].astype(float)+z['translation'][:,None];contact=np.concatenate([feet[e['start']:e['end']+1,e['marker']] for e in notes])
    reference=np.median(contact,axis=0);n0=z['normal'].astype(float);n0/=np.linalg.norm(n0)
    height=float(reference@n0+z['pilot_support_offset']);F=(feet-reference)/leg;C=(contact-reference)/leg
    def plane(x):
        angles=.04*np.tanh(x[:2]);n=Rotation.from_rotvec([angles[0],0,angles[1]]).apply(n0)
        delta=.02*np.tanh(x[2]);h=height/leg+delta
        return n,h,angles,delta
    def residual(x):
        n,h,angles,delta=plane(x)
        signed=F@n+h
        return np.r_[(C@n+h-.015)/np.sqrt(len(C)),
            30*np.maximum(-signed-.01,0).ravel()/np.sqrt(signed.size),
            .03*angles/.04,.01*delta/.02]
    fitted=least_squares(residual,np.zeros(3),max_nfev=300,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    normal,h,angles,delta=plane(fitted.x);offset=h*leg-reference@normal
    observations=inputs(rec);m,uv=metrics(z,observations,z['background_transforms'],z['translation'],z['contact_weight'])
    save_geometry(dest,z,z['translation'],uv,z['contact_weight'],normal,float(offset))
    meta=json.loads(source.with_suffix('.json').read_text());meta.update(output_sha256=sha(dest),
        plane_refit=dict(solver_sha256=sha(__file__),source_sha256=sha(source),constant_plane=True,
            angular_change_radians=angles.tolist(),height_change_at_contact_centroid_leg=float(delta),
            pose_and_foot_coordinates_unchanged=True,measured_ground=False))
    save(dest.with_suffix('.json'),meta)
    folder=dest.parents[1];(folder/'ground').mkdir(exist_ok=True)
    for suffix in ['npz','json']:
        link=folder/'ground'/f'{cid}.{suffix}'
        if not link.exists():link.symlink_to((source.parents[1]/'ground'/f'{cid}.{suffix}').resolve())
    if not (folder/'motionbert').exists():(folder/'motionbert').symlink_to(ROOT/'motionbert',target_is_directory=True)
    print('PLANE',cid,meta['plane_refit'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    refit(next(r for r in manifest()['records'] if r['id']==a.id),a.source)
