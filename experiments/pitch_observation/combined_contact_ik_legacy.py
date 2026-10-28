"""Native fixed-length two-link leg IK for all observed support anchors.

This updates actual local joint rotations/root and saved native coordinates.
It is never a render-time foot clamp. Knees retain bounded flexion and their
existing small off-hinge components; the ankle's world orientation is retained.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from hybrid_initializer import global_rotations
from arm_cocking_native import align_direction
from support_transfer_root import metrics,save_geometry
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import time,traceback,argparse


def inverse_legs(base,ground,notes):
    a0=base['anatomical_keypoints'].astype(float);p0=base['keypoints'].astype(float)
    tr0=base['translation'].astype(float);f0=base['foot_keypoints'].astype(float)
    world=a0+tr0[:,None];feet=f0+tr0[:,None];n=len(a0);leg=float(base['model_leg'])
    r0=base['smpl_rotations'].astype(float);G=global_rotations(r0)
    chosen=np.full((n,2),-1,int)
    primary=ground['primary_support_episode'];weights=ground['support_episode_weights']
    for t in range(n):
        for side in range(2):
            valid=[i for i,e in enumerate(notes) if e['marker']%2==side and e['start']<=t<=e['end']]
            if valid:chosen[t,side]=int(primary[t]) if int(primary[t]) in valid else max(valid,key=lambda i:weights[t,i])
    models=[];root_delta=np.zeros((n,3))
    for side in range(2):
        hip,knee,ankle,toe=1+side,4+side,7+side,10+side
        rest=lambda v,g:np.median(np.einsum('tji,tj->ti',g,v),axis=0)
        u=rest(world[:,knee]-world[:,hip],G[:,hip]);v=rest(world[:,ankle]-world[:,knee],G[:,knee])
        foot=rest(world[:,toe]-world[:,ankle],G[:,ankle]);kv=Rotation.from_matrix(r0[:,knee]).as_rotvec()
        grid=np.linspace(-.08,2.8,257)
        vectors=np.broadcast_to(kv[:,None],(n,len(grid),3)).copy();vectors[:,:,0]=grid
        matrices=Rotation.from_rotvec(vectors.reshape(-1,3)).as_matrix().reshape(n,len(grid),3,3)
        distance=np.linalg.norm(u+(matrices@v[None,None,:,None])[...,0],axis=-1)
        lo=distance.min(1)+1e-7;hi=distance.max(1)-1e-7
        target=world[:,ankle].copy();valid=chosen[:,side]>=0
        for t in np.flatnonzero(valid):
            episode=chosen[t,side];marker=notes[episode]['marker']
            target[t]=ground['support_episode_anchors'][episode]-(feet[t,marker]-world[t,ankle])
        models.append(dict(side=side,hip=hip,knee=knee,ankle=ankle,toe=toe,u=u,v=v,foot=foot,
            kv=kv,grid=grid,distance=distance,lo=lo,hi=hi,target=target,valid=valid))
    def project_delta(delta):
        for unused in range(32):
            for m in models:
                mask=m['valid'];center=m['target']-world[:,m['hip']];d=center-delta
                size=np.linalg.norm(d,axis=-1);wanted=np.clip(size,m['lo'],m['hi'])
                change=d*(1-wanted/np.maximum(size,1e-10))[:,None];delta[mask]+=change[mask]
        return delta
    root_delta=project_delta(root_delta)
    root_delta=project_delta(gaussian_filter1d(root_delta,1.,axis=0))
    tr=tr0+root_delta;a=a0.copy();p=p0.copy();f=f0.copy();rot=r0.copy();unreachable=[]
    for m in models:
        ids=np.flatnonzero(m['valid']);hip,knee,ankle,toe=[m[k] for k in ['hip','knee','ankle','toe']]
        if not len(ids):continue
        H=world[ids,hip]+root_delta[ids];D=m['target'][ids]-H;length=np.linalg.norm(D,axis=-1)
        flex=[]
        for t,d in zip(ids,length):
            residual=m['distance'][t]-d;roots=np.flatnonzero(residual[:-1]*residual[1:]<=0)
            if len(roots):
                j=min(roots,key=lambda k:abs(m['grid'][k]-m['kv'][t,0]));left,right=m['grid'][j:j+2]
                for unused in range(35):
                    middle=(left+right)/2;vector=m['kv'][t].copy();vector[0]=middle
                    q=Rotation.from_rotvec(vector).as_matrix()@m['v']+m['u'];value=np.linalg.norm(q)-d
                    vv=m['kv'][t].copy();vv[0]=left;first=np.linalg.norm(Rotation.from_rotvec(vv).as_matrix()@m['v']+m['u'])-d
                    if first*value<=0:right=middle
                    else:left=middle
                flex.append((left+right)/2)
            else:
                j=int(np.argmin(abs(residual)));flex.append(m['grid'][j]);unreachable.append(dict(frame=int(t),side=m['side'],distance_error_model=float(abs(residual[j]))))
        kneevec=m['kv'][ids].copy();kneevec[:,0]=flex;RK=Rotation.from_rotvec(kneevec).as_matrix()
        endpoint=m['u']+(RK@m['v'][None,:,None])[...,0]
        old_vector=(G[ids,hip]@endpoint[:,:,None])[...,0]
        align,_=align_direction(old_vector,D,np.cross(world[ids,knee]-world[ids,hip],world[ids,ankle]-world[ids,knee]))
        GH=align@G[ids,hip];GK=GH@RK;K=H+(GH@m['u'][None,:,None])[...,0];A=K+(GK@m['v'][None,:,None])[...,0]
        T=A+(G[ids,ankle]@m['foot'][None,:,None])[...,0]
        rot[ids,hip]=np.swapaxes(G[ids,0],-1,-2)@GH;rot[ids,knee]=RK;rot[ids,ankle]=np.swapaxes(GK,-1,-2)@G[ids,ankle]
        for j,q in [(knee,K),(ankle,A),(toe,T)]:a[ids,j]=q-tr[ids]
        ck,ca=13+m['side'],15+m['side']
        offk=np.einsum('tji,tj->ti',G[ids,hip],p0[ids,ck]-a0[ids,knee])
        offa=np.einsum('tji,tj->ti',G[ids,knee],p0[ids,ca]-a0[ids,ankle])
        p[ids,ck]=a[ids,knee]+np.einsum('tij,tj->ti',GH,offk)
        p[ids,ca]=a[ids,ankle]+np.einsum('tij,tj->ti',GK,offa)
        marker_ids=np.array([m['side'],m['side']+2,m['side']+4])
        shift=A-world[ids,ankle]-root_delta[ids]
        f[ids[:,None],marker_ids[None]]=f0[ids[:,None],marker_ids[None]]+shift[:,None]
    result={**base,'keypoints':p,'anatomical_keypoints':a,'foot_keypoints':f,'translation':tr,'smpl_rotations':rot}
    return result,dict(root_delta_max_leg=float(np.linalg.norm(root_delta,axis=-1).max()/leg),
        unreachable=unreachable,root_optimized_for_reachability=True,
        native_segment_lengths_preserved=True,knee_flexion_bounds_radians=[-.08,2.8],
        ankle_world_orientations_preserved=True,render_time_clamp=False)


def process(rec,source_root):
    cid=rec['id'];source=source_root/'combined'/f'{cid}.npz'
    if not source.exists():return False
    label='base' if source_root==ROOT else source_root.name
    folder=ROOT/'polished_v1'/label;folder.mkdir(parents=True,exist_ok=True)
    for name,target in [('ground',source_root/'ground'),('motionbert',ROOT/'motionbert')]:
        if not (folder/name).exists():(folder/name).symlink_to(target,target_is_directory=True)
    dest=folder/'combined'/source.name;meta=source.with_suffix('.json')
    hashes=dict(source=sha(source),solver=sha(__file__))
    if dest.exists() and dest.with_suffix('.json').exists():
        assert json.loads(dest.with_suffix('.json').read_text())['contact_ik_hashes']==hashes;return True
    base=load(source);ground=load(source_root/'ground'/source.name);notes=json.loads((source_root/'ground'/source.name).with_suffix('.json').read_text())['contact_episodes']
    z,report=inverse_legs(base,ground,notes);h=inputs(rec)
    m,uv=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight'])
    save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    old=json.loads(meta.read_text());old.update(output_sha256=sha(dest),contact_ik_hashes=hashes,contact_ik=report,
        metrics=m,body_and_feet_unchanged=False,arm_local_rotations_preserved_after_motionbert=True)
    save(dest.with_suffix('.json'),old)
    print('CONTACT IK',label,cid,report['root_delta_max_leg'],len(report['unreachable']),flush=True);return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--id')
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    while True:
        for folder in [ROOT]+sorted(ROOT.glob('retry_*')):
            for rec in manifest()['records'][a.worker::a.workers]:
                if a.id and rec['id']!=a.id:continue
                try:process(rec,folder)
                except Exception:print(traceback.format_exc(),flush=True)
        if a.once or a.id:break
        time.sleep(20)
