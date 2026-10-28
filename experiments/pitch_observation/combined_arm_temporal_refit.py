"""Source/continuity repair initialized and depth-regularized by actual MotionBERT.

Small collar/shoulder/elbow SO(3) corrections preserve all native bone lengths.
Root, lower body, support anchors and foot coordinates stay exactly fixed.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from hybrid_initializer import global_rotations
from pitch_study_fit import stable_rotation
from support_transfer_root import metrics,save_geometry
from arm_cocking_evaluate import direction_returns
from scipy.spatial.transform import Rotation,Slerp
import argparse


def refit(rec,source):
    import torch
    torch.set_num_threads(1);cid=rec['id'];dest=ROOT/'retry_armtemporal/combined'/f'{cid}.npz';base=load(source);h=inputs(rec)
    meta=json.loads(source.with_suffix('.json').read_text());hashes=dict(source=sha(source),solver=sha(__file__))
    if dest.exists() and dest.with_suffix('.json').exists():return
    G=global_rotations(base['smpl_rotations']).astype(float);rot=base['smpl_rotations'].astype(float).copy();a=base['anatomical_keypoints'].astype(float).copy();p=base['keypoints'].astype(float).copy();tr=base['translation'].astype(float)
    xyz=a+tr[:,None];bg=base['background_transforms'];xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
    T=lambda x:torch.as_tensor(x,dtype=torch.float32);N=len(a);sc=scores(h);height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));leg=float(base['model_leg']);reports=[]
    for side in range(2):
        c,s,e,w,hand=13+side,16+side,18+side,20+side,22+side;cs,ce,cw=5+side,7+side,9+side
        rest=lambda q,R:np.median(np.einsum('tji,tj->ti',R,q),axis=0)
        vs=rest(xyz[:,s]-xyz[:,c],G[:,c]);ve=rest(xyz[:,e]-xyz[:,s],G[:,s]);vw=rest(xyz[:,w]-xyz[:,e],G[:,e]);vh=rest(xyz[:,hand]-xyz[:,w],G[:,w])
        offsets=[np.einsum('tji,tj->ti',G[:,j],p[:,cj]-a[:,nj]) for cj,nj,j in [(cs,s,c),(ce,e,s),(cw,w,e)]]
        links=(np.diff(h['frame_index'])==1)&(np.diff(h['source_time'])>0)&(np.diff(h['source_time'])<.1)
        observed,_=direction_returns(h['keypoints'][:,cw]-h['keypoints'][:,ce],links)
        old=load(OUT/rec['source'])['anatomical_keypoints'];existing,_=direction_returns(old[:,w]-old[:,e],links)
        initial_returns,_=direction_returns(a[:,w]-a[:,e],links)
        repair=[int(i) for i in initial_returns if i not in observed and i not in existing]
        for frame in repair:
            reference=[Slerp(h['source_time'][[frame-1,frame+1]],Rotation.from_matrix(G[[frame-1,frame+1],j]))([h['source_time'][frame]]).as_matrix()[0] for j in [c,s,e]]
            rot[frame,c]=G[frame,9].T@reference[0];rot[frame,s]=reference[0].T@reference[1];rot[frame,e]=reference[1].T@reference[2]
        parent=T(G[:,9]);C=T(xyz[:,c]);initial=T(rot[:,[c,s,e]]);rW=T(rot[:,w]);V=[T(v) for v in [vs,ve,vw,vh]];O=[T(o) for o in offsets]
        controls=torch.nn.Parameter(torch.zeros((N,3,3)));optimizer=torch.optim.Adam([controls],lr=.015)
        target=T(np.nan_to_num(xy[:,[cs,ce,cw]]));weight=T(sc[:,[cs,ce,cw]]**2*np.array([.5,1.,2.]));principal=T(base['principal']);focal=float(base['focal'])
        mb=load(ROOT/'motionbert'/f'{cid}.npz')['flip'];ms,me,mw=[11,12,13] if side==0 else [14,15,16]
        u=mb[:,me]-mb[:,ms];v=mb[:,mw]-mb[:,me]
        u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-8);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)
        depth=T(np.stack([np.linalg.norm(ve)*u[:,2],np.linalg.norm(ve)*u[:,2]+np.linalg.norm(vw)*v[:,2]],axis=1)/leg)
        bound=np.broadcast_to(np.array([.5,.65,.65])[None,:,None],(N,3,3)).copy()
        bound[repair]=.18;limits=T(bound)
        def geometry():
            delta=limits*torch.tanh(controls);r=initial@stable_rotation(delta);GC=parent@r[:,0];GS=GC@r[:,1];GE=GS@r[:,2]
            S=C+(GC@V[0][:,None])[...,0];E=S+(GS@V[1][:,None])[...,0];W=E+(GE@V[2][:,None])[...,0];H=W+(GE@rW@V[3][:,None])[...,0]
            points=torch.stack([S+(GC@O[0][:,:,None])[...,0],E+(GS@O[1][:,:,None])[...,0],W+(GE@O[2][:,:,None])[...,0]],dim=1)
            return delta,r,S,E,W,H,points
        for step in range(700):
            optimizer.param_groups[0]['lr']=.015*(.05+.95*(1-step/699)**2);optimizer.zero_grad()
            delta,r,S,E,W,H,points=geometry();uv=points[...,:2]/points[...,2:].clamp_min(.2)*focal+principal
            residual=(uv-target)/height;image=((torch.sqrt(residual.square().sum(-1)+.001**2)-.001)*weight).sum()/weight.sum().clamp_min(1)
            current_depth=(torch.stack([E-S,W-S],dim=1)[...,2])/leg
            res=current_depth-depth
            temporal_depth=.25*(res[1:]-res[:-1]).square().mean()+.5*(res[2:]+res[:-2]-2*res[1:-1]).square().mean()
            loss=image+.10*res.square().mean()+temporal_depth+.0002*delta.square().mean()+.003*(delta[2:]+delta[:-2]-2*delta[1:-1]).square().mean()
            loss.backward();optimizer.step()
        with torch.no_grad():delta,r,S,E,W,H,points=geometry()
        r=r.numpy();S,E,W,H,points=[v.numpy() for v in [S,E,W,H,points]]
        rot[:,[c,s,e]]=r
        for j,q in [(s,S),(e,E),(w,W),(hand,H)]:a[:,j]=q-tr
        p[:,[cs,ce,cw]]=points-tr[:,None]
        links=(np.diff(h['frame_index'])==1)&(np.diff(h['source_time'])>0)&(np.diff(h['source_time'])<.1)
        observed,_=direction_returns(h['keypoints'][:,cw]-h['keypoints'][:,ce],links)
        original=load(OUT/rec['source'])['anatomical_keypoints'];existing,_=direction_returns(original[:,w]-original[:,e],links)
        returns,angles=direction_returns(a[:,w]-a[:,e],links)
        unsupported=[int(t) for t in returns if t not in observed and t not in existing]
        reports.append(dict(side=side,selected=dict(name='MotionBERT initialized source/collar refit',strength=.003,
            unsupported_returns=unsupported,ray_clamped_frames=0),maximum_correction_radians=float(abs(delta.numpy()).max()),initialized_return_frames=repair,depth_template='actual flip-restored MotionBERT, not ray-path depth'))
    z={**base,'smpl_rotations':rot,'anatomical_keypoints':a,'keypoints':p}
    assert np.array_equal(z['foot_keypoints'],base['foot_keypoints']) and np.array_equal(z['translation'],base['translation'])
    m,uv=metrics(z,h,bg,z['translation'],z['contact_weight']);save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    parent=meta['hashes']['solver'];meta['hashes']['solver']=sha(__file__)
    meta.update(arms=reports,metrics=m,output_sha256=sha(dest),arm_refit=hashes,parent_motionbert_arm_solver_sha256=parent,
        motionbert_depth_regularization=True,root_and_foot_coordinates_preserved=True)
    save(dest.with_suffix('.json'),meta)
    folder=dest.parents[1];(folder/'ground').mkdir(exist_ok=True)
    for suffix in ['npz','json']:
        link=folder/'ground'/f'{cid}.{suffix}'
        if not link.exists():link.symlink_to((source.parents[1]/'ground'/f'{cid}.{suffix}').resolve())
    if not (folder/'motionbert').exists():(folder/'motionbert').symlink_to(ROOT/'motionbert',target_is_directory=True)
    print('ARM REPAIR',cid,m['wrist_p95_px'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    refit(next(r for r in manifest()['records'] if r['id']==a.id),a.source)
