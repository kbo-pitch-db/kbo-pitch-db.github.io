"""Actual MotionBERT depth-conditioned arm reconstruction on the grounded body.

Image-ray/fixed-length candidates preserve native segment lengths. A temporal
path selects complete depth hypotheses without averaging roots. Both arms are
processed, so uncertain throwing-hand metadata does not select the wrong arm.
The grounded body, camera, root and six foot markers remain exactly unchanged.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from arm_depth_rays import sphere_on_ray,project
import arm_depth_rays as rays
from arm_cocking_native import native_graft
from hybrid_initializer import global_rotations
from arm_cocking_refine import phase_window
from arm_cocking_evaluate import direction_returns
from support_transfer_root import metrics,save_geometry
from stride_review import native_leg
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import argparse,traceback,time

CONFIG=dict(version='combined-motionbert-arms-v1',all_reliable_observations_used=True,
    marker_model='Baseline rigid offsets attached to native humerus/forearm; not whole-mesh SMPL pose correctives',
    body_root_camera_feet='exactly unchanged from support reconstruction',
    motionbert_role='actual adapted MotionBERT single/flip relative-depth branch priors, not image XY',
    depth_averaging=False,arm_length_scaling=False,
    temporal='second-order path; stronger acceleration controls retained, no moving-average pose filter')


def apply_arm(base,rot,side):
    s,e,w,palm=16+side,18+side,20+side,22+side;ce,cw=7+side,9+side
    old=global_rotations(base['smpl_rotations']).astype(float);new=global_rotations(rot).astype(float)
    native=base['anatomical_keypoints'].astype(float);world=native+base['translation'][:,None]
    rest=lambda v,R:np.einsum('tji,tj->ti',R,v)
    ve=rest(world[:,e]-world[:,s],old[:,s]);vw=rest(world[:,w]-world[:,e],old[:,e])
    vh=rest(world[:,palm]-world[:,w],old[:,w]);E=world[:,s]+np.einsum('tij,tj->ti',new[:,s],ve)
    W=E+np.einsum('tij,tj->ti',new[:,e],vw);H=W+np.einsum('tij,tj->ti',new[:,w],vh)
    p=base['keypoints'].astype(float).copy();a=native.copy()
    for j,q in [(e,E),(w,W),(palm,H)]:a[:,j]=q-base['translation']
    for cj,nj,R,j in [(ce,e,new[:,s],e),(cw,w,new[:,e],w)]:
        offset=rest(base['keypoints'][:,cj]-native[:,nj],old[:,s if cj==ce else e])
        p[:,cj]=a[:,j]+np.einsum('tij,tj->ti',R,offset)
    return {**base,'smpl_rotations':rot,'anatomical_keypoints':a,'keypoints':p}


def arm_candidates(base,h,phase,side,mb):
    n=len(h['frame_index']);s,e,w=16+side,18+side,20+side;ce,cw=7+side,9+side
    world=base['anatomical_keypoints'].astype(float)+base['translation'][:,None]
    coco=base['keypoints'].astype(float)+base['translation'][:,None];S=world[:,s]
    l1=float(np.median(np.linalg.norm(world[:,e]-S,axis=-1)));l2=float(np.median(np.linalg.norm(world[:,w]-world[:,e],axis=-1)))
    leg=native_leg(base['anatomical_keypoints']);f=float(base['focal']);principal=base['principal'];sc=scores(h)
    xy=np.einsum('tij,tkj->tki',base['background_transforms'],np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
    target=[];support=[];times=h['source_time']
    for nj,cj in [(e,ce),(w,cw)]:
        residual=np.zeros((n,2));valid=(sc[:,cj]>.5)&np.isfinite(xy[:,cj]).all(-1);ix=np.flatnonzero(valid)
        if len(ix)>=2:
            raw=xy[ix,cj]-project(coco[ix,cj],f,principal)
            residual=PchipInterpolator(times[ix],raw,axis=0)(np.clip(times,times[ix[0]],times[ix[-1]]))
            residual[ix]=raw
        target.append(project(world[:,nj],f,principal)+residual);support.append(int(len(ix)))
    target=np.stack(target,axis=1);Q=np.empty((n,4,2,3));clamped=np.zeros((n,4),bool)
    for t in range(n):
        elbow_roots,ec=sphere_on_ray(S[t],target[t,0],l1,f,principal)
        for ie,E in enumerate(elbow_roots):
            wrists,wc=sphere_on_ray(E,target[t,1],l2,f,principal)
            for iw,W in enumerate(wrists):Q[t,2*ie+iw]=[E-S[t],W-S[t]];clamped[t,2*ie+iw]=ec or wc
    depth=np.stack([Q[:,:,0,2]/l1,(Q[:,:,1,2]-Q[:,:,0,2])/l2],-1)
    roll=Rotation.from_euler('z',np.arctan2(base['background_transforms'][:,1,0],base['background_transforms'][:,0,0])).as_matrix()
    ms,me,mw=[11,12,13] if side==0 else [14,15,16]
    gate,_=phase_window(h,phase);confidence=np.minimum(sc[:,ce],sc[:,cw]);outputs=[]
    links=(np.diff(h['frame_index'])==1)&(np.diff(times)>0)&(np.diff(times)<.1)&(np.diff(h['track_id'])==0)
    observed,unused=direction_returns(h['keypoints'][:,cw]-h['keypoints'][:,ce],links)
    original_returns,_=direction_returns(world[:,w]-world[:,e],links)
    for name,pred in mb.items():
        p=pred['keypoints'];u=np.einsum('tij,tj->ti',roll,p[:,me]-p[:,ms]);v=np.einsum('tij,tj->ti',roll,p[:,mw]-p[:,me])
        u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-8);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)
        desired=np.stack([u[:,2],v[:,2]],-1)
        node=.08*(.25+.75*gate[:,None])*confidence[:,None]*((depth-desired[:,None])**2).sum(-1)
        for strength in [1.,4.]:
            rays.CONFIG={**rays.CONFIG,'velocity_weight':2.*strength,'acceleration_weight':.50*strength}
            path=rays.depth_path(Q,node,times,np.zeros(n),leg)
            q=Q[np.arange(n),path];native=np.zeros((n,17,3));native[:,ms]=S;native[:,me]=S+q[:,0];native[:,mw]=S+q[:,1]
            rot,adapter=native_graft(base,dict(keypoints=native,frame_index=base['frame_index'],source_time=base['source_time']),np.tile(np.eye(3),(n,1,1)),side)
            z=apply_arm(base,rot,side);m,uv=metrics(z,h,base['background_transforms'],z['translation'],base['contact_weight'])
            error=np.linalg.norm(uv[:,[ce,cw]]-h['keypoints'][:,[ce,cw]],axis=-1)
            reliable=sc[:,[ce,cw]]>.5
            source=float(np.percentile(error[reliable],95)) if reliable.any() else 0.
            a=z['anatomical_keypoints'];returns,angles=direction_returns(a[:,w]-a[:,e],links)
            unsupported=[int(t) for t in returns if t not in observed and t not in original_returns]
            step=np.linalg.norm(np.diff(a[:,[e,w]]-a[:,s,None],axis=0),axis=-1)/leg
            outputs.append(dict(name=name,strength=strength,z=z,metrics=m,
                arm_source_p95_px=source,unsupported_returns=unsupported,
                max_step_leg=float(step.max()),max_forearm_step_degrees=float(np.rad2deg(angles).max()),
                depth_path=path.tolist(),ray_clamped_frames=int(clamped[np.arange(n),path].sum()),
                source_support=support,adapter=adapter))
    selected=min(outputs,key=lambda c:(len(c['unsupported_returns']),c['arm_source_p95_px']+.1*c['max_step_leg']))
    z=selected.pop('z');summary={k:v for k,v in selected.items()}
    return z,dict(side=side,selected=summary,candidates=[{k:v for k,v in c.items() if k!='z'} for c in outputs])


def fit(rec):
    cid=rec['id'];source=ROOT/'ground'/f'{cid}.npz';mbpath=ROOT/'motionbert'/f'{cid}.npz';dest=ROOT/'combined'/f'{cid}.npz'
    if not source.exists() or not mbpath.exists():return False
    hashes=dict(ground=sha(source),motionbert=sha(mbpath),solver=sha(__file__),ray_code=sha(Path(__file__).with_name('arm_depth_rays.py')),adapter_code=sha(Path(__file__).with_name('arm_cocking_native.py')))
    if dest.exists() and dest.with_suffix('.json').exists():
        old=json.loads(dest.with_suffix('.json').read_text());assert old['hashes']==hashes;return True
    start=time.monotonic();base=load(source);h=inputs(rec);m=load(mbpath)
    mb={name:dict(keypoints=m[tag],frame_index=m['frame_index'],source_time=m['source_time']) for name,tag in [('single','single'),('flip','flip')]}
    for pred in mb.values():
        assert np.array_equal(pred['frame_index'],h['frame_index']) and np.array_equal(pred['source_time'],h['source_time'])
    result=base;arms=[]
    for side in range(2):result,report=arm_candidates(result,h,rec['phase'],side,mb);arms.append(report)
    assert np.array_equal(result['translation'],base['translation'])
    assert np.array_equal(result['anatomical_keypoints'][:,:18],base['anatomical_keypoints'][:,:18])
    assert np.array_equal(result['foot_keypoints'],base['foot_keypoints'])
    metrics_after,uv=metrics(result,h,base['background_transforms'],base['translation'],base['contact_weight'])
    save_geometry(dest,result,base['translation'],uv,base['contact_weight'],base['normal'],float(base['pilot_support_offset']))
    report=dict(id=cid,hashes=hashes,config=CONFIG,arms=arms,metrics=metrics_after,
        frames=len(h['frame_index']),output_sha256=sha(dest),seconds=time.monotonic()-start,
        motionbert_applied=True,both_arms_processed=True,body_and_feet_unchanged=True,
        actual_3d_accuracy_validated=False)
    save(dest.with_suffix('.json'),report);print('COMBINED',cid,round(report['seconds'],2),'s',flush=True)
    return True


def run(worker,workers,once=False):
    records=manifest()['records'][worker::workers];failures={}
    while True:
        pending=[]
        for r in records:
            try:
                if not fit(r):pending.append(r['id'])
            except Exception as e:
                failures[r['id']]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        save(ROOT/f'arm_worker{worker}.json',dict(total=len(records),pending=pending,failures=failures))
        if not pending or once:return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--once',action='store_true');p.add_argument('--id')
    a=p.parse_args()
    if a.id:fit(next(r for r in manifest()['records'] if r['id']==a.id))
    else:run(a.worker,a.workers,a.once)
