"""Fixed-shape support reconstruction with exact primary toe contacts.

The primary-support equation is part of reconstruction, before arm fitting
and packaging. Saved root and native FK geometry obey it; the renderer never
edits trajectories. Secondary contacts, image agreement, penetration and
transitions are independently audited rather than inferred from this equation.
"""
from common import *
from combined_population import ROOT as BASE_ROOT,manifest,inputs
from combined_contact_evidence import episodes as all_episodes
def episodes(h,phase,bg):
    weights,seeds,notes=all_episodes(h,phase,bg)
    keep=[i for i,e in enumerate(notes) if e['marker']<2]
    return weights[:,keep],seeds[keep],[notes[i] for i in keep]
PROFILE=int(os.environ.get("KBO_COMBINED_RETRY_PROFILE","5"))
ROOT=BASE_ROOT/f"retry_{PROFILE}"
ROOT.mkdir(exist_ok=True)
from pitch_study_fit import StudyFitter,stable_rotation,spline_basis
from support_transfer_root import save_geometry,metrics
from stride_review import native_leg
from scipy.spatial.transform import Rotation
import argparse,traceback,fcntl,gc

CONFIG=dict(version='combined-coherent-toe-support-v1',steps=2400,profile=PROFILE,
    shape='fixed sequence SMPL shape',upper_limb_rotations='unchanged until MotionBERT stage',
    primary_contact='exact root=constant episode anchor - FK foot point',
    secondary_contact='finite-confidence quadratic joint fitting; independently screened',
    plane='one fitted plane per clip, not calibrated ground',
    source='final unchanged RTMPose Halpe26 observations',
    supported_marker='image-supported big toe, release on missing/moving observations',
    stride='no target population stride or limb stretching',cuda_graph=True)


def assignments(notes,weights):
    n=len(weights);primary=np.zeros_like(weights);active=None
    for t in range(n):
        candidates=[i for i,e in enumerate(notes) if e['start']<=t<=e['end']]
        if active not in candidates:
            active=max(candidates,key=lambda i:(notes[i]['end']-t,float(weights[:,i].sum()))) if candidates else None
        if active is not None:primary[t,active]=1
    return primary


def fit(rec,body,steps=None):
    import torch
    steps=steps or CONFIG['steps'];cid=rec['id'];path=ROOT/'ground'/f'{cid}.npz'
    config={**CONFIG,'steps':steps,'solver_sha256':sha(__file__),
        'detector_sha256':sha(Path(__file__).with_name('combined_contact_evidence.py')),
        'population_helpers_sha256':sha(Path(__file__).with_name('combined_population.py'))}
    if path.exists() and path.with_suffix('.json').exists():
        old=json.loads(path.with_suffix('.json').read_text())
        assert old['config']==config and old['source_sha256']==rec['source_sha256'],'Changed production inputs need a new version'
        return old
    base=load(OUT/rec['source']);h=inputs(rec);N=len(h['frame_index']);leg=native_leg(base['anatomical_keypoints'])
    assert sha(OUT/rec['source'])==rec['source_sha256']
    weights,seeds,notes=episodes(h,rec['phase'],base['background_transforms'])
    # A high knee plus a toe visibly above the other foot is an airborne
    # pause, not a second support. Preserve the rejected evidence in a log.
    xy=np.einsum('tij,tkj->tki',base['background_transforms'],np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
    height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])))
    kept=[]
    for i,e in enumerate(notes):
        ids=np.arange(e['start'],e['end']+1);side=e['marker']%2
        high=(xy[ids,13+side,1]-xy[ids,11+side,1])<.08*height
        toe_above=xy[ids,20+side,1]<xy[ids,21-side,1]-.04*height
        if not (high&toe_above).mean()>.6:kept.append(i)
    notes=[notes[i] for i in kept];weights=weights[:,kept];seeds=seeds[kept]
    if not notes:raise ValueError('No image-supported contact; cannot certify this clip')
    primary=assignments(notes,weights);markers=np.array([e['marker'] for e in notes],int)
    cw=np.zeros((N,6))
    for i,j in enumerate(markers):cw[:,j]=np.maximum(cw[:,j],weights[:,i])
    T=lambda x:torch.as_tensor(x,dtype=torch.float32,device=body.device)
    beta=base['smpl_betas'];body.prepare_shape(beta);betas=T(np.repeat(beta[None],N,0))
    r0=T(base['smpl_rotations']);B=T(spline_basis(N,float(h['fps']),spacing=.08))
    tr0=base['translation']/leg
    controls=torch.nn.Parameter(T(np.linalg.lstsq(B.cpu().numpy(),tr0,rcond=None)[0]))
    delta=torch.nn.Parameter(torch.zeros((N,24,3),device=body.device))
    knee0=Rotation.from_matrix(base['smpl_rotations'][:,[4,5]].reshape(-1,3,3)).as_rotvec().reshape(N,2,3)
    prob=np.clip((np.clip(knee0[:,:,0],.02,2.7)+.08)/2.88,1e-4,1-1e-4)
    flex=torch.nn.Parameter(T(np.log(prob/(1-prob))))
    side=torch.nn.Parameter(T(np.arctanh(np.clip(knee0[:,:,1:]/.18,-.95,.95))))
    f0=base['foot_keypoints']/leg+tr0[:,None]
    anchors=torch.nn.Parameter(T(np.array([np.median(f0[max(0,k-1):min(N,k+2),j],axis=0) for j,k in zip(markers,seeds)])))
    pitch0=float(np.arctan2(-base['normal'][2],-base['normal'][1]));pitch=torch.nn.Parameter(T(0.))
    plane=torch.nn.Parameter(T(float(base['pilot_support_offset'])/leg))
    logf=torch.nn.Parameter(T(0.));limits=np.zeros((24,3),np.float32)
    limits[0]=.45;limits[[1,2]]=.95;limits[[3,6,9,12]]=.28;limits[[7,8,10,11]]=3.0
    limits=T(limits);W=T(weights);P=T(primary);ix=torch.as_tensor(markers,device=body.device)
    bg=base['background_transforms'];xy=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
    target=T(np.nan_to_num(np.concatenate([xy[:,:17],xy[:,20:26]],axis=1)))
    sc=scores(h);observation=np.concatenate([sc[:,:17]**2,sc[:,20:26]**2*80.],axis=1)
    observation[:,:5]*=.2;observation[:,7:11]*=6.;observation=T(observation)
    height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])))
    dt=T(np.diff(h['source_time']));middle=(dt[1:]+dt[:-1])/2
    good=np.diff(h['frame_index'])==1;triples=torch.as_tensor(np.flatnonzero(good[1:]&good[:-1]),device=body.device)
    initial_stride=T((base['anatomical_keypoints'][:,7]-base['anatomical_keypoints'][:,8])/leg)
    principal=T(base['principal']);initial_root=T(tr0[0]);knee_ref=T(knee0)
    optimizer=torch.optim.Adam([{'params':[controls,anchors,plane,delta,flex,side,pitch],'lr':T(.012)},
        {'params':[logf],'lr':T(.002)}],capturable=True,foreach=False)
    parameters=[controls,anchors,plane,delta,flex,side,pitch,logf]

    def evaluate():
        angles=limits*torch.tanh(delta);rot=r0@stable_rotation(angles)
        knee=torch.cat([(-.08+2.88*torch.sigmoid(flex))[...,None],.18*torch.tanh(side)],-1)
        rot=torch.cat([rot[:,:4],stable_rotation(knee),rot[:,6:]],1)
        p,a,f=body.model(rot,betas);p=p/leg;a=a/leg;f=f/leg
        free=B@controls
        # Differentiable exact primary support, not a display translation.
        tr=free+((anchors[None]-f.index_select(1,ix)-free[:,None])*P[...,None]).sum(1)
        theta=pitch0+.45*torch.tanh(pitch);normal=torch.stack([theta*0,-torch.cos(theta),-torch.sin(theta)])
        focal=float(base['focal'])*torch.exp(.3*torch.tanh(logf))
        points=torch.cat([p,f],1)+tr[:,None];uv=points[...,:2]/points[...,2:].clamp_min(.2)*focal+principal
        err=(uv-target)/height
        image=((torch.sqrt(err.square().sum(-1)+.005**2)-.005)*observation).sum()/observation.sum().clamp_min(1)
        contact=f.index_select(1,ix)+tr[:,None]-anchors[None]
        secondary=(contact.square().sum(-1)*W).sum()/W.sum().clamp_min(1)
        signed=((f+tr[:,None])*normal).sum(-1)+plane
        anchorheight=(anchors*normal).sum(-1)+plane
        floor=torch.relu(-signed-.01).square().mean()
        normal_loss=(anchorheight-.015).square().mean()
        root=(a[:,0]+tr);v=(root[1:]-root[:-1])/dt[:,None];acc=(v[1:]-v[:-1])/middle[:,None]
        smooth=(acc.index_select(0,triples)/30).square().mean()
        pose=angles.square().mean();temporal=(angles[2:]+angles[:-2]-2*angles[1:-1]).square().mean()
        stride=(a[:,7]-a[:,8]-initial_stride).square().mean()
        distance=(torch.linalg.vector_norm(a[:,7]-a[:,8],dim=-1)-torch.linalg.vector_norm(initial_stride,dim=-1)).square().mean()
        loss=image+150*secondary+80*floor+5.*normal_loss+.015*smooth+.002*pose+.003*temporal+.05*stride+.20*distance
        loss=loss+.002*(knee-knee_ref).square().mean()+.002*(knee[2:]+knee[:-2]-2*knee[1:-1]).square().mean()+.0001*logf.square()+.00001*((tr[0]-initial_root)/abs(initial_root[2]).clamp_min(1)).square().mean()
        return loss,(p,a,f,tr,rot,focal,normal)

    def train():
        optimizer.zero_grad(set_to_none=False);loss,_=evaluate();loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step();return loss
    start=time.monotonic();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):train()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=train()
    for step in range(4,steps):
        decay=.05+.95*(1-step/(steps-1))**2
        optimizer.param_groups[0]['lr'].fill_(.012*decay);optimizer.param_groups[1]['lr'].fill_(.002*decay)
        graph.replay()
        if step%200==0 and not torch.isfinite(loss):raise ValueError('Nonfinite ground fitting')
    with torch.no_grad():loss,out=evaluate()
    p,a,f,tr,rot,focal,normal=[x.detach().cpu().numpy() for x in out]
    z={**base,'keypoints':p*leg,'anatomical_keypoints':a*leg,'foot_keypoints':f*leg,
       'translation':tr*leg,'smpl_rotations':rot,'focal':focal,'model_leg':np.array(leg)}
    m,uv=metrics(z,h,bg,z['translation'],cw)
    save_geometry(path,z,z['translation'],uv,cw,normal,float(plane.detach())*leg)
    with np.load(path,allow_pickle=False) as q:payload={k:q[k] for k in q.files}
    atomic_npz(path,**payload,primary_support_episode=np.where(primary.any(1),primary.argmax(1),-1),
        support_episode_weights=weights,support_episode_markers=markers,
        support_episode_anchors=anchors.detach().cpu().numpy()*leg)
    result=dict(id=cid,config=config,source_sha256=rec['source_sha256'],output_sha256=sha(path),
        frames=N,contact_episodes=notes,metrics=m,seconds=time.monotonic()-start,
        actual_3d_accuracy_validated=False,primary_support_exact_by_construction=True)
    save(path.with_suffix('.json'),result);print('GROUND',cid,'seconds',round(result['seconds'],1),flush=True)
    return result


def run(records,worker):
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(float(os.environ.get('KBO_COMBINED_RETRY_MEMORY','.48')),0)
    body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False);failed={}
    for k,rec in enumerate(records):
        folder=ROOT/'locks';folder.mkdir(exist_ok=True)
        with (folder/(rec['id']+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            try:fit(rec,body)
            except Exception as e:
                failed[rec['id']]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
        save(ROOT/f'ground_worker{worker}.json',dict(processed=k+1,total=len(records),failed=failed))
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('--id');p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();r=manifest()['records'][a.worker::a.workers]
    if a.id:r=[x for x in manifest()['records'] if x['id']==a.id]
    run(r[:a.limit or None],a.worker)
