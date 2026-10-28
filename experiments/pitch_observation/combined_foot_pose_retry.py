"""Rigid shoe pose/contact refinement against original 2D and native leg reach.

One fixed shoe shape is fitted per sequence. Actual foot rotations, ankle
positions and contact anchors are optimized before native fixed-length leg IK.
Upper-body MotionBERT rotations, body shape and camera remain fixed.
"""
from common import *
from combined_population import ROOT,manifest,inputs
from hybrid_initializer import global_rotations
from combined_contact_ik import inverse_legs
from support_transfer_root import metrics,save_geometry
from scipy.spatial.transform import Rotation
import argparse,traceback


def fit(rec,choice):
    import torch
    cid=rec['id'];source=ROOT/choice['fit_path'];folder=ROOT/'foot_pose_v3';dest=folder/'combined'/f'{cid}.npz'
    if dest.exists() and dest.with_suffix('.json').exists():return
    base=load(source);h=inputs(rec);groundpath=source.parents[1]/'ground'/source.name;oldground=load(groundpath)
    gm=json.loads(groundpath.with_suffix('.json').read_text());notes=gm['contact_episodes'];N=len(h['frame_index']);E=len(notes);leg=float(base['model_leg'])
    G=global_rotations(base['smpl_rotations']);a=base['anatomical_keypoints'].astype(float);tr=base['translation'].astype(float)
    world=a+tr[:,None];feet=base['foot_keypoints'].astype(float)+tr[:,None];offsets=np.zeros((2,3,3));R0=np.zeros((N,2,3,3))
    for side in range(2):
        js=np.array([side,side+2,side+4]);local=np.einsum('tji,tkj->tki',G[:,7+side],feet[:,js]-world[:,7+side,None]);good=base['contact_weight'][:,js].sum(1)>.1
        offsets[side]=np.median(local[good] if good.any() else local,axis=0)
        X=offsets[side]-offsets[side].mean(0);Y=feet[:,js]-feet[:,js].mean(1,keepdims=True)
        U,S,Vt=np.linalg.svd(np.einsum('ki,tkj->tij',X,Y));R=np.swapaxes(Vt,-1,-2)@np.swapaxes(U,-1,-2)
        sign=np.linalg.det(R);Vt[sign<0,-1]*=-1;R0[:,side]=np.swapaxes(Vt,-1,-2)@np.swapaxes(U,-1,-2)
    # Smooth orientation initialization only; every final pose is reoptimized
    # against observations, reach and contact. No output coordinate filtering.
    from scipy.ndimage import gaussian_filter1d
    for side in range(2):
        q=Rotation.from_matrix(R0[:,side]).as_quat()
        for t in range(1,N):
            if q[t]@q[t-1]<0:q[t]*=-1
        q=gaussian_filter1d(q,1.5,axis=0);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        R0[:,side]=Rotation.from_quat(q).as_matrix()
    chosen=np.full((N,2),-1,int)
    for t in range(N):
        for side in range(2):
            active=[i for i,e in enumerate(notes) if e['marker']%2==side and e['start']<=t<=e['end']]
            if active:chosen[t,side]=max(active,key=lambda i:oldground['support_episode_weights'][t,i])
    T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
    r0=T(R0);O=T(offsets/leg);ankle0=T(world[:,[7,8]]/leg);hip=T(world[:,[1,2]]/leg)
    L=np.array([np.median(np.linalg.norm(a[:,4+s]-a[:,1+s],axis=-1)+np.linalg.norm(a[:,7+s]-a[:,4+s],axis=-1)) for s in range(2)])/leg
    maximum=T(L-.006);minimum=T(L*.22);native_foot_rest=T(np.stack([np.median(np.einsum('tji,tj->ti',G[:,7+s],a[:,10+s]-a[:,7+s]),axis=0) for s in range(2)])/leg);normal0=T(base['normal']);reference=T(np.median(oldground['support_episode_anchors'],axis=0)/leg);height0=T(float(np.median(oldground['support_episode_anchors'],axis=0)@base['normal']+base['pilot_support_offset'])/leg);normal_delta=torch.nn.Parameter(torch.zeros(2,device='cuda:0'));height_delta=torch.nn.Parameter(T(0.))
    delta=torch.nn.Parameter(torch.zeros((N,2,3),device='cuda:0'));free=torch.nn.Parameter(torch.zeros((N,2,3),device='cuda:0'))
    anchors=torch.nn.Parameter(T(oldground['support_episode_anchors']/leg));initial_anchor=T(oldground['support_episode_anchors']/leg)
    ids=torch.as_tensor(np.maximum(chosen,0),device='cuda:0');on=T((chosen>=0).astype(float));marker_local=np.zeros((N,2),int)
    for t in range(N):
        for s in range(2):
            if chosen[t,s]>=0:marker_local[t,s]=notes[chosen[t,s]]['marker']//2
    pick=torch.as_tensor(marker_local,device='cuda:0');sideids=torch.arange(2,device='cuda:0')[None].expand(N,-1)
    xy=np.einsum('tij,tkj->tki',base['background_transforms'],np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
    focal=float(base['focal']);principal=T(base['principal']);sc=scores(h);target=xy[:,[15,16,20,21,22,23,24,25]].copy()
    for s in range(2):
        native=world[:,7+s];coco=base['keypoints'][:,15+s]+tr
        target[:,s]=native[:,:2]/native[:,2:]*focal+base['principal']+xy[:,15+s]-(coco[:,:2]/coco[:,2:]*focal+base['principal'])
    target=T(np.nan_to_num(target));weight=T(sc[:,[15,16,20,21,22,23,24,25]]**2*np.array([.8,.8,1.,1.,1.,1.,1.,1.]))
    height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));primary=torch.as_tensor(pick,device='cuda:0')
    dt30=T(np.diff(h['source_time'])*30);dtmid=(dt30[1:]+dt30[:-1])/2
    rotation_limit=2*np.sqrt(2)*torch.sin(.8*dt30/2)
    parameters=[delta,free,anchors,normal_delta,height_delta];optimizer=torch.optim.Adam(parameters,lr=T(.012),capturable=True,foreach=False)
    from pitch_study_fit import stable_rotation
    def evaluate():
        angles=.06*torch.tanh(normal_delta);normal=stable_rotation(torch.stack([angles[0],angles[0]*0,angles[1]]))@normal0;plane=height0+.08*torch.tanh(height_delta)-(reference*normal).sum()
        R=r0@stable_rotation(2.8*torch.tanh(delta));A=anchors-((anchors*normal).sum(-1)+plane-.012)[:,None]*normal
        chosen_anchor=A[ids];local=O[sideids,primary];point=(R@local[:,:,:,None])[...,0]
        ankle=(ankle0+1.5*torch.tanh(free))*(1-on[...,None])+(chosen_anchor-point)*on[...,None]
        foot=ankle[:,:,None]+(R[:,:,None]@O[None,:,:,:,None])[...,0]
        ordered=foot.transpose(1,2).reshape(N,6,3);points=torch.cat([ankle,ordered],dim=1)
        uv=points[...,:2]/points[...,2:].clamp_min(.2)*focal+principal;err=(uv-target)/height
        image=((torch.sqrt(err.square().sum(-1)+.002**2)-.002)*weight).sum()/weight.sum().clamp_min(1)
        distance=torch.linalg.vector_norm(ankle-hip,dim=-1)
        reach=(torch.relu(distance-maximum).square()+torch.relu(minimum-distance).square()).mean()
        signed=(ordered*normal).sum(-1)+plane;ankle_height=(ankle*normal).sum(-1)+plane;bone_foot=ankle+(R@native_foot_rest[None,:,:,None])[...,0];bone_height=(bone_foot*normal).sum(-1)+plane;penetration=torch.relu(-signed-.006).square().mean()+torch.relu(.04-ankle_height).square().mean()+torch.relu(.012-bone_height).square().mean()
        rotation_step=torch.linalg.vector_norm((R[1:]-R[:-1]).reshape(N-1,2,9),dim=-1)
        rotation_jump=torch.relu(rotation_step-rotation_limit[:,None]).square().mean()
        temporal=((R[2:]+R[:-2]-2*R[1:-1])/dtmid[:,None,None,None].square()).square().mean()
        velocity=(ankle[1:]-ankle[:-1])/dt30[:,None,None]
        acceleration=(velocity[1:]-velocity[:-1])/dtmid[:,None,None]
        ankle_jump=torch.relu(torch.linalg.vector_norm(acceleration,dim=-1)-.12).square().mean()
        ankle_smooth=acceleration.square().mean()
        shift=(ankle-ankle0).square().mean();prior=(A-initial_anchor).square().mean()
        loss=image+3000*reach+1500*penetration+.08*temporal+100*rotation_jump+2000*ankle_jump+.03*ankle_smooth+.004*shift+.0005*prior+.0005*torch.tanh(normal_delta).square().mean()+.0005*torch.tanh(height_delta).square()
        return loss,(R,ankle,ordered,A,normal,plane)
    def step():
        optimizer.zero_grad(set_to_none=False);loss,_=evaluate();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step();return loss
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=step()
    for i in range(4,2400):optimizer.param_groups[0]['lr'].fill_(.012*(.05+.95*(1-i/2399)**2));graph.replay()
    with torch.no_grad():loss,result=evaluate()
    assert torch.isfinite(loss)
    R,ankle,foot,A,normal,plane=[x.detach().cpu().numpy() for x in result];ankle*=leg;foot*=leg;A*=leg;plane=float(plane)*leg
    rot=base['smpl_rotations'].astype(float).copy();native=a.copy();f=np.empty_like(feet)
    for side in range(2):
        rot[:,7+side]=np.swapaxes(G[:,4+side],-1,-2)@R[:,side]
        rest=np.median(np.einsum('tji,tj->ti',G[:,7+side],a[:,10+side]-a[:,7+side]),axis=0)
        native[:,10+side]=a[:,7+side]+np.einsum('tij,j->ti',R[:,side],rest)
        js=np.array([side,side+2,side+4]);f[:,js]=a[:,7+side,None]+np.einsum('tij,kj->tki',R[:,side],offsets[side])
    bundle={**oldground,'support_episode_anchors':A,'desired_ankle_world':ankle,'desired_ankle_valid':np.ones((N,2),bool)}
    z={**base,'smpl_rotations':rot,'anatomical_keypoints':native,'foot_keypoints':f,'normal':normal,'pilot_support_offset':np.array(plane)}
    z,ik=inverse_legs(z,bundle,notes)
    z.update(shoe_local_offsets=offsets.transpose(1,0,2).reshape(6,3),shoe_ankle_joints=np.array([7,8,7,8,7,8]),
        shoe_marker_model=np.array('Sequence-fixed rigid shoe pose fitted to 2D with exact contact and native leg reach'))
    m,uv=metrics(z,h,z['background_transforms'],z['translation'],z['contact_weight']);save_geometry(dest,z,z['translation'],uv,z['contact_weight'],z['normal'],float(z['pilot_support_offset']))
    gp=folder/'ground'/source.name;atomic_npz(gp,**{**z,**{k:v for k,v in bundle.items() if k.startswith('support_') or k=='primary_support_episode'}})
    gm.update(source_sha256=sha(groundpath),output_sha256=sha(gp));gm['config']={**gm['config'],'solver_sha256':sha(__file__),'prior_ground_solver_sha256':gm['config']['solver_sha256'],'version':'rigid-shoe-pose-and-contact-v3'};save(gp.with_suffix('.json'),gm)
    meta=json.loads(source.with_suffix('.json').read_text());meta['hashes']['ground']=sha(gp);meta.update(output_sha256=sha(dest),contact_ik=ik,
        contact_ik_hashes=dict(source=sha(source),solver=sha(Path(__file__).with_name('combined_contact_ik.py'))),
        rigid_sole_refit=dict(solver_sha256=sha(__file__),source_sha256=sha(source),constant_shape=True,foot_pose_optimized=True,rotation_and_ankle_temporal_optimization=True,
            no_output_coordinate_filter=True,rotation_step_target_radians_at30fps=.8,ankle_velocity_change_target_leg_at30fps=.12,
            original_2d_used=True,body_shape_and_upper_rotations_unchanged=True),metrics=m)
    save(dest.with_suffix('.json'),meta);print('FOOT POSE',cid,'root',ik['root_delta_max_leg'],flush=True)


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();import torch;torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.22,0)
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--id');a=p.parse_args()
    selected=json.loads((ROOT/'selection_frozen.json').read_text())['selected']
    for r in manifest()['records'][a.worker::a.workers]:
        if a.id and r['id']!=a.id:continue
        try:fit(r,selected[r['id']])
        except Exception:print(traceback.format_exc(),flush=True)
