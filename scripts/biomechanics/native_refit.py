"""A frozen native-joint reprojection experiment; never edits published poses.

The previous fitting targets WHAM surface landmarks. This experiment fits
native hip/knee/ankle joint centers to original image detections, retaining
shape, root/camera, upper body and explicit depth/rotation priors. Better image
fit is not ground-truth 3D accuracy. Operational adoption gates are fixed here.
Run in the existing local torch/smplx environment; licensed body assets stay local.
"""
import argparse,gzip,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
STUDY=REPO.parent/'research/pitch_observation_20261028'
OUT=REPO.parent/'research/biomechanics_20261029/native_refit'


def main():
    sys.path.insert(0,str(REPO/'experiments/pitch_observation'))
    from runtime import ensure_runtime
    ensure_runtime()
    import torch
    from smplx import SMPL
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--steps',type=int,default=100);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    reviews=json.loads((REPO/'data/metrics/source_reviews_20261029.json').read_text())['clips']
    if args.limit:reviews=reviews[:args.limit]
    clips={c['id']:c for c in json.loads((REPO/'data/clips.json').read_text())['clips']}
    catalog={r['dataset_clip_id']:r for r in __import__('csv').DictReader(open(REPO.parent/'dataset/clips.csv'))}
    # selection_frozen.json is an earlier intermediate. The actually published
    # subject/tail/contact corrections are in selection.json; verify its hash.
    selection=json.loads((STUDY/'combined_population_v1/selection.json').read_text())['selected']
    body=SMPL(model_path=str(REPO.parent/'external/WHAM/dataset/body_models/smpl'),gender='neutral',create_transl=False).to('cuda:0').eval();body.requires_grad_(False)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda:0')
    SMPL_IDS=[1,2,4,5,7,8];IMAGE_IDS=[11,12,13,14,15,16]
    from scipy.spatial.transform import Rotation
    reports=[]
    for review in reviews:
        cid=review['clipId'];dest=OUT/(cid+'.npz');meta=dest.with_suffix('.json')
        path=STUDY/'combined_population_v1'/selection[cid]['fit_path'];fit_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        published=json.load(gzip.open(REPO/'data'/clips[cid]['pose3d'],'rt'))
        assert published['source_fit_sha256']==fit_sha,('Wrong reconstruction lineage',cid)
        if dest.exists() and meta.exists():
            with np.load(dest) as saved:complete='foot_keypoints' in saved.files
            old=json.loads(meta.read_text())
            if complete and old['sourceFitSha256']==fit_sha and (cid!='4e09dd724a8c_c0011' or old.get('coordinateOriginCorrected')):reports.append(old);continue
        started=time.monotonic();clip=clips[cid];row=catalog[cid]
        with np.load(path,allow_pickle=False) as z:base={k:z[k] for k in ['smpl_rotations','smpl_betas','anatomical_keypoints','translation','focal','principal','withheld_frames']}
        pose=json.loads((REPO/'data'/clip['pose']).read_text());n=len(base['smpl_rotations']);raw=pose['frames'][:n]
        # Subject-corrected files can use a new crop (including full-image
        # pixels). The legacy clips.csv origin is not valid for those files.
        crop_origin=np.asarray(pose.get('crop_box',[float(row['crop_x1']),float(row['crop_y1'])])[:2],float)
        target=np.zeros((n,6,2));weight=np.zeros((n,6))
        for i,f in enumerate(raw):
            for j,index in enumerate(IMAGE_IDS):
                p=f['keypoints'][index]
                if p and len(p)>=3 and p[2]>=.5 and not f.get('interpolated',[False]*26)[index] and not f.get('outlier',[False]*26)[index]:
                    target[i,j]=np.asarray(p[:2])+crop_origin;weight[i,j]=p[2]**2
        weight[:,:2]*=.2;weight[:,4:]*=.6;weight[np.asarray(base['withheld_frames'],bool)]=0
        R0=tensor(base['smpl_rotations']);beta=np.asarray(base['smpl_betas']);beta=np.median(beta,axis=0) if beta.ndim==2 else beta;betas=tensor(np.repeat(beta[None],n,axis=0));origin=tensor(base['anatomical_keypoints'][:,0]);translation=tensor(base['translation']);principal=tensor(base['principal']);focal=float(base['focal'])
        with torch.no_grad():
            initial=body(body_pose=R0[:,1:],global_orient=R0[:,:1],betas=betas,pose2rot=False).joints[:,:24]
            reference=initial-initial[:,[0]]+origin[:,None]
        fk_error=float(torch.max(abs(reference-tensor(base['anatomical_keypoints']))).item())
        if fk_error>2e-4:raise ValueError('Native FK mismatch '+cid+': '+str(fk_error))
        delta=torch.nn.Parameter(torch.zeros((n,4,3),device='cuda:0'))
        optimizer=torch.optim.Adam([delta],lr=.012)
        # torch.matrix_exp avoids the zero-axis Rodrigues derivative singularity.
        def forward():
            angular=.7*torch.tanh(delta/.7);x,y,z=angular.unbind(-1);zero=torch.zeros_like(x)
            skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(n,4,3,3)
            rotations=R0.clone();rotations[:,[1,2,4,5]]=R0[:,[1,2,4,5]]@torch.matrix_exp(skew)
            output=body(body_pose=rotations[:,1:],global_orient=rotations[:,:1],betas=betas,pose2rot=False,return_verts=True)
            a=output.joints[:,:24];a=a-a[:,[0]]+origin[:,None];points=a[:,SMPL_IDS]+translation[:,None]
            uv=points[...,:2]/points[...,2:]*focal+principal
            vertices=output.vertices-output.joints[:,[0]]+origin[:,None]
            return a,uv,rotations,vertices,angular
        height=max(100,float(row['crop_y2'])-float(row['crop_y1']));goal=tensor(target);w=tensor(weight)
        with torch.no_grad():before=forward()[1].cpu().numpy()
        for step in range(args.steps):
            optimizer.zero_grad();a,uv,rotations,verts,angular=forward();image=(torch.sqrt(((uv-goal)/height).square().sum(-1)+.002**2)*w).sum()/w.sum().clamp_min(1)
            displacement=(a[:,SMPL_IDS,2]-reference[:,SMPL_IDS,2]).square().mean()
            regularizer=.015*angular.square().mean()+.012*displacement
            temporal=.004*(angular[1:]-angular[:-1]).square().mean() if n>1 else 0
            loss=image+regularizer+temporal;loss.backward();optimizer.step()
        from smplx.vertex_ids import vertex_ids
        foot_indices=[vertex_ids['smplh'][key] for key in ['LBigToe','RBigToe','LSmallToe','RSmallToe','LHeel','RHeel']]
        with torch.no_grad():a,uv,rotations,verts,angular=forward();feet=verts[:,foot_indices].cpu().numpy();a=a.cpu().numpy();uv=uv.cpu().numpy();rotations=rotations.cpu().numpy()
        good=weight>.05;err0=np.linalg.norm(before-target,axis=-1)[good];err1=np.linalg.norm(uv-target,axis=-1)[good]
        model=json.load(gzip.open(REPO/'data'/clip['pose3d'],'rt'));sceneR=np.asarray(model['scene']['rotation_camera_to_scene']);sceneOrigin=np.asarray(model['scene']['origin_camera'])
        old_scene=np.einsum('ij,tkj->tki',sceneR,base['anatomical_keypoints']+base['translation'][:,None]-sceneOrigin);new_scene=np.einsum('ij,tkj->tki',sceneR,a+base['translation'][:,None]-sceneOrigin)
        leg=np.median(np.linalg.norm(a[:,1]-a[:,4],axis=-1)+np.linalg.norm(a[:,4]-a[:,7],axis=-1))
        old_floor=np.maximum(0,-old_scene[:,[7,8],1]).max();new_floor=np.maximum(0,-new_scene[:,[7,8],1]).max()
        native_lengths=np.stack([np.linalg.norm(a[:,h]-a[:,k],axis=-1) for h,k in [(1,4),(4,7),(2,5),(5,8)]],1)
        bone_variation=float(np.ptp(native_lengths,axis=0).max()/leg)
        adopted=bool(np.median(err1)<.85*np.median(err0) and np.quantile(err1,.95)<=np.quantile(err0,.95)+2 and new_floor-old_floor<.03*leg and bone_variation<2e-5)
        report=dict(clipId=cid,version='native-joint-refit-20261029-v1',steps=args.steps,sourceFitSha256=hashlib.sha256(path.read_bytes()).hexdigest(),coordinateOriginCorrected=bool(np.any(crop_origin!=[float(row['crop_x1']),float(row['crop_y1'])])),usedPoseCropOrigin=True,cropOrigin=crop_origin.tolist(),
            nativeFkError=fk_error,beforeMedianPixels=float(np.median(err0)),afterMedianPixels=float(np.median(err1)),beforeP95Pixels=float(np.quantile(err0,.95)),afterP95Pixels=float(np.quantile(err1,.95)),
            ankleFloorIncreaseLeg=float((new_floor-old_floor)/leg),boneVariationLeg=bone_variation,adoptedForAnalysis=adopted,
            criterion='>=15% native-joint source median gain; p95 <= old+2px; added ankle penetration <3% leg; bone variation <2e-5 leg',
            externalAccuracyValidated=False,semantics='Source-consistency experiment, not 3D ground truth. Native SMPL-X adapter ambiguity remains.',seconds=time.monotonic()-started)
        np.savez_compressed(dest,anatomical_keypoints=a,foot_keypoints=feet,smpl_rotations=rotations,translation=base['translation'],frame_index=np.asarray([f['source_frame'] for f in raw]),source_time=np.asarray([f['source_time'] for f in raw]))
        meta.write_text(json.dumps(report,indent=2)+'\n');reports.append(report);print(cid,'adopt',adopted,'native image median',round(report['beforeMedianPixels'],2),'->',round(report['afterMedianPixels'],2),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(version='native-joint-refit-20261029-v1',clips=reports,adopted=sum(r['adoptedForAnalysis'] for r in reports),gtValidated=False),indent=2)+'\n')


if __name__=='__main__':main()
