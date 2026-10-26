#!/usr/bin/env python3
"""RTMPose Halpe-26 -> official WHAM -> temporal refinement -> numeric publication.

Run in a CUDA-enabled server shell using setup_wham.sh. Never falls back to a
different 3D model. WHAM source/weights and licensed SMPL assets stay outside
the public repository. Camera-relative output is used; SLAM is not assumed.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from refine_motion import atomic_npz, refine_trajectory, refine_3d, segments

COCO17 = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
          'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
          'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
HALPE26 = COCO17 + ['head','neck','hip','left_big_toe','right_big_toe',
                    'left_small_toe','right_small_toe','left_heel','right_heel']
HALPE_EDGES = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(17,18),(18,19),
    (18,5),(18,6),(5,7),(7,9),(6,8),(8,10),(19,11),(19,12),
    (11,13),(13,15),(12,14),(14,16),(15,20),(15,22),(15,24),
    (16,21),(16,23),(16,25)]
VERSION = 'rtmpose-halpe26-wham-v1'


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def paths_for(row):
    name = row['source_clip_id'] + '.npz'
    source = row['source_id']
    return {'pose_2d': Path('poses/2d_halpe26')/source/name,
        'pose_2d_refined': Path('poses/2d_halpe26_refined')/source/name,
        'prior_3d': Path('poses/3d_prior/wham')/source/name,
        'pose_3d': Path('poses/3d/wham')/source/name,
        'motion_quality': Path('quality/motion_wham')/(row['dataset_clip_id']+'.json')}


def coco_input(points, scores, missing, origin):
    """Halpe first 17 are COCO; extra feet/head landmarks stay in 2D assets."""
    if points.ndim != 3 or points.shape[1:] != (26,2) or scores.shape != points.shape[:2]:
        raise ValueError('Expected actual RTMPose Halpe-26 output, not padded COCO-17')
    points = points[:,:17].copy() - np.asarray(origin)
    confidence = scores[:,:17].copy()
    bad = missing[:,:17] | ~np.isfinite(points).all(-1) | ~np.isfinite(confidence)
    confidence[bad] = 0
    points[bad] = 0
    return np.concatenate((points,confidence[:,:,None]),axis=-1).astype(np.float32)


def camera_coordinates(joints):
    """WHAM camera/SMPL uses x right, y down, z away; display uses y up/z toward."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] < 17 or joints.shape[2] != 3:
        raise ValueError('Unexpected WHAM COCO regressor output')
    result = joints[:,:17].copy()
    result *= np.array([1,-1,-1], np.float32)
    result -= ((result[:,11]+result[:,12])/2)[:,None,:]
    return result


def required_assets(root):
    return [root/'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar',
        root/'checkpoints/hmr2a.ckpt', root/'dataset/body_models/smpl/SMPL_NEUTRAL.pkl',
        *[root/'dataset/body_models'/name for name in
          ['J_regressor_wham.npy','J_regressor_h36m.npy','J_regressor_feet.npy','smpl_mean_params.npz']]]


def check(args):
    report = {'backend_2d':'RTMPose Halpe-26','backend_3d':'WHAM', 'errors':[]}
    required = [args.wham_root/'lib/models/wham.py', args.pose_model, *required_assets(args.wham_root)]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            report['errors'].append('Missing: '+str(path))
    try:
        import torch
        report['torch'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
        report['cuda_device_count'] = torch.cuda.device_count()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA devices are not accessible in this session; use the server SSH terminal')
        # Actually allocate and execute on the requested GPU.
        report['cuda_smoke_sum'] = float(torch.ones(16,device=args.device).sum().item())
        report['gpu'] = torch.cuda.get_device_name(torch.device(args.device))
    except Exception as error:
        report['errors'].append('CUDA: '+str(error))
    for module in ['rtmlib','onnxruntime','smplx','joblib','yacs','einops','timm','cv2','scipy','progress','loguru']:
        try:
            __import__(module)
        except Exception as error:
            report['errors'].append(module+': '+str(error))
    if (args.wham_root/'lib/models/wham.py').is_file():
        import importlib
        before = Path.cwd()
        sys.path.insert(0,str(args.wham_root))
        try:
            os.chdir(args.wham_root)
            for module in ['configs.config','lib.models','lib.models.preproc.extractor','lib.data.datasets']:
                try:
                    importlib.import_module(module)
                except Exception as error:
                    report['errors'].append(module+': '+str(error))
        finally:
            os.chdir(before)
            sys.path.remove(str(args.wham_root))
    return report


def infer_halpe(args, row, pose_model, model_hash):
    import cv2
    paths = paths_for(row)
    original = args.dataset/row['pose_2d']
    video = args.dataset/row['crop_video']
    signature = hashlib.sha256((digest(original)+str(video.stat().st_size)+
        str(video.stat().st_mtime_ns)+model_hash+VERSION).encode()).hexdigest()
    target = args.dataset/paths['pose_2d']
    if target.exists():
        with np.load(target,allow_pickle=False) as old:
            if str(old['input_signature']) == signature:
                return {k:old[k].copy() for k in old.files}
    with np.load(original,allow_pickle=False) as raw:
        metadata = {k:raw[k].copy() for k in ['bbox','frame_index','pts','source_time','track_id'] if k in raw}
    count = len(metadata['frame_index'])
    points = np.full((count,26,2),np.nan,np.float32)
    scores = np.zeros((count,26),np.float32)
    origin = np.array([float(row['crop_x1']),float(row['crop_y1'])],np.float32)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        for i in range(count):
            ok,image = capture.read()
            if not ok:
                raise ValueError('Crop and pose frame counts differ: '+row['dataset_clip_id'])
            box = metadata['bbox'][i].copy() - np.tile(origin,2)
            if not np.isfinite(box).all() or min(box[2:]-box[:2]) < 8:
                continue
            keypoints,confidence = pose_model(image,bboxes=box[None])
            if np.asarray(keypoints).shape != (1,26,2) or np.asarray(confidence).shape != (1,26):
                raise ValueError('The supplied ONNX model does not output Halpe-26')
            points[i] = keypoints[0] + origin
            scores[i] = confidence[0]
    finally:
        capture.release()
    values = dict(metadata,keypoints=points,scores=scores,
        missing=~np.isfinite(points).all(-1)|(scores<.2),fps=fps,
        joints=np.array(HALPE26),layout='halpe26',edges=np.asarray(HALPE_EDGES),
        model='RTMPose-m Halpe-26',model_sha256=model_hash,
        source_pose_sha256=digest(original),input_signature=signature,
        coordinate_system='original source pixels',method=VERSION)
    atomic_npz(target,**values)
    return values


def build_tracking(halpe, origin):
    """Reset WHAM recurrent state at gaps, identity changes and unsupported boxes."""
    keypoints = coco_input(halpe['keypoints'],halpe['scores'],halpe['missing'],origin)
    bbox = halpe['bbox'] - np.tile(origin,2)
    good = np.isfinite(bbox).all(-1) & ((bbox[:,2:]-bbox[:,:2]).min(-1)>8)
    good &= (keypoints[:,:,2]>.3).sum(-1)>=6
    tracks = defaultdict(lambda:defaultdict(list))
    groups = segments(halpe['source_time'],halpe.get('track_id'),halpe['frame_index'])
    for group in groups:
        supported = group[good[group]]
        for run in np.split(supported,np.flatnonzero(np.diff(supported)>1)+1):
            if len(run)<2:
                continue
            # Match WHAM's cx, cy, scale parameterization, keeping clip boundaries intact.
            boxes = bbox[run]
            center = (boxes[:,:2]+boxes[:,2:])/2
            scale = (boxes[:,2:]-boxes[:,:2]).max(-1)*1.2/200
            from scipy.ndimage import median_filter
            kernel = min(len(run) if len(run)%2 else len(run)-1, max(1,int(float(halpe['fps'])/2)//2*2+1))
            packed = np.c_[center,scale].astype(np.float32)
            packed = median_filter(packed,size=(kernel,1),mode='nearest')
            tracks[len(tracks)].update(frame_id=run,keypoints=keypoints[run],bbox=packed)
    return tracks


def load_wham(args):
    import torch
    os.chdir(args.wham_root)
    sys.path.insert(0,str(args.wham_root))
    from configs.config import get_cfg_defaults
    from lib.models import build_body_model, build_network
    from lib.models.preproc.extractor import FeatureExtractor
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    cfg.DEVICE = args.device
    cfg.FLIP_EVAL = True
    body = build_body_model(args.device,1)
    network = build_network(cfg,body).eval()
    # Upstream permits absent/incomplete checkpoints. Reject uninitialized learned parameters.
    checkpoint = torch.load(cfg.TRAIN.CHECKPOINT,map_location='cpu')['model']
    missing = [k for k,_ in network.named_parameters() if not k.startswith('smpl.') and k not in checkpoint]
    if missing:
        raise ValueError('WHAM checkpoint missing learned parameters: '+str(missing[:8]))
    del checkpoint
    extractor = FeatureExtractor(args.device,True)
    checkpoint = torch.load('checkpoints/hmr2a.ckpt',map_location='cpu')['state_dict']
    missing = [k for k,_ in extractor.model.named_parameters() if k not in checkpoint]
    if missing:
        raise ValueError('HMR2A checkpoint missing learned parameters: '+str(missing[:8]))
    del checkpoint
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.wham_root,text=True).strip()
    hashes = {str(p.relative_to(args.wham_root)):digest(p) for p in required_assets(args.wham_root)}
    model_hash = hashlib.sha256(json.dumps({'revision':revision,'assets':hashes},sort_keys=True).encode()).hexdigest()
    return cfg,network,extractor,model_hash,{'revision':revision,'asset_sha256':hashes}


def infer_wham(args,row,halpe,cfg,network,extractor,model_hash,provenance):
    import torch
    import cv2
    from lib.data.datasets import CustomDataset
    from lib.utils.imutils import avg_preds
    target = args.dataset/paths_for(row)['prior_3d']
    signature = hashlib.sha256((str(halpe['input_signature'])+model_hash+VERSION+':flip-camera-local').encode()).hexdigest()
    if target.exists():
        with np.load(target,allow_pickle=False) as old:
            if str(old['input_signature']) == signature:
                return {k:old[k].copy() for k in old.files}
    origin = np.array([float(row['crop_x1']),float(row['crop_y1'])],np.float32)
    video = str(args.dataset/row['crop_video'])
    cap = cv2.VideoCapture(video)
    width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    count = len(halpe['keypoints'])
    xyz = np.full((count,17,3),np.nan,np.float32)
    xy = np.full((count,17,2),np.nan,np.float32)
    scores = np.zeros((count,17),np.float32)
    body_pose = np.full((count,23,3,3),np.nan,np.float32)
    root_pose = np.full((count,1,3,3),np.nan,np.float32)
    shape = np.full((count,10),np.nan,np.float32)
    with torch.no_grad():
        tracks = build_tracking(halpe,origin)
        if not tracks:
            raise ValueError('No supported continuous RTMPose track for WHAM: '+row['dataset_clip_id'])
        tracks = extractor.run(video,tracks)
        for track_id,track in tracks.items():
            frame_ids = np.asarray(track['frame_id'],dtype=int)
            # Identity camera angular velocity is only a local-coordinate convention.
            # Do not publish world trajectory from uncalibrated broadcast crops.
            slam = np.zeros((len(frame_ids),7));slam[:,3]=1
            dataset = CustomDataset(cfg,{track_id:track},slam,width,height,float(halpe['fps']))
            _,x,inits,features,mask,root,angvel,_,kwargs = dataset.load_data(0,True)
            flipped = network(x,inits,features,mask=mask,init_root=root,cam_angvel=angvel,return_y_up=True,**kwargs)
            _,x,inits,features,mask,root,angvel,_,kwargs = dataset.load_data(0,False)
            pred = network(x,inits,features,mask=mask,init_root=root,cam_angvel=angvel,return_y_up=True,**kwargs)
            pose,betas = avg_preds(pred['pose'].squeeze(0).reshape(-1,24,6),pred['betas'].squeeze(0),
                flipped['pose'].squeeze(0).reshape(-1,24,6),flipped['betas'].squeeze(0))
            network.pred_pose = pose.reshape_as(network.pred_pose)
            network.pred_shape = betas.reshape_as(network.pred_shape)
            network.pred_contact = (pred['contact']+flipped['contact'][...,[2,3,0,1]])/2
            network.forward_smpl(**kwargs)
            output = network.output
            xyz[frame_ids] = camera_coordinates(output.joints.detach().cpu().numpy().reshape(len(frame_ids),-1,3))
            xy[frame_ids] = output.full_joints2d.detach().cpu().numpy().reshape(len(frame_ids),-1,2)[:,:17]+origin
            scores[frame_ids] = halpe['scores'][frame_ids,:17]
            body_pose[frame_ids] = output.body_pose.cpu().numpy().reshape(-1,23,3,3)
            root_pose[frame_ids] = output.global_orient.cpu().numpy().reshape(-1,1,3,3)
            shape[frame_ids] = output.betas.cpu().numpy().reshape(-1,10)
    values = dict(keypoints=xyz,keypoints_2d=xy,scores=scores,
        missing=~np.isfinite(xyz).all(-1),source_time=halpe['source_time'],frame_index=halpe['frame_index'],
        fps=halpe['fps'],joints=np.array(COCO17),model='WHAM (RTMPose Halpe-26 input)',
        model_sha256=model_hash,source_2d_signature=halpe['input_signature'],input_signature=signature,
        smpl_body_pose=body_pose,smpl_global_orient=root_pose,smpl_betas=shape,
        score_semantics='2D RTMPose confidence; not a calibrated 3D confidence',
        provenance=json.dumps(provenance,sort_keys=True),method=VERSION,
        coordinate_system='hip-relative model meters; x right, y up, z toward camera')
    atomic_npz(target,**values)
    return values


def refine_clip(args,row,halpe,prior):
    paths = paths_for(row)
    times,frames = halpe['source_time'],halpe['frame_index']
    if not np.array_equal(times,prior['source_time']) or not np.array_equal(frames,prior['frame_index']):
        raise ValueError('WHAM and Halpe source timing differ')
    if 'WHAM' not in str(prior['model']) or str(prior['source_2d_signature'])!=str(halpe['input_signature']):
        raise ValueError('Expected an aligned genuine WHAM prior for these Halpe inputs')
    torso = np.linalg.norm((halpe['keypoints'][:,5]+halpe['keypoints'][:,6]-halpe['keypoints'][:,11]-halpe['keypoints'][:,12])/2,axis=-1)
    pool = torso[np.isfinite(torso)&(torso>1)]
    threshold = max(20,.35*np.median(pool)) if len(pool) else 50
    clean,filled,outlier = refine_trajectory(halpe['keypoints'],halpe['scores'],halpe['missing'],times,
        halpe.get('track_id'),frames,args.strength,args.max_gap,threshold)
    atomic_npz(args.dataset/paths['pose_2d_refined'],**dict(halpe,keypoints=clean,
        missing=~np.isfinite(clean).all(-1),interpolated=filled,outlier=outlier))
    result,fill3,spike3,lengths,quality = refine_3d(prior['keypoints'],prior['scores'],clean[:,:17],
        halpe['scores'][:,:17],times,halpe.get('track_id'),frames,args.strength,args.max_gap)
    atomic_npz(args.dataset/paths['pose_3d'],**dict(prior,keypoints=result,
        missing=~np.isfinite(result).all(-1),interpolated=fill3,outlier=spike3,bone_lengths=lengths))
    quality.update(model=str(prior['model']),model_sha256=str(prior['model_sha256']),
        pose_2d_model=str(halpe['model']),pose_2d_layout='halpe26',pose_3d_layout='coco17',
        strength=args.strength,max_gap=args.max_gap,method=VERSION,review_status='unreviewed',
        benchmark_eligible=False,interpretation='Camera-relative WHAM/SMPL estimate; no SLAM, OpenSim IK, calibrated scale or pitching ground truth.')
    error = quality['median_root_relative_reprojection_px']
    quality['median_reprojection_torso_ratio'] = error/float(np.median(pool)) if error is not None and len(pool) else None
    if quality['median_reprojection_torso_ratio'] is None or quality['median_reprojection_torso_ratio']>.35:
        quality['status'] = 'limited'
    atomic_json(args.dataset/paths['motion_quality'],quality)
    return dict(dataset_clip_id=row['dataset_clip_id'],**{k:str(v) for k,v in paths.items()},
        pose_3d_status=quality['status'],method=VERSION,review_status='unreviewed',benchmark_eligible='false',
        input_signature=str(prior['input_signature'])+f':{args.strength}:{args.max_gap}')


def write_index(dataset,record):
    index = dataset/'motion_wham.csv'
    records = {}
    if index.exists():
        with index.open() as stream:
            records = {r['dataset_clip_id']:r for r in csv.DictReader(stream)}
    records[record['dataset_clip_id']] = record
    temporary = index.with_suffix('.tmp.csv')
    with temporary.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(record))
        writer.writeheader();writer.writerows(records[k] for k in sorted(records))
    temporary.replace(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument('--dataset',type=Path,default=workspace/'dataset')
    parser.add_argument('--wham-root',type=Path,default=workspace/'external/WHAM')
    parser.add_argument('--pose-model',type=Path,default=workspace/'models/rtmpose_halpe26_m.onnx')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--strength',type=float,default=.45)
    parser.add_argument('--max-gap',type=int,default=3)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--deploy',action='store_true')
    args = parser.parse_args()
    for key in ['dataset','wham_root','pose_model']:
        setattr(args,key,getattr(args,key).resolve())
    if args.limit<0 or not 0<=args.strength<=10 or not 0<=args.max_gap<=10:
        parser.error('limit must be nonnegative, strength 0..10, max-gap 0..10')
    if not (args.device=='cuda' or args.device.startswith('cuda:')):
        parser.error('this pipeline requires a CUDA device, e.g. cuda:0')
    if args.deploy and not args.publish:
        parser.error('--deploy requires --publish')
    report = check(args)
    atomic_json(args.dataset/'wham_preflight.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if report['errors']:
        return 2
    if args.check:
        return 0
    from rtmlib import RTMPose
    pose_model = RTMPose(str(args.pose_model),model_input_size=(192,256),
        backend='onnxruntime',device=args.device,to_openpose=False)
    if 'CUDAExecutionProvider' not in pose_model.session.get_providers():
        raise RuntimeError('RTMPose silently selected CPU; CUDA is required for this run')
    if any(output.shape[1]!=26 for output in pose_model.session.get_outputs()):
        raise ValueError('Expected a 26-joint Halpe ONNX model')
    pose_hash = digest(args.pose_model)
    cfg,network,extractor,model_hash,provenance = load_wham(args)
    with (args.dataset/'clips.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if args.limit:
        rows = rows[:args.limit]
    for i,row in enumerate(rows,1):
        halpe = infer_halpe(args,row,pose_model,pose_hash)
        prior = infer_wham(args,row,halpe,cfg,network,extractor,model_hash,provenance)
        record = refine_clip(args,row,halpe,prior)
        write_index(args.dataset,record)
        print(f'WHAM {i}/{len(rows)} {row["dataset_clip_id"]} {record["pose_3d_status"]}',flush=True)
        if args.publish and (i%10==0 or i==len(rows)):
            from package_motion import package
            package(args.dataset,workspace/'kbo-pitch-db.github.io/data',backend='wham')
    if args.deploy:
        if not args.publish:
            parser.error('--deploy requires --publish')
        repository = workspace/'kbo-pitch-db.github.io'
        subprocess.run(['git','add','data'],cwd=repository,check=True)
        changes = subprocess.run(['git','diff','--cached','--quiet'],cwd=repository)
        if changes.returncode:
            subprocess.run(['git','commit','-q','-m','Publish WHAM and RTMPose Halpe-26 motion results'],cwd=repository,check=True)
        subprocess.run(['git','push','origin','main'],cwd=repository,check=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
