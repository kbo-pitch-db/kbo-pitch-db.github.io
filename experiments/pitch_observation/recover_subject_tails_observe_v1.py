"""Recover source-inspected pitcher-to-batter tracking transitions, versioned.

Good original frames and all timing stay exact. Original inputs remain read-only.
New observations, RGB features, GVHMR and MotionBERT are actually recomputed;
no old batter pose/shape is used as the replacement's initializer.
"""
import common as common
from common import *
import argparse,copy,shutil,gzip,traceback
ORIGINAL_OUT=OUT
PRODUCTION=OUT/'combined_population_v1'
TAIL=PRODUCTION/'tail_recovery_v1'
ORIGINAL_INPUTS=init_inputs
# Begin before the source-inspected transition, while a correct pitcher seed
# still exists. These are image/identity observations, not 3D accuracy labels.
CASES={'0b70417b94ab_c0010':71,'3ef5720dccec_c0010':82,
 '782c3ca51bd5_c0004':279,'1161553ed7d7_c0056':152,
 '3ef5720dccec_c0021':67,'e72b3a782760_c0009':146,
 'b54e0e701d77_c0006':161,'02dad52fd781_c0049':120}
SITE=WORKSPACE/'kbo-pitch-db.github.io'

def inputs(cid):
    r,c,s,h,w=ORIGINAL_INPUTS(cid);r=copy.deepcopy(r);c=copy.deepcopy(c);folder=TAIL/'inputs'/cid
    h=load(folder/'refined.npz');r['pose_2d']=str(folder/'raw.npz');r['pose_2d_refined']=str(folder/'refined.npz')
    c.update(crop_video=str(folder/'source.mkv'),crop_x1='0',crop_y1='0',crop_x2=s['width'],crop_y2=s['height'])
    feature=TAIL/'raw/hmr2a'/f'{cid}.npz'
    beta=np.median(load(feature)['smpl_betas'],axis=0) if feature.exists() else np.zeros(10,np.float32)
    w=dict(smpl_betas=np.repeat(beta[None],len(h['frame_index']),0))
    return r,c,s,h,w

def install():
    common.OUT=TAIL;common.init_inputs=inputs
    (TAIL/'metadata').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ORIGINAL_OUT/'metadata/player_hands.json',TAIL/'metadata/player_hands.json')

def observe(records):
    import cv2,torch
    from ultralytics import YOLO
    from rtmlib import RTMPose
    from refine_motion import refine_trajectory
    torch.set_num_threads(1);cv2.setNumThreads(1)
    detector_path=WORKSPACE/'models/yolov8n.pt';pose_path=WORKSPACE/'models/rtmpose_halpe26_m.onnx'
    detector=YOLO(str(detector_path));pose=RTMPose(str(pose_path),model_input_size=(192,256),backend='onnxruntime',device='cpu',to_openpose=False)
    index=json.loads((SITE/'data/clips.json').read_text())
    for cid in records:
        folder=TAIL/'inputs'/cid;dest=folder/'refined.npz'
        if dest.exists() and (folder/'provenance.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True);r,c,s,old,w=ORIGINAL_INPUTS(cid);raw=load(DATA/r['pose_2d']);start=CASES[cid];N=len(old['frame_index'])
        cap=cv2.VideoCapture(s['source_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(c['source_frame_start']));writer=None
        boxes=old['bbox'].copy();points=raw['keypoints'].copy();confidence=raw['scores'].copy();supported=[];details=[];last=boxes[start-1].astype(float)
        for t in range(N):
            ok,img=cap.read();assert ok
            if writer is None:
                writer=cv2.VideoWriter(str(folder/'source.mkv'),cv2.VideoWriter_fourcc(*'FFV1'),float(old['fps']),(img.shape[1],img.shape[0]));assert writer.isOpened()
            writer.write(img)
            if t<start:continue
            result=detector.predict(img,classes=[0],conf=.15,imgsz=1280,device=0,verbose=False)[0]
            b=result.boxes.xyxy.cpu().numpy();conf=result.boxes.conf.cpu().numpy()
            center=lambda q:(q[:2]+q[2:])/2
            heights=b[:,3]-b[:,1];previous_height=last[3]-last[1]
            distance=np.linalg.norm((b[:,:2]+b[:,2:])/2-center(last),axis=-1)/max(previous_height,100)
            eligible=[j for j in range(len(b)) if heights[j]>max(120,.45*previous_height) and distance[j]<.7]
            def score(j):
                lo=np.maximum(last[:2],b[j,:2]);hi=np.minimum(last[2:],b[j,2:]);inter=np.maximum(hi-lo,0).prod();union=(last[2:]-last[:2]).prod()+(b[j,2:]-b[j,:2]).prod()-inter
                return distance[j]+.25*abs(np.log(heights[j]/previous_height))-.4*inter/max(union,1)-.05*conf[j]
            if not eligible:raise ValueError(f'No identity-continuous pitcher detection: {cid} F{t}')
            j=min(eligible,key=score);last=b[j].astype(float);boxes[t]=last
            p,q=pose(img,bboxes=last[None]);assert p.shape==(1,26,2);points[t]=p[0];confidence[t]=q[0]
            supported.append(t);details.append(dict(frame=t,selected_bbox=last,all_boxes=b,confidence=conf,eligible=eligible))
        cap.release();writer.release()
        missing=~np.isfinite(points).all(-1)|(confidence<.2)
        values={**raw,'bbox':boxes,'keypoints':points,'scores':confidence,'missing':missing,'input_signature':np.array(sha(folder/'source.mkv')+sha(pose_path)),'method':np.array('versioned pitcher-tail tracking recovery, unchanged RTMPose Halpe26')}
        atomic_npz(folder/'raw.npz',**values)
        torso=np.linalg.norm((points[:,5]+points[:,6]-points[:,11]-points[:,12])/2,axis=-1);threshold=max(20,.35*np.nanmedian(torso))
        clean,filled,outlier=refine_trajectory(points,confidence,missing,old['source_time'],old['track_id'],old['frame_index'],.45,3,threshold)
        clean[:start]=old['keypoints'][:start];filled[:start]=old['interpolated'][:start];outlier[:start]=old['outlier'][:start]
        refined={**values,'keypoints':clean,'missing':~np.isfinite(clean).all(-1),'interpolated':filled,'outlier':outlier}
        refined['scores'][:start]=old['scores'][:start];refined['missing'][:start]=old['missing'][:start]
        atomic_npz(dest,**refined)
        assert np.array_equal(clean[:start],old['keypoints'][:start],equal_nan=True)
        old_clip=next(x for x in index['clips'] if x['id']==cid);save(folder/'old_clip.json',old_clip)
        save(folder/'detections.json',details)
        save(folder/'provenance.json',dict(id=cid,version='subject-tail-v1',cause='Source-inspected pitcher track switches to or mixes with background batter',
            recovery_start=start,frames=N,original_good_frames_preserved=True,original_files_overwritten=False,
            corrected_2d_sha256=sha(dest),corrected_raw_2d_sha256=sha(folder/'raw.npz'),original_local_2d_sha256=sha(DATA/r['pose_2d_refined']),
            original_public_2d_sha256=sha(SITE/'data'/old_clip['poseRefined']),source_video_sha256=sha(s['source_path']),source_start_frame=int(c['source_frame_start']),
            pose_weight_sha256=sha(pose_path),detector_weight_sha256=sha(detector_path),code_sha256=sha(__file__),detector_supported_frames=len(supported),
            source_pixels_published=False,human_review_complete=False,ground_truth_accuracy_validated=False))
        print('RECOVERED OBSERVATIONS',cid,start,N,flush=True)

def upstream(stage,ids):
    install()
    import all813_release
    records=[r for r in all813_release.population() if r['id'] in ids]
    if stage in ['features','network']:all813_release.run_upstream(stage,records,0);return
    import all813_fit
    for rec in records:
        cid=rec['id'];dest=TAIL/'all813_gvhmr_v1/background'/f'{cid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            z=load(ORIGINAL_OUT/'all813_gvhmr_v1/background'/f'{cid}.npz');z['crop_transforms']=z['full_transforms'].copy();atomic_npz(dest,**z)
            save(dest.with_suffix('.json'),dict(source='Existing person/graphics-masked full-frame stabilization',metric_camera=False))
    all813_fit.base(records)
    from stride_review import phase_info
    for rec in records:
        cid=rec['id'];src=TAIL/'all813_gvhmr_v1/base/gvhmr_temporal_f4'/f'{cid}.npz';z=load(src);r,c,s,h,w=inputs(cid)
        normal=z['normal'];offset=float(np.median(z['foot_signed_height']-((z['foot_keypoints']+z['translation'][:,None])*normal).sum(-1)))
        phase,_=phase_info(rec,h,z['background_transforms']);z['pilot_support_offset']=np.array(offset)
        path=TAIL/'base_selected'/src.name;atomic_npz(path,**z);save(path.with_suffix('.json'),dict(id=cid,phase=phase,actual_gvhmr_reinference=True,
            source_base_sha256=sha(src),corrected_2d_sha256=sha(TAIL/'inputs'/cid/'refined.npz'),shape_source='Fixed median HMR2a RGB shape of corrected pitcher; no old mixed WHAM shape/pose'))


def combine(ids):
    common.OUT=ORIGINAL_OUT
    import combined_population as population
    original=population.manifest();folder=TAIL/'combined';records=[]
    for cid in ids:
        rec=copy.deepcopy(next(r for r in original['records'] if r['id']==cid));path=TAIL/'base_selected'/f'{cid}.npz';meta=json.loads(path.with_suffix('.json').read_text());two=TAIL/'inputs'/cid/'refined.npz'
        rec.update(source=os.path.relpath(path,ORIGINAL_OUT),source_sha256=sha(path),pose_2d_refined=os.path.relpath(two,DATA),pose_2d_sha256=sha(two),phase=meta['phase'],
            subject_recovery_version='subject-tail-v1',subject_correction=f'subject_corrections/{cid}-tail-v1.json',shape_source=meta['shape_source'])
        records.append(rec)
    original['records']=records;save(folder/'inputs.json',original);population.ROOT=folder
    import torch
    torch.set_num_threads(1);population.infer(records,'recovered-tails')
    import combined_support as ground,combined_arms as arms,combined_contact_ik as ik
    from pitch_study_fit import StudyFitter
    body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False)
    for rec in records:ground.fit(rec,body)
    del body;torch.cuda.empty_cache()
    import combined_foot_pose_temporal as shoe,combined_root_support as roots,audit_combined_supported as audit
    body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    for rec in records:
        cid=rec['id'];arms.fit(rec);ik.process(rec,folder);shoe.fit(rec,dict(fit_path=f'polished_v1/base/combined/{cid}.npz'))
        audit.ROOT=folder/'foot_pose_v2';r=audit.audit(rec,body)
        if not r['geometry_pass']:
            roots.fit(rec,dict(fit_path=f'foot_pose_v2/combined/{cid}.npz'));audit.ROOT=folder/'foot_pose_v4';r=audit.audit(rec,body)
        assert r['geometry_pass'],(cid,r['checks'])
        save(folder/'ready'/f'{cid}.json',dict(id=cid,record=rec,fit_path=str((audit.ROOT/'combined'/f'{cid}.npz').relative_to(PRODUCTION)),audit_path=str((audit.ROOT/'audit'/f'{cid}.json').relative_to(PRODUCTION))))
        print('RECOVERED SUBJECT COMBINED READY',cid,flush=True)

if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('stage',choices=['observe','features','network','base','combine']);p.add_argument('--id');a=p.parse_args();ids=[a.id] if a.id else list(CASES)
    if a.stage=='observe':observe(ids)
    elif a.stage=='combine':combine(ids)
    else:upstream(a.stage,ids)
