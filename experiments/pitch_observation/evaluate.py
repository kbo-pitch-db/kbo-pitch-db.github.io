"""Observation evidence, numerical integrity, and sensitivity, never KBO MPJPE."""
from common import *
from candidates import CC,MB,EDGES,norm
from events import infer
from depth_ordinals import evidence
import argparse

FIT_DIRS=['fits_v2','fits_all2d','fits_preserve','fits_rgb','fits_shared','fits_unmerged','fits_final','fits_strong','fits_block','fits_contact_final','fits_nlf']

def current_projection(z,c):
 body=z['keypoints']@z['scene_rotation']+z['scene_origin'];feet=z['foot_keypoints']@z['scene_rotation']+z['scene_origin'];p=np.concatenate([body,feet],1);uv=p[...,:2]/p[...,2:]*float(z['fitted_camera_focal'])+z['camera_principal'];origin=np.array([float(c['crop_x1']),float(c['crop_y1'])]);return np.einsum('tij,tkj->tki',np.linalg.inv(z['background_transforms']),np.concatenate([uv-origin,np.ones((*uv.shape[:2],1))],-1))[...,:2]+origin

def collect(cid):
 r,c,s,h,w=init_inputs(cid);old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);p,leg=norm((old['keypoints']@old['scene_rotation'])[:,CC]);records=[dict(name='current_actual',common=p,uv=current_projection(old,c)[:,CC],leg=leg,global_root=old['root_translation']@old['scene_rotation'],native=old['anatomical_keypoints'],z=old,kind='legacy_actual')]
 for path in (OUT/'candidates').glob(f'*/{cid}.npz'):
  z=load(path);records.append(dict(name='raw/'+path.parent.name,common=z['keypoints'],uv=z['projected'],leg=float(z['model_leg']),z=z,kind='raw_camera_only'))
 for directory in FIT_DIRS:
  for path in (OUT/directory).glob(f'*/{cid}.npz'):
   z=load(path);p,leg=norm(z['keypoints'][:,CC]);records.append(dict(name=directory+'/'+path.parent.name,common=p,uv=z['projected'][:,CC],leg=leg,global_root=z['translation'],native=z['anatomical_keypoints'],z=z,kind='fit',path=str(path.relative_to(OUT))))
 return records,h,c,s

def pct(x,q):
 x=np.asarray(x);x=x[np.isfinite(x)];return float(np.percentile(x,q)) if len(x) else None

def evaluate_record(rec,h,cid,split,ordinal,event):
 p=rec['common'];uv=rec['uv'];target=h['keypoints'][:,CC];conf=scores(h)[:,CC];error=np.linalg.norm(uv-target,axis=-1);valid=(conf>.5)&np.isfinite(error);t=h['source_time'];dt=np.diff(t);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1)&np.isfinite(p[:-1]).all((1,2))&np.isfinite(p[1:]).all((1,2))
 if 'track_id' in h:links&=np.diff(h['track_id'])==0
 out=dict(id=cid,split=split,candidate=rec['name'],kind=rec['kind'],source_frames=len(p),finite_pose_ratio=float(np.isfinite(p).all((1,2)).mean()),global_metric_validated=False,absolute_stride_validated=False)
 for name,ids in [('common12',range(12)),('wrist',[10,11]),('elbow',[8,9]),('ankle',[4,5])]:
  for q in [50,95,99]:out[f'{name}_source_reprojection_p{q}_px']=pct(error[:,ids][valid[:,ids]],q)
 mean=np.nanmean(np.where(valid,error,np.nan),axis=1);out['worst_source_frames']=np.argsort(np.nan_to_num(mean,nan=-1))[-5:].tolist();out['peak_interankle_3d_leg']=pct(np.linalg.norm(p[:,4]-p[:,5],axis=-1),100)
 interval=event.get('plant_interval');out['plant_interval']=interval;out['plant_status']=event.get('plant_status');out['interankle_3d_interval_range_leg']=[pct(np.linalg.norm(p[interval[0]:interval[1]+1,4]-p[interval[0]:interval[1]+1,5],axis=-1),0),pct(np.linalg.norm(p[interval[0]:interval[1]+1,4]-p[interval[0]:interval[1]+1,5],axis=-1),100)] if interval else None
 # Never call this pitch stride: 3D ankle spacing, no inferred pitch-axis substitute.
 starts=[0,2,1,3,6,8,7,9];ends=[2,4,3,5,8,10,9,11];bones=p[:,ends]-p[:,starts];length=np.linalg.norm(bones,axis=-1);vectors=bones/np.maximum(length[...,None],1e-6);dots=(vectors[1:]*vectors[:-1]).sum(-1);out['limb_reversal_count']=int(((dots<0)&links[:,None]).sum());out['lower_limb_reversal_count']=int(((dots[:,:4]<0)&links[:,None]).sum());out['common_regressed_bone_length_cv_max']=float(np.nanmax(np.nanstd(length,0)/np.maximum(np.nanmean(length,0),1e-6)))
 if 'global_root' in rec:
  velocity=np.diff(rec['global_root'],axis=0)/np.maximum(dt[:,None],1e-6)/rec['leg'];out['uncalibrated_root_speed_p95_leg_s']=pct(np.linalg.norm(velocity[links],axis=-1),95);out['uncalibrated_root_speed_max_leg_s']=pct(np.linalg.norm(velocity[links],axis=-1),100)
 if 'native' in rec:
  z=rec['native'];parents=[0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21];lens=np.linalg.norm(z[:,1:]-z[:,parents],axis=-1);out['native_smpl_bone_cv_max']=float(np.nanmax(np.nanstd(lens,0)/np.maximum(np.nanmean(lens,0),1e-6)))
 for side in [0,1]:
  # Arm motion checked against original 2D in an observed delivery window.
  wrist=10+side;shoulder=6+side;obs=target[:,wrist]-target[:,shoulder];pred=uv[:,wrist]-uv[:,shoulder];ov=np.linalg.norm(np.diff(obs,axis=0),axis=-1)/np.maximum(dt,1e-6);pv=np.linalg.norm(np.diff(pred,axis=0),axis=-1)/np.maximum(dt,1e-6);good=links&valid[:-1,wrist]&valid[1:,wrist]&valid[:-1,shoulder]&valid[1:,shoulder];window=event.get('arm_fast_interval')
  if window:
   mask=np.zeros(len(good),bool);mask[max(0,window[0]-3):min(len(good),window[1]+4)]=True;good&=mask
  if good.any():
   a=int(np.argmax(np.where(good,ov,-1)));b=int(np.argmax(np.where(good,pv,-1)));out[f'arm{side}_projection_peak_delay_s']=float(t[b]-t[a]);out[f'arm{side}_projection_speed_ratio']=float(pv[good].max()/max(ov[good].max(),1e-6))
  motion=p[:,wrist]-p[:,shoulder];out[f'arm{side}_native_speed_p95_leg_s']=pct((np.linalg.norm(np.diff(motion,axis=0),axis=-1)/np.maximum(dt,1e-6))[links],95)
 for part in ['fit','withheld']:
  e=[]
  for q in ordinal:
   if q['split_observation']!=part:continue
   i=q['frame'];ref=p[i,[6,7]].mean(0)*(1-q['torso_fraction'])+p[i,:2].mean(0)*q['torso_fraction'];value=q['near_sign']*(p[i,CC.index(q['joint']),2]-ref[2]);e.append(value)
  out[f'{part}_pseudo_ordinal_count']=len(e);out[f'{part}_pseudo_ordinal_violations']=sum(v>.25 for v in e if np.isfinite(v));out[f'{part}_pseudo_ordinal_excess_mean']=float(np.nanmean(np.maximum(np.array(e)-.25,0))) if e else None
 # A second detector is a separate model observation, never independent 2D GT.
 altpath=OUT/'alternative_2d/rtmw384'/f'{cid}.npz'
 if altpath.exists():
  z=load(altpath);err=np.linalg.norm(uv-z['keypoints'][:,CC],axis=-1);v=(z['scores'][:,CC]>.6)&np.isfinite(err)
  out['separate_rtmw_common_p50_px']=pct(err[v],50);out['separate_rtmw_wrist_p95_px']=pct(err[:,[10,11]][v[:,[10,11]]],95)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',default='tune');ap.add_argument('--directories',nargs='+');ap.add_argument('--output-prefix',default='evaluation');ap.add_argument('--full-sample',action='store_true');a=ap.parse_args();allrecords=[]
 global FIT_DIRS
 if a.directories:FIT_DIRS=a.directories
 for sample_rec in sample(a.split,not a.full_sample):
  cid=sample_rec['id'];records,h,c,s=collect(cid);ordinal,probes=evidence(cid);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];event=infer(h,bg)
  for r in records:allrecords.append(evaluate_record(r,h,cid,sample_rec['split'],ordinal,event))
  print('EVALUATED',cid,len(records),flush=True)
 save(OUT/f'{a.output_prefix}_{a.split}.json',allrecords);csvwrite(OUT/f'{a.output_prefix}_{a.split}.csv',allrecords)
 summary=[]
 for candidate in sorted(set(r['candidate'] for r in allrecords)):
  x=[r for r in allrecords if r['candidate']==candidate];v=dict(candidate=candidate,clips=len(x))
  for key in ['common12_source_reprojection_p50_px','wrist_source_reprojection_p95_px','elbow_source_reprojection_p95_px','uncalibrated_root_speed_p95_leg_s','arm0_projection_peak_delay_s','arm1_projection_peak_delay_s','withheld_pseudo_ordinal_violations']:
   values=[r[key] for r in x if r.get(key) is not None];v[key+'_clip_median']=float(np.median(values)) if values else None
  summary.append(v)
 save(OUT/f'{a.output_prefix}_summary_{a.split}.json',summary);csvwrite(OUT/f'{a.output_prefix}_summary_{a.split}.csv',summary)
if __name__=='__main__':main()
