"""Stride-focused numerical review without replacing frozen study results.

Full 3D interankle separation, inferred landing intervals and cross-time ankle
reference displacements remain separate. None is measured rubber-to-foot stride.
"""
from common import *
from candidates import CC,norm
from events import infer,smooth_xy
from pool import load_pool
import pool,argparse
NATIVE=[1,2,4,5,7,8,16,17,18,19,20,21]

def native_leg(p):
 return float(np.nanmedian((np.linalg.norm(p[:,4]-p[:,1],axis=-1)+np.linalg.norm(p[:,7]-p[:,4],axis=-1)+np.linalg.norm(p[:,5]-p[:,2],axis=-1)+np.linalg.norm(p[:,8]-p[:,5],axis=-1))*.5))

def phase_info(rec,h,bg):
 ev=infer(h,bg);metadata=json.loads((OUT/'metadata/player_hands.json').read_text());m=metadata.get(rec['player'].split(':')[-1],{});hand=m.get('throwing_hand','unknown') if m.get('identity_matched') else 'unknown';lead='left' if hand=='right' else 'right' if hand=='left' else ev.get('lead_side');conflict=lead is not None and ev.get('lead_side') is not None and lead!=ev['lead_side'];plant=None if conflict else ev.get('plant_interval');ref=[];xy,sc=smooth_xy(h,bg)
 if lead and ev.get('lift_interval') and not conflict:
  side=0 if lead=='left' else 1;rear=1-side;feet=[20+rear,22+rear,24+rear];v=np.linalg.norm(np.gradient(xy[:,feet].mean(1),h['source_time'],axis=0),axis=-1);a,b=ev['lift_interval'];ids=np.arange(max(0,a-1),min(len(xy),b+1));good=ids[(sc[ids][:,feet].mean(1)>.5)&(v[ids]<.25)];ref=good.tolist() if len(good)>=2 else []
 obs_path=OUT/'camera_observations'/rec['id']/'observation.json';obs=json.loads(obs_path.read_text()) if obs_path.exists() else {};axes=None
 if obs.get('view')=='center_field_offset' and obs.get('axis_hypotheses',0)>=30:axes=load(obs_path.with_name('axes.npz'))['pitch_axes']
 return dict(events=ev,throwing_hand=hand,lead_side=lead,plant_interval=plant,phase_side_conflict=conflict,rear_ankle_reference_frames=ref,reference_semantics='Low image-speed rear ankle around observed lift; not observed rubber coordinate or proven ground contact',axis_hypotheses=len(axes) if axes is not None else 0),axes

def metric(native,global_native,phase,axes):
 leg=native_leg(native);sep=np.linalg.norm(native[:,7]-native[:,8],axis=-1)/leg;out=dict(native_interankle_3d_leg=sep,modeled_leg_length=leg,plant_interval=phase['plant_interval'],absolute_stride_validated=False)
 plant=phase['plant_interval'];out['plant_interankle_3d_leg_range']=[float(np.nanmin(sep[plant[0]:plant[1]+1])),float(np.nanmax(sep[plant[0]:plant[1]+1]))] if plant and np.isfinite(sep[plant[0]:plant[1]+1]).any() else None
 out['clip_peak_interankle_3d_leg']=float(np.nanmax(sep));out['rear_reference_to_lead_ankle_3d_leg']=None;out['pitch_axis_displacement_leg_range']=None
 refs=phase['rear_ankle_reference_frames']
 if global_native is not None and refs and phase['lead_side']:
  lead=7 if phase['lead_side']=='left' else 8;rear=15-lead;reference=np.nanmedian(global_native[refs,rear],axis=0);delta=(global_native[:,lead]-reference)/leg;out['rear_reference_to_lead_ankle_3d_leg']=np.linalg.norm(delta,axis=-1)
  if axes is not None:
   dot=delta@axes.T;out['pitch_axis_displacement_leg_range']=np.stack([dot.min(1),dot.max(1)],1)
 return out

def fitted(path):
 z=load(path);return z,z['anatomical_keypoints'],z['anatomical_keypoints']+z['translation'][:,None]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(OUT/'stride_review'));args=ap.parse_args();folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True);pool.VERSION='strong';catalog=[];flat=[];lineage=[]
 for split in ['tune','test']:
  for rec in sample(split,split=='tune'):
   cid=rec['id'];r,c,s,h,w=init_inputs(cid);payload=json.loads((OUT/'viewer_data'/f'{cid}.json').read_text());base=dict(load_pool(cid,split)[0]);old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);old_native=old['anatomical_keypoints']@old['scene_rotation'];old_global=old_native+old['scene_origin'];bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];phase,axes=phase_info(rec,h,bg);directory='fits_strong' if split=='tune' else 'fits_final';entries=[('current_actual',base['current_actual'],old_native,old_global,'legacy scene; original preserved')]
   for label,tag,stage,f in [('gvhmr_f4','gvhmr','temporal',4),('gvhmr_f1','gvhmr','temporal',1),('gvhmr_f12','gvhmr','temporal',12),('GVHMR f4 camera only','gvhmr','camera',4),('GVHMR f4 pose fit','gvhmr','pose',4),('WHAM raw camera only','WHAM_raw','camera',4),('WHAM raw + fitting','WHAM_raw','temporal',4)]:
    path=OUT/directory/f'{tag}_{stage}_f{f}'/f'{cid}.npz';z,nn,gg=fitted(path);entries.append((label,z,nn,gg,str(path.relative_to(OUT))))
   for label,path in [('NLF native shape + fitting',OUT/('fits_nlf_native_shape' if split=='tune' else 'fits_nlf_native_shape_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'),('NLF fixed native shape raw RGB',OUT/'nlf_native_fixed'/f'{cid}.npz'),('NLF WHAM-shape fitting',OUT/('fits_nlf' if split=='tune' else 'fits_nlf_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'),('NLF body + temporal arms',OUT/('fits_nlf_hybrid' if split=='tune' else 'fits_nlf_hybrid_test')/'nlf_mb_hybrid_temporal_f4'/f'{cid}.npz')]:
    if path.exists():z,nn,gg=fitted(path);entries.append((label,z,nn,gg,str(path.relative_to(OUT))))
   # Raw GVHMR geometry has no recovered global trajectory. Camera-only above
   # exposes the unchanged initial articulation with fitted nuisance camera.
   z=load(OUT/'adapters/gvhmr_f4'/f'{cid}.npz');entries.append(('GVHMR raw f4 SMPL adapter',z,z['anatomical_keypoints'],None,'Approximate shared-rotation SMPL adapter, native SMPL-X mesh unavailable'))
   models={}
   for name,z,nn,gg,path in entries:
    p,leg=norm(z['keypoints'][:,CC]);glob=None
    if gg is not None:
     if name=='current_actual':g=z['keypoints'][:,CC]
     else:g=z['keypoints'][:,CC]+z['translation'][:,None]
     glob=(g-g[0,:2].mean(0))/leg
    mm=metric(nn,gg,phase,axes);models[name]=dict(relative=p,global_fixed=glob,projected=z.get('projected',None)[:,CC] if z.get('projected') is not None else None,stride=mm,native_lower_hip_relative_leg=(nn[:,[1,2,4,5,7,8]]-nn[:,[1,2]].mean(1,keepdims=True))/mm['modeled_leg_length'],joint_definition='Common12 WHAM/COCO surface regressor for display; native SMPL ankles for separation',source=path)
    flat.append(dict(id=cid,split=split,player=rec['player_name'],candidate=name,plant_interval=phase['plant_interval'],plant_interankle_3d_leg_range=mm['plant_interankle_3d_leg_range'],clip_peak_interankle_3d_leg=mm['clip_peak_interankle_3d_leg'],rear_reference_frames=phase['rear_ankle_reference_frames'],phase_side_conflict=phase['phase_side_conflict'],absolute_stride_validated=False))
   # Preserve source observations, never include pixels in the public package.
   crop=np.array([float(c[k]) for k in ['crop_x1','crop_y1','crop_x2','crop_y2']]);payload.update(models=models,stride_phase=phase,crop_box=crop,source_url=s.get('url') or s.get('source_url') or 'https://www.youtube.com/watch?v='+s['provider_video_id'],default_model='gvhmr_f4');payload['extent']=max([3.2]+[float(np.nanmax(abs(np.asarray(m['global_fixed'],dtype=float)))) for m in models.values() if m['global_fixed'] is not None])*1.08
   save(folder/'data'/f'{cid}.json',payload);catalog.append(dict(id=cid,player=rec['player_name'],split=split,models=list(models),plant_interval=phase['plant_interval']));lineage.append(dict(id=cid,default='gvhmr_f4',focal_seed='4 x full-frame image diagonal; optimized, not measured',GVHMR='Official network with RTMPose/HMR2a inputs -> approximate SMPL shared-rotation adapter -> camera60+pose160+part-temporal80, depth prior 2',global_or_absolute_validated=False))
 save(folder/'catalog.json',catalog);save(folder/'metrics.json',flat);csvwrite(folder/'metrics.csv',flat);save(folder/'lineage.json',lineage);save(folder/'decision.json',dict(default='gvhmr_f4',comparison='NLF native shape + fitting when available; other hypotheses remain selectable',reason='User source-video stride preference plus stable observed motion; source appearance and reprojection do not identify true depth',review_feedback=['GVHMR f4 stride looked closest to source','NLF family also looked broadly reasonable'],scope='Post-hoc method choice, not new held-out accuracy validation',previous_frozen_results_changed=False,GT_KBO_3d=False))
 print('STRIDE REVIEW',len(catalog),'clips;',len(flat),'stage/model records; unknown phases preserved')
if __name__=='__main__':main()
