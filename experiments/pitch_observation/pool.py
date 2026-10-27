"""Fixed candidate pool and held-out RGB evidence. No coordinate averaging.

Tuning selects a source-corresponding reference, not a depth truth. Alternative
cameras and initializers with comparable evidence remain explicit hypotheses.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from common import *
from mesh_evidence import geometry,raster
from candidates import CC,norm
from evaluate import current_projection,pct
import argparse
VERSION='weak'
FAMILIES=['WHAM_raw','wham_single','wham_flip_restored','gvhmr','mb_obp_single_smpl','mb_obp_flip_smpl','mb_block_single_smpl','mb_block_flip_smpl']

def paths_for_pool(cid,split):
 records=[]
 for tag in FAMILIES:
  for factor in [1,4,12]:
   if split=='tune' and VERSION=='strong':directory='fits_strong'
   elif split=='tune':directory='fits_block' if tag in ['mb_block_single_smpl','mb_block_flip_smpl'] else 'fits_preserve' if tag in ['WHAM_raw','gvhmr'] else 'fits_unmerged'
   else:directory='fits_final'
   path=OUT/directory/f'{tag}_temporal_f{factor}'/f'{cid}.npz'
   if not path.exists():raise FileNotFoundError('Incomplete frozen candidate pool: '+str(path))
   records.append((f'{tag}_f{factor}',path))
 return records

def load_pool(cid,split):
 r,c,s,h,w=init_inputs(cid);old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);O=np.eye(3);O[:2,2]=[float(c['crop_x1']),float(c['crop_y1'])];bg=O[None]@old['background_transforms']@np.linalg.inv(O)[None]
 current=dict(keypoints=old['keypoints']@old['scene_rotation'],smpl_rotations=np.concatenate([old['smpl_global_orient'],old['smpl_body_pose']],1),smpl_betas=old['smpl_betas'],translation=old['camera_translation'],focal=old['fitted_camera_focal'],principal=old['camera_principal'],background_transforms=bg,projected=current_projection(old,c),source_time=h['source_time'],frame_index=h['frame_index'])
 records=[('current_actual',current)]+[(tag,load(path)) for tag,path in paths_for_pool(cid,split)]
 return records,h,c,s

def mesh_pool(split,full):
 import torch,cv2,pyrender
 from scene_fit import SceneFitter
 from scipy.ndimage import binary_erosion
 from scipy.stats import spearmanr
 torch.set_num_threads(1);cv2.setNumThreads(1);body=SceneFitter(WORKSPACE/'external/WHAM','cuda:0');faces=body.body.faces;colors=np.tile([200,170,130,255],(len(body.body.v_template),1)).astype(np.uint8);renderer=pyrender.OffscreenRenderer(256,512);allstats=[]
 for rec in sample(split,not full):
  cid=rec['id'];output=OUT/('pool_strong' if VERSION=='strong' else 'pool')/cid;output.mkdir(parents=True,exist_ok=True);dest=output/'mesh_metrics_v2.json'
  if dest.exists():allstats+=json.loads(dest.read_text());print('CACHE POOL',cid,flush=True);continue
  pool,h,c,s=load_pool(cid,split);start=time.monotonic();stats=[]
  for order,path in enumerate(sorted((OUT/'observations'/cid).glob('*.npz'))):
   idx=int(path.stem);obs=load(path)
   if 'mask' not in obs:raise ValueError('Missing mask '+str(path))
   hh,ww=obs['mask'].shape;scale=min(1.,360/max(hh,ww));H,W=int(round(hh*scale)),int(round(ww*scale));renderer.viewport_width=W;renderer.viewport_height=H;mask=cv2.resize(obs['mask'].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0;original=cv2.resize(cv2.imread(str(path.with_suffix('.jpg'))),(W,H));tiles=[]
   for name,z in pool:
    rot=z['smpl_rotations'][idx]
    if not np.isfinite(rot).all():continue
    beta=np.asarray(z['smpl_betas']);beta=beta[idx] if beta.ndim==2 else beta;vertices=geometry(body,rot,beta);tr=z['translation'][idx];f=float(z['focal']);principal=z['principal'];bg=z['background_transforms'][idx];iv=np.linalg.inv(bg);A=iv[:2,:2];sc=np.linalg.norm(A[0]);angle=np.arctan2(A[1,0],A[0,0]);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]);vertices=vertices@R.T;tr=tr@R.T;f*=sc;principal=(iv@np.r_[principal,1])[:2]
    color,depth=raster(vertices,faces,colors,tr,f,principal,obs['crop_origin'],scale,renderer);render_mask=depth>0;inter=render_mask&mask;union=render_mask|mask;iou=float(inter.sum()/max(union.sum(),1));q=dict(id=cid,split=split,frame=idx,observation_split='withheld' if order%3==2 else 'selection',candidate=name,silhouette_iou=iou,mask_source='SAM box prompt, not manual GT')
    for label,key in [('full','full_image_inverse_depth'),('crop','person_context_inverse_depth')]:
     zdepth=cv2.resize(obs[key].astype(np.float32),(W,H));good=binary_erosion(inter,iterations=2);v=1/np.maximum(depth,1e-6);q[label+'_surface_depth_correlation']=float(spearmanr(v[good],zdepth[good]).statistic) if good.sum()>100 else None
    # Arm-region mask agreement is based on source-defined ROIs, not model ROIs.
    for side in [0,1]:
     pts=(h['keypoints'][idx,[5+side,7+side,9+side]]-obs['crop_origin'])*scale
     if np.isfinite(pts).all():
      lo=np.maximum(0,np.floor(pts.min(0)-12*scale)).astype(int);hi=np.minimum([W,H],np.ceil(pts.max(0)+12*scale)).astype(int);sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]];q[f'arm{side}_mask_iou']=float(inter[sl].sum()/max(union[sl].sum(),1))
    stats.append(q)
    if name in ['current_actual','WHAM_raw_f4','gvhmr_f4','mb_block_single_smpl_f4']:
     panel=original.copy();panel[render_mask]=(panel[render_mask]*.4+color[...,:3][render_mask][:,::-1]*.6).astype(np.uint8);cv2.putText(panel,name,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1);tiles.append(panel)
   if tiles:cv2.imwrite(str(output/f'{idx:05d}.jpg'),np.concatenate([original]+tiles,1))
  save(dest,stats);allstats+=stats;print('POOL MESH',cid,len(stats),round(time.monotonic()-start,1),flush=True)
 renderer.delete();save(OUT/f'pool_mesh_{VERSION}_{split}.json',allstats);csvwrite(OUT/f'pool_mesh_{VERSION}_{split}.csv',allstats)

# Fixed before looking at KBO test24. Selection frames never become test evidence.
POLICY=dict(version='source_reference_v2',silhouette_weight=1.,reprojection_p95_weight=.10,new_reference_min_train_mask_gain=.005,max_wrist_p95_increase_px=2.,arm_speed_ratio=[.65,1.5],arm_peak_delay_limit_s=.10,near_best_mask_tolerance=.01,depth_disagreement_leg=.15,field_order_min_selection_frames=2,field_order_reverse_fraction=.5,field_order_max_mask_loss=.01,
 semantics='Select an observation-supported reference using selection keyframes only. SAM/depth are same-RGB model observations, never GT. Camera/depth truth remains unresolved; no global publication decision.')

def select(split,full):
 mesh=json.loads((OUT/f'pool_mesh_{VERSION}_{split}.json').read_text());records=[];candidate_rows=[]
 for rec in sample(split,not full):
  cid=rec['id'];pool,h,c,s=load_pool(cid,split);sc=scores(h);
  available=[set(x['frame'] for x in mesh if x['id']==cid and x['candidate']==name) for name,z in pool];common_frames=set.intersection(*available)
  from events import infer
  ev=infer(h,load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'])
  height=max(100,float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])));rows=[]
  for name,z in pool:
   uv=z['projected'];err=np.linalg.norm(uv[:,:17]-h['keypoints'][:,:17],axis=-1);valid=(sc[:,:17]>.5)&np.isfinite(err);wr=err[:,[9,10]][valid[:,[9,10]]];p,leg=norm(z['keypoints'][:,CC]);mr=[x for x in mesh if x['id']==cid and x['candidate']==name and x['frame'] in common_frames];train=[x['silhouette_iou'] for x in mr if x['observation_split']=='selection'];test=[x['silhouette_iou'] for x in mr if x['observation_split']=='withheld'];q=dict(id=cid,candidate=name,selection_mask_iou=float(np.mean(train)) if train else None,withheld_mask_iou=float(np.mean(test)) if test else None,source_reproj_p95_px=pct(err[valid],95),wrist_p95_px=pct(wr,95),finite_ratio=float(np.isfinite(p).all((1,2)).mean()),new_reference_eligible=True)
   q['arm_guard_status']='unknown';q['arm_guard_pass']=True
   if ev.get('arm_fast_interval'):
    side=0 if ev['arm_fast_side']=='left' else 1;j=9+side;sh=5+side;dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1);good=links&(sc[:-1,j]>.5)&(sc[1:,j]>.5)&(sc[:-1,sh]>.5)&(sc[1:,sh]>.5);window=np.zeros(len(good),bool);aa,bb=ev['arm_fast_interval'];window[max(0,aa-3):min(len(good),bb+4)]=True;good&=window
    if good.sum()>=2:
     obsrel=h['keypoints'][:,j]-h['keypoints'][:,sh];fitrel=uv[:,j]-uv[:,sh];ov=np.linalg.norm(np.diff(obsrel,axis=0),axis=-1)/np.maximum(dt,1e-6);pv=np.linalg.norm(np.diff(fitrel,axis=0),axis=-1)/np.maximum(dt,1e-6);ratio=float(pv[good].max()/max(ov[good].max(),1e-6));pa=int(np.argmax(np.where(good,ov,-1)));pb=int(np.argmax(np.where(good,pv,-1)));delay=float(h['source_time'][pb]-h['source_time'][pa]);q.update(arm_guard_status='observed_projection_only',arm_speed_ratio=ratio,arm_peak_delay_s=delay,arm_guard_pass=POLICY['arm_speed_ratio'][0]<=ratio<=POLICY['arm_speed_ratio'][1] and abs(delay)<=POLICY['arm_peak_delay_limit_s'])
   q['common_rgb_keyframes']=len(common_frames)
   from axis_order import setup as axis_setup,metrics as axis_metrics
   q['field_order']=axis_metrics(z['keypoints'][:,CC],axis_setup(cid,h))
   q['selection_score']=q['selection_mask_iou']-POLICY['reprojection_p95_weight']*(q['source_reproj_p95_px'] or 1000)/height;rows.append(q)
  old=next(x for x in rows if x['candidate']=='current_actual');eligible=[x for x in rows if x['finite_ratio']>=.95 and x['arm_guard_pass'] and x['wrist_p95_px'] is not None and x['wrist_p95_px']<=old['wrist_p95_px']+POLICY['max_wrist_p95_increase_px'] and x['selection_mask_iou'] is not None]
  best=max(eligible,key=lambda x:x['selection_score']) if eligible else old
  if best['candidate']!='current_actual' and best['selection_mask_iou']<old['selection_mask_iou']+POLICY['new_reference_min_train_mask_gain']:best=old
  mask_only=best
  source_reason='selection_RGB_mask'
  old_order=old['field_order'];order_usable=old_order.get('selection_frames',0)>=POLICY['field_order_min_selection_frames']
  if order_usable and (old_order.get('selection_robust_reverse_fraction') or 0)>=POLICY['field_order_reverse_fraction']:
   consistent=[q for q in eligible if q['candidate']!='current_actual' and q['field_order'].get('selection_frames',0)>=POLICY['field_order_min_selection_frames'] and q['field_order'].get('selection_robust_reverse_count',1)==0 and q['field_order'].get('selection_ambiguous_count',1)==0 and q['selection_mask_iou']>=old['selection_mask_iou']-POLICY['field_order_max_mask_loss']]
   if consistent:best=max(consistent,key=lambda q:q['selection_score']);source_reason='source_pitch_direction_order_correction'
  # Conditional source evidence also guards against a newly introduced robust reversal.
  bo=best['field_order']
  if best['candidate']!='current_actual' and bo.get('selection_frames',0)>=2 and (bo.get('selection_robust_reverse_fraction') or 0)>=.5 and (old_order.get('selection_robust_reverse_fraction') or 0)<.5:best=old;source_reason='rejected_new_source_direction_reversal'
  near=[x for x in eligible if x['selection_mask_iou']>=best['selection_mask_iou']-POLICY['near_best_mask_tolerance']];by=dict(pool);reference,_=norm(by[best['candidate']]['keypoints'][:,CC]);spread=[]
  for q in near:
   p,_=norm(by[q['candidate']]['keypoints'][:,CC]);spread.append(dict(candidate=q['candidate'],relative_pose_difference_p50_leg=float(np.nanmedian(np.linalg.norm(p-reference,axis=-1))),depth_difference_p95_leg=float(np.nanpercentile(abs(p[...,2]-reference[...,2]),95))))
  ambiguous=[q for q in spread if q['depth_difference_p95_leg']>POLICY['depth_disagreement_leg']]
  record=dict(id=cid,split=split,reference=best['candidate'],mask_only_reference=mask_only['candidate'],selection_reason=source_reason,old_field_order=old_order,new_field_order=best['field_order'],old_selection_mask_iou=old['selection_mask_iou'],new_selection_mask_iou=best['selection_mask_iou'],old_withheld_mask_iou=old['withheld_mask_iou'],new_withheld_mask_iou=best['withheld_mask_iou'],withheld_mask_delta=(best['withheld_mask_iou']-old['withheld_mask_iou']) if best['withheld_mask_iou'] is not None and old['withheld_mask_iou'] is not None else None,
   actual_estimate_changed=best['candidate']!='current_actual',near_best_hypotheses=spread,depth_status='unresolved_hypothesis_set' if ambiguous else 'unvalidated_monocular',relative_articulation_status='new_observation_fitted_candidate' if best['candidate']!='current_actual' else 'retained_legacy_candidate',global_trajectory_status='uncalibrated_not_validated',contact_status='image_interval_hypotheses_only',absolute_stride_status='unvalidated_do_not_measure',numeric_status='finite_screen_only',source_visual_status='automatic_source_mask_and_2d_evidence; agent visual notes separate',human_review_status='not_performed',classification_only_changes=0,published=False)
  records.append(record);candidate_rows+=rows
 save(OUT/f'selection_{VERSION}_{split}.json',records);csvwrite(OUT/f'selection_{VERSION}_{split}.csv',records);save(OUT/f'candidate_scores_{VERSION}_{split}.json',candidate_rows);csvwrite(OUT/f'candidate_scores_{VERSION}_{split}.csv',candidate_rows);save(OUT/f'selection_protocol_{VERSION}.json',POLICY)


if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['weak','strong'],default='strong');ap.add_argument('--split',default='tune');ap.add_argument('--full-sample',action='store_true');ap.add_argument('--mode',choices=['mesh','select'],default='mesh');a=ap.parse_args();VERSION=a.version
 if a.mode=='mesh':mesh_pool(a.split,a.full_sample)
 else:select(a.split,a.full_sample)
