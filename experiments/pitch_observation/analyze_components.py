"""Paired ablations and native/source checks; no monocular 3D accuracy claims."""
from common import *
from candidates import norm,CC
from events import infer,contact
from itertools import combinations
METRICS=['common12_source_reprojection_p50_px','wrist_source_reprojection_p95_px','elbow_source_reprojection_p95_px','ankle_source_reprojection_p95_px','uncalibrated_root_speed_p95_leg_s','limb_reversal_count','peak_interankle_3d_leg','arm0_projection_speed_ratio','arm1_projection_speed_ratio','arm0_projection_peak_delay_s','arm1_projection_peak_delay_s']

def main():
 evaluated=[]
 for filename in ['evaluation_tune.json','evaluation_component_tune.json']:
  p=OUT/filename
  if p.exists():evaluated+=json.loads(p.read_text())
 lookup={(r['id'],r['candidate']):r for r in evaluated};comparisons=[]
 pairs=[('legacy_to_raw_start_wham','current_actual','fits_all2d/WHAM_raw_temporal_f4'),('fitting_depth_preservation_WHAM','fits_preserve/WHAM_raw_temporal_f4','fits_strong/WHAM_raw_temporal_f4'),('initializer_head_to_lastblock','fits_unmerged/mb_obp_single_smpl_temporal_f4','fits_block/mb_block_single_smpl_temporal_f4'),('lastblock_fitting_depth_preservation','fits_block/mb_block_single_smpl_temporal_f4','fits_strong/mb_block_single_smpl_temporal_f4')]
 for init in ['WHAM_raw','mb_block_single_smpl']:
  for stage in ['hard','soft_confidence','soft_phase','temporal','combined']:
   pairs.append((f'contact_{init}_{stage}',f'fits_contact_final/{init}_control_f4',f'fits_contact_final/{init}_{stage}_f4'))
 for label,before,after in pairs:
  for rec in sample():
   aa=lookup.get((rec['id'],before));bb=lookup.get((rec['id'],after))
   if aa is None or bb is None:continue
   r=dict(comparison=label,id=rec['id'],split='tune',before=before,after=after)
   for key in METRICS:
    a,b=aa.get(key),bb.get(key);r[key+'_before']=a;r[key+'_after']=b;r[key+'_delta']=b-a if a is not None and b is not None else None
   comparisons.append(r)
 save(OUT/'ablation_pairs.json',comparisons);csvwrite(OUT/'ablation_pairs.csv',comparisons)
 summary=[]
 for label in dict.fromkeys(r['comparison'] for r in comparisons):
  x=[r for r in comparisons if r['comparison']==label];r=dict(comparison=label,n=len(x))
  for key in METRICS:
   for suffix in ['before','after','delta']:
    v=[q[key+'_'+suffix] for q in x if q.get(key+'_'+suffix) is not None];r[key+'_'+suffix+'_median']=float(np.median(v)) if v else None
  summary.append(r)
 save(OUT/'ablation_summary.json',summary);csvwrite(OUT/'ablation_summary.csv',summary)
 # Report both tangential and normal motion on hypothesis intervals. The plane
 # was fit from contact candidates: these are constraint diagnostics, not GT.
 contacts=[]
 for rec in sample():
  cid=rec['id'];r,c,s,h,w=init_inputs(cid);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];ev=infer(h,bg);cw=contact(h,ev,bg);dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1);legmean=None
  for path in (OUT/'fits_contact_final').glob(f'*/{cid}.npz'):
   if '_f4' not in path.parent.name:continue
   z=load(path);feet=z['foot_keypoints']+z['translation'][:,None];_,leg=norm(z['keypoints'][:,CC]);n=z['normal'];n=n.reshape(3);v=np.diff(feet,axis=0)/dt[:,None,None]/leg;normal=v@n;tangent=np.sqrt(np.maximum((v*v).sum(-1)-normal**2,0));good=(np.minimum(cw[:-1],cw[1:])>.03)&links[:,None];r=dict(id=cid,candidate=path.parent.name,assumed_contact_samples=int(good.sum()),tangent_speed_p50_leg_s=float(np.median(tangent[good])) if good.any() else None,normal_speed_p50_leg_s=float(np.median(abs(normal[good]))) if good.any() else None,plane_is_not_independent_evidence=True)
   if 'foot_signed_height' in z:r['penetration_p95_leg']=float(np.percentile(np.maximum(-z['foot_signed_height'],0),95)/leg)
   contacts.append(r)
 save(OUT/'contact_diagnostics.json',contacts);csvwrite(OUT/'contact_diagnostics.csv',contacts)
 # Camera sensitivity uses entire pose hypotheses, never coordinate means.
 sensitivity=[]
 from pool import FAMILIES
 for rec in sample():
  for family in FAMILIES:
   zs=[load(OUT/'fits_strong'/f'{family}_temporal_f{f}'/f'{rec["id"]}.npz') for f in [1,4,12]];ps=[norm(z['keypoints'][:,CC])[0] for z in zs];dist=[float(np.nanmedian(np.linalg.norm(a-b,axis=-1))) for a,b in combinations(ps,2)];dep=[float(np.nanpercentile(abs(a[...,2]-b[...,2]),95)) for a,b in combinations(ps,2)];ank=[float(np.nanmax(np.linalg.norm(p[:,4]-p[:,5],axis=-1))) for p in ps];sensitivity.append(dict(id=rec['id'],family=family,camera_pose_difference_max_p50_leg=max(dist),camera_depth_difference_max_p95_leg=max(dep),max_interankle_camera_range_leg=[min(ank),max(ank)],absolute_pitch_stride=False))
 save(OUT/'camera_sensitivity_tune.json',sensitivity);csvwrite(OUT/'camera_sensitivity_tune.csv',sensitivity)
 print('COMPONENTS',len(comparisons),len(contacts),len(sensitivity))
if __name__=='__main__':main()
