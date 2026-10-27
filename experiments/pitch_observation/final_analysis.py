"""Frozen-evaluation summary and scoped usability, including regressions."""
from common import *
from collections import Counter
from candidates import norm,CC
from axis_order import setup,metrics

def aggregate(rows,label):
 if not rows:return dict(split=label,n=0)
 def total(key,which):return sum((r[which].get(key) or 0) for r in rows)
 return dict(split=label,n=len(rows),actual_reference_changes=sum(r['actual_estimate_changed'] for r in rows),classification_only_changes=0,withheld_rgb_mask_delta_mean=float(np.mean([r['withheld_mask_delta'] for r in rows])),withheld_rgb_mask_better=sum(r['withheld_mask_delta']>.002 for r in rows),withheld_rgb_mask_worse=sum(r['withheld_mask_delta']<-.002 for r in rows),old_withheld_robust_reversals=total('withheld_robust_reverse_count','old_field_order'),new_withheld_robust_reversals=total('withheld_robust_reverse_count','new_field_order'),withheld_order_frames=total('withheld_frames','old_field_order'),order_fixed_clips=[r['id'] for r in rows if (r['old_field_order'].get('withheld_robust_reverse_count') or 0)>0 and r['new_field_order'].get('withheld_robust_reverse_count')==0],order_regressed_clips=[r['id'] for r in rows if (r['new_field_order'].get('withheld_robust_reverse_count') or 0)>(r['old_field_order'].get('withheld_robust_reverse_count') or 0)],depth_validated_clips=0,global_validated_clips=0,absolute_stride_validated_clips=0,human_reviewed_clips=0)

def public_summary(path,group,keys):
 x=json.loads(path.read_text());out=[]
 for label in dict.fromkeys(r[group] for r in x):
  a=[r for r in x if r[group]==label];q={group:label,'cases':len(a)}
  for key in keys:
   v=[r[key] for r in a if r.get(key) is not None];q[key+'_mean']=float(np.mean(v)) if v else None
  out.append(q)
 return out

def main():
 selections=[];summary=[];excluded=[]
 for split in ['tune','test']:
  path=OUT/f'selection_strong_{split}.json'
  if not path.exists():continue
  rows_=json.loads(path.read_text());summary.append(aggregate(rows_,split+'_all'));usable=[]
  for r in rows_:
   ob=json.loads((OUT/'camera_observations'/r['id']/'observation.json').read_text())
   r['input_transition_excluded_from_confirmatory_aggregate']='superposition' in ob.get('diagnostic_exclusion','')
   if r['input_transition_excluded_from_confirmatory_aggregate']:excluded.append(r['id'])
   else:usable.append(r)
  summary.append(aggregate(usable,split+'_input_clear'));selections+=rows_
 public=public_summary(OUT/'temporal_adaptation/confirmation.json','model',['mean_joint_error_leg','depth_error_leg','arm_error_leg','leg_error_leg','bone_direction_error_degrees','arm_peak_speed_ratios','arm_peak_absolute_delay_s'])
 fit=public_summary(OUT/'public_fitting_v2_confirmation.json','coefficient',['native_model_error_leg','smpl_initial_error_leg','fitted_error_leg','initial_depth_error_leg','fitted_depth_error_leg'])
 result=dict(kbo=summary,public_confirmation=public,public_fitting_confirmation=fit,input_transition_exclusions=excluded,scope='KBO has no measured 3D GT; ordering evidence conditional on source phase/view/axis hypotheses; masks and pseudo-depth are model-derived RGB observations',no_deployment=True)
 save(OUT/'final_summary.json',result)
 scoresall=[]
 for split in ['tune','test']:
  p=OUT/f'candidate_scores_strong_{split}.json'
  if p.exists():scoresall+=json.loads(p.read_text())
 # Observable parts of A/B/C. Unknown full-3D plausibility is never promoted.
 cause=[]
 for r in selections:
  candidates=[q for q in scoresall if q['id']==r['id']];known=[q for q in candidates if q['field_order'].get('withheld_frames',0)>0];good=[q['candidate'] for q in known if q['field_order'].get('withheld_robust_reverse_count',1)==0 and q['field_order'].get('withheld_ambiguous_count',1)==0];picked=next(q for q in candidates if q['candidate']==r['reference']);cause.append(dict(id=r['id'],scope='conditional source-direction order only; full pose/depth correctness unknown',A_no_consistent_observed_order_candidate=not bool(good) if known else None,B_consistent_candidate_missed=bool(good) and r['reference'] not in good if known else None,consistent_candidates=good,selected=r['reference'],C_full_3d_fitting_damage='not independently adjudicated on KBO; public known-camera experiment quantifies this effect'))
 save(OUT/'failure_scope.json',cause);csvwrite(OUT/'failure_scope.csv',cause)
 # Preserve all 813 old reason memberships; only actually processed rows acquire
 # new candidate fields. Offering a hypothesis is not a verified recovery.
 audit=json.loads((PREVIOUS/'audit_all.json').read_text());new={r['id']:r for r in selections};allstatus=[]
 for old in audit:
  row=dict(old);update=new.get(old['id']);row['study_processed']=update is not None;row['classification_only_changes']=0;row['relative_articulation_use']='legacy_diagnostic_unvalidated';row['global_trajectory_use']='not_validated';row['absolute_stride_use']='do_not_measure';row['human_review_status']='not_performed';row['published_in_this_study']=False
  if update:row.update(new_reference=update['reference'],estimate_changed=update['actual_estimate_changed'],source_visual_status=update['source_visual_status'],relative_articulation_use='observation_fitted_research_candidate; depth alternatives retained',numeric_status=update['numeric_status'],new_depth_status=update['depth_status'],confirmatory_exclusion=update['input_transition_excluded_from_confirmatory_aggregate'])
  row['hybrid_candidate_available']=(OUT/'relative_motion_candidates'/f'{old["id"]}.npz').exists();row['native_rgb_candidate_available']=(OUT/'nlf_native_fixed'/f'{old["id"]}.npz').exists();row['verified_kbo_3d_recovery']=None
  if row['hybrid_candidate_available']:row['hybrid_relative_candidate_path']='relative_motion_candidates/'+old['id']+'.npz'
  allstatus.append(row)
 save(OUT/'status_all_813.json',allstatus);csvwrite(OUT/'status_all_813.csv',allstatus);save(OUT/'per_clip_comparison.json',selections);csvwrite(OUT/'per_clip_comparison.csv',selections)
 save(OUT/'completion.json',dict(fixed_sample=48,tune_executed=sum(r['split']=='tune' for r in selections),test_executed=sum(r['split']=='test' for r in selections),remaining_tune_not_run=12,full_813_reconstruction_run=False,full_run_gate='No validated global/depth accuracy, frozen KBO evidence must be reviewed; preserve candidates and run checkpoints, do not overwrite production',legacy_outputs_preserved=True,final_halpe_2d_preserved=True,classification_only_added=0,real_3d_gt_kbo_available=False))
 print(json.dumps(clean(result),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
