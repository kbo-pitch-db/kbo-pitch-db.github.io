"""Separate observed alignment/dynamics from unvalidated 3D cocking proxies."""
from common import *
from all813_release import ROOT
from arm_cocking_refine import FOLDER,LABELS,deployment_guard
from audit_rear_pivot import source_detail
from stride_review import native_leg
import argparse
from arm_cocking_axial import LABELS as AXIAL_LABELS

RULE=dict(version='arm-selection-v1',wrist_p95_increase_px=2.,elbow_p95_increase_px=2.,
 holdout_wrist_p95_increase_px=2.,lower_body_max_change_model=.00002,
 foot_marker_max_change_leg=.003,projected_speed_ratio_range=[.65,1.5],
 separate_rtmw_wrist_p95_increase_px=2.,separate_detector='RTMW384 observations, never used for arm fitting; same RGB, not independent sensor or GT',
 ranking='Among MB settings only; GVHMR-only controls remain separate comparisons. Tune mean wrist+0.5 elbow held-out p95 change plus observed peak-delay/speed disagreement. Never rank by largest 3D hand-behind depth.',
 minimum_source_nonregression_tune_clips=4,isolated_return_step_radians=.9,isolated_return_net_ratio=.4,allow_new_source_unsupported_forearm_returns=0,
 validation='Withheld from arm refinement only. GVHMR and initializers saw all frames; no measured 3D GT.')

def percent(values,q):
 values=np.asarray(values);values=values[np.isfinite(values)]
 return float(np.percentile(values,q)) if len(values) else None

def direction_returns(v,links):
 v=np.asarray(v);size=np.linalg.norm(v,axis=-1);u=v/np.maximum(size[:,None],1e-8)
 step=np.arccos(np.clip((u[1:]*u[:-1]).sum(-1),-1,1));net=np.arccos(np.clip((u[2:]*u[:-2]).sum(-1),-1,1));valid=links[1:]&links[:-1]&(size[2:]>1e-6)&(size[1:-1]>1e-6)&(size[:-2]>1e-6)
 spike=valid&(np.minimum(step[:-1],step[1:])>.9)&(net<.4*np.minimum(step[:-1],step[1:]));return np.flatnonzero(spike)+1,step

def evaluate(cid,split):
 report=json.loads((FOLDER/'reports'/f'{cid}.json').read_text());base=load(FOLDER/'fits/current_GVHMR'/f'{cid}.npz');h=init_inputs(cid)[3];side=0 if report['throwing_side']=='left' else 1;leg=native_leg(base['anatomical_keypoints']);dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1)
 if 'track_id' in h:links&=np.diff(h['track_id'])==0
 source=scores(h);window=report['window'].get('frame_interval');mask=np.zeros(len(dt),bool)
 if window:mask[max(0,window[0]):min(len(dt),window[1]+1)]=True
 shoulder2=5+side;elbow2=7+side;wrist2=9+side;shoulder3=16+side;elbow3=18+side;wrist3=20+side
 target=h['keypoints'][:,:17];observed_returns,_=direction_returns(target[:,wrist2]-target[:,elbow2],links);holdout=np.arange(len(target))%5==2;results=[]
 for name in report['variants']:
  z=load(FOLDER/'fits'/name/f'{cid}.npz');m,uv=source_detail(z,h,base['contact_weight']);r=dict(id=cid,split=split,variant=name,**m)
  error=np.linalg.norm(uv[:,:17]-target,axis=-1)
  for joint,j in [('pitch_wrist',wrist2),('pitch_elbow',elbow2)]:
   valid=(source[:,j]>.5)&np.isfinite(error[:,j])
   for scope,subset in [('all',np.ones(len(target),bool)),('holdout',holdout),('phase',np.zeros(len(target),bool))]:
    if scope=='phase' and window:subset[max(0,window[0]):min(len(target),window[1]+1)]=True
    for q in [50,95,99,100]:r[f'{scope}_{joint}_p{q}_px']=percent(error[:,j][valid&subset],q)
  alternate=OUT/'alternative_2d/rtmw384'/f'{cid}.npz'
  for joint in ['pitch_wrist','pitch_elbow']:r[f'separate_rtmw_{joint}_p95_px']=None
  if alternate.exists():
   alt=load(alternate);assert np.array_equal(alt['frame_index'],h['frame_index'])
   for joint,j in [('pitch_wrist',wrist2),('pitch_elbow',elbow2)]:
    valid=(alt['scores'][:,j]>.6)&np.isfinite(alt['keypoints'][:,j]).all(-1);r[f'separate_rtmw_{joint}_p95_px']=percent(np.linalg.norm(uv[:,j]-alt['keypoints'][:,j],axis=-1)[valid],95)
  observed=target[:,wrist2]-target[:,shoulder2];predicted=uv[:,wrist2]-uv[:,shoulder2];ospeed=np.linalg.norm(np.diff(observed,axis=0)/dt[:,None],axis=-1);pspeed=np.linalg.norm(np.diff(predicted,axis=0)/dt[:,None],axis=-1)
  good=links&mask&(source[:-1,wrist2]>.5)&(source[1:,wrist2]>.5)&(source[:-1,shoulder2]>.5)&(source[1:,shoulder2]>.5);ix=np.flatnonzero(good)
  r.update(arm_projected_speed_ratio=None,arm_projected_peak_delay_s=None)
  if len(ix):
   oi=ix[np.argmax(ospeed[ix])];pi=ix[np.argmax(pspeed[ix])];r.update(arm_projected_speed_ratio=float(np.max(pspeed[ix])/max(np.max(ospeed[ix]),1e-6)),arm_projected_peak_delay_s=float(h['source_time'][pi]-h['source_time'][oi]))
  a=z['anatomical_keypoints'];return_frames,angular_step=direction_returns(a[:,wrist3]-a[:,elbow3],links);unsupported=[int(i) for i in return_frames if i not in observed_returns];r.update(forearm_source_unsupported_return_frames=unsupported,forearm_direction_step_p95_degrees=float(np.degrees(np.percentile(angular_step[links],95))),forearm_direction_step_max_degrees=float(np.degrees(angular_step[links].max())),return_diagnostic_semantics='Adjacent >0.9rad steps with net angle <0.4 of both, absent from source 2D; conservative continuity screen, not GT')
  world=a.astype(np.float64)+z['translation'][:,None].astype(np.float64);baseworld=base['anatomical_keypoints'].astype(np.float64)+base['translation'][:,None].astype(np.float64)
  up=a[:,12]-a[:,0];up/=np.maximum(np.linalg.norm(up,axis=-1,keepdims=True),1e-6);left=a[:,16]-a[:,17];left-=np.sum(left*up,-1,keepdims=True)*up;left/=np.maximum(np.linalg.norm(left,axis=-1,keepdims=True),1e-6);front=np.cross(left,up)
  behind=-np.sum((a[:,wrist3]-a[:,elbow3])*front,axis=-1)/leg;ids=np.arange(window[0],min(len(a),window[1]+1)) if window else np.arange(len(a));peak=ids[np.argmax(behind[ids])]
  r.update(model_forearm_posterior_peak_leg=float(behind[peak]),model_forearm_posterior_peak_frame=int(peak),model_forearm_posterior_curve_leg=behind.tolist(),posterior_proxy_semantics='Model torso cross-product basis; not observed wrist rotation or ground-truth cocking',lower_body_max_change_model=float(abs(world[:,:16]-baseworld[:,:16]).max()),foot_marker_max_change_leg=float(np.linalg.norm(z['foot_keypoints']+z['translation'][:,None]-base['foot_keypoints']-base['translation'][:,None],axis=-1).max()/leg),arm_change_p95_leg=float(np.percentile(np.linalg.norm(world[:,[elbow3,wrist3]]-baseworld[:,[elbow3,wrist3]],axis=-1)/leg,95)),human_review_complete=False)
  r['native_stride_change_max_leg']=float(abs(np.linalg.norm(a[:,7]-a[:,8],axis=-1)-np.linalg.norm(base['anatomical_keypoints'][:,7]-base['anatomical_keypoints'][:,8],axis=-1)).max()/leg)
  results.append(r)
 base_r=next(r for r in results if r['variant']=='current_GVHMR')
 for r in results:
  reasons=[]
  for key,limit in [('all_pitch_wrist_p95_px',2.),('all_pitch_elbow_p95_px',2.),('holdout_pitch_wrist_p95_px',2.)]:
   if r[key] is None or base_r[key] is None:reasons.append(key+'_unknown')
   elif r[key]>base_r[key]+limit:reasons.append(key+'_regression')
  if r['lower_body_max_change_model']>RULE['lower_body_max_change_model']:reasons.append('lower_body_changed')
  if r['foot_marker_max_change_leg']>RULE['foot_marker_max_change_leg']:reasons.append('foot_marker_pose_blend_change')
  alt_key='separate_rtmw_pitch_wrist_p95_px'
  if r[alt_key] is not None and base_r[alt_key] is not None and r[alt_key]>base_r[alt_key]+RULE['separate_rtmw_wrist_p95_increase_px']:reasons.append('separate_detector_wrist_regression')
  if set(r['forearm_source_unsupported_return_frames'])-set(base_r['forearm_source_unsupported_return_frames']):reasons.append('new_source_unsupported_forearm_return')
  ratio=r['arm_projected_speed_ratio']
  if ratio is None:reasons.append('observed_dynamics_unknown')
  elif not .65<=ratio<=1.5:reasons.append('projected_dynamics_disagreement')
  delay=r['arm_projected_peak_delay_s'];old=base_r['arm_projected_peak_delay_s']
  if delay is not None and old is not None and abs(delay)>max(abs(old)+1/float(h['fps']),2/float(h['fps'])):reasons.append('observed_peak_timing_regression')
  r['source_and_preservation_gate']=not reasons;r['gate_reasons']=reasons
 save(FOLDER/'evaluation'/f'{cid}.json',results);return results

def freeze_tune(rows):
 candidates=[]
 for name in ['raw_graft_single','raw_graft_flip','raw_graft_native_single','raw_graft_native_flip','raw_forearm_single','raw_forearm_flip','raw_axial_single','raw_axial_flip']+LABELS+AXIAL_LABELS:
  rr=[r for r in rows if r['variant']==name];delta=[]
  for r in rr:
   b=next(q for q in rows if q['id']==r['id'] and q['variant']=='current_GVHMR')
   if all(r[k] is not None and b[k] is not None for k in ['holdout_pitch_wrist_p95_px','holdout_pitch_elbow_p95_px']):delta.append((r['holdout_pitch_wrist_p95_px']-b['holdout_pitch_wrist_p95_px'])+.5*(r['holdout_pitch_elbow_p95_px']-b['holdout_pitch_elbow_p95_px'])+10*(abs(r['arm_projected_peak_delay_s'] or 0)-abs(b['arm_projected_peak_delay_s'] or 0)))
  candidates.append(dict(variant=name,source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr),score=float(np.mean(delta)) if delta else 1e6,clips=len(rr)))
 mb_candidates=[r for r in candidates if not r['variant'].startswith('gv_')];acceptable=[r for r in mb_candidates if r['source_and_preservation_pass']>=RULE['minimum_source_nonregression_tune_clips']];chosen=min(acceptable or mb_candidates,key=lambda r:r['score']);result=dict(rule=RULE,selected_variant=chosen['variant'],observationally_similar_variants=[r['variant'] for r in acceptable if r['score']<=chosen['score']+2.],unique_3d_solution_identified=False,tune_summary=candidates,adopted_for_main=False,source_nonregression_supported=bool(acceptable),selection_sha256=sha(__file__),confirmation_not_used_for_selection=True)
 path=FOLDER/'selection_rule.json'
 if path.exists():assert json.loads(path.read_text())==clean(result),'Frozen selection changed; new experiment version required'
 else:save(path,result)
 print(json.dumps(result,indent=2))

def main():
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['tune','confirmation'],default='tune');a=ap.parse_args();protocol=json.loads((FOLDER/'protocol.json').read_text());rows_=[]
 if a.cohort=='confirmation' and not (FOLDER/'selection_rule.json').exists():raise RuntimeError('Tune selection must be frozen first')
 for cid in protocol[a.cohort]:rows_+=evaluate(cid,a.cohort)
 save(FOLDER/f'evaluation_{a.cohort}.json',rows_);csvwrite(FOLDER/f'evaluation_{a.cohort}.csv',[{k:v for k,v in r.items() if k!='model_forearm_posterior_curve_leg'} for r in rows_])
 if a.cohort=='tune':freeze_tune(rows_)
 else:
  choice=json.loads((FOLDER/'selection_rule.json').read_text())['selected_variant'];rr=[r for r in rows_ if r['variant']==choice];save(FOLDER/'confirmation_summary.json',dict(selected_variant=choice,clips=len(rr),source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr),absolute_3d_accuracy_claim=False,main_release_changed=False,observational_not_GT=True))

if __name__=='__main__':main()
