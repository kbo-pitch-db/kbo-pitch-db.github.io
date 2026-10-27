"""V2 source/dynamics checks; the V1 confirmation was not used to design V2."""
from common import *
from arm_cocking_refine import deployment_guard
from arm_cocking_v2 import FOLDER,SETTINGS
import arm_cocking_evaluate as measure
import argparse

def main():
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['tune','confirmation'],default='tune');a=ap.parse_args();protocol=json.loads((FOLDER/'protocol.json').read_text());measure.FOLDER=FOLDER;all_rows=[]
 if a.cohort=='confirmation' and not (FOLDER/'selection_rule.json').exists():raise RuntimeError('Freeze V2 before confirmation evaluation')
 for cid in protocol[a.cohort]:all_rows+=measure.evaluate(cid,a.cohort)
 save(FOLDER/f'evaluation_{a.cohort}.json',all_rows);csvwrite(FOLDER/f'evaluation_{a.cohort}.csv',[{k:v for k,v in r.items() if k!='model_forearm_posterior_curve_leg'} for r in all_rows])
 if a.cohort=='tune':
  candidates=[]
  for name,tag,spline,depth in SETTINGS:
   rr=[r for r in all_rows if r['variant']==name];scores=[]
   for r in rr:
    b=next(q for q in all_rows if q['id']==r['id'] and q['variant']=='current_GVHMR');keys=['holdout_pitch_wrist_p95_px','holdout_pitch_elbow_p95_px']
    if all(r[k] is not None and b[k] is not None for k in keys):scores.append(r[keys[0]]-b[keys[0]]+.5*(r[keys[1]]-b[keys[1]])+10*(abs(r['arm_projected_peak_delay_s'] or 0)-abs(b['arm_projected_peak_delay_s'] or 0)))
   candidates.append(dict(variant=name,clips=len(rr),source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr),score=float(np.mean(scores)) if scores else 1e6,spline=spline,depth_weight=depth))
  eligible=[r for r in candidates if not r['variant'].startswith('gv_') and r['source_and_preservation_pass']>=4];mb=[r for r in candidates if not r['variant'].startswith('gv_')];chosen=min(eligible or mb,key=lambda x:x['score']);record=dict(rule={**measure.RULE,'version':'arm-local-marker-selection-v2','foot_marker_scope':'Frozen by model definition, not independent contact evidence','correction_spline_seconds':.06},selected_variant=chosen['variant'],observationally_similar_variants=[r['variant'] for r in eligible if r['score']<=chosen['score']+2],unique_3d_solution_identified=False,tune_summary=candidates,adopted_for_main=False,source_nonregression_supported=bool(eligible),confirmation_not_used_for_selection=True,v2_design_used_v1_tune_only=True,selection_sha256=sha(__file__))
  p=FOLDER/'selection_rule.json'
  if p.exists():assert json.loads(p.read_text())==clean(record),'Changed selection requires a new version'
  else:save(p,record)
  print(json.dumps(record,indent=2))
 else:
  choice=json.loads((FOLDER/'selection_rule.json').read_text())['selected_variant'];rr=[r for r in all_rows if r['variant']==choice];result=dict(selected_variant=choice,clips=len(rr),source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr),absolute_3d_accuracy_claim=False,main_release_changed=False,foot_marker_scope='Imposed invariant; not contact accuracy evidence',observational_not_GT=True);save(FOLDER/'confirmation_summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
