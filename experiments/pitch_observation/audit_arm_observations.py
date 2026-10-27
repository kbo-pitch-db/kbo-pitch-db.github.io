"""Post-hoc detector-score sensitivity; never alter frozen candidate selection."""
from common import *
from arm_cocking_refine import deployment_guard
from arm_cocking_v3 import FOLDER
import argparse

def main():
 deployment_guard();protocol=json.loads((FOLDER/'protocol.json').read_text());choice=json.loads((FOLDER/'selection_rule.json').read_text())['selected_variant'];results=[]
 for split in ['tune','confirmation']:
  for cid in protocol[split]:
   report=json.loads((FOLDER/'reports'/f'{cid}.json').read_text());side=0 if report['throwing_side']=='left' else 1;alt=load(OUT/'alternative_2d/rtmw384'/f'{cid}.npz');h=init_inputs(cid)[3];assert np.array_equal(alt['frame_index'],h['frame_index'])
   for joint,j in [('wrist',9+side),('elbow',7+side)]:
    score=alt['scores'][:,j];finite=np.isfinite(score)&np.isfinite(alt['keypoints'][:,j]).all(-1)
    for mode,q in [('frozen_raw_score_gt_0p6',None),('posthoc_score_rank_top_half',.5),('posthoc_score_rank_top_quarter',.75)]:
     threshold=.6 if q is None else float(np.quantile(score[finite],q));valid=finite&(score>threshold if q is None else score>=threshold)
     for variant in ['current_GVHMR',choice]:
      z=load(FOLDER/'fits'/variant/f'{cid}.npz');error=np.linalg.norm(z['projected'][:,j]-alt['keypoints'][:,j],axis=-1)[valid];results.append(dict(id=cid,split=split,joint=joint,mode=mode,variant=variant,threshold_native_decoder_units=threshold,frames=int(valid.sum()),native_score_min=float(score[finite].min()),native_score_max=float(score[finite].max()),score_is_probability=False,source_p95_px=float(np.percentile(error,95)) if len(error) else None,used_for_selection_or_fitting=False,meaning='Same RGB alternate estimator, not sensor GT; post-hoc rank sensitivity is descriptive'))
 save(FOLDER/'detector_score_sensitivity.json',results);csvwrite(FOLDER/'detector_score_sensitivity.csv',results);print('Native RTMW score range',min(r['native_score_min'] for r in results),max(r['native_score_max'] for r in results),'not calibrated probability; original frozen gates unchanged')

if __name__=='__main__':main()
