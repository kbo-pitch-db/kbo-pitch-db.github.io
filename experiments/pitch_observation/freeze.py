"""Seal model, split, optimizer and selector before confirmation evaluation."""
from common import *
from pool import POLICY,FAMILIES
if __name__=='__main__':
 dest=OUT/'frozen_protocol.json'
 if dest.exists():raise SystemExit('Already sealed; create a new study version to change the experiment.')
 scripts=['optimize.py','events.py','pool.py','axis_order.py','camera_observations.py','motionbert_smpl.py','public_temporal_adaptation.py','public_fitting_probe.py']
 save(dest,dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sample48_sha256=sha(OUT/'sample48.json'),subset_sha256=sha(OUT/'subset.json'),code_sha256={p:sha(Path(__file__).parent/p) for p in scripts},weights_sha256={p:sha(OUT/p) for p in ['adaptation/best_adapter.pt','temporal_adaptation/best.pt']},families=FAMILIES,focal_hypotheses=[1,4,12],schedule=[60,160,80],all_final_halpe_observations_used=True,raw_depth_preservation=2.,contact_in_final=False,contact_reason='Contact/ground assumptions have no independent support for global reconstruction; separate equal-budget ablations retained',selection=POLICY,final_test='24 clips, player/source-disjoint from tune; raw inference cached blindly; no current-iteration quality evaluation before this seal',public_confirmation='16 additional athletes, not previous public_test_A; never train or val',NLF='Exploratory independent RGB keyframes, excluded from frozen reference pool; successful checkpoint load required newer PyTorch',source_view_review='Original RGB only: confirm CF-offset if pitcher foreground and plate/catcher region behind; otherwise other/unknown. No inferred metric camera.',no_automatic_deployment=True))
 print('SEALED',dest)
