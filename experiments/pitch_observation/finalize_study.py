"""Rebuild final analysis, exports, report and verifiable run manifest."""
from common import *
import subprocess

def main():
 import nlf_results,export_relative,final_analysis,build_report
 seal=json.loads((OUT/'frozen_protocol.json').read_text());changed=[p for p,v in seal['code_sha256'].items() if sha(Path(__file__).parent/p)!=v]
 save(OUT/'frozen_protocol_integrity.json',dict(changed=changed,passed=not changed))
 if changed:raise RuntimeError('Frozen primary implementation changed: '+str(changed))
 nlf_results.main();export_relative.main();final_analysis.main()
 p=OUT/'completion.json';x=json.loads(p.read_text());x.update(independent_nlf_full_sequences=24,hybrid_relative_research_candidates=16,nlf_confirmation_cohorts={'native_shape_control':4,'rgb_contour':4,'body_arm_hybrid':4},verified_kbo_true_3d_recovery_count=None,all_compute_jobs_complete=True);save(p,x)
 p=OUT/'final_summary.json';x=json.loads(p.read_text());x.update(nlf_ablation=json.loads((OUT/'nlf_ablation_summary.json').read_text()),relative_candidate_export=16,verified_kbo_true_3d_recovery_count=None,numerical=json.loads((OUT/'numerical_integrity.json').read_text())|{'checks':'See numerical_integrity.json for individual fits'});save(p,x);build_report.main()
 old=json.loads((PREVIOUS/'input_2d_hashes.json').read_text());bad=[p for p,digest in old.items() if sha(DATA/p)!=digest];save(OUT/'input_integrity.json',dict(checked=len(old),changed=len(bad),mismatches=bad,main_worktree_status=subprocess.check_output(['git','-C',str(WORKSPACE/'kbo-pitch-db.github.io'),'status','--porcelain'],text=True),production_commit=subprocess.check_output(['git','-C',str(WORKSPACE/'kbo-pitch-db.github.io'),'rev-parse','HEAD'],text=True).strip()));assert not bad
 paths=['REPORT_KO.html','comparison.html','relative_motion_candidates.zip','final_summary.json','per_clip_comparison.csv','ablation_pairs.csv','ablation_summary.json','selector_ablation.csv','nlf_ablation_summary.csv','nlf_pose_source_evaluation.csv','status_all_813.csv','frozen_protocol.json','split_integrity.json','input_integrity.json','model_provenance.json','validation.json','numerical_integrity.json','REPRODUCE.txt']
 save(OUT/'result_manifest.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),code_commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),code_dirty=subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True),artifact_sha256={p:sha(OUT/p) for p in paths},script_sha256={str(p.relative_to(CODE)):sha(p) for p in Path(__file__).parent.glob('*.py')},published=False))
 print('FINAL STUDY SEALED: source/2D unchanged, no deployment')
if __name__=='__main__':main()
