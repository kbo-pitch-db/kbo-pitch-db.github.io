"""Publish numerical execution provenance; no weights, body meshes or source pixels."""
from common import *
from combined_population import ROOT,SITE,manifest
import copy,shutil

def main():
    selection=json.loads((ROOT/'selection.json').read_text());assert selection['summary']['selected_geometry_pass']==813
    records=json.loads((ROOT/'package_records.json').read_text());m=manifest()
    dest=SITE/'experiments/pitch_observation/releases/combined-20261028';dest.mkdir(parents=True,exist_ok=True)
    for source,name in [('inputs.json','inputs.json'),('selection_frozen.json','pre_shoe_selection.json'),('inputs_before_subject_update.json','inputs_before_subject_update.json'),('inputs_before_tail_recovery.json','inputs_before_tail_recovery.json'),('staged_release.json','summary.json')]:shutil.copy2(ROOT/source,dest/name)
    execution=[];rules=None
    for rec in m['records']:
        c=selection['selected'][rec['id']];fit=ROOT/c['fit_path'];audit=json.loads((ROOT/c['audit_path']).read_text());meta=json.loads(fit.with_suffix('.json').read_text())
        assert audit['geometry_pass'] and sha(fit)==audit['hashes']['combined'];rules=audit['rule']
        execution.append(dict(id=rec['id'],selected_fit_sha256=sha(fit),source_gvhmr_fit_sha256=rec['source_sha256'],
            unchanged_final_2d_npz_sha256=rec['pose_2d_sha256'],unchanged_final_2d_public_sha256=rec['public_2d_sha256'],
            actual_motionbert_prediction_sha256=meta['hashes']['motionbert'],selected_variant=c['folder'],
            source_pass=audit['source_pass'],source_checks=audit['source_checks'],
            ground_solver_sha256=json.loads((fit.parents[1]/'ground'/fit.name).with_suffix('.json').read_text())['config']['solver_sha256'],
            arm_solver_sha256=meta['hashes']['solver'],contact_ik_solver_sha256=meta['contact_ik_hashes']['solver'],
            rigid_shoe_solver_sha256=meta['rigid_sole_refit']['solver_sha256']))
    save(dest/'selection.json',execution);save(dest/'physical_rule.json',rules)
    files=list((SITE/'experiments/pitch_observation').glob('combined*.py'))+list((SITE/'experiments/pitch_observation').glob('audit_combined*.py'))
    files+=list((SITE/'experiments/pitch_observation').glob('recover_subject_tails*.py'))
    files+=[SITE/'experiments/pitch_observation'/n for n in ['common.py','pitch_study_fit.py','independent_support.py','arm_depth_rays.py','arm_cocking_evaluate.py','arm_cocking_native.py','arm_cocking_refine.py','hybrid_initializer.py','support_transfer_root.py','stride_review.py','events.py','public_adaptation.py','select_combined_release.py','adopt_combined_subject.py','package_combined_population.py','package_combined_provenance.py','verify_live_combined.py','repair_root_support.py','repair_combined_temporal.py','package_subject_tails.py','adopt_subject_tails.py','rgb_initializers.py','gvhmr_network.py','optimize.py','all813_fit.py']]
    files=[p if p.exists() else SITE/'scripts'/p.name for p in files]
    save(dest/'solver_sources.json',{str(p.relative_to(SITE)):sha(p) for p in sorted(set(files))})
    counts={k:sum(r['source_checks'][k] is False for r in execution) for k in ['source_p50_px','wrist_p95_px','feet_p95_px']}
    save(dest/'execution.json',dict(release='gvhmr-ground-motionbert-20261028-v1',clips=813,actual_motionbert_inference=True,
        motionbert_pretrained_checkpoint_sha256=m['base_motionbert_checkpoint_sha256'],motionbert_adapted_checkpoint_sha256=m['motionbert_checkpoint_sha256'],
        selection_policy=selection['summary']['selection_policy'],pre_shoe_selection_semantics='Frozen intermediate candidates; overridden for the independently corrected Oh pitcher input',
        source_regression_counts=counts,strict_source_pass=sum(r['source_pass'] for r in execution),
        no_render_time_trajectory_edits=True,constant_sequence_shape=True,constant_native_bones=True,constant_shoe_shape=True,
        versioned_final_2d_tail_recoveries=m.get('recovered_final_2d_clips',[]),original_input_files_overwritten=False,
        fixed_floor_and_coordinates_per_clip=True,native_fk_audited_all_valid_frames=True,
        camera_cut_unavailable_frames=14,source_pixels_or_model_weights_published=False,
        human_manually_reviewed=0,ground_truth_3d_accuracy_validated=False,
        limitations=['Inferred contact/plane checks are objective-linked, not independent accuracy evidence',
            'Source reprojection regressions remain failed/visible; no measured depth/scale/absolute stride',
            'Broadcast timestamps retain unknown slow-motion factor; no physical velocity claim']))
    (dest/'reproduce.txt').write_text('''Run in the separate research worktree with the prepared CUDA environment.
Required: original dataset, original official GVHMR/MotionBERT/HMR2a weights,
existing public-mocap MotionBERT adaptation and separately licensed SMPL.
No model weights, SMPL meshes, source image pixels/audio are published.

Restore inputs.json and the pre-combination canonical index from
 data/releases/support-20261028/original_index.json.gz into the private
 combined_population_v1 directory. Reproduce the upstream GVHMR estimates
 using the historical README and the separately versioned subject correction.
Never substitute the old Oh batter pose/shape for the corrected pitcher.

combined_population.py motionbert
combined_support.py
combined_arms.py --once
combined_contact_ik.py --once
Reproduce the versioned repairs identified by pre_shoe_selection.json.
combined_foot_pose_temporal.py
 audit_combined_temporal.py --once
 repair_combined_temporal.py --once
repair_root_support.py
recover_subject_tails.py observe
recover_subject_tails.py features
recover_subject_tails.py network
recover_subject_tails.py base
recover_subject_tails.py combine
package_subject_tails.py
adopt_subject_tails.py
combined_subject_update.py
adopt_combined_subject.py
select_combined_release.py
package_combined_population.py
Review saved-coordinate checks, source-error flags and staged geometry.
package_combined_population.py --promote
package_combined_provenance.py
Run Python unit tests and combined_population_browser_qa.cjs against staging.
After a push and successful Pages deployment, verify_live_combined.py checks
all813 deployed pose, quality and unchanged final-2D hashes. Browser verification
must also check actual fetched models, synchronization, floor, fixed camera,
cut gaps, JSON/CSV and desktop/mobile behavior. Generation never auto-pushes.

Recorded content hashes identify actual outputs. CUDA/solver executions need
not be bitwise reproducible; re-audit any regenerated coordinates rather than
copying the old pass labels. No predeclared accuracy test or human review is
claimed by this population consistency audit.
''')
    print('PROVENANCE',len(execution),'CLIPS',flush=True)

if __name__=='__main__':main()
