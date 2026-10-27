"""Document the actual promoted population, preserving the earlier README."""
from common import *
from all813_release import ROOT
import shutil,gzip

def main():
 assert (ROOT/'promotion.json').exists(),'Promote only after all population checks; document actual values afterwards'
 summary=json.loads((CODE/'data/gvhmr_release_summary.json').read_text());audit=json.loads((ROOT/'population_audit.json').read_text());assert summary['clips']==audit['summary']['clips']==813
 folder=CODE/'experiments/pitch_observation/releases/gvhmr-contact-20261028';folder.mkdir(parents=True,exist_ok=True)
 for name in ['input_manifest.json','population_audit.csv','selection_summary.csv','release_records.csv','cuda_graph_validation.json','cpu_execution_validation.json','promotion.json']:
  path=ROOT/name
  if path.exists():shutil.copy2(path,folder/name)
 for name in ['sample48.json','subset.json']:shutil.copy2(OUT/name,folder/name)
 for p in (ROOT/'shot_validity').glob('*.json'):shutil.copy2(p,folder/p.name)
 save(folder/'numerical_audit_summary.json',audit['summary']);save(folder/'release_summary.json',summary)
 configs={p.name:json.loads(p.read_text()) for p in (ROOT/'contact').glob('protocol*.json')};save(folder/'execution_protocols.json',configs)
 provenance=dict(method='Official GVHMR learned network + custom approximate SMPL adapter and confidence/contact sequence fitting',gvhmr_weight_sha256='4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91',hmr2a_weight_sha256='2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e',input_modifications=['Final RTMPose Halpe26 first17 instead of ViTPose','HMR2a original RGB features','30Hz network resampling; output original broadcast timestamps','Identity camera-angular-velocity hypotheses','Approximate SMPL-X22 shared rotations to SMPL24; separate WHAM-derived SMPL shape only, never SMPL-X beta transplanted'],absolute_scale='Not calibrated; no measured height or camera',base_fit=dict(focal_input_hypotheses=[1,4,12],chosen_input_family=4,steps_camera_pose_temporal=[60,160,80],depth_regularization=2,shape='Sequence-fixed',full_original_final2d=True),contact_fit=dict(steps=1100,hypotheses=8,prior='Native F4 ankle vector and distance, not population stride',rear_support='One toe during image-supported initial episode, heel/pivot/release allowed',unknown_contact='Zero weight; no invented full pitch/contact',selection='Frozen source/joint/contact screens; failed/unknown screens retained for available previews'),fixed48_sha256=sha(OUT/'sample48.json'),original_public_index_sha256=summary['original_index_sha256'],scripts_sha256={p.name:sha(p) for p in (CODE/'experiments/pitch_observation').glob('*.py') if p.name.startswith(('all813_','audit_all813','package_all813','supervise_all813','rear_pivot_','common','gvhmr_network'))},human_review_complete=False,gt3d_available=False)
 save(folder/'provenance.json',provenance)
 oldreadme=CODE/'docs/archive/README_wham_20261028.md'
 if not oldreadme.exists():oldreadme.parent.mkdir(parents=True,exist_ok=True);oldreadme.write_text('Archived before the GVHMR population release. The current workflow is in the root README.\n\n'+(CODE/'README.md').read_text())
 text=f'''# KBO Pitch Lab

The [Explorer](https://kbo-pitch-db.github.io/#explore) shows final RTMPose Halpe-26 and one published GVHMR-based 3D reconstruction per clip. The current canonical index contains **{summary['clips']} clips / {summary['players']} pitchers / {summary['frames']:,} source frames**. Every clip uses `gvhmr-phase-aware-contact`; there are **zero WHAM fallbacks**. No pitcher is selected automatically. `?clip=CLIP_ID#explore` opens a clip directly.

Release: `{summary['release_id']}`. All final 2D public JSON and source NPZ bytes are unchanged. The six previously accepted rear-pivot previews are reused exactly. The remaining clips use the same sequence fitting objective and contact rules. These are actual replacement 3D coordinates, not an availability-only reclassification.

All clips use the same viewer, one clip-wide camera/scale and one inferred ground frame. No framewise recentering, endpoint hold, population-stride target or display trajectory correction is applied. Playback stops at the last frame. JSON exports the saved coordinates; 3D CSV exports 24 native SMPL joint centers, source timing and availability. COCO-17 surface landmarks and six foot markers remain supplementary data.

## What the reconstruction means

The official GVHMR learned checkpoint runs with final RTMPose/HMR2a inputs, 30 Hz network resampling and identity camera-angular-velocity hypotheses. Its shared SMPL-X rotations are approximately adapted to SMPL, with a separate WHAM-derived **SMPL shape only**. This is **not** the complete native SMPL-X GVHMR demo/postprocessor. Camera, pose and root are then fitted in stages; the contact step jointly refines lower body, trunk, root and a shared focal estimate, preserving fixed sequence shape and local upper-limb rotations.

Contact fitting distinguishes toe/heel episodes and plane-normal/tangential terms, allows heel rise and release, and retains all eight plane/contact-strength candidates. Contact hypotheses are available in {summary['contact_hypotheses_available']} clips; initial rear-toe support is evaluable in {summary['rear_contact_evaluable']}. The other {summary['contact_hypotheses_unknown']} clips have unknown contact and zero contact weight. They can retain slipping, floating or global-depth errors. A rendered floor is an inferred reference, not a measured mound.

{summary['consistency_screen_passed']} clips satisfy the frozen objective-linked consistency screens. **None is labeled human manually reviewed or validated for absolute stride.** All 813 remain estimates with explicit failed/unknown checks. Numerical integrity and lower optimized contact error do not prove actual 3D accuracy. Source wrist/foot errors can regress; per-clip reasons and error tails are retained.

Six source clips have confirmed camera changes at their ends. Their primary shot was independently re-inferred and re-fitted. The {summary['unavailable_3d_frames']} post-cut frames keep their original 2D/timestamps, but have null 3D coordinates and `available=false`, `unavailable_reason=source_camera_change`. There are {summary['valid_3d_frames']:,} valid numerical 3D frames. The player does not interpolate across that boundary; CSV retains blank coordinate rows with the reason.

[Release summary](data/gvhmr_release_summary.json), [current model summary](data/motion_summary.json), and [execution/selection/audit provenance](experiments/pitch_observation/releases/gvhmr-contact-20261028/) describe coverage and limitations. Source time is broadcast playback time; an unknown slow-motion factor is not converted to physical time. Model meters, depth, camera, ground, contact, absolute stride and biomechanical accuracy remain unvalidated.

## Run locally

Serve only this repository directory: `python3 -m http.server 8000`, then open `http://localhost:8000`. Keep the parent workspace, source broadcast media, model weights and licensed body files outside the served directory.

The site is static GitHub Pages. Only skeleton media and numerical poses are hosted. Timestamped YouTube links and optional click-to-load embeds open source footage; no broadcast pixels/audio are packaged. Research comparisons are unlisted direct links with `noindex,nofollow,noarchive`, absent from the main navigation and disallowed in `robots.txt`; they are not access-controlled.

All 813 clips retain player identity evidence and all {summary['players']} pitchers retain 2024 KBO portraits with attribution. Identity evidence and 3D motion quality are separate. Estimates are not validated biomechanics or force data.

## Reproduce, resume and verify

Use the existing separate research worktree and prepared environment. Original local `dataset/`, separately acquired official weights and licensed neutral SMPL are required; they are not included in Pages. Copy the six confirmed-shot policy JSON files from the release provenance folder to `../all813_gvhmr_v1/shot_validity/` before reproducing the population on the same input NPZ bytes.

```bash
PY=../env/bin/python
EXP=experiments/pitch_observation
$PY $EXP/all813_release.py prepare
# One durable supervisor resumes missing inference, background, base and contact stages.
# Inspect ../all813_gvhmr_v1/pipeline_progress.json before starting another supervisor.
$PY $EXP/supervise_all813.py
# On the completed population:
$PY $EXP/all813_select.py
$PY $EXP/audit_all813_gvhmr.py
$PY $EXP/package_all813_gvhmr.py
# Only after auditing staging: switch the local canonical pointer.
$PY $EXP/package_all813_gvhmr.py --promote
$PY -m unittest discover -s tests
node tests/motion_viewer.test.cjs
node tests/reconstruction_registry.test.cjs
node tests/bone_interpolation.test.cjs
node tests/player_identity.test.cjs
# Live verification after a reviewed push/Pages deployment:
$PY $EXP/verify_live_all813.py
```

Completed per-clip NPZ/JSON and source hashes are cached. The supervisor never promotes, pushes or deploys. The optional `worker_device_overrides.json`, `worker_options.json` and `helper_workers.json` in the private result root allow resumable resource scheduling. Clip locks prevent duplicate writers. CUDA graph and single-thread CPU execution use the same 1100-step objective; execution-backend comparisons are recorded, not called accuracy tests. All candidates/checkpoints and local source-overlay diagnostics stay in `../all813_gvhmr_v1/`.

## Preserved earlier results

The original WHAM index is losslessly archived in `data/releases/wham-20261028/original_index.json.gz`. `raw_pose_archive.json` maps old raw paths to lossless gzip files and records decoded SHA-256 and original Git blob IDs. Old final scene JSON and Git history remain intact. Final 2D files were not repackaged. The [archived README](docs/archive/README_wham_20261028.md) documents the earlier WHAM workflow and its historical 44/769 screening counts; those labels are not transferred to this new method.
'''
 (CODE/'README.md').write_text(text);print('Documented actual full release',summary['clips'])

if __name__=='__main__':main()
