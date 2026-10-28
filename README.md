# KBO Pitch Lab

The [Explorer](https://kbo-pitch-db.github.io/#explore) shows final RTMPose Halpe-26 and one published GVHMR-based 3D reconstruction per clip. The current canonical index contains **813 clips / 123 pitchers / 135,741 source frames**. Every clip uses `gvhmr-phase-aware-contact`; there are **zero WHAM fallbacks**. No pitcher is selected automatically. `?clip=CLIP_ID#explore` opens a clip directly.

Release: `gvhmr-contact-20261028-v1`, with the targeted `independent-support-v7` repair described below. All final 2D public JSON and source NPZ bytes are unchanged. The six previously accepted rear-pivot previews are reused exactly. These are actual replacement 3D coordinates, not an availability-only reclassification.

All clips use the same viewer, one clip-wide camera/scale and one inferred ground frame. No framewise recentering, endpoint hold, population-stride target or display trajectory correction is applied. Playback stops at the last frame. JSON exports the saved coordinates; 3D CSV exports 24 native SMPL joint centers, source timing and availability. COCO-17 surface landmarks and six foot markers remain supplementary data.

## What the reconstruction means

The official GVHMR learned checkpoint runs with final RTMPose and HMR2a pitcher-crop inputs (with full-image bbox coordinates), 30 Hz network resampling and identity camera-angular-velocity hypotheses. Its shared SMPL-X rotations are approximately adapted to SMPL, with a separate WHAM-derived **SMPL shape only**. This is **not** the complete native SMPL-X GVHMR demo/postprocessor. Camera, pose and root are then fitted in stages; the contact step jointly refines lower body, trunk, root and a shared focal estimate, preserving fixed sequence shape and local upper-limb rotations.

The population contact fit distinguishes toe/heel episodes and plane-normal/tangential terms, allows heel rise and release, and retains all eight plane/contact-strength candidates. After the targeted repair, contact hypotheses are available in 510 clips; a rear-toe episode is evaluable in 510. The other 303 clips have unknown contact and zero contact weight. They can retain slipping, floating or global-depth errors. A rendered floor is an inferred reference, not a measured mound.

Targeted repair: [Choi Won-tae, 2024-08-16 slider c0026](https://kbo-pitch-db.github.io/?clip=d4ea211e62bd_c0026#explore) had no detected plant event, which disabled every contact constraint, including its observable rear stance. Independent, confidence-gated toe episodes now preserve rear support without a plant event, release on missing/moving observations, and permit later support/recontact. Root, lower body, trunk, foot rotations and a shared focal estimate are refitted with fixed shape/local upper-limb rotations, bounded knees and an all-foot nonpenetration penalty. There are no renderer trajectory corrections. Only this reported clip has this repair; the other 812 coordinates are unchanged.

The [repair audit](data/support_repairs/d4ea211e62bd_c0026-v7.json) records all candidates, checks and error tails. Initial rear-toe maximum displacement changes from 100.9% to 1.18% of the frozen model-leg reference; the later lead-toe interval changes from 61.7% to 2.90%. Foot reprojection p95 changes from 5.35 to 6.68 px; its worst error increases from 12.96 to 20.94 px at frame 33. All 62 frames were checked against CPU SMPL FK and fixed native segment lengths. This clip was used to tune the repair; contact residual reduction is objective-linked, not independent 3D accuracy validation. The original population poses and frozen audits remain available, and the frozen 377-pass count below is not relabeled by the repair.

377 clips satisfy the frozen objective-linked consistency screens. **None is labeled human manually reviewed or validated for absolute stride.** All 813 remain estimates with explicit failed/unknown checks. Numerical integrity and lower optimized contact error do not prove actual 3D accuracy. Source wrist/foot errors can regress; per-clip reasons and error tails are retained.

Six source clips have confirmed camera changes at their ends. Their primary shot was independently re-inferred and re-fitted. The 14 post-cut frames keep their original 2D/timestamps, but have null 3D coordinates and `available=false`, `unavailable_reason=source_camera_change`. There are 135,727 valid numerical 3D frames. The player does not interpolate across that boundary; CSV retains blank coordinate rows with the reason.

[Release summary](data/gvhmr_release_summary.json), [current model summary](data/motion_summary.json), and [execution/selection/audit provenance](experiments/pitch_observation/releases/gvhmr-contact-20261028/) describe coverage and limitations. Source time is broadcast playback time; an unknown slow-motion factor is not converted to physical time. Model meters, depth, camera, ground, contact, absolute stride and biomechanical accuracy remain unvalidated.

## Run locally

Serve only this repository directory: `python3 -m http.server 8000`, then open `http://localhost:8000`. Keep the parent workspace, source broadcast media, model weights and licensed body files outside the served directory.

The site is static GitHub Pages. Only skeleton media and numerical poses are hosted. Timestamped YouTube links and optional click-to-load embeds open source footage; no broadcast pixels/audio are packaged. Research comparisons are unlisted direct links with `noindex,nofollow,noarchive`, absent from the main navigation and disallowed in `robots.txt`; they are not access-controlled.

All 813 clips retain player identity evidence and all 123 pitchers retain 2024 KBO portraits with attribution. Identity evidence and 3D motion quality are separate. Estimates are not validated biomechanics or force data.

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

## Separate arm-cocking experiment

After verifying the full main release, three versioned arm experiments were executed on six tuning and six player/source-disjoint confirmation clips. The main release is unchanged. The unlisted `studies/arm-cocking-20261028/` contains synchronized fixed-axis comparisons, all 16 V3 candidates, the three-iteration report, CSV/JSON metrics, frozen selection, model provenance and `reproduce.txt`. It is absent from the homepage navigation and is not access-controlled.

Direct MotionBERT arm transplantation and additional fitting can worsen source correspondence. The final preview instead intersects source elbow/wrist rays with fixed-length limbs, retaining alternative depth branches; MotionBERT supplies a relative-depth prior. Confirmation wrist p95 median is 5.54 → 1.62 px (frames excluded from this arm refinement: 4.96 → 2.82 px). A GV-depth control also obtains similar pixel gains, and an independent RTMW observation provides little corroboration. These results do not establish correct arm depth, wrist flexion, biomechanical accuracy or a unique solution. Yang Hyeon-jong's source-fit failure and Choi Won-tae's reduced posterior-hand proxy are retained as counterexamples. Native wrist rotations, body/root/camera and feet remain fixed by experimental scope; their preservation is not new contact evidence. No arm candidate is adopted for the main dataset.
