# KBO Pitch Lab

The [Explorer](https://kbo-pitch-db.github.io/#explore) shows final RTMPose Halpe-26 and one published GVHMR + ground/contact IK + MotionBERT reconstruction per clip. The current canonical index contains **813 clips / 123 pitchers / 135,741 source frames**. Every clip uses `gvhmr-ground-motionbert-v1`; there are **zero WHAM fallbacks**. No pitcher is selected automatically. `?clip=CLIP_ID#explore` opens a clip directly.

Release: `gvhmr-ground-motionbert-20261028-v1`. All **813** clips use actual MotionBERT inference and both-arm refinement, fixed native body shape/segment lengths, exact inferred contact anchors and sequence-fixed rigid shoes. This replaces every default 3D payload, including the formerly targeted Choi Won-tae slider. All original 2D JSON/NPZ bytes and source timestamps are preserved. Eight source-inspected pitcher-to-batter tail transitions have separately versioned RTMPose recoveries; their normal prefix frames remain exactly unchanged. The other 805 active final-2D paths are unchanged from the pre-combination release. Earlier GVHMR/contact and arm-study results remain frozen as historical comparisons; they are not the current default.

All clips use the same viewer, one clip-wide camera/scale and one inferred ground frame. No framewise recentering, endpoint hold, population-stride target or display trajectory correction is applied. Playback stops at the last frame. JSON exports the saved coordinates; 3D CSV exports 24 native SMPL joint centers, source timing and availability. COCO-17 surface landmarks and six foot markers remain supplementary data. Rigid saved shoes also retain their shape between rendered frames; interpolation preserves a marker that is stationary in both saved frames.

## What the reconstruction means

The official GVHMR learned checkpoint runs with final RTMPose and HMR2a pitcher-crop inputs (with full-image bbox coordinates), 30 Hz network resampling and identity camera-angular-velocity hypotheses. Its shared SMPL-X rotations are approximately adapted to SMPL, with a separate WHAM-derived **SMPL shape only** for the original inputs. Versioned subject recoveries instead use the corrected pitcher’s HMR2a RGB shape. This is **not** the complete native SMPL-X GVHMR demo/postprocessor. Camera, pose and root are then fitted in stages; the contact step jointly refines lower body, trunk, root and a shared focal estimate, preserving fixed sequence shape and local upper-limb rotations.

The new contact reconstruction uses independent image-supported toe episodes, including clips without a detected plant event. It optimizes a clip-wide ground plane/camera and fixed-shape native joint rotations/root. Bounded two-link leg IK reconstructs actual saved rotations at supporting ankles; it does not stretch leg bones or clamp feet in the renderer. A final rigid-shoe pose fit jointly constrains rotation and ankle-trajectory continuity and uses the original reliable 2D foot observations, native leg reach and one shared plane. Shoe marker offsets are constant in each ankle frame, avoiding the variable toe–heel distance of skinned surface markers. Inferred support can release and recontact; unsupported moving feet remain free.

Actual MotionBERT inference uses the separately acquired pretrained model plus the existing public-mocap last-block adaptation. Original and flip-restored predictions remain separate depth hypotheses. Both arms are reconstructed from the source elbow/wrist rays and fixed limb lengths, with MotionBERT conditioning depth selection and temporal refinement. No claim is made that direct MotionBERT transplantation improves every arm or identifies unique depth. Per-clip checkpoint/prediction/solver hashes and selected refinements are included in the saved JSON.

All 813 outputs must pass checks on saved coordinates: inferred contact-marker diameter ≤1% of the frozen model-leg length, native FK/segment consistency, rigid shoe lengths, bounded knees, all-foot/native-body floor penetration, root/ankle velocity changes, shoe rotation continuity and the closest shoe remaining within 8% of the modeled leg length above the inferred plane. For 66 population clips and one recovered tail, actual root/foot/camera parameters are jointly refitted to prevent whole-body flight outside static toe episodes; this is an explicit pitching-support prior, not a measured contact detector. The absolute modeled root-speed diagnostic remains visible but does not reject continuous fast motion: neither broadcast slow-motion factor nor physical scale is calibrated. Contact residual reduction is objective-linked, **not independent evidence of correct KBO 3D motion**.

Strict source checks are retained separately: overall median, wrist p95 and foot p95 reprojection must each increase by no more than 2 px to pass. **38/813** pass all strict source checks. Foot p95 regresses by over 2 px in **774** clips, overall median in **90**, and wrist p95 in **0**. Rigid contact/shoe constraints trade away image agreement, so physical consistency is not a claim of uniformly improved accuracy. Every failed source check, before/after error and tail remains in the per-clip quality JSON and [combined release audit](data/combined_release_audit.csv). A geometry pass never relabels a failed source check. [The combined summary](data/combined_release_summary.json) records the actual population counts. No clip is labeled human manually reviewed, benchmark eligible or validated for absolute stride. Camera, depth, player scale and actual contact remain monocular estimates.

Six source clips have confirmed camera changes at their ends. Their 14 post-cut frames preserve final 2D/timestamps but retain null 3D coordinates, `available=false` and `unavailable_reason=source_camera_change`. There are **135,727 valid numerical 3D frames** out of 135,741 source frames. Playback and CSV preserve these gaps.

[Current model summary](data/motion_summary.json) and [frozen execution/input/selection provenance](experiments/pitch_observation/releases/combined-20261028/) describe the actual method. A single constant scene transform and inferred ground plane apply to every valid frame. The viewer does not invent a world trajectory, population stride or measured mound. Source time is broadcast playback time and is not converted into physical angular velocities.

## Run locally

Serve only this repository directory: `python3 -m http.server 8000`, then open `http://localhost:8000`. Keep the parent workspace, source broadcast media, model weights and licensed body files outside the served directory.

The site is static GitHub Pages. Only skeleton media and numerical poses are hosted. Timestamped YouTube links and optional click-to-load embeds open source footage; no broadcast pixels/audio are packaged. Research comparisons are unlisted direct links with `noindex,nofollow,noarchive`, absent from the main navigation and disallowed in `robots.txt`; they are not access-controlled.

All 813 clips retain player identity evidence and all 123 pitchers retain 2024 KBO portraits with attribution. Identity evidence and 3D motion quality are separate. Estimates are not validated biomechanics or force data.

## Reproduce, resume and verify

Use the existing separate research worktree and prepared Python/CUDA environment. Original local `dataset/`, separately acquired official model weights, the public-mocap adaptation and licensed neutral SMPL are required; weights/body files/source broadcast media are excluded from Pages. Frozen inputs, intermediate selection and solver source hashes are recorded in the release provenance. `common.py` resolves paths from the research worktree; copying scripts into an unrelated folder changes that root.

```bash
PY=../env/bin/python
EXP=experiments/pitch_observation
$PY $EXP/combined_population.py prepare
$PY $EXP/combined_population.py motionbert
$PY $EXP/combined_support.py
$PY $EXP/combined_arms.py --once
$PY $EXP/combined_contact_ik.py --once
# Reproduce the versioned repairs and selection_frozen.json recorded in provenance.
$PY $EXP/combined_foot_pose_temporal.py
$PY $EXP/audit_combined_temporal.py --once
$PY $EXP/repair_combined_temporal.py --once
$PY $EXP/repair_root_support.py
$PY $EXP/recover_subject_tails.py observe
$PY $EXP/recover_subject_tails.py features
$PY $EXP/recover_subject_tails.py network
$PY $EXP/recover_subject_tails.py base
$PY $EXP/recover_subject_tails.py combine
$PY $EXP/package_subject_tails.py
$PY $EXP/adopt_subject_tails.py
$PY $EXP/combined_subject_update.py
$PY $EXP/adopt_combined_subject.py
$PY $EXP/select_combined_release.py
$PY $EXP/package_combined_population.py
# After saved-coordinate and staged payload checks:
$PY $EXP/package_combined_population.py --promote
$PY -m unittest discover -s tests
node tests/motion_viewer.test.cjs
node tests/reconstruction_registry.test.cjs
node tests/combined_population_browser_qa.cjs http://127.0.0.1:8000/
# Read-only all-813 live index/pose/quality/final-2D content verification:
$PY $EXP/verify_live_combined.py
```

Completed per-clip NPZ/JSON/hashes remain resumable in private `combined_population_v1/`. Source and old published estimates are immutable. Generation/auditing/staging do not push or deploy. Failed candidates remain in private versioned folders; no fallback is silently substituted for a required combined output. Native FK and serialized contact/rigid-shoe checks are integrity checks, not held-out 3D accuracy validation.

## Preserved earlier results

The original WHAM index is losslessly archived in `data/releases/wham-20261028/original_index.json.gz`. `raw_pose_archive.json` and `scene_pose_archive.json` map old raw/scene paths to lossless gzip files and record decoded SHA-256 and original Git blob IDs. Historical scene JSON is losslessly compressed to keep the static site within its storage budget. The pre-combination canonical index is preserved byte-for-byte in `data/releases/support-20261028/original_index.json.gz`, and earlier GVHMR poses/Git history remain intact. Original final 2D files were not overwritten; the eight corrected tails use new versioned paths and preserve every earlier good frame. The [archived README](docs/archive/README_wham_20261028.md) documents the earlier WHAM workflow and its historical 44/769 screening counts; those labels are not transferred to this new method.

## Wrong-person input correction

Oh Seung-hwan Delivery 3 (`4e09dd724a8c_c0011`) originally tracked the batter from the 2D bbox onward. Its active 2D and GVHMR reconstruction now use a separately versioned pitcher track on the same 75 source frames; all original final 2D and 3D files are preserved. The original batter's WHAM pose and shape do not initialize the replacement. Its corrected input is retained in the combined release. Eight additional tail-only input recoveries are listed in `data/combined_release_summary.json`; their old 2D/3D assets remain available. The interval starts after stride: leg lift/landing are outside the observation. The historical subject-correction fit had unknown contact; the new combined release uses separately inferred image-supported toe episodes without inventing the missing stride interval. This is a subject-selection correction, not validation of absolute stride or 3D accuracy. See the [recovery commands and evidence](docs/subject_corrections/oh-delivery3.md) and [versioned asset hashes](data/subject_corrections/4e09dd724a8c_c0011-v1.json).

## Separate arm-cocking experiment

After verifying the full main release, three versioned arm experiments were executed on six tuning and six player/source-disjoint confirmation clips. That historical experiment did not change the then-current release. The unlisted `studies/arm-cocking-20261028/` contains synchronized fixed-axis comparisons, all 16 V3 candidates, the three-iteration report, CSV/JSON metrics, frozen selection, model provenance and `reproduce.txt`. It is absent from the homepage navigation and is not access-controlled.

Direct MotionBERT arm transplantation and additional fitting can worsen source correspondence. The final preview instead intersects source elbow/wrist rays with fixed-length limbs, retaining alternative depth branches; MotionBERT supplies a relative-depth prior. Confirmation wrist p95 median is 5.54 → 1.62 px (frames excluded from this arm refinement: 4.96 → 2.82 px). A GV-depth control also obtains similar pixel gains, and an independent RTMW observation provides little corroboration. These results do not establish correct arm depth, wrist flexion, biomechanical accuracy or a unique solution. Yang Hyeon-jong's source-fit failure and Choi Won-tae's reduced posterior-hand proxy are retained as counterexamples. Native wrist rotations, body/root/camera and feet remain fixed by experimental scope; their preservation is not new contact evidence. The frozen twelve-clip study is preserved as-is. The current all-813 combined release runs its own actual per-clip inference, arm/contact refinement and audits; it does not relabel the old study as a population validation.
