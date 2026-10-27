# KBO Pitch Lab

## Run WHAM + RTMPose Halpe-26 on the server

Run these commands in a server environment with NVIDIA devices exposed and outbound downloads enabled. The runner requires actual CUDA execution and WHAM checkpoints; it does not substitute another 3D model.

```bash
cd /home/eis/Workspace/Kichang/kbo-pitchers
# Install a separate .wham-env, download official models, then process one clip.
bash kbo-pitch-db.github.io/scripts/run_wham.sh --limit 1
# Resume and process the complete dataset, then commit numeric assets and push main.
bash kbo-pitch-db.github.io/scripts/run_wham.sh --deploy
```

The script clones the official [WHAM repository](https://github.com/yohanshin/WHAM) into `external/WHAM`, downloads its BEDLAM/3DPW checkpoint and HMR2A image-feature checkpoint, and obtains the official RTMPose-m Halpe-26 ONNX model. It reuses the existing licensed neutral SMPL file at `~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl`. Pass `--smpl-neutral /path/to/SMPL_NEUTRAL.pkl` if the existing model is elsewhere. Models, environments and meshes stay outside this public repository.

The dedicated Python 3.9 environment pins RTMLib 0.0.15, which supports the Halpe-26 RTMPose API. RTMLib 0.0.16 requires Python 3.10 or newer and is not used in this environment. The first run installs dependencies; subsequent runs resume cached results. Logs are in `dataset/wham_run.log`, preflight diagnostics in `dataset/wham_preflight.json`, and completed clip records in `dataset/motion_wham.csv`. `--prepare-only` installs and checks the environment without processing clips. `--device cuda:1` selects the second GPU. The setup command uses the selected GPU. For two GPUs after setup, run:

```bash
.wham-env/bin/python kbo-pitch-db.github.io/scripts/wham_parallel.py --gpus 0,1
```

Workers use disjoint clips and private indexes; one parent owns the merged index and publishes complete batches. Completed caches resume across restarts. HMR2A image features can be batched in float32 with `--feature-batch-size 16`. `wham_runtime.py` puts the environment's `torch/lib` ahead of system CUDA libraries before Python starts. Preflight checks cuDNN and executes a GPU LSTM as well as allocating a tensor.

The 2D model really predicts all 26 landmarks, including head, neck, hip, toes and heels. Existing pitcher boxes select the same person. Its first 17 joints feed the official WHAM feature encoder and recurrent/SMPL network, with flip evaluation. WHAM state resets at identity changes and frame gaps. The WHAM adapter supplies camera-relative priors; the scene fitter then infers a fixed local support frame. Zero camera angular velocity does not establish a calibrated world trajectory. The learned WHAM 3D joints and SMPL pose/shape parameters are cached separately from the older baseline. The displayed 3D confidence is inherited 2D RTMPose confidence, not a calibrated 3D score.

Only successfully inferred, aligned clips are switched to Halpe-26 2D and WHAM 3D in the public index. Unprocessed clips retain their actual previous model labels. `data/wham_progress.json` counts the real completed migration, and `data/motion_summary.json` describes active public models. Superseded public numerical copies/MP4s are removed after switching a clip; baseline dataset NPZs and Git history remain intact. This avoids doubling the deployment tree. The new 26-joint playback supports a browser clock and optional MP4s rendered from the same numerical Halpe-26 data. Per-view JSON/CSV exports preserve the actual 26-joint or 17-joint layout.

The adapter follows the official WHAM inference and flip-merging interfaces and validates checkpoint parameter coverage, so missing weights cannot silently produce randomly initialized results. Unit tests use explicitly synthetic temporary fixtures to check mapping, tracking boundaries and publication; they are not a substitute for the first real GPU smoke run above.

## Current 3D reliability and contact refinement

The original scene fit passed structural checks but missed visible motion failures. A new audit of all 813 v1 clips found 115 with adjacent root steps over 0.2 modeled leg lengths and 286 with both feet more than 0.15 leg lengths above the inferred floor for over 10% of supported frames. These are screening diagnostics, not labeled 3D errors. V1's independent per-frame camera depth, weak temporal/contact constraints, and one floor offset below the clip-wide foot minimum could preserve a good 2D projection while producing implausible 3D motion. V1 acceptance must not be treated as motion validation.

The full-dataset diagnostic refinement is `wham-pitch-contact-v2`; publication additionally requires `wham-pitch-anchor-v3`. WHAM still supplies the learned initialization; final RTMPose Halpe-26 2D data are retained. A single fixed SMPL shape fixes native segment lengths. Root motion in the support plane uses a cubic spline rather than independent depth estimates. The v2 root height is coupled to articulated support feet. In v3, root translation is solved from a fixed forefoot XYZ anchor during the inferred rear-foot drive and lead-foot stance. A fitted lead anchor and a short transition connect the two support phases. Observed lead-foot departure releases the anchor for the tail of the clip. The support equation is part of fitting, not a visual floor correction. Heel penetration and disagreement at the support transition are penalized and screened. A fixed local floor and camera apply to the entire clip; there is no renderer recentering or post-fit floor lowering. This model assumes at least one supporting foot; true airborne follow-through and uncertain support require review.

Initial leg lift and landing hypotheses come from stabilized 2D knees and forefeet. Missing foot observations are masked instead of being interpreted as zero-valued pixels; derivative-only gap interpolation retains its observation mask. A brief airborne foot pause is not treated as landing: the detector requires sustained later stability and backtracks to the preceding deceleration. The rear forefoot supports the drive and the lead forefoot supports the landing; heels can lift. Near landing, the lead ankle, pelvis and rear ankle have a soft ordering constraint in the inferred home-plate direction. No target stride length or population-average pitch is imposed. Broad knee flexion limits and a small allowance off the SMPL knee flexion axis limit implausible bending. Joint-specific temporal penalties allow the throwing arm to move faster than the pelvis and torso. The same numerical support frame is used throughout playback, and the bone renderer preserves limb lengths between saved frames.

Three effective camera-to-support-plane angle initializations (0°, 4°, 12°) and two focal initializations (4× and 12× full-image diagonal) are fitted in six combinations. A screened candidate is selected before comparing objectives. Quality JSON records all candidates and the range of inferred landing strides across screened candidates. This is initialization sensitivity, **not** a confidence interval or measured stride accuracy. Mound slope, unknown camera geometry, depth sign ambiguity, uncertain landing timing and WHAM's motion prior can still affect the result. A flat local support approximation does not establish surveyed mound geometry. The [official baseball rules, mound diagram](https://img.mlbstatic.com/mlb-images/image/upload/mlb/ub08blsefk8wkkd2oemz.pdf) describe a 1:12 downhill slope; the actual camera and mound in these clips are not calibrated.

The completed update has 813 contact-refitted clips and 297 fixed-stance rescue candidates. 44 source-frame-reviewed previews are published; 769 final 3D views are withheld, with all 813 final 2D views retained. These are preview-availability counts, not validated 3D accuracy. [The quality report](https://kbo-pitch-db.github.io/reliability.html) shows selected numerical comparisons. `data/scene_reliability_summary.json` records current completion, screening and withheld counts. `data/pitch_scene_audit.json` contains source alignment, saved-coordinate invariants, unchanged 2D hash counts and sampled official full-mesh reconstruction checks. Camera-stable, phase-resolved v2 sequences are eligible for v3 even when their initial stride was flagged short; a short-stride failure is not an exclusion from rescue fitting. Unresolved phase or camera spread remains held for wider review. Three anchor initializations (the v2 estimate and 1.2/1.8 modeled leg lengths) are optimized independently. These are starting points, never stride targets. Their resulting stride spread and the six-camera seed spread are both checked. The forefoot midpoint must remain stationary during the modeled stance; a moving swing foot is excluded from that test.

Only `screened_estimate` results appear in the final 3D panel or its downloads. Fits with large camera sensitivity, unresolved contacts, or an inferred plant separation below 0.8 modeled leg lengths are withheld for depth review. `stride_at_plant_leg` is native interankle spacing at the inferred landing, normalized by modeled leg length; it is not a measured rubber-to-landing pitch stride. The fitted forefoot-anchor separation is reported separately. The short-stride cutoff is a conservative review heuristic, not a universal pitching limit, and it never stretches a result to a target. For context, a [315-professional-pitcher study](https://pubmed.ncbi.nlm.nih.gov/34240663/) reported an average stride of 78.3% of height; that population statistic does not identify an individual KBO pitch. Failed or pending results retain usable 2D playback and an explicit 3D status, with no fallback to the old trajectory. Passing checks does not validate biomechanical accuracy: player height, absolute scale, depth and stride remain unvalidated. Published previews require source-video visual inspection bound to the exact fitted scene signature. This Codex review checks visible phase correspondence, not measured 3D accuracy. All clips remain ineligible as a ground-truth benchmark.

```bash
cd /home/eis/Workspace/Kichang/kbo-pitchers
# Refine existing genuine WHAM results; private indexes make workers resumable.
.wham-env/bin/python kbo-pitch-db.github.io/scripts/pitch_scene_batch.py --devices cuda:1,cuda:1,cpu,cpu --memory-fraction .15
# One CUDA worker is also supported with --device cuda:1 on refine_pitch_scene.py.
# The exact SMPL evaluation uses the 113 mesh vertices needed by the WHAM regressor and foot markers. CUDA graph acceleration preserves the Adam steps and learning-rate schedule; it was compared against eager optimization and official full-mesh SMPL positions and gradients. Workers share a queue, so CPU and GPU workers can finish different numbers of clips.
.wham-env/bin/python kbo-pitch-db.github.io/scripts/pitch_anchor_batch.py --dataset dataset \
  --devices cuda:1,cuda:1,cuda:1,cuda:1,cpu,cpu --memory-fraction .10
# Inspect candidates against the local source and record signature-bound preview decisions.
.venv/bin/python kbo-pitch-db.github.io/scripts/package_pitch_scene.py --dataset dataset \
  --output kbo-pitch-db.github.io/data --require-complete
.venv/bin/python kbo-pitch-db.github.io/scripts/audit_pitch_scene.py --dataset dataset \
  --public kbo-pitch-db.github.io/data
```

`dataset/motion_wham_publication.csv` and `quality/motion_wham_publication/` record the final publication decision separately from optimizer acceptance. Rebuild them by packaging after a refit.

`dataset/motion_wham_pitch_scene.csv` indexes `poses/3d_pitch_scene/wham/` and `quality/motion_wham_pitch_scene/`. The older `motion_wham_scene.csv` and v1 results remain local for comparison. `motion_wham_pitch_anchor.csv` indexes v3 candidates under `poses/3d_pitch_anchor/wham/`. `motion_wham_final_scene.csv` records the final overlay of v2 diagnostics and v3 candidates, while the publication index determines which can be displayed. Packaging prefers this overlay and never silently relabels an old 3D trajectory as a new result. Model/input/code signatures, fixed transforms, shared shape, native bone lengths, knee limits, source timestamps, saved detector scores and publication gates are checked separately from motion quality. The primary fixed-length body representation is `frames[].anatomy` / `anatomy_joints` (NPZ: `anatomical_keypoints`), and the 3D CSV exports these 24 native joint centers. Supplementary COCO-17 `keypoints` come from WHAM's surface-vertex regressor: their distances can vary with pose even with a fixed SMPL shape, and they must not be used as a rigid bone chain. JSON preserves both with explicit semantics; native-joint CSV scores are blank because no calibrated 3D confidence exists. Public JSON includes the exact saved joint centers, illustrative SMPL-24 anatomy, six foot markers, contact hypotheses and camera sensitivity. The bone illustration is not measured skeletal anatomy or an OpenSim simulation.

A separate publication screen checks for observed preparation, consistency between leg lift/landing and a fast-arm-motion proxy, and a stride directed approximately toward home plate. This catches clips that start in follow-through and cases where a returning rear leg was mislabeled as a new pitch. Wrist speed is not an annotated ball-release event; unresolved cases are held for review. A scene-specific visual preview record is also required; its scope is visual correspondence, not 3D measurement accuracy. An exact-scene-signature visual veto can additionally be recorded in `dataset/scene_review.json`. These vetoes do not rewrite the fitted coordinates or bypass numerical failures.

The “3D available” filter finds clips that passed the publication gate; every clip retains its original final 2D. The browser keeps final 2D and final 3D side by side, including on narrow screens. One clock controls playback, frame stepping, speed and seeking. Rotation changes the 3D viewing camera only. Three.js r169 is served locally with its MIT license; a canvas fallback supports browsers without WebGL.

The initial WHAM migration completed on 2026-10-27: all 813 source clips / 135,741 frames use RTMPose-m Halpe-26 and WHAM. Both RTX 3090 workers exited successfully, with no failed or incomplete clips. `data/wham_progress.json` records 813 completed and zero remaining baseline clips; `data/motion_summary.json` describes the active models and tracking/reprojection status. This baseline completion count does not mean every clip has publishable final 3D.

`data/wham_audit.json` records the initial hip-relative migration input-hash, frame/time, joint-layout, confidence and provenance checks, including all 3,252 public numerical payloads and links to 48 optional Halpe-26 MP4s. For that initial refinement, median refined/raw 3D acceleration is 0.360 across clips; this measures stability, not pitching accuracy. `data/wham_batch_report.json` records worker exit codes and completion counts.

The viewer displays only the final refined results: 2D on the left and monocular 3D on the right, including on narrow screens. One browser clock synchronizes playback, speed, seeking and frame stepping. Rotate or tilt the 3D view by dragging or using its sliders; the 2D projection stays fixed. Each panel downloads its exact final JSON/CSV independently. Amber joints mark short-gap interpolation; long gaps remain absent. Rendering uses fixed clip-wide bounds. Original/intermediate numerical assets and optional MP4s remain in the archive for reproducibility.

Inspired by [OpenCap Monocular](https://github.com/utahmobl/opencap-monocular), the implementation separates learned 3D initialization from image, body and temporal constraints. WHAM supplies the learned 3D and SMPL body model; the current contact-constrained SMPL fit uses bounded interpolation, image observations, fixed shape, knee limits and temporal constraints. Calibrated camera extrinsics, OpenSim IK and force estimates are not implemented. Pitching accuracy has not been validated. The original priors use hip-relative model meters. Final scene coordinates use one inferred local support frame per clip; y is normal to its estimated floor, and x/z lie in that plane.

```bash
# Validate actual WHAM assets against original source frames, times and hashes.
../.wham-env/bin/python scripts/audit_wham.py --dataset ../dataset --public data --require-complete
# Package completed WHAM clips, or add optional skeleton MP4 downloads.
../.venv/bin/python scripts/package_motion.py --dataset ../dataset --output data --backend wham
../.venv/bin/python scripts/render_motion_videos.py --output data --limit 48
```

`../dataset/motion_wham.csv` indexes raw/refined Halpe-26, raw/refined WHAM and quality reports. Model SHA256 and input signatures accompany priors. Detector scores remain distinct from interpolation/outlier flags. The browser loads the final refined 2D/3D trajectories directly. Quality reports describe support, acceleration and reprojection, not ground-truth accuracy. The earlier BlazePose experiment remains in dataset `motion.csv` and the legacy `infer_monocular.py`/`refine_motion.py` scripts; these are not the WHAM execution path.

Run `node tests/motion_viewer.test.cjs` and `python -m unittest discover -s tests -p 'test_*.py' -v` for view/export behavior and numerical invariants. Browser visual testing needs a browser-capable environment and local server.

A static, bilingual (English/Korean) explorer for KBO pitching motion and 2D/3D pose estimates. The two final motion panels render numerical keypoints on a plain background and offer JSON/CSV downloads. Optional skeleton-only H.264 MP4 files remain in the data archive. Thumbnails are rendered from skeletons; no broadcast images or audio are hosted.

## Run locally

Serve only this site folder, for example `python3 -m http.server 8000` from this directory, then open `http://localhost:8000`. Do not serve the parent workspace, which contains private source media and local backups.

The current package contains 813 real pose sequences. All clips retain final 2D playback and downloads. Only screened, visually reviewed 3D previews support the synchronized second panel and its downloads; remaining clips show a review status. A 48-clip subset also retains original Halpe-26 skeleton MP4s in the data archive. Names linked automatically through Naver are not verified identities, and pose estimates are not validated biomechanical measurements.

## Build data

The builder uses `../dataset/clips.csv`, `sources.csv`, and optional Naver metadata. It reads crop-video FPS locally but never copies source videos, broadcast overlays, or broadcast thumbnails into the output. Only numerical keypoints feed the animation renderer. Install `numpy`, `opencv-python`, and `imageio-ffmpeg` in the Python environment (already available in the workspace `.venv`).

```bash
../.venv/bin/python scripts/build_site_data.py --dataset ../dataset --output data --limit 0 --numeric-only
```

The default build replaces `data/assets/`; `--numeric-only` preserves existing MP4 files and adds numeric data and skeleton thumbnails. It reserves clips from recent source videos, then groups repeat pitchers; `--clip-ids ID,ID` and `--pitcher NAME` restrict pose selection. `--max-video-mb` limits the total generated skeleton video size. `--sample` generates an illustrative skeleton animation without source media.

Each clip can also show its original video through a timestamped YouTube link and an optional, click-to-load YouTube embed. The dataset provider video ID and exact start/end times populate `sourceVideoId`, `sourceId`, `sourceStart`, `sourceEnd`, and `sourceUrl`. Watch links round the start down to a whole second; the embed uses that start and rounds the end up. YouTube seeking is approximate and is independent of the pose controls. The iframe loads from `youtube-nocookie.com` only after expansion and is removed when collapsed or when a different clip is selected. A direct YouTube link remains available when embedding or regional playback is restricted. No broadcast video is copied into the site.

No broadcast samples are currently approved for self-hosted publication. Choosing representative broadcast samples later requires an explicit media policy and an allowlist; selecting pose clip IDs does not publish their broadcast footage.

## Public data boundary

The site index (`data/clips.json`, schema version 7) and every clip declare `mediaPolicy: "pose-only"`. `poseVideo`, `poseThumbnail`, and `pose` point to skeleton MP4, skeleton JPEG, and numerical JSON files. `poseRefined`, `pose3d`, `pose3dRaw`, and `motionQuality` link additional numerical assets. The browser does not use legacy `video`, `overlay`, or `thumbnail` fields, including when an old index is returned. `packageVersion` refreshes replaced media.

Each pose JSON declares FPS, coordinates, model and joint names. Halpe-26 2D records contain nullable `[x, y, confidence]` crop-pixel values; WHAM 3D records contain nullable `[x, y, z, score]` for COCO-17. Final scene files use model-estimated meters in a clip-fixed inferred support frame and add SMPL-24 anatomy and six foot markers. Archived raw priors remain hip-relative. Both retain source frame/time; WHAM scores inherit 2D confidence. Rendering uses fixed bounds for the complete clip, omits joints below 0.2 confidence, preserves missing frames, and adds no audio.

Previous packaged broadcast assets were moved outside this site to the local workspace's `videos/site_media_archive/`. Dataset originals remain local. Deploy only the current site files and `data/`; do not upload this archive or the repository's `.git` directory.

These local changes do not remove media from an already published deployment or from previous Git commits. Existing public hosting and repository history must be removed separately when retiring the GitHub repository.

## Player portraits

The pitcher group headers use KBO's 2024 profile portraits, matching the source clips' season. `data/player_profiles.json` contains 155 verified pitcher profiles from the source-game rosters, including all 123 pitchers represented by the 813 clips. Each profile records the Naver/KBO player code, season name, image URL, local image path, KBO profile-page URL, dimensions, SHA-256 and verification date. The two players whose KBO names have changed retain their 2024 names and also record their current names. The browser requires a matching name and player code before displaying a portrait.

Verified JPEGs are cached in `data/portraits/2024/`, outside the builder's replaceable `data/assets/` directory. The browser loads these local files first, retries the original KBO image URL on failure, then shows the person icon if both fail. The detail view retains the KBO attribution link. All 123 displayed portraits load without a request to KBO's CDN.

## Player identity evidence

Player identity is resolved separately from an exact clip-to-pitch link. The existing 390 automatically accepted Naver links remain unchanged. An additional 318 clips have exact pitcher-name agreement in at least two distinct in-clip scoreboard frames. Another 105 were checked by Codex reading broadcast text: 63 scoreboard names, 17 scoreboard inning/half/pitch-count states with a unique pitcher identity in the game records, and 25 uniform names/numbers checked against the original game lineup. These are automated and agent visual reviews, **not human ground-truth annotations**. No new pitch links, pitch types, speeds or pitch numbers are inferred from player identity alone.

All 813 clips now have names and player codes; no `Unknown pitcher` entries remain. `data/player_identity_summary.json` records coverage and missing portraits. `data/player_identity_evidence.json` preserves the 423 supplemental records with source-video timestamps, OCR readings or review notes and lineup references. The local dataset retains full evidence in `player_identity.json`, `player_identity_review.json`, and `naver_sports/evidence/identities/`. Broadcast screenshots are not copied into the public site.

`scripts/player_identity.py` validates clip/source/game IDs, clip intervals, the unique game-roster name/code pair and the evidence method. Uniform-number reviews also recheck the cached original lineup, including the home/away side. Both the full builder and metadata-only updater use this resolver. The portable evidence manifest preserves these corrections when the local OCR/review files are absent. Metadata-only updates preserve every pose, scene, media, source and pitch-statistics field.

Run from the workspace root:

```bash
# Reuse existing identities; update only names, evidence and coverage reports.
.venv/bin/python kbo-pitch-db.github.io/scripts/player_identity.py
# Recheck missing names using Korean OCR, including the two TVING score bugs.
.venv/bin/python kbo-pitch-db.github.io/scripts/recover_player_identity.py
# Verify the source-game roster against KBO and refresh local season portraits.
.venv/bin/python kbo-pitch-db.github.io/scripts/sync_player_profiles.py
python3 -m unittest discover -s kbo-pitch-db.github.io/tests -p test_player_identity.py
node kbo-pitch-db.github.io/tests/player_identity.test.cjs
```

Run the metadata-only updater after refreshing OCR or profiles. Unreadable or conflicting future observations remain unresolved and appear in the coverage report; fuzzy OCR names are not accepted.
