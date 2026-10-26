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

The 2D model really predicts all 26 landmarks, including head, neck, hip, toes and heels. Existing pitcher boxes select the same person. Its first 17 joints feed the official WHAM feature encoder and recurrent/SMPL network, with flip evaluation. WHAM state resets at identity changes and frame gaps. Only camera-relative joints are exported; zero camera angular velocity does not establish a calibrated world trajectory. The learned WHAM 3D joints and SMPL pose/shape parameters are cached separately from the older baseline. The displayed 3D confidence is inherited 2D RTMPose confidence, not a calibrated 3D score.

Only successfully inferred, aligned clips are switched to Halpe-26 2D and WHAM 3D in the public index. Unprocessed clips retain their actual previous model labels. `data/wham_progress.json` counts the real completed migration, and `data/motion_summary.json` describes active public models. Superseded public numerical copies/MP4s are removed after switching a clip; baseline dataset NPZs and Git history remain intact. This avoids doubling the deployment tree. The new 26-joint playback supports a browser clock and optional MP4s rendered from the same numerical Halpe-26 data. Per-view JSON/CSV exports preserve the actual 26-joint or 17-joint layout.

The adapter follows the official WHAM inference and flip-merging interfaces and validates checkpoint parameter coverage, so missing weights cannot silently produce randomly initialized results. Unit tests use explicitly synthetic temporary fixtures to check mapping, tracking boundaries and publication; they are not a substitute for the first real GPU smoke run above.

## 2D / 3D motion comparison

The archive contains 813 source clips / 135,741 frames. `data/wham_progress.json` records completed WHAM migration, and `data/motion_summary.json` describes the active public models and tracking/reprojection status. All remain unreviewed. `data/wham_audit.json`, when present, contains the final numerical audit.

The 813 packaged clips include original/refined 2D and learned/refined monocular 3D. Switch views, adjust the correction percentage, and rotate 3D by dragging or using rotation/tilt sliders. Playback speed, seeking and frame stepping use a browser clock for numerical motion views. Optional MP4 files remain available as downloads. JSON/CSV export the selected coordinates and current correction blend; the MP4 download remains the original 2D skeleton. Amber joints mark short-gap interpolation; long gaps remain absent. Rendering uses fixed clip-wide bounds.

Inspired by [OpenCap Monocular](https://github.com/utahmobl/opencap-monocular), the implementation separates learned 3D initialization from image, body and temporal constraints. WHAM supplies the learned 3D and SMPL body model; confidence-weighted acceleration smoothing, bounded interpolation, image observations, a weak-perspective scale and segment lengths refine the exported joints. Calibrated camera extrinsics, OpenSim IK, joint limits and force estimates are not implemented. Pitching accuracy has not been validated. Coordinates are model-estimated meters relative to the hips, with x right, y up and z toward the camera.

```bash
# Validate actual WHAM assets against original source frames, times and hashes.
../.wham-env/bin/python scripts/audit_wham.py --dataset ../dataset --require-complete
# Package completed WHAM clips, or add optional skeleton MP4 downloads.
../.venv/bin/python scripts/package_motion.py --dataset ../dataset --output data --backend wham
../.venv/bin/python scripts/render_motion_videos.py --output data --limit 48
```

`../dataset/motion_wham.csv` indexes raw/refined Halpe-26, raw/refined WHAM and quality reports. Model SHA256 and input signatures accompany priors. Detector scores remain distinct from interpolation/outlier flags. The browser correction slider blends two computed trajectories; it does not rerun temporal optimization. Zero correction restores original coordinates. Quality reports describe support, acceleration and reprojection, not ground-truth accuracy. The earlier BlazePose experiment remains in dataset `motion.csv` and the legacy `infer_monocular.py`/`refine_motion.py` scripts; these are not the WHAM execution path.

Run `node tests/motion_viewer.test.cjs` and `python -m unittest discover -s tests -p 'test_*.py' -v` for view/export behavior and numerical invariants. Browser visual testing needs a browser-capable environment and local server.

A static, bilingual (English/Korean) explorer for KBO pitching motion and 2D/3D pose estimates. Playback renders numerical keypoints on a plain background, with optional skeleton-only H.264 MP4 downloads and no broadcast images or audio. Thumbnails are rendered from the same skeletons.

## Run locally

Serve only this site folder, for example `python3 -m http.server 8000` from this directory, then open `http://localhost:8000`. Do not serve the parent workspace, which contains private source media and local backups.

The current package contains 813 real pose sequences. All clips support numeric playback; a 48-clip subset also provides Halpe-26 skeleton MP4 downloads. Playback supports speed selection, seeking, and frame stepping. Pose JSON, CSV, and skeleton MP4 downloads remain available. Names linked automatically through Naver are not verified identities, and pose estimates are not validated biomechanical measurements.

## Build data

The builder uses `../dataset/clips.csv`, `sources.csv`, and optional Naver metadata. It reads crop-video FPS locally but never copies source videos, broadcast overlays, or broadcast thumbnails into the output. Only numerical keypoints feed the animation renderer. Install `numpy`, `opencv-python`, and `imageio-ffmpeg` in the Python environment (already available in the workspace `.venv`).

```bash
../.venv/bin/python scripts/build_site_data.py --dataset ../dataset --output data --limit 0 --numeric-only
```

The default build replaces `data/assets/`; `--numeric-only` preserves existing MP4 files and adds numeric data and skeleton thumbnails. It reserves clips from recent source videos, then groups repeat pitchers; `--clip-ids ID,ID` and `--pitcher NAME` restrict pose selection. `--max-video-mb` limits the total generated skeleton video size. `--sample` generates an illustrative skeleton animation without source media.

Each clip can also show its original video through a timestamped YouTube link and an optional, click-to-load YouTube embed. The dataset provider video ID and exact start/end times populate `sourceVideoId`, `sourceId`, `sourceStart`, `sourceEnd`, and `sourceUrl`. Watch links round the start down to a whole second; the embed uses that start and rounds the end up. YouTube seeking is approximate and is independent of the pose controls. The iframe loads from `youtube-nocookie.com` only after expansion and is removed when collapsed or when a different clip is selected. A direct YouTube link remains available when embedding or regional playback is restricted. No broadcast video is copied into the site.

No broadcast samples are currently approved for self-hosted publication. Choosing representative broadcast samples later requires an explicit media policy and an allowlist; selecting pose clip IDs does not publish their broadcast footage.

## Public data boundary

The site index (`data/clips.json`, schema version 5) and every clip declare `mediaPolicy: "pose-only"`. `poseVideo`, `poseThumbnail`, and `pose` point to skeleton MP4, skeleton JPEG, and numerical JSON files. `poseRefined`, `pose3d`, `pose3dRaw`, and `motionQuality` link additional numerical assets. The browser does not use legacy `video`, `overlay`, or `thumbnail` fields, including when an old index is returned. `packageVersion` refreshes replaced media.

Each pose JSON declares FPS, coordinates, model and joint names. Halpe-26 2D records contain nullable `[x, y, confidence]` crop-pixel values; WHAM 3D records contain nullable `[x, y, z, score]` in hip-relative model meters for COCO-17. Both retain source frame/time; WHAM scores inherit 2D confidence. Rendering uses fixed bounds for the complete clip, omits joints below 0.2 confidence, preserves missing frames, and adds no audio.

Previous packaged broadcast assets were moved outside this site to the local workspace's `videos/site_media_archive/`. Dataset originals remain local. Deploy only the current site files and `data/`; do not upload this archive or the repository's `.git` directory.

These local changes do not remove media from an already published deployment or from previous Git commits. Existing public hosting and repository history must be removed separately when retiring the GitHub repository.

## Player portraits

The pitcher group headers use KBO's externally hosted 2024 profile portraits, matching the source clips' season. `data/player_profiles.json` maps the Naver/KBO player code to a name, portrait URL, KBO profile-page URL, image year, and verification date. The current 21 names and codes were checked against KBO player pages, and all image URLs were fetched and decoded successfully. The browser uses a portrait only when the clip's recorded name matches the entry for its player code. This confirms the profile mapping, not the automatic assignment of that identity to broadcast footage.

Photos are requested directly from KBO's image CDN without copying them into the repository; the clip detail includes a KBO photo-source link. Unknown pitchers and failed image requests show a generic person icon. Individual pitch cards and the pose player retain their skeleton previews. An unavailable portrait manifest also falls back to icons. The data builder replaces only `data/assets/` and `data/clips.json`, so it preserves the portrait manifest.
