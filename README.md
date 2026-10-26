# KBO Pitch Lab

A static, bilingual (English/Korean) explorer for KBO pitching motion and 2D pose estimates. Hosted media playback uses skeleton-only H.264 MP4 animations rendered from COCO-17 coordinates on a plain background, without broadcast images or audio. Thumbnails are rendered from the same skeletons.

## Run locally

Serve only this site folder, for example `python3 -m http.server 8000` from this directory, then open `http://localhost:8000`. Do not serve the parent workspace, which contains private source media and local backups.

The current package contains 48 real pose sequences. Playback supports speed selection, seeking, and frame stepping. Pose JSON, CSV, and skeleton MP4 downloads remain available. Names linked automatically through Naver are not verified identities, and pose estimates are not validated biomechanical measurements.

## Build data

The builder uses `../dataset/clips.csv`, `sources.csv`, and optional Naver metadata. It reads crop-video FPS locally but never copies source videos, broadcast overlays, or broadcast thumbnails into the output. Only numerical keypoints feed the animation renderer. Install `numpy`, `opencv-python`, and `imageio-ffmpeg` in the Python environment (already available in the workspace `.venv`).

```bash
../.venv/bin/python scripts/build_site_data.py --dataset ../dataset --output data --limit 48 --max-video-mb 185
```

The build replaces `data/assets/`. It reserves clips from recent source videos, then groups repeat pitchers; `--clip-ids ID,ID` and `--pitcher NAME` restrict pose selection. `--max-video-mb` limits the total generated skeleton video size. `--sample` generates an illustrative skeleton animation without source media.

Each clip can also show its original video through a timestamped YouTube link and an optional, click-to-load YouTube embed. The dataset provider video ID and exact start/end times populate `sourceVideoId`, `sourceId`, `sourceStart`, `sourceEnd`, and `sourceUrl`. Watch links round the start down to a whole second; the embed uses that start and rounds the end up. YouTube seeking is approximate and is independent of the pose controls. The iframe loads from `youtube-nocookie.com` only after expansion and is removed when collapsed or when a different clip is selected. A direct YouTube link remains available when embedding or regional playback is restricted. No broadcast video is copied into the site.

No broadcast samples are currently approved for self-hosted publication. Choosing representative broadcast samples later requires an explicit media policy and an allowlist; selecting pose clip IDs does not publish their broadcast footage.

## Public data boundary

The site index (`data/clips.json`, schema version 3) and every clip declare `mediaPolicy: "pose-only"`. `poseVideo`, `poseThumbnail`, and `pose` point to skeleton MP4, skeleton JPEG, and numerical JSON files. The browser does not use legacy `video`, `overlay`, or `thumbnail` fields, including when an old index is returned. `packageVersion` refreshes replaced media.

Each pose JSON contains FPS, coordinate system, the COCO-17 joint order, and frame records with source frame/time and nullable `[x, y, confidence]` values in crop-pixel coordinates. Rendering uses fixed bounds for the complete clip, omits joints below 0.2 confidence, preserves missing frames, and adds no audio.

Previous packaged broadcast assets were moved outside this site to the local workspace's `videos/site_media_archive/`. Dataset originals remain local. Deploy only the current site files and `data/`; do not upload this archive or the repository's `.git` directory.

These local changes do not remove media from an already published deployment or from previous Git commits. Existing public hosting and repository history must be removed separately when retiring the GitHub repository.
