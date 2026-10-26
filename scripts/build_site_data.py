#!/usr/bin/env python3
"""Build a small static site data bundle from the KBO dataset folder."""
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]



BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def render_pose_media(pose, target):
    """Render only numeric keypoints; never read broadcast images or audio."""
    import cv2
    import numpy as np
    import imageio_ffmpeg

    def visible(point):
        return (point is not None and len(point) >= 2
                and np.isfinite(point[:2]).all()
                and (len(point) < 3 or point[2] is None or point[2] >= .2))

    points = np.array([p[:2] for frame in pose['frames']
                       for p in frame['keypoints'] if visible(p)])
    if not pose["frames"] or not np.isfinite(pose["fps"]) or pose["fps"] <= 0:
        raise ValueError("Pose media requires frames and a positive finite FPS")
    width, height = 480, 480
    if points.size:
        low, high = points.min(axis=0), points.max(axis=0)
    else:
        low, high = np.array([0, 0]), np.array([1, 1])
    scale = min((width - 64) / max(1, high[0] - low[0]),
                (height - 64) / max(1, high[1] - low[1]))
    center = (low + high) / 2
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(pose['fps']), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target / 'pose.mp4')]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for i, frame in enumerate(pose['frames']):
                image = np.full((height, width, 3), (27, 32, 20), dtype=np.uint8)
                joints = [tuple(np.rint((np.array(p[:2]) - center) * scale + [width / 2, height / 2]).astype(int))
                          if visible(p) else None for p in frame['keypoints']]
                for a, b in BONES:
                    if a < len(joints) and b < len(joints) and joints[a] is not None and joints[b] is not None:
                        cv2.line(image, joints[a], joints[b], (104, 240, 180), 3, cv2.LINE_AA)
                for point in joints:
                    if point is not None:
                        cv2.circle(image, point, 4, (220, 250, 234), -1, cv2.LINE_AA)
                if i == min(len(pose['frames']) - 1, round(.8 * pose['fps'])):
                    cv2.imwrite(str(target / 'pose_thumbnail.jpg'), cv2.resize(image, (180, 180)))
                process.stdin.write(image.tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError(f'Pose video encoding failed: {target}')


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def render_thumbnail(pose, target):
    import cv2
    import numpy as np
    def valid(p):
        return p is not None and (p[2] is None or p[2] >= .2)
    all_points = np.array([p[:2] for f in pose['frames'] for p in f['keypoints'] if valid(p)])
    low, high = (all_points.min(0),all_points.max(0)) if all_points.size else (np.zeros(2),np.ones(2))
    center = (low+high)/2
    scale = 150/max(1,*(high-low))
    frame = pose['frames'][min(len(pose['frames'])-1,round(.8*pose['fps']))]
    points = [tuple(np.rint((np.array(p[:2])-center)*scale+90).astype(int)) if valid(p) else None for p in frame['keypoints']]
    picture = np.full((180,180,3),(27,32,20),np.uint8)
    for a,b in pose.get('edges',BONES):
        if points[a] and points[b]:cv2.line(picture,points[a],points[b],(104,240,180),2,cv2.LINE_AA)
    for point in points:
        if point:cv2.circle(picture,point,2,(230,250,230),-1,cv2.LINE_AA)
    cv2.imwrite(str(target/'pose_thumbnail.jpg'),picture)



def source_reference(source, clip):
    """Link the recorded provider ID and original timeline, without publishing media."""
    video_id = source.get("provider_video_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return {}
    try:
        start, end = float(clip["start_time"]), float(clip["end_time"])
    except (KeyError, TypeError, ValueError):
        return {}
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return {}
    return {"sourceId": source["source_id"], "sourceVideoId": video_id,
            "sourceStart": start, "sourceEnd": end,
            "sourceUrl": f"https://www.youtube.com/watch?v={video_id}&t={math.floor(start)}s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("../dataset"))
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=48, help="maximum number of clips to package (default: 48)")
    ap.add_argument("--max-video-mb", type=float, default=185, help="maximum total pose video size (default: 185 MB)")
    ap.add_argument("--featured-pitchers", type=int, default=10, help="number of repeat pitchers to feature first")
    ap.add_argument("--clips-per-pitcher", type=int, default=3, help="clips to package for each featured pitcher")
    ap.add_argument("--recent-sources", type=int, default=10, help="reserve one clip from each recently added source")
    ap.add_argument("--clip-ids", help="comma-separated dataset_clip_id values")
    ap.add_argument("--pitcher", help="only include clips with this automatically linked pitcher name")
    ap.add_argument("--sample", action="store_true", help="write illustrative demo data without copying real media")
    ap.add_argument('--numeric-only',action='store_true',help='render with a browser clock; preserve existing skeleton MP4s and add only numeric data/thumbnails')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.sample:
        write_sample(out)
        return
    dataset = args.dataset.resolve()
    clips = rows(dataset / "clips.csv")
    source_rows = rows(dataset / "sources.csv")
    sources = {r["source_id"]: r for r in source_rows}
    accepted = {r["dataset_clip_id"]: r for r in rows(dataset / "naver_sports/auto_accepted.csv")}
    pitches = {r["naver_pitch_id"]: r for r in rows(dataset / "naver_sports/pitches.csv")}
    def player_identity(clip):
        match = accepted.get(clip["dataset_clip_id"], {})
        name = match.get("pitcher_name", "").strip()
        code = pitches.get(match.get("naver_pitch_id", ""), {}).get("pitcher_code", "").strip()
        return ("naver:" + code if code else "name:" + name if name else "unknown:" + clip["dataset_clip_id"], code)
    selected_ids = set(args.clip_ids.split(",")) if args.clip_ids else None
    result = []
    copied_bytes = 0
    asset_root = out / "assets"
    if asset_root.exists() and not args.numeric_only:
        shutil.rmtree(asset_root)
    asset_root.mkdir(exist_ok=True)
    # Keep multiple clips for the same automatically linked pitcher name in the
    # static bundle, while rotating source videos before taking repeat clips.
    if not selected_ids and not args.pitcher:
        buckets = defaultdict(list)
        unknown = []
        for clip in clips:
            pitch = accepted.get(clip["dataset_clip_id"], {})
            if pitch.get("pitcher_name"):
                buckets[player_identity(clip)[0]].append(clip)
            else:
                unknown.append(clip)
        for name, group in list(buckets.items()):
            by_source = defaultdict(list)
            for clip in sorted(group, key=lambda item: (sources.get(item["source_id"], {}).get("game_date", ""), item["start_time"], item["dataset_clip_id"])):
                by_source[clip["source_id"]].append(clip)
            buckets[name] = [clip for round_index in range(max(map(len, by_source.values())))
                             for source_id in sorted(by_source)
                             if round_index < len(by_source[source_id])
                             for clip in [by_source[source_id][round_index]]]
        featured = sorted((name for name, group in buckets.items() if len(group) >= 2),
                          key=lambda name: (-len(buckets[name]), name))[:max(0, args.featured_pitchers)]
        def freshness(row):
            run_state = Path(row.get("run_dir", "")) / "run_state.json"
            run_dir = Path(row.get("run_dir", ""))
            source_file = Path(row.get("source_path", ""))
            if run_state.is_file():
                return run_state.stat().st_mtime
            if run_dir.is_dir():
                return run_dir.stat().st_mtime
            return source_file.stat().st_mtime if source_file.is_file() else 0
        recent_ids = [row["source_id"] for row in sorted(source_rows, key=freshness, reverse=True)[:args.recent_sources]]
        recent = []
        for source_id in recent_ids:
            candidates = [clip for clip in clips if clip["source_id"] == source_id]
            candidates.sort(key=lambda clip: (
                not bool(accepted.get(clip["dataset_clip_id"], {}).get("pitcher_name")),
                (dataset / clip["crop_video"]).stat().st_size if (dataset / clip["crop_video"]).is_file() else float("inf"),
                clip["dataset_clip_id"]))
            if candidates:
                recent.append(candidates[0])
        ordered = recent + [clip for name in featured for clip in buckets[name][:args.clips_per_pitcher]]
        ordered += [group[0] for name, group in sorted(buckets.items()) if name not in featured]
        selected = {clip["dataset_clip_id"] for clip in ordered}
        ordered += [clip for name in featured for clip in buckets[name][args.clips_per_pitcher:]
                    if clip["dataset_clip_id"] not in selected]
        clips = list({clip["dataset_clip_id"]: clip for clip in ordered + clips}.values())
    for clip in clips:
        cid = clip["dataset_clip_id"]
        pitch = accepted.get(cid, {})
        player = pitch.get("pitcher_name", "")
        player_key, player_code = player_identity(clip)
        if selected_ids and cid not in selected_ids:
            continue
        if args.pitcher and player != args.pitcher:
            continue
        src = sources.get(clip["source_id"], {})
        crop_rel = clip.get("crop_video")
        pose_rel = clip.get("pose_2d")
        if not crop_rel or not pose_rel or not (dataset / crop_rel).is_file() or not (dataset / pose_rel).is_file():
            continue
        import numpy as np
        try:
            import cv2
            video_capture = cv2.VideoCapture(str(dataset / crop_rel))
            video_fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            video_capture.release()
        except ImportError:
            video_fps = 30.0
        with np.load(dataset / pose_rel, allow_pickle=False) as z:
            points = z["keypoints"]
            scores = z["scores"] if "scores" in z.files else None
            frame_index = z["frame_index"] if "frame_index" in z.files else range(len(points))
            source_time = z["source_time"] if "source_time" in z.files else None
            missing = z["missing"] if "missing" in z.files else None
            fps = float(z["fps"].item()) if "fps" in z.files else video_fps
            x1, y1 = float(clip["crop_x1"]), float(clip["crop_y1"])
            frames = []
            for i, row in enumerate(points):
                joints = []
                for j, point in enumerate(row):
                    joint_missing = bool(missing[i]) if missing is not None and getattr(missing[i], "ndim", 0) == 0 else bool(missing[i][j]) if missing is not None else False
                    if joint_missing or len(point) < 2 or not np.isfinite(point[:2]).all():
                        joints.append(None)
                    else:
                        conf = float(scores[i][j]) if scores is not None and np.isfinite(scores[i][j]) else None
                        joints.append([round(float(point[0]) - x1, 2), round(float(point[1]) - y1, 2), round(conf, 4) if conf is not None else None])
                frames.append({"frame": i, "source_frame": int(frame_index[i]), "time": round(i / fps, 5), "source_time": round(float(source_time[i]), 5) if source_time is not None else None, "keypoints": joints})
        rel_folder = Path("assets") / cid
        target = out / rel_folder
        target.mkdir(parents=True, exist_ok=True)
        pose_name = "pose.json"
        pose_data = {"fps": fps, "coordinate_system": "crop pixels", "joints": JOINTS, "frames": frames}
        if args.numeric_only:
            render_thumbnail(pose_data, target)
        else:
            render_pose_media(pose_data, target)
        video_size = (target / "pose.mp4").stat().st_size if (target/'pose.mp4').exists() else 0
        if args.max_video_mb and copied_bytes + video_size > args.max_video_mb * 1024 * 1024:
            shutil.rmtree(target)
            continue
        copied_bytes += video_size
        with (target / pose_name).open("w", encoding="utf-8") as f:
            json.dump({"fps": fps, "coordinate_system": "crop pixels", "joints": JOINTS, "frames": frames}, f, ensure_ascii=False, separators=(",", ":"))
        duration = float(clip["end_time"]) - float(clip["start_time"])
        result.append({"id": cid, "player": player or "Unknown pitcher", "playerKnown": bool(player), "playerKey": player_key, "playerCode": player_code, "playerNameSource": "naver_auto_accepted" if player else "unknown", "team": src.get("teams_text", "Unknown"), "hand": "", "pitch": pitch.get("pitch_type_raw", ""), "speed": pitch.get("speed_kmh", ""), "date": src.get("game_date", ""), "matchup": src.get("teams_text", ""), "inning": pitch.get("inning", ""), "pitchNumber": pitch.get("pitcher_pitch_number", ""), "duration": round(duration, 2), "fps": fps, "frames": len(frames), "source": src.get("source_name", src.get("source_path", "")), **source_reference(src, clip), "model": "RTMPose", "coordinate": "17-joint COCO; crop pixels", "poseStatus": "missing frames present" if any(not any(p for p in fr["keypoints"]) for fr in frames) else "available", "confidence": "Per-joint confidence included in pose JSON", "mediaPolicy": "pose-only", "poseVideo": str(rel_folder / "pose.mp4") if (target/"pose.mp4").exists() else "", "poseThumbnail": str(rel_folder / "pose_thumbnail.jpg"), "pose": str(rel_folder / pose_name), "example": False, "review": clip.get("review_status", "unreviewed"), "poseFile": pose_name})
        if args.limit and len(result) >= args.limit:
            break
    index = {"schemaVersion": 3, "mediaPolicy": "pose-only", "generatedFrom": "dataset/clips.csv",
             "packageVersion": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
             "packageStats": {"videoBytes": copied_bytes, "videoMegabytes": round(copied_bytes / 1024 / 1024, 1)}, "clips": result}
    (out / "clips.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    if (dataset / 'motion.csv').is_file():
        from package_motion import package
        package(dataset, out)


def write_sample(out):
    sample = {"schemaVersion": 3, "mediaPolicy": "pose-only", "generatedFrom": "illustrative sample", "clips": [{"id": "DEMO-001", "player": "Sample Pitcher", "playerKo": "샘플 투수", "playerKnown": False, "playerKey": "unknown:DEMO-001", "team": "Demo club", "teamKo": "예시 구단", "hand": "", "pitch": "Fastball", "pitchKo": "직구", "date": "2024-08-22", "matchup": "Demo club vs Opponent", "inning": "1", "pitchNumber": "—", "duration": 2.8, "fps": 30, "frames": 84, "source": "Illustrative synthetic demo — no broadcast footage", "sourceUrl": "", "model": "Synthetic demo skeleton", "coordinate": "17-joint COCO; crop pixels", "poseStatus": "synthetic demo", "mediaPolicy": "pose-only", "poseVideo": "assets/demo/pose.mp4", "poseThumbnail": "assets/demo/pose_thumbnail.jpg", "pose": "assets/demo/pose.json", "example": True}]}
    if (out / "assets").exists():
        shutil.rmtree(out / "assets")
    folder = out / "assets/demo"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(84):
        phase = i / 83
        points = [[160, 35, .99], [153, 31, .98], [167, 31, .98], [147, 34, .95], [173, 34, .95], [137, 83, .97], [183, 83, .97], [125, 117, .94], [197, 108, .96], [121 + phase * 18, 157 - phase * 34, .91], [205, 142 - phase * 18, .90], [144, 165, .98], [176, 165, .98], [136, 224 - phase * 16, .94], [183, 223 - phase * 28, .95], [128, 285, .92], [193, 280, .93]]
        frames.append({"frame": i, "time": round(i / 30, 4), "keypoints": points})
    pose = {"fps": 30, "coordinate_system": "crop pixels", "joints": JOINTS, "frames": frames}
    (folder / "pose.json").write_text(json.dumps(pose, separators=(",", ":")), encoding="utf-8")
    render_pose_media(pose, folder)
    (out / "clips.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
