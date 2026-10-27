#!/usr/bin/env python3
"""Identify pitchers independently of exact pitch matching, with frame evidence.

This never promotes a pitch link or fills pitch-specific statistics. Run with the
workspace .venv (PyAV, OpenCV) and the bundled Korean Tesseract installation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'kbo_pitch_motion_pilot'))
import av
import cv2
from kbo_pitch_pilot.__main__ import frame_at
from kbo_pitch_pilot.match_naver import LAYOUTS, clean_name, name_match, ocr, scoreboard_crop

# TVING uses two upper-left score bugs absent from the older pitch-link OCR.
EXTRA_LAYOUTS = {
    'tving_march': (100, 197, 183, 223),
    'tving_june': (130, 165, 235, 193),
}
SOURCE_LAYOUTS = {'99f1a527c09b': 'tving_march', 'e56945a0fd80': 'tving_june',
                  '437c049ad9fc': 'mbc'}
VERSION = 'pitcher_identity_v1'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def signature(clip, source):
    stat = Path(source['source_path']).stat()
    return hashlib.sha256(json.dumps([VERSION, clip['source_id'], clip['start_time'],
        clip['end_time'], source['naver_game_id'], stat.st_size, stat.st_mtime_ns]).encode()).hexdigest()


def recover_one(dataset, clip, source, game, observation, previous, refresh):
    cid = clip['dataset_clip_id']
    sig = signature(clip, source)
    if not refresh and previous.get('signature') == sig:
        return previous
    roster = defaultdict(set)
    for pitch in game:
        roster[pitch['pitcher_name']].add(pitch['pitcher_code'])
    primary = SOURCE_LAYOUTS.get(clip['source_id'], observation.get('layout', 'spotv'))
    layouts = [primary]
    if primary in ('sbs', 'sbs_left'):
        layouts = ['sbs', 'sbs_left']
    start, end = float(clip['start_time']), float(clip['end_time'])
    samples = []
    with av.open(source['source_path']) as container:
        stream = container.streams.video[0]
        for fraction in (.22, .65, .88):
            frame, actual = frame_at(container, stream, start + (end - start) * fraction)
            if frame is None or not start <= actual < end:
                continue
            candidates = []
            for layout in layouts:
                box = EXTRA_LAYOUTS.get(layout) or LAYOUTS[layout]['pitcher']
                roi = scoreboard_crop(frame, box)
                scaled = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                raw = ocr(scaled, 'kor+eng', 7)
                name, match = name_match(raw, roster)
                if match != 'exact':
                    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                    for variant in (255-gray, cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]):
                        alternative = ocr(variant, 'kor+eng', 7)
                        alt_name, alt_match = name_match(alternative, roster)
                        if alt_match == 'exact':
                            raw, name, match = alternative, alt_name, alt_match
                            break
                # A label on SBS must be the pitcher (P), not the numbered batter.
                # The alternate layout is only accepted when its P marker is read.
                if layout != primary and layout.startswith('sbs') and not raw.lstrip().lower().startswith('p'):
                    continue
                if match == 'exact' and len(roster[name]) == 1:
                    candidates.append((name, next(iter(roster[name])), layout, raw, roi))
            names = {(item[0], item[1]) for item in candidates}
            if len(names) == 1:
                name, code, layout, raw, roi = candidates[0]
                relative = Path('naver_sports/evidence/identities') / f'{cid}_{len(samples)}.jpg'
                (dataset / relative).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dataset / relative), roi)
                samples.append({'time': round(actual, 5), 'name': name, 'code': code,
                    'layout': layout, 'raw': raw, 'image': str(relative)})
    votes = Counter((item['name'], item['code']) for item in samples)
    # All readable samples must agree; fuzzy hints are never accepted.
    resolved = len(votes) == 1 and max(votes.values()) >= 2
    name, code = next(iter(votes)) if resolved else ('', '')
    return {'clipId': cid, 'sourceId': clip['source_id'], 'gameId': clip['naver_game_id'],
        'start': start, 'end': end, 'signature': sig, 'name': name, 'code': code,
        'method': 'scoreboard_repeated_exact' if resolved else 'unresolved',
        'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=WORKSPACE / 'dataset')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    clips = rows(dataset / 'clips.csv')
    sources = {r['source_id']: r for r in rows(dataset / 'sources.csv')}
    accepted = {r['dataset_clip_id'] for r in rows(dataset / 'naver_sports/auto_accepted.csv')}
    observations = {r['dataset_clip_id']: r for r in rows(dataset / 'naver_sports/scoreboard_observations.csv')}
    games = defaultdict(list)
    for row in rows(dataset / 'naver_sports/pitches.csv'):
        games[row['naver_game_id']].append(row)
    destination = dataset / 'player_identity.json'
    old = json.loads(destination.read_text()).get('clips', {}) if destination.exists() else {}
    pending_ids = {c['dataset_clip_id'] for c in clips if c['dataset_clip_id'] not in accepted}
    records = {cid: r for cid, r in old.items() if cid in pending_ids}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(recover_one, dataset, c, sources[c['source_id']],
            games[c['naver_game_id']], observations.get(c['dataset_clip_id'], {}),
            old.get(c['dataset_clip_id'], {}), args.refresh): c for c in clips
            if c['dataset_clip_id'] not in accepted}
        for index, future in enumerate(as_completed(jobs), 1):
            c = jobs[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {'clipId': c['dataset_clip_id'], 'method': 'unresolved', 'name': '',
                          'code': '', 'error': f'{type(exc).__name__}: {exc}'}
            records[c['dataset_clip_id']] = record
            if index % 25 == 0 or index == len(jobs):
                print(f'Identity OCR {index}/{len(jobs)}; resolved {sum(bool(r["name"]) for r in records.values())}', flush=True)
                destination.write_text(json.dumps({'version': VERSION, 'clips': dict(sorted(records.items()))}, ensure_ascii=False, indent=2))
    print(Counter(r['method'] for r in records.values()))


if __name__ == '__main__':
    main()
