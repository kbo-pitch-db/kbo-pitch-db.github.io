#!/usr/bin/env python3
"""Verify KBO identity pages and cache season portraits for the dataset roster."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import cv2
import numpy as np

SITE = Path(__file__).resolve().parents[1]
CDN = 'https://6ptotvmi5753.edge.naverncp.com/KBO_IMAGE/person/middle'
# Keep the name recorded in the source season. KBO's stable player IDs now show
# these players' later names; this allowlist cannot accept arbitrary mismatches.
RENAMED = {'68419': ('최채흥', '최지명'), '76430': ('김상수', '김태혁')}


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'KBO-Pitch-Lab/1.0'}), timeout=25) as response:
        return response.read()


def sync_one(code, name, year, output):
    source = f'https://www.koreabaseball.com/Record/Player/PitcherDetail/Basic.aspx?playerId={code}'
    page = fetch(source).decode('utf-8-sig')
    found = re.search(r'id="[^"]*_lblName">([^<]+)</span>', page)
    official_name = html.unescape(found.group(1)).strip() if found else ''
    if re.sub(r'\s+', '', official_name) != re.sub(r'\s+', '', name) and RENAMED.get(code) != (name, official_name):
        raise ValueError(f'KBO name mismatch: expected {name}, found {official_name!r}')
    image_url = f'{CDN}/{year}/{code}.jpg'
    blob = fetch(image_url)
    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[0] < 50 or decoded.shape[1] < 40:
        raise ValueError('Portrait is not a decodable player image')
    relative = Path('portraits') / str(year) / f'{code}.jpg'
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return {'name': name, 'currentName': official_name, 'imageUrl': image_url, 'imagePath': str(relative),
        'sourceUrl': source, 'source': 'KBO', 'imageYear': str(year),
        'width': int(decoded.shape[1]), 'height': int(decoded.shape[0]),
        'sha256': hashlib.sha256(blob).hexdigest(),
        'verifiedAt': datetime.now(timezone.utc).date().isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=SITE.parent / 'dataset')
    parser.add_argument('--output', type=Path, default=SITE / 'data')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with (args.dataset / 'naver_sports/pitches.csv').open(encoding='utf-8-sig', newline='') as handle:
        roster = {(r['pitcher_code'], r['pitcher_name'], r['season']) for r in csv.DictReader(handle)}
    players, errors = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(sync_one, code, name, year, args.output): (code, name)
                for code, name, year in sorted(roster)}
        for future in as_completed(jobs):
            code, name = jobs[future]
            try:
                players[code] = future.result()
            except Exception as exc:
                errors[code] = {'name': name, 'error': f'{type(exc).__name__}: {exc}'}
            if (len(players) + len(errors)) % 25 == 0:
                print(f'KBO profiles: {len(players)} verified, {len(errors)} errors', flush=True)
    destination = args.output / 'player_profiles.json'
    old = json.loads(destination.read_text()) if destination.exists() else {}
    # A failed refresh must not discard previously verified portraits.
    merged = dict(old.get('players', {}))
    merged.update(players)
    destination.write_text(json.dumps({'schemaVersion': 2, 'source': 'KBO official player profiles',
        'players': dict(sorted(merged.items())), 'refreshErrors': errors}, ensure_ascii=False, indent=2))
    print(f'Verified {len(players)} portraits; errors: {json.dumps(errors, ensure_ascii=False)}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
