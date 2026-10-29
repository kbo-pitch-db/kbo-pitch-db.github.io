"""Checkpoint public Naver schedules and inning PTS replies for a date range.

Uses /game-polling, the route used by the original KBO collection code.
Successful replies are cached as JSON, never pickle. A failed request is not
treated as an empty day or a completed game. This does not change published data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://api-gw.sports.naver.com'
TEAMS = {'HH', 'HT', 'KT', 'LG', 'LT', 'NC', 'OB', 'SK', 'SS', 'WO'}


def tracked_pitches(payload):
    """Join text and tracking by native pitch ID, never array position."""
    relays = payload['result']['textRelayData']['textRelays']
    if not isinstance(relays, list):
        raise ValueError('textRelays is not a list')
    seen = {}
    unmatched = 0
    for block in relays:
        tracks = {}
        for track in block.get('ptsOptions') or []:
            key = str(track.get('pitchId', ''))
            if key in ('', '-1', '0'):
                continue
            if key in tracks and tracks[key] != track:
                raise ValueError('Conflicting tracking for pitch '+key)
            tracks[key] = track
        used = set()
        for event in block.get('textOptions') or []:
            key = str(event.get('ptsPitchId', ''))
            if key not in tracks or not event.get('pitchResult'):
                continue
            row = {'pitchId': key, 'tracking': tracks[key], 'event': event,
                   'inning': block.get('inn'), 'homeOrAway': block.get('homeOrAway')}
            if key in seen and seen[key] != row:
                raise ValueError('Conflicting events for pitch '+key)
            seen[key] = row
            used.add(key)
        unmatched += len(tracks.keys() - used)
    return list(seen.values()), unmatched


class Collector:
    def __init__(self, root, timeout=12, attempts=2):
        self.root, self.timeout, self.attempts = Path(root), timeout, attempts

    def fetch(self, path, params, cache):
        cache = self.root / cache
        if cache.exists():
            with gzip.open(cache, 'rt', encoding='utf-8') as stream:
                return json.load(stream)
        url = BASE + path + ('?' + urlencode(params) if params else '')
        for attempt in range(self.attempts):
            try:
                request = Request(url, headers={'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json', 'Referer': 'https://m.sports.naver.com/'})
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                payload = json.loads(raw)
                if payload.get('success') is False or not isinstance(payload.get('result'), dict):
                    raise ValueError('API reply has no usable result')
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(cache.name+'.tmp')
                temporary.write_bytes(gzip.compress(raw, mtime=0))
                temporary.replace(cache)
                return payload
            except Exception:
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(1)

    def game(self, game_id):
        detail = self.fetch('/schedule/games/'+game_id, {}, f'games/{game_id}/game.json.gz')
        game = detail['result']['game']
        if '퓨처스' in str(game.get('title', '')) or game.get('statusInfo') == '경기취소':
            return {'gameId': game_id, 'status': 'excluded'}
        # Only archive finished games; an in-progress response must not become
        # a permanently cached, apparently complete record.
        if '종료' not in str(game.get('statusInfo', '')) and game.get('statusCode') != 'RESULT':
            (self.root/f'games/{game_id}/game.json.gz').unlink(missing_ok=True)
            raise ValueError('Game is not confirmed finished: '+game_id)
        match = re.search(r'\d+', str(game.get('currentInning', '')))
        if not match:
            raise ValueError('Missing final inning: '+game_id)
        last = int(match.group())
        if not 1 <= last <= 30:
            raise ValueError('Invalid final inning')
        def inning(number):
            reply = self.fetch('/schedule/games/'+game_id+'/game-polling',
                {'inning': number}, f'games/{game_id}/inning-{number:02d}.json.gz')
            rows, unmatched = tracked_pitches(reply)
            for row in rows:
                row['gameId'] = game_id
            return rows, unmatched
        with ThreadPoolExecutor(max_workers=3) as pool:
            parts = list(pool.map(inning, range(1, last+1)))
        rows = [row for part, _ in parts for row in part]
        keys = [row['pitchId'] for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Pitch repeated across innings: '+game_id)
        if not rows:
            raise ValueError('Finished game has no tracked pitches: '+game_id)
        raw = ('\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in rows)+'\n').encode()
        target = self.root/f'games/{game_id}/pitches.jsonl.gz'
        target.write_bytes(gzip.compress(raw, mtime=0))
        return {'gameId': game_id, 'status': 'complete', 'innings': last, 'pitches': len(rows),
                'unmatchedTracking': sum(n for _, n in parts),
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

    def collect(self, start, end):
        report = {'startedAt': datetime.now(timezone.utc).isoformat(), 'from': str(start),
                  'to': str(end), 'complete': False, 'days': [], 'games': [], 'errors': []}
        current, failures = start, 0
        while current <= end:
            try:
                payload = self.fetch('/schedule/games', {'fields': 'basic',
                    'upperCategoryId': 'kbaseball', 'fromDate': str(current), 'toDate': str(current)},
                    f'schedules/{current}.json.gz')
                games = payload['result']['games']
                if not isinstance(games, list):
                    raise ValueError('Schedule games is not a list')
                for game in games:
                    key = game['gameId']
                    if not re.fullmatch(r'\d{8}[A-Z]{4}\d{5}', key) or key[8:10] not in TEAMS or key[10:12] not in TEAMS:
                        continue
                    if key[:8] != current.strftime('%Y%m%d'):
                        raise ValueError('Schedule date mismatch: '+key)
                    if game.get('statusInfo') == '경기취소':
                        continue
                    result = self.game(key)
                    report['games'].append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                report['days'].append(str(current))
                failures = 0
            except Exception as exc:
                failures += 1
                report['errors'].append({'date': str(current), 'error': type(exc).__name__, 'detail': str(exc)})
                print(json.dumps(report['errors'][-1], ensure_ascii=False), flush=True)
                if failures >= 3:
                    report['stoppedReason'] = 'Three consecutive dates failed; rerun to resume cached replies.'
                    break
            current += timedelta(days=1)
        report['complete'] = not report['errors'] and current > end
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'collection-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-date', required=True, type=date.fromisoformat)
    parser.add_argument('--to-date', required=True, type=date.fromisoformat)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=12)
    parser.add_argument('--attempts', type=int, default=2)
    args = parser.parse_args()
    today_kst = datetime.now(timezone(timedelta(hours=9))).date()
    if args.to_date < args.from_date or args.to_date > today_kst or args.timeout <= 0 or args.attempts < 1:
        parser.error('Use a valid past date range, positive timeout and at least one attempt.')
    report = Collector(args.output, args.timeout, args.attempts).collect(args.from_date, args.to_date)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
