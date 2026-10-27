"""Shared, conservative player metadata used by builds and metadata-only updates."""
import argparse
import copy
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class PlayerIdentities:
    def __init__(self, dataset):
        self.dataset = Path(dataset)
        self.clips = {r['dataset_clip_id']: r for r in rows(self.dataset / 'clips.csv')}
        self.sources = {r['source_id']: r for r in rows(self.dataset / 'sources.csv')}
        self.accepted = {r['dataset_clip_id']: r for r in rows(self.dataset / 'naver_sports/auto_accepted.csv')}
        self.pitches = {r['naver_pitch_id']: r for r in rows(self.dataset / 'naver_sports/pitches.csv')}
        self.roster = defaultdict(set)
        for pitch in self.pitches.values():
            self.roster[(pitch['naver_game_id'], pitch['pitcher_name'])].add(pitch['pitcher_code'])
        portable = Path(__file__).resolve().parents[1] / 'data/player_identity_evidence.json'
        evidence = json.loads(portable.read_text()).get('clips', {}) if portable.exists() else {}
        path = self.dataset / 'player_identity.json'
        self.recovered = dict(evidence)
        self.recovered.update(json.loads(path.read_text()).get('clips', {}) if path.exists() else {})
        path = self.dataset / 'player_identity_review.json'
        self.reviewed = {cid: r for cid, r in evidence.items() if r.get('method') == 'broadcast_text_review'}
        self.reviewed.update(json.loads(path.read_text()).get('clips', {}) if path.exists() else {})

    def resolve(self, clip):
        cid = clip.get('dataset_clip_id', clip.get('id'))
        source = self.clips.get(cid, clip)
        unknown = {'player': 'Unknown pitcher', 'playerKnown': False,
            'playerKey': 'unknown:' + cid, 'playerCode': '', 'playerNameSource': 'unknown',
            'playerEvidence': {}}
        accepted = self.accepted.get(cid, {})
        pitch = self.pitches.get(accepted.get('naver_pitch_id'), {})
        if pitch and pitch['naver_game_id'] == source.get('naver_game_id') and pitch['pitcher_name'] == accepted.get('pitcher_name'):
            return self.metadata(pitch['pitcher_name'], pitch['pitcher_code'], 'naver_auto_accepted',
                {'pitchId': pitch['naver_pitch_id'], 'sourceUrl': pitch['source_url']})
        for records, method in ((self.reviewed, 'broadcast_text_review'),
                                (self.recovered, 'scoreboard_repeated_exact')):
            record = records.get(cid, {})
            name, code = record.get('name', ''), record.get('code', '')
            if not name or not code or record.get('method') != method:
                continue
            # Source and interval binding prevent a reused clip ID carrying a stale identity.
            if record.get('sourceId') != source.get('source_id') or record.get('gameId') != source.get('naver_game_id'):
                continue
            if any(float(record.get(field, -1)) != float(source.get(csv_field, -2))
                   for field, csv_field in (('start', 'start_time'), ('end', 'end_time'))):
                continue
            if self.roster.get((source['naver_game_id'], name)) != {code}:
                continue
            samples = record.get('samples', [])
            if method == 'scoreboard_repeated_exact':
                agreeing = [s for s in samples if s.get('name') == name and s.get('code') == code
                    and record['start'] <= s.get('time', -1) < record['end']]
                if len(agreeing) < 2 or len(agreeing) != len(samples):
                    continue
                if max(s['time'] for s in agreeing) - min(s['time'] for s in agreeing) < .2:
                    continue
            else:
                try:
                    if not self.valid_review(record, samples):
                        continue
                except (OSError, KeyError, ValueError, TypeError):
                    continue
            return self.metadata(name, code, method, {'gameId': record['gameId'],
                'frameTimes': [s['time'] for s in samples],
                **({'reviewKind': record['reviewKind'], 'reviewer': 'Codex visual text review',
                    'reviewNote': record['reviewNote']} if method == 'broadcast_text_review' else {}),
                'evidenceFile': 'player_identity_evidence.json'})
        return unknown

    def valid_review(self, record, samples):
        if not samples or not record.get('reviewNote'):
            return False
        if any(not record['start'] <= s.get('time', -1) < record['end'] for s in samples):
            return False
        kind = record.get('reviewKind')
        if kind == 'scoreboard_name':
            return True
        if kind == 'scoreboard_state':
            state = record['observedState']
            identities = {(p['pitcher_name'], p['pitcher_code']) for p in self.pitches.values()
                if p['naver_game_id'] == record['gameId'] and p['inning'] == str(state['inning'])
                and p['half_inning'] == state['half']
                and int(p['pitcher_pitch_number']) == state['completedP'] + 1}
            return identities == {(record['name'], record['code'])}
        if kind == 'uniform_text':
            path = self.dataset / record['rosterFile']
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                relay = json.load(handle)['result']['textRelayData']
            if relay['gameId'] != record['gameId']:
                return False
            players = relay[record['rosterSide'] + 'Lineup']['pitcher']
            identities = {(p['name'], p['pcode']) for p in players
                          if str(p['backnum']) == record['uniformNumber']}
            return identities == {(record['name'], record['code'])}
        return False

    @staticmethod
    def metadata(name, code, method, evidence):
        return {'player': name, 'playerKnown': True, 'playerKey': 'naver:' + code,
                'playerCode': code, 'playerNameSource': method, 'playerEvidence': evidence}

    def export_evidence(self, output):
        records = {}
        for cid, clip in self.clips.items():
            identity = self.resolve(clip)
            method = identity['playerNameSource']
            if method not in ('broadcast_text_review', 'scoreboard_repeated_exact'):
                continue
            record = copy.deepcopy((self.reviewed if method == 'broadcast_text_review' else self.recovered)[cid])
            record.pop('signature', None)
            video_id = self.sources.get(clip['source_id'], {}).get('provider_video_id', '')
            for sample in record['samples']:
                sample.pop('image', None)
                sample['sourceUrl'] = f'https://www.youtube.com/watch?v={video_id}&t={int(sample["time"])}s' if video_id else ''
            if record.get('rosterFile'):
                record['rosterSourceUrl'] = f'https://api-gw.sports.naver.com/schedule/games/{record["gameId"]}/relay?inning=1'
            records[cid] = record
        manifest = {'schemaVersion': 1,
            'description': 'Supplemental player identity evidence, independent of pitch linking. '
                'Visual text reviews were performed by Codex, not human ground-truth annotation. '
                'Screenshots remain in the local dataset; source video links and times are provided here.',
            'clips': dict(sorted(records.items()))}
        (output / 'player_identity_evidence.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def update_index(dataset, output):
    identities = PlayerIdentities(dataset)
    path = output / 'clips.json'
    index = json.loads(path.read_text())
    for clip in index['clips']:
        clip.update(identities.resolve(clip))
    index['packageVersion'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    profiles = json.loads((output / 'player_profiles.json').read_text())['players']
    unknown = [c for c in index['clips'] if not c['playerKnown']]
    missing = sorted({c['playerCode'] for c in index['clips'] if c['playerKnown'] and
        (c['playerCode'] not in profiles or profiles[c['playerCode']]['name'] != c['player'] or
         not profiles[c['playerCode']].get('imagePath') or not (output / profiles[c['playerCode']]['imagePath']).is_file())})
    summary = {'generatedAt': datetime.now(timezone.utc).isoformat(), 'clips': len(index['clips']),
        'identifiedClips': len(index['clips']) - len(unknown), 'unidentifiedClips': len(unknown),
        'players': len({c['playerCode'] for c in index['clips'] if c['playerKnown']}),
        'identityMethods': dict(Counter(c['playerNameSource'] for c in index['clips'])),
        'reviewKinds': dict(Counter(c['playerEvidence']['reviewKind'] for c in index['clips']
            if c['playerNameSource'] == 'broadcast_text_review')),
        'missingPortraitCodes': missing,
        'unidentified': [{'id': c['id'], 'sourceUrl': c.get('sourceUrl', ''),
            'date': c.get('date'), 'matchup': c.get('matchup')} for c in unknown]}
    (output / 'player_identity_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    identities.export_evidence(output)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'unidentified'}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit('Player portraits are incomplete; run sync_player_profiles.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update only player metadata; preserve motion assets and pitch fields.')
    site = Path(__file__).resolve().parents[1]
    parser.add_argument('--dataset', type=Path, default=site.parent / 'dataset')
    parser.add_argument('--output', type=Path, default=site / 'data')
    args = parser.parse_args()
    update_index(args.dataset, args.output)
