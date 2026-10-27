"""Regression checks against mislabelling a player or inventing a pitch match."""
import copy
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from player_identity import PlayerIdentities, update_index


class PlayerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = dict(dataset_clip_id='video_c1', source_id='video', naver_game_id='game', start_time='10', end_time='13')
        self.pitch = dict(naver_pitch_id='pitch1', naver_game_id='game', pitcher_name='투수', pitcher_code='12345', source_url='https://example.org/relay')
        self.record = dict(clipId='video_c1', sourceId='video', gameId='game', start=10., end=13.,
            name='투수', code='12345', method='scoreboard_repeated_exact', samples=[
                dict(time=10.5, name='투수', code='12345'), dict(time=11.5, name='투수', code='12345')])
        self.write_csv('clips.csv', [self.clip])
        self.write_csv('naver_sports/pitches.csv', [self.pitch])

    def write_csv(self, relative, records):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def resolve(self, record=None):
        (self.root / 'player_identity.json').write_text(json.dumps({'clips': {'video_c1': record or self.record}}))
        return PlayerIdentities(self.root).resolve(self.clip)

    def test_repeated_scoreboard_recovers_player_without_pitch_fields(self):
        result = self.resolve()
        self.assertEqual(result['playerCode'], '12345')
        self.assertEqual(result['playerNameSource'], 'scoreboard_repeated_exact')
        self.assertNotIn('pitchNumber', result)
        self.assertNotIn('pitch', result)

    def test_rejects_wrong_source_game_interval_or_code(self):
        for key, value in [('sourceId', 'other'), ('gameId', 'other'), ('start', 9), ('end', 14), ('code', '99999')]:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                record[key] = value
                self.assertFalse(self.resolve(record)['playerKnown'])

    def test_rejects_single_frame_duplicate_frame_conflict_and_out_of_bounds(self):
        a, b = self.record['samples']
        for samples in ([a], [a, a], [a, {**b, 'name': '다른투수'}], [a, {**b, 'time': 13.1}]):
            with self.subTest(samples=samples):
                self.assertFalse(self.resolve({**self.record, 'samples': samples})['playerKnown'])

    def test_fuzzy_hints_are_not_identities(self):
        self.assertFalse(self.resolve({**self.record, 'method': 'fuzzy_review'})['playerKnown'])

    def test_ambiguous_name_cannot_select_a_player_code(self):
        self.write_csv('naver_sports/pitches.csv', [self.pitch, {**self.pitch, 'naver_pitch_id': 'pitch2', 'pitcher_code': '99999'}])
        self.assertFalse(self.resolve()['playerKnown'])

    def test_accepted_link_must_belong_to_the_game(self):
        self.write_csv('naver_sports/auto_accepted.csv', [dict(dataset_clip_id='video_c1', naver_pitch_id='pitch1', pitcher_name='투수')])
        self.assertEqual(self.resolve()['playerNameSource'], 'naver_auto_accepted')
        self.write_csv('naver_sports/pitches.csv', [{**self.pitch, 'naver_game_id': 'other'}])
        self.assertFalse(self.resolve()['playerKnown'])

    def review(self, **fields):
        record = {**self.record, 'method': 'broadcast_text_review',
                  'reviewKind': 'scoreboard_name', 'reviewNote': 'Read printed text', **fields}
        (self.root / 'player_identity_review.json').write_text(json.dumps({'clips': {'video_c1': record}}))
        return PlayerIdentities(self.root).resolve(self.clip)

    def test_review_requires_an_in_clip_evidence_frame(self):
        self.assertTrue(self.review()['playerKnown'])
        self.assertFalse(self.review(samples=[])['playerKnown'])
        self.assertFalse(self.review(samples=[dict(time=14)])['playerKnown'])

    def test_uniform_review_requires_the_correct_game_side_and_number(self):
        path = self.root / 'lineup.json.gz'
        with gzip.open(path, 'wt') as handle:
            json.dump({'result': {'textRelayData': {'gameId': 'game',
                'homeLineup': {'pitcher': [dict(name='투수', pcode='12345', backnum='27')]},
                'awayLineup': {'pitcher': [dict(name='다른투수', pcode='99999', backnum='27')]}}}}, handle)
        fields = dict(reviewKind='uniform_text', uniformNumber='27', rosterSide='home', rosterFile='lineup.json.gz')
        self.assertTrue(self.review(**fields)['playerKnown'])
        self.assertFalse(self.review(**{**fields, 'rosterSide': 'away'})['playerKnown'])
        self.assertFalse(self.review(**{**fields, 'uniformNumber': '28'})['playerKnown'])
        self.assertFalse(self.review(**{**fields, 'rosterFile': 'missing.json.gz'})['playerKnown'])

    def test_scoreboard_state_requires_one_player_identity(self):
        pitch = {**self.pitch, 'inning': '10', 'half_inning': 'bottom', 'pitcher_pitch_number': '8'}
        self.write_csv('naver_sports/pitches.csv', [pitch])
        fields = dict(reviewKind='scoreboard_state', observedState=dict(inning=10, half='bottom', completedP=7))
        self.assertTrue(self.review(**fields)['playerKnown'])
        self.write_csv('naver_sports/pitches.csv', [pitch, {**pitch, 'naver_pitch_id': 'pitch2', 'pitcher_name': '다른투수', 'pitcher_code': '99999'}])
        self.assertFalse(self.review(**fields)['playerKnown'])

    def test_metadata_update_preserves_existing_motion_and_pitch_data(self):
        self.resolve()
        output = self.root / 'site'
        output.mkdir()
        portrait = output / 'portraits/12345.jpg'
        portrait.parent.mkdir()
        portrait.write_bytes(b'image')
        (output / 'player_profiles.json').write_text(json.dumps({'players': {'12345': {'name': '투수', 'imagePath': 'portraits/12345.jpg'}}}))
        original = dict(id='video_c1', pose3d='assets/final.json', sceneSignature='untouched', pitch='', pitchNumber='', model3d='WHAM', mediaPolicy='pose-only')
        (output / 'clips.json').write_text(json.dumps({'clips': [original], 'custom': 'preserved'}))
        update_index(self.root, output)
        result = json.loads((output / 'clips.json').read_text())
        self.assertEqual(result['custom'], 'preserved')
        for key, value in original.items():
            self.assertEqual(result['clips'][0][key], value)
        self.assertEqual(result['clips'][0]['player'], '투수')


if __name__ == '__main__':
    unittest.main()
