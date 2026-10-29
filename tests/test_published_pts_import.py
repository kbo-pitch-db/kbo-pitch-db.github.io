import importlib.util
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('published_pts', ROOT/'scripts/import_published_pts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0,str(ROOT/'scripts'))
from build_biomechanics_database import refresh_records


class PublishedPtsTests(unittest.TestCase):
    def test_tracking_row_cannot_invent_relay_order_or_before_pitch_counts(self):
        row = dict(game_id='20240824HTNC02024', pitchId='240824_184301', inning='3',
                   pitch_num='2', pitcher='54640', picher_name='네일', batter='12345',
                   batter_name='타자', result='Ball', stuff='4-Seam', speed='149',
                   ball='2', strike='1', out='2', homeScore='1', awayScore='2',
                   x='0.1', crossPlateY='.7083', topSz='3.3', bottomSz='1.6')
        row.update({k:'1' for k in ['x0','y0','z0','vx0','vy0','vz0','ax','ay','az']})
        result = module.convert(row, ['pitcher_pitch_number','half_inning','balls_before',
                                     'strikes_before','outs_before','relay_seqno','at_bat_id'],
                                lambda _: (None, 'missing_tracking_fields'), 'raw.csv')
        for key in ['pitcher_pitch_number','half_inning','balls_before','strikes_before',
                    'outs_before','relay_seqno','at_bat_id']:
            self.assertEqual(result[key], '')
        self.assertEqual(result['pitch_number'], '2')
        self.assertEqual(result['naver_pitch_id'], '20240824HTNC02024:240824_184301')
        self.assertNotIn('trajectory_points_json', result)

    def test_bunt_and_swing_codes_match_existing_relay_semantics(self):
        # Cross-checked against the 9,588 overlapping relay/CSV records, including
        # 20 bunt fouls and eight bunt misses. They must not become normal fouls.
        self.assertEqual(module.RESULTS['BuntFoul'], 'W')
        self.assertEqual(module.RESULTS['BuntSwing'], 'V')
        self.assertEqual(module.RESULTS['Swing'], 'S')

    def test_same_pitcher_does_not_join_an_unidentified_pitch_to_a_clip(self):
        old=dict(id='old',pitchId='old',clipIds=[],unit='pitch_record',primaryClipId=None)
        base=dict(rows=[old],scope={})
        records=[dict(id=pid,clipIds=[],gameId='game',date='2024-08-24',pitch='직구',speed=149,sourceUrl='source') for pid in ['old','new']]
        clips=[dict(id='clip1',playerCode='1',player='투수',sourceId='video',date='2024-08-24',sourceUrl='video',pitch='',speed='')]
        details={'1':dict(records=records,clips={'clip1':dict(status='unresolved')})}
        result=refresh_records(base,clips,{'1':dict(name='투수',hand='R')},details,{'video':'game'},'expanded')
        self.assertEqual(result['rows'][0],old)
        self.assertIsNone(result['rows'][1]['primaryClipId'])
        self.assertIsNone(result['rows'][2]['pitchId'])
        self.assertEqual(result['scope']['linkedPitches'],0)
        self.assertEqual(result['scope']['recordPitches'],2)
        self.assertEqual(result['scope']['unlinkedClips'],1)
        with self.assertRaisesRegex(ValueError,'discard existing pitches'):
            refresh_records(base,clips,{}, {},{},'expanded')


if __name__ == '__main__':
    unittest.main()
