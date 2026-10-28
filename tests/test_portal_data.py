"""The portal must preserve pitch identity, denominators and pose provenance."""
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portal_builder', SITE/'scripts/build_portal_data.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PortalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((SITE/'data/portal/summary.json').read_text())
        cls.index = json.loads((SITE/'data/clips.json').read_text())
        cls.clips = {c['id']: c for c in cls.index['clips']}
        cls.details = [json.loads((SITE/'data'/p['detailPath']).read_text()) for p in cls.summary['players']]

    def test_manifest_matches_canonical_release_and_sample(self):
        s = self.summary
        self.assertEqual(s['poseVersion'], self.index['packageVersion'])
        self.assertEqual(s['clipIndexSha256'], hashlib.sha256((SITE/'data/clips.json').read_bytes()).hexdigest())
        self.assertEqual((s['clipCount'], s['playerCount'], s['gameCount']), (813, 155, 32))
        self.assertEqual(s['motionPlayerCount'], 123)
        self.assertEqual(sum(p['clipCount'] for p in s['players']), 813)
        self.assertEqual(sum(p['recordCount'] for p in s['players']), s['profileRecordCount'])
        self.assertEqual(sum(s['links'].values()), 813)

    def test_records_are_deduplicated_and_every_link_has_same_identity(self):
        seen_records, seen_clips, linked = set(), set(), 0
        for detail in self.details:
            p = detail['profile']
            records = {r['id']: r for r in detail['records']}
            self.assertEqual(len(records), p['recordCount'])
            self.assertFalse(seen_records.intersection(records))
            seen_records.update(records)
            for cid, link in detail['clips'].items():
                self.assertNotIn(cid, seen_clips)
                seen_clips.add(cid)
                clip = self.clips[cid]
                self.assertEqual(clip['playerCode'], p['code'])
                self.assertEqual(clip['player'], p['name'])
                if link['status'] == 'auto_accepted':
                    linked += 1
                    record = records[link['pitchId']]
                    self.assertEqual(link['auditDecision'], 'accepted')
                    self.assertEqual(clip['playerEvidence']['pitchId'], record['id'])
                    self.assertEqual(record['id'].split(':')[0], record['gameId'])
                    self.assertEqual(clip['pitch'], record['pitch'])
                    self.assertAlmostEqual(float(clip['speed']), record['speed'])
                    self.assertIn(cid, record['clipIds'])
                else:
                    self.assertIsNone(link['pitchId'])
                    self.assertFalse(any(cid in r['clipIds'] for r in records.values()))
                for q in link['metrics'].values():
                    self.assertGreaterEqual(q['validFrames'], 0)
                    self.assertLessEqual(q['validFrames'], q['totalFrames'])
                    self.assertEqual(q['totalFrames'], clip['frames'])
            for record in records.values():
                for cid in record['clipIds']:
                    self.assertEqual(detail['clips'][cid]['pitchId'], record['id'])
        self.assertEqual(linked, 390)
        self.assertEqual(len(seen_clips), 813)
        self.assertEqual(len(seen_records), 9754)

    def test_statistics_use_all_acquired_records_including_players_without_video(self):
        no_video = [d for d in self.details if d['profile']['clipCount'] == 0]
        self.assertEqual(len(no_video), 32)
        self.assertEqual(sum(len(d['records']) for d in no_video), 806)
        for detail in self.details:
            p = detail['profile']
            stats = builder.pitch_statistics(detail['records'])
            self.assertEqual(stats, p['statistics'])
            self.assertEqual(sum(stats['pitchCounts'].values()), p['recordCount'])
            self.assertEqual(sum(stats['byPitch'][k]['count'] for k in stats['byPitch']), p['recordCount'])
            if p['representative']:
                representative = self.clips[p['representative']['clipId']]
                self.assertEqual(representative['playerCode'], p['code'])
                self.assertGreater(p['representative']['candidateCount'], 0)
            else:
                self.assertEqual(p['clipCount'], 0)

    def test_result_denominators_exclude_bunts_and_missing_positions(self):
        records = [dict(pitch='직구', speed=140., result=result, plateX=0., plateZ=2., zoneBottom=1., zoneTop=3.)
                   for result in ['S','F','H','T','B','V','W']]
        stats = builder.pitch_statistics(records)
        self.assertEqual(stats['pitches'], 7)
        self.assertEqual(stats['swings'], 3)
        self.assertEqual(stats['ordinaryPitches'], 5)
        self.assertAlmostEqual(stats['whiffRate'], 33.333, places=3)
        self.assertEqual(stats['cswRate'], 40)
        self.assertEqual(stats['zoneRate'], 100)
        records[0]['plateX'] = None
        self.assertEqual(builder.pitch_statistics(records)['locations'], 6)

    def test_percentiles_have_explicit_eligible_cohorts_and_midrank_ties(self):
        profiles = self.summary['players']
        for metric, (denominator, minimum) in builder.BENCHMARKS.items():
            cohort = [p for p in profiles if p['recordCount'] >= 50 and p['statistics'][denominator] >= minimum
                      and p['statistics'][metric] is not None]
            self.assertEqual(len(cohort), self.summary['benchmarks'][metric]['players'])
            values = [p['statistics'][metric] for p in cohort]
            for p in profiles:
                b = p['benchmarks'][metric]
                if p not in cohort:
                    self.assertFalse(b['qualified'])
                    self.assertIsNone(b['percentile'])
                else:
                    v = p['statistics'][metric]
                    expected = round(100*(sum(x < v for x in values)+.5*sum(x == v for x in values))/len(values), 1)
                    self.assertEqual(expected, b['percentile'])

    def test_full_record_download_is_the_hashed_source(self):
        archive = gzip.decompress((SITE/'data/portal/pitch_records.csv.gz').read_bytes())
        self.assertEqual(hashlib.sha256(archive).hexdigest(), self.summary['provenance']['naver_sports/pitches.csv'])
        records = list(csv.DictReader(io.StringIO(archive.decode('utf-8-sig'))))
        self.assertEqual(len(records), 9754)
        self.assertEqual(len({r['naver_pitch_id'] for r in records}), 9754)
        self.assertEqual(len({r['naver_game_id'] for r in records}), 32)

    def test_angles_are_included_angles_with_explicit_quality_gates(self):
        frame = {'keypoints': [[1, 0, .9], [0, 0, .9], [0, 1, .9]]}
        self.assertEqual(builder.joint_angle(frame, [0, 1, 2]), 90)
        frame['keypoints'][2] = [-1, 0, .9]
        self.assertEqual(builder.joint_angle(frame, [0, 1, 2]), 180)
        frame['keypoints'][2][2] = .49
        self.assertIsNone(builder.joint_angle(frame, [0, 1, 2]))
        frame['keypoints'][2][2] = .9
        for flag in ['interpolated', 'outlier']:
            frame[flag] = [False, True, False]
            self.assertIsNone(builder.joint_angle(frame, [0, 1, 2]))
            del frame[flag]
        frame['keypoints'][1] = frame['keypoints'][0]
        self.assertIsNone(builder.joint_angle(frame, [0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
