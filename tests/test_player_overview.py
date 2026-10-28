"""The public summary must never promote candidate, duplicate or single-pitch values."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_overview', ROOT / 'scripts/build_player_overview.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def row(identity, value, status='provisional_3d', pitch='직구'):
    return dict(id=identity, pitchId=identity, primaryClipId=identity + '_clip', pitch=pitch,
                date='2024-01-01', metrics={'metric': dict(value=value, status=status)})


class PlayerOverviewTests(unittest.TestCase):
    def test_no_single_pitch_player_statistic(self):
        summary = build.metric_summary([row('one', 40)], 'metric')
        self.assertEqual(summary['n'], 1)
        self.assertEqual(summary['status'], 'single')
        self.assertIsNone(summary['value'])

    def test_duplicates_candidates_and_unstable_values_do_not_enter_headline(self):
        rows = [row('one', 10), row('two', 20), row('three', 30), row('one', 10),
                row('candidate', 5000, 'event_candidate'), row('sensitive', 1000, 'sensitive_3d'),
                row('missing', None), row('nan', float('nan'))]
        summary = build.metric_summary(rows, 'metric')
        self.assertEqual(summary['n'], 3)
        self.assertEqual(summary['value'], 20)
        self.assertEqual(summary['observedN'], 4)
        self.assertEqual(len({s['id'] for s in summary['samples']}), 4)

    def test_two_stable_pitches_are_insufficient_not_unstable(self):
        summary = build.metric_summary([row('one', 10), row('two', 20)], 'metric')
        self.assertIsNone(summary['value'])
        self.assertEqual(summary['status'], 'insufficient')

    def test_sensitive_only_withheld(self):
        summary = build.metric_summary([row('one', 10, 'sensitive_3d'), row('two', 90, 'sensitive_3d')], 'metric')
        self.assertIsNone(summary['value'])
        self.assertEqual(summary['status'], 'sensitive')
        self.assertEqual(summary['n'], 0)

    def test_release_coverage_and_actual_representatives(self):
        overview = json.loads((ROOT / 'data/portal/player_overview.json').read_text())
        clips = json.loads((ROOT / 'data/clips.json').read_text())
        portal = json.loads((ROOT / 'data/portal/summary.json').read_text())
        self.assertEqual(overview['poseVersion'], clips['packageVersion'])
        self.assertEqual(set(overview['players']), {p['code'] for p in portal['players']})
        by_id = {c['id']: c for c in clips['clips']}
        for code, player in overview['players'].items():
            if player['representative']:
                self.assertEqual(by_id[player['representative']]['playerCode'], code)
            for pitch, clip_id in player['byPitch'].items():
                self.assertEqual(by_id[clip_id]['pitch'], pitch)
                self.assertEqual(by_id[clip_id]['playerCode'], code)
            for metric in player['metrics'].values():
                if metric['value'] is not None:
                    self.assertGreaterEqual(metric['n'], 3)
                self.assertFalse(any(s['status'] == 'event_candidate' for s in metric['samples']))


if __name__ == '__main__':
    unittest.main()
