"""Prevent silent PTS misjoins and false completion during collection failures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from datetime import date

SPEC = importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1]/'scripts/collect_naver_pts.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CollectionTests(unittest.TestCase):
    def test_native_id_join_survives_missing_and_reordered_tracking(self):
        payload = {'result': {'textRelayData': {'textRelays': [{
            'ptsOptions': [{'pitchId': 'b', 'x0': 2}, {'pitchId': 'a', 'x0': 1}, {'pitchId': 'orphan'}],
            'textOptions': [{'ptsPitchId': 'missing', 'pitchResult': 'B'},
                            {'ptsPitchId': 'a', 'pitchResult': 'T'},
                            {'ptsPitchId': 'b', 'pitchResult': 'S'}]}]}}}
        rows, unmatched = module.tracked_pitches(payload)
        self.assertEqual([(r['pitchId'], r['tracking']['x0']) for r in rows], [('a', 1), ('b', 2)])
        self.assertEqual(unmatched, 1)
        payload['result']['textRelayData']['textRelays'][0]['ptsOptions'].append({'pitchId': 'a', 'x0': 999})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            module.tracked_pitches(payload)

    def test_outage_is_not_empty_success_and_stops_after_three_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = module.Collector(directory)
            def failed(*args):
                raise TimeoutError('test outage')
            collector.fetch = failed
            report = collector.collect(date(2026, 8, 5), date(2026, 8, 10))
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['errors']), 3)
            self.assertEqual(report['days'], [])
            self.assertEqual(report['games'], [])
            self.assertFalse(json.loads((Path(directory)/'collection-report.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
