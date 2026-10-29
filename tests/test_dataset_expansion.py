"""Protect prior observations and avoid inventing identities for published PTS."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'data/releases/combined-20261029-increment-v1'


class DatasetExpansionTests(unittest.TestCase):
    def test_original_clip_entries_and_every_active_pose_byte_are_preserved(self):
        previous=json.load(gzip.open(RELEASE/'previous_index.json.gz','rt'))
        current=json.loads((ROOT/'data/clips.json').read_text())
        by_id={c['id']:c for c in current['clips']}
        for clip in previous['clips']:
            self.assertEqual(by_id[clip['id']],clip)
        hashes=json.load(gzip.open(RELEASE/'baseline_pose_hashes.json.gz','rt'))
        self.assertEqual(len(hashes),3*len(previous['clips']))
        for relative,expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT/'data'/relative).read_bytes()).hexdigest(),expected,relative)
        added=json.loads((RELEASE/'clips.json').read_text())['clips']
        self.assertFalse({c['id'] for c in previous['clips']}&{c['id'] for c in added})
        self.assertEqual(len(current['clips']),len(previous['clips'])+len(added))

    def test_published_tracking_rows_do_not_gain_missing_counts_or_pitch_links(self):
        records=[r for p in (ROOT/'data/portal/players').glob('*.json')
                 for r in json.loads(p.read_text())['records']
                 if r.get('recordSource')=='published_naver_pts_csv']
        provenance=json.loads((RELEASE/'record_provenance.json').read_text())
        self.assertEqual(len(records),provenance['recordCount'])
        self.assertEqual(len({r['id'] for r in records}),len(records))
        self.assertEqual({r['gameId'] for r in records},set(provenance['games'].values()))
        for r in records:
            for field in ['sequence','balls','strikes','outs']:
                self.assertIsNone(r[field])
            self.assertIn(r['half'],['',None])
            self.assertEqual(r['clipIds'],[])


if __name__=='__main__':unittest.main()
