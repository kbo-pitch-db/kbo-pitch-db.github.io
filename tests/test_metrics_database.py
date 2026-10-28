"""Regression checks for immutable v1 (retained as quality-only legacy data)."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_metrics_database as builder


class MetricsDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(gzip.decompress((ROOT/'data/releases/metrics-20261028-v1/database.json.gz').read_bytes()))
        cls.clips={c['id']:c for c in json.loads((ROOT/'data/clips.json').read_text())['clips']}
        cls.provenance=json.loads(gzip.decompress((ROOT/'data/releases/metrics-20261028-v1/clip_provenance.json.gz').read_bytes()))['clips']

    def test_record_and_motion_units_are_exhaustive_and_separate(self):
        rows=self.data['rows'];records=[r for r in rows if r['unit']=='pitch_record'];unlinked=[r for r in rows if r['unit']=='unlinked_clip']
        self.assertEqual(len(rows),10177);self.assertEqual(len(records),9754);self.assertEqual(len(unlinked),423)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(sum(bool(r['primaryClipId']) for r in records),390)
        all_clips=[cid for r in rows for cid in r['clipIds']]
        self.assertEqual(len(all_clips),len(set(all_clips)));self.assertEqual(set(all_clips),set(self.clips))
        for r in unlinked:
            self.assertIsNone(r['pitchId']);self.assertIsNone(r['recordSourceUrl']);self.assertTrue(r['gameId'])
            self.assertNotEqual(r['linkStatus'],'auto_accepted')

    def test_dictionary_definitions_and_versions(self):
        metrics=self.data['dictionary'];self.assertEqual(len(metrics),24)
        self.assertEqual(len({m['id'] for m in metrics}),24)
        for m in metrics:
            for key in ['unit','coordinateSystem','referenceEvent','requiredJoints','calculationVersion','conditions','formula']:self.assertTrue(m[key],(m['id'],key))
            self.assertTrue(m['name']['ko']);self.assertTrue(m['name']['en'])
            self.assertTrue(m['definition']['ko']);self.assertTrue(m['definition']['en'])
        self.assertIn('observed_clip',{m['referenceEvent'] for m in metrics})
        self.assertIn('provisional_landing_release',{m['referenceEvent'] for m in metrics})

    def test_every_cell_has_a_finite_value_or_an_explicit_reason(self):
        no_motion=0
        for row in self.data['rows']:
            if row['metrics'] is None:
                self.assertIsNone(row['primaryClipId']);no_motion+=1;continue
            self.assertEqual(set(row['metrics']),{m['id'] for m in self.data['dictionary']})
            for key,cell in row['metrics'].items():
                self.assertIn(cell['status'],self.data['statusCodes'])
                self.assertLessEqual(cell['validFrames'],cell['totalFrames'])
                if cell['value'] is None:self.assertNotIn(cell['status'],['observed_2d','needs_review_3d','provisional_event'])
                else:
                    self.assertTrue(math.isfinite(cell['value']))
                    self.assertIn(cell['status'],['observed_2d','needs_review_3d','provisional_event'])
        self.assertEqual(no_motion,9364)
        for counts in self.data['statusCounts'].values():self.assertEqual(sum(counts.values()),len(self.data['rows']))

    def test_unverified_dependencies_remain_null_and_3d_needs_review(self):
        for row in self.data['rows']:
            if row['metrics'] is None:continue
            for key in ['landing_release_time_ms','pelvis_trunk_peak_gap_ms']:
                self.assertIsNone(row['metrics'][key]['value']);self.assertEqual(row['metrics'][key]['status'],'physical_time_unverified')
            for key in ['trunk_forward_lean_release','trunk_lateral_lean_release','normalized_stride_landing']:
                self.assertIsNone(row['metrics'][key]['value']);self.assertEqual(row['metrics'][key]['status'],'pitch_direction_unverified')
            for key in ['separation_p90_3d','pelvis_rotation_range_3d','ankle_spread_p90_3d']:
                if row['metrics'][key]['value'] is not None:self.assertEqual(row['metrics'][key]['status'],'needs_review_3d')
        self.assertFalse(self.data['defaults']['include3dInAggregates'])

    def test_event_statistics_use_only_audited_source_pitches(self):
        pilot={s['clipId']:s for s in json.loads((ROOT/'data/mechanics/65320.json').read_text())['samples']}
        provided=[r for r in self.data['rows'] if r['metrics'] and r['metrics']['knee_extension_2d']['value'] is not None]
        self.assertEqual(len(provided),6)
        for row in provided:
            self.assertIn(row['primaryClipId'],pilot)
            self.assertEqual(row['metrics']['knee_extension_2d']['value'],pilot[row['primaryClipId']]['metrics']['extension'])
            self.assertEqual(row['metrics']['knee_extension_2d']['status'],'provisional_event')

    def test_recompute_raw_2d_metrics_and_hashes(self):
        for row in [r for r in self.data['rows'] if r['metrics']][::113]:
            cid=row['primaryClipId'];path=ROOT/'data'/self.clips[cid]['pose'];pose=json.loads(path.read_text())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.provenance[cid]['poseSha256'])
            ids=[11,13,15] if row['hand']=='R' else [12,14,16]
            values=[builder.bend(*(frame['keypoints'][i] for i in ids),2) if builder.good2(frame,ids) else None for frame in pose['frames']]
            expected=builder.summary_cell(values,len(values),'range','2d')
            self.assertEqual(row['metrics']['lead_knee_range_2d'],expected)

    def test_per_metric_failure_does_not_drop_other_values(self):
        row=next(r for r in self.data['rows'] if r['metrics'] and r['metrics']['throw_elbow_range_2d']['value'] is None)
        self.assertIsNotNone(row['metrics']['lead_knee_range_2d']['value'])
        self.assertEqual(row['metrics']['throw_elbow_range_2d']['status'],'insufficient_observations')
        self.assertEqual(builder.summary_cell([0.]*20,20,'p90','2d')['value'],0.)
        self.assertIsNone(builder.summary_cell([None]*20,20,'p90','2d')['value'])
        self.assertEqual(builder.unwrap_longest([179,-179,None,10,20,30]),[None,None,None,10,20,30])
        self.assertEqual(builder.unwrap_longest([179,-179,-178]),[179,181,182])

    def test_model_alignment_failure_leaves_observable_2d(self):
        with tempfile.TemporaryDirectory() as temp:
            old=builder.DATA;builder.DATA=Path(temp)
            try:
                points=[[float(i),float(i%4),.9] for i in range(26)]
                frames=[dict(keypoints=points,source_frame=i,source_time=i/30) for i in range(20)]
                (builder.DATA/'pose.json').write_text(json.dumps(dict(frames=frames,fps=30)))
                (builder.DATA/'model.json').write_text(json.dumps(dict(frames=[dict(source_frame=-1,source_time=i/30) for i in range(20)],fps=30)))
                clip=dict(id='synthetic',pose='pose.json',pose3d='model.json',sourceId='test',reconstructionVersion='test',sourceUrl='https://example.invalid')
                result=builder.clip_metrics(clip,'R',builder.dictionary(),{})['metrics']
                self.assertIsNotNone(result['lead_knee_range_2d']['value'])
                self.assertEqual(result['separation_p90_3d']['status'],'model_alignment_mismatch')
            finally:builder.DATA=old

    def test_full_csv_preserves_missing_and_public_compressed_payload(self):
        plain=gzip.decompress((ROOT/'data/releases/metrics-20261028-v1/database.json.gz').read_bytes())
        self.assertEqual(gzip.decompress((ROOT/'data/releases/metrics-20261028-v1/database.json.gz').read_bytes()),plain)
        with gzip.open(ROOT/'data/releases/metrics-20261028-v1/pitches.csv.gz','rt',encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),10177)
        missing=next(r for r in rows if r['pitch_link_status']=='no_motion')
        self.assertEqual(missing['lead_knee_range_2d__value'],'')
        self.assertEqual(missing['lead_knee_range_2d__status'],'no_motion')
        self.assertEqual(missing['lead_knee_range_2d__unit'],'°')
        self.assertTrue(missing['lead_knee_range_2d__calculation_version'])


if __name__=='__main__':unittest.main()
