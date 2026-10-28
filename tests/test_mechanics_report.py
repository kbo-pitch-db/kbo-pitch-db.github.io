"""Source/event/denominator checks for the exploratory image-pattern pilot."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_mechanics_report import flexion, summarize, smooth


class MechanicsReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'data/mechanics/65320.json').read_text())
        cls.audit = json.loads((ROOT/'data/mechanics/event_audit.json').read_text())
        cls.detail = json.loads((ROOT/'data/portal/players/65320.json').read_text())
        cls.clips = {c['id']:c for c in json.loads((ROOT/'data/clips.json').read_text())['clips']}

    def test_pitches_are_unique_and_scope_is_exhaustive(self):
        r = self.report
        self.assertEqual(len(r['samples']), 6)
        self.assertEqual(len({s['pitchId'] for s in r['samples']}), 6)
        self.assertEqual({s['gameId'] for s in r['samples']}, {'20240530LGSK02024'})
        self.assertEqual({s['pitch']:sum(t['pitch']==s['pitch'] for t in r['samples']) for s in r['samples']}, {'직구':3,'슬라이더':3})
        accounted = [s['clipId'] for s in r['samples']+r['excluded']]
        self.assertEqual(len(accounted), len(set(accounted)))
        self.assertEqual(set(accounted), set(self.detail['clips']))
        self.assertIn({'clipId':'0d84991ce9f6_c0009','reason':'landing_occluded'}, r['excluded'])

    def test_original_event_frames_and_pose_hashes(self):
        for s in self.report['samples']:
            clip = self.clips[s['clipId']]
            path = ROOT/'data'/clip['pose']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), s['poseSha256'])
            pose = json.loads(path.read_text())
            self.assertEqual([pose['frames'][i]['source_frame'] for i in s['landingInterval']],s['sourceLandingInterval'])
            self.assertEqual([pose['frames'][i]['source_frame'] for i in s['releaseInterval']],s['sourceReleaseInterval'])
            self.assertLess(s['landingInterval'][1],s['releaseInterval'][0])
            self.assertEqual(s['leadSide'],'left')
            self.assertFalse(s['modelInspection']['eligibleForAggregation'])

    def test_metrics_recompute_from_original_observations(self):
        for s in self.report['samples']:
            pose=json.loads((ROOT/'data'/self.clips[s['clipId']]['pose']).read_text())
            raw=[flexion(pose['frames'][i],'R') for i in s['rawFrames']]
            for a,b in zip(raw,s['rawFlexion']): self.assertAlmostEqual(a,b,places=3)
            offset=s['rawFrames'][0]
            expected=summarize(smooth(raw,1),s['landingFrame']-offset,s['releaseFrame']-offset)
            self.assertAlmostEqual(expected['extension'],s['metrics']['extension'],places=3)
            self.assertEqual(len(s['waveform']),41)
            self.assertTrue(all(math.isfinite(v) for v in s['waveform']))

    def test_sensitivity_is_separate_and_contains_all_processing_settings(self):
        for s in self.report['samples']:
            for setting in s['settings'].values():
                low,high=s['sensitivity']['extension']
                self.assertGreaterEqual(setting['extension']+1e-4,low)
                self.assertLessEqual(setting['extension']-1e-4,high)
            self.assertGreater(s['sensitivity']['extension'][1]-s['sensitivity']['extension'][0],0)

    def test_clock_and_quality_are_not_falsely_validated(self):
        self.assertFalse(self.report['humanReviewed'])
        self.assertFalse(self.report['physicalTimeValidated'])
        self.assertFalse(self.report['modelSensitivityValidated'])
        self.assertEqual(sum(s['strictObservationEligible'] for s in self.report['samples']),1)
        for s in self.report['samples']:
            self.assertFalse(s['cadence']['playbackRateVerified'])
            self.assertGreater(s['cadence']['nearDuplicatePairs'],0)
            self.assertLess(len(s['observationFrameCandidates']),len(s['rawFrames']))
            self.assertNotIn('angularVelocity',s['metrics'])

    def test_role_mapping_confidence_and_missing_observations(self):
        frame={'keypoints':[[0,0,.9] for _ in range(26)],'interpolated':[False]*26,'outlier':[False]*26}
        for i,p in zip([11,13,15],[[1,0,.9],[0,0,.9],[-1,0,.9]]):frame['keypoints'][i]=p
        for i,p in zip([12,14,16],[[1,0,.9],[0,0,.9],[0,1,.9]]):frame['keypoints'][i]=p
        self.assertEqual(flexion(frame,'R'),0)
        self.assertEqual(flexion(frame,'L'),90)
        frame['keypoints'][13][2]=.45
        self.assertEqual(flexion(frame,'R'),0)
        self.assertIsNone(flexion(frame,'R',.5))
        frame['interpolated'][13]=True
        self.assertIsNone(flexion(frame,'R'))


if __name__=='__main__': unittest.main()
