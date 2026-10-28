"""A missing landing event must not disable an observable supporting toe."""
import sys
import unittest
import gzip
import json
import math
import subprocess
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/pitch_observation'))
from independent_support import contact_state


def observation():
    n = 40
    p = np.zeros((n, 26, 2))
    p[:, :, 0] = 500
    p[:, :, 1] = 400
    p[:, 20, 1] = 200  # lead foot paused in the air
    p[:, 21, 1] = 900  # rear foot supports the pitcher
    h = dict(keypoints=p, scores=np.ones((n, 26)), missing=np.zeros((n, 26), bool),
             bbox=np.tile([0, 0, 1000, 1000], (n, 1)), fps=np.array(30.),
             source_time=np.arange(n) / 30, frame_index=np.arange(n))
    phase = dict(lead_side='left', plant_interval=None, phase_side_conflict=False,
                 events=dict(plant_interval=None, contact_confidence=0., lift_interval=[4, 8]))
    return h, phase


class IndependentSupportTests(unittest.TestCase):
    def test_unknown_plant_preserves_rear_support_and_rejects_airborne_pause(self):
        h, phase = observation()
        weights, _, notes = contact_state(h, phase, None)
        self.assertTrue(notes)
        self.assertTrue(all(e['marker'] == 1 for e in notes))
        self.assertGreater(weights.sum(), 0)

    def test_confidence_gap_releases_contact_without_inventing_observations(self):
        h, phase = observation()
        h['missing'][16:24, 21] = True
        weights, _, notes = contact_state(h, phase, None)
        self.assertEqual(weights[16:24].sum(), 0)
        self.assertFalse(any(e['start'] < 16 and e['end'] >= 24 for e in notes))

    def test_time_discontinuity_splits_the_same_stationary_marker(self):
        h, phase = observation()
        h['source_time'][20:] += .5
        _, _, notes = contact_state(h, phase, None)
        self.assertFalse(any(e['start'] < 20 <= e['end'] for e in notes))
        self.assertEqual(len(notes), 2)

    def test_published_repair_has_stationary_toes_and_fixed_native_bones(self):
        site = Path(__file__).resolve().parents[1]
        cid = 'd4ea211e62bd_c0026'
        audit = json.loads((site / f'data/support_repairs/{cid}-v7.json').read_text())
        current = json.loads((site / 'data/clips.json').read_text())
        clips = {c['id']: c for c in current['clips']}
        before = json.loads(subprocess.check_output(
            ['git', 'show', '2dfb149974bae3486e714cdc6b0999cc04a52c7d:data/clips.json'], cwd=site))
        changed = [c['id'] for c in before['clips'] if c != clips[c['id']]]
        self.assertEqual(changed, [cid])
        self.assertEqual(audit['status'], 'support_consistency_passed')
        self.assertEqual(audit['fk_checked_frames'], 62)
        self.assertLess(audit['fk_max_error_model'], 1e-5)
        self.assertFalse(audit['actual_3d_accuracy_validated'])
        pose = json.loads(gzip.decompress((site / 'data' / clips[cid]['pose3d']).read_bytes()))
        self.assertFalse(pose['scene']['per_frame_recentering'])
        self.assertEqual(pose['scene']['floor']['height'], 0)
        a = pose['frames'][0]['anatomy']
        leg = audit['normalization_leg_model']
        seen = set()
        for e in audit['after_episodes']:
            foot = np.array([f['feet'][e['marker']] for f in pose['frames'][e['start']:e['end']+1]])
            extent = np.linalg.norm(foot-foot[0], axis=-1).max()/leg
            self.assertAlmostEqual(extent, e['extent_from_start_leg'], delta=1e-5)
            self.assertLessEqual(extent, .05 if e['marker'] in seen else .03)
            seen.add(e['marker'])
        parents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
        native = np.array([f['anatomy'] for f in pose['frames']])
        lengths = np.linalg.norm(native[:,1:]-native[:,parents[1:]],axis=-1)
        self.assertLess(np.ptp(lengths,axis=0).max(),5e-6)
        self.assertGreater(audit['after']['feet_p100_px'],audit['before']['feet_p100_px'])
        self.assertLessEqual(audit['after']['feet_p95_px'],audit['before']['feet_p95_px']+2)


if __name__ == '__main__':
    unittest.main()
