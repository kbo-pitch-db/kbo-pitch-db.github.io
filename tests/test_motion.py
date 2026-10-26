"""Numerical/temporal invariants; no claim of anatomical ground truth."""
import unittest
import numpy as np
from scipy.sparse import diags
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from refine_motion import smooth_run, refine_trajectory, constrain_bones, BONES


class MotionTests(unittest.TestCase):
    def test_banded_matches_objective(self):
        rng = np.random.default_rng(5)
        for n in (3, 4, 10, 80):
            y = rng.normal(size=(n, 3))
            w = rng.uniform(.1, 1, n)
            d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n)).toarray()
            expected = np.linalg.solve(np.diag(w) + .45*d.T@d + np.eye(n)*1e-8, w[:,None]*y)
            np.testing.assert_allclose(smooth_run(y, w), expected, atol=1e-9)

    def test_gaps_and_identity_are_preserved(self):
        x = np.tile(np.arange(24)[:,None,None], (1,1,2)).astype(float)
        original = x.copy()
        missing = np.zeros((24,1), bool)
        missing[3:5] = True
        missing[10:15] = True
        missing[19:21] = True
        track = np.r_[np.zeros(20), np.ones(4)]
        y, filled, _ = refine_trajectory(x, np.ones((24,1)), missing,
            np.arange(24)/30, track, max_gap=3)
        self.assertTrue(np.isfinite(y[3:5]).all())
        self.assertTrue(filled[3:5].all())
        self.assertTrue(np.isnan(y[10:15]).all())
        self.assertTrue(np.isnan(y[19:21]).all())
        np.testing.assert_array_equal(x, original)

    def test_single_spike_but_keep_fast_linear_motion(self):
        x = np.tile(np.arange(20)[:,None,None]*100., (1,1,2))
        x[8] += 1000
        y, _, outlier = refine_trajectory(x, np.ones((20,1)), np.zeros((20,1),bool),
            np.arange(20)/30, threshold=500)
        self.assertTrue(outlier[8,0])
        np.testing.assert_allclose(y[:,0,0], np.arange(20)*100., atol=.01)

    def test_bones_and_root(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40,17,3))*.03
        base = rng.normal(size=(17,3))
        x += base
        result, targets = constrain_bones(x, np.ones((40,17)), iterations=60)
        np.testing.assert_allclose((result[:,11]+result[:,12])/2, 0, atol=1e-6)
        before = np.mean([np.std(np.linalg.norm(x[:,b]-x[:,a],axis=-1)) for a,b in BONES])
        after = np.mean([np.std(np.linalg.norm(result[:,b]-result[:,a],axis=-1)) for a,b in BONES])
        self.assertLess(after, before*.1)

    def test_no_phase_delay_on_fast_supported_motion(self):
        time = np.arange(180) / 30
        signal = np.sin(2*np.pi*3*time)
        result = smooth_run(signal[:,None], np.ones(180))[:,0]
        # Offline symmetric smoothing preserves timing and most of a 3 Hz motion.
        middle = slice(15,-15)
        self.assertEqual(int(np.argmax(np.correlate(signal[middle],result[middle],mode='full'))),len(signal[middle])-1)
        self.assertGreater(np.std(result[middle])/np.std(signal[middle]), .9)


if __name__ == '__main__':
    unittest.main()
