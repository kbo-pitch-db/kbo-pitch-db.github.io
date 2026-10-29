"""Map coverage, source identity and measured distortion of the 2D layout."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_pitcher_map import embed, scores, unpack


class PitcherMapTests(unittest.TestCase):
    def test_every_eligible_pitcher_is_in_the_correct_fixed_layout(self):
        index = json.loads((ROOT/'data/visuals/pitcher-map.json').read_text())
        source = json.loads((ROOT/'data/visuals/affinity.json').read_text())
        self.assertEqual(set(index['seasons']), set(source['seasons']))
        for year, layout in index['seasons'].items():
            raw = (ROOT/'data'/source['seasons'][year]['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), layout['cohortSha256'])
            cohort = unpack(json.loads(gzip.decompress(raw)))
            self.assertEqual([p['key'] for p in layout['points']], [p['key'] for p in cohort['players']])
            self.assertEqual((layout['dateStart'], layout['dateEnd']), (cohort['dateStart'], cohort['dateEnd']))
            self.assertTrue(all(p['pitches'] >= 500 for p in cohort['players']))
            xy = np.array([[p['x'], p['y']] for p in layout['points']])
            self.assertTrue(np.isfinite(xy).all())
            self.assertGreater(np.linalg.matrix_rank(xy), 1)
            matrix = scores(cohort['players'])
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            np.testing.assert_allclose(matrix.diagonal(), 1, atol=1e-12)
            target = 1-matrix
            np.fill_diagonal(target, 0)
            observed = np.linalg.norm(xy[:, None]-xy, axis=2)
            stress = np.sqrt(np.square(observed-target).sum()/np.square(target).sum())
            self.assertAlmostEqual(float(stress), layout['audit']['normalizedStress'], places=6)
            self.assertLess(stress, .31)
            # A 2D projection loses real neighbours; keep the measured recall
            # visible in the audit instead of claiming exact distances.
            expected = np.argsort(target, axis=1)[:, 1:11]
            projected = np.argsort(observed, axis=1)[:, 1:11]
            recall = np.mean([len(set(a)&set(b))/10 for a,b in zip(expected, projected)])
            self.assertAlmostEqual(float(recall), layout['audit']['nearest10Recall'])
            self.assertGreater(recall, .4)

    def test_same_distribution_and_usage_share_match_existing_metric(self):
        def player(counts, values):
            return {'counts': counts, 'types': [None if value is None else {'projections': [[value]*65]*13} for value in values]}
        a = player([100, 0], [0, None])
        same = player([1000, 0], [0, None])
        mixed = player([50, 50], [0, 0])
        shifted = player([100, 0], [1, None])
        other = player([0, 100], [None, 0])
        result = scores([a, same, mixed, shifted, other])
        self.assertAlmostEqual(result[0, 1], 1)
        self.assertAlmostEqual(result[0, 2], .5)
        self.assertAlmostEqual(result[0, 3], np.exp(-1))
        self.assertEqual(result[0, 4], 0)

    def test_embedding_retains_known_planar_distances(self):
        xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        distances = np.linalg.norm(xy[:, None]-xy, axis=2)
        result, audit = embed(distances, starts=1, iterations=200)
        np.testing.assert_allclose(np.linalg.norm(result[:, None]-result, axis=2), distances, atol=1e-8)
        self.assertLess(audit['normalizedStress'], 1e-10)


if __name__ == '__main__':
    unittest.main()
