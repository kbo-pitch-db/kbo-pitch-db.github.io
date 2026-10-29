"""Embed every eligible pitcher from the existing seasonal similarity cohorts."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def unpack(data):
    if data['similarity'].get('projectionEncoding') == 'delta-int-1e4':
        for p in data['players']:
            for t in p['types']:
                if t:
                    t['projections'] = (np.cumsum(t['projections'], axis=1)/10000).tolist()
    return data


def scores(players):
    counts = np.asarray([p['counts'] for p in players], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    result = np.zeros((len(players), len(players)))
    for k in range(counts.shape[1]):
        indices = np.flatnonzero(counts[:, k])
        points = np.asarray([players[i]['types'][k]['projections'] for i in indices])
        for j, i in enumerate(indices):
            distance = np.abs(points - points[j]).mean(axis=(1, 2))
            result[i, indices] += np.minimum(shares[i, k], shares[indices, k])*np.exp(-distance)
    return np.clip(result, 0, 1)


def embed(distances, starts=5, iterations=1800):
    """Metric SMACOF with classical MDS + fixed-seed starts; retain least stress."""
    n = len(distances)
    squared = distances**2
    centered = -.5*(squared-squared.mean(0)-squared.mean(1)[:, None]+squared.mean())
    values, vectors = np.linalg.eigh(centered)
    classical = vectors[:, -2:] * np.sqrt(np.maximum(values[-2:], 0))
    rng = np.random.default_rng(20261030)
    best = None
    for trial in range(starts):
        x = classical.copy() if trial == 0 else rng.normal(size=(n, 2))*.3
        previous = np.inf
        for step in range(iterations):
            observed = np.linalg.norm(x[:, None]-x, axis=2)
            ratio = distances/np.maximum(observed, 1e-12)
            np.fill_diagonal(ratio, 0)
            b = -ratio
            np.fill_diagonal(b, ratio.sum(axis=1))
            updated = b @ x / n
            difference = np.linalg.norm(updated[:, None]-updated, axis=2)-distances
            stress = np.square(difference).sum()/2
            x = updated
            if previous-stress <= 1e-10*max(1, previous) and np.isfinite(previous):
                break
            previous = stress
        if best is None or stress < best[0]:
            best = stress, x, step+1
    stress, x, steps = best
    x -= x.mean(axis=0)
    # Rotation/reflection have no statistical meaning. Fix orientation for
    # deterministic files without altering pairwise geometry.
    _, _, rotation = np.linalg.svd(x, full_matrices=False)
    x = x @ rotation.T
    for axis in range(2):
        if x[np.argmax(np.abs(x[:, axis])), axis] < 0:
            x[:, axis] *= -1
    actual = np.linalg.norm(x[:, None]-x, axis=2)
    neighbours = np.argsort(distances, axis=1)[:, 1:11]
    projected = np.argsort(actual, axis=1)[:, 1:11]
    recall = np.mean([len(set(a)&set(b))/len(a) for a, b in zip(neighbours, projected)])
    return x, {'normalizedStress': float(np.sqrt(stress/(squared.sum()/2))),
               'nearest10Recall': float(recall), 'iterations': steps}


def build():
    manifest = json.loads((ROOT/'data/visuals/affinity.json').read_text())
    output = {'schemaVersion': 1, 'version': 'pitcher-map-20261030-v1',
              'similarityModel': 'usage-overlap-sliced-wasserstein-v1',
              'layout': 'metric-smacof-2d', 'distance': '1 - similarity/100',
              'latestSeason': manifest['latestSeason'], 'seasons': {}}
    for year, meta in sorted(manifest['seasons'].items()):
        raw = (ROOT/'data'/meta['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise ValueError('Cohort SHA256 mismatch: '+year)
        data = unpack(json.loads(gzip.decompress(raw)))
        matrix = scores(data['players'])
        np.fill_diagonal(matrix, 1)
        xy, audit = embed(1-matrix)
        points = [{'key': p['key'], 'x': round(float(x), 7), 'y': round(float(y), 7)}
                  for p, (x, y) in zip(data['players'], xy)]
        output['seasons'][year] = {'season': int(year), 'cohortSha256': meta['sha256'],
            'dateStart': data['dateStart'], 'dateEnd': data['dateEnd'],
            'minimumPitches': data['minimumPitches'], 'minimumCoverage': data['minimumCoverage'],
            'points': points, 'audit': audit}
        print(json.dumps({'season': year, 'players': len(points), **audit}), flush=True)
    (ROOT/'data/visuals/pitcher-map.json').write_text(json.dumps(output, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    build()
