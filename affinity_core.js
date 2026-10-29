/* Small, shared numerical contract for the observed-pitch comparison. */
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AffinityCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  function proportions(counts) {
    if (!Array.isArray(counts) || counts.length !== 5 ||
        counts.some(v => !Number.isFinite(v) || v < 0)) throw Error('Invalid outcome counts');
    const total = counts.reduce((a, b) => a + b, 0);
    if (!total) throw Error('Empty profile');
    return counts.map(v => v / total);
  }
  function similarity(a, b) {
    const p = proportions(a), q = proportions(b);
    let divergence = 0;
    for (let i = 0; i < p.length; i++) {
      const m = (p[i] + q[i]) / 2;
      if (p[i]) divergence += .5 * p[i] * Math.log2(p[i] / m);
      if (q[i]) divergence += .5 * q[i] * Math.log2(q[i] / m);
    }
    return 100 * (1 - Math.sqrt(Math.max(0, Math.min(1, divergence))));
  }
  function rank(players, selected) {
    return players.filter(p => p.key !== selected.key).map(player => ({
      player, score: similarity(selected.counts, player.counts)
    })).sort((a, b) => b.score - a.score || a.player.key.localeCompare(b.player.key));
  }
  function parse(hash) {
    const params = new URLSearchParams(hash.split('?')[1] || '');
    const raw = params.get('s'), number = raw === null || raw.trim() === '' ? NaN : Number(raw);
    return {
      role: params.get('players') === 'pitchers' ? 'pitchers' : 'hitters',
      key: params.get('player') || '', compare: params.get('compare') || '',
      threshold: Number.isFinite(number) ? Math.round(Math.max(0, Math.min(100, number)) * 10) / 10 : 95
    };
  }
  function fragment(state) {
    const params = new URLSearchParams({players: state.role, player: state.key, s: String(state.threshold)});
    if (state.compare) params.set('compare', state.compare);
    return '#visuals/affinity?' + params;
  }
  return {proportions, similarity, rank, parse, fragment};
});
