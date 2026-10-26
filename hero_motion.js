/* Animate the hero with a real, packaged RTMPose sequence. */
(() => {
  const skeleton = document.querySelector('.hero-skeleton');
  if (!skeleton) return;
  const bones = document.querySelector('#hero-bones');
  const joints = document.querySelector('#hero-joints');
  const trail = document.querySelector('#hero-wrist-trail');
  const timeLabel = document.querySelector('#hero-time');
  const frameLabel = document.querySelector('#hero-frame');
  const fpsLabel = document.querySelector('#hero-fps');
  const countLabel = document.querySelector('#hero-visible-joints');
  const progress = document.querySelector('#hero-progress');
  const edges = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
    [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
  const indexUrl = new URL('data/clips.json', document.currentScript.src);

  const valid = point => Array.isArray(point) && Number.isFinite(point[0]) &&
    Number.isFinite(point[1]) && (point[2] == null || point[2] >= .3);
  const percentile = (values, fraction) => {
    const sorted = values.sort((a, b) => a - b);
    return sorted[Math.floor((sorted.length - 1) * fraction)];
  };

  async function start() {
    const indexResponse = await fetch(indexUrl, { cache: 'no-store' });
    if (!indexResponse.ok) return;
    const index = await indexResponse.json();
    const clip = (index.clips || []).find(item => item.pose && !item.example && item.frames >= 30);
    if (!clip) return;
    const poseUrl = new URL(clip.pose, indexUrl);
    if (index.packageVersion) poseUrl.searchParams.set('v', index.packageVersion);
    const poseResponse = await fetch(poseUrl);
    if (!poseResponse.ok) return;
    const pose = await poseResponse.json();
    const skeletonEdges = pose.edges || edges;
    const jointCount = document.querySelector('.art-tag.tag-left span');
    if (jointCount) jointCount.textContent = String((pose.joints || []).length || 17);
    const frames = (pose.frames || []).slice(0, 240);
    const points = frames.flatMap(frame => (frame.keypoints || []).filter(valid));
    if (frames.length < 2 || points.length < 30) return;
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    const left = percentile(xs, .02), right = percentile(xs, .98);
    const top = percentile(ys, .02), bottom = percentile(ys, .98);
    const scale = Math.min(236 / Math.max(1, right - left), 300 / Math.max(1, bottom - top));
    const map = point => [160 + (point[0] - (left + right) / 2) * scale,
      195 + (point[1] - (top + bottom) / 2) * scale];
    const fps = Number(pose.fps || clip.fps) || 30;
    fpsLabel.textContent = String(Math.round(fps));
    skeleton.setAttribute('aria-label', `Animated 2D pose from clip ${clip.id}`);

    function draw(index) {
      const frame = frames[index];
      const keypoints = frame.keypoints || [];
      const mapped = keypoints.map(point => valid(point) ? map(point) : null);
      bones.innerHTML = skeletonEdges.filter(([a, b]) => mapped[a] && mapped[b]).map(([a, b]) =>
        `<path d="M${mapped[a][0].toFixed(1)} ${mapped[a][1].toFixed(1)}L${mapped[b][0].toFixed(1)} ${mapped[b][1].toFixed(1)}"/>`).join('');
      joints.innerHTML = mapped.filter(Boolean).map(([x, y]) =>
        `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>`).join('');
      const wristHistory = [];
      for (let offset = 14; offset >= 0; offset--) {
        if (index - offset < 0) continue;
        const candidate = frames[index - offset].keypoints || [];
        const wrist = valid(candidate[10]) ? candidate[10] : valid(candidate[9]) ? candidate[9] : null;
        if (wrist) wristHistory.push(map(wrist).map(value => value.toFixed(1)).join(','));
      }
      trail.setAttribute('points', wristHistory.join(' '));
      const seconds = Number(frame.time) || index / fps;
      timeLabel.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
      frameLabel.textContent = `F${String(frame.frame ?? index).padStart(4, '0')}`;
      countLabel.textContent = String(mapped.filter(Boolean).length);
      progress.style.width = `${Math.max(2, 100 * (index + 1) / frames.length)}%`;
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) { draw(Math.floor(frames.length / 3)); return; }
    let visible = true;
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(entries => { visible = entries[0].isIntersecting; },
        { threshold: .05 }).observe(skeleton);
    }
    let started = performance.now();
    let previous = -1;
    function tick(now) {
      if (visible && !document.hidden) {
        const index = Math.floor((now - started) * fps / 1000) % frames.length;
        if (index !== previous) { draw(index); previous = index; }
      } else {
        started += now - (tick.lastTime || now);
      }
      tick.lastTime = now;
      requestAnimationFrame(tick);
    }
    draw(0);
    requestAnimationFrame(tick);
  }
  start().catch(() => {});
})();
