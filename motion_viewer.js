/* Numeric-only 2D/3D viewer. The existing MP4 supplies the shared playback clock. */
window.MotionViewer = (() => {
  let state = null, animation = 0, generation = 0;
  const edges = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  const copy = {
    en: {view:'Motion view',raw2:'2D · original',smooth2:'2D · refined',raw3:'3D · learned',smooth3:'3D · refined',
      blend:'Correction',yaw:'Rotate',pitch:'Tilt',reset:'Reset view',
      note:'3D depth and scale are monocular estimates. Hip-relative coordinates; pitching accuracy has not been validated.',
      gap:'No supported joints in this frame',limited:'Limited tracking',ready:'3D estimate available',
      load:'Motion data could not be loaded.', method:'Method & quality', interpolated:'Amber = interpolated joint',
      adjustment:'Correction blends original and refined coordinates. Downloads include this setting.'},
    ko: {view:'동작 보기',raw2:'2D · 원본',smooth2:'2D · 보정',raw3:'3D · 모델 추정',smooth3:'3D · 보정',
      blend:'보정 적용',yaw:'회전',pitch:'기울기',reset:'시점 초기화',
      note:'3D 깊이와 크기는 단안 추정치입니다. 골반 중심 상대 좌표이며 투구 정확도는 검증되지 않았습니다.',
      gap:'이 프레임에 표시할 관절이 없습니다',limited:'추적 제한 있음',ready:'3D 추정 결과 있음',
      load:'동작 데이터를 불러오지 못했습니다.',method:'추정 방식·품질',interpolated:'주황색 = 보간된 관절',
      adjustment:'원본과 보정 좌표를 섞는 비율입니다. 내려받는 데이터에도 적용됩니다.'}
  };
  function dispose() {
    generation++; cancelAnimationFrame(animation); state = null;
  }
  function createClock(element, frames, fps) {
    const events = new EventTarget();
    let time = 0, paused = true, rate = 1, timer = 0, previous = 0;
    const emit = name => events.dispatchEvent(new Event(name));
    const duration = frames / fps;
    const tick = now => {
      if (paused) return;
      time = Math.min(duration, time + (now-previous)/1000*rate); previous = now;
      emit('timeupdate');
      if (time >= duration) {paused=true;emit('pause');emit('ended');return;}
      timer = requestAnimationFrame(tick);
    };
    return {
      get currentTime(){return time;}, set currentTime(value){time=Math.max(0,Math.min(duration,Number(value)||0));emit('timeupdate');},
      get paused(){return paused;},get duration(){return duration;},
      get playbackRate(){return rate;},set playbackRate(value){rate=Number(value)||1;},
      videoWidth:0,videoHeight:0,classList:element.classList,
      addEventListener:(...args)=>events.addEventListener(...args),
      play(){if(!paused)return Promise.resolve();if(time>=duration)time=0;paused=false;previous=performance.now();emit('play');timer=requestAnimationFrame(tick);return Promise.resolve();},
      pause(){if(paused)return;paused=true;cancelAnimationFrame(timer);emit('pause');}
    };
  }
  async function mount({clip, panel, lang, url, raw, onPose, clock}) {
    dispose();
    if (!clip.poseRefined) return;
    const ticket = generation, t = copy[lang] || copy.en;
    const control = document.createElement('section'); control.className = 'motion-controls';
    control.innerHTML = `<label>${t.view}<select class="motion-mode" aria-label="${t.view}"><option value="raw2">${t.raw2}</option><option value="smooth2">${t.smooth2}</option>${clip.pose3d ? `<option value="raw3">${t.raw3}</option><option value="smooth3">${t.smooth3}</option>` : ''}</select></label><label class="motion-blend">${t.blend} <output>100%</output><input type="range" min="0" max="100" value="100" aria-label="${t.blend}"></label><div class="motion-orbit" hidden><label>${t.yaw}<input class="motion-yaw" type="range" min="-180" max="180" value="0" aria-label="${t.yaw}"></label><label>${t.pitch}<input class="motion-pitch" type="range" min="-60" max="60" value="0" aria-label="${t.pitch}"></label><button class="download-btn motion-reset" type="button">${t.reset}</button></div><p class="motion-adjustment">${t.adjustment}</p><p class="motion-note">${t.interpolated}</p><p class="motion-3d-note" hidden>${t.note}</p><a class="motion-quality" target="_blank" rel="noopener">${t.method} ↗</a><p class="motion-status" role="status"></p>`;
    panel.querySelector('.player-bar').after(control);
    const canvas = document.createElement('canvas'); canvas.className = 'motion-canvas';
    canvas.width = 720; canvas.height = 720; canvas.hidden = true;
    canvas.setAttribute('aria-label', t.view); canvas.setAttribute('role','img');
    panel.querySelector('.player-wrap').append(canvas);
    control.querySelector('.motion-quality').href = url(clip.motionQuality);
    const mode = control.querySelector('.motion-mode'); mode.disabled = true;
    try {
      const assets = {raw2:raw};
      await Promise.all([['smooth2',clip.poseRefined],['raw3',clip.pose3dRaw],['smooth3',clip.pose3d]].filter(([,p]) => p).map(async ([key,path]) => {
        const response = await fetch(url(path)); if (!response.ok) throw Error(path);
        assets[key] = await response.json();
        if (assets[key].frames.length !== raw.frames.length) throw Error('unaligned motion');
      }));
      if (generation !== ticket || !panel.isConnected) return;
      state = {clip,panel,control,canvas,video:clock || panel.querySelector('video'),assets,onPose,
        mode:'smooth2',blend:1,yaw:0,pitch:0,t};
      mode.disabled = false; mode.value = 'smooth2';
      mode.addEventListener('change', () => {state.mode = mode.value; update();});
      const slider = control.querySelector('.motion-blend input');
      slider.addEventListener('input', () => {state.blend = Number(slider.value)/100; update();});
      for (const axis of ['yaw','pitch']) control.querySelector('.motion-'+axis).addEventListener('input', e => {
        state[axis] = Number(e.target.value)*Math.PI/180; draw();
      });
      control.querySelector('.motion-reset').addEventListener('click', () => {
        state.yaw = state.pitch = 0;
        for (const axis of ['yaw','pitch']) control.querySelector('.motion-'+axis).value = '0';
        draw();
      });
      let drag = null;
      canvas.addEventListener('pointerdown',e => {
        if (!state?.mode.endsWith('3')) return;
        drag = [e.clientX,e.clientY,state.yaw,state.pitch]; canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove',e => {
        if (!drag || !state) return;
        state.yaw = Math.max(-Math.PI, Math.min(Math.PI, drag[2]+(e.clientX-drag[0])*.01));
        state.pitch = Math.max(-Math.PI/3, Math.min(Math.PI/3, drag[3]+(e.clientY-drag[1])*.008));
        control.querySelector('.motion-yaw').value = String(state.yaw*180/Math.PI);
        control.querySelector('.motion-pitch').value = String(state.pitch*180/Math.PI); draw();
      });
      canvas.addEventListener('pointerup',()=>{drag=null}); canvas.addEventListener('pointercancel',()=>{drag=null});
      update();
      const tick = () => {if (!state) return; draw(); animation = requestAnimationFrame(tick);}; tick();
    } catch (error) {
      if (generation === ticket && panel.isConnected) control.querySelector('.motion-status').textContent = t.load;
    }
  }
  function update() {
    const s = state; if (!s) return;
    const refined = s.mode.startsWith('smooth'), is3d = s.mode.endsWith('3');
      const data = s.assets[s.mode];
    const original = s.assets[is3d?'raw3':'raw2'];
    const dim = is3d?3:2;
    s.active = {...data, dimension:dim, display_correction:refined?s.blend:0,
      frames:data.frames.map((frame,i) => ({...frame,keypoints:frame.keypoints.map((p,j) => {
        if (!refined || s.blend === 1) return p;
        const q = original.frames[i].keypoints[j];
        if (s.blend === 0) return q;
        if (!p) return null; if (!q) return p;
        return p.map((v,k) => k<dim?v*s.blend+q[k]*(1-s.blend):v);
      })}))};
    if (refined && s.blend === 0) s.active.frames = original.frames;
    s.onPose(s.active);
    const modelLabel = s.panel.querySelector('[data-meta="model"] .meta-value');
    const coordinatesLabel = s.panel.querySelector('[data-meta="coords"] .meta-value');
    if (modelLabel) modelLabel.title = modelLabel.textContent = data.model || 'RTMPose';
    if (coordinatesLabel) coordinatesLabel.title = coordinatesLabel.textContent = data.coordinate_system || 'crop pixels';
    s.control.querySelector('.motion-blend').hidden = !refined;
    s.control.querySelector('.motion-adjustment').hidden = !refined;
    s.control.querySelector('output').value = Math.round(s.blend*100)+'%';
    s.control.querySelector('.motion-orbit').hidden = !is3d;
    s.control.querySelector('.motion-3d-note').hidden = !is3d;
    s.canvas.hidden = s.mode === 'raw2' && Boolean(s.clip.poseVideo);
    s.video.classList.toggle('motion-clock', !s.canvas.hidden);
    const label = s.panel.querySelector('.video-label');
    label.querySelector('span').textContent = s.t[s.mode];
    label.querySelector('i').textContent = is3d?'3D POSE':'2D POSE';
    s.control.querySelector('.motion-status').textContent = is3d?s.t[s.clip.pose3dStatus === 'available'?'ready':'limited']:'';
    // Fixed clip-wide bounds; no frame-wise zoom or recentering.
    const points = s.active.frames.flatMap(f=>f.keypoints.filter((p,j)=>p && ((p[dim]??1)>=.2 || f.interpolated?.[j])));
    if (is3d) {
      s.radius = Math.max(.9, ...points.map(p=>Math.hypot(p[0],p[1],p[2])))*1.12;
    } else {
      const low = [Infinity,Infinity], high = [-Infinity,-Infinity];
      for (const p of points) for(let k=0;k<2;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}
      s.center = points.length?low.map((v,k)=>(v+high[k])/2):[0,0];
      s.extent = points.length?Math.max(high[0]-low[0],high[1]-low[1],1)*1.18:1;
    }
    draw();
  }
  function draw() {
    const s = state; if (!s || s.canvas.hidden) return;
    const ctx = s.canvas.getContext('2d'), w=s.canvas.width, h=s.canvas.height;
    ctx.fillStyle='#14201b'; ctx.fillRect(0,0,w,h);
    const is3d=s.mode.endsWith('3'), dim=is3d?3:2;
    const position=Math.min(s.active.frames.length-1,Math.max(0,s.video.currentTime*s.active.fps));
    const frame=s.video.paused?Math.round(position):Math.floor(position), fraction=s.video.paused?0:position-frame;
    const current=s.active.frames[frame], next=s.active.frames[Math.min(frame+1,s.active.frames.length-1)];
    if (!current) return;
    const project = p => {
      if (!is3d) return [(p[0]-s.center[0])*w/s.extent+w/2,(p[1]-s.center[1])*h/s.extent+h/2,0];
      const x=p[0]*Math.cos(s.yaw)+p[2]*Math.sin(s.yaw), z=-p[0]*Math.sin(s.yaw)+p[2]*Math.cos(s.yaw);
      const y=p[1]*Math.cos(s.pitch)-z*Math.sin(s.pitch), depth=p[1]*Math.sin(s.pitch)+z*Math.cos(s.pitch);
      return [w/2+x*w/(2*s.radius),h/2-y*h/(2*s.radius),depth];
    };
    const points=current.keypoints.map((p,j)=>{
      if(!p || ((p[dim]??1)<.2 && !current.interpolated?.[j]))return null;
      const candidate=next.keypoints[j];
      const q=candidate && ((candidate[dim]??1)>=.2 || next.interpolated?.[j]) ? candidate : null;
      const value=p.slice(0,dim).map((v,k)=>q?v+(q[k]-v)*fraction:v);
      return project(value);
    });
    if(is3d){
      const origin=project([0,0,0]);
      for(const [axis,color,name] of [[[.25,0,0],'#e9a49b','X'],[[0,.25,0],'#b4f068','Y'],[[0,0,.25],'#86b8e5','Z']]){
        const p=project(axis);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...origin.slice(0,2));ctx.lineTo(...p.slice(0,2));ctx.stroke();ctx.fillStyle=color;ctx.font='16px sans-serif';ctx.fillText(name,p[0]+6,p[1]);
      }
    }
    const ordered=edges.filter(([a,b])=>points[a]&&points[b]).sort(([a,b],[c,d])=>(points[a][2]+points[b][2])-(points[c][2]+points[d][2]));
    ctx.lineWidth=5;ctx.lineCap='round';
    for(const [a,b] of ordered){
      ctx.strokeStyle=a%2===1&&a>=5?'#b4f068':'#77cde0';
      ctx.beginPath();ctx.moveTo(points[a][0],points[a][1]);ctx.lineTo(points[b][0],points[b][1]);ctx.stroke();
    }
    for(let j=0;j<points.length;j++)if(points[j]){
      ctx.fillStyle=current.interpolated?.[j]?'#ffc47a':'#eafadc';
      ctx.beginPath();ctx.arc(points[j][0],points[j][1],6,0,2*Math.PI);ctx.fill();
    }
    if(!points.some(Boolean)){ctx.fillStyle='#c6d4c9';ctx.font='19px sans-serif';ctx.textAlign='center';ctx.fillText(s.t.gap,w/2,h/2);ctx.textAlign='start';}
  }
  return {mount,dispose,draw,createClock};
})();
