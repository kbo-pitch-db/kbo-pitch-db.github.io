/* Final 2D and 3D motion share one playback clock. */
window.MotionViewer = (() => {
  let state = null, animation = 0, generation = 0;
  const edges = [[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  const copy = {
    en: {view2:'Final 2D',view3:'Final 3D',yaw:'3D rotation',pitch:'Tilt',reset:'Reset view',
      note:'3D depth and scale are monocular estimates. Hip-relative coordinates; pitching accuracy has not been validated.',
      gap:'No supported joints in this frame',limited:'Limited tracking',ready:'3D estimate available',scene:'Fixed clip frame · inferred floor',
      load:'Motion data could not be loaded.', unavailable:'3D data unavailable',method:'Method & quality',interpolated:'Amber = interpolated joint'},
    ko: {view2:'최종 2D',view3:'최종 3D',yaw:'3D 회전',pitch:'기울기',reset:'시점 초기화',
      note:'3D 깊이와 크기는 단안 추정치입니다. 골반 중심 상대 좌표이며 투구 정확도는 검증되지 않았습니다.',
      gap:'이 프레임에 표시할 관절이 없습니다',limited:'추적 제한 있음',ready:'3D 추정 결과 있음',scene:'클립 고정 좌표 · 추정 바닥',
      load:'동작 데이터를 불러오지 못했습니다.',unavailable:'3D 데이터 없음',method:'추정 방식·품질',interpolated:'주황색 = 보간된 관절'}
  };
  function dispose() {
    generation++; cancelAnimationFrame(animation);
    for(const view of state?.views||[])view.mesh?.dispose();
    state = null;
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
  async function mount({clip, panel, lang, url, pose2d, onPoses, clock}) {
    dispose();
    const ticket = generation, t = copy[lang] || copy.en;
    const control = document.createElement('section'); control.className = 'motion-controls';
    control.innerHTML = `<div class="motion-orbit" hidden><label>${t.yaw}<input class="motion-yaw" type="range" min="-180" max="180" value="0" aria-label="${t.yaw}"></label><label>${t.pitch}<input class="motion-pitch" type="range" min="-60" max="60" value="0" aria-label="${t.pitch}"></label><button class="download-btn motion-reset" type="button">${t.reset}</button></div><p class="motion-note">${t.interpolated}</p><a class="motion-quality" target="_blank" rel="noopener">${t.method} ↗</a><p class="motion-status" role="status"></p>`;
    panel.querySelector('.player-bar').after(control);
    control.querySelector('.motion-quality').href = url(clip.motionQuality);
    control.querySelector('.motion-quality').hidden = !clip.motionQuality;
    const views = [{data:pose2d,canvas:panel.querySelector('.motion-canvas[data-dimension="2"]')}];
    const s = state = {clip,panel,control,views,video:clock,yaw:0,pitch:0,t};
    function ready(view) {
      const {data,canvas} = view, dim = data.dimension === 3 ? 3 : 2;
      const points = data.frames.flatMap(f=>f.keypoints.filter((p,j)=>p && ((p[dim]??1)>=.2 || f.interpolated?.[j])));
      if (dim === 3) {
        const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
        for(const p of points)for(let k=0;k<3;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}
        view.center3=points.length?low.map((v,k)=>(v+high[k])/2):[0,.8,0];
        view.radius=points.reduce((r,p)=>Math.max(r,Math.hypot(...view.center3.map((v,k)=>p[k]-v))),1)*1.08;
      } else {
        const low=[Infinity,Infinity], high=[-Infinity,-Infinity];
        for (const p of points) for(let k=0;k<2;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}
        view.center=points.length?low.map((v,k)=>(v+high[k])/2):[0,0];
        view.extent=points.length?Math.max(high[0]-low[0],high[1]-low[1],1)*1.18:1;
      }
      canvas.hidden=false;
      panel.querySelector(`.motion-placeholder[data-dimension="${dim}"]`).hidden=true;
      panel.querySelectorAll(`.motion-downloads [data-dimension="${dim}"]`).forEach(button=>{button.disabled=false;});
      onPoses(Object.fromEntries(views.map(v=>[v.data.dimension===3?3:2,v.data])));
      draw();
    }
    ready(views[0]);
    const tick = () => {if (state!==s) return;if(s.video.currentTime!==s.lastTime||s.video.paused!==s.lastPaused)draw();animation=requestAnimationFrame(tick);}; tick();
    if (!clip.pose3d) {
      panel.querySelector('.motion-placeholder[data-dimension="3"]').textContent=t.unavailable;
      return;
    }
    try {
      const response=await fetch(url(clip.pose3d)); if(!response.ok) throw Error(clip.pose3d);
      const pose3d=await response.json();
      if (generation!==ticket || !panel.isConnected) return;
      if (pose3d.dimension!==3 || pose3d.frames.length!==pose2d.frames.length || pose3d.fps!==pose2d.fps ||
          pose3d.frames.some((f,i)=>f.source_frame!==pose2d.frames[i].source_frame || f.source_time!==pose2d.frames[i].source_time)) {
        throw Error('unaligned final motion');
      }
      const view3={data:pose3d,canvas:panel.querySelector('.motion-canvas[data-dimension="3"]')};
      try{
        const module=window.BoneScene || await import('./bone_scene.js?v=20261027-scene1');
        if(generation!==ticket || !panel.isConnected)return;
        view3.mesh=module.createBoneScene(view3.canvas,pose3d);
      }catch(error){
        // A separate canvas permits a 2D projection when WebGL is unavailable.
        const fallback=view3.canvas.cloneNode(false);view3.canvas.replaceWith(fallback);view3.canvas=fallback;
        view3.canvas.dataset.renderer='canvas-fallback';
      }
      views.push(view3);
      ready(views[1]);
      control.querySelector('.motion-orbit').hidden=false;
      control.querySelector('.motion-status').textContent=pose3d.scene?t.scene+(clip.pose3dStatus==='limited'?' · '+t.limited:''):t[clip.pose3dStatus==='available'?'ready':'limited'];
      const coordinates=panel.querySelector('[data-meta="coords"] .meta-value');
      if(coordinates)coordinates.title=coordinates.textContent='2D: crop pixels · 3D: '+pose3d.coordinate_system;
      for (const axis of ['yaw','pitch']) control.querySelector('.motion-'+axis).addEventListener('input', e => {
        s[axis] = Number(e.target.value)*Math.PI/180; draw();
      });
      control.querySelector('.motion-reset').addEventListener('click', () => {
        s.yaw = s.pitch = 0;
        view3.mesh?.resetZoom();
        for (const axis of ['yaw','pitch']) control.querySelector('.motion-'+axis).value = '0';
        draw();
      });
      const canvas=views[1].canvas;
      let drag = null;
      canvas.addEventListener('pointerdown',e => {
        drag = [e.clientX,e.clientY,s.yaw,s.pitch]; canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove',e => {
        if (!drag || state!==s) return;
        s.yaw = Math.max(-Math.PI, Math.min(Math.PI, drag[2]+(e.clientX-drag[0])*.01));
        s.pitch = Math.max(-Math.PI/3, Math.min(Math.PI/3, drag[3]+(e.clientY-drag[1])*.008));
        control.querySelector('.motion-yaw').value = String(s.yaw*180/Math.PI);
        control.querySelector('.motion-pitch').value = String(s.pitch*180/Math.PI); draw();
      });
      canvas.addEventListener('pointerup',()=>{drag=null}); canvas.addEventListener('pointercancel',()=>{drag=null});
    } catch (error) {
      if (generation === ticket && panel.isConnected) {
        panel.querySelector('.motion-placeholder[data-dimension="3"]').textContent=t.load;
        control.querySelector('.motion-status').textContent=t.load;
      }
    }
  }
  function draw() {
    const s=state; if(!s) return;
    s.lastTime=s.video.currentTime;s.lastPaused=s.video.paused;
    for (const view of s.views) drawView(s,view);
  }
  function drawView(s,view) {
    const {data,canvas}=view;
    if(canvas.hidden) return;
    const is3d=data.dimension===3, dim=is3d?3:2;
    const position=Math.min(data.frames.length-1,Math.max(0,s.video.currentTime*data.fps));
    const frame=s.video.paused?Math.round(position):Math.floor(position), fraction=s.video.paused?0:position-frame;
    const current=data.frames[frame], next=data.frames[Math.min(frame+1,data.frames.length-1)];
    canvas.dataset.frame=String(frame);
    if (!current) return;
    if(view.mesh){view.mesh.render(current,next,fraction,s.yaw,s.pitch);return;}
    const ctx = canvas.getContext('2d'), w=canvas.width, h=canvas.height;
    if(!ctx)return;
    ctx.fillStyle='#14201b'; ctx.fillRect(0,0,w,h);
    const project = p => {
      if (!is3d) return [(p[0]-view.center[0])*w/view.extent+w/2,(p[1]-view.center[1])*h/view.extent+h/2,0];
      const a=p.map((v,k)=>v-view.center3[k]);
      const x=a[0]*Math.cos(s.yaw)+a[2]*Math.sin(s.yaw), z=-a[0]*Math.sin(s.yaw)+a[2]*Math.cos(s.yaw);
      const y=a[1]*Math.cos(s.pitch)-z*Math.sin(s.pitch), depth=a[1]*Math.sin(s.pitch)+z*Math.cos(s.pitch);
      return [w/2+x*w/(2*view.radius),h/2-y*h/(2*view.radius),depth];
    };
    const points=current.keypoints.map((p,j)=>{
      if(!p || ((p[dim]??1)<.2 && !current.interpolated?.[j]))return null;
      const candidate=next.keypoints[j];
      const q=candidate && ((candidate[dim]??1)>=.2 || next.interpolated?.[j]) ? candidate : null;
      const value=p.slice(0,dim).map((v,k)=>q?v+(q[k]-v)*fraction:v);
      return project(value);
    });
    if(is3d){
      if(data.scene){
        ctx.strokeStyle='#395847';ctx.lineWidth=1;
        const floor=data.scene.floor.height;
        for(let q=-2;q<=2;q+=.25)for(const pair of [[[q,floor,-2],[q,floor,2]],[[-2,floor,q],[2,floor,q]]]){
          const a=project(pair[0]),b=project(pair[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
        }
      }
      const origin=project([0,0,0]);
      for(const [axis,color,name] of [[[.25,0,0],'#e9a49b','X'],[[0,.25,0],'#b4f068','Y'],[[0,0,.25],'#86b8e5','Z']]){
        const p=project(axis);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...origin.slice(0,2));ctx.lineTo(...p.slice(0,2));ctx.stroke();ctx.fillStyle=color;ctx.font='16px sans-serif';ctx.fillText(name,p[0]+6,p[1]);
      }
    }
    const skeleton=data.edges || edges;
    const ordered=skeleton.filter(([a,b])=>points[a]&&points[b]).sort(([a,b],[c,d])=>(points[a][2]+points[b][2])-(points[c][2]+points[d][2]));
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
