/* Final 2D and 3D motion share one playback clock. */
window.MotionViewer = (() => {
  let state = null, animation = 0, generation = 0;
  const edges = [[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  const copy = {
    en: {view2:'Final 2D',view3:'Final 3D',yaw:'3D rotation',pitch:'Tilt',reset:'Reset view',
      note:'3D depth and scale are monocular estimates. Hip-relative coordinates; pitching accuracy has not been validated.',
      preview:'3D preview · review required',reviewReasons:{short_stride_depth_review:'short stride / uncertain depth',depth_initialization_sensitivity:'depth depends on initialization',stride_direction_depth_review:'uncertain stride direction',pitch_event_order_review:'uncertain pitch-event order',pitch_preparation_unresolved:'preparation unresolved',arm_phase_unresolved:'arm phase unresolved',pitch_contact_phase_ambiguous:'uncertain landing timing',floor_penetration:'foot-floor penetration',insufficient_supported_frames:'insufficient supported frames',poor_2d_alignment:'limited 2D alignment',visual_review_failed:'visual review requires follow-up',visual_review_pending:'visual review pending',stance_anchor_refinement_required:'stance-anchor refinement pending'},
      gap:'No supported joints in this frame',limited:'Limited tracking',ready:'Stability screened · depth and stride unvalidated',scene:'Fixed clip frame · inferred floor',
      load:'Motion data could not be loaded.', unavailable:'3D data unavailable',depthheld:'3D withheld: stride depth needs review.',sequenceheld:'3D withheld: pitch-event order needs review.',contactheld:'3D withheld: landing timing is uncertain.',visualpending:'3D is awaiting visual review.',pending:'3D refitting and quality checks in progress.',withheld:'3D withheld: motion, support or depth needs review.',method:'Method & quality',interpolated:'Amber = interpolated joint'},
    ko: {view2:'최종 2D',view3:'최종 3D',yaw:'3D 회전',pitch:'기울기',reset:'시점 초기화',
      note:'영상에서 추정한 3D 동작입니다. 실제 깊이와 크기는 차이가 날 수 있습니다.',
      preview:'3D 미리보기 · 검토 필요',reviewReasons:{short_stride_depth_review:'짧은 보폭·깊이 불확실',depth_initialization_sensitivity:'초기값에 따른 깊이 차이',stride_direction_depth_review:'보폭 방향 불확실',pitch_event_order_review:'투구 단계 순서 불확실',pitch_preparation_unresolved:'준비 자세 불확실',arm_phase_unresolved:'팔 동작 단계 불확실',pitch_contact_phase_ambiguous:'착지 시점 불확실',floor_penetration:'발의 바닥 관통',insufficient_supported_frames:'관측 지원 부족',poor_2d_alignment:'2D 정합 제한',visual_review_failed:'원본 시각 대조 후 추가 검토 필요',visual_review_pending:'시각 검토 대기',stance_anchor_refinement_required:'접지 위치 추가 보정 대기'},
      gap:'이 프레임에 표시할 관절이 없습니다',limited:'추적 제한 있음',ready:'움직임 검사 통과 · 깊이·스트라이드 미검증',scene:'클립 고정 좌표 · 추정 바닥',
      load:'동작 데이터를 불러오지 못했습니다.',unavailable:'3D 데이터 없음',depthheld:'보폭 깊이를 신뢰하기 어려워 3D를 보류했습니다.',sequenceheld:'투구 단계의 연결을 확인하지 못해 3D를 보류했습니다.',contactheld:'착지 시점을 확정하지 못해 3D를 보류했습니다.',visualpending:'3D 시각 검토를 기다리고 있습니다.',pending:'3D 재처리와 품질 확인 중입니다.',withheld:'3D 품질 검사를 통과하지 못해 3D 표시를 보류했습니다.',method:'추정 방식·품질',interpolated:'주황색 = 보간된 관절'}
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
    control.innerHTML = `<div class="motion-orbit" hidden><label>${t.yaw}<input class="motion-yaw" type="range" min="-180" max="180" value="0" aria-label="${t.yaw}"></label><label>${t.pitch}<input class="motion-pitch" type="range" min="-60" max="60" value="0" aria-label="${t.pitch}"></label><button class="download-btn motion-reset" type="button">${t.reset}</button></div><p class="motion-note">${t.interpolated}</p><p class="motion-status" role="status"></p>`;
    panel.querySelector('.player-bar').after(control);
    const views = [{data:pose2d,canvas:panel.querySelector('.motion-canvas[data-dimension="2"]')}];
    const s = state = {clip,panel,control,views,video:clock,yaw:0,pitch:0,t};
    function ready(view) {
      const {data,canvas} = view, dim = data.dimension === 3 ? 3 : 2;
      const points = data.frames.flatMap(f=>dim===3&&f.anatomy?f.anatomy.filter(Boolean):f.keypoints.filter((p,j)=>p && ((p[dim]??1)>=.2 || f.interpolated?.[j])));
      if (dim === 3) {
        const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
        for(const p of points)for(let k=0;k<3;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}
        view.center3=points.length?low.map((v,k)=>(v+high[k])/2):[0,.8,0];
        view.radius=points.reduce((r,p)=>Math.max(r,Math.hypot(...view.center3.map((v,k)=>p[k]-v))),1)*1.08;
        if(data.scene?.comparison_bounds){const b=data.scene.comparison_bounds;view.center3=b.min.map((v,k)=>(v+b.max[k])/2);view.radius=Math.max(.95,Math.hypot(...b.min.map((v,k)=>b.max[k]-v))/2)*1.08;}
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
    if (!clip.pose3d || clip.pose3dStatus==='review_required') {
      panel.querySelector('.motion-placeholder[data-dimension="3"]').textContent=clip.pose3dReasons?.includes('contact_fit_pending')?t.pending:clip.pose3dReasons?.includes('visual_review_pending')?t.visualpending:clip.pose3dReasons?.some(reason=>['short_stride_depth_review','depth_initialization_sensitivity','stride_direction_depth_review'].includes(reason))?t.depthheld:clip.pose3dReasons?.some(reason=>['pitch_event_order_review','pitch_preparation_unresolved','arm_phase_unresolved'].includes(reason))?t.sequenceheld:clip.pose3dReasons?.includes('pitch_contact_phase_ambiguous')?t.contactheld:clip.pose3dStatus==='review_required'?t.withheld:t.unavailable;
      return;
    }
    try {
      const response=await fetch(url(clip.pose3d)); if(!response.ok) throw Error(clip.pose3d);
      const pose3d=window.ReconstructionRegistry?await window.ReconstructionRegistry.decode(response,clip.pose3d):await response.json();
      if (generation!==ticket || !panel.isConnected) return;
      if (pose3d.dimension!==3 || pose3d.frames.length!==pose2d.frames.length || pose3d.fps!==pose2d.fps ||
          pose3d.frames.some((f,i)=>f.source_frame!==pose2d.frames[i].source_frame || f.source_time!==pose2d.frames[i].source_time)) {
        throw Error('unaligned final motion');
      }
      const view3={data:pose3d,canvas:panel.querySelector('.motion-canvas[data-dimension="3"]')};
      try{
        const module=window.BoneScene || await import('./bone_scene.js?v=20261028-combined-v1');
        if(generation!==ticket || !panel.isConnected)return;
        view3.interpolateAnatomy=module.interpolateAnatomy;
        view3.mesh=module.createBoneScene(view3.canvas,pose3d);
      }catch(error){
        // A separate canvas permits a 2D projection when WebGL is unavailable.
        const fallback=view3.canvas.cloneNode(false);view3.canvas.replaceWith(fallback);view3.canvas=fallback;
        view3.canvas.dataset.renderer='canvas-fallback';
      }
      views.push(view3);
      ready(views[1]);
      control.querySelector('.motion-orbit').hidden=false;
      control.querySelector('.motion-status').textContent=lang==='ko'?'영상에서 추정한 3D · 실제 거리와 다를 수 있습니다':'Monocular 3D estimate · fixed scene · ground and scale unvalidated';
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
    if(is3d){
      const placeholder=s.panel.querySelector('.motion-placeholder[data-dimension="3"]');
      if(current.available===false){
        placeholder.textContent=current.unavailable_reason==='source_camera_change'?(s.t===copy.ko?'카메라가 전환된 구간은 3D를 제공하지 않습니다.':'3D is unavailable after the source camera changes.'):s.t.unavailable;
        placeholder.hidden=false;view.frameUnavailable=true;
        if(view.mesh)view.mesh.render(current,current,0,s.yaw,s.pitch);
        else{const context=canvas.getContext('2d');context.fillStyle='#14201b';context.fillRect(0,0,canvas.width,canvas.height);}
        return;
      }
      if(view.frameUnavailable){placeholder.hidden=true;view.frameUnavailable=false;}
    }
    if(view.mesh){view.mesh.render(current,next?.available===false?current:next,next?.available===false?0:fraction,s.yaw,s.pitch);return;}
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
    const native=is3d&&current.anatomy&&data.anatomy_parents;
    canvas.dataset.jointLayout=native?'SMPL-24':data.joints.length.toString();
    const points=native?(view.interpolateAnatomy?view.interpolateAnatomy(current.anatomy,next.anatomy,fraction,data.anatomy_parents):current.anatomy).map(p=>p?project(p):null):current.keypoints.map((p,j)=>{
      if(!p || ((p[dim]??1)<.2 && !current.interpolated?.[j]))return null;
      const candidate=next.keypoints[j];
      const q=candidate && ((candidate[dim]??1)>=.2 || next.interpolated?.[j]) ? candidate : null;
      const value=p.slice(0,dim).map((v,k)=>q?v+(q[k]-v)*fraction:v);
      return project(value);
    });
    if(is3d){
      if(data.scene?.floor && data.scene.show_floor!==false){
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
    const skeleton=native?data.anatomy_parents.map((parent,j)=>[parent,j]).filter(([parent])=>parent>=0):data.edges || edges;
    const ordered=skeleton.filter(([a,b])=>points[a]&&points[b]).sort(([a,b],[c,d])=>(points[a][2]+points[b][2])-(points[c][2]+points[d][2]));
    ctx.lineWidth=5;ctx.lineCap='round';
    for(const [a,b] of ordered){
      ctx.strokeStyle=native?({1:1,4:1,7:1,10:1,13:1,16:1,18:1,20:1,22:1}[b]?'#77cde0':'#e99d65'):a%2===1&&a>=5?'#b4f068':'#77cde0';
      ctx.beginPath();ctx.moveTo(points[a][0],points[a][1]);ctx.lineTo(points[b][0],points[b][1]);ctx.stroke();
    }
    for(let j=0;j<points.length;j++)if(points[j]){
      ctx.fillStyle=current.interpolated?.[j]?'#ffc47a':'#eafadc';
      ctx.beginPath();ctx.arc(points[j][0],points[j][1],6,0,2*Math.PI);ctx.fill();
    }
    if(!is3d&&s.highlightKey){const result=window.PitcherExperience?.paintJointHighlight?.(ctx,points,s.highlightKey,w);canvas.dataset.selectedJoint=s.highlightKey;canvas.dataset.highlightVisible=String(result?.visible??false);canvas.dataset.highlightedBones=String(result?.bones??0)}
    if(!points.some(Boolean)){ctx.fillStyle='#c6d4c9';ctx.font='19px sans-serif';ctx.textAlign='center';ctx.fillText(s.t.gap,w/2,h/2);ctx.textAlign='start';}
  }
  function setHighlightedJoint(key){if(state){state.highlightKey=key;const note=state.control?.querySelector('.motion-note');if(note)note.textContent=state.t===copy.ko?'주황색 강조: 선택 관절·연결된 두 뼈 · 옅은 살구색 관절: 보간':'Orange emphasis: selected joint and its two bones · pale amber joints: interpolated';draw()}}
  return {mount,dispose,draw,createClock,setHighlightedJoint};
})();
