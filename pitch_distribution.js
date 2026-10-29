/* Seasonal velocity distributions from the existing observed pitch records. */
window.PitchDistribution=(()=>{
  'use strict';
  const core=window.PitchDistributionCore;
  const state={request:0,players:[],cache:new Map(),model:null,player:'',year:'',pitch:'',hoverPitch:'',trail:true,playing:false,timer:0,frame:0,display:null,target:null,transitioning:false,signature:'',serial:0};
  const ko=()=>document.documentElement.lang==='ko',text=(a,b)=>ko()?a:b;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=n=>Number(n).toLocaleString(ko()?'ko-KR':'en-US');
  const date=s=>s?s.replaceAll('-','.'):'';
  const routePath=()=>location.hash.split('?')[0];
  const reduced=()=>matchMedia('(prefers-reduced-motion:reduce)').matches;
  const href=hash=>{const u=new URL(location.href);u.search='';u.hash=hash;return u.pathname+u.hash;};
  const title=()=>text('시즌별 구속 분포','Pitch velocity over the seasons');
  const playerLabel=p=>p.name+(state.players.some(q=>q.code!==p.code&&q.name===p.name)?' · '+p.code:'');
  const current=()=>state.model?.years.find(y=>y.year===state.year);
  const index=()=>state.model.years.findIndex(y=>y.year===state.year);
  function fragment(){const p=new URLSearchParams({player:state.player,year:state.year});if(state.pitch)p.set('pitch',state.pitch);if(!state.trail)p.set('trail','0');return '#visuals/distribution?'+p;}
  function sync(push=false){const next=href(fragment());if(location.hash!==fragment()||location.search)history[push?'pushState':'replaceState'](null,'',next);state.signature=fragment()+'|'+document.documentElement.lang;}
  function card(){
    const curves=[['#df704b','M80 226 C220 226 300 210 345 120 S410 40 450 126 S490 225 550 226'],['#5889b2','M80 226 C130 226 165 90 210 100 S280 225 550 226'],['#c09c36','M80 226 C175 226 250 158 290 164 S360 226 550 226']];
    return `<a class="visuals-card" href="${href('#visuals/distribution')}"><div class="visuals-thumbnail"><svg viewBox="0 0 620 310" role="img" aria-label="${title()}"><path d="M70 60V226H550" fill="none" stroke="#bad0c5"/>${curves.map(([c,d])=>`<path d="${d}" fill="${c}" fill-opacity=".15" stroke="${c}" stroke-width="3"/>`).join('')}<text x="310" y="285" text-anchor="middle" fill="#55736a" font-size="16">KBO · PITCH DISTRIBUTIONS</text></svg></div><div class="visuals-card-copy"><span class="visuals-kicker">03 / PITCH DISTRIBUTIONS</span><h2>${title()}</h2><p>${text('구종별 구속이 시즌마다 어떻게 달라졌는지 재생해 보세요.','Play through each season to explore changes in a pitcher’s arsenal.')}</p><span class="visuals-card-link">${text('살펴보기','Explore')} <span aria-hidden="true">↗</span></span></div></a>`;
  }
  async function data(code){
    if(!state.cache.has(code)){
      state.cache.set(code,SeasonVisuals.records(code).then(rows=>core.build(rows,SeasonRecords.index.pitchTypes)).catch(e=>{state.cache.delete(code);throw e}));
      while(state.cache.size>12)state.cache.delete(state.cache.keys().next().value);
    }
    return state.cache.get(code);
  }
  async function route(force=false){
    if(routePath()!=='#visuals/distribution')return;
    const q=new URLSearchParams(location.hash.split('?')[1]||''),ticket=++state.request,host=document.getElementById('visuals-distribution');
    if(!force&&state.signature===location.hash+'|'+document.documentElement.lang&&host.querySelector('#distribution-chart'))return;
    stop();
    try{
      await SeasonRecords.load();
      if(!state.players.length)state.players=SeasonVisuals.players();
      const requested=q.get('player')||q.get('playerId');
      const selected=state.players.find(p=>p.code===requested)||state.players.find(p=>p.name==='류현진')||state.players[0];
      host.innerHTML=`<p role="status">${text('투구 기록을 불러오는 중입니다.','Loading pitch records…')}</p>`;
      const model=await data(selected.code);
      if(ticket!==state.request||routePath()!=='#visuals/distribution')return;
      state.model=model;state.player=selected.code;state.trail=q.get('trail')!=='0';state.hoverPitch='';
      if(!model.years.length){host.innerHTML=`<a class="visuals-back" href="${href('#visuals')}">← Visuals</a><p>${text('이 투수의 구속 기록이 없습니다.','No recorded velocities are available for this pitcher.')}</p>`;return;}
      state.year=model.years.some(y=>y.year===q.get('year'))?q.get('year'):model.years[0].year;
      state.pitch=q.get('pitch')==='all'?'all':model.types.some(t=>t.key===q.get('pitch'))?q.get('pitch'):'';
      state.display=current().series;state.target=state.display;sync();render(host);
      if(!q.has('year')&&model.years.length>1&&!reduced()&&!document.hidden)play();
    }catch(error){
      if(ticket!==state.request||routePath()!=='#visuals/distribution')return;
      host.innerHTML=`<p role="alert">${text('투구 기록을 불러오지 못했습니다.','Could not load pitch records.')}</p><button class="button-outline" id="distribution-retry">${text('다시 불러오기','Try again')}</button>`;
      host.querySelector('button').onclick=()=>route(true);
    }
  }
  function render(host){
    const p=state.players.find(p=>p.code===state.player),m=state.model;
    host.innerHTML=`<a class="visuals-back" href="${href('#visuals')}">← Visuals</a><header class="visuals-heading distribution-heading"><div><span class="visuals-kicker">03 / PITCH DISTRIBUTIONS</span><h1>${title()}</h1><p>${text('구종별 구속과 구사 비율의 변화를 시즌 순서대로 살펴보세요.','Watch how pitch velocity and usage change from season to season.')}</p></div><button class="button-outline" id="distribution-share">${text('링크 복사','Copy link')}</button></header><div class="distribution-picker"><label for="distribution-search">${text('투수 찾기','Find a pitcher')}</label><input id="distribution-search" type="search" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="distribution-players" autocomplete="off" value="${esc(playerLabel(p))}" placeholder="${text('이름으로 검색','Search by name')}"><ul class="season-player-options" id="distribution-players" role="listbox" aria-label="${text('투수 선택','Select a pitcher')}" hidden></ul></div>
      <section class="distribution-panel" aria-labelledby="distribution-player-title"><div class="distribution-chart-head"><div><h2 id="distribution-player-title">${esc(playerLabel(p))}</h2><p id="distribution-scope"></p></div><strong id="distribution-year">${state.year}</strong></div><div class="distribution-plot-wrap"><svg id="distribution-chart" role="img" aria-label="${text('구종별 구속 분포','Pitch velocity distributions')}"></svg><div id="distribution-tooltip" role="tooltip" hidden></div></div><div class="season-legend" id="distribution-legend" aria-label="${text('구종 강조','Highlight a pitch type')}"></div>
      <div class="distribution-timeline"><button id="distribution-play" class="button-primary"></button><button id="distribution-prev" aria-label="${text('이전 시즌','Previous season')}">‹</button><div class="distribution-slider"><label class="sr-only" for="distribution-season">${text('시즌 선택','Choose a season')}</label><input id="distribution-season" type="range" min="0" max="${m.years.length-1}" step="1" value="${index()}"><div class="distribution-year-ticks">${m.years.map((y,i)=>`<button data-distribution-year="${y.year}" style="--tick:${i}">${y.year}</button>`).join('')}</div></div><button id="distribution-next" aria-label="${text('다음 시즌','Next season')}">›</button></div><div class="distribution-options"><label><input id="distribution-trail" type="checkbox" ${state.trail?'checked':''}>${text('이전 시즌 겹쳐보기','Show earlier seasons')}</label><span id="distribution-single"></span></div><p class="distribution-note">${text('가로축은 구속, 세로축은 해당 시즌 전체 구속 기록에서 차지하는 비율입니다.','Velocity is on the horizontal axis; frequency is the share of all recorded velocities in that season.')}</p></section><p id="distribution-status" class="sr-only" role="status"></p>`;
    picker();
    host.querySelector('#distribution-play').onclick=()=>state.playing?stop():play();
    host.querySelector('#distribution-prev').onclick=()=>{stop(false);go(Math.max(0,index()-1),true);};
    host.querySelector('#distribution-next').onclick=()=>{stop(false);go(Math.min(m.years.length-1,index()+1),true);};
    host.querySelector('#distribution-season').oninput=e=>{const value=Number(e.target.value);stop(false);go(value,false);};
    host.querySelectorAll('[data-distribution-year]').forEach(b=>b.onclick=()=>{stop(false);go(m.years.findIndex(y=>y.year===b.dataset.distributionYear),true);});
    host.querySelector('#distribution-trail').onchange=e=>{state.trail=e.target.checked;sync();draw();update();};
    host.querySelector('#distribution-share').onclick=async e=>{try{await navigator.clipboard.writeText(location.href);e.target.textContent=text('복사했습니다.','Copied.');}catch{host.querySelector('#distribution-status').textContent=text('주소창의 링크를 복사해 주세요.','Copy the address from your browser.');}};
    host.querySelector('svg').onpointermove=inspect;host.querySelector('svg').onpointerleave=hideTip;
    host.querySelector('.distribution-options').insertAdjacentHTML('beforeend',`<a id="distribution-movement-link" class="visuals-back">${text('무브먼트 변화 보기','View movement changes')} ↗</a>`);
    legend();draw();update();
  }
  function picker(){
    SeasonVisuals.picker({prefix:'distribution',players:state.players,current:()=>state.player,label:playerLabel,choose:async code=>{stop();history.pushState(null,'',href('#visuals/distribution?'+new URLSearchParams({player:code})));await route(true);}});
  }
  function update(){
    if(routePath()!=='#visuals/distribution'||!document.getElementById('distribution-play')||!state.model)return;
    const y=current(),i=index(),single=state.model.years.length===1;
    document.getElementById('distribution-year').textContent=state.year;
    document.getElementById('distribution-scope').textContent=`${date(y.dateStart)}–${date(y.dateEnd)} · ${text('구속 기록','Recorded velocities')} ${fmt(y.n)}${text('구',' pitches')}${y.missing?text(` · 구속 누락 ${fmt(y.missing)}구`,` · ${fmt(y.missing)} missing`):''}`;
    const play=document.getElementById('distribution-play');play.disabled=single;play.textContent=state.playing?text('Ⅱ 일시정지','Ⅱ Pause'):i===state.model.years.length-1?text('↻ 다시 재생','↻ Replay'):text('▶ 재생','▶ Play');play.setAttribute('aria-pressed',String(state.playing));
    document.getElementById('distribution-prev').disabled=i===0;document.getElementById('distribution-next').disabled=i===state.model.years.length-1;
    const slider=document.getElementById('distribution-season');slider.value=i;slider.disabled=single;slider.setAttribute('aria-valuetext',text(`${state.year}년`,state.year));
    document.getElementById('distribution-single').textContent=single?text('구속 기록이 있는 시즌은 한 시즌입니다.','Only one season has recorded velocities.') : '';
    document.getElementById('distribution-movement-link').href=href('#visuals/movement?'+new URLSearchParams({player:state.player,year:state.year,...(state.pitch&&state.pitch!=='all'?{pitch:state.pitch}:{}),...(!state.trail?{trail:'0'}:{})}));
    document.querySelectorAll('[data-distribution-year]').forEach(b=>b.setAttribute('aria-current',b.dataset.distributionYear===state.year?'true':'false'));
    document.getElementById('distribution-status').textContent=text(`${state.year}년, 구속 기록 ${fmt(y.n)}구`,`${state.year}, ${fmt(y.n)} recorded velocities`);
  }
  function legend(){
    const y=current(),host=document.getElementById('distribution-legend');if(!host)return;
    const items=[{key:'all',name:{ko:'전체 투구',en:'All pitches'},color:'#173c36'},...state.model.types];
    host.innerHTML=items.map(t=>{const g=y.groups.get(t.key);return `<button data-distribution-type="${t.key}" aria-pressed="${state.pitch===t.key}" ${t.key!=='all'&&!g?'class="is-absent"':''}><i style="background:${t.color}" ${t.key==='all'?'class="is-total"':''}></i><span>${esc(t.name[ko()?'ko':'en'])}</span><strong>${t.key==='all'?'100%':g?(100*g.n/y.n).toFixed(1)+'%':'—'}</strong></button>`;}).join('');
    host.querySelectorAll('button').forEach(b=>{
      b.onpointerenter=e=>{if(e.pointerType==='mouse'){state.hoverPitch=b.dataset.distributionType;emphasize();}};b.onpointerleave=()=>{state.hoverPitch='';emphasize();};
      b.onfocus=()=>{state.hoverPitch=b.dataset.distributionType;emphasize();};b.onblur=()=>{state.hoverPitch='';emphasize();};
      b.onclick=()=>{state.pitch=state.pitch===b.dataset.distributionType?'':b.dataset.distributionType;host.querySelectorAll('button').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.distributionType===state.pitch)));sync();emphasize();update();};
    });
  }
  function draw(){
    const svg=document.getElementById('distribution-chart');if(!svg||!state.model||routePath()!=='#visuals/distribution')return;
    const m=state.model,width=Math.max(240,svg.clientWidth),height=svg.clientHeight,left=width<500?46:62,right=18,top=24,bottom=58,w=width-left-right,h=height-top-bottom;
    const x=v=>left+(v-m.xDomain[0])/(m.xDomain[1]-m.xDomain[0])*w,y=v=>top+h-v/m.yMax*h;
    state.geometry={width,height,left,right,top,bottom,w,h,x,y};
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const xStep=width<500?10:5,xTicks=[];for(let v=Math.ceil(m.xDomain[0]/xStep)*xStep;v<=m.xDomain[1];v+=xStep)xTicks.push(v);
    const yTicks=[];for(let v=0;v<=m.yMax+.001;v+=m.yStep)yTicks.push(v);
    const grid=yTicks.map(v=>`<line x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-10}" y="${y(v)+5}" text-anchor="end">${v}%</text>`).join('')+xTicks.map(v=>`<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${y(0)}"/><text x="${x(v)}" y="${height-30}" text-anchor="middle">${v}</text>`).join('');
    const types=m.types,paths=types.map(t=>`<path class="distribution-area" data-curve="${t.key}" fill="${t.color}" stroke="${t.color}"/>`).join('')+`<path class="distribution-total" data-curve="all"/>`;
    const traces=state.trail?m.years.slice(0,index()).map(yr=>`<g data-trail-year="${yr.year}">${types.map(t=>`<path d="${core.path(yr.series[t.key],m.bins,x,y,true)}" fill="${t.color}" stroke="${t.color}"/>`).join('')}</g>`).join(''):'';
    svg.innerHTML=`<title>${esc(title())} · ${state.year}</title><g class="distribution-grid">${grid}</g><text class="distribution-axis-label" x="${left+w/2}" y="${height-5}" text-anchor="middle">${text('구속','Velocity')} · km/h</text><g class="distribution-traces" aria-hidden="true">${traces}</g><g id="distribution-curves">${paths}</g><g id="distribution-cursor" hidden><line y1="${top}" y2="${y(0)}"/></g>`;
    paint(state.display||current().series);emphasize();
  }
  function paint(series){
    const g=state.geometry;if(!g)return;
    document.querySelectorAll('#distribution-curves [data-curve]').forEach(p=>p.setAttribute('d',core.path(series[p.dataset.curve],state.model.bins,g.x,g.y,p.dataset.curve!=='all')));
  }
  function emphasize(){
    const key=state.hoverPitch||state.pitch;
    document.querySelectorAll('#distribution-curves [data-curve]').forEach(p=>{p.classList.toggle('is-muted',!!key&&p.dataset.curve!==key);p.classList.toggle('is-emphasized',p.dataset.curve===key);});
    document.querySelectorAll('[data-distribution-type]').forEach(b=>b.classList.toggle('is-emphasized',b.dataset.distributionType===key));
  }
  function finish(){
    cancelAnimationFrame(state.frame);state.frame=0;
    if(state.target){state.display=state.target;paint(state.display);}
    state.transitioning=false;document.getElementById('distribution-chart')?.setAttribute('aria-busy','false');
  }
  function stop(settle=true){clearTimeout(state.timer);state.timer=0;state.playing=false;if(settle)finish();else{cancelAnimationFrame(state.frame);state.frame=0;state.transitioning=false;}state.hoverPitch='';hideTip();emphasize();update();}
  function go(i,push=false){
    i=Math.max(0,Math.min(state.model.years.length-1,i));
    const from=state.display||current().series;cancelAnimationFrame(state.frame);hideTip();
    state.year=state.model.years[i].year;state.target=current().series;sync(push);update();legend();draw();
    const duration=reduced()?0:650,started=performance.now(),token=++state.serial;
    state.transitioning=!!duration;document.getElementById('distribution-chart').setAttribute('aria-busy',String(!!duration));
    function step(now){
      if(token!==state.serial||routePath()!=='#visuals/distribution')return;
      const t=duration?Math.min(1,(now-started)/duration):1,e=t*t*(3-2*t);
      state.display=core.blend(from,state.target,e);paint(state.display);
      if(t<1)state.frame=requestAnimationFrame(step);else finish();
    }
    if(duration)state.frame=requestAnimationFrame(step);else step(started);
  }
  function play(){
    if(!state.model||state.model.years.length<2)return;
    if(index()===state.model.years.length-1)go(0,false);
    state.playing=true;update();
    function tick(){
      if(!state.playing||routePath()!=='#visuals/distribution'||document.hidden){stop();return;}
      if(index()>=state.model.years.length-1){stop();return;}
      go(index()+1,false);
      state.timer=setTimeout(tick,2300);
    }
    clearTimeout(state.timer);state.timer=setTimeout(tick,1800);
  }
  function hideTip(){const tip=document.getElementById('distribution-tooltip');if(tip)tip.hidden=true;document.getElementById('distribution-cursor')?.setAttribute('hidden','');}
  function inspect(e){
    if(state.transitioning||!state.geometry)return;
    const svg=document.getElementById('distribution-chart'),g=state.geometry,rect=svg.getBoundingClientRect(),px=e.clientX-rect.left;
    if(px<g.left||px>g.width-g.right){hideTip();return;}
    const speed=state.model.xDomain[0]+(px-g.left)/g.w*(state.model.xDomain[1]-state.model.xDomain[0]),bin=Math.max(state.model.bins[0],Math.min(state.model.bins.at(-1),Math.floor(speed))),yr=current();
    const items=state.model.types.map(t=>({...t,...core.at(yr,t.key,bin)})).filter(t=>t.count).sort((a,b)=>b.count-a.count);
    const total=core.at(yr,'all',bin),tip=document.getElementById('distribution-tooltip');
    tip.innerHTML=`<strong>${bin}–${bin+1} km/h</strong><span>${yr.year} · ${fmt(total.count)}${text('구',' pitches')} · ${total.percent.toFixed(1)}%</span>${items.map(t=>`<p><i style="background:${t.color}"></i><span>${esc(t.name[ko()?'ko':'en'])}</span><b>${fmt(t.count)}${text('구',' pitches')} · ${t.percent.toFixed(1)}%</b></p>`).join('')}`;
    tip.hidden=false;tip.style.left=Math.max(8,Math.min(g.width-tip.offsetWidth-8,px+16))+'px';tip.style.top='12px';
    const cursor=document.getElementById('distribution-cursor');cursor.removeAttribute('hidden');cursor.querySelector('line').setAttribute('x1',g.x(bin+.5));cursor.querySelector('line').setAttribute('x2',g.x(bin+.5));
  }
  let resizeTimer;
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(routePath()==='#visuals/distribution'&&state.model){finish();draw();}},120);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  matchMedia('(prefers-reduced-motion:reduce)').addEventListener('change',e=>{if(e.matches)stop();});
  return {card,route,stop,state,go};
})();
