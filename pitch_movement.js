/* Seasonal HB/IVB clouds, paired mean changes and observed-year playback. */
window.PitchMovement=(()=>{
  'use strict';
  const core=window.PitchMovementCore;
  const state={request:0,players:[],cache:new Map(),model:null,player:'',year:'',base:'previous',pitch:'',hover:'',trail:true,playing:false,timer:0,frame:0,display:null,target:null,transitioning:false,signature:''};
  const ko=()=>document.documentElement.lang==='ko',text=(a,b)=>ko()?a:b;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=n=>Number(n).toLocaleString(ko()?'ko-KR':'en-US'),date=s=>s?s.replaceAll('-','.'):'';
  const signed=v=>v===null?'—':Math.abs(v)<.05?'0.0':(v>0?'+':'−')+Math.abs(v).toFixed(1);
  const value=v=>v===null?'—':v.toFixed(1);
  const path=()=>location.hash.split('?')[0],reduced=()=>matchMedia('(prefers-reduced-motion:reduce)').matches;
  const title=()=>text('시즌별 무브먼트 변화','Pitch movement over the seasons');
  const label=p=>p.name+(state.players.some(q=>q.code!==p.code&&q.name===p.name)?' · '+p.code:'');
  const href=hash=>{const u=new URL(location.href);u.search='';u.hash=hash;return u.pathname+u.hash;};
  const now=()=>state.model.years.find(y=>y.year===state.year),idx=()=>state.model.years.findIndex(y=>y.year===state.year);
  const base=()=>core.baseline(state.model,state.year,state.base);
  const visibleTypes=()=>state.model.types.filter(t=>!state.pitch||t.key===state.pitch);
  const typeName=t=>t.name[ko()?'ko':'en'];
  function fragment(){const p=new URLSearchParams({player:state.player,year:state.year});if(state.pitch)p.set('pitch',state.pitch);if(state.base!=='previous')p.set('base',state.base);if(!state.trail)p.set('trail','0');return '#visuals/movement?'+p;}
  function sync(push=false){if(location.hash!==fragment()||location.search)history[push?'pushState':'replaceState'](null,'',href(fragment()));state.signature=fragment()+'|'+document.documentElement.lang;}
  function card(){
    return `<a class="visuals-card" href="${href('#visuals/movement')}"><div class="visuals-thumbnail"><svg viewBox="0 0 620 310" role="img" aria-label="${title()}"><path d="M110 142H510M310 40V245" stroke="#bad0c5"/>${[[210,90,235,70,'#df704b'],[380,150,405,135,'#5889b2'],[270,200,250,212,'#c09c36']].map(([x,y,a,b,c])=>`<path d="M${x} ${y}L${a} ${b}" stroke="${c}" stroke-width="3"/><circle cx="${x}" cy="${y}" r="18" fill="none" stroke="${c}" stroke-dasharray="4 3" opacity=".5"/><circle cx="${a}" cy="${b}" r="20" fill="${c}" fill-opacity=".15" stroke="${c}" stroke-width="3"/>`).join('')}<text x="310" y="285" text-anchor="middle" fill="#55736a" font-size="16">KBO · MOVEMENT EVOLUTION</text></svg></div><div class="visuals-card-copy"><span class="visuals-kicker">04 / MOVEMENT EVOLUTION</span><h2>${title()}</h2><p>${text('같은 투수의 구종이 시즌마다 얼마나 달라졌는지 비교해 보세요.','Compare how a pitcher’s pitch shapes change across seasons.')}</p><span class="visuals-card-link">${text('살펴보기','Explore')} <span aria-hidden="true">↗</span></span></div></a>`;
  }
  async function data(code){
    if(!state.cache.has(code)){state.cache.set(code,SeasonVisuals.records(code).then(rows=>core.build(rows,SeasonRecords.index.pitchTypes)).catch(e=>{state.cache.delete(code);throw e}));while(state.cache.size>12)state.cache.delete(state.cache.keys().next().value);}
    return state.cache.get(code);
  }
  async function route(force=false){
    if(path()!=='#visuals/movement')return;
    const ticket=++state.request,host=document.getElementById('visuals-movement'),q=new URLSearchParams(location.hash.split('?')[1]||'');
    if(!force&&state.signature===location.hash+'|'+document.documentElement.lang&&host.querySelector('#movement-chart'))return;
    stop();
    try{
      await SeasonRecords.load();if(!state.players.length)state.players=SeasonVisuals.players();
      const p=state.players.find(p=>p.code===q.get('player'))||state.players.find(p=>p.name==='류현진')||state.players[0];
      host.innerHTML=`<p role="status">${text('투구 기록을 불러오는 중입니다.','Loading pitch records…')}</p>`;
      const model=await data(p.code);if(ticket!==state.request||path()!=='#visuals/movement')return;
      state.model=model;state.player=p.code;state.hover='';
      if(!model.years.length){host.innerHTML=`<a class="visuals-back" href="${href('#visuals')}">← Visuals</a><p>${text('이 투수의 무브먼트 기록이 없습니다.','No movement records are available for this pitcher.')}</p>`;return;}
      state.year=model.years.some(y=>y.year===q.get('year'))?q.get('year'):model.years[0].year;
      state.base=model.years.some(y=>y.year===q.get('base'))?q.get('base'):'previous';state.pitch=model.types.some(t=>t.key===q.get('pitch'))?q.get('pitch'):'';state.trail=q.get('trail')!=='0';
      state.display=state.target=core.snapshot(now());sync();render(host);
      if(!q.has('year')&&model.years.length>1&&!reduced()&&!document.hidden)play();
    }catch(e){if(ticket!==state.request||path()!=='#visuals/movement')return;host.innerHTML=`<p role="alert">${text('투구 기록을 불러오지 못했습니다.','Could not load pitch records.')}</p><button class="button-outline" id="movement-retry">${text('다시 불러오기','Try again')}</button>`;host.querySelector('button').onclick=()=>route(true);}
  }
  function render(host){
    const p=state.players.find(p=>p.code===state.player),m=state.model;
    host.innerHTML=`<a class="visuals-back" href="${href('#visuals')}">← Visuals</a><header class="visuals-heading movement-heading"><div><span class="visuals-kicker">04 / MOVEMENT EVOLUTION</span><h1>${title()}</h1><p>${text('같은 구종의 움직임이 시즌마다 어떻게 달라졌는지 살펴보세요.','Explore how the movement of each pitch type changes over time.')}</p></div><button class="button-outline" id="movement-share">${text('링크 복사','Copy link')}</button></header><div class="distribution-picker"><label for="movement-search">${text('투수 찾기','Find a pitcher')}</label><input id="movement-search" type="search" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="movement-players" autocomplete="off" value="${esc(label(p))}"><ul class="season-player-options" id="movement-players" role="listbox" aria-label="${text('투수 선택','Select a pitcher')}" hidden></ul></div>
      <div class="movement-workspace"><section class="distribution-panel movement-panel" aria-labelledby="movement-player-title"><div class="distribution-chart-head"><div><h2 id="movement-player-title">${esc(label(p))}</h2><p id="movement-scope"></p></div><strong id="movement-year">${state.year}</strong></div><div class="movement-plot-wrap"><canvas id="movement-cloud" aria-hidden="true"></canvas><svg id="movement-chart" role="group" aria-label="${text('HB·IVB 무브먼트 지도','HB and IVB movement map')}"></svg><p class="movement-empty" id="movement-empty" hidden></p><div id="movement-tooltip" role="tooltip" hidden></div></div><div class="season-legend" id="movement-legend" aria-label="${text('구종 선택과 확대','Select and zoom a pitch type')}"></div><div class="distribution-timeline"><button id="movement-play" class="button-primary"></button><button id="movement-prev" aria-label="${text('이전 시즌','Previous season')}">‹</button><div class="distribution-slider"><label class="sr-only" for="movement-season">${text('시즌 선택','Season')}</label><input id="movement-season" type="range" min="0" max="${m.years.length-1}" step="1"><div class="distribution-year-ticks">${m.years.map(y=>`<button data-movement-year="${y.year}">${y.year}</button>`).join('')}</div></div><button id="movement-next" aria-label="${text('다음 시즌','Next season')}">›</button></div><div class="movement-options"><label>${text('비교 시즌','Compare with')}<select id="movement-base"></select></label><label><input id="movement-trail" type="checkbox" ${state.trail?'checked':''}>${text('이전 시즌 평균 보기','Show earlier means')}</label></div><p class="distribution-note">${text('점은 실제 투구, 큰 원은 구종별 평균입니다. 화살표는 비교 시즌에서 현재 시즌으로의 변화입니다. 구종을 고르면 확대해서 볼 수 있습니다.','Dots are observed pitches; large circles are pitch-type means. Arrows show changes from the comparison season. Select a pitch type to zoom in.')}</p></section><aside class="movement-summary"><h2 id="movement-compare-title"></h2><div id="movement-cards"></div><a id="movement-velocity-link" class="visuals-back">${text('구속 분포 보기','View velocity distributions')} ↗</a></aside></div><p id="movement-status" class="sr-only" role="status"></p>`;
    SeasonVisuals.picker({prefix:'movement',players:state.players,current:()=>state.player,label,choose:async code=>{stop();history.pushState(null,'',href('#visuals/movement?'+new URLSearchParams({player:code})));await route(true);}});
    host.querySelector('#movement-play').onclick=()=>state.playing?stop():play();
    host.querySelector('#movement-prev').onclick=()=>{stop(false);go(idx()-1,true);};host.querySelector('#movement-next').onclick=()=>{stop(false);go(idx()+1,true);};
    host.querySelector('#movement-season').oninput=e=>{const n=Number(e.target.value);stop(false);go(n,false);};
    host.querySelectorAll('[data-movement-year]').forEach(b=>b.onclick=()=>{stop(false);go(m.years.findIndex(y=>y.year===b.dataset.movementYear),true);});
    host.querySelector('#movement-base').onchange=e=>{state.base=e.target.value;stop();sync();update();draw();cards();};
    host.querySelector('#movement-trail').onchange=e=>{state.trail=e.target.checked;sync();draw();};
    host.querySelector('#movement-share').onclick=async e=>{try{await navigator.clipboard.writeText(location.href);e.target.textContent=text('복사했습니다.','Copied.');}catch{host.querySelector('#movement-status').textContent=text('주소창의 링크를 복사해 주세요.','Copy the link from the address bar.');}};
    const svg=host.querySelector('#movement-chart');svg.onpointermove=e=>{if(e.pointerType!=='touch')inspect(e);};svg.onpointerleave=clearHover;svg.onpointercancel=clearHover;
    svg.onclick=e=>{if(e.detail===0)return;const rect=svg.getBoundingClientRect(),hit=meanHit(e.clientX-rect.left,e.clientY-rect.top,22);if(hit)choosePitch(hit.key);};
    svg.onkeydown=e=>{if(e.key==='Escape'){clearHover();e.target.blur();}};
    update();legend();draw();cards();
  }
  function update(){
    if(path()!=='#visuals/movement'||!state.model||!document.getElementById('movement-year'))return;
    const y=now(),i=idx(),one=state.model.years.length===1;
    document.getElementById('movement-year').textContent=state.year;
    document.getElementById('movement-scope').textContent=`${date(y.dateStart)}–${date(y.dateEnd)} · ${fmt(y.n)}${text('구',' pitches')}${y.missing?text(` · 무브먼트 없음 ${fmt(y.missing)}구`,` · ${fmt(y.missing)} missing`):''}`;
    const b=document.getElementById('movement-play');b.disabled=one;b.textContent=state.playing?text('Ⅱ 일시정지','Ⅱ Pause'):i===state.model.years.length-1?text('↻ 다시 재생','↻ Replay'):text('▶ 재생','▶ Play');b.setAttribute('aria-pressed',String(state.playing));
    document.getElementById('movement-prev').disabled=i===0;document.getElementById('movement-next').disabled=i===state.model.years.length-1;
    const slider=document.getElementById('movement-season');slider.value=i;slider.disabled=one;slider.setAttribute('aria-valuetext',text(`${state.year}년`,state.year));
    document.querySelectorAll('[data-movement-year]').forEach(b=>b.setAttribute('aria-current',b.dataset.movementYear===state.year?'true':'false'));
    const select=document.getElementById('movement-base');select.innerHTML=`<option value="previous">${text('직전 시즌','Previous available season')}</option>`+state.model.years.map(y=>`<option value="${y.year}" ${y.year===state.year?'disabled':''}>${y.year}</option>`).join('');select.value=state.base;
    document.getElementById('movement-velocity-link').href=href('#visuals/distribution?'+new URLSearchParams({player:state.player,year:state.year,...(state.pitch?{pitch:state.pitch}:{})}));
    document.getElementById('movement-status').textContent=text(`${state.year}년, 무브먼트 ${fmt(y.n)}구`,`${state.year}, ${fmt(y.n)} movement records`);
  }
  function legend(){
    const host=document.getElementById('movement-legend');host.innerHTML=`<button data-movement-type="" aria-pressed="${!state.pitch}">${text('전체 구종','All pitch types')}</button>`+state.model.types.map(t=>`<button data-movement-type="${esc(t.key)}" aria-pressed="${state.pitch===t.key}"><i style="background:${t.color}"></i>${esc(typeName(t))}</button>`).join('');
    host.querySelectorAll('button').forEach(b=>{b.onclick=()=>choosePitch(b.dataset.movementType);b.onpointerenter=e=>{if(e.pointerType==='mouse')hover(b.dataset.movementType);};b.onpointerleave=clearHover;b.onfocus=()=>hover(b.dataset.movementType);b.onblur=clearHover;});
  }
  function choosePitch(key){const focused=document.activeElement?.dataset;const button=focused?.movementType,mean=focused?.movementMean;stop();state.pitch=key===state.pitch?'':key;sync();legend();draw();cards();update();if(button!==undefined)document.querySelector(`[data-movement-type="${CSS.escape(button)}"]`)?.focus({preventScroll:true});else if(mean)document.querySelector(`[data-movement-mean="${CSS.escape(mean)}"]`)?.focus({preventScroll:true});}
  function cards(){
    const y=now(),ref=base();document.getElementById('movement-compare-title').textContent=ref?text(`${ref.year}년 대비 변화`,`Changes from ${ref.year}`):text('구종별 평균','Pitch-type means');
    document.getElementById('movement-cards').innerHTML=visibleTypes().map(t=>{
      const c=core.compare(y,ref,t.key),g=c.current;
      const count=ref?`${ref.year} ${fmt(c.base?.n||0)}${text('구',' pitches')} → ${y.year} ${fmt(g?.n||0)}${text('구',' pitches')}`:`${y.year} · ${fmt(g?.n||0)}${text('구',' pitches')}`;
      const speedScope=g&&(g.speedN!==g.n||c.base&&c.base.speedN!==c.base.n)?`<p class="movement-card-sample">${text('구속 표본','Velocity sample')}: ${ref?`${fmt(c.base?.speedN||0)} → `:''}${fmt(g.speedN)}${text('구',' pitches')}</p>`:'';
      const note=!g?text('이 시즌에는 기록이 없습니다.','No records in this season.'):g.n<core.MINIMUM?text('평균 비교에 필요한 표본이 부족합니다.','Too few pitches for mean comparison.'):ref&&!c.base?text('비교 시즌에는 기록이 없습니다.','No records in the comparison season.'):c.base&&c.base.n<core.MINIMUM?text('비교 시즌의 표본이 부족합니다.','Too few pitches in the comparison season.') : '';
      return `<article class="movement-card" data-movement-card="${esc(t.key)}"><h3><i style="background:${t.color}"></i>${esc(typeName(t))}</h3><p class="movement-card-sample">${count}</p><div class="movement-card-values"><div><span>HB · cm</span><strong>${value(g?.hb??null)}</strong><small data-delta-hb>${c.status==='ok'?'Δ '+signed(c.hb):'—'}</small></div><div><span>IVB · cm</span><strong>${value(g?.ivb??null)}</strong><small data-delta-ivb>${c.status==='ok'?'Δ '+signed(c.ivb):'—'}</small></div></div><p class="movement-card-speed">${text('구속','Velocity')} <b>${value(g?.speed??null)}</b> km/h <span data-delta-speed>${c.speed===null?'':'(Δ '+signed(c.speed)+')'}</span></p>${speedScope}${note?`<p class="movement-card-note">${note}</p>`:''}</article>`;
    }).join('');
  }
  function draw(){
    const svg=document.getElementById('movement-chart');if(!svg||path()!=='#visuals/movement')return;
    const focused=document.activeElement?.dataset?.movementMean,yNow=now(),ref=base(),domain=state.model.domains[state.pitch]||state.model.domain;
    const width=Math.max(230,svg.clientWidth),height=svg.clientHeight,size=Math.min(width-88,height-100),left=(width-size)/2+12,top=(height-size)/2-12;
    const x=v=>left+(v-domain.x[0])/(domain.x[1]-domain.x[0])*size,y=v=>top+size-(v-domain.y[0])/(domain.y[1]-domain.y[0])*size;
    state.geometry={width,height,size,left,top,x,y,domain};svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const span=domain.x[1]-domain.x[0],step=span<=40?5:span<=140?20:Math.ceil(span/6/10)*10;
    const xt=[],yt=[];for(let n=Math.ceil(domain.x[0]/step)*step;n<=domain.x[1];n+=step)xt.push(n);for(let n=Math.ceil(domain.y[0]/step)*step;n<=domain.y[1];n+=step)yt.push(n);
    const grid=xt.map(n=>`<line x1="${x(n)}" x2="${x(n)}" y1="${top}" y2="${top+size}" class="${n===0?'is-zero':''}"/><text x="${x(n)}" y="${top+size+23}" text-anchor="middle">${n}</text>`).join('')+yt.map(n=>`<line x1="${left}" x2="${left+size}" y1="${y(n)}" y2="${y(n)}" class="${n===0?'is-zero':''}"/><text x="${left-10}" y="${y(n)+5}" text-anchor="end">${n}</text>`).join('');
    const history=state.trail?visibleTypes().map(t=>{
      const parts=[];let previous=null;
      for(const yr of state.model.years.slice(0,idx()+1)){
        const g=yr.groups.get(t.key);if(g?.hb==null){previous=null;continue;}
        if(previous)parts.push(`<path d="M${x(previous.hb)} ${y(previous.ivb)}L${x(g.hb)} ${y(g.ivb)}"/>`);
        if(yr.year!==state.year)parts.push(`<circle cx="${x(g.hb)}" cy="${y(g.ivb)}" r="4"/>`);
        previous=g;
      }return `<g data-movement-history="${esc(t.key)}" stroke="${t.color}">${parts.join('')}</g>`;
    }).join(''):'';
    const references=ref?visibleTypes().map(t=>{const g=ref.groups.get(t.key);return g?.hb!=null?`<circle class="movement-reference" data-reference="${esc(t.key)}" cx="${x(g.hb)}" cy="${y(g.ivb)}" r="7" stroke="${t.color}"/>${state.pitch?`<text class="movement-base-year" x="${x(g.hb)-10}" y="${y(g.ivb)-16}" text-anchor="end">${ref.year}</text>`:''}`:'';}).join(''):'';
    svg.innerHTML=`<defs><clipPath id="movement-clip"><rect x="${left}" y="${top}" width="${size}" height="${size}"/></clipPath></defs><g class="movement-grid">${grid}</g><text class="movement-axis" x="${left+size/2}" y="${top+size+49}" text-anchor="middle">HB · cm</text><text class="movement-axis" transform="translate(${left-38} ${top+size/2}) rotate(-90)" text-anchor="middle">IVB · cm</text><g class="movement-history" clip-path="url(#movement-clip)">${history}</g><g clip-path="url(#movement-clip)">${references}<g id="movement-means">${visibleTypes().map(t=>`<g data-movement-mean="${esc(t.key)}" role="button" tabindex="0" aria-label="${esc(typeName(t))}"><circle class="movement-mean-halo" r="12"/><circle r="7" fill="${t.color}" fill-opacity=".3" stroke="${t.color}" stroke-width="3"/></g>`).join('')}</g><g id="movement-arrows"></g></g><g id="movement-mean-labels" aria-hidden="true"></g>`;
    const canvas=document.getElementById('movement-cloud'),dpr=Math.min(2,devicePixelRatio||1);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);state.context=canvas.getContext('2d');state.context.setTransform(dpr,0,0,dpr,0,0);
    const empty=state.pitch&&!yNow.groups.has(state.pitch);
    const notice=document.getElementById('movement-empty');notice.hidden=!empty;notice.textContent=text('이 시즌에는 선택한 구종의 무브먼트 기록이 없습니다.','No movement records for this pitch type in this season.');svg.style.visibility=canvas.style.visibility=empty?'hidden':'visible';
    svg.querySelectorAll('[data-movement-mean]').forEach(el=>{const key=el.dataset.movementMean;el.onfocus=()=>{stop();hover(key);showMean(key);};el.onblur=clearHover;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choosePitch(key);}};el.onclick=e=>{if(e.detail===0)choosePitch(key);};});
    paint();if(focused)svg.querySelector(`[data-movement-mean="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
  }
  function paint(){
    const g=state.geometry,ctx=state.context,view=state.display;if(!g||!ctx||!view||!document.getElementById('movement-means'))return;
    const focus=state.hover||state.pitch,types=visibleTypes();ctx.clearRect(0,0,g.width,g.height);ctx.save();ctx.beginPath();ctx.rect(g.left,g.top,g.size,g.size);ctx.clip();
    for(const yr of state.model.years){const weight=view.weights[yr.year]||0;if(weight<.0001)continue;for(const t of types){const points=yr.groups.get(t.key)?.points;if(!points)continue;ctx.globalAlpha=weight*(focus?(focus===t.key? .24:.035):.17);ctx.fillStyle=t.color;ctx.beginPath();for(const p of points){const x=g.x(p.hb),y=g.y(p.ivb);ctx.moveTo(x+1.7,y);ctx.arc(x,y,1.7,0,Math.PI*2);}ctx.fill();}}
    ctx.restore();
    const ref=base(),arrows=[],labels=[],occupied=[];
    for(const t of types){const m=view.means[t.key],node=document.querySelector(`[data-movement-mean="${CSS.escape(t.key)}"]`),current=now().groups.get(t.key);if(!node)continue;
      node.style.opacity=m?m.alpha*(focus&&focus!==t.key? .2:1):0;node.style.pointerEvents=m&&m.alpha>.9?'auto':'none';node.tabIndex=current?.hb!=null?0:-1;
      if(!m){node.setAttribute('transform','translate(-10000 -10000)');continue;}node.setAttribute('transform',`translate(${g.x(m.hb)} ${g.y(m.ivb)})`);node.classList.toggle('is-hovered',state.hover===t.key);node.setAttribute('aria-label',`${typeName(t)} · HB ${value(current?.hb??null)} cm · IVB ${value(current?.ivb??null)} cm · ${fmt(current?.n||0)}${text('구',' pitches')}`);
      const b=ref?.groups.get(t.key),c=core.compare(now(),ref,t.key);
      if(b?.hb!=null&&c.status==='ok'&&m.alpha>.9){const ax=g.x(b.hb),ay=g.y(b.ivb),bx=g.x(m.hb),by=g.y(m.ivb),dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy);if(len>1){const ux=dx/len,uy=dy/len,head=Math.min(7,len*.6);arrows.push(`<g data-movement-arrow="${esc(t.key)}" data-delta-hb="${c.hb}" data-delta-ivb="${c.ivb}" stroke="${t.color}" fill="${t.color}" opacity="${focus&&focus!==t.key? .15:1}"><path class="movement-arrow-halo" d="M${ax} ${ay}L${bx} ${by}"/><path class="movement-arrow-line" d="M${ax} ${ay}L${bx} ${by}"/><path stroke="none" d="M${bx} ${by}L${bx-ux*head-uy*head*.55} ${by-uy*head+ux*head*.55}L${bx-ux*head+uy*head*.55} ${by-uy*head-ux*head*.55}Z"/></g>`);}}
      if(m.alpha>.8){
        const px=g.x(m.hb),py=g.y(m.ivb),name=typeName(t),w=[...name].length*17+8;
        for(const [x,y]of [[px,py+25],[px,py-18],[px+w/2+16,py+5],[px-w/2-16,py+5]]){
          const box={x:x-w/2,y:y-16,w,h:20};
          if(box.x<g.left||box.x+w>g.left+g.size||box.y<g.top||box.y+20>g.top+g.size||occupied.some(b=>box.x<b.x+b.w+4&&box.x+w+4>b.x&&box.y<b.y+b.h+3&&box.y+23>b.y))continue;
          if(types.some(other=>{const q=view.means[other.key];return other.key!==t.key&&q&&Math.abs(g.x(q.hb)-x)<w/2+10&&Math.abs(g.y(q.ivb)-(y-8))<19;}))continue;
          occupied.push(box);labels.push(`<text x="${x}" y="${y}" text-anchor="middle" opacity="${focus&&focus!==t.key? .2:1}">${esc(name)}</text>`);break;
        }
      }
    }
    document.getElementById('movement-arrows').innerHTML=arrows.join('');document.getElementById('movement-mean-labels').innerHTML=labels.join('');
  }
  function hover(key){if(state.hover===key)return;state.hover=key;paint();document.querySelectorAll('[data-movement-card]').forEach(el=>el.classList.toggle('is-hovered',el.dataset.movementCard===key));}
  function clearHover(){state.hover='';hideTip();paint();document.querySelectorAll('.movement-card.is-hovered').forEach(el=>el.classList.remove('is-hovered'));}
  function hideTip(){const el=document.getElementById('movement-tooltip');if(el)el.hidden=true;}
  function meanHit(x,y,r=18){return visibleTypes().map(t=>({key:t.key,p:now().groups.get(t.key)})).filter(a=>a.p?.hb!=null).map(a=>({...a,d:Math.hypot(state.geometry.x(a.p.hb)-x,state.geometry.y(a.p.ivb)-y)})).filter(a=>a.d<=r).sort((a,b)=>a.d-b.d)[0];}
  function showMean(key){
    if(state.transitioning)return;const t=state.model.types.find(t=>t.key===key),g=now().groups.get(key);if(!t||g?.hb==null)return;
    const c=core.compare(now(),base(),key);tip(`${typeName(t)} · ${state.year}`,`HB ${value(g.hb)} cm · IVB ${value(g.ivb)} cm`,`${fmt(g.n)}${text('구',' pitches')} · ${text('평균 구속','Mean velocity')} ${value(g.speed)} km/h`,c.status==='ok'?`${base().year} → ${state.year}: ΔHB ${signed(c.hb)} cm · ΔIVB ${signed(c.ivb)} cm`:'',state.geometry.x(g.hb),state.geometry.y(g.ivb));
  }
  function tip(name,line,sample,change,x,y){const el=document.getElementById('movement-tooltip');el.innerHTML=`<strong>${esc(name)}</strong><span>${line}</span><span>${sample}</span>${change?`<p>${change}</p>`:''}`;el.hidden=false;el.style.left=Math.max(8,Math.min(state.geometry.width-el.offsetWidth-8,x+18))+'px';el.style.top=Math.max(8,Math.min(state.geometry.height-el.offsetHeight-8,y-el.offsetHeight-18))+'px';}
  function inspect(e){
    if(state.transitioning)return;const rect=e.currentTarget.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,g=state.geometry;if(x<g.left||x>g.left+g.size||y<g.top||y>g.top+g.size){clearHover();return;}
    const mean=meanHit(x,y);if(mean){hover(mean.key);showMean(mean.key);return;}
    let found=null,distance=36;for(const p of now().points){if(state.pitch&&p.key!==state.pitch)continue;const d=(g.x(p.hb)-x)**2+(g.y(p.ivb)-y)**2;if(d<distance){distance=d;found=p;}}
    if(!found){clearHover();return;}const t=state.model.types.find(t=>t.key===found.key);hover(found.key);tip(typeName(t),`HB ${value(found.hb)} cm · IVB ${value(found.ivb)} cm`,`${date(found.date)} · ${value(found.speed)} km/h`,text('실제 투구 1개','One observed pitch'),x,y);
  }
  function finish(){cancelAnimationFrame(state.frame);state.frame=0;if(state.target)state.display=state.target;state.transitioning=false;document.getElementById('movement-chart')?.setAttribute('aria-busy','false');paint();}
  function stop(settle=true){clearTimeout(state.timer);state.timer=0;state.playing=false;state.hover='';hideTip();if(settle)finish();else{cancelAnimationFrame(state.frame);state.frame=0;state.transitioning=false;}update();}
  function go(i,push=false){
    i=Math.max(0,Math.min(state.model.years.length-1,i));const from=state.display;cancelAnimationFrame(state.frame);hideTip();state.hover='';state.year=state.model.years[i].year;state.target=core.snapshot(now());sync(push);update();cards();draw();
    const start=performance.now(),duration=reduced()?0:650;state.transitioning=!!duration;document.getElementById('movement-chart').setAttribute('aria-busy',String(!!duration));
    function step(t){if(path()!=='#visuals/movement')return;const n=duration?Math.min(1,(t-start)/duration):1,e=n*n*(3-2*n);state.display=core.blend(from,state.target,e);paint();if(n<1)state.frame=requestAnimationFrame(step);else finish();}
    if(duration)state.frame=requestAnimationFrame(step);else step(start);
  }
  function play(){if(state.model.years.length<2)return;if(idx()===state.model.years.length-1)go(0);state.playing=true;update();function tick(){if(!state.playing||path()!=='#visuals/movement'||document.hidden){stop();return;}if(idx()===state.model.years.length-1){stop();return;}go(idx()+1);state.timer=setTimeout(tick,2300);}clearTimeout(state.timer);state.timer=setTimeout(tick,1800);}
  let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>{if(path()==='#visuals/movement'&&state.model){finish();hideTip();draw();}},120);});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});matchMedia('(prefers-reduced-motion:reduce)').addEventListener('change',e=>{if(e.matches)stop();});
  return {card,route,stop,state};
})();
