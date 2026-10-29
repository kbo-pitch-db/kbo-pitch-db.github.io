/* League-wide view of the same seasonal pitcher similarity; fixed coordinates. */
window.PitcherMap=(()=>{
  'use strict';
  const URL_DATA=new URL('data/visuals/pitcher-map.json?v=20261030-map1',document.currentScript.src);
  const state={index:null,pending:null,request:0,data:null,layout:null,selected:'',query:'',zoom:1,dx:0,dy:0,positions:[],signature:''};
  const ko=()=>document.documentElement.lang==='ko';
  const text=(a,b)=>ko()?a:b;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=v=>Number(v).toLocaleString(ko()?'ko-KR':'en-US');
  const title=()=>text('투수 유사도 지도','Pitcher similarity map');
  const routePath=()=>location.hash.split('?')[0];
  const label=p=>p.name+(state.data.players.some(q=>q.key!==p.key&&q.name===p.name)?' · '+p.code:'');
  const href=hash=>{const u=new URL(location.href);u.search='';u.hash=hash;return u.pathname+u.hash;};
  const date=s=>s.slice(0,4)+'.'+s.slice(4,6)+'.'+s.slice(6);
  function fragment(year,player='') {const q=new URLSearchParams({year:String(year)});if(player)q.set('player',player);return '#visuals/map?'+q;}
  function navigate(year,player='',replace=false){
    const focus=document.activeElement?.id;
    history[replace?'replaceState':'pushState'](null,'',href(fragment(year,player)));
    return route().then(()=>{if(focus)document.getElementById(focus)?.focus();});
  }
  function card(){
    // Gallery illustration only. The interactive map always uses actual data.
    const clusters=[[180,120],[300,140],[400,200],[245,215],[460,100]];
    const dots=clusters.flatMap(([x,y],j)=>Array.from({length:7},(_,i)=>{
      const angle=i*2.399,r=13+Math.sqrt(i)*13;
      return `<circle cx="${x+Math.cos(angle)*r}" cy="${y+Math.sin(angle)*r}" r="6" fill="${['#187c67','#c85b3b','#5889b2','#8972aa','#c09c36'][j]}"/>`;
    })).join('');
    return `<a class="visuals-card" href="${href('#visuals/map')}"><div class="visuals-thumbnail"><svg viewBox="0 0 620 310" role="img" aria-label="${title()}"><path d="M180 120L300 140L400 200L460 100M300 140L245 215L400 200" stroke="#bad0c5" stroke-width="2" fill="none"/>${dots}<text x="310" y="285" text-anchor="middle" fill="#55736a" font-size="16">KBO · ALL PITCHERS</text></svg></div><div class="visuals-card-copy"><span class="visuals-kicker">02 / PITCHER MAP</span><h2>${title()}</h2><p>${text('전체 투수의 구종과 공의 움직임을 한눈에 비교해 보세요.','Explore the full field of pitcher repertoires and movement.')}</p><span class="visuals-card-link">${text('살펴보기','Explore')} <span aria-hidden="true">↗</span></span></div></a>`;
  }
  async function route(force=false){
    if(routePath()!=='#visuals/map')return;
    const ticket=++state.request,host=document.getElementById('visuals-map');
    const q=new URLSearchParams(location.hash.split('?')[1]||'');
    try{
      if(!state.pending)state.pending=fetch(URL_DATA,{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error('Map unavailable');return r.json()}).catch(e=>{state.pending=null;throw e});
      if(!host.querySelector('#pitcher-map-plot'))host.innerHTML=`<p role="status">${text('투수 지도를 불러오는 중입니다.','Loading the pitcher map…')}</p>`;
      const [index,data]=await Promise.all([state.pending,KBOVisuals.cohort(q.get('year')||'')]);
      if(ticket!==state.request||routePath()!=='#visuals/map')return;
      const year=String(data.season),layout=index.seasons?.[year];
      if(index.schemaVersion!==1||!layout||layout.cohortSha256!==KBOVisuals.state.manifest.seasons[year].sha256||layout.points.length!==data.players.length||new Set(layout.points.map(p=>p.key)).size!==data.players.length||layout.points.some(p=>!data.players.some(q=>q.key===p.key)||!Number.isFinite(p.x)||!Number.isFinite(p.y)))throw Error('Map cohort mismatch');
      state.index=index;state.data=data;state.layout=layout;
      state.selected=data.players.some(p=>p.key===q.get('player'))?q.get('player'):'';
      const canonical=fragment(year,state.selected);
      if(location.hash!==canonical||location.search)history.replaceState(null,'',href(canonical));
      const signature=year+'|'+document.documentElement.lang;
      if(force||signature!==state.signature||!host.querySelector('#pitcher-map-plot')){
        state.query='';state.zoom=1;state.dx=0;state.dy=0;state.signature=signature;render(host);
      }else{draw();detail();}
    }catch(error){
      if(ticket!==state.request||routePath()!=='#visuals/map')return;
      host.innerHTML=`<p role="alert">${text('투수 지도를 불러오지 못했습니다.','Could not load the pitcher map.')}</p><button class="button-outline" id="pitcher-map-retry">${text('다시 불러오기','Try again')}</button>`;
      host.querySelector('button').onclick=()=>{state.pending=null;return route(true);};
    }
  }
  function render(host){
    const d=state.data;
    host.innerHTML=`<a class="visuals-back" href="${href('#visuals')}">← Visuals</a><header class="visuals-heading pitcher-map-heading"><div><span class="visuals-kicker">02 / PITCHER MAP</span><h1>${title()}</h1><p>${text('구종과 공의 움직임이 비슷한 투수들을 함께 살펴보세요.','Explore pitchers with similar repertoires and movement.')}</p></div><button class="button-outline" id="pitcher-map-share">${text('링크 복사','Copy link')}</button></header>
      <div class="pitcher-map-controls"><label for="pitcher-map-season">${text('기록 연도','Season')}<select id="pitcher-map-season">${Object.keys(state.index.seasons).sort().reverse().map(y=>`<option value="${y}" ${Number(y)===d.season?'selected':''}>${y}</option>`).join('')}</select></label><div class="pitcher-map-searchbox"><label for="pitcher-map-search">${text('투수 찾기','Find a pitcher')}</label><input id="pitcher-map-search" type="search" autocomplete="off" placeholder="${text('이름으로 검색','Search by name')}" aria-controls="pitcher-map-results"><div id="pitcher-map-results" aria-live="polite"></div></div><span>${date(d.dateStart)}–${date(d.dateEnd)} · ${text('투수','Pitchers')} <strong id="pitcher-map-count">${d.players.length}${text('명','')}</strong> · ${text('유효 투구','Minimum')} ${fmt(d.minimumPitches)}${text('구 이상',' usable pitches')}</span></div>
      <div class="pitcher-map-workspace"><div class="pitcher-map-chart"><div class="pitcher-map-tools"><span>${text('원을 눌러 투수를 선택하세요.','Select a pitcher on the map.')}</span><div role="group" aria-label="${text('지도 확대와 이동','Map zoom and navigation')}"><button id="pitcher-map-minus" aria-label="${text('축소','Zoom out')}">−</button><button id="pitcher-map-plus" aria-label="${text('확대','Zoom in')}">+</button><button id="pitcher-map-reset">${text('전체 보기','Fit all')}</button></div></div><svg id="pitcher-map-plot" role="group" tabindex="0" aria-label="${title()}" aria-describedby="pitcher-map-help"></svg><p class="sr-only" id="pitcher-map-help">${text('방향키로 투수 사이를 이동하고 Enter로 선택합니다. 더하기와 빼기로 확대하거나 축소하고 Home으로 전체를 봅니다.','Use arrow keys to move between pitchers, Enter to select, plus or minus to zoom, and Home to fit all pitchers.')}</p><div id="pitcher-map-legend"></div><p class="pitcher-map-note">${text('원의 색은 구종 비율입니다. 거리는 유사도를 2차원으로 줄인 근사치이며, 정확한 점수는 선수를 선택하면 볼 수 있습니다.','Ring colours show pitch usage. Distances approximate similarity in two dimensions; select a pitcher for exact scores.')}</p></div>
      <aside class="pitcher-map-sidebar"><div id="pitcher-map-detail" aria-live="polite"></div></aside></div><p id="pitcher-map-announcement" class="sr-only" role="status"></p>`;
    host.querySelector('#pitcher-map-season').onchange=e=>navigate(e.target.value,state.selected);
    host.querySelector('#pitcher-map-search').oninput=e=>{state.query=e.target.value;search();draw();};
    host.querySelector('#pitcher-map-search').onkeydown=e=>{if(e.key==='Escape'){e.target.value='';state.query='';search();draw();}else if(e.key==='Enter'){const matches=matching();if(state.query.trim()&&matches.length===1)select(matches[0].key,true);}};
    host.querySelector('#pitcher-map-plus').onclick=()=>zoom(1.4);
    host.querySelector('#pitcher-map-minus').onclick=()=>zoom(1/1.4);
    host.querySelector('#pitcher-map-reset').onclick=()=>{state.zoom=1;state.dx=state.dy=0;draw();};
    host.querySelector('#pitcher-map-share').onclick=async e=>{try{await navigator.clipboard.writeText(location.href);e.target.textContent=text('복사했습니다.','Copied.');}catch{host.querySelector('#pitcher-map-announcement').textContent=text('주소창의 링크를 복사해 주세요.','Copy the link from the address bar.');}};
    gestures(host.querySelector('svg'));
    search();draw();detail();
  }
  function matching(){const q=state.query.trim().toLowerCase();return state.data.players.filter(p=>(p.name+' '+p.code).toLowerCase().includes(q)).sort((a,b)=>a.name.localeCompare(b.name,'ko')||a.key.localeCompare(b.key));}
  function search(){
    const results=document.getElementById('pitcher-map-results');
    if(!state.query.trim()){results.innerHTML='';return;}
    const matches=matching();
    results.innerHTML=matches.length?`<p>${text('검색 결과','Matches')} ${matches.length}</p><ul>${matches.map(p=>`<li><button data-map-search="${esc(p.key)}">${esc(label(p))}<small>${fmt(p.pitches)}${text('구',' pitches')}</small></button></li>`).join('')}</ul>`:`<p role="status">${text('검색 결과가 없습니다.','No matching pitchers.')}</p>`;
    results.querySelectorAll('button').forEach(b=>b.onclick=()=>select(b.dataset.mapSearch,true));
  }
  function select(key,reveal=false){
    state.selected=key;state.query='';document.getElementById('pitcher-map-search').value='';search();
    if(reveal&&state.zoom>1){const p=state.positions.find(p=>p.key===key);state.dx=state.width/2-state.zoom*p.x;state.dy=state.height/2-state.zoom*p.y;}
    if(location.hash!==fragment(state.data.season,key))history.pushState(null,'',href(fragment(state.data.season,key)));draw();detail();
  }
  function neighbors(){const p=state.data.players.find(p=>p.key===state.selected);return p?AffinityCore.rank(state.data.players,p).slice(0,5):[];}
  function detail(){
    const host=document.getElementById('pitcher-map-detail'),p=state.data.players.find(p=>p.key===state.selected);
    if(!p){host.innerHTML=`<div class="pitcher-map-empty"><h2>${text('투수를 선택해 보세요.','Choose a pitcher.')}</h2><p>${text('지도에서 원을 누르거나 이름을 검색하면 비슷한 투수와 유사도 점수가 나옵니다.','Select a ring or search by name to see similar pitchers and their scores.')}</p></div>`;return;}
    const near=neighbors(),sample=text(`${fmt(p.pitches)}구 · ${fmt(p.games)}경기`,`${fmt(p.pitches)} pitches · ${fmt(p.games)} games`);
    host.innerHTML=`<div class="pitcher-map-player"><span>${text('선택한 투수','Selected pitcher')}</span><h2>${esc(label(p))}</h2><p>${sample}</p><div class="pitcher-map-mix">${p.counts.map((n,i)=>n?`<span style="width:${100*n/p.pitches}%;background:${state.data.pitchTypes[i].color}" title="${esc(state.data.pitchTypes[i].name[ko()?'ko':'en'])} ${(100*n/p.pitches).toFixed(1)}%"></span>`:'').join('')}</div></div><div class="pitcher-map-neighbors"><h3>${text('비슷한 투수','Similar pitchers')}</h3><ol>${near.map(({player:q,score},i)=>`<li><button data-map-neighbor="${esc(q.key)}"><span class="pitcher-map-rank">${i+1}</span><span>${esc(label(q))}</span><strong>${score.toFixed(1)}</strong></button><a href="${href(AffinityCore.fragment({key:p.key,compare:q.key,year:String(state.data.season),threshold:0}))}" aria-label="${esc(label(p)+' · '+label(q)+' '+text('구종별 비교','repertoire comparison'))}">${text('비교','Compare')} ↗</a></li>`).join('')}</ol></div>`;
    host.querySelectorAll('[data-map-neighbor]').forEach(b=>b.onclick=()=>select(b.dataset.mapNeighbor,true));
    document.getElementById('pitcher-map-announcement').textContent=text(`${label(p)} 선택. 비슷한 투수 5명을 표시합니다.`,`${label(p)} selected. Showing five similar pitchers.`);
  }
  function geometry(){
    const svg=document.getElementById('pitcher-map-plot'),width=Math.max(230,svg.clientWidth),height=svg.clientHeight;
    const points=state.layout.points,xs=points.map(p=>p.x),ys=points.map(p=>p.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const scale=Math.min((width-66)/(maxX-minX||1),(height-66)/(maxY-minY||1));
    state.width=width;state.height=height;state.positions=points.map(p=>({...p,x:width/2+(p.x-(minX+maxX)/2)*scale,y:height/2+(p.y-(minY+maxY)/2)*scale}));
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  }
  function draw(){
    const svg=document.getElementById('pitcher-map-plot');if(!svg||routePath()!=='#visuals/map')return;
    const focused=document.activeElement?.dataset?.mapPlayer;geometry();
    const byKey=new Map(state.data.players.map(p=>[p.key,p])),positions=new Map(state.positions.map(p=>[p.key,p]));
    const near=neighbors(),nearKeys=new Set(near.map(p=>p.player.key)),matches=new Set(matching().map(p=>p.key));
    const edges=state.selected?near.map(({player})=>[state.selected,player.key]):[];
    const line=edges.map(([a,b])=>{const p=positions.get(a),q=positions.get(b);return `<line x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}"/>`;}).join('');
    const nodes=state.positions.map(point=>{
      const p=byKey.get(point.key),selected=p.key===state.selected,nearby=nearKeys.has(p.key),dim=state.query.trim()&&!matches.has(p.key),radius=selected?10:nearby?7.5:6;
      let offset=0;const circumference=2*Math.PI*radius;
      const ring=p.counts.map((n,i)=>{if(!n)return '';const length=n/p.pitches*circumference,dash=`<circle r="${radius}" fill="none" stroke="${state.data.pitchTypes[i].color}" stroke-width="${selected?5:4}" stroke-dasharray="${length} ${circumference}" stroke-dashoffset="${-offset}" transform="rotate(-90)"/>`;offset+=length;return dash;}).join('');
      return `<g class="pitcher-map-node${selected?' is-selected':''}${nearby?' is-near':''}${dim?' is-dim':''}" data-map-player="${esc(p.key)}" data-map-x="${point.x}" data-map-y="${point.y}" transform="translate(${point.x} ${point.y})" role="button" tabindex="${p.key===(focused||state.selected||state.positions[0].key)?0:-1}" aria-pressed="${selected}" aria-label="${esc(label(p))} · ${fmt(p.pitches)}${text('구',' pitches')}"><title>${esc(label(p))} · ${fmt(p.pitches)}${text('구',' pitches')}</title><circle class="pitcher-map-focus" r="${radius+5}"/><circle r="${radius}" fill="#fff"/>${ring}</g>`;
    }).join('');
    svg.innerHTML=`<g id="pitcher-map-viewport" transform="translate(${state.dx} ${state.dy}) scale(${state.zoom})"><g class="pitcher-map-edges">${line}</g>${nodes}</g><g id="pitcher-map-labels" aria-hidden="true"></g>`;
    svg.querySelectorAll('[data-map-player]').forEach(g=>{g.onkeydown=e=>keyboard(e,g);g.onclick=e=>{if(e.detail===0)select(g.dataset.mapPlayer);};});
    labels(byKey,nearKeys,matches);
    const used=state.data.pitchTypes.filter((t,i)=>state.data.players.some(p=>p.counts[i]));
    document.getElementById('pitcher-map-legend').innerHTML=used.map(t=>`<span><i style="background:${t.color}"></i>${esc(t.name[ko()?'ko':'en'])}</span>`).join('');
    if(focused)svg.querySelector(`[data-map-player="${CSS.escape(focused)}"]`)?.focus({preventScroll:true});
  }
  function labels(byKey,nearKeys,matches){
    const width=state.width,height=state.height,occupied=[];
    const ordered=[...state.positions].sort((a,b)=>{
      const priority=p=>(p.key===state.selected?1e9:0)+(state.query.trim()&&matches.has(p.key)?1e8:0)+(nearKeys.has(p.key)?1e7:0)+byKey.get(p.key).pitches;
      return priority(b)-priority(a);
    });
    const result=[];
    for(const point of ordered){
      if(state.query.trim()&&!matches.has(point.key)&&point.key!==state.selected)continue;
      const p=byKey.get(point.key),x=point.x*state.zoom+state.dx,y=point.y*state.zoom+state.dy;
      if(x<12||x>width-12||y<12||y>height-12)continue;
      const selected=p.key===state.selected,important=selected||nearKeys.has(p.key),name=label(p),size=selected?17:important?15:state.zoom>=2?15:12;
      const labelWidth=[...name].reduce((n,c)=>n+(c.charCodeAt(0)>255?size:size*.6),0)+8;
      const radius=(selected?10:important?7.5:6)*state.zoom+7;
      const candidates=[[x,y+radius+size],[x,y-radius],[x+radius+labelWidth/2,y+size/3],[x-radius-labelWidth/2,y+size/3]];
      for(const [lx,top]of candidates){
        const box={x:lx-labelWidth/2,y:top-size,w:labelWidth,h:size+6};
        if(box.x<2||box.x+box.w>width-2||box.y<2||box.y+box.h>height-2)continue;
        if(occupied.some(b=>box.x<b.x+b.w+3&&box.x+box.w+3>b.x&&box.y<b.y+b.h+2&&box.y+box.h+2>b.y))continue;
        if(!selected&&state.positions.some(q=>q.key!==p.key&&Math.abs(q.x*state.zoom+state.dx-lx)<labelWidth/2+7*state.zoom&&Math.abs(q.y*state.zoom+state.dy-(top-size/2))<size/2+7*state.zoom))continue;
        occupied.push(box);result.push(`<text x="${lx}" y="${top}" text-anchor="middle" font-size="${size}" class="${important?'is-important':''}" style="pointer-events:none">${esc(name)}</text>`);break;
      }
    }
    // Paint important labels last, on top of the ordinary labels.
    document.getElementById('pitcher-map-labels').innerHTML=result.reverse().join('');
  }
  function keyboard(e,node){
    if(e.key==='Enter'||e.key===' '){e.preventDefault();select(node.dataset.mapPlayer);return;}
    const vectors={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]},direction=vectors[e.key];
    if(!direction)return;e.preventDefault();e.stopPropagation();
    const source=state.positions.find(p=>p.key===node.dataset.mapPlayer);
    const target=state.positions.filter(p=>p.key!==source.key).map(p=>{const dx=p.x-source.x,dy=p.y-source.y,dot=dx*direction[0]+dy*direction[1],cross=Math.abs(dx*direction[1]-dy*direction[0]);return {p,dot,cost:Math.hypot(dx,dy)+cross*2};}).filter(v=>v.dot>0).sort((a,b)=>a.cost-b.cost)[0]?.p;
    if(target){
      const x=target.x*state.zoom+state.dx,y=target.y*state.zoom+state.dy;
      if(x<20||x>state.width-20||y<20||y>state.height-20){state.dx+=state.width/2-x;state.dy+=state.height/2-y;draw();}
      document.querySelectorAll('#pitcher-map-plot [data-map-player]').forEach(g=>g.tabIndex=g.dataset.mapPlayer===target.key?0:-1);document.querySelector(`[data-map-player="${CSS.escape(target.key)}"]`).focus({preventScroll:true});
    }
  }
  function zoom(factor,cx=state.width/2,cy=state.height/2){
    const previous=state.zoom;state.zoom=Math.min(6,Math.max(1,previous*factor));
    state.dx=cx-(cx-state.dx)*state.zoom/previous;state.dy=cy-(cy-state.dy)*state.zoom/previous;
    if(state.zoom===1)state.dx=state.dy=0;draw();
  }
  function gestures(svg){
    const pointers=new Map();let origin=null,moved=false,pinch=null;
    svg.onpointerdown=e=>{
      if(e.button&&e.pointerType==='mouse')return;
      pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});svg.setPointerCapture(e.pointerId);
      if(pointers.size===1){origin={x:e.clientX,y:e.clientY,dx:state.dx,dy:state.dy};moved=false;}
      if(pointers.size===2){const [a,b]=[...pointers.values()];pinch={distance:Math.hypot(a.x-b.x,a.y-b.y),zoom:state.zoom};moved=true;}
    };
    svg.onpointermove=e=>{
      if(!pointers.has(e.pointerId))return;
      pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(pointers.size===2&&pinch){const [a,b]=[...pointers.values()],rect=svg.getBoundingClientRect(),distance=Math.hypot(a.x-b.x,a.y-b.y);zoom(pinch.zoom*distance/pinch.distance/state.zoom,(a.x+b.x)/2-rect.left,(a.y+b.y)/2-rect.top);return;}
      if(!origin)return;
      const dx=e.clientX-origin.x,dy=e.clientY-origin.y;
      if(Math.hypot(dx,dy)>5)moved=true;
      if(moved){state.dx=origin.dx+dx;state.dy=origin.dy+dy;draw();}
    };
    svg.onpointerup=e=>{
      const hadPinch=!!pinch;pointers.delete(e.pointerId);if(pointers.size<2)pinch=null;
      if(!moved&&!hadPinch){const rect=svg.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const closest=state.positions.map(p=>({p,d:Math.hypot(p.x*state.zoom+state.dx-x,p.y*state.zoom+state.dy-y)})).sort((a,b)=>a.d-b.d)[0];if(closest?.d<=18)select(closest.p.key);}
      origin=null;
    };
    svg.onpointercancel=e=>{pointers.delete(e.pointerId);origin=null;pinch=null;moved=true;};
    svg.onwheel=e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();const r=svg.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.004),e.clientX-r.left,e.clientY-r.top);}};
    svg.onkeydown=e=>{
      if(e.target===svg&&e.key.startsWith('Arrow')){const node=svg.querySelector('[tabindex="0"]');if(node)keyboard(e,node);}
      if(['+','=','-','Home'].includes(e.key)){e.preventDefault();if(e.key==='Home'){state.zoom=1;state.dx=state.dy=0;draw();}else zoom(e.key==='-'?1/1.4:1.4);}
    };
  }
  let resized;
  window.addEventListener('resize',()=>{clearTimeout(resized);resized=setTimeout(()=>{if(routePath()==='#visuals/map'&&state.layout){state.zoom=1;state.dx=state.dy=0;draw();}},120);});
  return {card,route,state};
})();
