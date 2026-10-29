/* Visuals load their own cohort; player/motion explorer parameters stay separate. */
window.KBOVisuals = (() => {
  'use strict';
  const dataURL = new URL('data/visuals/affinity.json?v=20261030-arsenal1', document.currentScript.src).href;
  const core = window.AffinityCore;
  const state = {data:null,pending:null,signature:'',compare:'',ranked:[],selected:null,pitch:null};
  const copy = {
    ko: {
      intro:'야구 데이터를 다른 각도에서 살펴보세요.',name:'비슷한 투수 찾기',
      card:'구종 구성과 구속, 공의 움직임이 비슷한 투수를 찾아보세요.',
      open:'살펴보기',back:'← Visuals',title:'비슷한 투수 찾기',
      subtitle:'구종별 구사 비율과 구속·무브먼트 분포를 함께 비교합니다.',
      player:'투수 찾기',choose:'투수 선택',search:'이름으로 검색',noPlayer:'검색 결과가 없습니다.',
      loading:'투수 기록을 불러오는 중입니다.',failed:'기록을 불러오지 못했습니다.',retry:'다시 불러오기',
      near:'비슷한 투수',score:'유사도',threshold:'유사도 기준',compare:'구종별 비교',
      selected:'기준 투수',counterpart:'비교 투수',recenter:'이 투수 중심으로 보기',
      none:'이 기준에 맞는 투수가 없습니다. 유사도 기준을 낮춰 보세요.',
      hint:'투수를 누르면 아래에서 구종과 분포를 비교할 수 있습니다.',
      allScores:'전체 투수의 유사도',distribution:'점 하나가 비교 가능한 투수 한 명입니다.',
      copy:'링크 복사',copied:'링크를 복사했습니다.',copyFailed:'주소창의 링크를 복사해 주세요.',
      method:'비교 기준 · 출처',
      methodText:'같은 구종끼리 구속·HB·IVB 분포를 비교하고, 두 투수의 구사 비율을 반영합니다. 한쪽만 던지는 구종도 점수에 영향을 줍니다. 평균뿐 아니라 분포의 폭과 세 값의 관계를 함께 비교합니다.',
      methodScope:'HB는 포수 시점의 좌우 움직임, IVB는 중력에 의한 낙하를 제외한 수직 움직임입니다. 네이버 궤적으로 계산한 값이며, 확보한 기록을 기준으로 비교합니다. 점수는 실력이나 성적 순위가 아닙니다.',
      methodMath:'구속·HB·IVB가 모두 있는 투구만 사용합니다. 자료의 90% 이상을 사용할 수 있고 유효 투구가 500구 이상인 투수를 포함합니다. 자세한 계산식은 아래 문서에 있습니다.',
      source:'원본 기록',reference:'Savant 원본',math:'계산식과 자료 범위',
      barsNote:'구사 비율은 비교에 사용한 유효 투구를 기준으로 합니다.',
      emptyCompare:'비교할 투수가 없습니다.',graphLabel:'선택한 투수와 비슷한 투수 지도',
      mix:'구종 구성 · 구사 비율',shapes:'구종별 분포',pitch:'비교할 구종',
      axes:['구속','수평 무브먼트 · HB','수직 무브먼트 · IVB'],
      median:'중앙값',missing:'해당 구종 기록 없음',
      graphNote:'같은 구종의 실제 투구 분포입니다. 두 투수에게 같은 축을 사용합니다.'
    },
    en: {
      intro:'A different view of baseball data.',name:'Pitcher similarity',
      card:'Find pitchers with similar repertoires, velocity and movement.',
      open:'Explore',back:'← Visuals',title:'Pitcher similarity',
      subtitle:'Compare pitch usage and joint velocity–movement distributions.',
      player:'Find a pitcher',choose:'Select a pitcher',search:'Search by name',noPlayer:'No matching pitchers.',
      loading:'Loading pitcher records…',failed:'Could not load the records.',retry:'Try again',
      near:'Similar pitchers',score:'Similarity',threshold:'Minimum similarity',compare:'Compare repertoires',
      selected:'Selected pitcher',counterpart:'Comparison',recenter:'Explore this pitcher',
      none:'No pitchers meet this threshold. Try a lower threshold.',
      hint:'Select a pitcher to compare repertoires and distributions below.',
      allScores:'Similarity across all pitchers',distribution:'Each dot is one eligible pitcher.',
      copy:'Copy link',copied:'Link copied.',copyFailed:'Copy the link from the address bar.',
      method:'Comparison & sources',
      methodText:'Compare joint speed/HB/IVB distributions within the same pitch type, weighted by usage overlap. Pitches thrown by only one pitcher also reduce the score. Spread and relationships between features matter, not just averages.',
      methodScope:'HB is signed horizontal movement from the catcher’s view; IVB excludes gravity. Both are derived from Naver trajectories. Scores describe acquired records, not performance or ability.',
      methodMath:'Only pitches with speed, HB and IVB are used. Pitchers need 500 usable pitches and at least 90% coverage of their acquired pitches. See the linked document for the formula.',
      source:'Source records',reference:'Savant original',math:'Formula & coverage',
      barsNote:'Usage shares use the valid pitches included in the comparison.',
      emptyCompare:'No comparison pitcher is available.',graphLabel:'Map of pitchers similar to the selected pitcher',
      mix:'Pitch repertoire & usage',shapes:'Pitch distributions',pitch:'Pitch type',
      axes:['Velocity','Horizontal break · HB','Induced vertical break · IVB'],
      median:'Median',missing:'No pitches of this type',
      graphNote:'Observed distributions of the selected pitch type. Both pitchers share the same axes.'
    }
  };
  const lang = () => document.documentElement.lang === 'ko' ? 'ko' : 'en';
  const t = () => copy[lang()];
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const number = v => Number(v).toLocaleString(lang() === 'ko' ? 'ko-KR' : 'en-US');
  const date = value => `${value.slice(0,4)}.${value.slice(4,6)}.${value.slice(6)}`;
  const identity = p => state.data?.players.some(q=>q.code!==p.code&&q.name===p.name)?p.code:'';
  const label = p => `${p.name}${identity(p)?' · '+identity(p):''}`;
  const pitchName=i=>state.data.pitchTypes[i].name[lang()];
  const pitchColor=i=>state.data.pitchTypes[i].color;
  const sample = p => lang() === 'ko' ? `${number(p.pitches)}구 · ${number(p.games)}경기` : `${number(p.pitches)} pitches · ${number(p.games)} games`;
  const href = hash => {const u=new URL(location.href);u.search='';u.hash=hash;return u.pathname+u.hash;};
  const currentRoute = () => location.hash.split('?')[0];
  function donut(p, x, y, radius, selected=false) {
    let offset = 0;
    const values = core.proportions(p.counts), circumference = 2 * Math.PI * radius;
    return `<g aria-hidden="true"><circle cx="${x}" cy="${y}" r="${radius+9}" fill="${selected?'#173c36':'#fcfdf9'}" stroke="${selected?'#173c36':'#d4dfd7'}"/>` + values.map((v,i) => {
      const s=`<circle cx="${x}" cy="${y}" r="${radius}" fill="none" stroke="${pitchColor(i)}" stroke-width="${selected?13:8}" stroke-dasharray="${v*circumference} ${circumference}" stroke-dashoffset="${-offset*circumference}" transform="rotate(-90 ${x} ${y})"/>`;
      offset += v; return s;
    }).join('') + `<circle cx="${x}" cy="${y}" r="${selected?6:3}" fill="${selected?'#eff5df':'#173c36'}"/></g>`;
  }
  function preview() {
    return `<svg viewBox="0 0 620 310" role="img" aria-label="${esc(t().name)}"><g stroke="#bad0c5" stroke-width="2"><path d="M310 150L170 66M310 150L450 70M310 150L480 215M310 150L150 225"/></g>${[[310,150,37],[170,66,22],[450,70,22],[480,215,22],[150,225,22]].map(([x,y,r],i)=>`<circle cx="${x}" cy="${y}" r="${r+10}" fill="${i?'#fff':'#173c36'}"/><circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#187c67" stroke-width="10"/><circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#c85b3b" stroke-width="10" stroke-dasharray="${r*.8} ${r*7}" transform="rotate(-75 ${x} ${y})"/>`).join('')}<text x="310" y="285" text-anchor="middle" fill="#55736a" font-size="16">KBO · PITCHER SIMILARITY</text></svg>`;
  }
  function gallery() {
    document.getElementById('visuals-gallery').innerHTML=`<header class="visuals-heading"><span class="visuals-kicker">KBO PITCH LAB</span><h1>Visuals</h1><p>${t().intro}</p></header><div class="visuals-grid"><a class="visuals-card" href="${esc(href('#visuals/affinity'))}"><div class="visuals-thumbnail">${preview()}</div><div class="visuals-card-copy"><span class="visuals-kicker">01 / PITCHER SIMILARITY</span><h2>${t().name}</h2><p>${t().card}</p><span class="visuals-card-link">${t().open} <span aria-hidden="true">↗</span></span></div></a></div>`;
  }
  function install() {
    I18N.ko.navVisuals='Visuals'; I18N.en.navVisuals='Visuals';
    document.querySelector('.main-nav a[href="#visuals"]').addEventListener('click', e => {
      e.preventDefault(); location.href = href('#visuals');
    });
    let resized;
    window.addEventListener('resize',()=>{clearTimeout(resized);resized=setTimeout(()=>{
      if(currentRoute()==='#visuals/affinity'&&state.selected&&document.getElementById('affinity-map')){drawMatches(state.ranked.filter(p=>p.score>=state.current.threshold));}
    },120);});
  }
  async function route(force=false) {
    const path = currentRoute();
    if (!['#visuals', '#visuals/affinity'].includes(path)) return;
    if (path === '#visuals') {gallery(); return;}
    const host = document.getElementById('visuals-affinity');
    if (!state.data) {
      host.innerHTML=`<p role="status">${t().loading}</p>`;
      try {
        if (!state.pending) state.pending=fetch(dataURL,{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error('affinity');return r.json()}).then(data=>{
          if(data.schemaVersion!==2||data.similarity?.id!=='usage-overlap-sliced-wasserstein-v1'||!data.players?.length)throw Error('Unsupported repertoire data');
          for(const p of data.players){
            if(p.counts.length!==data.pitchTypes.length||p.pitches!==p.counts.reduce((a,b)=>a+b,0)||p.pitches<data.minimumPitches||p.pitches/p.observedPitches<data.minimumCoverage)throw Error('Invalid denominator');
            core.proportions(p.counts);
            p.types.forEach((type,i)=>{
              if(!p.counts[i]){if(type!==null)throw Error('Unexpected distribution');return;}
              if(type?.n!==p.counts[i]||type.projections.length!==data.similarity.directions.length)throw Error('Invalid distribution');
              for(const q of type.projections)if(q.length!==data.similarity.quantileCount||q.some(v=>!Number.isFinite(v)))throw Error('Invalid projections');
              for(const h of type.histograms)if(h.reduce((n,pair)=>n+pair[1],0)!==type.n)throw Error('Invalid histogram');
            });
          }
          return data;
        });
        state.data=await state.pending;
      } catch(error) {
        state.pending=null;
        host.innerHTML=`<p role="alert">${t().failed}</p><button class="button-outline" id="affinity-retry">${t().retry}</button>`;
        host.querySelector('button').onclick=()=>route(true);return;
      }
    }
    if(currentRoute()!=='#visuals/affinity')return;
    const parsed=core.parse(location.hash), players=state.data.players;
    const fallback=players.find(p=>p.name==='네일')||players.reduce((a,b)=>a.pitches>b.pitches?a:b);
    const selected=players.find(p=>p.key===parsed.key)||fallback;
    parsed.key=selected.key;
    const ranked=core.rank(players,selected);
    const visible=ranked.filter(p=>p.score>=parsed.threshold);
    if(!visible.some(p=>p.player.key===parsed.compare))parsed.compare=visible[0]?.player.key||'';
    const canonical=core.fragment(parsed);
    if(location.hash!==canonical || location.search)history.replaceState(null,'',href(canonical));
    const signature=canonical+'|'+lang();
    if(!force&&state.signature===signature&&host.querySelector('#affinity-map'))return;
    state.signature=signature;state.current=parsed;state.selected=selected;state.ranked=ranked;state.compare=parsed.compare;
    render(host,players,visible);
  }
  function change(next,replace=false) {
    const activeId=document.activeElement?.id;
    const s={...state.current,...next};
    const target=href(core.fragment(s));
    history[replace?'replaceState':'pushState'](null,'',target);
    route(true);
    if(activeId)document.getElementById(activeId)?.focus();
  }
  function render(host,players,visible) {
    const s=state.current,d=state.data,p=state.selected;
    const cohort=lang()==='ko'?`${d.season} · ${date(d.dateStart)}–${date(d.dateEnd)} · 확보한 투구 ${number(d.usablePitches)}구`:`${d.season} · ${date(d.dateStart)}–${date(d.dateEnd)} · ${number(d.usablePitches)} acquired pitches`;
    const coverage=lang()==='ko'?`유효 투구 ${number(d.minimumPitches)}구 이상 · 투수 ${players.length}명`:`At least ${number(d.minimumPitches)} usable pitches · ${players.length} pitchers`;
    host.innerHTML=`<a class="visuals-back" href="${esc(href('#visuals'))}">${t().back}</a><header class="visuals-heading affinity-heading"><div><span class="visuals-kicker">01 / PITCHER SIMILARITY</span><h1>${t().title}</h1><p>${t().subtitle}</p></div><button class="button-outline" id="affinity-share">${t().copy}</button></header>
      <div class="affinity-controls"><div class="affinity-picker"><label for="affinity-search">${t().player}</label><input id="affinity-search" type="text" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="affinity-options" autocomplete="off" value="${esc(label(p))}" placeholder="${t().search}"><ul id="affinity-options" role="listbox" aria-label="${t().choose}" hidden></ul></div><span class="affinity-cohort">${coverage}</span></div>
      <p class="affinity-scope">${cohort}</p><div class="affinity-workspace"><div class="affinity-map-panel"><div class="affinity-map-head"><div><span>${t().selected}</span><h2>${esc(label(p))}</h2><p>${sample(p)}</p></div><span class="affinity-badge">${t().score}<b>100</b></span></div><div id="affinity-map"></div><p class="affinity-map-hint">${t().hint}</p></div><aside class="affinity-neighbors" aria-labelledby="affinity-near-heading"><div class="affinity-neighbors-title"><h2 id="affinity-near-heading">${t().near}</h2><span id="affinity-count"></span></div><label class="affinity-threshold-label" for="affinity-threshold">${t().threshold}<output id="affinity-threshold-value">${s.threshold.toFixed(1)}</output></label><input id="affinity-threshold" type="range" min="0" max="100" step="0.1" value="${s.threshold}" aria-valuetext="${s.threshold.toFixed(1)} / 100"><div id="affinity-ranking" aria-live="polite"></div></aside></div>
      <section class="affinity-distribution"><h2>${t().allScores}</h2><div id="affinity-beeswarm"></div><p>${t().distribution}</p></section><section class="affinity-comparison" aria-labelledby="affinity-compare-title"><div class="affinity-comparison-top"><div><span class="visuals-kicker">PITCH REPERTOIRE</span><h2 id="affinity-compare-title">${t().compare}</h2></div><button class="button-outline" id="affinity-recenter">${t().recenter} ↗</button></div><div id="affinity-compare-head"></div><h3 class="affinity-subtitle">${t().mix}</h3><div id="affinity-bars"></div><p class="affinity-bars-note">${t().barsNote}</p><div class="affinity-shapes-heading"><h3>${t().shapes}</h3><label for="affinity-pitch">${t().pitch}<select id="affinity-pitch"></select></label></div><div id="affinity-type-samples"></div><div id="affinity-shapes"></div><p class="affinity-bars-note">${t().graphNote}</p></section>
      <details class="affinity-method"><summary>${t().method}</summary><p>${t().methodText}</p><p>${t().methodScope}</p><p>${t().methodMath}</p><p><a href="${esc(d.source.url)}" target="_blank" rel="noopener noreferrer">${t().source} ↗</a> · <a href="https://baseballsavant.mlb.com/affinity-pitchers-bySHV" target="_blank" rel="noopener noreferrer">${t().reference} ↗</a> · <a href="docs/visuals-affinity.md">${t().math} ↗</a></p></details><p class="sr-only" id="affinity-announcement" role="status"></p>`;
    host.querySelector('#affinity-share').onclick=async()=>{
      const announcement=host.querySelector('#affinity-announcement');
      try{await navigator.clipboard.writeText(location.href);announcement.textContent=t().copied;host.querySelector('#affinity-share').textContent=t().copied;}catch(error){announcement.textContent=t().copyFailed;}
    };
    host.querySelector('#affinity-threshold').oninput=e=>{
      s.threshold=Number(e.target.value);
      e.target.setAttribute('aria-valuetext',`${s.threshold.toFixed(1)} / 100`);
      host.querySelector('#affinity-threshold-value').textContent=s.threshold.toFixed(1);
      const matches=state.ranked.filter(item=>item.score>=s.threshold);
      if(!matches.some(item=>item.player.key===state.compare))state.compare=matches[0]?.player.key||'';
      s.compare=state.compare;history.replaceState(null,'',href(core.fragment(s)));state.signature=core.fragment(s)+'|'+lang();
      drawMatches(matches);
    };
    host.querySelector('#affinity-recenter').onclick=()=>{if(state.compare)change({key:state.compare,compare:''})};
    picker(players);drawMatches(visible);
  }
  function picker(players) {
    const input=document.getElementById('affinity-search'), list=document.getElementById('affinity-options');
    let matches=[],active=-1;
    const sorted=[...players].sort((a,b)=>a.name.localeCompare(b.name,'ko')||a.key.localeCompare(b.key));
    function close(){list.hidden=true;input.setAttribute('aria-expanded','false');input.removeAttribute('aria-activedescendant');active=-1;}
    function show(query=''){
      matches=sorted.filter(p=>`${p.name} ${p.code}`.toLowerCase().includes(query.toLowerCase().trim()));active=-1;
      list.innerHTML=matches.map((p,i)=>`<li id="affinity-option-${i}" role="option" aria-selected="${p.key===state.selected.key}" data-index="${i}"><span>${esc(label(p))}</span><small>${number(p.pitches)}${lang()==='ko'?'구':' pitches'}</small></li>`).join('')||`<li role="presentation">${t().noPlayer}</li>`;
      list.hidden=false;input.setAttribute('aria-expanded','true');input.removeAttribute('aria-activedescendant');
    }
    function choose(i){const p=matches[i];if(p){close();change({key:p.key,compare:''});document.getElementById('affinity-search').focus();document.getElementById('affinity-search').select();}}
    input.onfocus=()=>{input.select();};input.onclick=()=>show();input.oninput=()=>show(input.value);
    input.onblur=()=>{setTimeout(()=>{if(!list.contains(document.activeElement)){close();input.value=label(state.selected);}},0);};
    list.tabIndex=-1;
    list.onpointerdown=e=>{if(e.pointerType==='mouse')e.preventDefault();else list.focus({preventScroll:true});};
    list.onclick=e=>{const option=e.target.closest('[data-index]');if(option)choose(Number(option.dataset.index));};
    list.onblur=()=>{close();input.value=label(state.selected);};
    input.onkeydown=e=>{
      if(['ArrowDown','ArrowUp'].includes(e.key)){
        e.preventDefault();if(list.hidden)show();if(!matches.length)return;
        active=active<0?(e.key==='ArrowDown'?0:matches.length-1):(active+(e.key==='ArrowDown'?1:-1)+matches.length)%matches.length;
        list.querySelectorAll('[data-index]').forEach((el,i)=>el.classList.toggle('is-active',i===active));
        input.setAttribute('aria-activedescendant',`affinity-option-${active}`);list.children[active].scrollIntoView({block:'nearest'});
      }else if(e.key==='Enter'&&!list.hidden&&matches.length){e.preventDefault();choose(Math.max(0,active));}
      else if(e.key==='Escape'){close();input.value=label(state.selected);}
    };
  }
  function selectComparison(key) {
    state.compare=key;state.current.compare=key;
    history.replaceState(null,'',href(core.fragment(state.current)));state.signature=core.fragment(state.current)+'|'+lang();
    document.querySelectorAll('[data-affinity-compare]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.affinityCompare===key)));
    drawComparison();drawDistribution();
    const other=state.ranked.find(item=>item.player.key===key)?.player;
    document.getElementById('affinity-announcement').textContent=other?(lang()==='ko'?`${label(state.selected)}, ${label(other)}의 구종과 분포를 표시합니다.`:`Comparing repertoires for ${label(state.selected)} and ${label(other)}.`):'';
  }
  function drawMatches(matches) {
    document.getElementById('affinity-count').textContent=lang()==='ko'?`${matches.length}명`:`${matches.length} pitchers`;
    const ranked=document.getElementById('affinity-ranking');
    ranked.innerHTML=matches.length?`<ol>${matches.map(({player:p,score},i)=>`<li><button data-affinity-compare="${esc(p.key)}" aria-pressed="${state.compare===p.key}"><span class="affinity-rank">${i+1}</span><span class="affinity-match-name">${esc(label(p))}<small>${sample(p)}</small></span><strong>${score.toFixed(1)}</strong></button></li>`).join('')}</ol>`:`<p class="affinity-empty" role="status">${t().none}</p>`;
    const map=document.getElementById('affinity-map'), mobile=matchMedia('(max-width:620px)').matches;
    const shown=matches.slice(0,mobile?6:12), p=state.selected;
    const width=mobile?Math.max(240,map.clientWidth):760, height=mobile?520:570;
    const cx=width/2,cy=mobile?250:285;
    // A neighborhood diagram, not a league-wide embedding. Angles are layout only.
    const nodes=shown.map(({player,score},i)=>{
      const angle=-Math.PI/2+2*Math.PI*i/shown.length;
      const radius=shown.length<5?205:(i%2?248:190);
      return {player,score,x:mobile?width*(i%2?.78:.22):cx+Math.sin(angle)*radius*1.07,y:mobile?90+Math.floor(i/2)*160:cy-Math.cos(angle)*radius*.84};
    });
    map.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="group" aria-label="${esc(t().graphLabel)}">${mobile?'':`<circle cx="${cx}" cy="${cy}" r="198" fill="none" stroke="#dbe7dd" stroke-dasharray="3 9"/><circle cx="${cx}" cy="${cy}" r="100" fill="none" stroke="#e0e9e1"/>`}${nodes.map(n=>`<line x1="${cx}" y1="${cy}" x2="${n.x}" y2="${n.y}" stroke="#b5ccbf" stroke-width="${1+(n.score/100)**8*4}"/>`).join('')}${donut(p,cx,cy,mobile?28:38,true)}${mobile?'':`<rect x="${cx-78}" y="${cy+54}" width="156" height="38" rx="19" fill="#173c36"/><text x="${cx}" y="${cy+79}" text-anchor="middle" fill="#fff" font-size="19" font-weight="700">${esc(p.name)}</text>`}${nodes.map(n=>`<g class="affinity-node" role="button" tabindex="0" data-affinity-compare="${esc(n.player.key)}" aria-pressed="${state.compare===n.player.key}" aria-label="${esc(label(n.player))} · ${t().score} ${n.score.toFixed(1)}"><rect class="affinity-node-hit" x="${n.x-56}" y="${n.y-31}" width="112" height="103" rx="14" fill="transparent"/>${donut(n.player,n.x,n.y,20)}<text x="${n.x}" y="${n.y+48}" text-anchor="middle" font-size="18" font-weight="600" fill="#173c36">${esc(n.player.name)}</text><text x="${n.x}" y="${n.y+68}" text-anchor="middle" font-size="${mobile?17:14}" fill="#4e6a60">${esc(identity(n.player))}${identity(n.player)?' · ':''}${n.score.toFixed(1)}</text></g>`).join('')}</svg><div class="affinity-legend">${state.data.pitchTypes.map((type,i)=>p.counts[i]||nodes.some(n=>n.player.counts[i])?`<span><i style="background:${pitchColor(i)}"></i>${esc(pitchName(i))}</span>`:'').join('')}</div><p class="affinity-map-count">${lang()==='ko'?`지도에는 유사도가 높은 ${shown.length}명을 표시합니다.`:`The map shows the ${shown.length} closest matches.`}</p>`;
    document.querySelectorAll('[data-affinity-compare]').forEach(el=>{
      el.onclick=()=>selectComparison(el.dataset.affinityCompare);
      if(el.tagName.toLowerCase()==='g')el.onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();selectComparison(el.dataset.affinityCompare);}};
    });
    drawDistribution();drawComparison();
  }
  function drawDistribution() {
    const host=document.getElementById('affinity-beeswarm');
    const width=Math.max(230,host.clientWidth), padding=24, span=width-padding*2;
    const values=state.ranked, threshold=state.current.threshold, bins=new Map();
    // Fixed score domain [0,100], collision-packed rows; never rescale scores.
    const dots=values.map(item=>{
      const x=padding+item.score/100*span, bin=Math.round(x/6), row=bins.get(bin)||0;bins.set(bin,row+1);
      return {...item,x,y:60+(row%2?1:-1)*Math.ceil(row/2)*5.5};
    });
    const ys=dots.map(d=>d.y),low=Math.min(0,...ys)-8, high=Math.max(88,...ys)+40;
    host.innerHTML=`<svg viewBox="0 ${low} ${width} ${high-low}" role="img" aria-label="${esc(t().allScores)}"><title>${esc(t().distribution)}</title><rect x="${padding+threshold/100*span}" y="${low}" width="${(100-threshold)/100*span}" height="${high-low-32}" fill="#e2ede1"/>${dots.map(d=>`<circle cx="${d.x}" cy="${d.y}" r="${d.player.key===state.compare?4:2.5}" fill="${d.player.key===state.compare?'#c85b3b':d.score>=threshold?'#245f4e':'#adbcb4'}"><title>${esc(label(d.player))}: ${d.score.toFixed(1)}</title></circle>`).join('')}<line x1="${padding}" x2="${width-padding}" y1="${high-29}" y2="${high-29}" stroke="#9aafa3"/>${[0,25,50,75,100].map(n=>`<text x="${padding+n/100*span}" y="${high-7}" text-anchor="middle" font-size="17" fill="#486257">${n}</text>`).join('')}<line x1="${padding+threshold/100*span}" x2="${padding+threshold/100*span}" y1="${low}" y2="${high-29}" stroke="#245f4e" stroke-dasharray="4 4"/></svg>`;
  }
  function drawComparison() {
    const p=state.selected, match=state.ranked.find(item=>item.player.key===state.compare), q=match?.player;
    document.getElementById('affinity-recenter').hidden=!q;
    document.getElementById('affinity-compare-head').innerHTML=`<div class="affinity-person"><span>${t().selected}</span><strong>${esc(label(p))}</strong><small>${sample(p)}</small></div>${q?`<div class="affinity-person is-comparison"><span>${t().counterpart} · ${t().score} ${match.score.toFixed(1)}</span><strong>${esc(label(q))}</strong><small>${sample(q)}</small></div>`:`<p>${t().emptyCompare}</p>`}`;
    const a=core.proportions(p.counts), b=q?core.proportions(q.counts):null;
    const barName=player=>q&&p.name===q.name?label(player):player.name;
    const types=state.data.pitchTypes.map((_,i)=>i).filter(i=>p.counts[i]||(q&&q.counts[i]));
    document.getElementById('affinity-bars').innerHTML=types.map(i=>`<div class="affinity-pitch-row" data-pitch-index="${i}"><h3><i style="background:${pitchColor(i)}"></i>${esc(pitchName(i))}</h3><div class="affinity-bar-pair"><div class="affinity-bar-row"><span class="affinity-bar-name">${esc(barName(p))}</span><div class="affinity-bar-track"><i style="width:${a[i]*100}%;background-color:${pitchColor(i)}"></i></div><strong>${(a[i]*100).toFixed(1)}%</strong></div>${q?`<div class="affinity-bar-row is-comparison"><span class="affinity-bar-name">${esc(barName(q))}</span><div class="affinity-bar-track"><i style="width:${b[i]*100}%;background-color:${pitchColor(i)}"></i></div><strong>${(b[i]*100).toFixed(1)}%</strong></div>`:''}</div></div>`).join('');
    if(!types.includes(state.pitch))state.pitch=types.reduce((best,i)=>p.counts[i]>p.counts[best]?i:best,types[0]);
    const select=document.getElementById('affinity-pitch');
    select.innerHTML=types.map(i=>`<option value="${i}" ${i===state.pitch?'selected':''}>${esc(pitchName(i))}</option>`).join('');
    select.onchange=()=>{state.pitch=Number(select.value);drawShapes();};
    drawShapes();
  }
  function drawShapes(){
    const p=state.selected,q=state.ranked.find(item=>item.player.key===state.compare)?.player,i=state.pitch;
    const players=[p,q].filter(Boolean), samples=players.map(player=>player.types[i]);
    document.getElementById('affinity-type-samples').innerHTML=players.map((player,j)=>`<span class="affinity-shape-key ${j?'is-comparison':''}"><i></i><strong>${esc(label(player))}</strong> · ${samples[j]?number(samples[j].n)+(lang()==='ko'?'구':' pitches'):t().missing}</span>`).join('');
    const host=document.getElementById('affinity-shapes');
    host.innerHTML=state.data.axes.map((axis,a)=>`<article class="affinity-shape" data-axis="${axis.id}"><h4>${t().axes[a]} <small>${axis.unit}</small></h4><div class="affinity-density"></div><div class="affinity-density-values">${players.map((player,j)=>`<p><span>${esc(label(player))}</span><strong>${samples[j]?samples[j].quantiles[a][2].toFixed(1)+' '+axis.unit:'—'}</strong></p>`).join('')}<span>${t().median}</span></div></article>`).join('');
    host.querySelectorAll('.affinity-shape').forEach((article,a)=>{
      const axis=state.data.axes[a],step=axis.step,sets=samples.map(type=>type?new Map(type.histograms[a]):null);
      const indices=sets.flatMap(set=>set?[...set.keys()]:[]);
      const min=Math.floor((Math.min(...indices)*step-step)/5)*5,max=Math.ceil(((Math.max(...indices)+1)*step+step)/5)*5;
      const span=Math.max(10,max-min),end=min+span;
      const probabilities=sets.flatMap((set,j)=>set?[...set.values()].map(n=>n/samples[j].n):[]);
      const peak=Math.max(.05,Math.ceil(Math.max(...probabilities)*20)/20);
      const container=article.querySelector('.affinity-density'),width=Math.max(230,container.clientWidth),height=205;
      const x=v=>44+(v-min)/span*(width-58),y=v=>163-v/peak*137;
      const paths=sets.map((set,j)=>{
        if(!set)return '';
        let d=`M ${x(min)} ${y(0)}`;
        for(let bin=Math.floor(min/step);bin<Math.ceil(end/step);bin++){
          const value=(set.get(bin)||0)/samples[j].n;
          d+=` H ${x(Math.max(min,bin*step))} V ${y(value)} H ${x(Math.min(end,(bin+1)*step))}`;
        }
        d+=` V ${y(0)}`;
        return `<path class="affinity-density-line ${j?'is-comparison':''}" d="${d}" fill="none" stroke="${j?'#173c36':pitchColor(i)}" stroke-width="${j?2.5:3}" ${j?'stroke-dasharray="6 4"':''}/>`;
      }).join('');
      container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(t().axes[a]+' · '+pitchName(i))}"><title>${esc(t().graphNote)}</title>${[0,peak/2,peak].map(v=>`<line x1="44" x2="${width-14}" y1="${y(v)}" y2="${y(v)}" stroke="#dbe5da"/><text x="37" y="${y(v)+5}" text-anchor="end" font-size="14" fill="#4d655c">${Math.round(v*100)}%</text>`).join('')}${min<0&&end>0?`<line x1="${x(0)}" x2="${x(0)}" y1="26" y2="163" stroke="#a5b8ab" stroke-dasharray="3 4"/>`:''}${paths}${[min,(min+end)/2,end].map(v=>`<text x="${x(v)}" y="190" text-anchor="middle" font-size="17" fill="#4d655c">${Number.isInteger(v)?v:v.toFixed(1)}</text>`).join('')}</svg>`;
    });
  }
  return {install,route,state};
})();
