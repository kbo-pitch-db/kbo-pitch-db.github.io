/* Pitch-level lookup and export. No metric is calculated from display pixels. */
window.MetricsDatabase = (() => {
  const VERSION='mechanics-db-20261028-v1',contexts=new Map();
  const defaults=['lead_knee_range_2d','throw_elbow_range_2d','trunk_tilt_p90_2d'];
  let dataset=null,loading=null,dialog=null,lastFocus=null;
  const t=(ko,en)=>app.lang==='ko'?ko:en,e=v=>escapeHtml(v),name=m=>m.name[app.lang],definition=m=>m.definition[app.lang];
  const number=v=>typeof v==='number'&&Number.isFinite(v),format=v=>number(v)?v.toFixed(1):'—';
  const groupName=k=>({lead_leg:t('앞다리','Lead leg'),arm:t('팔','Arm'),pelvis_trunk:t('골반·몸통','Pelvis / trunk'),posture:t('자세','Posture'),space:t('공간','Space'),timing:t('타이밍','Timing')})[k]||k;
  const statusName=k=>dataset?.statusCodes[k]?.[app.lang]||k;
  const unit=m=>m.unit==='ratio'?t('비율','ratio'):m.unit;
  function dataUrl(path){const u=new URL(url('metrics/'+path));u.searchParams.set('v',VERSION);return u.href}
  function install(){
    Object.assign(I18N.ko,{navMetrics:'지표 DB',reportMetrics:'지표 집계',reportMechanics:'투구 상세 분석',browsePitchers:'선수별 집계',heroDesc:'KBO 투구 기록과 영상 기반 메커니즘 지표를 조회하고 내려받는 연구 데이터베이스입니다.',researchLine:'투구 하나에서 시작하는 데이터베이스',researchDesc:'지표별 값과 상태를 비교하고, 원본 투구에서 근거를 확인하세요.'});
    Object.assign(I18N.en,{navMetrics:'Metrics DB',reportMetrics:'Metric summary',reportMechanics:'Pitch detail',browsePitchers:'Player summaries',heroDesc:'A research database for KBO pitch records and video-derived mechanics metrics. Query, compare and download.',researchLine:'A database built around individual pitches',researchDesc:'Compare metric values and calculation status, then trace the evidence to the actual pitch.'});
  }
  async function load(){
    if(dataset)return dataset;
    if(!loading)loading=(async()=>{
      let d;
      if(typeof DecompressionStream==='function'){
        try{const r=await fetch(dataUrl('database.json.gz'));if(!r.ok)throw Error('compressed database');d=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json()}catch{d=null}
      }
      if(!d){const r=await fetch(dataUrl('database.json'));if(!r.ok)throw Error('database');d=await r.json()}
      if(d.schemaVersion!==1||d.version!==VERSION||!Array.isArray(d.rows)||!Array.isArray(d.dictionary))throw Error('database version mismatch');
      if(app.reconstructions?.version&&d.poseVersion!==app.reconstructions.version)throw Error('pose release mismatch');
      dataset=d;renderGuideDictionary();return d;
    })().catch(error=>{loading=null;throw error});
    return loading;
  }
  function cell(row,key){return row.metrics?.[key]||{value:null,status:'no_motion',validFrames:0,totalFrames:0}}
  function eligible(value,include3d=false){return number(value.value)&&(['observed_2d','provisional_event'].includes(value.status)||(include3d&&value.status==='needs_review_3d'))}
  function summarize(rows,key,include3d=false){
    const values=rows.map(r=>cell(r,key)).filter(v=>eligible(v,include3d)).map(v=>v.value),q=PitchCharts.quantile;
    return {n:values.length,median:q(values,.5),q25:values.length>=3?q(values,.25):null,q75:values.length>=3?q(values,.75):null,values};
  }
  function encodeCsv(value){
    let s=value==null?'':String(value);
    if(typeof value==='string'&&/^[=+@\-]/.test(s))s="'"+s;
    return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;
  }
  function csv(headers,rows){return '\ufeff'+[headers,...rows].map(row=>row.map(encodeCsv).join(',')).join('\r\n')+'\r\n'}
  function download(filename,contents,type='text/csv;charset=utf-8'){const link=document.createElement('a'),objectUrl=URL.createObjectURL(new Blob([contents],{type}));link.href=objectUrl;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(objectUrl),1000)}
  function exportPitches(rows,columns){
    const headers=['row_id','unit','pitch_id','player_code','player','throwing_hand','game_id','date','pitch_type','speed_kmh','primary_clip_id','all_clip_ids','pitch_link_status','database_version'];
    for(const key of columns)for(const suffix of ['value','unit','status','valid_frames','total_frames','calculation_version','reference_event','source_clip_id'])headers.push(key+'__'+suffix);
    return csv(headers,rows.map(row=>{
      const out=[row.id,row.unit,row.pitchId,row.playerCode,row.player,row.hand,row.gameId,row.date,row.pitch,row.speed,row.primaryClipId,row.clipIds.join('|'),row.linkStatus,VERSION];
      for(const key of columns){const m=dataset.dictionary.find(m=>m.id===key),v=cell(row,key);out.push(v.value,m.unit,v.status,v.validFrames,v.totalFrames,m.calculationVersion,m.referenceEvent,row.primaryClipId)}
      return out;
    }));
  }
  function grouped(rows){
    const map=new Map();for(const row of rows){const id=[row.unit,row.playerCode,row.pitch||''].join('|');if(!map.has(id))map.set(id,{id,unit:row.unit,playerCode:row.playerCode,player:row.player,pitch:row.pitch,rows:[]});map.get(id).rows.push(row)}return [...map.values()];
  }
  function sortGroups(groups,state){return [...groups].sort((a,b)=>{
    const isMetric=dataset.dictionary.some(m=>m.id===state.sort),x=isMetric?summarize(a.rows,state.sort,state.include3d).median:a[state.sort],y=isMetric?summarize(b.rows,state.sort,state.include3d).median:b[state.sort];
    if(x==null&&y==null)return a.id.localeCompare(b.id);if(x==null)return 1;if(y==null)return -1;
    const diff=typeof x==='number'?x-y:String(x).localeCompare(String(y),'ko');return (state.direction==='asc'?diff:-diff)||a.id.localeCompare(b.id);
  })}
  function aggregateCsv(groups,columns,include3d){
    const head=['unit','player_code','player','pitch_type','rows','games','database_version','include_review_3d'];
    for(const key of columns)for(const suffix of ['median','q25','q75','valid_n','unit','calculation_version','status_counts'])head.push(key+'__'+suffix);
    return csv(head,groups.map(g=>{
      const row=[g.unit,g.playerCode,g.player,g.pitch,g.rows.length,new Set(g.rows.map(r=>r.gameId).filter(Boolean)).size,VERSION,include3d];
      for(const key of columns){const s=summarize(g.rows,key,include3d),m=dataset.dictionary.find(m=>m.id===key),counts={};g.rows.forEach(r=>{const status=cell(r,key).status;counts[status]=(counts[status]||0)+1});row.push(s.median,s.q25,s.q75,s.n,m.unit,m.calculationVersion,JSON.stringify(counts))}return row;
    }));
  }
  function parseOptions(fixedPlayer){
    const p=new URL(location.href).searchParams,prefix=fixedPlayer?'pm_':'db_',get=k=>p.get(prefix+k),metrics=dataset.dictionary.map(m=>m.id),valid=k=>metrics.includes(k);
    const fallbackScope=!fixedPlayer||dataset.rows.some(r=>r.playerCode===fixedPlayer&&r.unit==='pitch_record'&&r.primaryClipId)?'linked':dataset.rows.some(r=>r.playerCode===fixedPlayer&&r.unit==='unlinked_clip')?'unlinked':'records';
    return {prefix,scope:['records','unlinked','linked'].includes(get('scope'))?get('scope'):fallbackScope,view:['pitches','aggregate','dictionary'].includes(get('view'))?get('view'):(fixedPlayer?'aggregate':'pitches'),
      player:fixedPlayer||get('player')||'',pitch:get('pitch')||'',game:get('game')||'',source:get('source')||'',search:get('search')||'',status:get('status')||'',
      columns:get('columns')!==null?get('columns').split(',').filter(valid):[...defaults],metric:valid(get('metric'))?get('metric'):defaults[0],
      sort:valid(get('sort'))||['player','pitch','speed','date'].includes(get('sort'))?get('sort'):'date',direction:get('direction')==='asc'?'asc':'desc',
      x:get('x')==='speed'||valid(get('x'))?get('x'):'speed',y:valid(get('y'))?get('y'):defaults[0],include3d:get('review')==='yes',plot:get('plot')==='scatter'?'scatter':'distribution',plotsOpen:get('plots')==='yes',page:0};
  }
  function clearPlayerLink(link){for(const key of [...link.searchParams.keys()])if(key.startsWith('pm_'))link.searchParams.delete(key)}
  function openPlayerContext(code,options={}){
    const link=new URL(location.href);clearPlayerLink(link);window.MechanicsReport?.clearLink(link);link.searchParams.delete('clip');link.searchParams.set('pitcher',code);
    for(const [key,value]of Object.entries(options))if(value!==''&&value!=null)link.searchParams.set('pm_'+key,Array.isArray(value)?value.join(','):String(value));
    link.searchParams.set('pm_view','aggregate');history.replaceState(null,'',link);contexts.delete('player-metrics-root:'+code);PitchPortal.openPlayer(code);
  }
  function save(state){
    const u=new URL(location.href),values={scope:state.scope,view:state.view,player:state.player,pitch:state.pitch,game:state.game,source:state.source,search:state.search,status:state.status,columns:state.columns.join(','),metric:state.metric,sort:state.sort,direction:state.direction,x:state.x,y:state.y,review:state.include3d?'yes':'no',plot:state.plot,plots:state.plotsOpen?'yes':'no'};
    if(state.prefix==='db_'&&(!location.hash||location.hash==='#database'))u.searchParams.delete('clip');
    for(const [k,v]of Object.entries(values)){if(v!==''||k==='columns')u.searchParams.set(state.prefix+k,v);else u.searchParams.delete(state.prefix+k)}history.replaceState(null,'',u);
  }
  function filtered(state){
    return dataset.rows.filter(row=>{
      if(state.scope==='unlinked'?row.unit!=='unlinked_clip':row.unit!=='pitch_record')return false;
      if(state.scope==='linked'&&!row.primaryClipId)return false;
      if(state.player&&row.playerCode!==state.player||state.pitch&&(row.pitch||'unknown')!==state.pitch||state.game&&row.gameId!==state.game||state.source&&row.sourceId!==state.source)return false;
      if(state.search&&!`${row.player} ${row.playerCode} ${row.id} ${row.clipIds.join(' ')}`.toLowerCase().includes(state.search.toLowerCase()))return false;
      const value=cell(row,state.metric);
      if(state.status==='missing'&&value.value!==null||state.status&&state.status!=='missing'&&value.status!==state.status)return false;
      return true;
    });
  }
  function sortRows(rows,state){
    return [...rows].sort((a,b)=>{
      const metric=dataset.dictionary.some(m=>m.id===state.sort),x=metric?cell(a,state.sort).value:a[state.sort],y=metric?cell(b,state.sort).value:b[state.sort];
      if(x==null&&y==null)return a.id.localeCompare(b.id);if(x==null)return 1;if(y==null)return -1;
      const diff=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),'ko');return (state.direction==='asc'?diff:-diff)||a.id.localeCompare(b.id);
    });
  }
  function options(values,current,all,label=value=>value){return `<option value="">${e(all)}</option>`+values.map(value=>`<option value="${e(value)}" ${value===current?'selected':''}>${e(label(value))}</option>`).join('')}
  async function mount(root,fixedPlayer=null){
    if(!root)return;
    const token={};root._metricsToken=token;
    root.innerHTML=`<p class="db-loading" role="status">${t('메커니즘 지표를 불러오는 중…','Loading mechanics metrics…')}</p>`;
    try{await load();if(!root.isConnected||root._metricsToken!==token)return;
      const key=root.id+':'+(fixedPlayer||''),state=contexts.get(key)||parseOptions(fixedPlayer);contexts.set(key,state);
      render(root,state,fixedPlayer);
    }catch(error){if(root.isConnected&&root._metricsToken===token)root.innerHTML=`<p class="db-error" role="alert">${t('지표 데이터베이스를 불러오지 못했습니다.','The metrics database could not be loaded.')} <button class="button-outline" data-db-retry>${t('다시 시도','Retry')}</button></p>`;root.querySelector('[data-db-retry]')?.addEventListener('click',()=>mount(root,fixedPlayer))}
  }
  function refresh(){if(document.querySelector('.shell')?.dataset.route==='database')mount(document.querySelector('#metrics-database-root'));else load().then(renderGuideDictionary).catch(()=>{});renderGuideDictionary()}
  function render(root,state,fixedPlayer,focusId){
    if(!root.isConnected)return;
    root._dbResize?.disconnect();
    const all=filtered(state),rows=sortRows(all,state),groups=sortGroups(grouped(rows),state),defs=dataset.dictionary,selected=defs.find(m=>m.id===state.metric),pageSize=20,maxPage=Math.max(0,Math.ceil((state.view==='aggregate'?groups.length:rows.length)/pageSize)-1);state.page=Math.min(state.page,maxPage);
    const baseRows=dataset.rows.filter(r=>(state.scope==='unlinked'?r.unit==='unlinked_clip':r.unit==='pitch_record')&&(!fixedPlayer||r.playerCode===fixedPlayer));
    const players=[...new Map(baseRows.map(r=>[r.playerCode,r.player])).entries()].filter(([code])=>code).sort((a,b)=>a[1].localeCompare(b[1],'ko'));
    const pitches=[...new Set(baseRows.map(r=>r.pitch||'unknown'))].sort(),games=[...new Set(baseRows.filter(r=>!state.player||r.playerCode===state.player).map(r=>r.gameId).filter(Boolean))].sort(),sources=[...new Set(baseRows.filter(r=>!state.player||r.playerCode===state.player).map(r=>r.sourceId).filter(Boolean))].sort();
    root.innerHTML=`<div class="db-workspace" data-db-context="${fixedPlayer?'player':'global'}">
      ${fixedPlayer?'':`<div class="db-heading"><div><span class="db-eyebrow">KBO · MECHANICS DATABASE</span><h1>${t('투구 메커니즘 지표','Pitch mechanics metrics')}</h1><p>${t('투구별 값을 찾고, 선수·구종별로 집계하고, 원본에서 확인하세요.','Find pitch-level values, summarize by player and pitch type, and inspect the source.')}</p></div><a class="button-outline" href="${e(dataUrl('pitches.csv.gz'))}" download>${t('전체 CSV.gz','Full CSV.gz')} ↓</a></div><div class="db-coverage"><span><strong>${dataset.scope.recordPitches.toLocaleString()}</strong>${t('투구 기록','pitch records')}</span><span><strong>${dataset.scope.linkedPitches}</strong>${t('동작 연결 투구','pitches linked to motion')}</span><span><strong>${dataset.scope.unlinkedClips}</strong>${t('미연결 클립','unlinked clips')}</span><span><strong>${defs.length}</strong>${t('정의된 지표','defined metrics')}</span></div>`}
      <div class="db-tabs" role="tablist" aria-label="${t('데이터 보기','Data view')}">${[['pitches',t('투구별 지표','Pitch metrics')],['aggregate',t('선수·구종별 집계','Player / pitch summary')],['dictionary',t('지표 사전','Metric dictionary')]].map(([v,label])=>`<button role="tab" data-db-view="${v}" aria-selected="${state.view===v}" tabindex="${state.view===v?'0':'-1'}">${label}</button>`).join('')}</div>
      ${state.view==='dictionary'?dictionaryMarkup():`<div class="db-filter-grid">
      <label>${t('자료 범위','Data scope')}<select id="${root.id}-scope" data-db-filter="scope">${[['linked',t('동작이 연결된 투구','Pitches with linked motion')],['records',t('전체 투구 기록','All pitch records')],['unlinked',t('미연결 동작 클립','Unlinked motion clips')]].map(([v,label])=>`<option value="${v}" ${state.scope===v?'selected':''}>${label}</option>`).join('')}</select></label>
      ${fixedPlayer?'':`<label>${t('선수','Player')}<select id="${root.id}-player" data-db-filter="player">${options(players.map(([code])=>code),state.player,t('모든 선수','All players'),code=>players.find(p=>p[0]===code)?.[1]+' · '+code)}</select></label>`}
      <label>${t('구종','Pitch type')}<select id="${root.id}-pitch" data-db-filter="pitch">${options(pitches,state.pitch,t('모든 구종','All pitch types'),p=>p==='unknown'?t('미확인','Unknown'):pitchLabel(p))}</select></label>
      <label>${t('경기','Game')}<select id="${root.id}-game" data-db-filter="game">${options(games,state.game,t('모든 경기','All games'))}</select></label>
      <label>${t('원본 영상','Source video')}<select id="${root.id}-source" data-db-filter="source">${options(sources,state.source,t('모든 원본','All sources'))}</select></label>
      <label>${t('이름·투구 ID 검색','Search name / pitch ID')}<input id="${root.id}-search" data-db-search type="search" value="${e(state.search)}"></label></div>
      <p class="db-scope-note">${state.scope==='unlinked'?t('미연결 클립은 같은 실제 투구의 리플레이일 수 있습니다. 이 보기의 행·집계 단위는 클립이며, 고유 투구 수로 해석하지 않습니다.','Unlinked clips may include replays of the same pitch. Rows and summary counts in this view refer to clips, not verified unique pitches.'):t('한 행은 경기 기록의 투구 하나입니다. 자동 연결은 사람 검수 전이며, 대체 영상이 있어도 투구 수에는 한 번만 포함합니다.','Each row is one provider pitch record. Automatic motion links await human review. Alternate clips do not add to the pitch denominator.')}</p>
      <details class="db-controls" ${state.controlsOpen?'open':''}><summary>${t('표 열·지표 묶음·품질 필터 설정','Configure columns, metric groups and quality filters')} · ${state.columns.length} ${t('지표 열','metric columns')}</summary><div class="db-column-tools"><span>${t('지표 묶음','Metric group')}</span>${[['default',t('기본 2D','Default 2D')],['pelvis_trunk',groupName('pelvis_trunk')],['lead_leg',groupName('lead_leg')],['arm',groupName('arm')],['posture',groupName('posture')],['timing',groupName('timing')],['space',groupName('space')]].map(([v,label])=>`<button data-db-preset="${v}" class="db-chip">${label}</button>`).join('')}<label><span class="sr-only">${t('추가할 지표','Metric to add')}</span><select data-db-add-choice>${defs.filter(m=>!state.columns.includes(m.id)).map(m=>`<option value="${m.id}">${e(name(m))}</option>`).join('')}</select></label><button data-db-add class="button-outline">+ ${t('열 추가','Add column')}</button></div>
      <div class="db-selected-columns">${state.columns.map(key=>`<button class="db-chip" data-db-remove="${key}" aria-label="${e(name(defs.find(m=>m.id===key)))} ${t('열 제거','remove column')}">${e(name(defs.find(m=>m.id===key)))} ×</button>`).join('')}</div>
      <div class="db-inspection-controls"><label>${t('분포·상태 필터의 기준 지표','Metric for distribution / status filter')}<select id="${root.id}-metric" data-db-filter="metric">${defs.map(m=>`<option value="${m.id}" ${m.id===state.metric?'selected':''}>${e(name(m))}</option>`).join('')}</select></label><label>${t('기준 지표의 산출 상태','Selected metric status')}<select id="${root.id}-status" data-db-filter="status">${[['',t('모든 상태','All statuses')],['observed_2d',statusName('observed_2d')],['provisional_event',statusName('provisional_event')],['needs_review_3d',statusName('needs_review_3d')],['missing',t('결측만','Missing only')]].map(([v,label])=>`<option value="${v}" ${state.status===v?'selected':''}>${e(label)}</option>`).join('')}</select></label><label class="db-check"><input type="checkbox" data-db-review ${state.include3d?'checked':''}>${t('검토 필요 3D를 집계·차트에 포함','Include review-required 3D in summaries / charts')}</label></div></details>
      <div class="db-table-actions"><p role="status"><strong>${(state.view==='aggregate'?groups.length:rows.length).toLocaleString()}</strong> ${state.view==='aggregate'?t('집계 행','summary rows'):t('행','rows')} · ${t('필터 범위','filtered scope')} ${rows.length.toLocaleString()} ${state.scope==='unlinked'?t('클립','clips'):t('투구','pitches')}</p><button class="button-outline" data-db-csv>${t('현재 열·필터 CSV','Selected columns / filtered CSV')} ↓</button><button class="button-outline" data-db-reset>${t('필터 초기화','Reset filters')}</button></div>
      <p class="db-quality-note">${state.include3d?`<strong class="db-review-active">${t('현재 검토 필요 3D 포함','Review-required 3D included')} · </strong>`:''}${state.status?`${e(state.status==='missing'?t('결측만','Missing only'):statusName(state.status))} · `:''}${t('2D는 관측 클립 구간의 투영 통계입니다. 이벤트 지표는 검수된 잠정 범위가 있는 표본만 제공합니다. 3D 추정값은 기본 집계에서 제외됩니다. 결측은 —로 표시하며 0으로 채우지 않습니다.','2D values describe the observed clip projection. Event metrics require provisional audited brackets. 3D estimates are excluded from default summaries. Missing values display as — and are never imputed as zero.')}</p>
      <div class="db-table-scroll" tabindex="0" role="region" aria-label="${t('지표 테이블, 열이 많으면 좌우로 이동','Metrics table; scroll horizontally for additional columns')}">${state.view==='aggregate'?aggregateTable(groups,state):pitchTable(rows.slice(state.page*pageSize,(state.page+1)*pageSize),state)}</div>
      <div class="db-pagination"><button class="button-outline" data-db-page="previous" ${state.page===0?'disabled':''}>← ${t('이전','Previous')}</button><span>${state.page+1} / ${maxPage+1} · ${t('페이지당 20행, CSV는 필터된 전체 행','20 rows per page; CSV exports every filtered row')}</span><button class="button-outline" data-db-page="next" ${state.page>=maxPage?'disabled':''}>${t('다음','Next')} →</button></div>
      <details class="db-plots" ${state.plotsOpen?'open':''}><summary>${t('선택 지표의 분포 · X/Y 산점도','Selected metric distribution · X/Y scatter')}</summary><div class="db-plot-controls"><button data-db-plot="distribution" aria-pressed="${state.plot==='distribution'}">${t('투구별 점 분포','Pitch-level distribution')}</button><button data-db-plot="scatter" aria-pressed="${state.plot==='scatter'}">${t('X/Y 산점도','X/Y scatter')}</button>${state.plot==='scatter'?`<label>X<select data-db-axis="x"><option value="speed" ${state.x==='speed'?'selected':''}>${t('구속 (km/h)','Velocity (km/h)')}</option>${defs.map(m=>`<option value="${m.id}" ${state.x===m.id?'selected':''}>${e(name(m))} (${unit(m)})</option>`).join('')}</select></label><label>Y<select data-db-axis="y">${defs.map(m=>`<option value="${m.id}" ${state.y===m.id?'selected':''}>${e(name(m))} (${unit(m)})</option>`).join('')}</select></label>`:''}<button class="db-text-button" data-db-definition="${selected.id}">${t('기준 지표 정의 보기','View selected metric definition')}</button></div><div class="db-plot-host"></div><p class="db-chart-caption"></p></details>`}
    </div>`;
    const rerender=focus=>{save(state);render(root,state,fixedPlayer,focus)};
    root.querySelectorAll('[data-db-view]').forEach(button=>{
      button.onclick=()=>{state.view=button.dataset.dbView;state.page=0;rerender()};
      button.onkeydown=event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const views=['pitches','aggregate','dictionary'],index=views.indexOf(state.view);state.view=event.key==='Home'?views[0]:event.key==='End'?views[2]:views[(index+(event.key==='ArrowRight'?1:2))%3];state.page=0;rerender();root.querySelector(`[data-db-view="${state.view}"]`).focus()}};
    });
    root.querySelectorAll('[data-db-definition]').forEach(button=>button.onclick=()=>openDefinition(button.dataset.dbDefinition));
    if(state.view==='dictionary'){bindDictionary(root);save(state);return}
    root.querySelector('.db-controls').ontoggle=event=>state.controlsOpen=event.target.open;
    root.querySelectorAll('[data-db-filter]').forEach(select=>select.onchange=()=>{state[select.dataset.dbFilter]=select.value;state.page=0;if(select.dataset.dbFilter==='metric'){state.plotsOpen=true;state.status=''}if(['player','scope'].includes(select.dataset.dbFilter)){state.game='';state.source='';if(select.dataset.dbFilter==='scope'&&!fixedPlayer)state.player=''}rerender(select.id)});
    root.querySelector('[data-db-search]').oninput=event=>{const position=event.target.selectionStart;state.search=event.target.value;state.page=0;rerender(event.target.id);root.querySelector('[data-db-search]').setSelectionRange(position,position)};
    root.querySelectorAll('[data-db-preset]').forEach(button=>button.onclick=()=>{state.columns=button.dataset.dbPreset==='default'?[...defaults]:defs.filter(m=>m.group===button.dataset.dbPreset).map(m=>m.id);rerender()});
    root.querySelector('[data-db-add]').onclick=()=>{const value=root.querySelector('[data-db-add-choice]').value;if(value&&!state.columns.includes(value))state.columns.push(value);rerender()};
    root.querySelectorAll('[data-db-remove]').forEach(button=>button.onclick=()=>{state.columns=state.columns.filter(k=>k!==button.dataset.dbRemove);rerender()});
    root.querySelector('[data-db-review]').onchange=event=>{state.include3d=event.target.checked;rerender()};
    root.querySelectorAll('[data-db-distribution]').forEach(button=>button.onclick=()=>{state.metric=button.dataset.dbDistribution;state.plotsOpen=true;state.plot='distribution';rerender();root.querySelector('.db-plots').scrollIntoView({block:'start'})});
    root.querySelectorAll('[data-db-sort]').forEach(button=>button.onclick=()=>{state.direction=state.sort===button.dataset.dbSort&&state.direction==='desc'?'asc':'desc';state.sort=button.dataset.dbSort;rerender()});
    root.querySelectorAll('[data-db-page]').forEach(button=>button.onclick=()=>{state.page+=button.dataset.dbPage==='next'?1:-1;render(root,state,fixedPlayer);root.querySelector('.db-table-scroll').scrollIntoView({block:'start'})});
    root.querySelector('[data-db-reset]').onclick=()=>{Object.assign(state,{scope:'linked',player:fixedPlayer||'',pitch:'',game:'',source:'',search:'',status:'',include3d:false,page:0});rerender()};
    root.querySelector('[data-db-csv]').onclick=()=>download(`kbo-mechanics-${state.view}-${VERSION}.csv`,state.view==='aggregate'?aggregateCsv(groups,state.columns,state.include3d):exportPitches(rows,state.columns));
    root.querySelectorAll('[data-db-row]').forEach(button=>button.onclick=()=>openRow(dataset.rows.find(r=>r.id===button.dataset.dbRow),state.columns));
    root.querySelectorAll('[data-db-player]').forEach(button=>button.onclick=()=>openPlayerContext(button.dataset.dbPlayer,{scope:state.scope,pitch:state.pitch,game:state.game,source:state.source,columns:state.columns,review:state.include3d?'yes':'no'}));
    root.querySelectorAll('[data-db-group]').forEach(button=>button.onclick=()=>{const g=groups.find(g=>g.id===button.dataset.dbGroup);state.player=g.playerCode;state.pitch=g.pitch||'unknown';state.view='pitches';state.page=0;state.plotsOpen=true;if(button.dataset.dbGroupMetric)state.metric=button.dataset.dbGroupMetric;rerender()});
    const observePlot=()=>{const host=root.querySelector('.db-plot-host');let width=host.clientWidth;root._dbResize?.disconnect();root._dbResize=new ResizeObserver(()=>{if(host.clientWidth>0&&host.clientWidth!==width){width=host.clientWidth;drawPlot(root,rows,state)}});root._dbResize.observe(host)};
    root.querySelector('.db-plots').ontoggle=event=>{state.plotsOpen=event.target.open;save(state);if(event.target.open){drawPlot(root,rows,state);observePlot()}};
    root.querySelectorAll('[data-db-plot]').forEach(button=>button.onclick=()=>{state.plot=button.dataset.dbPlot;state.plotsOpen=true;rerender()});
    root.querySelectorAll('[data-db-axis]').forEach(select=>select.onchange=()=>{state[select.dataset.dbAxis]=select.value;rerender()});
    if(state.plotsOpen){drawPlot(root,rows,state);observePlot()};
    if(focusId)root.querySelector('#'+focusId)?.focus({preventScroll:true});save(state);
  }
  function header(key,label,state,metric=false){return `<th aria-sort="${state.sort===key?(state.direction==='asc'?'ascending':'descending'):'none'}"><button data-db-sort="${key}">${e(label)} ${state.sort===key?(state.direction==='asc'?'↑':'↓'):'↕'}</button>${metric?`<div class="db-column-actions"><button class="db-define" data-db-definition="${key}">${t('정의','Definition')}</button><button class="db-define" data-db-distribution="${key}">${t('분포','Distribution')}</button></div>`:''}</th>`}
  function metricCell(value,m){return `<td class="db-value-cell" data-metric="${m.id}" data-value="${value.value==null?'':value.value}" data-status="${value.status}"><span class="db-number">${format(value.value)}${number(value.value)?`<small>${unit(m)}</small>`:''}</span><span class="db-cell-status ${value.status==='needs_review_3d'?'review':value.status==='provisional_event'?'provisional':''}">${e(statusName(value.status))}</span>${number(value.value)?`<span class="db-frame-count">${t('유효','valid')} ${value.validFrames}/${value.totalFrames} ${t('프레임','frames')}</span>`:''}</td>`}
  function pitchTable(rows,state){return `<table class="db-pitch-table"><caption class="sr-only">${t('한 행은 투구 기록 또는 별도 표시된 미연결 클립 하나','One row per pitch record, or per explicitly separated unlinked clip')}</caption><thead><tr>${header('player',t('선수','Player'),state)}${header('pitch',t('구종','Pitch type'),state)}${header('speed',t('구속 (km/h)','Speed (km/h)'),state)}${header('date',t('날짜 · 경기','Date / game'),state)}${state.columns.map(key=>header(key,`${name(dataset.dictionary.find(m=>m.id===key))} (${unit(dataset.dictionary.find(m=>m.id===key))})`,state,true)).join('')}<th>${t('근거','Evidence')}</th></tr></thead><tbody>${rows.map(row=>`<tr data-row-id="${e(row.id)}"><td><button class="db-text-button" data-db-player="${e(row.playerCode)}">${e(row.player)}</button><small>${e(row.playerCode)} · ${row.hand||'—'}</small></td><td>${e(row.pitch?pitchLabel(row.pitch):t('미확인','Unknown'))}</td><td>${format(row.speed)}</td><td>${e(row.date)}<small>${e(row.gameId||row.sourceId||'—')}</small></td>${state.columns.map(key=>metricCell(cell(row,key),dataset.dictionary.find(m=>m.id===key))).join('')}<td><button class="button-outline db-detail-button" data-db-row="${e(row.id)}">${t('투구 상세','Pitch details')}</button><small>${row.primaryClipId?`${row.clipIds.length} ${t('클립','clip(s)')}`:t('동작 없음','No motion')}</small></td></tr>`).join('')||`<tr><td colspan="${5+state.columns.length}">${t('조건에 맞는 행이 없습니다. 필터를 초기화해 보세요.','No matching rows. Try resetting the filters.')}</td></tr>`}</tbody></table>`}
  function spark(values){if(!values.length)return '';const lo=Math.min(...values),hi=Math.max(...values),x=v=>8+(hi===lo?.5:(v-lo)/(hi-lo))*100;return `<svg class="db-spark" viewBox="0 0 116 22" aria-hidden="true"><path d="M8 11H108"/>${values.map((v,i)=>`<circle cx="${x(v)}" cy="${8+(i%3)*3}" r="2"/>`).join('')}</svg>`}
  function aggregateTable(groups,state){
    const sorted=groups;
    return `<table class="db-aggregate-table"><caption>${t('중앙값 · 사분위 범위(IQR) · 지표별 유효 표본 n. IQR은 n≥3일 때만 표시합니다.','Median · interquartile range (IQR) · valid n per metric. IQR requires n≥3.')}</caption><thead><tr>${header('player',t('선수','Player'),state)}${header('pitch',t('구종','Pitch type'),state)}<th>${t('범위 표본','Rows in scope')}</th>${state.columns.map(key=>header(key,`${name(dataset.dictionary.find(m=>m.id===key))} (${unit(dataset.dictionary.find(m=>m.id===key))})`,state,true)).join('')}<th>${t('개별 투구','Individual pitches')}</th></tr></thead><tbody>${sorted.slice(state.page*20,(state.page+1)*20).map(g=>`<tr><td><button class="db-text-button" data-db-player="${e(g.playerCode)}">${e(g.player)}</button></td><td>${e(g.pitch?pitchLabel(g.pitch):t('미확인','Unknown'))}</td><td>${g.rows.length} ${g.unit==='unlinked_clip'?t('클립','clips'):t('투구','pitches')}<small>${new Set(g.rows.map(r=>r.gameId).filter(Boolean)).size} ${t('경기 기록','recorded games')}</small></td>${state.columns.map(key=>{const s=summarize(g.rows,key,state.include3d);return `<td><button class="db-summary-value" data-db-group="${e(g.id)}" data-db-group-metric="${key}"><strong>${format(s.median)}</strong><span>IQR ${s.n>=3?`${format(s.q25)}–${format(s.q75)}`:'—'} · n=${s.n}</span>${spark(s.values)}</button></td>`}).join('')}<td><button class="button-outline" data-db-group="${e(g.id)}">${t('투구 보기','View pitches')} →</button></td></tr>`).join('')||`<tr><td colspan="${4+state.columns.length}">${t('조건에 맞는 집계가 없습니다.','No matching summary groups.')}</td></tr>`}</tbody></table>`;
  }
  function showDialog(title,html){
    if(!dialog){dialog=document.createElement('dialog');dialog.className='db-dialog';dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close()});dialog.addEventListener('close',()=>lastFocus?.isConnected&&lastFocus.focus({preventScroll:true}));document.body.append(dialog)}
    if(dialog.open)dialog.close();lastFocus=document.activeElement;
    dialog.innerHTML=`<div class="db-dialog-head"><h2 id="db-dialog-title">${e(title)}</h2><button class="button-outline" data-db-close aria-label="${t('닫기','Close')}">×</button></div>${html}`;dialog.setAttribute('aria-labelledby','db-dialog-title');dialog.querySelector('[data-db-close]').onclick=()=>dialog.close();dialog.showModal();
  }
  function openRow(row,columns=defaults){
    if(!row)return;
    showDialog(`${row.player||''} · ${row.pitch?pitchLabel(row.pitch):t('구종 미확인','Pitch type unknown')} · ${row.speed??'—'} km/h`,
      `<p class="db-detail-id">${e(row.id)} · ${e(row.date)} · ${e(row.gameId||row.sourceId)}</p><p>${e(statusName(row.linkStatus==='no_motion'?'no_motion':row.metrics?'observed_2d':'no_motion'))} · ${t('투구 기록 연결','Pitch-record link')}: ${e(({auto_accepted:t('자동 연결 · 미검수','Automatically linked · unreviewed'),no_motion:t('연결 동작 없음','No linked motion'),held:t('보류','Held'),unresolved:t('미연결','Unresolved')})[row.linkStatus]||row.linkStatus)}</p><div class="db-detail-actions">${row.primaryClipId?`<button class="button-primary" data-db-open-clip>${t('동작·파형 상세 열기','Open motion and curve details')} ↗</button><a href="${e(row.sourceUrl)}" target="_blank" rel="noopener noreferrer">${t('원본 중계','Original broadcast')} ↗</a>`:`<span>${t('이 투구에 연결된 동작 영상이 없습니다.','No motion clip is linked to this pitch.')}</span>`}${row.recordSourceUrl?`<a href="${e(row.recordSourceUrl)}" target="_blank" rel="noopener noreferrer">${t('경기 투구 기록','Game pitch record')} ↗</a>`:''}${row.playerCode?`<button class="button-outline" data-db-open-player>${t('선수 집계 보기','Player summary')}</button>`:''}</div><div class="db-detail-cells">${dataset.dictionary.map(m=>{const v=cell(row,m.id);return `<article><h3><button class="db-text-button" data-db-definition="${m.id}">${e(name(m))}</button></h3><strong>${format(v.value)} ${number(v.value)?unit(m):''}</strong><span class="db-cell-status">${e(statusName(v.status))}</span><p>${e(definition(m))}</p><small>${m.calculationVersion} · ${v.validFrames}/${v.totalFrames} ${t('유효 프레임','valid frames')}</small></article>`}).join('')}</div><p>${t('원본 클립','Source clip')}: ${e(row.primaryClipId||'—')} · ${t('대체 클립','Alternate clips')}: ${e(row.clipIds.join(', ')||'—')}</p>`);
    dialog.querySelector('[data-db-open-clip]')?.addEventListener('click',()=>{dialog.close();PitchPortal.openClip(row.primaryClipId)});
    dialog.querySelector('[data-db-open-player]')?.addEventListener('click',()=>{dialog.close();openPlayerContext(row.playerCode,{scope:row.unit==='unlinked_clip'?'unlinked':row.primaryClipId?'linked':'records',pitch:row.pitch||'',columns})});
    dialog.querySelectorAll('[data-db-definition]').forEach(button=>button.onclick=()=>openDefinition(button.dataset.dbDefinition));
  }
  function openDefinition(key){const m=dataset.dictionary.find(m=>m.id===key);if(!m)return;showDialog(name(m),definitionMarkup(m))}
  function definitionMarkup(m){
    const conditions=m.referenceEvent==='provisional_landing_release'?t('잠정 착지·릴리스 범위 필요. 세 관절 점수 ≥0.4, 보간·이상값 제외, 3프레임 평균. 엄격 기준 및 이벤트 민감도는 기존 파일럿 JSON에 있습니다.',m.conditions):m.pendingReason?t('필요한 기준 이벤트·실제 시간·투구 기준축의 검증 전으로, 현재 값은 결측입니다.',m.conditions):t('필요 관절의 원본 2D 점수 ≥0.5, 보간·이상값 제외. 유효 관측 10프레임 이상이며 클립의 50% 이상. 누락 구간을 채우지 않습니다.',m.conditions);
    return `<dl class="db-definition"><dt>${t('지표 ID','Metric ID')}</dt><dd><code>${m.id}</code></dd><dt>${t('정의','Definition')}</dt><dd>${e(definition(m))}</dd><dt>${t('단위','Unit')}</dt><dd>${unit(m)}</dd><dt>${t('좌표계','Coordinate system')}</dt><dd>${e(m.coordinateSystem)}</dd><dt>${t('기준 이벤트·구간','Reference event / interval')}</dt><dd>${m.referenceEvent==='observed_clip'?t('관측된 클립 구간 전체. 영상 시작·끝과 촬영 조건이 다를 수 있습니다.','Entire observed clip; boundaries and filming conditions may vary.'):m.referenceEvent==='provisional_landing_release'?t('잠정 착지–릴리스 범위 중앙 정렬','Midpoint alignment of provisional landing / release brackets'):t('검증된 착지–릴리스 이벤트 필요','Audited landing / release events required')}</dd><dt>${t('필요 관절','Required joints')}</dt><dd>${e(m.requiredJoints)}</dd><dt>${t('계산식','Formula')}</dt><dd><code>${e(m.formula)}</code></dd><dt>${t('계산 버전','Calculation version')}</dt><dd>${e(m.calculationVersion)}</dd><dt>${t('산출 조건','Calculation conditions')}</dt><dd>${e(conditions)}</dd><dt>${t('해석 범위','Interpretation')}</dt><dd>${m.basis==='2d'?t('영상 투영과 클립 구간의 영향을 받습니다. 해부학적 3D 관절각이나 시즌 전체의 선수 특성이 아닙니다. 프레임 수에는 중복 영상이 포함될 수 있으며 독립 표본 수나 정확도를 뜻하지 않습니다.',m.limitations):t('검증 전 단안 3D 추정 기하량입니다. 기본 집계에서 제외하며 실험실 기준값과 비교하지 않습니다.',m.limitations)}</dd></dl>`;
  }
  function dictionaryMarkup(){return `<div class="db-dictionary"><p>${t('값이 없어도 정의와 결측 사유는 유지합니다. 관측 구간 통계와 착지–릴리스 지표는 별개의 열입니다.','Definitions and missing reasons are retained even when no value is available. Observed-clip statistics and event-aligned metrics use separate columns.')}</p><a href="${e(dataUrl('dictionary.json'))}" download>${t('지표 사전 JSON','Metric dictionary JSON')} ↓</a>${dataset.dictionary.map(m=>`<details><summary>${e(name(m))} <span>${unit(m)} · ${groupName(m.group)} · ${m.basis.toUpperCase()}</span></summary>${definitionMarkup(m)}</details>`).join('')}</div>`}
  function bindDictionary(root){root.querySelectorAll('[data-db-definition]').forEach(button=>button.onclick=()=>openDefinition(button.dataset.dbDefinition))}
  function renderGuideDictionary(){const root=document.querySelector('#metric-dictionary');if(dataset&&root){root.innerHTML=dictionaryMarkup();bindDictionary(root)}}
  function drawPlot(root,rows,state){
    const host=root.querySelector('.db-plot-host'),caption=root.querySelector('.db-chart-caption');if(!host||!root.querySelector('.db-plots').open)return;
    const defs=dataset.dictionary,metric=defs.find(m=>m.id===state.metric),width=Math.max(260,Math.round(host.clientWidth||700)),height=300,left=60,right=24,top=32,bottom=height-54;
    const plotted=rows.filter(row=>state.plot==='scatter'?(state.x==='speed'?number(row.speed):eligible(cell(row,state.x),state.include3d))&&eligible(cell(row,state.y),state.include3d):eligible(cell(row,state.metric),state.include3d));
    if(!plotted.length){
      const axisKeys=state.plot==='scatter'?[state.x,state.y]:[state.metric],reasons=axisKeys.map(key=>{const counts={};for(const row of rows){const status=key==='speed'?(number(row.speed)?'observed_2d':'insufficient_observations'):cell(row,key).status;counts[status]=(counts[status]||0)+1}return `${key==='speed'?t('구속','Velocity'):name(defs.find(m=>m.id===key))}: ${Object.entries(counts).map(([k,n])=>`${statusName(k)} ${n}`).join(', ')}`});
      host.innerHTML=`<p class="db-empty-plot">${t('현재 조건에서 차트에 사용할 유효 값이 없습니다.','No eligible values under these conditions.')}</p><p>${e(reasons.join(' · '))}</p>`;caption.textContent=`n=0 / ${rows.length} · ${t('검토 필요 3D는 명시적으로 포함한 경우에만 사용합니다.','Review-required 3D is used only when explicitly included.')}`;return;
    }
    const ykey=state.plot==='scatter'?state.y:state.metric,xvalues=plotted.map(row=>state.plot==='scatter'?(state.x==='speed'?row.speed:cell(row,state.x).value):cell(row,state.metric).value),yvalues=plotted.map(row=>cell(row,ykey).value);
    const padded=values=>{const a=Math.min(...values),b=Math.max(...values),p=(b-a||Math.max(1,Math.abs(a)*.1))*.08;return [a-p,b+p]},[xmin,xmax]=padded(xvalues),[ymin,ymax]=padded(yvalues),x=v=>left+(v-xmin)/(xmax-xmin)*(width-left-right),y=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
    const points=plotted.map((row,i)=>({row,x:x(xvalues[i]),y:state.plot==='scatter'?y(yvalues[i]):130+((i%9)-4)*7}));
    let background='';
    if(state.plot==='distribution'){
      const stats=summarize(plotted,state.metric,state.include3d);
      background=`${stats.n>=3?`<rect x="${x(stats.q25)}" y="87" width="${Math.max(1,x(stats.q75)-x(stats.q25))}" height="86" class="db-iqr-band"/>`:''}<path d="M${x(stats.median)} 70V190" class="db-median-line"/><text x="${left}" y="25">${t('중앙값','Median')} ${format(stats.median)} ${unit(metric)} · n=${stats.n}</text>`;
    }else background=[0,.5,1].map(p=>{const v=ymin+p*(ymax-ymin);return `<path class="db-plot-grid" d="M${left} ${y(v)}H${width-right}"/><text x="${left-9}" y="${y(v)+5}" text-anchor="end">${format(v)}</text>`}).join('');
    host.innerHTML=`${state.plot==='scatter'?`<p class="db-axis-label">Y ↑: ${e(name(defs.find(m=>m.id===state.y)))} (${unit(defs.find(m=>m.id===state.y))})</p>`:''}<svg class="db-data-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${e(state.plot==='scatter'?t('선택한 X/Y 지표 산점도','Scatter of selected X/Y variables'):name(metric))}">${background}<path class="db-axis" d="M${left} ${bottom}H${width-right}"/>${[0,.5,1].map(p=>`<text x="${left+p*(width-left-right)}" y="${bottom+25}" text-anchor="middle">${format(xmin+p*(xmax-xmin))}</text>`).join('')}${points.map(({row,x,y})=>`<circle class="db-data-point" data-db-point="${e(row.id)}" tabindex="0" role="button" aria-label="${e(row.player)} ${e(row.pitch||'')} ${format(cell(row,ykey).value)}" cx="${x}" cy="${y}" r="6"><title>${e(row.player)} · ${e(row.pitch||'')} · ${format(cell(row,ykey).value)} · ${e(row.id)}</title></circle>`).join('')}</svg><p class="db-axis-label">${e(state.plot==='scatter'?(state.x==='speed'?t('구속 (km/h)','Velocity (km/h)'):name(defs.find(m=>m.id===state.x))+' ('+unit(defs.find(m=>m.id===state.x))+')'):name(metric)+' ('+unit(metric)+')')}${state.plot==='scatter'?` · Y: ${e(name(defs.find(m=>m.id===state.y)))} (${unit(defs.find(m=>m.id===state.y))})`:''}</p>`;
    host.querySelectorAll('[data-db-point]').forEach(point=>{const action=()=>openRow(dataset.rows.find(r=>r.id===point.dataset.dbPoint),state.columns);point.onclick=action;point.onkeydown=event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();action()}}});
    caption.textContent=`${plotted.length} / ${rows.length} ${t('행 사용. 점 하나는 현재 범위의 투구 또는 미연결 클립 하나입니다. 분포 띠는 투구 간 IQR이며 추정 신뢰구간이 아닙니다. 서로 다른 클립 경계·촬영 조건을 고려하세요. 산점도는 관찰 관계이며 원인을 설명하지 않습니다.','rows used. Each dot is one pitch or unlinked clip in the current scope. The band is across-row IQR, not estimation confidence. Consider differing clip boundaries and filming conditions. Scatter shows association, not causation.')}`;
  }
  return {install,load,mount,refresh,cell,eligible,summarize,exportPitches,aggregateCsv,grouped,filtered,sortRows,clearPlayerLink,openRow,renderGuideDictionary,get data(){return dataset}};
})();
