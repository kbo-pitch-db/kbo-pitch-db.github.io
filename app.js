const I18N={en:{navExplore:'Explore clips',originalVideo:'Original video',overlayVideo:'Pose overlay',missingOverlay:'Overlay video is not included in this package.',navGuide:'Data guide',staticData:'Static dataset',eyebrow:'AN OPEN MOTION ARCHIVE',heroTitle:'Every delivery,<br><em>in a new light.</em>',heroDesc:'Explore KBO pitching motion through 2D skeleton animations. Find a pitcher, study the movement, and download the pose data.',browse:'Browse the collection',heroNote:'A visual research archive for the game, its players, and the people who coach them.',frameAnalysis:'FRAME ANALYSIS',poseOverlay:'2D POSE · SKELETON VIEW',motionSnapshot:'MOTION SNAPSHOT',keypoints:'keypoints<br>tracked',library:'THE LIBRARY',collectionTitle:'Find your next <em>frame.</em>',collectionDesc:'Search and filter the archive. Select a clip to watch its 2D pose animation.',searchPlaceholder:'Search pitchers, teams or clips',team:'Team',allTeams:'All teams',hand:'Throwing hand',allHands:'All hands',pitch:'Pitch type',allPitches:'All pitches',clear:'Clear',clipsFound:'clips found',curated:'CURATED DATASET',loading:'Loading clips…',emptyTitle:'No clips found',emptyDesc:'Try another name or clear one of the filters.',clearFilters:'Clear all filters',yourWorkspace:'YOUR WORKSPACE',selectClip:'Select a clip<br>to get started.',detailHint:'Choose a clip to watch its 2D skeleton animation and inspect pose data.',stepFind:'Find a pitcher',stepStudy:'Study each frame',stepDownload:'Take the data',guideEyebrow:'BUILT FOR EXPLORATION',guideTitle:'From broadcast<br>to <em>better questions.</em>',guideDesc:'Each clip includes a 2D skeleton animation and source metadata. Timestamped YouTube links let you view the original footage.',guideDisclaimer:'Pose keypoints are image-space estimates, not validated biomechanical measurements. Treat them as a visual starting point for exploration.',formatTitle:'Open, readable files',formatDesc:'Pose JSON contains COCO-17 keypoints, frame number, timestamp, and confidence. CSV export makes quick review simple.',workflowTitle:'Make it your own',workflowDesc:'Run your own models, compare clips, or build a research workflow on top of the files. Additions stay portable.',contextTitle:'Know what you’re seeing',contextDesc:'Source, frame rate, pose model, coordinate format, and missing frames are shown with every clip.',flowStep1:'BROWSE',flowStep2:'INSPECT',flowStep3:'DOWNLOAD',sourceTag:'SOURCE',frameTag:'FRAME RATE',statusTag:'POSE STATUS',bannerKicker:'THE GAME, IN EVERY FRAME',bannerTitle:'Start with a pitch.<br><em>See what you find.</em>',exploreClips:'Explore clips',footerNote:'An open, evolving archive for the study of pitching motion.',staticSite:'STATIC SITE',unknownPitcher:'Unknown pitcher',unknownTeam:'Unknown team',unknown:'Not provided',right:'Right',left:'Left',notRecorded:'Not recorded',missingVideo:'Video file is not included in this data package.',missingPose:'Pose file is missing or unavailable for this clip.',badData:'Clip data could not be loaded. Check data/clips.json and try again.',example:'ILLUSTRATIVE SAMPLE',estimated:'Estimated',frame:'FRAME',play:'Play',pause:'Pause',poseOn:'Pose on',poseOff:'Pose off',downloadJson:'POSE JSON',downloadCsv:'POSE CSV',downloadVideo:'VIDEO',source:'Source',fps:'FPS',model:'Pose model',coords:'Coordinates',poseStatus:'Pose status',clipDuration:'Duration',frameCount:'Frames',teamLabel:'Team',handLabel:'Throwing hand',pitchLabel:'Pitch type',dateLabel:'Game date',inningLabel:'Inning',pitchNumberLabel:'Pitcher pitch #',confidence:'Joint confidence',noPose:'Pose data unavailable',noConfidence:'Not provided',filesMissing:'This clip has no downloadable pose data.',searchCount:'Search results',npzUnavailable:'The original NPZ pose file is not part of this package.'},ko:{navExplore:'클립 탐색',originalVideo:'원본 영상',overlayVideo:'포즈 오버레이',missingOverlay:'이 패키지에 오버레이 영상이 없습니다.',navGuide:'데이터 안내',staticData:'정적 데이터셋',eyebrow:'열린 투구 동작 아카이브',heroTitle:'모든 투구를,<br><em>새로운 시선으로.</em>',heroDesc:'KBO 투구 동작을 2D 관절 애니메이션으로 살펴보세요. 투수를 찾고 움직임을 확인한 뒤 포즈 데이터를 내려받을 수 있습니다.',browse:'클립 둘러보기',heroNote:'야구와 선수, 코치를 위한 시각 연구 아카이브입니다.',frameAnalysis:'프레임 분석',poseOverlay:'2D 포즈 · 관절 보기',motionSnapshot:'동작 스냅샷',keypoints:'추적 관절점',library:'클립 라이브러리',collectionTitle:'다음 <em>프레임을</em> 찾아보세요.',collectionDesc:'검색과 필터로 자료를 살펴보세요. 클립을 선택하면 2D 포즈 영상을 확인할 수 있습니다.',searchPlaceholder:'투수, 팀 또는 클립 검색',team:'팀',allTeams:'모든 팀',hand:'투구 손',allHands:'모든 투수',pitch:'구종',allPitches:'모든 구종',clear:'초기화',clipsFound:'개 클립',curated:'정리된 데이터',loading:'클립을 불러오는 중…',emptyTitle:'클립이 없습니다',emptyDesc:'다른 검색어를 입력하거나 필터를 초기화해 보세요.',clearFilters:'필터 모두 지우기',yourWorkspace:'클립 작업 공간',selectClip:'클립을 선택해<br>시작하세요.',detailHint:'목록에서 클립을 선택하면 2D 관절 애니메이션과 포즈 데이터를 확인할 수 있습니다.',stepFind:'투수 찾기',stepStudy:'프레임 확인',stepDownload:'데이터 받기',guideEyebrow:'탐색을 위한 데이터',guideTitle:'중계 영상에서<br><em>더 나은 질문으로.</em>',guideDesc:'각 클립은 2D 관절 애니메이션과 출처 정보를 제공합니다. 시간 정보가 포함된 YouTube 링크로 원본 장면을 확인할 수 있습니다.',guideDisclaimer:'포즈 관절점은 영상 좌표 기반 추정치이며 검증된 생체역학 측정값이 아닙니다. 시각적 탐색의 출발점으로 활용하세요.',formatTitle:'열어 볼 수 있는 파일',formatDesc:'포즈 JSON에 COCO-17 관절점, 프레임 번호, 시각, 신뢰도 값이 담깁니다. 빠르게 확인할 수 있도록 CSV도 제공합니다.',workflowTitle:'원하는 방식으로 활용',workflowDesc:'직접 실행한 모델을 적용하거나 클립을 비교하고 연구 작업 흐름을 구성해 보세요. 파일은 그대로 옮겨 쓸 수 있습니다.',contextTitle:'데이터 맥락 확인',contextDesc:'출처, 프레임 레이트, 포즈 모델, 좌표 형식과 누락 프레임 정보를 각 클립에 표시합니다.',flowStep1:'탐색',flowStep2:'확인',flowStep3:'다운로드',sourceTag:'출처',frameTag:'프레임 레이트',statusTag:'포즈 상태',bannerKicker:'모든 프레임에서 야구를',bannerTitle:'투구에서 시작해,<br><em>새로운 장면을 찾으세요.</em>',exploreClips:'클립 탐색',footerNote:'투구 동작 연구를 위한 개방형 아카이브입니다.',staticSite:'정적 사이트',unknownPitcher:'투수 미상',unknownTeam:'팀 미상',unknown:'정보 없음',right:'우투',left:'좌투',notRecorded:'기록 없음',missingVideo:'이 데이터 패키지에는 영상 파일이 없습니다.',missingPose:'이 클립의 포즈 파일이 없거나 확인할 수 없습니다.',badData:'클립 데이터를 불러오지 못했습니다. data/clips.json을 확인해 주세요.',example:'예시 데이터',estimated:'추정',frame:'프레임',play:'재생',pause:'일시정지',poseOn:'포즈 켜기',poseOff:'포즈 끄기',downloadJson:'포즈 JSON',downloadCsv:'포즈 CSV',downloadVideo:'영상',source:'출처',fps:'FPS',model:'포즈 모델',coords:'좌표 형식',poseStatus:'포즈 상태',clipDuration:'재생 시간',frameCount:'프레임 수',teamLabel:'팀',handLabel:'투구 손',pitchLabel:'구종',dateLabel:'경기 날짜',inningLabel:'이닝',pitchNumberLabel:'투수 투구 순서',confidence:'관절 신뢰도',noPose:'포즈 데이터 없음',noConfidence:'미제공',filesMissing:'다운로드할 포즈 데이터가 없습니다.',searchCount:'검색 결과',npzUnavailable:'원본 NPZ 포즈 파일은 이 패키지에 포함되지 않았습니다.'}};
I18N.en.reliabilityReport='3D quality report';I18N.ko.reliabilityReport='3D 품질 검토 보고서';I18N.en.estimateOnly='Estimate · 3D accuracy unvalidated';I18N.ko.estimateOnly='추정치 · 3D 정확도 미검증';I18N.en.poseStatus='2D status';I18N.ko.poseStatus='2D 상태';I18N.en.only3d='3D available';I18N.ko.only3d='3D 제공 클립만';I18N.en.pitchersFound='groups'; I18N.ko.pitchersFound='개 묶음'; I18N.en.team='Matchup'; I18N.ko.team='대진'; I18N.en.allTeams='All matchups'; I18N.ko.allTeams='모든 대진'; I18N.en.teamLabel='Matchup'; I18N.ko.teamLabel='대진'; I18N.en.scrollToBrowse='SCROLL TO BROWSE ↓'; I18N.ko.scrollToBrowse='목록 안에서 스크롤 ↓'; I18N.en.delivery='Delivery'; I18N.ko.delivery='투구'; I18N.en.autoName='AUTO LINKED NAME'; I18N.ko.autoName='자동 연결 이름'; I18N.en.unidentified='UNIDENTIFIED'; I18N.ko.unidentified='신원 미확인'; I18N.en.moreFromPitcher='More from this pitcher'; I18N.ko.moreFromPitcher='이 투수의 다른 영상'; I18N.en.clipCount='clips'; I18N.ko.clipCount='개 영상'; I18N.en.clipSingular='clip'; I18N.en.playbackUnavailable='This browser cannot play the MP4. Download the clip to view it.'; I18N.ko.playbackUnavailable='이 브라우저에서 MP4를 재생할 수 없습니다. 클립을 내려받아 확인하세요.';
I18N.en.poseVideo='2D pose animation';I18N.ko.poseVideo='2D 포즈 영상';I18N.en.downloadVideo='POSE MP4';I18N.ko.downloadVideo='포즈 MP4';
I18N.en.profilePhoto='profile photo';I18N.ko.profilePhoto='프로필 사진';I18N.en.photoSource='Profile photo: KBO ↗';I18N.ko.photoSource='프로필 사진: KBO ↗';
Object.assign(I18N.en,{scoreboardName:'SCOREBOARD NAME',reviewedName:'BROADCAST TEXT REVIEWED',nameEvidence:'Name source'});
Object.assign(I18N.ko,{scoreboardName:'점수판 반복 확인',reviewedName:'중계 표기 판독 확인',nameEvidence:'선수명 확인 근거'});
Object.assign(I18N.en,{youtubeSource:'Original on YouTube',youtubeOpen:'Open on YouTube ↗',youtubeShow:'Show source video',youtubeHide:'Hide source video',youtubeNote:'Plays on YouTube at the recorded source time. If playback is restricted, open the link above. Pose playback is controlled separately.'});
Object.assign(I18N.ko,{youtubeSource:'YouTube 원본',youtubeOpen:'YouTube에서 열기 ↗',youtubeShow:'원본 영상 펼치기',youtubeHide:'원본 영상 접기',youtubeNote:'기록된 원본 시각에서 YouTube 영상을 재생합니다. 재생이 제한되면 위 링크를 이용하세요. 포즈 영상은 별도로 조작합니다.'});
const $=s=>document.querySelector(s);function getSavedLanguage(){try{return localStorage.getItem('kbo-lang')==='ko'?'ko':'en'}catch(e){return'en'}}function saveLanguage(v){try{localStorage.setItem('kbo-lang',v)}catch(e){}}const app={lang:getSavedLanguage(),clips:[],filtered:[],selected:null,expanded:new Set(),profiles:{},failedPortraits:new Set(),pose:null,poses:{},poseUrl:null,video:null,toastTimer:null};const tr=k=>I18N[app.lang][k]||I18N.en[k]||k;const text=(el,k)=>{if(el)el.textContent=tr(k)};const escapeHtml=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const local=()=>location.protocol==='file:';let DATA_VERSION='';const url=p=>{if(!p)return '';const value=new URL(p,DATA_URL);if(DATA_VERSION)value.searchParams.set('v',DATA_VERSION);return value.href};
const DATA_URL=new URL('data/clips.json',document.currentScript.src).href;
async function init(){try{const [res,profiles]=await Promise.all([fetch(DATA_URL,{cache:'no-store'}),fetch(new URL('player_profiles.json',DATA_URL),{cache:'no-store'}).then(r=>r.ok?r.json():{}).catch(()=>({}))]);app.profiles=profiles.players||{};if(!res.ok)throw Error('index');const data=await res.json();DATA_VERSION=data.packageVersion||'';app.clips=Array.isArray(data.clips)?data.clips.map(c=>({...c,video:c.mediaPolicy==='pose-only'?c.poseVideo:'',thumbnail:c.mediaPolicy==='pose-only'?c.poseThumbnail:'',overlay:'',overlayThumbnail:''})):[];if(!app.clips.length)throw Error('empty');populateFilters();filter();const counts=new Map();for(const clip of app.filtered)counts.set(groupKey(clip),(counts.get(groupKey(clip))||0)+1);const first=[...app.filtered].filter(clip=>clip.playerKnown&&clip.pose3d).sort((a,b)=>(counts.get(groupKey(b))||0)-(counts.get(groupKey(a))||0)||displayPlayer(a).localeCompare(displayPlayer(b)))[0]||app.filtered.find(clip=>clip.pose3d)||app.filtered[0];if(first)selectClip(first.id,{scroll:false});}catch(e){$('#clip-list').innerHTML=`<div class="empty-state"><h3>${escapeHtml(tr('badData'))}</h3><p>${local()?'Opening index.html as file may block data loading. Use python3 -m http.server 8000.':''}</p></div>`;$('#result-count').textContent='0';}}
function bind(){$('#lang-toggle').addEventListener('click',()=>{app.lang=app.lang==='en'?'ko':'en';saveLanguage(app.lang);applyLanguage();populateFilters();renderList();if(app.selected)renderDetail(app.selected)});$('#search').addEventListener('input',filter);['team-filter','hand-filter','pitch-filter'].forEach(id=>$('#'+id).addEventListener('change',filter));$('#only-3d').addEventListener('change',()=>{filter();if($('#only-3d').checked&&app.filtered.length&&!app.selected?.pose3d)selectClip(app.filtered[0].id,{scroll:false})});$('#clear-filters').addEventListener('click',clearFilters);$('#empty-clear').addEventListener('click',clearFilters);$('#menu-btn').addEventListener('click',()=>$('.main-nav').classList.toggle('open'));document.querySelectorAll('.nav-link').forEach(a=>a.addEventListener('click',()=>$('.main-nav').classList.remove('open')));document.addEventListener('keydown',e=>{if(e.key==='/'&&!['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();$('#search').focus()}if(app.video&&e.key==='ArrowLeft'){e.preventDefault();stepFrame(-1)}if(app.video&&e.key==='ArrowRight'){e.preventDefault();stepFrame(1)}if(app.video&&e.code==='Space'&&!['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)){e.preventDefault();togglePlayback()}})}
function applyLanguage(){document.documentElement.lang=app.lang;document.title=app.lang==='ko'?'KBO Pitch Lab — 투구 동작을 살펴보다':'KBO Pitch Lab — Pitch motion, made observable';document.querySelectorAll('[data-i18n]').forEach(el=>{const val=tr(el.dataset.i18n);if(val.includes('<'))el.innerHTML=val;else el.textContent=val});document.querySelectorAll('[data-i18n-placeholder]').forEach(el=>el.placeholder=tr(el.dataset.i18nPlaceholder));$('#lang-toggle').innerHTML=app.lang==='en'?'한국어 <span>↗</span>':'ENGLISH <span>↗</span>';}
function displayTeam(c){return app.lang==='ko'?(c.teamKo||translateTeam(c.team)):c.team||tr('unknownTeam')}function displayPlayer(c){if(!c.playerKnown&&(!c.player||/^unknown pitcher$/i.test(c.player)))return tr('unknownPitcher');return app.lang==='ko'?(c.playerKo||c.player||tr('unknownPitcher')):(c.player||tr('unknownPitcher'))}function displayPitch(c){return app.lang==='ko'?(c.pitchKo||translatePitch(c.pitch)):c.pitch||tr('unknown')};function translateTeam(s){return ({'SSG vs LG':'SSG 대 LG','LG vs 한화':'LG 대 한화','한화 vs LG':'한화 대 LG','SSG vs 삼성':'SSG 대 삼성','삼성 vs SSG':'삼성 대 SSG','KIA vs KT':'KIA 대 KT','KT vs NC':'KT 대 NC','Unknown':'팀 미상'})[s]||s||tr('unknownTeam')}function translatePitch(s){return ({Fastball:'직구',Slider:'슬라이더',Curveball:'커브',Changeup:'체인지업','Two-seam':'투심'})[s]||s||tr('unknown')}
function populateFilters(){const teams=[...new Set(app.clips.map(c=>c.team).filter(Boolean))].sort();const pitches=[...new Set(app.clips.map(c=>c.pitch).filter(Boolean))].sort();$('#team-filter').innerHTML=`<option value="">${escapeHtml(tr('allTeams'))}</option>`+teams.map(x=>`<option value="${escapeHtml(x)}">${escapeHtml(app.lang==='ko'?translateTeam(x):x)}</option>`).join('');$('#pitch-filter').innerHTML=`<option value="">${escapeHtml(tr('allPitches'))}</option>`+pitches.map(x=>`<option value="${escapeHtml(x)}">${escapeHtml(app.lang==='ko'?translatePitch(x):x)}</option>`).join('');}
function filter(){const q=$('#search').value.trim().toLowerCase(),team=$('#team-filter').value,hand=$('#hand-filter').value,pitch=$('#pitch-filter').value;app.filtered=app.clips.filter(c=>(!q||[c.player,c.playerKo,c.team,c.teamKo,c.id,c.pitch,c.pitchKo,c.matchup].join(' ').toLowerCase().includes(q))&&(!team||c.team===team)&&(!hand||c.hand===hand)&&(!pitch||c.pitch===pitch)&&(!$('#only-3d').checked||!!c.pose3d));$('#result-count').textContent=app.filtered.length;$('#player-count').textContent=new Set(app.filtered.map(groupKey)).size;renderList();}
function clearFilters(){$('#search').value='';$('#team-filter').value='';$('#hand-filter').value='';$('#pitch-filter').value='';$('#only-3d').checked=false;filter()}
function groupKey(c){return c.playerKnown && c.player ? (c.playerKey || `name:${c.player.trim()}`) : `unknown:${c.id}`}
function clipCountWord(n){return app.lang==='ko'?tr('clipCount'):tr(n===1?'clipSingular':'clipCount')}
function formatFps(c){const fps=Number(c.fps);if(!Number.isFinite(fps))return '—';return Math.abs(fps-Math.round(fps))<.03?String(Math.round(fps)):fps.toFixed(2)}
function playerProfile(c){
  const profile=app.profiles[c.playerCode];
  return c.playerKnown&&profile&&profile.name.replace(/\s+/g,'')===String(c.player||'').replace(/\s+/g,'')?profile:null;
}
function playerPortrait(c){
  const profile=playerProfile(c);
  const fallback='<svg class="portrait-placeholder" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="17" r="8"/><path d="M8 44v-4a16 16 0 0 1 32 0v4Z"/></svg>';
  return fallback+(profile&&!app.failedPortraits.has(c.playerCode)?`<img class="player-portrait" data-player-code="${escapeHtml(c.playerCode)}" src="${escapeHtml(profile.imagePath?url(profile.imagePath):profile.imageUrl)}" data-fallback-src="${escapeHtml(profile.imagePath?profile.imageUrl:'')}" alt="${escapeHtml(displayPlayer(c)+' '+tr('profilePhoto'))}" title="${escapeHtml(`KBO · ${profile.imageYear}`)}" loading="lazy" decoding="async" referrerpolicy="no-referrer">`:'');
}
function nameEvidence(c){return tr(({naver_auto_accepted:'autoName',scoreboard_repeated_exact:'scoreboardName',broadcast_text_review:'reviewedName'})[c.playerNameSource]||'unidentified')}
function bindPortraits(root){
  root.querySelectorAll('.player-portrait').forEach(img=>{
    const fallback=()=>{
      if(img.dataset.fallbackSrc){const next=img.dataset.fallbackSrc;delete img.dataset.fallbackSrc;img.src=next;return}
      app.failedPortraits.add(img.dataset.playerCode);img.remove();
    };
    img.addEventListener('error',fallback);
    if(img.complete&&!img.naturalWidth)fallback();
  });
}
function renderList(){
  const list=$('#clip-list'),empty=$('#empty-state'),scroll=$('.clip-list-wrap');
  const previousScroll=scroll.scrollTop;
  if(!app.filtered.length){list.innerHTML='';empty.classList.remove('hidden');showEmptyDetail();return}
  empty.classList.add('hidden');
  const groups=new Map();
  for(const clip of app.filtered){const key=groupKey(clip);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(clip)}
  const sortedGroups=[...groups].sort((a,b)=>Number(b[1][0].playerKnown)-Number(a[1][0].playerKnown)||b[1].length-a[1].length||displayPlayer(a[1][0]).localeCompare(displayPlayer(b[1][0])));
  list.innerHTML=sortedGroups.map(([key,clips])=>{
    const first=clips[0],expanded=app.expanded.has(key),name=displayPlayer(first);
    const dates=[...new Set(clips.map(c=>c.date).filter(Boolean))].sort();
    const dateLabel=dates.length>1?`${dates[0]} — ${dates[dates.length-1]}`:(dates[0]||displayTeam(first));
    const children=clips.map((c,index)=>`<article class="clip-card ${app.selected?.id===c.id?'selected':''}" data-id="${escapeHtml(c.id)}" tabindex="0" role="button" aria-label="${escapeHtml(name)} · ${escapeHtml(c.date||'')} · ${escapeHtml(displayPitch(c))}"><div class="clip-thumb">${c.thumbnail?`<img src="${escapeHtml(url(c.thumbnail))}" alt="" loading="lazy">`:''}<span class="thumb-frame">${escapeHtml(String(c.frames||0).padStart(4,'0'))} FR</span><span class="thumb-play">▶</span></div><div class="clip-info"><div class="clip-topline"><span class="clip-player">${escapeHtml(displayPitch(c)==tr('unknown')?`${tr('delivery')} ${index+1}`:displayPitch(c))}</span><span class="clip-id">${escapeHtml(c.id.split('_').pop())}</span></div><div class="clip-subline">${escapeHtml(c.date||tr('unknown'))} · ${escapeHtml(displayTeam(c))}</div><div class="clip-tags"><span class="clip-tag green">${escapeHtml(c.inning?`${c.inning} INN`:`${index+1} / ${clips.length}`)}</span>${c.pitchNumber?`<span class="clip-tag">P${escapeHtml(c.pitchNumber)}</span>`:''}<span class="clip-tag">${escapeHtml(formatFps(c))} FPS</span><span class="clip-tag ${c.pose3d?'green':''}">${c.pose3d?'2D + 3D':'2D'}</span></div></div></article>`).join('');
    return `<section class="player-group ${expanded?'is-open':''}"><button type="button" class="player-group-toggle" data-player-key="${escapeHtml(key)}" aria-expanded="${expanded}" aria-label="${escapeHtml(name)} · ${clips.length} ${escapeHtml(clipCountWord(clips.length))}"><span class="group-cover">${playerPortrait(first)}</span><span class="group-copy"><strong>${escapeHtml(name)}</strong><small>${escapeHtml(nameEvidence(first))} · ${escapeHtml(dateLabel)}</small></span><span class="group-count">${clips.length} <small>${escapeHtml(clipCountWord(clips.length))}</small></span><span class="group-chevron" aria-hidden="true">⌄</span></button><div class="player-group-clips">${children}</div></section>`
  }).join('');
  scroll.scrollTop=previousScroll;
  bindPortraits(list);
  list.querySelectorAll('.player-group-toggle').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.playerKey;app.expanded.has(key)?app.expanded.delete(key):app.expanded.add(key);renderList()}));
  list.querySelectorAll('.clip-card').forEach(card=>{const open=()=>selectClip(card.dataset.id);card.addEventListener('click',open);card.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open()}})});
  if(app.selected&&!app.filtered.some(c=>c.id===app.selected.id)){app.selected=null;showEmptyDetail()}
}
function sourceReference(c){
  if(!/^[A-Za-z0-9_-]{11}$/.test(c.sourceVideoId||'')||typeof c.sourceStart!=='number'||!Number.isFinite(c.sourceStart)||c.sourceStart<0)return null;
  const start=Math.floor(c.sourceStart),end=typeof c.sourceEnd==='number'&&Number.isFinite(c.sourceEnd)&&c.sourceEnd>c.sourceStart?Math.ceil(c.sourceEnd):null;
  return {watch:`https://www.youtube.com/watch?v=${c.sourceVideoId}&t=${start}s`,embed:`https://www.youtube-nocookie.com/embed/${c.sourceVideoId}?start=${start}${end===null?'':`&end=${end}`}&playsinline=1&rel=0`};
}
function sourceTime(seconds){const n=Math.floor(seconds);return `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`}
function renderSource(c,panel){
  const source=sourceReference(c);if(!source)return;
  const section=document.createElement('section');section.className='youtube-source';
  const interval=`${sourceTime(c.sourceStart)}${c.sourceEnd>c.sourceStart?' – '+sourceTime(c.sourceEnd):''}`;
  section.innerHTML=`<div class="youtube-source-head"><div><h4>${escapeHtml(tr('youtubeSource'))}</h4><span class="source-time">${escapeHtml(interval)}</span></div><a class="download-btn" href="${escapeHtml(source.watch)}" target="_blank" rel="noopener noreferrer">${escapeHtml(tr('youtubeOpen'))}</a></div><button type="button" class="youtube-toggle" aria-expanded="false" aria-controls="youtube-embed">${escapeHtml(tr('youtubeShow'))}</button><div id="youtube-embed" class="youtube-embed" hidden></div><p class="youtube-source-note">${escapeHtml(tr('youtubeNote'))}</p>`;
  panel.querySelector('.player-bar').after(section);
  const button=section.querySelector('.youtube-toggle'),container=section.querySelector('.youtube-embed');
  button.addEventListener('click',()=>{
    const open=button.getAttribute('aria-expanded')!=='true';
    button.setAttribute('aria-expanded',String(open));button.textContent=tr(open?'youtubeHide':'youtubeShow');container.hidden=!open;
    container.replaceChildren();
    if(open){
      app.video?.pause();
      const iframe=document.createElement('iframe');iframe.src=source.embed;
      iframe.title=`${tr('youtubeSource')} · ${interval} · ${c.matchup||c.id}`;
      iframe.allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
      iframe.allowFullscreen=true;iframe.referrerPolicy='strict-origin-when-cross-origin';
      container.append(iframe);
    }
  });
}
function selectClip(id,{scroll=true}={}){const c=app.clips.find(x=>x.id===id);if(!c)return;app.video?.pause();app.selected=c;app.expanded.add(groupKey(c));app.pose=null;app.poses={};app.poseUrl=null;renderList();renderDetail(c);if(scroll&&innerWidth<760)$('#detail-panel').scrollIntoView({behavior:'smooth',block:'start'})}
function renderMotionPanels(c){
  return `<div class="video-pair">${[2,3].map(dimension=>`<section class="video-cell" data-dimension="${dimension}"><div class="video-label"><span>${escapeHtml(tr(dimension===2?'final2d':'final3d'))}</span><i>${dimension===2?'HALPE-26':'WHAM'}</i></div><div class="player-wrap"><canvas class="motion-canvas" data-dimension="${dimension}" width="720" height="720" role="img" aria-label="${escapeHtml(tr(dimension===2?'final2d':'final3d'))}" hidden></canvas><div class="motion-placeholder" data-dimension="${dimension}" role="status">${escapeHtml(tr('loading'))}</div></div><div class="motion-downloads"><button class="download-btn" type="button" data-dimension="${dimension}" data-format="json" disabled aria-label="${dimension}D JSON">JSON ↓</button><button class="download-btn" type="button" data-dimension="${dimension}" data-format="csv" disabled aria-label="${dimension}D CSV">CSV${dimension===3?' · 24':''} ↓</button></div></section>`).join('')}</div>`;
}
async function renderDetail(c){window.MotionViewer?.dispose();const panel=$('#detail-panel');app.video?.pause();app.video=null;const val=(k,v)=>`<div class="meta-item" data-meta="${k}"><span class="meta-key">${escapeHtml(tr(k))}</span><span class="meta-value" title="${escapeHtml(v||'')}">${escapeHtml(v||tr('unknown'))}</span></div>`;const tags=[c.pitch&&`<span class="detail-tag accent">${escapeHtml(displayPitch(c))}</span>`,c.hand&&`<span class="detail-tag">${escapeHtml(c.hand==='R'?tr('right'):tr('left'))}</span>`,c.example&&`<span class="detail-tag">${escapeHtml(tr('example'))}</span>`].filter(Boolean).join('');panel.innerHTML=`<div class="detail-content">${renderMotionPanels(c)}<div class="player-bar"><button class="control-btn" id="back-frame" title="Previous frame">‹</button><button class="control-btn" id="play-btn" title="Play or pause">▶</button><button class="control-btn" id="next-frame" title="Next frame">›</button><input class="frame-range" id="frame-range" type="range" min="0" max="${Math.max(0,(c.frames||1)-1)}" value="0" aria-label="Frame position"><span class="time-code" id="time-code">00:00.00</span><select class="speed-select" id="speed-select" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></div><div class="detail-body"><div class="detail-title-row"><div><h3 class="detail-player">${escapeHtml(displayPlayer(c))}</h3><div class="detail-matchup">${escapeHtml(c.matchup||displayTeam(c))}${c.date?' · '+escapeHtml(c.date):''}${c.inning?' · '+escapeHtml(c.inning)+' INN':''}</div></div><span class="detail-id">${escapeHtml(c.id)}</span></div><div class="detail-tags">${tags}</div><div class="metadata-grid">${val('nameEvidence',nameEvidence(c))}${val('teamLabel',displayTeam(c))}${val('handLabel',c.hand?(c.hand==='R'?tr('right'):tr('left')):tr('notRecorded'))}${val('pitchLabel',displayPitch(c))}${val('dateLabel',c.date)}${val('inningLabel',c.inning)}${val('pitchNumberLabel',c.pitchNumber)}${val('clipDuration',c.duration?`${c.duration.toFixed?c.duration.toFixed(2):c.duration} s`:'' )}${val('fps',c.fps?`${c.fps} fps`:'')}${val('frameCount',c.frames?String(c.frames):'')}${val('source',c.source)}${val('model',c.model3d?`${c.model} / ${c.model3d}`:c.model)}${val('coords',c.sceneFrame?'2D: crop pixels · 3D: clip-fixed inferred floor':c.pose3d?'2D: crop pixels · 3D: hip-relative model meters':c.coordinate)}<div class="meta-item"><span class="meta-key">${escapeHtml(tr('poseStatus'))}</span><span class="meta-value pose-status"><i></i>${escapeHtml(c.poseStatus||tr('unknown'))}</span></div></div><div class="download-row"><span class="download-note">${escapeHtml(c.example?tr('example'):`${tr('estimateOnly')} · ${tr('confidence')}: ${c.confidence||tr('noConfidence')}`)}</span></div></div></div>`;
  renderSource(c,panel);
  const profile=playerProfile(c);
  if(profile){
    const source=document.createElement('a');source.className='profile-source';source.href=profile.sourceUrl;
    source.target='_blank';source.rel='noopener noreferrer';source.textContent=tr('photoSource');
    panel.querySelector('.detail-tags').after(source);
  }
  const related=app.clips.filter(other=>other.id!==c.id && groupKey(other)===groupKey(c));
  if(related.length){
    const box=document.createElement('section');box.className='related-clips';
    box.innerHTML=`<h4>${escapeHtml(tr('moreFromPitcher'))} <span>${related.length}</span></h4><div class="related-strip">${related.map(other=>`<button class="related-clip" type="button" data-related-id="${escapeHtml(other.id)}">${other.thumbnail?`<img src="${escapeHtml(url(other.thumbnail))}" alt="" loading="lazy">`:''}<span><strong>${escapeHtml(displayPitch(other))}</strong><small>${escapeHtml(other.date||tr('unknown'))}</small></span></button>`).join('')}</div>`;
    panel.querySelector('.detail-tags').after(box);
    box.querySelectorAll('.related-clip').forEach(button=>button.addEventListener('click',()=>selectClip(button.dataset.relatedId)));
  }
  const video=window.MotionViewer.createClock(panel.querySelector('.player-wrap'),c.frames,c.fps);app.video=video;app.pose=null;app.poses={};
  video.addEventListener('timeupdate',updateFrameUi);
  video.addEventListener('play',()=>{$('#play-btn').textContent='Ⅱ'});
  const paused=()=>{$('#play-btn').textContent='▶'};
  video.addEventListener('pause',paused);video.addEventListener('ended',paused);
  $('#play-btn').addEventListener('click',togglePlayback);
  $('#back-frame').addEventListener('click',()=>stepFrame(-1));
  $('#next-frame').addEventListener('click',()=>stepFrame(1));
  $('#speed-select').addEventListener('change',e=>{video.playbackRate=Number(e.target.value)});
  $('#frame-range').addEventListener('input',e=>seekFrame(Number(e.target.value)));
  panel.querySelectorAll('.motion-downloads button').forEach(button=>button.addEventListener('click',()=>downloadPose(button.dataset.format,Number(button.dataset.dimension))));
  const posePath=c.poseRefined||c.pose;
  if(posePath){
    try{
      const response=await fetch(url(posePath));if(!response.ok)throw Error(posePath);
      const pose=await response.json();if(app.selected?.id!==c.id||app.video!==video)return;
      app.pose=pose;app.poses={2:pose};app.poseUrl=posePath;
      await window.MotionViewer.mount({clip:c,panel,lang:app.lang,url,pose2d:pose,clock:video,onPoses:poses=>{app.poses=poses;app.pose=poses[2]}});
    }catch(e){
      if(app.selected?.id!==c.id||app.video!==video)return;
      showToast(tr('missingPose'));app.pose=null;app.poses={};
      panel.querySelectorAll('.motion-placeholder').forEach(el=>{el.textContent=tr('missingPose')});
    }
  }else{showToast(tr('missingPose'))}
}

function showEmptyDetail(){window.MotionViewer?.dispose();app.video?.pause();app.video=null;const p=$('#detail-panel');p.innerHTML=`<div class="detail-empty"><div class="empty-visual"><div class="empty-video-icon">▶</div><div class="empty-cross cross-a"></div><div class="empty-cross cross-b"></div><span class="empty-coordinate">FRAME 0000</span></div><span class="detail-kicker">${escapeHtml(tr('yourWorkspace'))}</span><h3>${tr('selectClip')}</h3><p>${escapeHtml(tr('detailHint'))}</p></div>`}
function togglePlayback(){if(!app.video)return;if(app.video.paused)app.video.play().catch(()=>showToast(tr('missingVideo')));else app.video.pause()}
function currentFps(){return Number(app.selected?.fps)||30}function localFrame(){return Math.round((app.video?.currentTime||0)*currentFps())}function stepFrame(n){if(!app.video)return;const v=app.video,fps=currentFps();if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=Math.max(0,Math.min(v.duration,Math.min(((app.selected?.frames||1)-1)/fps,v.currentTime+n/fps)));else seekFrame(localFrame()+n)}function seekFrame(n){const v=app.video;if(!v)return;const fps=currentFps();if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=Math.max(0,Math.min(v.duration,Math.min(((app.selected?.frames||1)-1)/fps,n/fps)));$('#frame-range').value=String(n);updateFrameUi();drawPose()}function updateFrameUi(){if(!app.video)return;const n=Math.min(localFrame(),Math.max(0,(app.selected?.frames||1)-1));const range=$('#frame-range');if(range&&document.activeElement!==range)range.value=String(n);const t=app.video.currentTime||0;if($('#time-code'))$('#time-code').textContent=`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')} · F${String(n).padStart(4,'0')}`}
const BONES=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];function drawPose(){window.MotionViewer?.draw()}
async function downloadPose(kind,dimension=2){
  const pose=app.poses[dimension];
  if(!pose){showToast(tr('filesMissing'));return}
  let blob,name;
  if(kind==='json'){
    blob=new Blob([JSON.stringify(pose,null,2)],{type:'application/json'});
    name=`${app.selected.id}-${pose.dimension===3?'3d':'2d'}-final-pose.json`;
  }else{
    const dim=pose.dimension===3?3:2;
    const native=dim===3&&Array.isArray(pose.anatomy_joints);
    const names=native?pose.anatomy_joints:pose.joints;
    const lines=[dim===3?'clip_id,frame,source_frame,time_seconds,source_time,joint,x_model_m,y_model_m,z_model_m,visibility,interpolated,outlier,coordinate_frame':'clip_id,frame,source_frame,time_seconds,source_time,joint,x_pixels,y_pixels,confidence,interpolated,outlier,coordinate_frame'];
    for(const f of pose.frames||[])(native?f.anatomy||[]:f.keypoints||[]).forEach((p,i)=>{
      if(p&&p[0]!=null)lines.push([csv(app.selected.id),f.frame,f.source_frame??'',f.time,f.source_time??'',names?.[i]||`joint_${i}`,...p.slice(0,dim),native?'':p[dim]??'',native?(f.interpolated||[]).some(Boolean):f.interpolated?.[i]??false,native?'':f.outlier?.[i]??false,csv(pose.scene?.frame||(dim===3?'hip-relative':'crop-pixels'))].join(','))
    });
    blob=new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'});
    name=`${app.selected.id}-${dim}d-final-pose.csv`;
  }
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)
}function csv(s){return `"${String(s).replaceAll('"','""')}"`}
function showToast(message){const el=$('#toast');el.textContent=message;el.classList.add('show');clearTimeout(app.toastTimer);app.toastTimer=setTimeout(()=>el.classList.remove('show'),2400)}
Object.assign(I18N.en,{
  heroDesc:'Watch final 2D and estimated 3D pitching motion side by side. Play, inspect and download both results.',
  collectionDesc:'Search the archive. Compare 2D motion and rotate the estimated 3D skeleton.',
  final2d:'Final 2D',final3d:'Final 3D',detailHint:'Choose a clip to watch final 2D and 3D motion together.',
  guideDesc:'Final 2D and 3D share playback, seeking and frame stepping. Rotate the 3D view and download either result. Timestamped YouTube links open the source footage.',
  guideDisclaimer:'3D uses an inferred floor and a fixed frame for each clip. Scale, camera and ground are estimated; stride accuracy and forces have not been validated.',
  formatDesc:'Final 2D exports Halpe-26. The 3D CSV exports 24 fixed-length SMPL joint centers; JSON also includes supplementary COCO-17 landmarks and source timing.',
  contextDesc:'Final 2D uses RTMPose Halpe-26. Available 3D previews fit a constant WHAM SMPL body with fixed stance contacts, temporal constraints and one floor frame. Other 3D results remain withheld for review.'
});
Object.assign(I18N.ko,{
  heroDesc:'최종 2D와 추정 3D 투구 동작을 나란히 살펴보세요. 함께 재생하고 각 결과의 데이터를 내려받을 수 있습니다.',
  collectionDesc:'자료를 검색하고 2D 동작을 비교하거나 추정 3D 관절을 회전해 보세요.',
  final2d:'최종 2D',final3d:'최종 3D',detailHint:'클립을 선택하면 최종 2D와 3D 동작을 동시에 볼 수 있습니다.',
  guideDesc:'최종 2D·3D를 함께 재생·탐색하고 프레임을 이동하세요. 3D는 회전할 수 있고 각 결과를 따로 내려받을 수 있습니다. YouTube 링크에서 원본 장면도 확인할 수 있습니다.',
  guideDisclaimer:'3D는 클립 전체에 고정된 추정 바닥 좌표를 사용합니다. 크기·카메라·바닥은 추정값이며 실제 보폭 정확도와 힘은 검증되지 않았습니다.',
  formatDesc:'2D는 Halpe-26입니다. 3D CSV는 뼈 길이가 고정된 SMPL 24관절 중심 좌표이며, JSON에는 보조 COCO-17과 원본 시각도 담깁니다.',
  contextDesc:'최종 2D는 RTMPose Halpe-26을 사용합니다. 제공되는 3D 미리보기는 WHAM 체형·관절 길이를 고정하고 접지 위치·시간 제약과 동일한 바닥 좌표계를 적용합니다. 신뢰도가 부족한 3D는 보류합니다.'
});
bind();applyLanguage();init();
