'use strict';
const $ = id => document.getElementById(id);
const names = {current_final:'기존 최종 fitting',H_local_temporal:'Phase-aware soft contact + 부위별 시간 제약',WHAM_raw_articulation:'WHAM raw — 상대 자세',MotionBERT_articulation:'MotionBERT — 상대 자세'};
let manifest = [], data = null, playing = false, startWall = 0, startTime = 0, loadVersion = 0;
const canvases = [];
for (const name of ['before','after']) for (let i=0;i<3;i++) {
  const canvas=document.createElement('canvas');canvas.width=360;canvas.height=360;
  canvas.setAttribute('aria-label',`${name==='before'?'기존':'비교 후보'} ${['정면','측면','상단'][i]}`);
  $(name).append(canvas);canvases.push(canvas);
}
async function json(url) { const r=await fetch(url);if(!r.ok)throw Error(`${url}: ${r.status}`);return r.json(); }
async function load() {
  const version=++loadVersion;playing=false;$('play').textContent='재생';$('loading').textContent='클립을 불러오는 중…';
  const selected=$('model').value;
  try {
    const next=await json(`viewer/${encodeURIComponent($('clip').value)}.json`);
    if(version!==loadVersion)return;
    data=next;$('frame').max=data.times.length-1;$('frame').value=0;$('model').replaceChildren();
    for(const name of Object.keys(data.models)){const option=document.createElement('option');option.value=name;option.textContent=names[name];$('model').append(option);}
    $('model').value=data.models[selected]?selected:'H_local_temporal';
    const source=new URL(data.sourceUrl);if(!['www.youtube.com','youtube.com','youtu.be'].includes(source.hostname))throw Error('Invalid source host');
    $('source').href=source.href;$('download').href=`viewer/${encodeURIComponent(data.id)}.json`;
    const status=data.status;
    $('status').textContent=[`클립: ${data.id} · ${data.player}`,`기존 판정: ${status.old_status}`,`기존 검토 사유: ${status.old_reasons.join(', ')||'없음 (3D 정확도 검증을 의미하지 않음)'}`,
      `비교 규칙의 선택: ${status.new_primary==='H_local_temporal'?'개선 후보 (새 검증 통과 아님)':'기존 결과 유지'}`,
      `개선 후보의 회귀/보류 사유: ${status.selection_guard_reasons.join(', ')||'비교 규칙 충족'}`,
      `상대 자세: ${status.root_relative_articulation}`,`전역 이동 / 깊이 / 접지: ${status.global_translation} / ${status.depth} / ${status.ground_contact}`,
      `보폭·절대 거리: ${status.stride_absolute_distance}`,`사람 검수 완료: 아님 · 새 3D 검증: ${status.new_validation?'있음':'없음'}`].join('\n');
    $('loading').textContent='';draw();
  } catch(error) { if(version===loadVersion){$('loading').textContent=`불러오지 못했습니다: ${error.message}`;$('loading').className='error';} }
}
function valid(p) { return p && p.every(Number.isFinite); }
function draw3(canvas, points, edges, axes, label) {
  const g=canvas.getContext('2d'),bounds=data.bounds;
  const span=Math.max(...bounds.max.map((v,i)=>v-bounds.min[i]))*1.1;
  const screen=p=>[180+(p[axes[0]]-(bounds.max[axes[0]]+bounds.min[axes[0]])/2)*310/span,185-(p[axes[1]]-(bounds.max[axes[1]]+bounds.min[axes[1]])/2)*310/span];
  g.fillStyle='#fff';g.fillRect(0,0,360,360);g.fillStyle='#253b30';g.font='14px system-ui';g.fillText(label,12,21);
  if(axes[1]===1){const y=185+(bounds.max[1]+bounds.min[1])/2*310/span;g.strokeStyle='#ccd7cf';g.beginPath();g.moveTo(15,y);g.lineTo(345,y);g.stroke();g.font='10px system-ui';g.fillText('추정 평면 y=0',12,y-5);}
  for(const [a,b] of edges){if(!valid(points[a])||!valid(points[b]))continue;g.strokeStyle=a%2?'#3478b2':'#bc4a3e';g.lineWidth=3;g.beginPath();g.moveTo(...screen(points[a]));g.lineTo(...screen(points[b]));g.stroke();}
  g.fillStyle='#5b6d60';g.font='10px system-ui';g.fillText('동일한 고정 축 · 화면 범위',12,350);
}
function draw() {
  if(!data)return;
  const i=+$('frame').value,name=$('model').value,model=data.models[name];
  $('newlabel').textContent=`${names[name]} — 정면 / 측면 / 상단`;$('time').textContent=`${i} / ${data.times.length-1} · ${data.times[i].toFixed(3)}초`;
  const g=$('overlay').getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,960,540);
  g.strokeStyle='#edf1ec';for(let x=0;x<960;x+=80){g.beginPath();g.moveTo(x,0);g.lineTo(x,540);g.stroke();}
  const sets=[[data.observations[i],'#273d32'],[data.models.current_final.projection[i],'#1676cf'],[model.projection?model.projection[i]:[],'#bd5424']];
  for(const [points,color] of sets){g.fillStyle=color;for(const p of points){if(!p||!p.slice(0,2).every(Number.isFinite))continue;g.beginPath();g.arc(p[0]*data.imageScale,p[1]*data.imageScale,3,0,Math.PI*2);g.fill();}}
  const axes=[[0,1],[2,1],[0,2]];
  for(let a=0;a<3;a++){draw3(canvases[a],data.models.current_final.points[i],data.edges,axes[a],['정면 x/y','측면 z/y','상단 x/z'][a]);draw3(canvases[a+3],model.points[i],model.edges||data.edges,axes[a],['정면 x/y','측면 z/y','상단 x/z'][a]);}
  for(const canvas of [$('overlay'),...canvases])canvas.dataset.frame=String(i);
}
$('clip').onchange=load;$('model').onchange=draw;
$('frame').oninput=()=>{startWall=performance.now();startTime=data?.times[+$('frame').value]||0;draw();};
$('play').onclick=()=>{if(!data)return;if(+$('frame').value===data.times.length-1)$('frame').value=0;playing=!playing;$('play').textContent=playing?'정지':'재생';startWall=performance.now();startTime=data.times[+$('frame').value];draw();};
function animate(){if(data&&playing){const time=startTime+(performance.now()-startWall)/1000;let frame=0;while(frame<data.times.length-1&&data.times[frame+1]<=time)frame++;$('frame').value=frame;draw();if(frame===data.times.length-1){playing=false;$('play').textContent='재생';}}requestAnimationFrame(animate);}
animate();
json('viewer/manifest.json').then(rows=>{manifest=rows;for(const row of rows){const o=document.createElement('option');o.value=row.id;o.textContent=`${row.player_name||'선수 미확인'} · ${row.id} · ${row.split==='test'?'최종 비교':'튜닝'}`;$('clip').append(o);}return load();}).catch(error=>{$('loading').textContent=error.message;});
