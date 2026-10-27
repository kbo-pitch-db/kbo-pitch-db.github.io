import {createBoneScene} from '../../bone_scene.js';
const $=id=>document.getElementById(id),edges=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[15,20],[15,22],[15,24],[16,21],[16,23],[16,25]];
const state={ready:false,playing:false,index:0,data:null,two:null,scenes:[],frameTimes:[],yaw:.65,pitch:.12,position:0,lastTick:0,ticket:0};window.armComparison=state;
const positions={oblique:[.65,.12],front:[0,-.25],side:[Math.PI/2,-.25],top:[0,1]};
function stop(){state.playing=false;$('play').textContent='재생';}
function draw2d(frame){
 const canvas=$('two'),ctx=canvas.getContext('2d'),[cx,cy,extent]=state.twoBounds,project=p=>[(p[0]-cx)*640/extent+360,(p[1]-cy)*640/extent+360];
 ctx.fillStyle='#1b3028';ctx.fillRect(0,0,720,720);ctx.lineWidth=3;ctx.strokeStyle='#a9db87';
 for(const [a,b] of edges){const p=frame.keypoints[a],q=frame.keypoints[b];if(!p||!q||p[2]<.2||q[2]<.2)continue;ctx.beginPath();ctx.moveTo(...project(p));ctx.lineTo(...project(q));ctx.stroke()}
 for(const p of frame.keypoints){if(!p||p[2]<.2)continue;ctx.beginPath();ctx.arc(...project(p),4,0,Math.PI*2);ctx.fillStyle='#e1f2bb';ctx.fill()}
 canvas.dataset.frame=String(state.index);
}
function draw(){
 if(!state.ready)return;
 const i=state.index;draw2d(state.two.frames[i]);
 for(const [j,key] of ['nlf_mb','gvhmr'].entries()){const m=state.data.models[key];state.scenes[j].render(m.frames[i],m.frames[i],0,state.yaw,state.pitch);document.querySelectorAll('article canvas')[j+1].dataset.frame=String(i)}
 $('seek').value=i;$('time').textContent=`${state.frameTimes[i].toFixed(2)} s · ${i+1} / ${state.frameTimes.length}`;
}
function seek(i){if(!state.ready)return;stop();state.index=Math.max(0,Math.min(state.frameTimes.length-1,Number(i)));state.position=state.frameTimes[state.index];draw()}
state.seek=seek;
async function load(cid){
 const ticket=++state.ticket;stop();state.ready=false;$('status').textContent='불러오는 중…';$('error').textContent='';for(const id of ['play','prev','next','seek'])$(id).disabled=true;
 try{
  const rec=state.catalog.find(r=>r.id===cid);if(!rec)throw Error('비교 결과가 없는 클립입니다.');
  const response=await fetch(rec.file);if(!response.ok)throw Error('3D 파일을 불러올 수 없습니다.');const data=await window.ReconstructionRegistry.decode(response,rec.file);const r=await fetch(data.pose2d);if(!r.ok)throw Error('2D 파일을 불러올 수 없습니다.');const two=await r.json();
  if(ticket!==state.ticket)return;
  for(const m of Object.values(data.models)){if(m.frames.length!==two.frames.length||m.frames.some((f,i)=>f.source_frame!==two.frames[i].source_frame||f.source_time!==two.frames[i].source_time))throw Error('모델과 2D의 프레임·시간이 일치하지 않습니다.');}
  for(const s of state.scenes)s.dispose();state.scenes=[];
  for(const [id,key] of [['nlf','nlf_mb'],['gv','gvhmr']]){const old=$(id),canvas=old.cloneNode(false);old.replaceWith(canvas);canvas.addEventListener('wheel',e=>e.stopImmediatePropagation(),{capture:true});state.scenes.push(createBoneScene(canvas,data.models[key]));}
  state.data=data;state.two=two;state.index=0;state.position=0;state.frameTimes=data.times.map(t=>t-data.times[0]);
  const points=two.frames.flatMap(f=>f.keypoints.filter(p=>p&&p[2]>.2&&p.slice(0,2).every(Number.isFinite)));let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of points){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])}state.twoBounds=[(minX+maxX)/2,(minY+maxY)/2,Math.max(maxX-minX,maxY-minY,1)];
  state.ready=true;$('seek').max=two.frames.length-1;$('status').textContent=`${data.player} · ${rec.date}`;$('source').href=data.source_url;$('explorer').href=`../../?preview=rear-pivot&clip=${encodeURIComponent(cid)}#explore`;
  for(const id of ['play','prev','next','seek'])$(id).disabled=false;
  const url=new URL(location.href);url.searchParams.set('clip',cid);history.replaceState(null,'',url);draw();
 }catch(e){if(ticket!==state.ticket)return;$('error').textContent=e.message;$('status').textContent='비교 불가';console.error(e)}
}
state.load=load;
$('clip').onchange=()=>load($('clip').value);$('seek').oninput=()=>seek($('seek').value);$('prev').onclick=()=>seek(state.index-1);$('next').onclick=()=>seek(state.index+1);
$('play').onclick=()=>{if(!state.ready)return;if(state.playing){stop();return}if(state.index===state.frameTimes.length-1)seek(0);state.playing=true;state.lastTick=performance.now();$('play').textContent='정지'};
$('views').onclick=e=>{const button=e.target.closest('[data-view]');if(!button)return;[state.yaw,state.pitch]=positions[button.dataset.view];for(const b of $('views').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===button));draw()};
function tick(now){
 if(state.playing&&state.ready){state.position+=(now-state.lastTick)/1000*Number($('speed').value);const last=state.frameTimes.length-1;while(state.index<last&&state.frameTimes[state.index+1]<=state.position)state.index++;if(state.position>=state.frameTimes[last]){state.index=last;stop()}draw()}
 state.lastTick=now;requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
try{const response=await fetch('catalog.json');if(!response.ok)throw Error('클립 목록을 불러올 수 없습니다.');state.catalog=await response.json();for(const r of state.catalog){const option=document.createElement('option');option.value=r.id;option.textContent=`${r.player} · ${r.date}`;$('clip').append(option)}$('clip').disabled=false;const requested=new URL(location.href).searchParams.get('clip');const cid=state.catalog.some(r=>r.id===requested)?requested:'3fdcd2dbaf58_c0014';$('clip').value=cid;await load(cid)}catch(e){$('error').textContent=e.message;console.error(e)}
