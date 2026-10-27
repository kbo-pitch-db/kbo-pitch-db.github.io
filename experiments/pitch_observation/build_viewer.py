"""Local static source/3D comparisons. Fixed axes, explicit coordinate semantics."""
from common import *
from candidates import CC,MB,norm
from pool import load_pool
import pool
from events import infer


def make_data():
 folder=OUT/'viewer_data';folder.mkdir(exist_ok=True);catalog=[]
 for split in ['tune','test']:
  selected=OUT/f'selection_strong_{split}.json'
  if not selected.exists():continue
  selection={r['id']:r for r in json.loads(selected.read_text())};pool.VERSION='strong'
  for rec in sample(split,split=='tune'):
   cid=rec['id'];records,h,c,s=load_pool(cid,split);by=dict(records);sel=selection[cid];old=by['current_actual'];names=['current_actual',sel['reference']]
   names=list(dict.fromkeys(names+['WHAM_raw_f4','gvhmr_f4','mb_obp_single_smpl_f4','mb_block_single_smpl_f4']))
   models={}
   for name in names:
    z=by[name];xyz=z['keypoints'][:,CC];relative,leg=norm(xyz);glob=xyz.copy()
    if name!='current_actual':glob+=z['translation'][:,None]
    anchor=glob[0,:2].mean(0);glob=(glob-anchor)/leg
    models[name]=dict(relative=relative,global_fixed=glob,projected=z['projected'][:,CC],global_validated=False)
   # Independent native lifting outputs use their own clearly labeled relative mode.
   for name,tag in [('MotionBERT pretrained native','mb_single'),('MotionBERT head adapter native','mb_obp_single'),('MotionBERT last block native','mb_block_single')]:
    p=OUT/'raw'/tag/f'{cid}.npz'
    if p.exists():models[name]=dict(relative=norm(load(p)['keypoints'][:,MB])[0],global_fixed=None,projected=None,global_validated=False)
   nlfpath=OUT/('fits_nlf' if split=='tune' else 'fits_nlf_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'
   if nlfpath.exists():
    z=load(nlfpath);xyz=z['keypoints'][:,CC];relative,leg=norm(xyz);glob=xyz+z['translation'][:,None];glob=(glob-glob[0,:2].mean(0))/leg;models['NLF RGB + fitting (exploratory)']=dict(relative=relative,global_fixed=glob,projected=z['projected'][:,CC],global_validated=False)
   for label,path in [('NLF RGB body + temporal MotionBERT arms',OUT/('fits_nlf_hybrid' if split=='tune' else 'fits_nlf_hybrid_test')/'nlf_mb_hybrid_temporal_f4'/f'{cid}.npz'),('NLF fixed native shape · raw RGB',OUT/'nlf_native_fixed'/f'{cid}.npz'),('NLF native shape + 2D fitting',OUT/('fits_nlf_native_shape' if split=='tune' else 'fits_nlf_native_shape_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'),('NLF native shape + RGB contour fitting',OUT/('fits_nlf_rgb' if split=='tune' else 'fits_nlf_rgb_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz')]:
    if path.exists():
     z=load(path);xyz=z['keypoints'][:,CC];relative,leg=norm(xyz);glob=xyz+z['translation'][:,None];glob=(glob-glob[0,:2].mean(0))/leg;models[label]=dict(relative=relative,global_fixed=glob,projected=z['projected'][:,CC],global_validated=False)
   fullnlf=OUT/'raw/nlf_frames'/cid
   if len(list(fullnlf.glob('*_f4.npz')))==len(h['source_time']):
    nz=[load(fullnlf/f'{i:05d}_f4.npz') for i in range(len(h['source_time']))];nj=[1,2,4,5,7,8,16,17,18,19,20,21];native=np.concatenate([q['joints3d_nonparam'] for q in nz])[:,nj];relative,leg=norm(native);glob=(native-native[0,:2].mean(0))/leg
    models['NLF raw native joint centers (SMPL24 mapping)']=dict(relative=relative,global_fixed=glob,projected=np.concatenate([q['joints2d_nonparam'] for q in nz])[:,nj],global_validated=False)
   finite=[np.max(abs(np.asarray(m['global_fixed'])[np.isfinite(m['global_fixed'])])) for m in models.values() if m['global_fixed'] is not None];extent=max([3.2]+finite)*1.08
   bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];event=infer(h,bg)
   payload=dict(id=cid,player=rec['player_name'],split=split,width=int(s['width']),height=int(s['height']),video=f'original_clips/{cid}.mp4',times=h['source_time']-h['source_time'][0],source_times=h['source_time'],halpe=h['keypoints'],scores=scores(h),models=models,extent=extent,selection=sel,events=event,legacy_status=rec['old_status'],legacy_reasons=rec['old_reasons'],source_note='Original-resolution H264 inspection copy; broadcast encoded timestamps, physical slow-motion timing not calibrated')
   save(folder/f'{cid}.json',payload);catalog.append(dict(id=cid,player=rec['player_name'],split=split,reference=sel['reference'],changed=sel['actual_estimate_changed']))
 save(OUT/'viewer_catalog.json',catalog)

HTML=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>KBO 원본 · 초기 후보 · 3D 비교</title><style>
:root{color-scheme:dark}body{font:15px system-ui;margin:20px;background:#10161e;color:#e8edf3}a{color:#80cbff}h1{font-size:23px}select,button,input{font:inherit;padding:6px;background:#243244;color:white;border:1px solid #536376;border-radius:5px}label{margin-right:14px}.bar>*{min-width:0;max-width:100%}select{max-width:min(85vw,850px)}pre{overflow-wrap:anywhere}.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}.notice{background:#26303d;padding:12px;line-height:1.6;border-left:4px solid #ffcf70}.source{position:relative;max-width:1100px}video{display:block;width:100%}.source canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}.grids{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.plot{width:100%;height:auto;background:#17212d}.rowlabel{margin:18px 0 6px;font-weight:bold}.legend{font-size:13px;color:#b6c5d7}#seek{width:min(65vw,800px)}pre{white-space:pre-wrap;background:#17212d;padding:12px;max-height:380px;overflow:auto}img{max-width:100%}@media(max-width:700px){.grids{grid-template-columns:minmax(0,1fr)}body{margin:10px}}
</style><h1>KBO 원본과 3D 후보 비교 — 로컬 연구 결과</h1><p><a href="REPORT_KO.html">연구 보고서</a> · <a href="ablation_pairs.csv">Ablation CSV</a> · <a href="frozen_protocol.json">동결한 평가 설정</a> · <a href="REPRODUCE.txt">재현·재개 명령</a></p>
<div class="notice">공개 3D GT가 없는 KBO 단안 복원입니다. 기본 화면은 첫 프레임을 원점으로 고정하며 모든 모델에 동일 축과 범위를 사용합니다. 전역 이동·접촉·절대 보폭은 검증되지 않았습니다. 상대 자세 모드는 골반을 매 프레임 제거한 별도 비교이며 전역 이동 개선의 증거로 사용할 수 없습니다. 사람의 수동 검수는 수행되지 않았습니다.</div>
<div class="bar"><label>클립 <select id="clip"></select></label><button id="play">재생 / 정지</button><button id="prev">−1 frame</button><button id="next">+1 frame</button><span id="time"></span></div>
<div class="source"><video id="video" playsinline controls muted></video><canvas id="overlay"></canvas></div>
<div class="bar"><input id="seek" type="range" min="0" value="0"><label><input id="obs" type="checkbox" checked>최종 Halpe 2D 초록</label><label><input id="old2d" type="checkbox">기존 투영 주황</label><label><input id="new2d" type="checkbox" checked>신규 투영 하늘</label></div>
<div class="bar"><label>좌표 <select id="mode"><option value="global_fixed">첫 프레임 고정 원점 · 전역 미검증</option><option value="relative">골반 기준 상대 자세 · 이동 제거</option></select></label><label>비교 후보 <select id="candidate"></select></label></div>
<p id="status" class="notice"></p><div class="rowlabel">기존 결과 <span class="legend">주황</span></div><div id="oldplots" class="grids"></div><div class="rowlabel">선택한 후보 <span class="legend">하늘 · 정면/측면/상단 모두 같은 축</span></div><div id="newplots" class="grids"></div>
<p class="legend">단위: 각 모델의 시퀀스 고정 다리 길이. 실제 m/mm가 아닙니다. 시점 이름은 카메라 축 기준 XY / ZY / XZ이며 측량된 경기장 좌표가 아닙니다. 카메라별로 동일한 전역 원점을 알 수 없어 첫 프레임의 상수 이동만 제거합니다. 프레임별 확대·바닥 이동은 없습니다.</p>
<details><summary>이벤트·상태·남은 깊이 가설</summary><pre id="details"></pre></details><h2>같은 RGB에서 얻은 마스크·깊이 관측</h2><p class="legend">아래 자료는 GT가 아닙니다. 현재 시간과 가장 가까운 관측 프레임 번호를 표시합니다.</p><div id="evidenceLabel"></div><img id="evidence"><script>
let data=null,index=0,token=0;const $=id=>document.getElementById(id),edges=[[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]],halpeMap=[11,12,13,14,15,16,5,6,7,8,9,10];
function canvases(id){for(let i=0;i<3;i++){let c=document.createElement('canvas');c.width=420;c.height=360;c.className='plot';$(id).append(c)}}canvases('oldplots');canvases('newplots');
function plot(canvas,points,color,view,range){let ctx=canvas.getContext('2d'),W=420,H=360;ctx.clearRect(0,0,W,H);ctx.fillStyle='#17212d';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#334454';ctx.lineWidth=1;let scale=135/range;for(let n=-Math.ceil(range);n<=range;n++){let x=W/2+n*scale;ctx.beginPath();ctx.moveTo(x,35);ctx.lineTo(x,H-20);ctx.stroke();let y=H/2+n*scale;ctx.beginPath();ctx.moveTo(20,y);ctx.lineTo(W-20,y);ctx.stroke()}ctx.fillStyle='#d4dfec';ctx.font='13px system-ui';ctx.fillText(['카메라 XY (정면)','카메라 ZY (측면)','카메라 XZ (상단)'][view]+' ±'+range.toFixed(2)+' leg',12,21);if(!points){ctx.fillText('이 모델은 전역 궤적을 제공하지 않음 — 상대 자세 모드에서 확인',10,70);return}let dims=[[0,1],[2,1],[0,2]][view],xy=points.map(p=>[W/2+p[dims[0]]*scale,H/2+p[dims[1]]*scale]);ctx.strokeStyle=color;ctx.lineWidth=3;for(let [a,b]of edges){if(!xy[a]?.every(Number.isFinite)||!xy[b]?.every(Number.isFinite))continue;ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke()}for(let j=0;j<xy.length;j++){if(!xy[j]?.every(Number.isFinite))continue;ctx.fillStyle=j%2===0?'#ff7991':color;ctx.beginPath();ctx.arc(...xy[j],4,0,7);ctx.fill()}}
function sourceLine(ctx,p,color){if(!p)return;let xy=p[index];ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,data.width/650);for(let[a,b]of edges){if(!xy[a]?.every(Number.isFinite)||!xy[b]?.every(Number.isFinite))continue;ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke()}}
function render(){if(!data)return;let name=$('candidate').value,mode=$('mode').value,old=data.models.current_actual,now=data.models[name];let range=mode==='relative'?1.85:data.extent;for(let v=0;v<3;v++){plot($('oldplots').children[v],old[mode]?.[index],'#f5ab59',v,range);plot($('newplots').children[v],now[mode]?.[index],'#70d8ff',v,range)}let c=$('overlay'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if($('obs').checked){ctx.fillStyle='#54ff8d';for(let j=0;j<26;j++){let p=data.halpe[index][j];if(data.scores[index][j]>.3&&p.every(Number.isFinite)){ctx.beginPath();ctx.arc(...p,data.width/500,0,7);ctx.fill()}}}if($('old2d').checked)sourceLine(ctx,old.projected,'#f5ab59');if($('new2d').checked)sourceLine(ctx,now.projected,'#70d8ff');$('seek').value=index;$('time').textContent=`${index+1}/${data.times.length} · 원본 ${data.source_times[index].toFixed(3)} s`;
let key=Array.from({length:13},(_,i)=>Math.round(i*(data.times.length-1)/12)).reduce((a,b)=>Math.abs(a-index)<=Math.abs(b-index)?a:b);let src=`observations/${data.id}/${String(key).padStart(5,'0')}_evidence.jpg`;if(!$('evidence').src.endsWith(src)){$('evidence').src=src;$('evidenceLabel').textContent=`관측 frame ${key} · 현재 frame ${index}`}}
async function load(id){let mine=++token;$('video').pause();let r=await fetch(`viewer_data/${id}.json`);let z=await r.json();if(mine!==token)return;data=z;index=0;$('video').onerror=()=>{if(!$('video').src.endsWith('.webm'))$('video').src=data.video.replace('.mp4','.webm')};$('video').src=$('video').canPlayType('video/mp4; codecs="avc1.42E01E"')?data.video:data.video.replace('.mp4','.webm');$('overlay').width=data.width;$('overlay').height=data.height;$('seek').max=data.times.length-1;$('candidate').replaceChildren();for(let name of Object.keys(data.models)){let o=new Option(name,name);$('candidate').add(o)}let requested=new URLSearchParams(location.search).get('model');$('candidate').value=requested&&data.models[requested]?requested:data.selection.reference;$('status').textContent=`${data.player} · ${data.split} · 기존 ${data.legacy_status} → ${data.selection.reference}; 깊이: ${data.selection.depth_status}. 보류 RGB 윤곽 변화 ${data.selection.withheld_mask_delta.toFixed(4)}. ${data.selection.selection_reason}`;$('details').textContent=JSON.stringify({events:data.events,selection:data.selection,source_note:data.source_note},null,2);render()}
function goto(i){if(!data)return;index=Math.max(0,Math.min(data.times.length-1,i));$('video').currentTime=data.times[index];render()}
$('play').onclick=()=>{$('video').paused?$('video').play():$('video').pause()};$('prev').onclick=()=>goto(index-1);$('next').onclick=()=>goto(index+1);$('seek').oninput=e=>goto(+e.target.value);$('clip').onchange=e=>load(e.target.value);for(let id of ['mode','candidate','obs','old2d','new2d'])$(id).onchange=render;function tick(){if(data&&!$('video').paused){let t=$('video').currentTime;index=data.times.reduce((a,v,i)=>Math.abs(v-t)<Math.abs(data.times[a]-t)?i:a,0);render()}requestAnimationFrame(tick)}tick();fetch('viewer_catalog.json').then(r=>r.json()).then(rows=>{for(let r of rows)$('clip').add(new Option(`${r.split} · ${r.player} · ${r.id}${r.changed?' · 신규 후보':''}`,r.id));let id=new URLSearchParams(location.search).get('clip');$('clip').value=rows.some(x=>x.id===id)?id:rows.find(x=>x.id==='4e09dd724a8c_c0031')?.id||rows[0].id;load($('clip').value)});
</script></html>'''
if __name__=='__main__':
 make_data();(OUT/'comparison.html').write_text(HTML);print('VIEWER BUILT')
