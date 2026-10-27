/* Direct-link experiment only: all clips/variants, fixed cameras, exact sync. */
const assert=require('node:assert/strict'),sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8776/';const pageUrl=base+'studies/arm-cocking-20261028/';
 const target=await(await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'})).json();const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));let seq=0;const pending=new Map(),errors=[];
 ws.addEventListener('message',e=>{const x=JSON.parse(e.data);if(x.id){const p=pending.get(x.id);pending.delete(x.id);x.error?p.reject(x.error):p.resolve(x.result)}else if(x.method==='Runtime.exceptionThrown')errors.push(x.params)});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value};
 const wait=async expression=>{for(let i=0;i<200;i++){if(await ev(expression))return;await sleep(200)}throw Error('Timeout: '+expression)};
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:pageUrl});await wait('window.armCocking?.ready');
 assert.equal(await ev("document.querySelector('meta[name=robots]').content"),'noindex,nofollow,noarchive');const clips=await ev('armCocking.catalog.map(c=>c.id)');assert.equal(clips.length,12);let variants=0;
 for(const [ci,cid] of clips.entries()){
  await ev(`armCocking.load('${cid}')`);await wait('armCocking.ready');assert.equal(await ev("document.getElementById('error').textContent"),'');
  const names=await ev('Object.keys(armCocking.data.models)');assert.equal(names.length,16);variants+=names.length;
  for(const variant of ci===0?names:[await ev('armCocking.data.default_variant')]){
   await ev(`document.getElementById('variant').value='${variant}';document.getElementById('variant').dispatchEvent(new Event('change'))`);
   const first=await ev("document.getElementById('baseline').dataset.cameraMatrix");assert.equal(await ev("document.getElementById('candidate').dataset.cameraMatrix"),first);
   for(const expression of ['0','Math.floor(armCocking.times.length/2)','armCocking.times.length-1']){
    await ev(`armCocking.seek(${expression})`);const frames=await ev("['two','baseline','candidate'].map(id=>document.getElementById(id).dataset.frame)");assert.ok(frames.every(f=>f===frames[0]));assert.equal(await ev("document.getElementById('baseline').dataset.cameraMatrix"),first);assert.equal(await ev("document.getElementById('candidate').dataset.cameraMatrix"),first);
   }
   assert.equal(await ev("document.getElementById('candidate').dataset.floorVisible"),'true');
  }
 }
 for(const view of ['front','side','top']){
  await ev(`document.querySelector('[data-view=${view}]').click()`);assert.equal(await ev("document.getElementById('baseline').dataset.cameraMatrix"),await ev("document.getElementById('candidate').dataset.cameraMatrix"));
 }
 for(const width of [1440,390,320]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:width<500});await ev('armCocking.seek(0)');assert.ok(await ev('document.documentElement.scrollWidth<=innerWidth+1'),'overflow '+width)}
 await ev("armCocking.seek(armCocking.times.length-2);document.getElementById('speed').value='1';document.getElementById('play').click()");await wait('!armCocking.playing');assert.equal(await ev('armCocking.index===armCocking.times.length-1'),true);assert.equal(errors.length,0,JSON.stringify(errors));
 console.log(JSON.stringify({base,clips:12,variants:16,clip_variant_payloads:variants,all_payloads_load:true,synchronized:true,identical_fixed_camera:true,floor:true,mobile_widths:[1440,390,320],stops_at_end:true,errors:0},null,2));ws.close();await fetch('http://127.0.0.1:9223/json/close/'+target.id);
})().catch(e=>{console.error(e);process.exit(1)});
