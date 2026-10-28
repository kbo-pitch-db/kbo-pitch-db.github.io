/* Verify the actual default Explorer payload, fixed camera and toe trajectories. */
const assert=require('node:assert/strict'),fs=require('node:fs');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8779/';
 const target=await(await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
 let seq=0;const pending=new Map(),errors=[],requests=[];
 ws.addEventListener('message',event=>{
  const value=JSON.parse(event.data);
  if(value.id){const p=pending.get(value.id);pending.delete(value.id);value.error?p.reject(value.error):p.resolve(value.result)}
  else if(value.method==='Runtime.exceptionThrown')errors.push(value.params);
  else if(value.method==='Network.requestWillBeSent')requests.push(value.params.request.url);
 });
 const call=(method,params={})=>new Promise((resolve,reject)=>{
  const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));
 });
 const ev=async expression=>{
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;
 };
 try{
  await call('Runtime.enable');await call('Page.enable');await call('Network.enable');
  await call('Network.setCacheDisabled',{cacheDisabled:true});
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:base+'?clip=d4ea211e62bd_c0026#explore'});
  let ready=false;
  for(let i=0;i<200;i++){
   if(await ev("typeof app!=='undefined'&&app.selected?.id==='d4ea211e62bd_c0026'&&!!app.poses[3]")){ready=true;break}
   await sleep(200);
  }
  assert.ok(ready,'Default repaired 3D did not load');
  assert.equal(await ev('app.reconstructions'),null);
  assert.equal(await ev('app.poses[3].postprocessing_version'),'independent-support-v7');
  assert.equal(await ev('app.selected.reconstructionId'),'published');
  assert.ok(requests.some(url=>url.includes('gvhmr-release-v1/support-v7/d4ea211e62bd_c0026.json.gz')));
  assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.renderer"),'bone-mesh');
  const matrix=await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix");
  for(const frame of [0,16,26,32,42,54,61]){
   await ev(`app.video.pause();seekFrame(${frame})`);
   assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix"),matrix);
   assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.frame"),String(frame));
  }
  const toe=await ev(`(async()=>{
   const audit=await fetch(new URL(app.selected.supportAudit,DATA_URL)).then(r=>r.json());
   return audit.after_episodes.map(e=>{
    const p=app.poses[3].frames.slice(e.start,e.end+1).map(f=>f.feet[e.marker]);
    return {marker:e.marker,start:e.start,end:e.end,extent:Math.max(...p.map(q=>Math.hypot(...q.map((v,k)=>v-p[0][k]))))/audit.normalization_leg_model};
   });
  })()`);
  const seen=new Set();
  for(const e of toe){assert.ok(e.extent<=(seen.has(e.marker)?.05:.03));seen.add(e.marker)}
  await ev("seekFrame(42);document.querySelector('#detail-panel').scrollIntoView({block:'start'});document.querySelector('.motion-yaw').value='90';document.querySelector('.motion-yaw').dispatchEvent(new Event('input'))");
  await sleep(150);
  const rect=await ev("(()=>{const r=document.querySelector('.motion-canvas[data-dimension=\"3\"]').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()");
  const screenshot=await call('Page.captureScreenshot',{format:'png',clip:rect,captureBeyondViewport:true});
  fs.writeFileSync('/tmp/choi-support-viewer.png',Buffer.from(screenshot.data,'base64'));
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({base,default_repaired_clip:true,fixed_camera:true,renderer:'bone-mesh',toe,errors:0},null,2));
 }finally{ws.close();await fetch('http://127.0.0.1:9223/json/close/'+target.id)}
})().catch(error=>{console.error(error);process.exit(1)});
