/* Same Explorer, fixed ground/camera and no fallback for explicit previews. */
const assert=require('node:assert/strict'),sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8776/';
 const target=await(await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));let seq=0;const pending=new Map(),errors=[],requests=[];
 ws.addEventListener('message',e=>{const x=JSON.parse(e.data);if(x.id){const p=pending.get(x.id);pending.delete(x.id);x.error?p.reject(x.error):p.resolve(x.result)}else if(x.method==='Runtime.exceptionThrown')errors.push(x.params);else if(x.method==='Network.requestWillBeSent')requests.push(x.params.request.url)});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value};
 const wait=async expression=>{for(let i=0;i<200;i++){if(await ev(expression))return;await sleep(200)}throw Error('Timeout: '+expression)};
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});
 await send('Page.navigate',{url:base});await wait("typeof app!=='undefined'&&app.clips.length===813");assert.equal(await ev('app.reconstructions'),null);assert.equal(await ev('app.selected'),null);assert.equal(requests.some(u=>u.includes('preview_rear_pivot_registry')),false);
 const results=[];
 for(const cid of ['3fdcd2dbaf58_c0014','d30b22cfc0b0_c0036']){
  await send('Page.navigate',{url:base+'?preview=rear-pivot&clip='+cid+'#explore'});await wait(`typeof app!=='undefined'&&app.selected?.id==='${cid}'&&app.poses[3]?.method==='gvhmr-rear-pivot-v14'`);
  assert.equal(await ev("document.querySelector('#reconstruction-method,#only-current-3d')"),null);assert.equal(await ev("document.querySelector('meta[name=robots]').content.includes('noindex')"),true);
  const matrix=await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix");
  for(const i of [0,20,40,80]){await ev(`seekFrame(${i})`);assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix"),matrix)}
  assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.floorVisible"),'true');
  results.push(await ev(`(()=>{const p=app.poses[3],m=p.fit_diagnostics.selected_metrics,[a,b]=m.rear_interval,j=m.rear_marker;return {id:app.selected.id,method:p.method,frames:p.frames.length,toe_extent_model:Math.max(...p.frames.slice(a,b+1).map(f=>Math.hypot(...f.feet[j].map((v,k)=>v-p.frames[a].feet[j][k])))),floor:p.scene.floor.height,fixed_camera:true}})()`));
  for(const width of [1440,390,320]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:width<500});assert.ok(await ev('document.documentElement.scrollWidth<=innerWidth+1'))}
 }
 await ev("selectClip('1e24233962b6_c0018',{scroll:false})");assert.equal(await ev('app.selected.pose3d'),null);assert.equal(await ev('app.poses[3]'),undefined);
 assert.equal(errors.length,0,JSON.stringify(errors));console.log(JSON.stringify({base,public_release_unchanged:true,previews:results,no_new_ui_controls:true,missing_preview_never_substituted:true,mobile_widths:[1440,390,320],errors:0},null,2));ws.close();await fetch('http://127.0.0.1:9223/json/close/'+target.id);
})().catch(e=>{console.error(e);process.exit(1)});
