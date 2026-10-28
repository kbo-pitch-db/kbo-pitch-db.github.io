/* Public dataset and explicit-link preview use one fixed-scene viewer. CDP :9223. */
const assert=require('node:assert/strict'),sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8779/';
 const target=await(await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));let seq=0;const pending=new Map(),errors=[],requests=[];
 ws.addEventListener('message',e=>{const x=JSON.parse(e.data);if(x.id){const p=pending.get(x.id);pending.delete(x.id);x.error?p.reject(x.error):p.resolve(x.result)}else if(x.method==='Runtime.exceptionThrown')errors.push(x.params);else if(x.method==='Network.requestWillBeSent')requests.push(x.params.request.url)});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value};
 const wait=async expression=>{for(let i=0;i<200;i++){if(await ev(expression))return;await sleep(200)}throw Error('Timeout: '+expression)};
 const navigate=async suffix=>{await send('Page.navigate',{url:base+suffix});await wait("typeof app!=='undefined'&&app.clips.length===813");};
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await navigate('');assert.equal(await ev('app.selected'),null);assert.equal(await ev('app.reconstructions'),null);
 assert.equal(await ev("document.querySelector('#only-current-3d,#only-3d,#reconstruction-method,#reconstruction-coverage')"),null);
 assert.equal(await ev("document.querySelector('a[href*=\"studies/\"]')"),null);
 assert.equal(requests.some(u=>/data\/(reconstruction_registry|preview_registry)\.json/.test(u)),false);
 assert.equal(await ev("app.clips.every(c=>c.motionMethod==='gvhmr-ground-motionbert-v1'&&c.pose3d.includes('combined-v1'))"),true);
 const cid='3fdcd2dbaf58_c0014';await ev(`selectClip('${cid}',{scroll:false})`);await wait('!!app.poses[3]');
 assert.equal(await ev('app.selected.reconstructionId'),'published');assert.equal(await ev('app.poses[3].method'),'gvhmr-ground-motionbert-v1');assert.equal(await ev('app.selected.pose3d===app.clips.find(c=>c.id===app.selected.id).pose3d'),true);
 assert.equal(await ev("document.querySelector('#reconstruction-method,.motion-quality')"),null);
 assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.floorVisible"),'true');
 const firstMatrix=await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix");
 for(const f of [40,80,140]){await ev(`seekFrame(${f})`);assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix"),firstMatrix);}
 await ev('app.video.currentTime=app.video.duration;drawPose()');assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.frame"),'140');
 await ev("window.__downloads=[];window.__anchorClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){__downloads.push({name:this.download,text:fetch(this.href).then(r=>r.text())})};downloadPose('json',3)");
 assert.equal(await ev("(async()=>JSON.stringify(JSON.parse(await __downloads[0].text))===JSON.stringify(app.poses[3]))()"),true);await ev('HTMLAnchorElement.prototype.click=window.__anchorClick');
 await navigate('?clip=8eeef70dce48_c0013');await wait("app.selected?.id==='8eeef70dce48_c0013'&&!!app.poses[3]");
 const cut=await ev('app.poses[3].quality.valid_3d_frames');assert.equal(cut,274);await ev(`seekFrame(${cut})`);
 assert.equal(await ev("document.querySelector('.motion-placeholder[data-dimension=\"3\"]').hidden"),false);
 assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"2\"]').hidden"),false);
 await ev("window.__downloads=[];HTMLAnchorElement.prototype.click=function(){__downloads.push({name:this.download,text:fetch(this.href).then(r=>r.text())})};downloadPose('csv',3)");
 const csv=await ev("__downloads[0].text");assert.ok(csv.split('\n')[0].endsWith('available,unavailable_reason'));assert.equal(csv.split('\n').length,1+276*24);assert.ok(csv.includes('false,"source_camera_change"'));await ev('HTMLAnchorElement.prototype.click=window.__anchorClick');
 await ev(`seekFrame(${cut-1})`);assert.equal(await ev("document.querySelector('.motion-placeholder[data-dimension=\"3\"]').hidden"),true);
 for(const sample of ['d4ea211e62bd_c0026','3fdcd2dbaf58_c0014']){
  await navigate('?clip='+sample+'#explore');await wait('!!app.poses[3]');
  assert.equal(await ev('app.poses[3].fit_provenance.motionbert_applied'),true);
  assert.equal(await ev('app.poses[3].quality.constant_shoe_shape'),true);
  assert.equal(await ev('app.poses[3].quality.contact_consistency_passed'),true);
  const camera=await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix");
  for(const f of [0,10,30]){await ev(`seekFrame(${f})`);assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix"),camera)}
 }
 for(const sample of ['02dad52fd781_c0049','0b70417b94ab_c0010','1161553ed7d7_c0056','3ef5720dccec_c0010','3ef5720dccec_c0021','782c3ca51bd5_c0004','b54e0e701d77_c0006','e72b3a782760_c0009']){
  await navigate('?clip='+sample+'#explore');await wait('!!app.poses[3]&&!!app.poses[2]');
  assert.equal(await ev('app.poses[2].subject'), 'pitcher');
  assert.equal(await ev('app.poses[3].subject'), 'pitcher');
  assert.equal(await ev('app.poses[3].fit_provenance.motionbert_applied'),true);
  assert.ok(await ev("app.selected.poseRefined.includes('subject-tail-v1')"));
  assert.equal(await ev('app.poses[3].quality.full_support_check_passed'),true);
  await ev('seekFrame(app.poses[3].frames.length-1)');
  assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.floorVisible"),'true');
 }
 await navigate('?new3d=1');assert.equal(await ev('app.filtered.length'),813);assert.equal(await ev('app.selected'),null);
 await navigate('?preview=support-v4&clip='+cid);await wait("app.poses[3]?.method==='gvhmr-support-v4'");
 assert.equal(await ev("document.querySelector('meta[name=\"robots\"]').content.includes('noindex')"),true);
 assert.equal(await ev('app.poses[3].scene.show_floor'),true);assert.equal(await ev('app.poses[3].scene.floor.height'),0);assert.equal(await ev('app.poses[3].scene.per_frame_recentering'),false);
 assert.equal(await ev("document.querySelector('#reconstruction-method')"),null);
 const previewMatrix=await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix");
 for(const f of [0,40,80,140]){await ev(`seekFrame(${f})`);assert.equal(await ev("document.querySelector('.motion-canvas[data-dimension=\"3\"]').dataset.cameraMatrix"),previewMatrix);}
 assert.ok(await ev('Math.hypot(...app.poses[3].frames.at(-1).anatomy[0].map((v,k)=>v-app.poses[3].frames[0].anatomy[0][k]))>.3'));
 for(const width of [1440,390,320]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:width<500});assert.ok(await ev('document.documentElement.scrollWidth<=innerWidth+1'),'overflow '+width);}
 await ev("selectClip('1e24233962b6_c0018',{scroll:false})");assert.equal(await ev('app.selected.pose3d'),null);assert.equal(await ev('app.poses[3]'),undefined);
 await navigate('?clip=1e24233962b6_c0018');await wait("app.selected?.id==='1e24233962b6_c0018'&&!!app.poses[3]");assert.equal(await ev('app.selected.reconstructionId'),'published');
 assert.equal(errors.length,0,JSON.stringify(errors));
 console.log(JSON.stringify({base,public_clips:813,public_single_release:true,actual_gvhmr_clips:813,actual_motionbert_clips:813,rigid_shoes:true,recovered_pitcher_tails:8,camera_cut_gap_and_seek_restore:true,no_research_filters_or_links:true,no_auto_selection:true,public_floor:true,explicit_preview:true,preview_fixed_ground_and_camera:true,no_framewise_recentering:true,playback_stops_at_final_frame:true,preview_root_not_reset:true,unavailable_preview_never_substitutes_old_pose:true,exact_download:true,mobile_widths:[1440,390,320],errors:0},null,2));
 ws.close();await fetch('http://127.0.0.1:9223/json/close/'+target.id);
})().catch(e=>{console.error(e);process.exit(1)});
