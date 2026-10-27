/* Browser integration against local or deployed static pages; requires CDP :9223. */
const assert=require('node:assert/strict'),sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8776/studies/support-transfer-20261028/';
 const target=await(await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));let seq=0;
 const pending=new Map(),errors=[],requests=[];
 ws.addEventListener('message',e=>{const x=JSON.parse(e.data);if(x.id){const p=pending.get(x.id);pending.delete(x.id);x.error?p.reject(x.error):p.resolve(x.result)}else if(x.method==='Runtime.exceptionThrown')errors.push(x.params);else if(x.method==='Network.requestWillBeSent')requests.push(x.params.request.url)});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const ev=async expression=>{const x=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(x.exceptionDetails)throw Error(JSON.stringify(x.exceptionDetails));return x.result.value};
 const wait=async expression=>{for(let i=0;i<180;i++){if(await ev(expression))return;await sleep(200)}throw Error('Timeout: '+expression)};
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});
 await send('Network.setBlockedURLs',{urls:['https://www.youtube-nocookie.com/*']});
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:base+'?clip=3fdcd2dbaf58_c0014&frame=40'});await wait("typeof data!=='undefined'&&!!data&&!document.querySelector('#loading').textContent");
 assert.equal(await ev('frame'),40);assert.equal(await ev("document.querySelectorAll('canvas').length"),8);
 assert.equal(await ev("document.querySelector('#baseline').value"),'gvhmr_f4');assert.equal(await ev("document.querySelector('#candidate').value"),'root_support_transfer');
 assert.equal(await ev("document.querySelector('iframe')"),null);assert.ok(requests.every(u=>!u.includes('youtube-nocookie.com')));
 const extent=await ev('data.extent');await ev("$('candidate').value='root_temporal_only';$('candidate').onchange()");assert.equal(await ev('data.extent'),extent);assert.equal(await ev('frame'),40);
 await ev("$('plant').click()");assert.equal(await ev('frame===Math.round((data.phase.plant_interval[0]+data.phase.plant_interval[1])/2)'),true);
 await ev("$('play').click()");await sleep(250);assert.ok(await ev('frame>40'));await ev("$('play').click()");
 await ev("$('source-video').open=true");await wait("!!document.querySelector('#source-player iframe')");
 const embed=await ev("document.querySelector('#source-player iframe').src");assert.match(embed,/https:\/\/www.youtube-nocookie.com\/embed\/[A-Za-z0-9_-]{11}\?start=\d+&end=\d+/);
 await ev("$('source-video').open=false");await wait("!document.querySelector('iframe')");
 const ids=await ev("[...$('clip').options].map(o=>o.value)");assert.equal(ids.length,12);
 for(const id of ids){
  await ev(`$('clip').value=${JSON.stringify(id)};load()`);assert.equal(await ev('data.id'),id);
  assert.equal(await ev('Object.keys(data.models).length'),3);await ev('step(Math.min(20,data.times.length-1))');
  assert.equal(await ev("[...document.querySelectorAll('canvas')].every(c=>c.dataset.frame===String(frame))"),true);
  assert.equal(await ev("$('plant').disabled===!data.phase.plant_interval"),true);
  assert.equal(await ev("JSON.stringify(data.models.gvhmr_f4.relative)===JSON.stringify(data.models.root_support_transfer.relative)"),true);
  assert.match(await ev("$('download').href"),/\.json\.gz$/);
 }
 for(const width of [1440,390,320]){await send('Emulation.setDeviceMetricsOverride',{width,height:1100,deviceScaleFactor:1,mobile:width<500});assert.ok(await ev('document.documentElement.scrollWidth<=innerWidth+1'),'overflow '+width);}
 assert.equal(errors.length,0,JSON.stringify(errors));
 assert.equal(requests.some(u=>/source_frames|original_clips|\.mp4(?:[?#]|$)/.test(u)),false);
 console.log(JSON.stringify({base,clips:12,models:3,canvases:8,deep_link_frame:true,all_clips_loaded:true,fixed_extent:true,synchronized:true,playback:true,plant_jump:true,unchanged_relative_pose:true,gzip:true,original_youtube_embed_url_checked:true,youtube_playback_not_verified:true,no_self_hosted_source_media:true,mobile_widths:[1440,390,320],errors:0},null,2));
 ws.close();await fetch('http://127.0.0.1:9223/json/close/'+target.id);
})().catch(e=>{console.error(e);process.exit(1)});
