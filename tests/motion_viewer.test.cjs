/* Exercise the numeric viewer without browser/network dependencies. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Element {
  constructor(){this.nodes={};this.listeners={};this.value='';this.hidden=false;this.isConnected=true;this.classList={toggle(){}};}
  querySelector(q){return this.nodes[q] ||= new Element();}
  addEventListener(key,handler){this.listeners[key]=handler;}
  setAttribute(){}
  append(){}
  after(){}
  getContext(){return new Proxy({}, {get:(o,k)=>o[k] ||= (()=>{}),set:(o,k,v)=>(o[k]=v,true)});}
  event(key, value){this.value=value;this.listeners[key]({target:this});}
}
const root=path.resolve(__dirname,'..');
const index=JSON.parse(fs.readFileSync(path.join(root,'data/clips.json')));
const clip=index.clips.find(c=>c.pose3d);
assert.ok(clip,'fixture has genuine learned 3D');
const load=p=>JSON.parse(fs.readFileSync(path.join(root,'data',p)));
const panel=new Element();
panel.querySelector('video').currentTime=.8;
panel.querySelector('video').paused=true;
const controls=[];
let now=0, nextTimer=0;
const timers=new Map();
const context={window:{},console,Math,fetch:async p=>({ok:true,json:async()=>load(p)}),
  EventTarget,Event,performance:{now:()=>now},
  document:{createElement:()=>{const e=new Element();controls.push(e);return e;}},
  cancelAnimationFrame(id){timers.delete(id);},requestAnimationFrame(fn){timers.set(++nextTimer,fn);return nextTimer;}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'motion_viewer.js'),'utf8'),context);
(async()=>{
  let active;
  const raw=load(clip.pose);
  await context.window.MotionViewer.mount({clip,panel,lang:'ko',url:p=>p,raw,onPose:p=>{active=p;}});
  const control=controls[0],mode=control.querySelector('.motion-mode');
  assert.equal(mode.disabled,false);
  assert.equal(active.dimension,2);
  mode.event('change','smooth3');
  assert.equal(active.dimension,3);
  assert.equal(active.frames.length,raw.frames.length);
  const p=active.frames.flatMap(f=>f.keypoints).find(Boolean);
  assert.equal(p.length,4,'3D has xyz and visibility');
  const expected=load(clip.pose3dRaw);
  control.querySelector('.motion-blend input').event('input','0');
  assert.equal(JSON.stringify(active.frames),JSON.stringify(expected.frames),'zero correction is exactly original learned pose');
  control.querySelector('.motion-blend input').event('input','50');
  assert.equal(active.display_correction,.5);
  control.querySelector('.motion-yaw').event('input','90');
  context.window.MotionViewer.draw();
  mode.event('change','raw2');
  assert.equal(JSON.stringify(active.frames),JSON.stringify(raw.frames));
  assert.equal(active.dimension,2);
  context.window.MotionViewer.dispose();
  const clock=context.window.MotionViewer.createClock(new Element(),90,30);
  assert.equal(clock.duration,3);
  let updates=0;clock.addEventListener('timeupdate',()=>updates++);
  clock.currentTime=1;assert.equal(clock.currentTime,1);
  clock.playbackRate=.5;await clock.play();assert.equal(clock.paused,false);
  now=1000;[...timers.values()].at(-1)(now);
  assert.equal(clock.currentTime,1.5,'numeric playback respects speed and elapsed time');
  clock.pause();assert.equal(clock.paused,true);
  clock.currentTime=99;assert.equal(clock.currentTime,3);
  assert.ok(updates>=3);
  const numericClip={...clip,poseVideo:''};
  const before=controls.length;
  await context.window.MotionViewer.mount({clip:numericClip,panel,lang:'en',url:p=>p,raw,clock,onPose:p=>{active=p;}});
  controls[before].querySelector('.motion-mode').event('change','raw2');
  assert.equal(controls[before+1].hidden,false,'numeric-only original 2D renders on canvas');
  context.window.MotionViewer.dispose();
  const halpeNames=['head','neck','hip','left_big_toe','right_big_toe','left_small_toe','right_small_toe','left_heel','right_heel'];
  const halpe={...raw,model:'RTMPose Halpe-26 synthetic UI fixture',joints:[...raw.joints,...halpeNames],
    edges:[[5,7],[7,9],[15,20],[15,22],[15,24],[16,21],[16,23],[16,25]],
    frames:raw.frames.map(f=>({...f,keypoints:[...f.keypoints,...halpeNames.map((_,i)=>[100+i,300+i,.9])]}))};
  context.fetch=async()=>({ok:true,json:async()=>halpe});
  const halpeClip={...clip,poseVideo:'',pose3d:undefined,pose3dRaw:undefined,poseRefined:'synthetic-halpe-fixture'};
  const halpeBefore=controls.length;
  await context.window.MotionViewer.mount({clip:halpeClip,panel,lang:'en',url:p=>p,raw:halpe,clock,onPose:p=>{active=p;}});
  assert.equal(active.joints.length,26);
  assert.equal(active.frames[0].keypoints.length,26);
  assert.equal(active.edges.at(-1)[1],25);
  controls[halpeBefore].querySelector('.motion-mode').event('change','raw2');
  context.window.MotionViewer.draw();
  context.window.MotionViewer.dispose();
  console.log('PASS: view modes, shared frames, 3D export, correction endpoints, rotation and numeric playback clock');
})().catch(e=>{console.error(e);process.exitCode=1});
