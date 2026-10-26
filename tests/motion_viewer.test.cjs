/* Exercise final-pair playback and async loading without browser dependencies. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
class Element {
  constructor(){this.nodes={};this.listeners={};this.dataset={};this.hidden=true;this.disabled=true;this.isConnected=true;this.width=this.height=720;this.calls=[];this.classList={toggle(){}};}
  querySelector(q){return this.nodes[q] ||= new Element();}
  querySelectorAll(q){return [this.querySelector(q+' first'),this.querySelector(q+' second')];}
  addEventListener(key,handler){this.listeners[key]=handler;}
  after(){}
  getContext(){return this.context ||= new Proxy({}, {get:(o,k)=>o[k] ||= ((...args)=>this.calls.push([k,...args])),set:(o,k,v)=>(o[k]=v,true)});}
  event(key,value){this.value=value;this.listeners[key]({target:this});}
}
const root=path.resolve(__dirname,'..');
const index=JSON.parse(fs.readFileSync(path.join(root,'data/clips.json')));
const clip=index.clips.find(c=>c.pose3d && c.poseRefined);
const load=p=>JSON.parse(fs.readFileSync(path.join(root,'data',p)));
const pose2d=load(clip.poseRefined),pose3d=load(clip.pose3d);
const controls=[],requests=[];
let now=0,nextTimer=0;
const timers=new Map();
let disposedScenes=0;
const context={window:{BoneScene:{createBoneScene(canvas){return {
 render(current,next,fraction,yaw,pitch){canvas.calls.push(['bone-render',current.frame,fraction,yaw,pitch]);},
 resetZoom(){},dispose(){disposedScenes++;}
};}}},console,Math,Object,fetch:async p=>{requests.push(p);return {ok:true,json:async()=>load(p)};},
 EventTarget,Event,performance:{now:()=>now},document:{createElement:()=>{const e=new Element();controls.push(e);return e;}},
 cancelAnimationFrame(id){timers.delete(id);},requestAnimationFrame(fn){timers.set(++nextTimer,fn);return nextTimer;}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'motion_viewer.js'),'utf8'),context);
const viewer=context.window.MotionViewer;
const canvas=(panel,dim)=>panel.querySelector(`.motion-canvas[data-dimension="${dim}"]`);
(async()=>{
 const panel=new Element(),clock=viewer.createClock(new Element(),clip.frames,clip.fps);
 clock.currentTime=.5;
 let poses;
 await viewer.mount({clip,panel,lang:'ko',url:p=>p,pose2d,clock,onPoses:p=>{poses=p;}});
 assert.deepEqual(requests,[clip.pose3d],'only final 3D is fetched after final 2D');
 assert.equal(poses[2],pose2d,'2D export is the exact final result');
 assert.equal(JSON.stringify(poses[3]),JSON.stringify(pose3d),'3D export is the exact final result');
 assert.equal(poses[2].joints.length,26);assert.equal(poses[3].joints.length,17);
 assert.equal(canvas(panel,2).hidden,false);assert.equal(canvas(panel,3).hidden,false);
 assert.equal(canvas(panel,2).dataset.frame,canvas(panel,3).dataset.frame);
 assert.ok(!controls[0].innerHTML.includes('motion-mode') && !controls[0].innerHTML.includes('motion-blend'));
 clock.currentTime=1;viewer.draw();
 assert.equal(canvas(panel,2).dataset.frame,String(Math.round(clip.fps)));
 assert.equal(canvas(panel,2).dataset.frame,canvas(panel,3).dataset.frame,'seeking updates both panels');
 canvas(panel,2).calls=[];canvas(panel,3).calls=[];viewer.draw();
 const before2=JSON.stringify(canvas(panel,2).calls),before3=JSON.stringify(canvas(panel,3).calls);
 canvas(panel,2).calls=[];canvas(panel,3).calls=[];
 controls[0].querySelector('.motion-yaw').event('input','90');
 assert.equal(JSON.stringify(canvas(panel,2).calls),before2,'rotation preserves 2D projection');
 assert.notEqual(JSON.stringify(canvas(panel,3).calls),before3,'rotation updates 3D projection');
 await clock.play();now=200;[...timers.values()].forEach(fn=>fn(now));
 assert.equal(canvas(panel,2).dataset.frame,canvas(panel,3).dataset.frame,'playback shares the same clock');
 clock.pause();viewer.dispose();
 assert.equal(disposedScenes,1,'clip disposal releases its 3D scene');
 const plain=viewer.createClock(new Element(),90,30);
 assert.equal(plain.duration,3);plain.currentTime=1;plain.playbackRate=.5;await plain.play();
 now=1200;[...timers.values()].at(-1)(now);assert.equal(plain.currentTime,1.5);
 plain.pause();plain.currentTime=99;assert.equal(plain.currentTime,3);
 context.fetch=async()=>({ok:false});
 const failurePanel=new Element();let supported;
 await viewer.mount({clip,panel:failurePanel,lang:'en',url:p=>p,pose2d,clock:plain,onPoses:p=>{supported=p;}});
 assert.equal(supported[2],pose2d);assert.equal(supported[3],undefined);
 assert.equal(canvas(failurePanel,2).hidden,false,'3D loading failure preserves 2D');
 assert.equal(canvas(failurePanel,3).hidden,true);
 viewer.dispose();
 const misaligned=JSON.parse(JSON.stringify(pose3d));misaligned.frames[0].source_time+=1;
 context.fetch=async()=>({ok:true,json:async()=>misaligned});
 await viewer.mount({clip,panel:new Element(),lang:'en',url:p=>p,pose2d,clock:plain,onPoses:p=>{supported=p;}});
 assert.equal(supported[3],undefined,'misaligned 3D cannot appear synchronized');viewer.dispose();
 let resolveOld,oldUpdates=0;
 context.fetch=()=>new Promise(resolve=>{resolveOld=resolve;});
 const pending=viewer.mount({clip,panel:new Element(),lang:'en',url:p=>p,pose2d,clock:plain,onPoses:()=>oldUpdates++});
 context.fetch=async()=>({ok:true,json:async()=>pose3d});
 await viewer.mount({clip,panel:new Element(),lang:'en',url:p=>p,pose2d,clock:plain,onPoses:p=>{supported=p;}});
 resolveOld({ok:true,json:async()=>pose3d});await pending;
 assert.equal(oldUpdates,1,'a stale request cannot replace the newly selected clip');
 assert.equal(supported[3],pose3d);viewer.dispose();
 console.log('PASS: final 2D/3D, synchronized playback and seeking, independent 3D rotation, exact exports, loading failure and clip-switch races');
})().catch(e=>{console.error(e);process.exitCode=1});
