const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const code=fs.readFileSync(path.join(__dirname,'../bone_interpolation.js'),'utf8');
 const {interpolateAnatomy}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
 const a=[[0,0,0],[1,0,0],[2,0,0]],b=[[.5,.5,0],[.5,1.5,0],[-.5,1.5,0]],parents=[-1,0,1];
 for(const t of [.1,.25,.5,.9]){
  const p=interpolateAnatomy(a,b,t,parents);
  for(let j=1;j<3;j++)assert.ok(Math.abs(Math.hypot(...p[j].map((x,k)=>x-p[parents[j]][k]))-1)<1e-10,'bone length is preserved');
 }
 assert.deepEqual(interpolateAnatomy(a,b,0,parents),a);assert.deepEqual(interpolateAnatomy(a,b,1,parents),b);
 const reversed=interpolateAnatomy([[0,0,0],[1,0,0]],[[0,0,0],[-1,0,0]],.5,[-1,0]);
 assert.ok(Math.abs(Math.hypot(...reversed[1])-1)<1e-10,'a reversal cannot collapse a bone');
 assert.equal(interpolateAnatomy(a,[null,b[1],b[2]],.5,parents)[1],null,'a missing parent does not manufacture a connected pose');
 console.log('PASS: constant bone lengths between frames, exact saved frames, reversals and missing joints');
})().catch(error=>{console.error(error);process.exit(1)});
