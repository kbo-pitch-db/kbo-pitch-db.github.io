const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const code=fs.readFileSync(path.join(__dirname,'../bone_interpolation.js'),'utf8');
 const {interpolateAnatomy,interpolateRigidFeet}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
 const a=[[0,0,0],[1,0,0],[2,0,0]],b=[[.5,.5,0],[.5,1.5,0],[-.5,1.5,0]],parents=[-1,0,1];
 for(const t of [.1,.25,.5,.9]){
  const p=interpolateAnatomy(a,b,t,parents);
  for(let j=1;j<3;j++)assert.ok(Math.abs(Math.hypot(...p[j].map((x,k)=>x-p[parents[j]][k]))-1)<1e-10,'bone length is preserved');
 }
 assert.deepEqual(interpolateAnatomy(a,b,0,parents),a);assert.deepEqual(interpolateAnatomy(a,b,1,parents),b);
 const reversed=interpolateAnatomy([[0,0,0],[1,0,0]],[[0,0,0],[-1,0,0]],.5,[-1,0]);
 assert.ok(Math.abs(Math.hypot(...reversed[1])-1)<1e-10,'a reversal cannot collapse a bone');
 assert.equal(interpolateAnatomy(a,[null,b[1],b[2]],.5,parents)[1],null,'a missing parent does not manufacture a connected pose');
 const feet=[[0,0,0],[2,0,0],[.1,0,0],[2.1,0,0],[0,0,.25],[2,0,.25]];
 const turn=feet.map((p,j)=>{const anchor=j%2===0?[0,0,0]:[2,0,.25],v=p.map((x,k)=>x-anchor[k]);return [anchor[0]-v[2],anchor[1]+v[1],anchor[2]+v[0]]});
 for(const t of [.1,.25,.5,.9]){
  const f=interpolateRigidFeet(feet,turn,t);
  assert.deepEqual(f[0],feet[0],'stationary supporting toe is exact between frames');
  assert.deepEqual(f[5],feet[5],'stationary supporting heel is exact between frames');
  for(const side of [0,1])for(const [i,j] of [[side,side+2],[side,side+4],[side+2,side+4]])assert.ok(Math.abs(Math.hypot(...f[i].map((x,k)=>x-f[j][k]))-Math.hypot(...feet[i].map((x,k)=>x-feet[j][k])))<1e-10,'rotating shoe cannot shrink');
 }
 assert.deepEqual(interpolateRigidFeet(feet,turn,0),feet);assert.deepEqual(interpolateRigidFeet(feet,turn,1),turn);
 assert.equal(interpolateRigidFeet(feet,turn.map((p,j)=>j===0?null:p),.5)[0],null,'missing shoe observations are not manufactured');
 console.log('PASS: constant bone lengths between frames, exact saved frames, reversals, rigid shoes, stationary support markers and missing joints');
})().catch(error=>{console.error(error);process.exit(1)});
