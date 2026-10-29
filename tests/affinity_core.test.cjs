const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const core=require('../affinity_core.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../data/visuals/affinity.json')));
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
function fixture(counts,pointsByType){
 return {counts,types:pointsByType.map(points=>points?{projections:data.similarity.directions.map(dir=>{
  const values=points.map(p=>p.reduce((sum,v,i)=>sum+v/10*dir[i],0)).sort((a,b)=>a-b);
  return Array.from({length:65},(_,i)=>{const x=(i+.5)/65*(values.length-1),lo=Math.floor(x),hi=Math.ceil(x);return values[lo]+(values[hi]-values[lo])*(x-lo);});
 })}:null)};
}
const a=fixture([100,0],[[[140,10,30],[150,20,40]],null]);
near(core.similarity(a,a),100);
near(core.similarity(a,fixture([1000,0],[[[140,10,30],[150,20,40]],null])),100);
const other=fixture([0,100],[null,[[140,10,30],[150,20,40]]]);near(core.similarity(a,other),0);
const mixed=fixture([50,50],[[[140,10,30],[150,20,40]],[[140,10,30],[150,20,40]]]);near(core.similarity(a,mixed),50);
for(let feature=0;feature<3;feature++){
 const shifted=fixture([100,0],[[[140,10,30],[150,20,40]].map(p=>p.map((v,i)=>v+(i===feature?10:0))),null]);
 assert.ok(core.similarity(a,shifted)<80,`Feature ${feature} must influence the score`);
}
// Identical means, changed variance; a mean-only comparison would miss this.
const constant=fixture([100,0],[[[145,15,35],[145,15,35]],null]);assert.ok(core.similarity(a,constant)<100);
// Equal means and variances, different shapes: the comparison uses more than moments.
const twoPeaks=fixture([100,0],[[[145,-10,30],[145,-10,30],[145,10,30],[145,10,30]],null]);
const threePeaks=fixture([100,0],[[[145,-Math.sqrt(200),30],[145,0,30],[145,0,30],[145,Math.sqrt(200),30]],null]);
assert.ok(core.similarity(twoPeaks,threePeaks)<100);
// Same marginal histograms, opposite HB/IVB covariance. Joint projections differ.
const corr=fixture([100,0],[[[145,-10,-10],[145,10,10]],null]);
const anti=fixture([100,0],[[[145,-10,10],[145,10,-10]],null]);
assert.ok(core.similarity(corr,anti)<100);
// Type identity matters even with identical aggregate three-dimensional clouds.
const swapped=fixture([50,50],[[[150,20,40]],[[140,10,30]]]);
const unSwapped=fixture([50,50],[[[140,10,30]],[[150,20,40]]]);assert.ok(core.similarity(swapped,unSwapped)<100);
assert.throws(()=>core.proportions([0,0]));assert.throws(()=>core.proportions([NaN,1]));
assert.throws(()=>core.similarity(a,{counts:[100,0],types:[null,null]}));
for(const p of data.players){
 near(core.similarity(p,p),100);
 const ranked=core.rank(data.players,p);assert.equal(ranked.length,data.players.length-1);
 for(let i=0;i<ranked.length;i++){
  const q=ranked[i];assert.notEqual(p.key,q.player.key);assert.ok(Number.isFinite(q.score)&&q.score>=0&&q.score<=100);
  near(q.score,core.similarity(q.player,p));if(i)assert.ok(ranked[i-1].score>=q.score);
 }
}
for(const [raw,expected]of [['',30],['NaN',30],['Infinity',30],['-10',0],['1000',100],['45.25',45.3]])assert.equal(core.parse('#visuals/affinity?v=arsenal1&s='+raw).threshold,expected);
const state={key:'54640',threshold:43.2,compare:'67246'};assert.deepEqual(core.parse(core.fragment(state)),state);
const legacy=core.parse('#visuals/affinity?players=pitchers&player=54640&s=95');assert.equal(legacy.key,'54640');assert.equal(legacy.threshold,30);
assert.equal(core.fragment(state).includes('players='),false);
console.log('PASS: usage, each feature, spread, covariance, pitch-type identity, all-pitcher rankings and link migration.');
