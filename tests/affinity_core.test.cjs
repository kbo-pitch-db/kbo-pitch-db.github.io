const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../affinity_core.js');

const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
near(core.similarity([500,0,0,0,0],[30,0,0,0,0]),100);
near(core.similarity([500,0,0,0,0],[0,500,0,0,0]),0);
// Analytic JS divergence: H(3/4,1/4) - (H(1,0)+H(1/2,1/2))/2.
const entropy=-(.75*Math.log2(.75)+.25*Math.log2(.25));
near(core.similarity([1,0,0,0,0],[1,1,0,0,0]),100*(1-Math.sqrt(entropy-.5)));
assert.throws(()=>core.similarity([0,0,0,0,0],[1,0,0,0,0]));
assert.throws(()=>core.proportions([-1,1,1,1,1]));
assert.throws(()=>core.proportions([NaN,1,1,1,1]));
assert.throws(()=>core.proportions([1,2]));
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../data/visuals/affinity.json')));
for(const players of Object.values(data.groups)){
  for(const p of players){
    const ranked=core.rank(players,p);
    assert.equal(ranked.length,players.length-1);
    assert.ok(!ranked.some(q=>q.player.key===p.key));
    for(let i=0;i<ranked.length;i++){
      const q=ranked[i];assert.ok(Number.isFinite(q.score)&&q.score>=0&&q.score<=100);
      near(q.score,core.similarity(q.player.counts,p.counts));
      if(i)assert.ok(ranked[i-1].score>=q.score);
    }
  }
}
for(const [raw,expected]of [['',95],['NaN',95],['Infinity',95],['-10',0],['1000',100],['95.25',95.3]]){
  assert.equal(core.parse('#visuals/affinity?s='+raw).threshold,expected);
}
const state={role:'hitters',key:'68050-L',threshold:96.1,compare:'79530-R'};
assert.deepEqual(core.parse(core.fragment(state)),state);
assert.equal(core.parse('#visuals/affinity?players=invalid').role,'hitters');
console.log('Affinity numerical contract, all-player rankings and shared state passed.');
