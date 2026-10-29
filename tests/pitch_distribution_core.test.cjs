const assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const core=require('../pitch_distribution_core.js');
const manifest=JSON.parse(fs.readFileSync('data/seasons/manifest.json'));
const types=manifest.pitchTypes;
const record=(speed,pitch='직구',date='2024-04-01')=>({speed,pitch,date});
const m=core.build({2024:[record(140),record(140.9),record(141),record(130,'슬라이더'),record(null),record(NaN),record(0)],2026:[record(150),record(120,'슬라이더'),record(110,null)]},types);
assert.deepEqual(m.years.map(y=>y.year),['2024','2026']);assert.equal(m.years[0].n,4);assert.equal(m.years[0].missing,3);
assert.deepEqual(core.at(m.years[0],'4-Seam',140),{count:2,percent:50});assert.equal(core.at(m.years[0],'all',141).count,1);
assert.equal(core.at(m.years[1],'other',110).count,1);assert.equal(m.years[1].series.other.reduce((a,b)=>a+b,0),100/3);
for(const year of m.years){
 assert.ok(Math.abs(year.series.all.reduce((a,b)=>a+b,0)-100)<1e-9);
 for(let i=0;i<m.bins.length;i++)assert.ok(Math.abs(year.series.all[i]-m.types.reduce((n,t)=>n+year.series[t.key][i],0))<1e-9);
}
assert.equal(core.build({2024:[record(null)]},types).years.length,0);
const interpolated=core.blend(m.years[0].series,m.years[1].series,.5);
assert.ok(Math.abs(interpolated.all.reduce((a,b)=>a+b,0)-100)<1e-9);
assert.deepEqual(core.blend(m.years[0].series,m.years[1].series,0),m.years[0].series);
assert.deepEqual(core.blend(m.years[0].series,m.years[1].series,1),m.years[1].series);
// The spline conserves usage mass and has no negative frequency controls.
const values=[0,0,0,10,30,5,0,0,0],bins=values.map((_,i)=>i);
const d=core.path(values,bins,x=>x,y=>y);let last=[1.5,0],area=0;
for(const match of d.matchAll(/C\s+([^CMLZ]+)/g)){
 const q=match[1].trim().split(/\s+/).map(Number);const [x1,y1,x2,y2,x3,y3]=q;
 assert.ok([y1,y2,y3].every(y=>y>=0));
 assert.ok(Math.abs((x1-last[0])-(x3-last[0])/3)<1e-9);
 area+=(x3-last[0])*(last[1]+y1+y2+y3)/4;last=[x3,y3];
}
assert.ok(Math.abs(area-values.reduce((a,b)=>a+b,0))<1e-9);
// Read the real existing source shards, without the 500-pitch similarity gate.
const source={};
for(const [year,meta]of Object.entries(manifest.seasons)){
 const shard=meta.shards[76715%manifest.buckets];const payload=JSON.parse(zlib.gunzipSync(fs.readFileSync('data/'+shard.path)));
 source[year]=payload.players['76715'].map(row=>record(row[3],manifest.pitchTypes[row[2]]?.name.ko,meta.games[row[1]].slice(0,8)));
}
const actual=core.build(source,types);
assert.deepEqual(actual.years.map(y=>[y.year,y.n]),[['2024',2490],['2025',2267],['2026',1529]]);
const fixed=[...actual.xDomain,actual.yMax];
for(const y of actual.years){assert.ok(Math.abs(y.series.all.reduce((a,b)=>a+b,0)-100)<1e-8);assert.ok(Math.max(...y.series.all)<=actual.yMax);}
assert.ok(fixed[0]<=90&&fixed[1]>=150);
console.log('PASS: observed bins, denominators, missing seasons, unknown types, spline mass, interpolation endpoints and real three-season Ryu records');
