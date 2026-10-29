const assert=require('node:assert/strict'),fs=require('node:fs'),zlib=require('node:zlib');
const core=require('../pitch_movement_core.js'),manifest=JSON.parse(fs.readFileSync('data/seasons/manifest.json')),types=manifest.pitchTypes;
const row=(hb,ivb,speed=140,pitch='직구')=>({flight:{status:'derived_pts',hb_cm:hb,ivb_cm:ivb},speed,pitch,date:'2024-04-01',id:'test'});
const a=[row(0,0,140),row(2,4,142),row(4,8,144),row(10,1,null,'슬라이더'),row(11,2,120,'슬라이더'),row(null,2),row(2,NaN),{...row(0,0),flight:{status:'no_tracking',hb_cm:0,ivb_cm:0}}];
const b=[row(1,2,141),row(3,6,143),row(5,10,145),row(3,5,130,'커터'),row(4,6,132,'커터'),row(5,7,null,'커터')];
const c=[row(12,2,122,'슬라이더'),row(13,3,123,'슬라이더'),row(14,4,124,'슬라이더')];
const m=core.build({2024:a,2025:b,2026:c},types),first=m.years[0],second=m.years[1];
assert.equal(first.n,5);assert.equal(first.missing,3);assert.equal(first.groups.get('4-Seam').hb,2);assert.equal(first.groups.get('4-Seam').ivb,4);assert.equal(first.groups.get('4-Seam').speed,142);
assert.equal(first.groups.get('Slider').hb,null);assert.equal(second.groups.get('Cutter').speed,null);
const change=core.compare(second,first,'4-Seam');assert.equal(change.status,'ok');assert.equal(change.hb,1);assert.equal(change.ivb,2);assert.equal(change.speed,1);
assert.equal(core.compare(second,first,'Cutter').status,'missing_baseline');assert.equal(core.compare(second,first,'Cutter').hb,null);
assert.equal(core.compare(m.years[2],first,'Slider').status,'few_baseline');assert.equal(core.compare(first,m.years[2],'Slider').status,'few_current');
assert.equal(core.compare(second,first,'Slider').status,'missing_current');assert.equal(core.compare(first,null,'4-Seam').hb,null);
assert.equal(core.baseline(m,'2024'),null);assert.equal(core.baseline(m,'2026').year,'2025');assert.equal(core.baseline(m,'2026','2024').year,'2024');assert.equal(core.baseline(m,'2025','2025'),null);
for(const y of m.years)for(const p of y.points){assert.ok(p.hb>=m.domain.x[0]&&p.hb<=m.domain.x[1]);assert.ok(p.ivb>=m.domain.y[0]&&p.ivb<=m.domain.y[1]);}
for(const domain of [m.domain,...Object.values(m.domains)])assert.equal(domain.x[1]-domain.x[0],domain.y[1]-domain.y[0]);
const before=JSON.stringify(a),s0=core.snapshot(first),s1=core.snapshot(second),mix=core.blend(s0,s1,.5);
assert.equal(mix.means['4-Seam'].hb,2.5);assert.equal(mix.means['4-Seam'].ivb,5);assert.equal(mix.means.Cutter.hb,s1.means.Cutter.hb);assert.equal(mix.means.Cutter.alpha,.5);
assert.equal(mix.weights['2024'],.5);assert.equal(mix.weights['2025'],.5);assert.equal(JSON.stringify(a),before);
const fade=core.blend(s1,core.snapshot(m.years[2]),.5);assert.equal(fade.means['4-Seam'].hb,s1.means['4-Seam'].hb);assert.equal(fade.means['4-Seam'].alpha,.5);
assert.deepEqual(core.blend(s0,s1,0),s0);assert.deepEqual(core.blend(s0,s1,1),s1);
const zero=core.build({2024:[row(0,0),row(0,0),row(0,0)]},types);assert.equal(zero.years[0].n,3);assert.equal(zero.years[0].groups.get('4-Seam').hb,0);
assert.equal(core.build({2024:[row(null,null)]},types).years.length,0);
const source={};for(const [year,meta]of Object.entries(manifest.seasons)){
 const payload=JSON.parse(zlib.gunzipSync(fs.readFileSync('data/'+meta.shards[76715%16].path)));
 source[year]=payload.players['76715'].map(r=>({...row(r[4]===null?null:r[4]/1000,r[5]===null?null:r[5]/1000,r[3],types[r[2]]?.name.ko),date:meta.games[r[1]].slice(0,8)}));
}
const real=core.build(source,types);assert.deepEqual(real.years.map(y=>y.n),[2490,2267,1529]);
for(const y of real.years){const raw=source[y.year].filter(r=>r.pitch==='직구');const g=y.groups.get('4-Seam');assert.equal(raw.length,g.n);assert.ok(Math.abs(raw.reduce((n,r)=>n+r.flight.hb_cm,0)/raw.length-g.hb)<1e-9);}
assert.equal(core.compare(real.years[2],real.years[1],'Slider').status,'missing_baseline');
console.log('PASS: paired movement means, zero/null handling, sample gates, exact signed deltas, missing types, isotropic fixed domains, real three-season records and non-fabricated fades');
