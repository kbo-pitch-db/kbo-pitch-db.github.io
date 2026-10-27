const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.resolve(__dirname,'..');
const clips=JSON.parse(fs.readFileSync(path.join(root,'data/clips.json'))).clips;
const profiles=JSON.parse(fs.readFileSync(path.join(root,'data/player_profiles.json'))).players;
const evidence=JSON.parse(fs.readFileSync(path.join(root,'data/player_identity_evidence.json'))).clips;
const context={URL,document:{currentScript:{src:'https://example.org/lab/app.js'}},window:{}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8').replace('bind();applyLanguage();init();',''),context);
const run=code=>vm.runInContext(code,context);
context.profiles=profiles;
run('app.profiles=profiles');
const keys=new Set();
assert.equal(clips.length,813);
for(const clip of clips){
  assert.equal(clip.playerKnown,true,clip.id);
  assert.notEqual(clip.player,'Unknown pitcher');
  assert.equal(profiles[clip.playerCode].name,clip.player,clip.id);
  assert.ok(fs.existsSync(path.join(root,'data',profiles[clip.playerCode].imagePath)),clip.id);
  if(clip.playerNameSource!=='naver_auto_accepted'){
    assert.equal(evidence[clip.id].name,clip.player);
    assert.equal(evidence[clip.id].code,clip.playerCode);
  }
  context.clip=clip;
  keys.add(run('groupKey(clip)'));
}
assert.equal(keys.size,123);
context.clip=clips.find(c=>c.playerNameSource==='broadcast_text_review');
assert.ok(run('playerPortrait(clip)').includes('/data/portraits/2024/'));
assert.equal(run('playerProfile({...clip,player:"wrong name"})'),null);
assert.equal(run('playerProfile({...clip,playerKnown:false})'),null);
assert.equal(run('nameEvidence(clip)'),'BROADCAST TEXT REVIEWED');
run("app.lang='ko'");
assert.equal(run("displayPlayer({playerKnown:false,player:'Unknown pitcher'})"),'투수 미상');
assert.equal(run('nameEvidence(clip)'),'중계 표기 판독 확인');
const img={dataset:{playerCode:'test',fallbackSrc:'https://cdn.example/portrait.jpg'},
  complete:false,remove(){this.removed=true},addEventListener(event,fn){this[event]=fn}};
context.root={querySelectorAll:()=>[img]};
run('bindPortraits(root)');
img.error();
assert.equal(img.src,'https://cdn.example/portrait.jpg');
assert.equal(run('app.failedPortraits.has("test")'),false);
img.error();
assert.equal(img.removed,true);
assert.equal(run('app.failedPortraits.has("test")'),true);
assert.equal(run('app.failedPortraits.has(clip.playerCode)'),false);
console.log('PASS: 813 identities, 123 player groups, complete portraits, name guards, provenance labels and image fallback');
