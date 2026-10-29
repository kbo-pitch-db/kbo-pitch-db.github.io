/* Pitcher-only repertoire overlap and joint speed/HB/IVB distribution metric. */
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AffinityCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  const MODEL='arsenal1', DEFAULT_THRESHOLD=30;
  function unpack(data){
    if(data.similarity.projectionEncoding==='delta-int-1e4'){
      for(const p of data.players)for(const t of p.types)if(t)t.projections=t.projections.map(q=>{let n=0;return q.map(v=>{n+=v;return n/10000});});
      data.similarity.projectionEncoding='float';
    }
    return data;
  }
  function proportions(counts) {
    if(!Array.isArray(counts)||!counts.length||counts.some(v=>!Number.isFinite(v)||v<0))throw Error('Invalid repertoire');
    const total=counts.reduce((a,b)=>a+b,0);
    if(!total)throw Error('Empty repertoire');
    return counts.map(v=>v/total);
  }
  function slicedDistance(a,b) {
    if(!a?.length||a.length!==b?.length)throw Error('Missing joint distribution');
    let sum=0,n=0;
    for(let i=0;i<a.length;i++) {
      if(!a[i].length||a[i].length!==b[i]?.length)throw Error('Incompatible quantiles');
      for(let j=0;j<a[i].length;j++) {
        if(!Number.isFinite(a[i][j])||!Number.isFinite(b[i][j]))throw Error('Invalid quantiles');
        sum+=Math.abs(a[i][j]-b[i][j]);n++;
      }
    }
    return sum/n;
  }
  function compare(a,b) {
    const p=proportions(a.counts),q=proportions(b.counts);
    if(p.length!==q.length)throw Error('Incompatible pitch types');
    let value=0,overlap=0;
    const types=p.map((share,i)=>{
      const mass=Math.min(share,q[i]);overlap+=mass;
      if(!mass)return {overlap:0,distance:null,contribution:0};
      const distance=slicedDistance(a.types[i]?.projections,b.types[i]?.projections);
      const contribution=mass*Math.exp(-distance);value+=contribution;
      return {overlap:mass,distance,contribution};
    });
    return {score:100*Math.max(0,Math.min(1,value)),overlap:100*overlap,types};
  }
  const similarity=(a,b)=>compare(a,b).score;
  function rank(players,selected) {
    return players.filter(p=>p.key!==selected.key).map(player=>({player,score:similarity(selected,player)}))
      .sort((a,b)=>b.score-a.score||a.player.key.localeCompare(b.player.key));
  }
  function parse(hash) {
    const p=new URLSearchParams(hash.split('?')[1]||'');
    // Old scores had a different meaning. Preserve valid pitcher identities,
    // reset old thresholds, and never remount the removed player group.
    const raw=p.get('s'),n=raw===null||raw.trim()===''?NaN:Number(raw);
    return {key:p.get('player')||'',compare:p.get('compare')||'',year:/^202[456]$/.test(p.get('year')||'')?p.get('year'):p.has('v')||p.has('players')?'2024':'',
      threshold:p.get('v')===MODEL&&Number.isFinite(n)?Math.round(Math.max(0,Math.min(100,n))*10)/10:DEFAULT_THRESHOLD};
  }
  function fragment(s) {
    const p=new URLSearchParams({v:MODEL,player:s.key,s:String(s.threshold)});
    if(s.year)p.set('year',s.year);
    if(s.compare)p.set('compare',s.compare);
    return '#visuals/affinity?'+p;
  }
  return {unpack,proportions,slicedDistance,compare,similarity,rank,parse,fragment};
});
