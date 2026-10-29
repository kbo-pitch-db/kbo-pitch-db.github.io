/* Seasonal records stay separate from the immutable 2024 motion catalog. */
window.SeasonRecords=(()=>{
  const BASE=new URL('data/',document.currentScript.src),VERSION='20261030-seasons1';
  let pending=null,index=null;const summaries=new Map(),buckets=new Map();
  async function json(path){
    const u=new URL(path,BASE);u.searchParams.set('v',VERSION);
    const r=await fetch(u,{cache:'no-cache'});if(!r.ok)throw Error('Season records unavailable');
    return path.endsWith('.gz')?ReconstructionRegistry.decode(r,path):r.json();
  }
  async function load(){if(!pending)pending=json('seasons/manifest.json').then(d=>{if(d.schemaVersion!==1)throw Error('Season schema');index=d;return d}).catch(e=>{pending=null;throw e});return pending;}
  const years=()=>index?Object.keys(index.seasons).sort().reverse():['2024'];
  const year=value=>index?.seasons[String(value)]?String(value):String(index?.latestSeason||2024);
  const info=value=>index?.seasons[year(value)];
  async function summary(value){await load();const y=year(value);if(!summaries.has(y))summaries.set(y,json(index.seasons[y].summary.path).catch(e=>{summaries.delete(y);throw e}));return summaries.get(y);}
  async function bucket(y,id){const key=y+':'+id;if(!buckets.has(key)){buckets.set(key,json(index.seasons[y].shards[id].path).catch(e=>{buckets.delete(key);throw e}));while(buckets.size>20)buckets.delete(buckets.keys().next().value);}return buckets.get(key);}
  function decode(row,code,y){
    const meta=index.seasons[y],game=meta.games[row[1]],type=index.pitchTypes[row[2]],id=game+':'+row[0],player=meta.players[code];
    const value=(n,scale)=>row[n]===null?null:row[n]/scale;
    return {id,pitchId:id,playerCode:code,player:player.name,gameId:game,date:game.slice(0,4)+'-'+game.slice(4,6)+'-'+game.slice(6,8),season:y,
      pitch:type?.name.ko||null,speed:row[3],flight:{hb_cm:value(4,1000),ivb_cm:value(5,1000),vaa_deg:value(6,1000),status:row[4]===null?'no_tracking':'derived_pts'},
      plateX:value(7,10000),plateZ:value(8,10000),zoneBottom:value(9,10000),zoneTop:value(10,10000),result:row[11],batterHand:row[12],inning:row[13],atBatPitch:row[14],
      half:null,atBat:null,sequence:null,balls:null,strikes:null,outs:null,batter:null,clipIds:[],primaryClipId:null,linkStatus:'unlinked_record',unit:'pitch_record',
      hand:null,sourceId:null,captureCondition:'no_motion',analysisModel:'no_motion',calculationVersion:index.version,timeStatus:'no_motion',delivery:'unresolved',
      trackingStatus:row[4]===null?'no_tracking':'derived_validated_x',recordSource:row[15]?'retained_naver_relay':'published_naver_pts_csv',
      sourceUrl:'https://api-gw.sports.naver.com/schedule/games/'+game+'/game-polling?inning='+row[13],recordSourceUrl:meta.source.url};
  }
  async function records(value,code=''){
    await load();const y=year(value),meta=index.seasons[y];if(code&&!meta.players[code])return [];
    const ids=code?[Number(code)%index.buckets]:Array.from({length:index.buckets},(_,i)=>i),result=[];
    // Four concurrent small shards; never fetch every season for one profile.
    for(let offset=0;offset<ids.length;offset+=4){const payloads=await Promise.all(ids.slice(offset,offset+4).map(i=>bucket(y,i)));
      for(const payload of payloads)for(const [id,rows]of Object.entries(payload.players))if(!code||id===code)for(const r of rows)result.push(decode(r,id,y));}
    return result;
  }
  function profile(base,value){
    const y=year(value),p=index?.seasons[y].players[base.code];
    return {...base,...(p||{recordCount:0,recordGames:0,recordStart:null,recordEnd:null}),season:y,teams:y==='2024'?base.teams:[],_seasonMissing:!p,
      statistics:p?{...base.statistics,fastballVelocity:p.fastballVelocity,archetype:p.archetype}:{...base.statistics,fastballVelocity:null,archetype:'small_sample'}};
  }
  async function extend(summaryData){
    await load();const existing=new Map(summaryData.players.map(p=>[p.code,p]));
    for(const y of years())for(const [code,p]of Object.entries(index.seasons[y].players))if(!existing.has(code))existing.set(code,{code,name:p.name,season:y,teams:[],hand:null,clipCount:0,clipGames:0,linkedClips:0,dateStart:null,dateEnd:null,detailPath:null,
      recordCount:p.recordCount,recordGames:p.recordGames,recordStart:p.recordStart,recordEnd:p.recordEnd,statistics:{fastballVelocity:p.fastballVelocity,archetype:p.archetype},benchmarks:{},pitchCounts:{},metricEligibleClips:{}});
    summaryData.players=[...existing.values()];summaryData.playerCount=existing.size;summaryData.recordSeasons=years();summaryData.seasonRecordCount=Object.values(index.seasons).reduce((n,s)=>n+s.recordCount,0);
    return summaryData;
  }
  function choices(selected){return years().map(y=>`<option value="${y}" ${y===String(selected)?'selected':''}>${y}</option>`).join('');}
  return {load,years,year,info,summary,records,profile,extend,choices,get index(){return index}};
})();
