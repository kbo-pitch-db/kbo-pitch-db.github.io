/* Comparable PTS movement samples only; no pitch relabelling or imputation. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PitchMovementCore=api;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
  'use strict';
  const MINIMUM=3;
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  function bounds(points,origin=false){
    if(!points.length)return {x:[-10,10],y:[-10,10]};
    let minX=origin?0:Infinity,maxX=origin?0:-Infinity,minY=origin?0:Infinity,maxY=origin?0:-Infinity;
    for(const p of points){minX=Math.min(minX,p.hb);maxX=Math.max(maxX,p.hb);minY=Math.min(minY,p.ivb);maxY=Math.max(maxY,p.ivb);}
    const cx=origin?0:(minX+maxX)/2,cy=origin?0:(minY+maxY)/2;
    const radius=origin?Math.ceil((Math.max(Math.abs(minX),Math.abs(maxX),Math.abs(minY),Math.abs(maxY))+5)/10)*10:Math.ceil((Math.max(maxX-minX,maxY-minY)/2+4)/5)*5;
    return {x:[cx-radius,cx+radius],y:[cy-radius,cy+radius]};
  }
  function build(seasons,types){
    const known=new Map(types.map(t=>[t.name.ko,String(t.id)]));
    const definitions=[...types.map(t=>({...t,key:String(t.id)})),{key:'other',name:{ko:'미분류',en:'Unclassified'},color:'#72786f'}];
    const all=[],used=new Set();
    const years=Object.entries(seasons).sort(([a],[b])=>Number(a)-Number(b)).map(([year,rows])=>{
      const groups=new Map(),points=[],dates=[];
      for(const r of rows){
        const f=r.flight;if(f?.status!=='derived_pts'||!finite(f.hb_cm)||!finite(f.ivb_cm))continue;
        const key=known.get(r.pitch)||'other',speed=finite(r.speed)&&r.speed>0&&r.speed<250?r.speed:null;
        const p={key,hb:f.hb_cm,ivb:f.ivb_cm,speed,date:r.date,id:r.id};
        if(!groups.has(key))groups.set(key,{key,n:0,hbSum:0,ivbSum:0,speedN:0,speedSum:0,points:[]});
        const g=groups.get(key);g.n++;g.hbSum+=p.hb;g.ivbSum+=p.ivb;g.points.push(p);
        if(speed!==null){g.speedN++;g.speedSum+=speed;}
        points.push(p);all.push(p);dates.push(r.date);used.add(key);
      }
      for(const g of groups.values()){g.hb=g.n>=MINIMUM?g.hbSum/g.n:null;g.ivb=g.n>=MINIMUM?g.ivbSum/g.n:null;g.speed=g.speedN>=MINIMUM?g.speedSum/g.speedN:null;}
      dates.sort();
      return {year:String(year),n:points.length,recordCount:rows.length,missing:rows.length-points.length,dateStart:dates[0]||null,dateEnd:dates.at(-1)||null,groups,points};
    }).filter(y=>y.n);
    const active=definitions.filter(t=>used.has(t.key));
    return {years,types:active,domain:bounds(all,true),domains:Object.fromEntries(active.map(t=>[t.key,bounds(all.filter(p=>p.key===t.key))])),minimum:MINIMUM};
  }
  function baseline(model,year,mode='previous'){
    const i=model.years.findIndex(y=>y.year===String(year));
    return mode==='previous'?(model.years[i-1]||null):model.years.find(y=>y.year===String(mode)&&y.year!==String(year))||null;
  }
  function compare(now,base,key){
    const a=now.groups.get(key),b=base?.groups.get(key);
    const status=!a?'missing_current':!base?'no_baseline':!b?'missing_baseline':a.n<MINIMUM?'few_current':b.n<MINIMUM?'few_baseline':'ok';
    return {status,current:a||null,base:b||null,hb:status==='ok'?a.hb-b.hb:null,ivb:status==='ok'?a.ivb-b.ivb:null,speed:status==='ok'&&a.speed!==null&&b.speed!==null?a.speed-b.speed:null};
  }
  function snapshot(year){return {means:Object.fromEntries([...year.groups].filter(([,g])=>g.hb!==null).map(([key,g])=>[key,{hb:g.hb,ivb:g.ivb,alpha:1}])),weights:{[year.year]:1}};}
  function blend(from,to,t){
    if(t<=0)return from;if(t>=1)return to;const means={},weights={};
    for(const key of new Set([...Object.keys(from.means),...Object.keys(to.means)])){
      const a=from.means[key],b=to.means[key];
      means[key]=a&&b?{hb:a.hb+(b.hb-a.hb)*t,ivb:a.ivb+(b.ivb-a.ivb)*t,alpha:a.alpha+(b.alpha-a.alpha)*t}:a?{...a,alpha:a.alpha*(1-t)}:{...b,alpha:b.alpha*t};
    }
    for(const key of new Set([...Object.keys(from.weights),...Object.keys(to.weights)]))weights[key]=(from.weights[key]||0)*(1-t)+(to.weights[key]||0)*t;
    return {means,weights};
  }
  return {build,bounds,baseline,compare,snapshot,blend,MINIMUM};
});
