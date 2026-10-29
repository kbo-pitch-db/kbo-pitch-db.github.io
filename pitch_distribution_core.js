/* Actual 1 km/h histograms; display interpolation never changes source counts. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PitchDistributionCore=api;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
  'use strict';
  const OTHER='other';
  function build(seasons,types){
    const known=new Map(types.map(t=>[t.name.ko,String(t.id)]));
    const definitions=new Map(types.map(t=>[String(t.id),{...t,key:String(t.id)}]));
    definitions.set(OTHER,{key:OTHER,name:{ko:'미분류',en:'Unclassified'},color:'#72786f'});
    let low=Infinity,high=-Infinity;
    const years=Object.entries(seasons).sort(([a],[b])=>Number(a)-Number(b)).map(([year,rows])=>{
      const groups=new Map(),all=new Map(),dates=[];let n=0,missing=0,totalSpeed=0;
      for(const r of rows){
        const speed=r.speed;
        if(typeof speed!=='number'||!Number.isFinite(speed)||speed<=0||speed>=250){missing++;continue;}
        const key=known.get(r.pitch)||OTHER,bin=Math.floor(speed);
        if(!groups.has(key))groups.set(key,{key,n:0,sum:0,bins:new Map()});
        const g=groups.get(key);g.n++;g.sum+=speed;g.bins.set(bin,(g.bins.get(bin)||0)+1);
        all.set(bin,(all.get(bin)||0)+1);n++;totalSpeed+=speed;dates.push(r.date);low=Math.min(low,bin);high=Math.max(high,bin);
      }
      dates.sort();
      return {year:String(year),n,missing,recordCount:rows.length,dateStart:dates[0]||null,dateEnd:dates.at(-1)||null,mean:n?totalSpeed/n:null,groups,all};
    }).filter(y=>y.n);
    if(!years.length)return {years:[],types:[],bins:[],xDomain:[0,1],yMax:1};
    const min=Math.floor((low-3)/5)*5,max=Math.ceil((high+4)/5)*5;
    const bins=Array.from({length:max-min+1},(_,i)=>min+i);
    const used=new Set(years.flatMap(y=>[...y.groups.keys()]));
    const active=[...definitions.values()].filter(t=>used.has(t.key));
    let peak=0;
    for(const y of years){
      y.series={all:bins.map(b=>100*(y.all.get(b)||0)/y.n)};
      for(const type of active){const g=y.groups.get(type.key);y.series[type.key]=bins.map(b=>100*(g?.bins.get(b)||0)/y.n);}
      peak=Math.max(peak,...y.series.all);
    }
    const step=peak>25?10:peak>12?5:peak>5?2:1;
    return {years,types:active,bins,xDomain:[min,max+1],yMax:Math.ceil(peak/step)*step,yStep:step};
  }
  function blend(from,to,t){
    if(t<=0)return from;if(t>=1)return to;
    const out={};for(const key of Object.keys(to))out[key]=to[key].map((v,i)=>(from[key]?.[i]||0)+(v-(from[key]?.[i]||0))*t);return out;
  }
  // Uniform cubic B-spline. Positive basis weights avoid negative densities;
  // padded zero bins retain each type's usage share under the displayed curve.
  function path(values,bins,x,y,area=false){
    if(values.length!==bins.length||values.length<4)return '';
    const p=values.map((v,i)=>[x(bins[i]+.5),y(v)]);
    const weighted=(a,b,c)=>[(a[0]+4*b[0]+c[0])/6,(a[1]+4*b[1]+c[1])/6];
    const start=weighted(p[0],p[1],p[2]);
    let d=`M ${p[0][0]} ${y(0)} L ${start[0]} ${start[1]}`;
    for(let i=0;i<p.length-3;i++){
      const a=p[i+1],b=p[i+2],end=weighted(a,b,p[i+3]);
      d+=` C ${(2*a[0]+b[0])/3} ${(2*a[1]+b[1])/3} ${(a[0]+2*b[0])/3} ${(a[1]+2*b[1])/3} ${end[0]} ${end[1]}`;
    }
    d+=` L ${p.at(-1)[0]} ${y(0)}`;
    return area?d+` L ${p[0][0]} ${y(0)} Z`:d;
  }
  function at(year,type,bin){
    const count=type==='all'?(year.all.get(bin)||0):(year.groups.get(type)?.bins.get(bin)||0);
    return {count,percent:year.n?100*count/year.n:0};
  }
  return {build,blend,path,at};
});
