/* Display transforms only. Source angles and exported coordinates stay untouched. */
window.PitchCharts = (() => {
  const finite = value => typeof value==='number' && Number.isFinite(value);
  const clamp = (value,min,max) => Math.max(min,Math.min(max,value));
  function prepareSeries(values,fps=30,smooth=true,options={}){
    const raw=values.map(v=>finite(v)?v:null),filled=[...raw],inferred=raw.map(()=>false);
    const maxGap=Math.max(0,Math.min(3,Math.floor(fps*.1+1e-6)-1));
    for(let start=0;start<raw.length;start++){
      if(raw[start]!==null)continue;
      let end=start;while(end<raw.length&&raw[end]===null)end++;
      const count=end-start;
      if(smooth&&options.bridgeGaps!==false&&start>0&&end<raw.length&&count<=maxGap&&Math.abs(raw[end]-raw[start-1])<=45){
        for(let j=start;j<end;j++){filled[j]=raw[start-1]+(raw[end]-raw[start-1])*(j-start+1)/(count+1);inferred[j]=true}
      }
      start=end-1;
    }
    const display=filled.map((v,i)=>{
      if(!smooth||v===null)return v;
      // Five-frame weighted mean; no smoothing across unsupported intervals.
      let total=v*3,weight=3;
      for(const direction of [-1,1])for(let d=1;d<=2;d++){
        const j=i+direction*d;if(j<0||j>=filled.length||filled[j]===null)break;
        const w=3-d;total+=filled[j]*w;weight+=w;
      }
      return total/weight;
    });
    const gaps=[];for(let i=0;i<display.length;i++){if(display[i]!==null)continue;const start=i;while(i+1<display.length&&display[i+1]===null)i++;gaps.push([start,i])}
    return {raw,display,inferred,gaps,bridgedFrames:inferred.filter(Boolean).length,smooth};
  }
  function paths(series,width=720,height=240,padding={left:44,right:18,top:18,bottom:36}){
    const values=series.display, count=Math.max(1,values.length-1);
    const x=i=>padding.left+i/count*(width-padding.left-padding.right);
    const y=v=>height-padding.bottom-clamp(v,0,180)/180*(height-padding.top-padding.bottom);
    let measured='',estimated='';
    for(let i=1;i<values.length;i++){
      if(values[i-1]===null||values[i]===null)continue;
      const x0=x(i-1),x1=x(i),y0=y(values[i-1]),y1=y(values[i]);
      const prev=values[Math.max(0,i-2)],next=values[Math.min(values.length-1,i+1)];
      const slope0=prev===null?y1-y0:(y1-y(prev))/2,slope1=next===null?y1-y0:(y(next)-y0)/2;
      const low=Math.min(y0,y1),high=Math.max(y0,y1);
      const segment=`M${x0.toFixed(2)},${y0.toFixed(2)}C${(x0+(x1-x0)/3).toFixed(2)},${clamp(y0+slope0/3,low,high).toFixed(2)} ${(x1-(x1-x0)/3).toFixed(2)},${clamp(y1-slope1/3,low,high).toFixed(2)} ${x1.toFixed(2)},${y1.toFixed(2)}`;
      if(series.inferred[i-1]||series.inferred[i])estimated+=segment;else measured+=segment;
    }
    return {measured,estimated,x,y};
  }
  function quantile(values,fraction){if(!values.length)return null;const sorted=[...values].sort((a,b)=>a-b),p=(sorted.length-1)*fraction,i=Math.floor(p);return sorted[i]+(sorted[Math.ceil(p)]-sorted[i])*(p-i)}
  function stats(records){
    const swings=records.filter(r=>['S','F','H'].includes(r.result)),normal=records.filter(r=>['B','T','S','F','H'].includes(r.result));
    const located=records.filter(r=>[r.plateX,r.plateZ,r.zoneTop,r.zoneBottom].every(finite)&&r.zoneTop>r.zoneBottom),fastballs=records.filter(r=>r.pitch==='직구'&&finite(r.speed));
    const ratio=(a,b)=>b?100*a/b:null;
    return {pitches:records.length,swings:swings.length,whiffs:swings.filter(r=>r.result==='S').length,
      whiffRate:ratio(swings.filter(r=>r.result==='S').length,swings.length),cswRate:ratio(normal.filter(r=>['S','T'].includes(r.result)).length,normal.length),
      zoneRate:ratio(located.filter(r=>Math.abs(r.plateX)<=17/24&&r.plateZ>=r.zoneBottom&&r.plateZ<=r.zoneTop).length,located.length),
      fastballVelocity:fastballs.length?fastballs.reduce((s,r)=>s+r.speed,0)/fastballs.length:null};
  }
  return {prepareSeries,paths,quantile,stats};
})();
