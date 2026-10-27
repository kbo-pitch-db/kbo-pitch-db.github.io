/* Interpolate world-space bone directions through the kinematic tree.
 * Joint-center interpolation would shorten a rotating limb between saved frames.
 */
export function interpolateAnatomy(a,b,t,parents){
  if(t<=0||!b)return a.map(p=>p?.slice(0,3)||null);
  if(t>=1)return b.map(p=>p?.slice(0,3)||null);
  const output=Array(a.length).fill(null);
  for(let j=0;j<a.length;j++){
    if(!a[j]||!b[j])continue;
    const parent=parents[j];
    if(parent<0){output[j]=a[j].map((v,k)=>v+(b[j][k]-v)*t);continue;}
    if(!output[parent]||!a[parent]||!b[parent])continue;
    let u=a[j].map((v,k)=>v-a[parent][k]),v=b[j].map((x,k)=>x-b[parent][k]);
    const la=Math.hypot(...u),lb=Math.hypot(...v);
    if(la<1e-8||lb<1e-8)continue;
    u=u.map(x=>x/la);v=v.map(x=>x/lb);
    const dot=Math.max(-1,Math.min(1,u.reduce((s,x,k)=>s+x*v[k],0)));
    let direction;
    if(dot>.9995){direction=u.map((x,k)=>x+(v[k]-x)*t);}
    else if(dot<-.9995){
      // A deterministic great circle avoids a collapsed limb at a reversal.
      const axis=Math.abs(u[0])<.8?[1,0,0]:[0,1,0],proj=u.reduce((s,x,k)=>s+x*axis[k],0);
      let tangent=axis.map((x,k)=>x-proj*u[k]);const length=Math.hypot(...tangent);tangent=tangent.map(x=>x/length);
      direction=u.map((x,k)=>x*Math.cos(Math.PI*t)+tangent[k]*Math.sin(Math.PI*t));
    }else{
      const angle=Math.acos(dot),denominator=Math.sin(angle);
      direction=u.map((x,k)=>(x*Math.sin((1-t)*angle)+v[k]*Math.sin(t*angle))/denominator);
    }
    const norm=Math.hypot(...direction),length=la+(lb-la)*t;
    output[j]=output[parent].map((x,k)=>x+length*direction[k]/norm);
  }
  return output;
}
