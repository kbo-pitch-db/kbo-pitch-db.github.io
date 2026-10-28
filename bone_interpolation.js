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

/* Interpolate an already rigid saved shoe without shrinking its triangle.
 * A marker stationary in both saved frames stays stationary. This never adds
 * a floor/contact clamp or changes any stored frame or root trajectory.
 */
export function interpolateRigidFeet(a,b,t){
  if(t<=0||!b)return a.map(p=>p?.slice(0,3)||null);
  if(t>=1)return b.map(p=>p?.slice(0,3)||null);
  const out=Array(6).fill(null),sub=(u,v)=>u.map((x,k)=>x-v[k]);
  const dot=(u,v)=>u.reduce((s,x,k)=>s+x*v[k],0);
  const cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const unit=v=>{const n=Math.hypot(...v);return n>1e-10?v.map(x=>x/n):null;};
  function basis(p){
    const x=unit(sub(p[1],p[0]));if(!x)return null;
    const z=unit(cross(x,sub(p[2],p[0])));return z?[x,cross(z,x),z]:null;
  }
  function quaternion(m){
    const q=[0,0,0,0],trace=m[0][0]+m[1][1]+m[2][2];
    if(trace>0){const s=2*Math.sqrt(1+trace);q[3]=s/4;q[0]=(m[2][1]-m[1][2])/s;q[1]=(m[0][2]-m[2][0])/s;q[2]=(m[1][0]-m[0][1])/s;}
    else{
      let i=0;for(let k=1;k<3;k++)if(m[k][k]>m[i][i])i=k;
      const j=(i+1)%3,k=(i+2)%3,s=2*Math.sqrt(Math.max(0,1+m[i][i]-m[j][j]-m[k][k]));
      q[i]=s/4;q[j]=(m[i][j]+m[j][i])/s;q[k]=(m[i][k]+m[k][i])/s;q[3]=(m[k][j]-m[j][k])/s;
    }
    const n=Math.hypot(...q),sign=q[3]<0?-1:1;return q.map(x=>x*sign/n);
  }
  for(let side=0;side<2;side++){
    const ids=[side,side+2,side+4];
    if(ids.some(j=>!a[j]||!b[j]))continue;
    const pa=ids.map(j=>a[j].slice(0,3)),pb=ids.map(j=>b[j].slice(0,3)),A=basis(pa),B=basis(pb);
    if(!A||!B){for(const j of ids)out[j]=a[j].slice(0,3).map((x,k)=>x+(b[j][k]-x)*t);continue;}
    const relative=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>B.reduce((s,column,k)=>s+column[i]*A[k][j],0)));
    const q=quaternion(relative),half=Math.acos(Math.max(-1,Math.min(1,q[3]))),sin=Math.sin(half);
    const axis=sin>1e-8?q.slice(0,3).map(x=>x/sin):[1,0,0],angle=2*half*t,c=Math.cos(angle),s=Math.sin(angle);
    let anchor=2;
    for(let j=0;j<3;j++)if(Math.hypot(...sub(pb[j],pa[j]))<1e-5&&Math.hypot(...sub(pb[j],pa[j]))<Math.hypot(...sub(pb[anchor],pa[anchor])))anchor=j;
    const origin=pa[anchor].map((x,k)=>x+(pb[anchor][k]-x)*t);
    for(let j=0;j<3;j++){
      const v=sub(pa[j],pa[anchor]),av=cross(axis,v),along=dot(axis,v);
      out[ids[j]]=origin.map((x,k)=>x+v[k]*c+av[k]*s+axis[k]*along*(1-c));
    }
  }
  return out;
}
