/* A real linked pitch drives the overview demo, including pitch-type changes. */
window.HeroMotion = (() => {
  let current=null,ticket=0,frame=0,paused=window.matchMedia('(prefers-reduced-motion: reduce)').matches,last=0,visible=true;
  const cache=new Map(),skeleton=document.querySelector('.hero-skeleton'),button=document.querySelector('#hero-pause');
  if('IntersectionObserver' in window)new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;last=0},{threshold:.05}).observe(skeleton);
  const valid=p=>p&&p.length>=3&&p.slice(0,3).every(Number.isFinite)&&p[2]>=.3;
  function updateButton(){button.textContent=typeof tr==='function'?tr(paused?'playDemo':'pauseDemo'):(paused?'Play demo':'Pause demo');button.setAttribute('aria-pressed',String(paused))}
  button.addEventListener('click',()=>{paused=!paused;last=0;updateButton()});
  async function show(clip){
    updateButton();if(current?.clip.id===clip.id)return;
    const request=++ticket;
    try{
      const path=clip.poseRefined||clip.pose;
      if(!cache.has(path)){const response=await fetch(url(path));if(!response.ok)throw Error('pose');cache.set(path,await response.json())}
      if(request!==ticket)return;const pose=cache.get(path),points=pose.frames.flatMap(f=>f.keypoints.filter(valid));if(!points.length)throw Error('empty');
      const low=[Infinity,Infinity],high=[-Infinity,-Infinity];for(const p of points)for(let i=0;i<2;i++){low[i]=Math.min(low[i],p[i]);high[i]=Math.max(high[i],p[i])}
      current={clip,pose,center:low.map((v,i)=>(v+high[i])/2),scale:Math.min(250/Math.max(1,high[0]-low[0]),300/Math.max(1,high[1]-low[1]))};frame=0;last=0;


      skeleton.setAttribute('aria-label',`${clip.player} · ${clip.pitch} · ${clip.speed} km/h · ${clip.id}`);skeleton.dataset.clip=clip.id;draw();
    }catch(error){if(request===ticket){current=null;document.querySelector('#hero-bones').replaceChildren();document.querySelector('#hero-joints').replaceChildren();document.querySelector('#hero-frame').textContent='—'}}
  }
  function draw(){if(!current)return;const {pose,center,scale}=current,source=pose.frames[frame],map=p=>[160+(p[0]-center[0])*scale,195+(p[1]-center[1])*scale],mapped=source.keypoints.map(p=>valid(p)?map(p):null);
    document.querySelector('#hero-bones').innerHTML=pose.edges.filter(([a,b])=>mapped[a]&&mapped[b]).map(([a,b])=>`<path d="M${mapped[a].join(' ')}L${mapped[b].join(' ')}"/>`).join('');
    document.querySelector('#hero-joints').innerHTML=mapped.filter(Boolean).map(([x,y])=>`<circle cx="${x}" cy="${y}" r="4"/>`).join('');
    const points=[];for(let i=Math.max(0,frame-12);i<=frame;i++){const f=pose.frames[i],p=f.keypoints[10];if(valid(p))points.push(map(p).join(','))}document.querySelector('#hero-wrist-trail').setAttribute('points',points.join(' '));
    document.querySelector('#hero-frame').textContent=`F${String(frame).padStart(4,'0')}`;document.querySelector('#hero-time').textContent=`${Number(source.time).toFixed(2)} s`;document.querySelector('#hero-progress').style.width=`${100*(frame+1)/pose.frames.length}%`;
  }
  function tick(now){if(current&&!paused&&visible&&!document.hidden){if(!last)last=now;if(now-last>=1000/current.pose.fps){frame=(frame+Math.floor((now-last)*current.pose.fps/1000))%current.pose.frames.length;last=now;draw()}}else last=0;requestAnimationFrame(tick)}requestAnimationFrame(tick);
  return {show};
})();
