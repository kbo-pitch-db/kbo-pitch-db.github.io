/* Shared source access and accessible player search for seasonal visuals. */
window.SeasonVisuals=(()=>{
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function players(){
    const found=new Map();for(const y of SeasonRecords.years())for(const p of Object.values(SeasonRecords.info(y).players))if(!found.has(p.code))found.set(p.code,{code:p.code,name:p.name});
    return [...found.values()].sort((a,b)=>a.name.localeCompare(b.name,'ko')||a.code.localeCompare(b.code));
  }
  async function records(code){
    await SeasonRecords.load();const years=SeasonRecords.years().filter(y=>SeasonRecords.info(y).players[code]).sort();
    return Object.fromEntries(await Promise.all(years.map(async y=>[y,await SeasonRecords.records(y,code)])));
  }
  function picker({prefix,players,current,label,choose}){
    const input=document.getElementById(prefix+'-search'),list=document.getElementById(prefix+'-players');let matches=[],active=-1;
    function close(){list.hidden=true;active=-1;input.setAttribute('aria-expanded','false');input.removeAttribute('aria-activedescendant');}
    function show(query=''){
      matches=players.filter(p=>(p.name+' '+p.code).toLowerCase().includes(query.trim().toLowerCase()));active=-1;
      list.innerHTML=matches.map((p,i)=>`<li role="option" id="${prefix}-option-${i}" data-index="${i}" aria-selected="${p.code===current()}">${esc(label(p))}</li>`).join('')||`<li role="presentation">${document.documentElement.lang==='ko'?'검색 결과가 없습니다.':'No matching pitchers.'}</li>`;
      list.hidden=false;input.setAttribute('aria-expanded','true');input.removeAttribute('aria-activedescendant');
    }
    async function select(i){const p=matches[i];if(!p)return;close();await choose(p.code);document.getElementById(prefix+'-search')?.focus();}
    input.onfocus=()=>input.select();input.onclick=()=>show();input.oninput=()=>show(input.value);
    input.onblur=()=>setTimeout(()=>{if(!list.contains(document.activeElement)){close();input.value=label(players.find(p=>p.code===current()));}},0);
    list.tabIndex=-1;list.onpointerdown=e=>{if(e.pointerType==='mouse')e.preventDefault();else list.focus({preventScroll:true});};list.onclick=e=>{const li=e.target.closest('[data-index]');if(li)select(Number(li.dataset.index));};list.onblur=()=>close();
    input.onkeydown=e=>{
      if(e.key==='ArrowDown'||e.key==='ArrowUp'){
        e.preventDefault();if(list.hidden)show();if(!matches.length)return;
        active=active<0?(e.key==='ArrowDown'?0:matches.length-1):(active+(e.key==='ArrowDown'?1:-1)+matches.length)%matches.length;
        list.querySelectorAll('[data-index]').forEach((el,i)=>el.classList.toggle('is-active',i===active));input.setAttribute('aria-activedescendant',prefix+'-option-'+active);list.children[active].scrollIntoView({block:'nearest'});
      }else if(e.key==='Enter'&&!list.hidden&&matches.length){e.preventDefault();select(Math.max(0,active));}
      else if(e.key==='Escape'){close();input.value=label(players.find(p=>p.code===current()));}
    };
  }
  return {players,records,picker};
})();
