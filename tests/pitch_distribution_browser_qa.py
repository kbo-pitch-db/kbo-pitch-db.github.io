"""Real seasonal scrubbing, interpolated paths, replay, history and mobile controls."""
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

BASE=(sys.argv[1]if len(sys.argv)>1 else 'http://127.0.0.1:50103').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-pitch-distribution-qa');OUT.mkdir(exist_ok=True)
with sync_playwright()as p:
 b=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1440,'height':1100});page=ctx.new_page();errors,failed,requests=[],[],[]
 page.add_init_script("localStorage.setItem('kbo-lang','ko')");page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append(r.url)if r.status>=400 and r.url.startswith(BASE)else None);page.on('request',lambda r:requests.append(r.url))
 page.goto(BASE+'/',wait_until='networkidle',timeout=60000);assert page.locator('#home').is_visible();assert page.locator('#visuals-distribution').is_hidden()
 page.locator('.main-nav a[href="#visuals"]').click();page.wait_for_selector('.visuals-card');assert page.locator('.visuals-card').count()==4
 page.locator('.visuals-card[href$="#visuals/distribution"]').click();page.wait_for_selector('[data-curve]');assert page.locator('#visuals-distribution').is_visible();assert page.locator('.data-navigation').is_hidden()
 assert page.evaluate('PitchDistribution.state.player')=='76715';assert page.evaluate('PitchDistribution.state.playing')
 page.wait_for_function('PitchDistribution.state.year==="2026"&&!PitchDistribution.state.playing',timeout=12000)
 assert page.locator('#distribution-play').inner_text()=='↻ 다시 재생';assert page.locator('.distribution-traces [data-trail-year]').count()==2
 page.locator('#distribution-play').click();assert page.evaluate('PitchDistribution.state.playing');assert page.evaluate('PitchDistribution.state.year')=='2024'
 page.locator('#distribution-play').click();assert not page.evaluate('PitchDistribution.state.playing');assert not page.evaluate('PitchDistribution.state.transitioning')
 page.goto(BASE+'/#visuals/distribution?player=76715&year=2024',wait_until='networkidle');page.wait_for_selector('[data-curve]');assert not page.evaluate('PitchDistribution.state.playing')
 assert page.evaluate('PitchDistribution.state.model.years.map(y=>y.n)')==[2490,2267,1529]
 assert '2,490'in page.locator('#distribution-scope').inner_text()
 assert page.locator('#distribution-play').evaluate('(e)=>getComputedStyle(e).color')=='rgb(255, 255, 255)'
 fixed=page.evaluate('JSON.stringify([PitchDistribution.state.model.xDomain,PitchDistribution.state.model.yMax])');path=page.locator('[data-curve="all"]').get_attribute('d')
 page.locator('#distribution-next').click();page.wait_for_timeout(150)
 assert page.evaluate('PitchDistribution.state.transitioning');assert page.locator('[data-curve="all"]').get_attribute('d')!=path
 assert page.evaluate('PitchDistribution.state.display.all.some((v,i)=>Math.abs(v-PitchDistribution.state.target.all[i])>1e-8)')
 page.wait_for_function('!PitchDistribution.state.transitioning');assert page.locator('#distribution-year').inner_text()=='2025';assert '2,267'in page.locator('#distribution-scope').inner_text()
 assert page.evaluate('JSON.stringify([PitchDistribution.state.model.xDomain,PitchDistribution.state.model.yMax])')==fixed
 assert page.locator('.distribution-traces [data-trail-year]').count()==1
 # Back restores the manually selected season; slider input pauses playback.
 page.go_back();page.wait_for_function('PitchDistribution.state.year==="2024"');page.wait_for_selector('[data-curve]')
 page.locator('#distribution-season').fill('2');page.wait_for_function('!PitchDistribution.state.transitioning');assert page.evaluate('PitchDistribution.state.year')=='2026';assert not page.evaluate('PitchDistribution.state.playing');assert '2026.07.31'in page.locator('#distribution-scope').inner_text()
 assert 'year=2026'in page.url
 page.locator('#distribution-trail').uncheck();assert page.locator('[data-trail-year]').count()==0;assert 'trail=0'in page.url
 page.locator('#distribution-trail').check();assert page.locator('[data-trail-year]').count()==2
 # Each legend share has the same all-velocity denominator as the curves.
 for button in page.locator('[data-distribution-type]').all():
  key=button.get_attribute('data-distribution-type')
  expected=page.evaluate('''key=>{const y=PitchDistribution.state.model.years.find(y=>y.year===PitchDistribution.state.year);const g=y.groups.get(key);return key==='all'?'100%':g?(100*g.n/y.n).toFixed(1)+'%':'—'}''',key)
  assert button.locator('strong').inner_text()==expected
 page.locator('[data-distribution-type="4-Seam"]').hover();assert page.locator('[data-curve="4-Seam"]').get_attribute('class').find('is-emphasized')>=0
 page.locator('[data-distribution-type="4-Seam"]').click();assert 'pitch=4-Seam'in page.url
 page.mouse.move(10,10,steps=3);assert page.locator('[data-curve="4-Seam"]').get_attribute('class').find('is-emphasized')>=0
 shared=page.url;page.reload(wait_until='networkidle');page.wait_for_selector('[data-curve]');assert page.evaluate('PitchDistribution.state.pitch')=='4-Seam';assert page.evaluate('PitchDistribution.state.year')=='2026';assert not page.evaluate('PitchDistribution.state.playing')
 # Tooltip shows actual unsmoothed bin counts, not interpolated observations.
 plot=page.locator('#distribution-chart');plot.scroll_into_view_if_needed();box=plot.bounding_box();x=page.evaluate('PitchDistribution.state.geometry.x(142.5)');page.mouse.move(box['x']+x,box['y']+120,steps=3);page.wait_for_selector('#distribution-tooltip:not([hidden])')
 total=page.evaluate('PitchDistributionCore.at(PitchDistribution.state.model.years.find(y=>y.year==="2026"),"all",142)')
 assert '142–143'in page.locator('#distribution-tooltip').inner_text();assert str(total['count'])+'구'in page.locator('#distribution-tooltip').inner_text()
 page.screenshot(path=str(OUT/'desktop.png'),full_page=True);page.wait_for_timeout(180)
 # Language and link copy preserve a reproducible, paused endpoint.
 page.locator('#lang-toggle').click();page.wait_for_selector('[data-curve]');assert page.locator('.distribution-heading h1').inner_text()=='Pitch velocity over the seasons';assert page.url==shared
 page.locator('#lang-toggle').click();page.wait_for_selector('[data-curve]');ctx.grant_permissions(['clipboard-read','clipboard-write']);page.locator('#distribution-share').click();assert page.evaluate('navigator.clipboard.readText()')==shared
 # Keyboard search supports homonyms, exact IDs and a pitcher with one season.
 search=page.locator('#distribution-search');search.fill('이승현');assert page.locator('#distribution-players [role=option]').count()==2;assert '51454'in page.locator('#distribution-players').inner_text()and'60146'in page.locator('#distribution-players').inner_text();search.press('Escape')
 search.fill('김광현');search.press('ArrowDown');search.press('Enter');page.wait_for_function('PitchDistribution.state.players.find(p=>p.code===PitchDistribution.state.player).name==="김광현"');page.wait_for_selector('[data-curve]');page.locator('#distribution-play').click()if page.evaluate('PitchDistribution.state.playing')else None
 single=page.evaluate('''()=>{const years=SeasonRecords.years();return Object.keys(SeasonRecords.info(years[0]).players).find(c=>years.filter(y=>SeasonRecords.info(y).players[c]).length===1&&SeasonRecords.info(years[0]).players[c].recordCount>10)}''')
 assert single
 page.goto(BASE+f'/#visuals/distribution?player={single}',wait_until='networkidle');page.wait_for_selector('[data-curve]');assert page.evaluate('PitchDistribution.state.model.years.length')==1;assert page.locator('#distribution-play').is_disabled();assert page.locator('#distribution-season').is_disabled();assert page.locator('#distribution-single').inner_text()
 # Existing map and similarity still occupy their own routes; timers stop.
 page.goto(BASE+'/#visuals/distribution',wait_until='networkidle');page.wait_for_selector('[data-curve]');assert page.evaluate('PitchDistribution.state.playing')
 page.locator('.visuals-back').click();page.wait_for_selector('.visuals-card');assert not page.evaluate('PitchDistribution.state.playing');assert page.evaluate('PitchDistribution.state.frame')==0
 page.goto(BASE+'/#visuals/map?year=2024',wait_until='networkidle');page.wait_for_selector('.pitcher-map-node');assert page.locator('.pitcher-map-node').count()==154;assert page.locator('#visuals-distribution').is_hidden()
 # Mobile controls remain usable, without a horizontal chart or table scroll.
 for width in [768,390,320]:
  page.set_viewport_size({'width':width,'height':844});page.goto(BASE+'/#visuals/distribution?player=76715&year=2025',wait_until='networkidle');page.wait_for_selector('[data-curve]')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  for control in ['#distribution-search','#distribution-season','#distribution-play','#distribution-next','#distribution-trail']:
   assert page.locator(control if control!='#distribution-trail'else'.distribution-options label').bounding_box()['height']>=44
  page.locator('#distribution-season').focus();page.keyboard.press('End');page.wait_for_function('!PitchDistribution.state.transitioning');assert page.locator('#distribution-year').inner_text()=='2026'
  page.screenshot(path=str(OUT/f'mobile-{width}.png'),full_page=True);page.wait_for_timeout(180)
 assert not errors,errors;assert not failed,failed
 # Reduced motion removes automatic playback and interpolated transitions.
 reduced=b.new_context(viewport={'width':1000,'height':900},reduced_motion='reduce');r=reduced.new_page();r.add_init_script("localStorage.setItem('kbo-lang','ko')");r.goto(BASE+'/#visuals/distribution',wait_until='networkidle');r.wait_for_selector('[data-curve]');assert not r.evaluate('PitchDistribution.state.playing');r.locator('#distribution-next').click();assert not r.evaluate('PitchDistribution.state.transitioning');assert r.locator('#distribution-year').inner_text()=='2025';reduced.close()
 # A failed seasonal request can be retried; it is never called a missing year.
 retry=ctx.new_page();attempts=[]
 def fail_once(route):
  attempts.append(True)
  if len(attempts)==1:route.fulfill(status=503,body='temporary failure')
  else:route.continue_()
 retry.route('**/data/seasons/2025/bucket-11.json.gz*',fail_once)
 retry.goto(BASE+'/#visuals/distribution?player=76715&year=2025',wait_until='networkidle');retry.wait_for_selector('#distribution-retry');retry.locator('#distribution-retry').click();retry.wait_for_selector('[data-curve]');assert retry.locator('#distribution-year').inner_text()=='2025';retry.close()
 mobile=b.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True);touch=mobile.new_page();touch.add_init_script("localStorage.setItem('kbo-lang','ko')");touch.goto(BASE+'/#visuals/distribution?player=76715&year=2024',wait_until='networkidle');touch.wait_for_selector('[data-curve]');touch.locator('[data-distribution-year="2026"]').tap();touch.wait_for_function('!PitchDistribution.state.transitioning');assert touch.locator('#distribution-year').inner_text()=='2026';mobile.close()
 print(json.dumps({'passed':True,'base':BASE,'realSeasonCounts':[2490,2267,1529],'autoplay':True,'slider':True,'interpolation':True,'fixedAxes':True,'mobile':[768,390,320],'reducedMotion':True,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False));b.close()
