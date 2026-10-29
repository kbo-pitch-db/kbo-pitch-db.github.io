"""Actual movement deltas, equal physical scales, playback and shared context."""
import json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=(sys.argv[1]if len(sys.argv)>1 else'http://127.0.0.1:50104').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-pitch-movement-qa');OUT.mkdir(exist_ok=True)
with sync_playwright()as p:
 b=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1440,'height':1100});page=ctx.new_page();errors,failed=[],[]
 page.add_init_script("localStorage.setItem('kbo-lang','ko')");page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append(r.url)if r.status>=400 and r.url.startswith(BASE)else None)
 page.goto(BASE+'/',wait_until='networkidle',timeout=60000);assert page.locator('#home').is_visible();assert page.locator('#visuals-movement').is_hidden()
 page.locator('.main-nav a[href="#visuals"]').click();page.wait_for_selector('.visuals-card');assert page.locator('.visuals-card').count()==4
 page.goto(BASE+'/#visuals/movement?player=76715&year=2026',wait_until='networkidle');page.wait_for_selector('[data-movement-mean]')
 assert not page.evaluate('PitchMovement.state.playing');assert page.locator('.data-navigation').is_hidden();assert '2025'in page.locator('#movement-compare-title').inner_text()
 assert page.evaluate('PitchMovement.state.model.years.map(y=>y.n)')==[2490,2267,1529]
 expected=page.evaluate('''async()=>{const a=await SeasonRecords.records(2025,'76715'),b=await SeasonRecords.records(2026,'76715');const mean=(rs,key)=>{const v=rs.filter(r=>r.pitch==='직구'&&r.flight.status==='derived_pts');return v.reduce((n,r)=>n+r.flight[key],0)/v.length};return {hb:mean(b,'hb_cm')-mean(a,'hb_cm'),ivb:mean(b,'ivb_cm')-mean(a,'ivb_cm')}}''')
 arrow=page.locator('[data-movement-arrow="4-Seam"]');assert abs(float(arrow.get_attribute('data-delta-hb'))-expected['hb'])<1e-9;assert abs(float(arrow.get_attribute('data-delta-ivb'))-expected['ivb'])<1e-9
 assert 'Δ +1.7'in page.locator('[data-movement-card="4-Seam"] [data-delta-hb]').inner_text()
 assert page.locator('[data-movement-arrow="Slider"]').count()==0;assert page.locator('[data-movement-card="Slider"] [data-delta-hb]').inner_text()=='—'
 fixed=page.evaluate('JSON.stringify(PitchMovement.state.geometry.domain)')
 assert page.evaluate('Math.abs((PitchMovement.state.geometry.x(1)-PitchMovement.state.geometry.x(0))+(PitchMovement.state.geometry.y(1)-PitchMovement.state.geometry.y(0)))<1e-9')
 page.screenshot(path=str(OUT/'overview.png'),full_page=True);page.wait_for_timeout(180)
 # A real comparison season, not a zero or a renormalized coordinate frame.
 page.locator('#movement-base').select_option('2024');assert '2024'in page.locator('#movement-compare-title').inner_text();assert 'base=2024'in page.url
 assert float(page.locator('[data-movement-arrow="4-Seam"]').get_attribute('data-delta-hb'))!=expected['hb']
 page.locator('[data-movement-type="4-Seam"]').click();assert page.locator('[data-movement-card]').count()==1
 zoom=page.evaluate('JSON.stringify(PitchMovement.state.geometry.domain)');assert zoom!=fixed
 assert page.evaluate('PitchMovement.state.geometry.domain.x[1]-PitchMovement.state.geometry.domain.x[0]')<120
 page.locator('#movement-season').fill('1');page.wait_for_timeout(150)
 assert page.evaluate('PitchMovement.state.transitioning');assert page.evaluate('Object.values(PitchMovement.state.display.weights).filter(v=>v>0).length')==2
 page.wait_for_function('!PitchMovement.state.transitioning');assert page.locator('#movement-year').inner_text()=='2025';assert page.evaluate('JSON.stringify(PitchMovement.state.geometry.domain)')==zoom
 assert page.evaluate('JSON.stringify(PitchMovement.state.display)===JSON.stringify(PitchMovementCore.snapshot(PitchMovement.state.model.years.find(y=>y.year===PitchMovement.state.year)))')
 # Mean markers have the same preview by keyboard; a click selects a pitch.
 mean=page.locator('[data-movement-mean="4-Seam"]');mean.focus();assert page.locator('#movement-tooltip').is_visible();assert '평균 구속'in page.locator('#movement-tooltip').inner_text();page.keyboard.press('Escape');assert page.locator('#movement-tooltip').is_hidden()
 page.locator('#movement-trail').uncheck();assert page.locator('[data-movement-history]').count()==0
 page.locator('#movement-trail').check();assert page.locator('[data-movement-history]').count()==1
 shared=page.url;page.reload(wait_until='networkidle');page.wait_for_selector('[data-movement-mean]');assert page.evaluate('PitchMovement.state.base')=='2024';assert page.evaluate('PitchMovement.state.pitch')=='4-Seam';assert not page.evaluate('PitchMovement.state.playing')
 # The velocity link keeps this exact player/year/pitch and stops animation.
 page.locator('#movement-velocity-link').click();page.wait_for_selector('[data-curve]');assert page.evaluate('PitchDistribution.state.player')=='76715';assert page.evaluate('PitchDistribution.state.year')=='2025';assert page.evaluate('PitchDistribution.state.pitch')=='4-Seam';assert not page.evaluate('PitchMovement.state.playing')
 page.go_back();page.wait_for_selector('[data-movement-mean]');assert page.url==shared
 ctx.grant_permissions(['clipboard-read','clipboard-write']);page.locator('#movement-share').click();assert page.evaluate('navigator.clipboard.readText()')==shared
 page.locator('#lang-toggle').click();page.wait_for_selector('[data-movement-mean]');assert page.locator('.movement-heading h1').inner_text()=='Pitch movement over the seasons';assert page.url==shared;page.locator('#lang-toggle').click();page.wait_for_selector('[data-movement-mean]')
 # Absent types cannot be plotted at (0,0) or get made-up changes.
 page.goto(BASE+'/#visuals/movement?player=76715&year=2025&pitch=Slider',wait_until='networkidle');page.wait_for_selector('#movement-empty');assert page.locator('#movement-empty').is_visible();assert page.locator('#movement-chart').evaluate('(e)=>getComputedStyle(e).visibility')=='hidden';assert page.locator('[data-movement-arrow]').count()==0
 page.locator('#movement-season').fill('2');page.wait_for_function('!PitchMovement.state.transitioning');assert page.locator('#movement-empty').is_hidden();assert page.locator('[data-movement-arrow]').count()==0
 assert '비교 시즌에는 기록이 없습니다'in page.locator('[data-movement-card="Slider"]').inner_text()
 # Autoplay advances actual seasons once, then stops; replay and leaving work.
 page.goto(BASE+'/#visuals/movement',wait_until='networkidle');page.wait_for_selector('[data-movement-mean]');assert page.evaluate('PitchMovement.state.playing')
 page.wait_for_function('PitchMovement.state.year==="2026"&&!PitchMovement.state.playing',timeout=12000)
 page.locator('#movement-play').click();assert page.evaluate('PitchMovement.state.playing');page.locator('#visuals-movement > .visuals-back').click();page.wait_for_selector('.visuals-card');assert not page.evaluate('PitchMovement.state.playing');assert page.evaluate('PitchMovement.state.frame')==0
 # Shared player search also handles homonyms and a single observed season.
 page.goto(BASE+'/#visuals/movement?year=2024',wait_until='networkidle');page.wait_for_selector('[data-movement-mean]');search=page.locator('#movement-search');search.fill('이승현');assert page.locator('#movement-players [role=option]').count()==2;assert '51454'in page.locator('#movement-players').inner_text();search.press('Escape')
 search.fill('김광현');search.press('ArrowDown');search.press('Enter');page.wait_for_function('PitchMovement.state.players.find(p=>p.code===PitchMovement.state.player).name==="김광현"');page.wait_for_selector('[data-movement-mean]');page.locator('#movement-play').click()if page.evaluate('PitchMovement.state.playing')else None
 single=page.evaluate('''()=>{const ys=SeasonRecords.years();return Object.keys(SeasonRecords.info(ys[0]).players).find(c=>ys.filter(y=>SeasonRecords.info(y).players[c]).length===1&&SeasonRecords.info(ys[0]).players[c].recordCount>10)}''')
 page.goto(BASE+f'/#visuals/movement?player={single}',wait_until='networkidle');page.wait_for_selector('[data-movement-mean]');assert page.locator('#movement-play').is_disabled();assert page.locator('#movement-season').is_disabled()
 for width in [768,390,320]:
  page.set_viewport_size({'width':width,'height':844});page.goto(BASE+'/#visuals/movement?player=76715&year=2026&pitch=4-Seam&base=2024',wait_until='networkidle');page.wait_for_selector('[data-movement-mean]')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  for control in ['#movement-search','#movement-play','#movement-base','#movement-season']:
   assert page.locator(control).bounding_box()['height']>=44
  assert page.evaluate('Math.abs((PitchMovement.state.geometry.x(1)-PitchMovement.state.geometry.x(0))+(PitchMovement.state.geometry.y(1)-PitchMovement.state.geometry.y(0)))<1e-9')
  page.locator('#movement-season').focus();page.keyboard.press('Home');page.wait_for_function('!PitchMovement.state.transitioning');assert page.locator('#movement-year').inner_text()=='2024';assert page.locator('[data-movement-arrow]').count()==0
  page.screenshot(path=str(OUT/f'mobile-{width}.png'),full_page=True);page.wait_for_timeout(180)
 assert not errors,errors;assert not failed,failed
 reduced=b.new_context(viewport={'width':1000,'height':900},reduced_motion='reduce');r=reduced.new_page();r.add_init_script("localStorage.setItem('kbo-lang','ko')");r.goto(BASE+'/#visuals/movement',wait_until='networkidle');r.wait_for_selector('[data-movement-mean]');assert not r.evaluate('PitchMovement.state.playing');r.locator('#movement-next').click();assert not r.evaluate('PitchMovement.state.transitioning');assert r.locator('#movement-year').inner_text()=='2025';reduced.close()
 retry=ctx.new_page();attempts=[]
 def fail_once(route):
  attempts.append(True)
  if len(attempts)==1:route.fulfill(status=503,body='temporary failure')
  else:route.continue_()
 retry.route('**/data/seasons/2025/bucket-11.json.gz*',fail_once);retry.goto(BASE+'/#visuals/movement?player=76715&year=2025',wait_until='networkidle');retry.wait_for_selector('#movement-retry');retry.locator('#movement-retry').click();retry.wait_for_selector('[data-movement-mean]');assert retry.locator('#movement-year').inner_text()=='2025';retry.close()
 mobile=b.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True);touch=mobile.new_page();touch.add_init_script("localStorage.setItem('kbo-lang','ko')");touch.goto(BASE+'/#visuals/movement?player=76715&year=2024',wait_until='networkidle');touch.wait_for_selector('[data-movement-mean]');touch.locator('[data-movement-year="2026"]').tap();touch.wait_for_function('!PitchMovement.state.transitioning');assert touch.locator('#movement-year').inner_text()=='2026';touch.locator('[data-movement-type="4-Seam"]').tap();assert touch.locator('[data-movement-card]').count()==1;mobile.close()
 print(json.dumps({'passed':True,'base':BASE,'pairedDeltas':True,'rawCloudCrossfade':True,'fixedIsotropicAxes':True,'missingIsNotZero':True,'autoplay':True,'singleSeason':True,'mobile':[768,390,320],'reducedMotion':True,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False));b.close()
