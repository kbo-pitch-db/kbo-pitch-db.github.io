"""Real-browser repertoire, distributions, sharing, removal and mobile checks."""
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

BASE=(sys.argv[1]if len(sys.argv)>1 else'http://127.0.0.1:50100').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-pitcher-visuals-qa');OUT.mkdir(exist_ok=True)

with sync_playwright()as p:
    browser=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1100})
    page=context.new_page();errors,failed,requests=[],[],[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failed.append(r.url)if r.status>=400 and r.url.startswith(BASE)else None)
    page.on('request',lambda r:requests.append(r.url))
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page.goto(BASE+'/',wait_until='networkidle',timeout=60000)
    assert page.locator('#home').is_visible()
    assert page.locator('#visuals-gallery').is_hidden()
    assert not any('/data/visuals/'in u for u in requests)
    page.locator('.main-nav a[href="#visuals"]').click();page.wait_for_selector('.visuals-card')
    assert page.locator('.data-navigation').is_hidden()
    assert '타자'not in page.locator('#visuals-gallery').inner_text()
    assert '비슷한 투수 찾기'in page.locator('.visuals-card').first.inner_text()
    page.screenshot(path=str(OUT/'gallery.png'),full_page=True)
    page.locator('.visuals-card').first.click();page.wait_for_selector('#affinity-shapes svg')
    assert page.locator('#affinity-season').input_value()=='2026'
    page.locator('#affinity-season').select_option('2024');page.wait_for_function('KBOVisuals.state.data.season===2024');page.wait_for_selector('#affinity-shapes svg')
    assert page.locator('.main-nav a[href="#visuals"]').get_attribute('aria-current')=='page'
    assert page.locator('#visuals-affinity').is_visible()
    assert page.locator('#affinity-search').input_value()=='네일'
    assert page.locator('[data-affinity-role]').count()==0
    assert page.locator('#affinity-shapes .affinity-shape').count()==3
    assert '215,525'in page.locator('.affinity-scope').inner_text()
    assert '타자'not in page.locator('#visuals-affinity').inner_text()
    assert '154명'in page.locator('.affinity-cohort').inner_text()
    assert page.evaluate("!('groups'in KBOVisuals.state.data)")
    # Rings and all usage bars have the same real type counts and denominator.
    for row in page.locator('.affinity-pitch-row').all():
        i=int(row.get_attribute('data-pitch-index'))
        expected=page.evaluate('(i)=>{const p=KBOVisuals.state.selected;return(100*p.counts[i]/p.pitches).toFixed(1)+"%"}',i)
        assert row.locator('.affinity-bar-row strong').first.inner_text()==expected
    assert page.locator('.affinity-bar-row.is-comparison .affinity-bar-track i').first.evaluate('(e)=>getComputedStyle(e).backgroundImage')!='none'
    # The sample count, median and curve change with the selected type.
    select=page.locator('#affinity-pitch')
    before=page.locator('.affinity-density-line').first.get_attribute('d')
    options=select.locator('option').all();first=select.input_value()
    different=next(o.get_attribute('value')for o in options if o.get_attribute('value')!=first)
    select.select_option(different)
    assert page.locator('.affinity-density-line').first.get_attribute('d')!=before
    for axis in range(3):
        expected=page.evaluate('(axis)=>KBOVisuals.state.selected.types[KBOVisuals.state.pitch]?.quantiles[axis][2].toFixed(1)',axis)
        if expected is not None:assert expected in page.locator('.affinity-density-values').nth(axis).inner_text()
    # One-sided pitch types must show missingness, never an invented second line.
    page.goto(BASE+'/#visuals/affinity?v=arsenal1&player=54640&s=0&compare=52206',wait_until='networkidle')
    page.wait_for_selector('#affinity-shapes svg')
    page.locator('#affinity-pitch').select_option('6')
    assert '해당 구종 기록 없음'in page.locator('#affinity-type-samples').inner_text()
    assert page.locator('#affinity-shapes .affinity-density-line').count()==3
    assert all(value.inner_text()=='—'for value in page.locator('.affinity-density-values p:first-child strong').all())
    # Keyboard map selection, comparison, recenter and Back preserve identities.
    node=page.locator('.affinity-node').nth(2);key=node.get_attribute('data-affinity-compare')
    node.focus();page.keyboard.press('Enter')
    assert page.evaluate('KBOVisuals.state.compare')==key
    old=page.url;page.locator('#affinity-recenter').click()
    assert page.evaluate('KBOVisuals.state.selected.key')==key
    page.go_back();page.wait_for_function('(k)=>KBOVisuals.state.compare===k',arg=key);assert page.url==old
    # Keyboard search and distinct IDs for homonyms.
    search=page.locator('#affinity-search');search.fill('투수없음')
    assert '검색 결과가 없습니다'in page.locator('#affinity-options').inner_text();page.keyboard.press('Escape')
    search.fill('김광현');page.keyboard.press('ArrowDown');page.keyboard.press('Enter');page.wait_for_function('KBOVisuals.state.selected.name==="김광현"')
    assert page.locator('#affinity-search').input_value()=='김광현'
    page.locator('#affinity-search').fill('이승현')
    assert page.locator('#affinity-options [role=option]').count()==2
    assert '51454'in page.locator('#affinity-options').inner_text()
    assert '60146'in page.locator('#affinity-options').inner_text();page.keyboard.press('Escape')
    # A strict threshold has no stale comparison. Zero includes every other pitcher.
    slider=page.locator('#affinity-threshold');slider.fill('100')
    assert page.locator('#affinity-ranking button').count()==0
    assert page.locator('#affinity-recenter').is_hidden()
    assert page.locator('#affinity-shapes .affinity-density-line').count()==3
    slider.fill('0');assert page.locator('#affinity-ranking button').count()==153
    slider.focus();page.keyboard.press('End');assert slider.input_value()=='100';slider.fill('30')
    page.locator('#affinity-ranking button').nth(1).click()
    shared=page.url;expected=page.evaluate('JSON.stringify(KBOVisuals.state.current)')
    page.reload(wait_until='networkidle');page.wait_for_selector('#affinity-shapes svg')
    assert page.evaluate('JSON.stringify(KBOVisuals.state.current)')==expected
    page.locator('#lang-toggle').click();assert page.locator('.affinity-heading h1').inner_text()=='Pitcher similarity'
    assert page.url==shared;assert 'Hitters'not in page.locator('#visuals-affinity').inner_text()
    page.locator('#lang-toggle').click();context.grant_permissions(['clipboard-read','clipboard-write'])
    page.locator('#affinity-share').click();assert page.evaluate('navigator.clipboard.readText()')==shared
    # Old pitch-outcome score thresholds are migrated, not reused on a new model.
    page.goto(BASE+'/#visuals/affinity?players=pitchers&player=54640&s=95',wait_until='networkidle')
    page.wait_for_selector('#affinity-shapes svg')
    assert page.evaluate('KBOVisuals.state.selected.key')=='54640'
    assert page.locator('#affinity-threshold').input_value()=='30'
    assert 'players='not in page.url
    page.goto(BASE+'/#visuals/affinity?players=hitters&player=68050-L&s=95',wait_until='networkidle')
    page.wait_for_selector('#affinity-shapes svg')
    assert page.locator('#affinity-search').input_value()=='네일'
    assert '68050'not in page.url
    page.screenshot(path=str(OUT/'desktop.png'),full_page=True)
    for width in [768,390,320]:
        page.set_viewport_size({'width':width,'height':844});page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
        assert page.locator('#visuals-affinity table').count()==0
        assert page.locator('.affinity-node').count()==min(6 if width<=620 else 12,page.locator('#affinity-ranking button').count())
        assert page.locator('#affinity-shapes svg').count()==3
        for selector in ['.affinity-heading p','.affinity-scope','.affinity-bar-name','.affinity-shapes-heading label','.affinity-density-values p']:
            assert page.locator(selector).first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)')>=17
        for selector in ['#affinity-season','#affinity-search','#affinity-share','#affinity-threshold','#affinity-pitch','#affinity-ranking button']:
            assert page.locator(selector).first.bounding_box()['height']>=44
        if width<1000:
            cells=page.locator('.affinity-shape').all()
            assert cells[1].bounding_box()['y']>cells[0].bounding_box()['y']
        page.screenshot(path=str(OUT/f'mobile-{width}.png'),full_page=True)
    # Existing visitor and advanced data links still open their own screens.
    page.goto(BASE+'/?pitcher=65320&pm_view=mechanics&mechanics=lead-knee#analysis',wait_until='networkidle')
    page.wait_for_selector('#analysis-profile .report-tabs');assert page.locator('#visuals-affinity').is_hidden()
    page.goto(BASE+'/?pitcher=65320#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]')
    assert page.locator('#pitcher-profile').is_visible();assert page.locator('#visuals-affinity').is_hidden()
    assert not errors,errors;assert not failed,failed
    # Retry after a failed lazy data request.
    retry=context.new_page();attempts=[]
    def fail_once(route):
        attempts.append(True)
        if len(attempts)==1:route.fulfill(status=503,body='temporary unavailable')
        else:route.continue_()
    retry.route('**/data/visuals/affinity.json*',fail_once)
    retry.goto(BASE+'/#visuals/affinity',wait_until='networkidle');retry.wait_for_selector('#affinity-retry')
    retry.locator('#affinity-retry').click();retry.wait_for_selector('#affinity-shapes svg');assert len(attempts)==2;retry.close()
    mobile=browser.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True)
    touch=mobile.new_page();touch.add_init_script("localStorage.setItem('kbo-lang','ko')")
    touch.goto(BASE+'/#visuals/affinity',wait_until='networkidle');touch.wait_for_selector('#affinity-search')
    touch.locator('#affinity-season').select_option('2024');touch.wait_for_function('KBOVisuals.state.data.season===2024');touch.wait_for_selector('#affinity-search')
    touch.locator('#affinity-search').tap();touch.locator('#affinity-search').fill('김광현')
    touch.locator('#affinity-options [role=option]').tap();touch.wait_for_function('KBOVisuals.state.selected.name==="김광현"');assert touch.locator('#affinity-search').input_value()=='김광현'
    assert touch.locator('#affinity-shapes svg').count()==3;mobile.close()
    print(json.dumps({'result':'passed','base':BASE,'shared':shared,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False))
    browser.close()
