"""Whole-cohort map, accurate pair scores, navigation, touch and mobile layouts."""
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:50101').rstrip('/')
CHROME = '/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT = Path('/tmp/kbo-pitcher-map-qa'); OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    context = browser.new_context(viewport={'width':1440,'height':1100})
    page = context.new_page(); errors, failed, requests = [], [], []
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('response', lambda r: failed.append(r.url) if r.status>=400 and r.url.startswith(BASE) else None)
    page.on('request', lambda r: requests.append(r.url))
    page.goto(BASE+'/', wait_until='networkidle', timeout=60000)
    assert page.locator('#home').is_visible()
    assert not any('/data/visuals/' in u for u in requests)
    page.locator('.main-nav a[href="#visuals"]').click(); page.wait_for_selector('.visuals-card')
    assert page.locator('.visuals-card').count()==4
    page.locator('.visuals-card[href$="#visuals/map"]').click(); page.wait_for_selector('.pitcher-map-node')
    assert page.locator('#pitcher-map-season').input_value()=='2026'
    assert page.locator('.pitcher-map-node').count()==125
    assert page.locator('#visuals-affinity').is_hidden()
    assert page.locator('.data-navigation').is_hidden()
    assert page.locator('.main-nav a[href="#visuals"]').get_attribute('aria-current')=='page'
    assert '2026.08.04' in page.locator('.pitcher-map-controls').inner_text()
    page.screenshot(path=str(OUT/'overview.png'), full_page=True)
    original = page.evaluate('JSON.stringify(PitcherMap.state.positions)')
    # Search highlights, but never removes pitchers or changes coordinates.
    search = page.locator('#pitcher-map-search'); search.fill('소형준')
    assert page.locator('[data-map-search]').count()==1
    assert page.locator('.pitcher-map-node').count()==125
    page.locator('[data-map-search]').click()
    assert page.evaluate('PitcherMap.state.data.players.find(p=>p.key===PitcherMap.state.selected).name')=='소형준'
    assert page.evaluate('JSON.stringify(PitcherMap.state.positions)')==original
    expected = page.evaluate('AffinityCore.rank(PitcherMap.state.data.players,PitcherMap.state.data.players.find(p=>p.key===PitcherMap.state.selected)).slice(0,5).map(p=>({key:p.player.key,score:p.score.toFixed(1)}))')
    for item, row in zip(expected, page.locator('[data-map-neighbor]').all()):
        assert row.get_attribute('data-map-neighbor')==item['key']
        assert row.locator('strong').inner_text()==item['score']
    # Shared selection and actual full-distribution comparison are reproducible.
    shared=page.url; selected=page.evaluate('PitcherMap.state.selected')
    compare_href=page.locator('.pitcher-map-neighbors a').first.get_attribute('href')
    page.reload(wait_until='networkidle'); page.wait_for_selector('.pitcher-map-neighbors')
    assert page.evaluate('PitcherMap.state.selected')==selected
    page.locator('.pitcher-map-neighbors a').first.click(); page.wait_for_selector('#affinity-shapes svg')
    assert page.evaluate('KBOVisuals.state.selected.key')==selected
    assert page.evaluate('KBOVisuals.state.compare')==expected[0]['key']
    page.go_back(); page.wait_for_selector('.pitcher-map-neighbors'); assert page.url==shared
    # Keyboard navigation and actual pointer selection work on the SVG.
    active=page.locator(f'[data-map-player="{selected}"]'); active.focus(); page.keyboard.press('ArrowRight')
    next_key=page.evaluate('document.activeElement.dataset.mapPlayer'); assert next_key!=selected
    page.keyboard.press('Enter'); assert page.evaluate('PitcherMap.state.selected')==next_key
    point=page.locator('.pitcher-map-node').nth(15); point_key=point.get_attribute('data-map-player'); point.click(force=True)
    page.wait_for_function('(key)=>PitcherMap.state.selected===key',arg=point_key)
    # Zoom, drag and fit all; selection does not re-run or rotate the embedding.
    page.locator('#pitcher-map-plus').click(); assert page.evaluate('PitcherMap.state.zoom')>1
    plot=page.locator('#pitcher-map-plot'); box=plot.bounding_box()
    before=page.evaluate('PitcherMap.state.dx'); page.mouse.move(box['x']+100,box['y']+80); page.mouse.down(); page.mouse.move(box['x']+150,box['y']+110,steps=4); page.mouse.up()
    assert page.evaluate('PitcherMap.state.dx')!=before
    page.locator('#pitcher-map-reset').click(); assert page.evaluate('[PitcherMap.state.zoom,PitcherMap.state.dx,PitcherMap.state.dy]')==[1,0,0]
    # Enlarged rings remain clickable away from their centre, with a touch
    # target that grows with the visible marker rather than a fixed radius.
    for _ in range(4):page.locator('#pitcher-map-plus').click()
    target=page.evaluate('''()=>{const s=PitcherMap.state;return s.positions.map(p=>({...p,cx:p.x*s.zoom+s.dx,cy:p.y*s.zoom+s.dy})).find(p=>p.cx>40&&p.cx<s.width-40&&p.cy>40&&p.cy<s.height-40&&!s.positions.some(q=>q.key!==p.key&&Math.hypot(q.x*s.zoom+s.dx-p.cx-24,q.y*s.zoom+s.dy-p.cy)<=24))}''')
    assert target is not None
    box=page.locator('#pitcher-map-plot').bounding_box(); page.mouse.click(box['x']+target['cx']+24,box['y']+target['cy'])
    assert page.evaluate('PitcherMap.state.selected')==target['key']
    page.locator('#pitcher-map-reset').click()
    # All years use their own entire eligible cohort; homonyms remain distinct.
    for year, count in [('2024',154),('2025',146),('2026',125)]:
        page.locator('#pitcher-map-season').select_option(year); page.wait_for_function('(y)=>PitcherMap.state.data.season===Number(y)',arg=year)
        assert page.locator('.pitcher-map-node').count()==count
        assert len(set(page.locator('.pitcher-map-node').evaluate_all('(es)=>es.map(e=>e.dataset.mapPlayer)')))==count
        assert f'year={year}' in page.url
    page.locator('#pitcher-map-season').select_option('2024'); page.wait_for_function('PitcherMap.state.data.season===2024')
    page.locator('#pitcher-map-search').fill('이승현'); assert page.locator('[data-map-search]').count()==2
    assert '51454' in page.locator('#pitcher-map-results').inner_text() and '60146' in page.locator('#pitcher-map-results').inner_text()
    page.locator('#pitcher-map-search').fill('없는투수'); assert '검색 결과가 없습니다' in page.locator('#pitcher-map-results').inner_text()
    page.locator('#pitcher-map-search').press('Escape'); assert page.locator('.is-dim').count()==0
    page.locator('#pitcher-map-search').fill('네일'); page.locator('#pitcher-map-search').press('Enter'); assert '네일' in page.locator('.pitcher-map-player h2').inner_text()
    page.screenshot(path=str(OUT/'selected.png'),full_page=True)
    # Language and history retain identities; no data fetch is needed twice.
    shared=page.url; page.locator('#lang-toggle').click(); page.wait_for_selector('.pitcher-map-player')
    assert page.locator('.pitcher-map-heading h1').inner_text()=='Pitcher similarity map'
    assert page.url==shared
    page.locator('#lang-toggle').click(); context.grant_permissions(['clipboard-read','clipboard-write'])
    page.locator('#pitcher-map-share').click(); assert page.evaluate('navigator.clipboard.readText()')==shared
    # Real mobile viewports retain every point and keep detail after the map.
    for width in [768,390,320]:
        page.set_viewport_size({'width':width,'height':844}); page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
        assert page.locator('.pitcher-map-node').count()==154
        assert page.locator('.pitcher-map-sidebar').bounding_box()['y']>page.locator('.pitcher-map-chart').bounding_box()['y']
        for control in ['#pitcher-map-season','#pitcher-map-search','#pitcher-map-plus','#pitcher-map-minus','#pitcher-map-reset']:
            assert page.locator(control).bounding_box()['height']>=44
        page.screenshot(path=str(OUT/f'mobile-{width}.png'),full_page=True)
    assert not errors,errors; assert not failed,failed
    # Invalid/old parameters cannot silently change the cohort or other screens.
    page.goto(BASE+'/?db_season=2024#visuals/map?year=2099&player=does-not-exist',wait_until='networkidle'); page.wait_for_selector('.pitcher-map-node')
    assert page.locator('#pitcher-map-season').input_value()=='2026'; assert not page.evaluate('PitcherMap.state.selected'); assert '2099' not in page.url and 'does-not-exist' not in page.url
    # Recover from transient map failure without breaking existing similarity.
    retry=context.new_page(); attempts=[]
    def fail_once(route):
        attempts.append(True)
        if len(attempts)==1:route.fulfill(status=503,body='temporary failure')
        else:route.continue_()
    retry.route('**/data/visuals/pitcher-map.json*',fail_once)
    retry.goto(BASE+'/#visuals/map',wait_until='networkidle'); retry.wait_for_selector('#pitcher-map-retry'); retry.locator('#pitcher-map-retry').click(); retry.wait_for_selector('.pitcher-map-node'); retry.close()
    mobile=browser.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True)
    touch=mobile.new_page(); touch.add_init_script("localStorage.setItem('kbo-lang','ko')")
    touch.goto(BASE+'/#visuals/map?year=2024',wait_until='networkidle'); touch.wait_for_selector('.pitcher-map-node'); touch.locator('.pitcher-map-node').nth(12).tap(force=True); assert touch.evaluate('!!PitcherMap.state.selected')
    touch.locator('#pitcher-map-plot').scroll_into_view_if_needed(); box=touch.locator('#pitcher-map-plot').bounding_box(); cx=box['x']+box['width']/2; cy=box['y']+box['height']/2
    session=mobile.new_cdp_session(touch)
    def points(distance):return [{'x':cx-distance,'y':cy,'id':1},{'x':cx+distance,'y':cy,'id':2}]
    session.send('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':points(30)})
    session.send('Input.dispatchTouchEvent',{'type':'touchMove','touchPoints':points(70)})
    session.send('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
    assert touch.evaluate('PitcherMap.state.zoom')>1.5
    mobile.close()
    print(json.dumps({'passed':True,'base':BASE,'cohorts':[154,146,125],'mobile':[768,390,320],'scoresMatchExistingMetric':True,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False))
    browser.close()
