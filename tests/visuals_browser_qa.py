"""Real interaction, data, deep-link and mobile checks for the first Visual."""
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

BASE=(sys.argv[1]if len(sys.argv)>1 else'http://127.0.0.1:50100').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-visuals-qa');OUT.mkdir(exist_ok=True)

with sync_playwright()as p:
    browser=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1100})
    page=context.new_page()
    errors,failed,requests=[],[],[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failed.append(r.url)if r.status>=400 and r.url.startswith(BASE)else None)
    page.on('request',lambda r:requests.append(r.url))
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page.goto(BASE+'/',wait_until='networkidle',timeout=60000)
    assert page.locator('#home').is_visible()
    assert page.locator('#visuals-gallery').is_hidden()
    assert page.locator('#visuals-affinity').is_hidden()
    assert not any('/data/visuals/'in u for u in requests)
    assert page.locator('.main-nav a').count()==4
    page.locator('.main-nav a[href="#visuals"]').click()
    page.wait_for_selector('.visuals-card')
    assert page.locator('#home').is_hidden()
    assert page.locator('.data-navigation').is_hidden()
    assert page.locator('.main-nav a[href="#visuals"]').get_attribute('aria-current')=='page'
    page.screenshot(path=str(OUT/'gallery-desktop.png'),full_page=True)
    page.locator('.visuals-card').click()
    page.wait_for_selector('#affinity-bars .affinity-outcome')
    assert page.locator('#visuals-affinity').is_visible()
    assert page.locator('#visuals-gallery').is_hidden()
    assert page.locator('#affinity-bars .affinity-outcome').count()==5
    assert page.locator('#affinity-bars .affinity-bar-row').count()==10
    assert page.locator('.affinity-node').count()==12
    assert page.locator('#affinity-search').input_value()=='김도영 · 우타'
    assert '214,732' in page.locator('.affinity-scope').inner_text()
    assert page.locator('#affinity-ranking button').count()==19
    assert page.locator('.affinity-bar-row.is-comparison .affinity-bar-track i').first.evaluate('(el)=>getComputedStyle(el).backgroundImage')!='none'
    assert page.locator('.affinity-bar-row:not(.is-comparison) .affinity-bar-track i').first.evaluate('(el)=>getComputedStyle(el).backgroundImage')=='none'
    # Displayed fractions come directly from counts with an all-pitch denominator.
    for i in range(5):
        expected=page.evaluate('(i)=>{const p=KBOVisuals.state.selected;return(100*p.counts[i]/p.pitches).toFixed(1)+"%"}',i)
        assert page.locator('.affinity-outcome').nth(i).locator('.affinity-bar-row strong').first.inner_text()==expected
    # Keyboard-select a comparison, then move it to the center.
    node=page.locator('.affinity-node').nth(2);key=node.get_attribute('data-affinity-compare')
    node.focus();page.keyboard.press('Enter')
    assert page.locator('.affinity-node[aria-pressed=true]').get_attribute('data-affinity-compare')==key
    assert page.evaluate('KBOVisuals.state.compare')==key
    before=page.url
    page.locator('#affinity-recenter').click()
    assert page.evaluate('KBOVisuals.state.selected.key')==key
    page.go_back();page.wait_for_function('(key)=>KBOVisuals.state.compare===key',arg=key)
    assert page.url==before
    # Search is a keyboard combobox, including no matches, Escape and both arrows.
    search=page.locator('#affinity-search');search.fill('선수없음')
    assert '검색 결과가 없습니다' in page.locator('#affinity-options').inner_text()
    page.keyboard.press('Escape')
    assert search.get_attribute('aria-expanded')=='false'
    search.fill('강백호');page.keyboard.press('ArrowDown');page.keyboard.press('Enter')
    assert page.evaluate('KBOVisuals.state.selected.key')=='68050-L'
    assert page.locator('#affinity-search').input_value()=='강백호 · 좌타'
    page.locator('#affinity-search').click();page.keyboard.press('ArrowUp')
    assert page.locator('#affinity-search').get_attribute('aria-activedescendant')==f'affinity-option-{page.locator("#affinity-options [role=option]").count()-1}'
    page.keyboard.press('Escape')
    # Slider uses precise, unrounded scores; zero matches leaves no stale player.
    slider=page.locator('#affinity-threshold');slider.fill('100')
    assert page.locator('#affinity-ranking button').count()==0
    assert page.locator('.affinity-node').count()==0
    assert page.locator('#affinity-recenter').is_hidden()
    assert page.locator('#affinity-bars .affinity-bar-row').count()==5
    slider.fill('0')
    assert page.locator('#affinity-ranking button').count()==138
    slider.focus();page.keyboard.press('End')
    assert slider.input_value()=='100'
    slider.fill('95')
    assert page.locator('#affinity-ranking button').count()>0
    # Batting sides of switch hitters remain separate; pitcher group is not mixed.
    page.locator('#affinity-search').fill('로하스')
    assert page.locator('#affinity-options [role=option]').count()==2
    page.locator('#affinity-options [role=option]').last.click()
    assert page.evaluate('KBOVisuals.state.selected.side')=='R'
    page.locator('[data-affinity-role=pitchers]').click()
    assert page.locator('[data-affinity-role=pitchers]').get_attribute('aria-pressed')=='true'
    assert page.evaluate('KBOVisuals.state.current.role')=='pitchers'
    assert '154명' in page.locator('.affinity-cohort').inner_text()
    assert page.locator('#affinity-search').input_value()=='네일'
    assert page.evaluate('document.activeElement.dataset.affinityRole')=='pitchers'
    page.locator('#affinity-search').fill('이승현')
    assert page.locator('#affinity-options [role=option]').count()==2
    assert '51454'in page.locator('#affinity-options').inner_text()
    assert '60146'in page.locator('#affinity-options').inner_text()
    page.keyboard.press('Escape')
    # Compare and threshold state survives refresh, language changes and sharing.
    page.locator('#affinity-threshold').fill('93.2')
    page.locator('#affinity-ranking button').nth(3).click()
    shared=page.url
    expected=page.evaluate('JSON.stringify(KBOVisuals.state.current)')
    page.reload(wait_until='networkidle');page.wait_for_selector('#affinity-map svg')
    assert page.evaluate('JSON.stringify(KBOVisuals.state.current)')==expected
    page.locator('#lang-toggle').click()
    assert page.locator('.affinity-heading h1').inner_text()=='Player similarity'
    assert page.url==shared
    assert page.locator('.affinity-outcome h3').first.inner_text()=='Ball'
    page.locator('#lang-toggle').click()
    context.grant_permissions(['clipboard-read','clipboard-write'])
    page.locator('#affinity-share').click()
    assert page.evaluate('navigator.clipboard.readText()')==shared
    page.screenshot(path=str(OUT/'affinity-desktop.png'),full_page=True)
    # Malformed links are canonicalized within the same group.
    page.goto(BASE+'/?db_player=65320#visuals/affinity?players=pitchers&player=missing&s=NaN&compare=68050-L',wait_until='networkidle')
    page.wait_for_selector('#affinity-map svg')
    assert page.evaluate('KBOVisuals.state.current.role')=='pitchers'
    assert page.evaluate('KBOVisuals.state.current.threshold')==95
    assert '?db_player'not in page.url
    assert page.evaluate('KBOVisuals.state.selected.name')=='네일'
    # Small layouts keep legible, responsive SVGs, real touch targets, and no tables.
    for width in [768,390,320]:
        page.set_viewport_size({'width':width,'height':844})
        page.goto(BASE+'/#visuals/affinity?players=hitters&player=68050-L&s=95',wait_until='networkidle')
        page.wait_for_selector('#affinity-map svg');page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
        assert page.locator('#visuals-affinity table').count()==0
        assert page.locator('.affinity-node').count()==(6 if width<=620 else 12)
        for selector in ['.affinity-heading p','.affinity-scope','.affinity-bar-name','.affinity-method']:
            assert page.locator(selector).first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)')>=17
        for selector in ['#affinity-search','#affinity-share','#affinity-threshold','#affinity-ranking button']:
            assert page.locator(selector).first.bounding_box()['height']>=44
        if width<=620:
            # SVG labels use a responsive viewBox rather than shrinking a desktop chart.
            svg=page.locator('#affinity-map svg')
            rendered=svg.bounding_box()['width']
            logical=float(svg.get_attribute('viewBox').split()[2])
            assert rendered/logical>=.99
        page.screenshot(path=str(OUT/f'affinity-{width}.png'),full_page=True)
    # Mobile menu, root route, and existing advanced/player links still resolve.
    page.goto(BASE+'/#visuals',wait_until='networkidle')
    page.locator('#menu-btn').click();page.locator('.main-nav a[href="#home"]').click()
    assert page.locator('#home').is_visible()
    assert page.locator('#visuals-gallery').is_hidden()
    assert page.locator('#visuals-affinity').is_hidden()
    page.goto(BASE+'/?pitcher=65320&pm_view=mechanics&mechanics=lead-knee#analysis',wait_until='networkidle')
    page.wait_for_selector('#analysis-profile .report-tabs')
    assert page.locator('#player-analysis').is_visible()
    assert page.locator('#visuals-affinity').is_hidden()
    assert 'pm_view=mechanics'in page.url
    page.goto(BASE+'/?pitcher=65320#pitchers',wait_until='networkidle')
    page.wait_for_selector('#visitor-form[data-ready=true]')
    assert page.locator('#pitcher-profile').is_visible()
    assert page.locator('#visuals-affinity').is_hidden()
    assert not errors,errors
    assert not failed,failed
    # A failed Visuals request has a working retry, independent of the old views.
    retry=context.new_page()
    attempts=[]
    def fail_once(route):
        attempts.append(True)
        if len(attempts)==1:route.fulfill(status=503,body='temporary unavailable')
        else:route.continue_()
    retry.route('**/data/visuals/affinity.json',fail_once)
    retry.goto(BASE+'/#visuals/affinity',wait_until='networkidle')
    retry.wait_for_selector('#affinity-retry')
    retry.locator('#affinity-retry').click()
    retry.wait_for_selector('#affinity-map svg')
    assert len(attempts)==2
    retry.close()
    # Touch selection also works; the list allows scrolling without selecting on pointerdown.
    mobile=browser.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True)
    touch=mobile.new_page();touch.add_init_script("localStorage.setItem('kbo-lang','ko')")
    touch.goto(BASE+'/#visuals/affinity',wait_until='networkidle')
    touch.wait_for_selector('#affinity-search')
    touch.locator('#affinity-search').tap();touch.locator('#affinity-search').fill('강백호')
    touch.locator('#affinity-options [role=option]').tap()
    assert touch.evaluate('KBOVisuals.state.selected.key')=='68050-L'
    touch.locator('[data-affinity-role=pitchers]').tap()
    assert touch.evaluate('KBOVisuals.state.current.role')=='pitchers'
    mobile.close()
    print(json.dumps({'result':'passed','base':BASE,'shared':shared,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False))
    browser.close()
