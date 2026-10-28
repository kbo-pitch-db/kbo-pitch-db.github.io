"""Exercise the public player page and the preserved analysis links in Chromium."""
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:50092').rstrip('/')
CHROME = sys.argv[2] if len(sys.argv) > 2 else '/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT = Path('/tmp/kbo-visitor-qa')
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1100})
    errors, failed, requests = [], [], []
    page.on('pageerror', lambda err: errors.append({'message':str(err),'stack':err.stack,'url':page.url}))
    page.on('response', lambda r: failed.append(r.url) if r.status >= 400 and r.url.startswith(BASE) else None)
    page.on('request', lambda r: requests.append(r.url))
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page.goto(BASE + '/', wait_until='networkidle')
    assert page.locator('#home').is_visible()
    assert page.locator('.main-nav a').count() == 3
    assert not page.locator('.home-steps,.method-pipeline,.model-details').count()
    page.goto(BASE + '/?pitcher=65320#pitchers', wait_until='networkidle')
    page.wait_for_selector('#visitor-form[data-ready=true]')
    root = page.locator('#pitcher-profile')
    assert root.locator('.visitor-section').count() == 3
    assert root.locator('table,[role=tablist],details,[download]').count() == 0
    assert root.locator('select:visible').count() == 1
    assert root.locator('.visitor-mechanism').count() == 3
    assert root.locator('.motion-pitch-label').is_hidden()
    assert '구종 미상' not in root.locator('.visitor-delivery-meta').inner_text()
    assert root.locator('.visitor-delivery-meta').inner_text().endswith('km/h')
    assert root.locator('a[target=_blank]').get_attribute('href').startswith('https://www.youtube.com/')
    assert not any('/metrics/database.' in u for u in requests)
    for selector in ['.visitor-mechanism p','.visitor-delivery-meta']:
        assert root.locator(selector).first.evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)')>=17
    for selector in ['#visitor-play','#visitor-pitch','.visitor-card-foot button']:
        assert root.locator(selector).first.bounding_box()['height']>=44
    for text in ['GVHMR', 'MotionBERT', 'RTMPose', '감사', '이관', '모델', '계산 방법']:
        assert text not in root.inner_text(), text
    # Graphs are visible below the cards without opening the details dialog.
    page.wait_for_selector('#visitor-graphs[data-clip]')
    assert root.locator('.visitor-inline-graph').count() == 2
    assert root.locator('#visitor-graphs').evaluate('(el)=>!!(el.compareDocumentPosition(document.querySelector("#visitor-cards"))&Node.DOCUMENT_POSITION_PRECEDING)')
    slider = root.locator('#visitor-frame')
    maximum = int(slider.get_attribute('max'))
    def selected_frame(expected):
        assert root.locator('canvas').evaluate_all('(els)=>els.map(e=>Number(e.dataset.frame))') == [expected, expected]
        assert root.locator('.visitor-wave-cursor').evaluate_all('(els)=>els.map(e=>Number(e.dataset.frame))') == [expected, expected]
        assert slider.input_value() == str(expected)
        assert root.locator('#visitor-frame-value').inner_text() == f'{expected+1} / {maximum+1}'
    for frame in [maximum // 2, maximum, 0]:
        slider.fill(str(frame)); selected_frame(frame)
    slider.focus();page.keyboard.press('ArrowRight');selected_frame(1)
    page.keyboard.press('End');selected_frame(maximum)
    page.keyboard.press('Home');selected_frame(0)
    first_graph_clip = root.locator('#visitor-graphs').get_attribute('data-clip')
    # Both canvases advance together with the only playback button.
    page.locator('#visitor-play').click()
    page.wait_for_function("[...document.querySelectorAll('#visitor-form canvas')].every(c=>Number(c.dataset.frame)>2)")
    frames = root.locator('canvas').evaluate_all('(cs)=>cs.map(c=>Number(c.dataset.frame))')
    assert frames[0] == frames[1], frames
    page.locator('#visitor-play').click()
    assert page.locator('#visitor-play').get_attribute('aria-pressed') == 'false'
    selected_frame(int(slider.input_value()))
    # Scrubbing while playing pauses on the exact requested frame.
    page.locator('#visitor-play').click();slider.fill('3');selected_frame(3)
    assert page.locator('#visitor-play').get_attribute('aria-pressed') == 'false'

    # 3D has keyboard and touch rotation, without opening another screen.
    canvas = root.locator('canvas[data-dimension="3"]')
    before = root.locator('.motion-yaw').input_value()
    canvas.focus(); page.keyboard.press('ArrowRight')
    assert root.locator('.motion-yaw').input_value() != before
    canvas.dispatch_event('pointerdown', {'clientX': 100, 'clientY': 100, 'pointerId': 1})
    canvas.dispatch_event('pointermove', {'clientX': 130, 'clientY': 100, 'pointerId': 1})
    canvas.dispatch_event('pointerup', {'pointerId': 1})
    # Pitch selection highlights direct-labelled bars and changes the actual delivery.
    page.locator('#visitor-pitch').select_option('슬라이더')
    page.wait_for_function("document.querySelector('.visitor-delivery-meta')?.textContent.includes('슬라이더')")
    page.wait_for_selector('#visitor-form[data-ready=true]')
    page.wait_for_selector('#visitor-graphs[data-clip]')
    assert root.locator('#visitor-graphs').get_attribute('data-clip') != first_graph_clip
    maximum = int(slider.get_attribute('max'))
    slider.fill(str(maximum // 2));selected_frame(maximum // 2)
    assert root.locator('[data-pitch-row="슬라이더"]').get_attribute('aria-pressed') == 'true'
    assert root.locator('[data-pitch-row="직구"]').get_attribute('aria-pressed') == 'false'
    assert root.locator('[data-pitch-row="슬라이더"] strong').first.inner_text() == '32.6%'
    assert set(parse_qs(urlparse(page.url).query)) == {'pitcher', 'pitch'}
    separation = root.locator('[data-mechanism="thorax_pelvis_axial_fc"]')
    assert '16.5' in separation.inner_text() and '3구' in separation.inner_text()
    knee = root.locator('[data-mechanism="lead_knee_net_extension"]')
    assert knee.locator('.visitor-metric-value strong').count() == 0
    assert '보류' in knee.inner_text()
    # Only the selected card opens a detailed distribution, waveform and definition.
    separation.locator('button').click()
    page.wait_for_selector('#visitor-wave svg')
    assert page.locator('dialog[open]').count() == 1
    assert '중앙값' in page.locator('dialog').inner_text()
    assert page.locator('#visitor-distribution svg circle').count() == 3
    page.screenshot(path=str(OUT / 'detail-desktop.png'))
    page.keyboard.press('Escape')
    page.wait_for_selector('dialog',state='detached')
    assert separation.locator('button').evaluate('(el)=>el===document.activeElement')
    page.screenshot(path=str(OUT / 'player-desktop.png'), full_page=True)
    # Real PTS data supply both the movement map and fixed-axis VAA chart.
    assert root.locator('#visitor-flight .flight-panel').count() == 2
    assert root.locator('#visitor-flight .flight-point').count() == 10
    assert 'n=356' in root.locator('#visitor-flight').inner_text()
    assert root.locator('#visitor-flight [data-flight-pitch="슬라이더"]').first.get_attribute('aria-pressed') == 'true'
    page.locator('#visitor-pitch').select_option('커브')
    page.wait_for_function("document.querySelector('.visitor-delivery-meta')?.textContent.includes('커브')")
    # Existing advanced player links retain all parameters on a separate route.
    legacy = '/?pitcher=65320&pm_scope=linked&pm_view=aggregate&pm_pitch=슬라이더&pm_columns=lead_knee_net_extension,thorax_pelvis_axial_fc&pm_metric=lead_knee_net_extension&pm_x=speed&pm_y=thorax_pelvis_axial_fc&pm_plots=yes#pitchers'
    page.goto(BASE + legacy, wait_until='networkidle')
    page.wait_for_selector('#player-metrics-root .db-aggregate-table')
    assert urlparse(page.url).fragment == 'analysis'
    params = parse_qs(urlparse(page.url).query)
    assert params['pm_pitch'] == ['슬라이더']
    assert params['pm_columns'] == ['lead_knee_net_extension,thorax_pelvis_axial_fc']
    assert page.locator('#pitcher-profile').is_hidden()
    assert page.locator('#analysis-profile').is_visible()
    page.screenshot(path=str(OUT / 'legacy-analysis.png'), full_page=True)
    # The older event report is also preserved as analysis rather than the default.
    page.goto(BASE + '/?pitcher=65320&mechanics=lead-knee&mpitch=직구&mview=wave&mquality=exploratory&msmooth=3&mprogress=45#pitchers', wait_until='networkidle')
    page.wait_for_selector('#knee-phase')
    assert urlparse(page.url).fragment == 'analysis'
    assert page.locator('#knee-phase').input_value() == '45'
    # Bare player URLs never inherit the previous analysis context.
    page.goto(BASE + '/?pitcher=65320#pitchers', wait_until='networkidle')
    page.wait_for_selector('#visitor-form[data-ready=true]')
    page.locator('[data-visitor-explore]').click()
    page.wait_for_selector('#player-metrics-root .db-pitch-table')
    assert urlparse(page.url).fragment == 'analysis'
    page.go_back(wait_until='networkidle')
    page.wait_for_selector('#visitor-form[data-ready=true]')
    assert page.locator('#pitcher-profile table').count() == 0
    page.locator('.main-nav a[href="#home"]').click()
    page.locator('#lang-toggle').click();page.locator('#lang-toggle').click()
    assert page.locator('#home').is_visible()
    page.locator('.main-nav a[href="#pitchers"]').click()
    assert page.locator('#pitcher-grid').is_visible()
    assert not urlparse(page.url).query
    # Global DB links and exports are still accessible.
    page.goto(BASE + '/?db_scope=linked&db_player=65320&db_metric=lead_knee_net_extension&db_status=#database', wait_until='networkidle')
    page.wait_for_selector('#metrics-database-root table')
    assert page.locator('.data-navigation').is_visible()
    assert parse_qs(urlparse(page.url).query)['db_player'] == ['65320']
    # Real mobile layout: no table, no horizontal page scroll, stacked motion/cards.
    for width in [768, 390, 320]:
        page.set_viewport_size({'width': width, 'height': 844})
        page.goto(BASE + '/?pitcher=65320#pitchers', wait_until='networkidle')
        page.wait_for_selector('#visitor-form[data-ready=true]')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        if width < 620:
            boxes = root.locator('.visitor-motion-cell').evaluate_all('(els)=>els.map(e=>({x:e.offsetLeft,y:e.offsetTop}))')
            assert boxes[1]['y'] > boxes[0]['y'], boxes
        assert root.locator('table').count() == 0
        page.screenshot(path=str(OUT / f'player-{width}.png'), full_page=True)
    # A player without video keeps real record charts and honest mechanics states.
    page.goto(BASE + '/?pitcher=50030#pitchers', wait_until='networkidle')
    assert '수집된 동작 영상이 없습니다' in root.inner_text()
    assert root.locator('.visitor-pitch-row').count() == 3
    assert root.locator('.visitor-metric-value strong').count() == 0
    assert root.locator('#visitor-frame,.visitor-inline-graph').count() == 0
    # English uses the same three-section screen.
    page.locator('#lang-toggle').click()
    assert 'Featured delivery' in root.inner_text()
    assert 'No delivery has been collected' in root.inner_text()
    assert not errors, errors
    assert not failed, failed
    print(json.dumps({'base': BASE, 'passed': True, 'synchronizedFrames': frames, 'mobileWidths': [768, 390, 320], 'legacyLinks': ['pm_*', 'mechanics=lead-knee', 'db_*'], 'errors': errors, 'failed': failed, 'screenshots': str(OUT)}))
    browser.close()
