"""Real browser: audited phases, condition changes, evidence and share restore."""
import json
import sys
from playwright.sync_api import sync_playwright

base=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8789').rstrip('/')
chrome=sys.argv[2] if len(sys.argv)>2 else '/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=chrome,headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1100})
    context.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page=context.new_page();errors=[];failed=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('response',lambda response:failed.append(response.url) if response.status>=400 and base in response.url else None)
    page.goto(base+'/?pitcher=65320&mechanics=lead-knee#pitchers',wait_until='networkidle')
    page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('#report-tab-mechanics').get_attribute('aria-selected')=='true'
    assert page.locator('#single-pitch-inspector').get_attribute('open') is None
    assert page.locator('.knee-iqr').count()==1
    assert page.locator('#knee-distribution [data-pitch-point]').count()==3
    assert '3 투구' in page.locator('.knee-scope').inner_text()
    assert page.locator('#knee-pose').get_attribute('data-highlighted-bones')=='2'
    page.locator('#knee-phase').fill('63')
    assert abs(float(page.locator('#knee-pose').get_attribute('data-phase'))-63)<.1
    assert '63%' in page.locator('.knee-cursor-value').text_content()
    assert int(page.locator('#knee-pose').get_attribute('data-frame'))==95
    page.locator('[data-knee-event="maximum"]').click()
    assert 0<float(page.locator('#knee-pose').get_attribute('data-phase'))<100
    page.locator('#knee-play').click();page.wait_for_timeout(100)
    assert float(page.locator('#knee-pose').get_attribute('data-phase'))>0
    page.wait_for_timeout(500)
    assert page.locator('#knee-phase-value').inner_text()=='100%'
    page.locator('[data-knee-view="compare"]').click();page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('.knee-iqr').count()==2
    assert page.locator('#knee-distribution [data-pitch-point]').count()==6
    slider=page.locator('#knee-distribution [data-pitch-point="0d84991ce9f6_c0038"]')
    slider.focus();page.keyboard.press('Enter');page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('#knee-selected-clip').input_value()=='0d84991ce9f6_c0038'
    assert page.locator('#knee-selected-clip').input_value() in page.url
    assert page.locator('#knee-source-links a').count()==2
    page.locator('#knee-smoothing').select_option('5');page.locator('#knee-phase').fill('41')
    share=page.url
    page.goto(share,wait_until='networkidle');page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('#knee-smoothing').input_value()=='5'
    assert page.locator('#knee-pitch').input_value()==''
    assert page.locator('#knee-selected-clip').input_value()=='0d84991ce9f6_c0038'
    assert abs(float(page.locator('#knee-pose').get_attribute('data-phase'))-41)<.1
    page.locator('#knee-quality').select_option('strict');page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('.knee-iqr').count()==0
    assert '1개만 남아' in page.locator('.knee-observation').inner_text()
    page.locator('#knee-pitch').select_option('슬라이더')
    assert page.locator('#knee-selected-clip').is_disabled()
    assert '0개만 남아' in page.locator('.knee-observation').inner_text()
    page.locator('#knee-quality').select_option('exploratory');page.wait_for_selector('#knee-pose[data-frame]')
    page.locator('[data-knee-view="speed"]').click();page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('.knee-speed-chart [data-pitch-point]').count()==3
    page.locator('.knee-speed-chart [data-pitch-point]').last.click();page.wait_for_selector('#knee-pose[data-frame]')
    assert page.locator('#knee-selected-clip').input_value()=='0d84991ce9f6_c0002'
    page.locator('.knee-methods summary').click()
    assert page.locator('.knee-methods tbody tr').count()==3
    assert '31' in page.locator('.knee-methods ul').inner_text()
    assert 'modelInspection' in page.locator('.knee-methods').inner_text()
    page.locator('[data-knee-view="compare"]').click();page.wait_for_selector('#knee-pose[data-frame]')
    page.locator('.knee-evidence-layout').scroll_into_view_if_needed()
    page.screenshot(path='/tmp/kbo-knee-charts.png')
    for width in [1440,1280,1100,1024,800,768,600,580,390,320]:
        page.set_viewport_size({'width':width,'height':1050});page.wait_for_timeout(50)
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,('overflow',width)
        # SVG labels must retain their actual rendered size, not just CSS px
        # before a whole-chart mobile scale transform.
        rendered=page.locator('.knee-wave-chart text').first.evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)*el.getScreenCTM().a')
        assert rendered>=14,('chart label size',width,rendered)
        if width<=1100:
            assert page.locator('.knee-selected').bounding_box()['y']>page.locator('.knee-chart-block').bounding_box()['y']
    page.set_viewport_size({'width':390,'height':844});page.locator('#knee-report-title').scroll_into_view_if_needed()
    page.screenshot(path='/tmp/kbo-knee-mobile.png')
    page.locator('#lang-toggle').click();page.wait_for_selector('#knee-pose[data-frame]')
    assert 'How does the lead leg' in page.locator('#knee-report-title').inner_text()
    assert page.locator('.knee-iqr').count()==2
    page.locator('#report-tab-overview').click()
    assert 'mechanics=' not in page.url
    page.locator('#report-tab-mechanics').click();page.wait_for_selector('#knee-pose[data-frame]')
    chosen=page.locator('#knee-selected-clip').input_value()
    page.locator('#knee-open-inspector').click();page.wait_for_function('!!app.poses[3]')
    assert page.evaluate('app.selected.id')==chosen
    assert page.locator('#explore').is_visible()
    page.goto(base+'/?pitcher=50030#pitchers',wait_until='networkidle');page.wait_for_selector('#report-tab-mechanics')
    page.locator('#report-tab-mechanics').click()
    assert page.locator('#open-knee-pilot').count()==1
    page.locator('#open-knee-pilot').click();page.wait_for_selector('#knee-report-title')
    assert page.locator('#report-tab-mechanics').get_attribute('aria-selected')=='true'
    assert not errors,errors
    assert not failed,failed
    print(json.dumps({'ok':True,'checks':['event alignment','unique pitch distributions','source pose synchronization','phase playback','pitch comparison','strict/empty screen','processing changes','velocity evidence','URL restore','source opening','bilingual','10 viewport widths','pilot discoverability'],'pageErrors':errors,'failedResponses':failed},ensure_ascii=False))
    browser.close()
