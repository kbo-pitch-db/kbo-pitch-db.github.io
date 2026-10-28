"""Exercise the research portal in a real browser.

Requires Playwright Python and Chromium. Run against an existing static server:
  python3 tests/portal_browser_qa.py http://127.0.0.1:8789 /path/to/chrome
"""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

base = (sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8789').rstrip('/')
executable = sys.argv[2] if len(sys.argv)>2 else None
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440,'height':1000})
    errors, failed = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: failed.append(response.url) if response.status>=400 and base in response.url else None)
    page.goto(base+'/', wait_until='networkidle')
    page.wait_for_function('PitchPortal.state.summary?.playerCount===123')
    assert page.evaluate('app.selected===null')
    assert page.locator('.pitcher-card').count()==12
    assert '9,754' in page.locator('#overview-stats').inner_text()
    page.wait_for_function("!!document.querySelector('.hero-skeleton').dataset.clip")
    old_demo = page.locator('.hero-skeleton').get_attribute('data-clip')
    options=page.locator('#demo-pitch option').all()
    page.locator('#demo-pitch').select_option(options[-1].get_attribute('value'))
    page.wait_for_function('(old)=>document.querySelector(".hero-skeleton").dataset.clip!==old', arg=old_demo)
    page.locator('#hero-pause').click()
    assert page.locator('#hero-pause').get_attribute('aria-pressed')=='true'
    page.screenshot(path='/tmp/kbo-portal-home.png')
    page.locator('#pitcher-search').fill('65320')
    assert page.locator('.pitcher-card').count()==1
    page.locator('.pitcher-card').click()
    page.wait_for_selector('[data-scope="records"]')
    assert '38' in page.locator('.profile-stats').inner_text()
    assert page.locator('[data-play-clip]').count()>1
    page.locator('[data-scope="records"]').click()
    assert page.locator('.heatmap').count()==2
    assert page.locator('.location-chart .chart-point').count()>100
    # Overlapping observations remain individually reachable from keyboard and select.
    page.locator('.speed-chart .chart-point').first.focus()
    page.keyboard.press('Enter')
    assert '144.0' in page.locator('#record-evidence').inner_text()
    assert page.locator('#record-evidence a[href*="api-gw.sports.naver.com"]').count()==1
    page.locator('#observation-select').select_option('10')
    assert page.locator('#record-evidence .record-facts').count()==1
    page.locator('.heatmap').first.locator('button:not([disabled])').first.click()
    assert page.locator('#record-evidence .record-examples button').count()>0
    page.locator('#reset-chart').click()
    page.locator('.heatmap').nth(1).locator('button:not([disabled])').first.click()
    page.locator('#record-evidence .record-examples button').first.click()
    assert page.locator('#record-evidence .previous-record').count()==1
    page.locator('#reset-chart').click()
    page.locator('.zone-cell').nth(4).focus();page.keyboard.press('Enter')
    assert page.locator('#reset-chart').count()==1
    page.locator('#reset-chart').click()
    page.locator('#record-hand').select_option('L')
    assert page.evaluate("PitchPortal.state.hand==='L'")
    page.locator('#lang-toggle').click()
    assert page.locator('html').get_attribute('lang')=='ko'
    assert page.locator('#record-hand').input_value()=='L'
    assert page.locator('#pitcher-profile [data-scope="records"]').get_attribute('aria-pressed')=='true'
    assert '확보한 경기 기록' in page.locator('#pitcher-profile').inner_text()
    page.evaluate("selectClip('0e1f00c192b9_c0028',{scroll:false})")
    page.wait_for_function('!!app.poses[3]')
    assert page.locator('.angle-curve').count()==4
    page.locator('.angle-curve').first.focus();page.keyboard.press('End')
    assert page.evaluate('localFrame()===app.selected.frames-1')
    assert page.locator('.motion-canvas[data-dimension="3"]').get_attribute('data-frame')=='63'
    page.locator('#clip-compare').click()
    page.locator('#compare-b').select_option('3fdcd2dbaf58_c0014')
    page.locator('#load-compare').click()
    page.wait_for_selector('#compare-progress')
    page.locator('#compare-progress').fill('500')
    assert page.locator('#compare-canvas-0').get_attribute('data-frame')=='32'
    assert page.locator('#compare-canvas-1').get_attribute('data-frame')=='70'
    page.locator('#compare-metric').select_option('rightElbow')
    assert page.locator('#compare-curves path').count()==2
    page.locator('#compare-play').click();page.wait_for_timeout(300)
    assert int(page.locator('#compare-progress').input_value())>500
    page.locator('#compare-play').click()
    assert page.locator('#compare-workspace a[href*="youtube.com"]').count()==2
    widths=[1440,1024,800,768,580,390,320]
    for width in widths:
        page.set_viewport_size({'width':width,'height':950})
        page.wait_for_timeout(80)
        scroll=page.evaluate('document.documentElement.scrollWidth')
        assert scroll<=width+1, (width,scroll,page.evaluate('[...document.querySelectorAll("body *")].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&e.getBoundingClientRect().width>0).slice(0,8).map(e=>e.tagName+"."+e.className)'))
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate('window.scrollTo(0,0)');page.screenshot(path='/tmp/kbo-portal-mobile.png')
    page.locator('#menu-btn').click()
    assert page.locator('#menu-btn').get_attribute('aria-expanded')=='true'
    page.locator('.main-nav a[href="#pitchers"]').click()
    assert page.locator('#menu-btn').get_attribute('aria-expanded')=='false'
    # Saved language, direct profile, and direct clip work after a fresh load.
    page.goto(base+'/?pitcher=65320&clip=0e1f00c192b9_c0028#explore', wait_until='networkidle')
    page.wait_for_function('!!app.poses[3]')
    page.wait_for_selector('#pitcher-profile .profile-identity')
    assert page.locator('html').get_attribute('lang')=='ko'
    assert page.evaluate("app.selected.id==='0e1f00c192b9_c0028'")
    page.locator('#lang-toggle').click();page.wait_for_function('!!app.poses[3]')
    assert page.locator('html').get_attribute('lang')=='en'
    assert page.evaluate('(()=>{const e=document.querySelector(".metadata-grid");return e.scrollWidth<=e.clientWidth+1})()')
    page.locator('#detail-panel').screenshot(path='/tmp/kbo-portal-explorer.png')
    assert not errors,errors
    assert not failed,failed
    print(json.dumps({'language_persistence':True,'profile_search':True,'pitch_demo':True,'record_filters':True,'heatmaps_and_zone_selection':True,'pitch_evidence':True,'synchronized_2d_3d_curves':True,'two_clip_comparison':True,'responsive_widths':widths,'direct_links':True,'browser_errors':errors,'failed_local_assets':failed},indent=2))
    browser.close()
