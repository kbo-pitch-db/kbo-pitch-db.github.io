"""Real-browser checks for the player report. Requires Playwright + Chromium.
Run against a static server: python3 tests/portal_browser_qa.py URL /path/to/chrome
"""
import json
import sys
from playwright.sync_api import sync_playwright

base=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8789').rstrip('/')
executable=sys.argv[2] if len(sys.argv)>2 else None
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050})
    errors,failed=[],[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('response',lambda response:failed.append(response.url) if response.status>=400 and base in response.url else None)
    page.goto(base+'/',wait_until='networkidle')
    page.wait_for_function('PitchPortal.state.summary?.playerCount===155')
    assert page.evaluate('app.selected===null')
    assert page.locator('#home-featured-players button').count()==4
    assert page.locator('#pitchers').is_hidden()
    assert page.locator('#explore').is_hidden()
    page.wait_for_function('!!document.querySelector(".hero-skeleton").dataset.clip')
    demo=page.locator('.hero-skeleton').get_attribute('data-clip')
    choices=page.locator('#demo-pitch option').all()
    page.locator('#demo-pitch').select_option(choices[-1].get_attribute('value'))
    page.wait_for_function('(before)=>document.querySelector(".hero-skeleton").dataset.clip!==before',arg=demo)
    page.locator('#hero-pause').click()
    assert page.locator('#hero-pause').get_attribute('aria-pressed')=='true'
    page.screenshot(path='/tmp/kbo-ux2-home.png')
    page.locator('.main-nav a[href="#pitchers"]').click()
    page.locator('#pitcher-search').fill('65320')
    assert page.locator('.pitcher-card').count()==1
    page.locator('.pitcher-card').click()
    page.wait_for_selector('.characteristic-radar')
    assert page.locator('#pitcher-grid').is_hidden()
    assert page.locator('.report-coverage>strong').inner_text()=='356'
    assert page.locator('.percentile-row').count()==5
    assert page.locator('.radar-player').count()==1
    assert page.locator('#report-tab-overview').get_attribute('aria-selected')=='true'
    page.locator('#report-tab-overview').focus();page.keyboard.press('ArrowRight')
    assert page.locator('#report-tab-arsenal').get_attribute('aria-selected')=='true'
    assert page.locator('.simple-arsenal-table').count()==1
    assert page.locator('.density-cell').count()==25
    assert page.locator('#report-linked').is_checked() is False
    assert '356 / 356' in page.locator('.pane-intro').inner_text()
    # The default density view is readable without understanding coordinate units.
    cell=page.locator('.density-cell').all()
    nonempty=next(c for c in cell if int(c.locator('span').inner_text().split()[0])>0)
    nonempty.click()
    page.locator('.matching-records button').first.click()
    assert page.locator('#profile-record-detail .human-record').count()==1
    assert page.locator('#profile-record-detail a[href*="api-gw.sports.naver.com"]').count()==1
    page.locator('[data-location-mode="points"]').click()
    page.locator('.arrival-point').first.focus();page.keyboard.press('Enter')
    assert page.locator('#profile-record-detail').is_visible()
    page.locator('#report-batter').select_option('L')
    page.locator('#lang-toggle').click()
    assert page.locator('html').get_attribute('lang')=='ko'
    assert page.locator('#report-tab-arsenal').get_attribute('aria-selected')=='true'
    assert page.locator('#report-batter').input_value()=='L'
    page.locator('#report-clear').click()
    assert '356 / 356' in page.locator('.pane-intro').inner_text()
    page.locator('[data-report-pitch="슬라이더"]').first.click()
    assert page.evaluate('PitchPortal.state.pitch')=='슬라이더'
    page.locator('#report-clear').click()
    advanced=page.locator('.report-advanced').last
    advanced.locator('summary').first.click()
    advanced.locator('.heatmap').first.locator('button:not([disabled])').first.click()
    assert page.locator('.matching-records button').count()>0
    page.locator('#report-tab-mechanics').click()
    page.wait_for_selector('.guided-angle-chart')
    assert page.locator('.dimension-3').get_attribute('aria-pressed')=='true'
    assert len(page.locator('.curve-observed').get_attribute('d'))>100
    assert page.locator('[data-sequence]').count()==3
    page.locator('#profile-play').click();page.wait_for_timeout(300)
    assert int(page.locator('#profile-progress').input_value())>0
    page.locator('#profile-play').click()
    page.locator('[data-snapshot="0.8"]').click()
    expected=int(page.locator('#profile-progress').input_value())
    assert int(page.locator('.guided-angle-chart').get_attribute('aria-valuenow'))==expected
    page.locator('.guided-angle-chart').focus();page.keyboard.press('End')
    maximum=int(page.locator('#profile-progress').get_attribute('max'))
    assert int(page.locator('.guided-angle-chart').get_attribute('aria-valuenow'))==maximum
    page.locator('.dimension-2').click();page.locator('.curve-raw').click()
    assert page.locator('.curve-raw').get_attribute('aria-pressed')=='true'
    page.locator('.guided-metric-select').select_option('rightKnee')
    page.locator('.dimension-3').click();page.locator('.curve-readable').click()
    page.screenshot(path='/tmp/kbo-ux2-form-ko.png',full_page=True)
    widths=[1440,1024,800,768,580,390,320]
    for width in widths:
        page.set_viewport_size({'width':width,'height':1000});page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,('form',width)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path='/tmp/kbo-ux2-mobile-form.png',full_page=True)
    page.locator('#report-tab-overview').click()
    assert page.locator('.percentile-row').count()==5
    page.screenshot(path='/tmp/kbo-ux2-mobile-overview.png',full_page=True)
    page.locator('#report-tab-arsenal').click()
    for width in [390,320]:
        page.set_viewport_size({'width':width,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,('arsenal',width)
    page.screenshot(path='/tmp/kbo-ux2-mobile-arsenal.png',full_page=True)
    page.locator('#report-tab-mechanics').click();page.wait_for_selector('.guided-angle-chart')
    current=page.locator('#reference-clip').input_value()
    page.locator('#profile-open-3d').click()
    page.wait_for_function('!!app.poses[3]')
    assert page.evaluate('app.selected.id')==current
    assert page.locator('#explore').is_visible()
    assert page.locator('#pitchers').is_hidden()
    page.wait_for_selector('#detail-panel .guided-angle-chart')
    page.locator('#detail-panel .guided-angle-chart').focus();page.keyboard.press('End')
    assert page.evaluate('localFrame()===app.selected.frames-1')
    page.locator('#clip-compare').click()
    page.locator('#compare-b').select_option('3fdcd2dbaf58_c0014')
    page.locator('#load-compare').click();page.wait_for_selector('#compare-progress')
    assert page.evaluate('PitchPortal.state.compare.modelAngles')
    page.locator('#compare-progress').fill('500')
    assert page.locator('#compare-canvas-1').get_attribute('data-frame')=='70'
    page.locator('#compare-metric').select_option('rightElbow')
    assert page.locator('#compare-curves path').count()==4
    assert page.evaluate('document.documentElement.scrollWidth')<=321
    # A player without any collected motion still has a complete pitch-record report.
    page.goto(base+'/?pitcher=50030#pitchers',wait_until='networkidle')
    page.wait_for_selector('.report-identity')
    assert page.locator('html').get_attribute('lang')=='ko'
    assert page.locator('.report-coverage>strong').inner_text()=='27'
    assert page.locator('.radar-player').count()==0
    page.locator('#report-tab-mechanics').click()
    assert page.locator('.empty-report-motion').count()==1
    page.locator('#motion-read-stats').click()
    assert '27 / 27' in page.locator('.pane-intro').inner_text()
    page.locator('.back-to-players').click()
    assert page.locator('#pitcher-grid').is_visible()
    page.locator('#pitcher-search').fill('50030')
    assert page.locator('.pitcher-card').count()==1
    page.locator('#pitcher-motion').select_option('motion')
    assert page.locator('.pitcher-card').count()==0
    # Drawer and language switch remain usable on a phone.
    page.locator('#menu-btn').click()
    assert page.locator('#menu-btn').get_attribute('aria-expanded')=='true'
    page.locator('.main-nav a[href="#home"]').click()
    assert page.locator('#menu-btn').get_attribute('aria-expanded')=='false'
    page.locator('#lang-toggle').click()
    assert page.locator('html').get_attribute('lang')=='en'
    assert not errors,errors
    assert not failed,failed
    print(json.dumps({'all_acquired_records':9754,'players_including_no_video':155,'report_tabs':True,'sample_relative_percentiles':True,'readable_velocity_and_density':True,'explained_mechanics_3d_and_2d':True,'synchronized_curve_and_motion':True,'no_video_report':True,'language_and_filter_persistence':True,'responsive_widths':widths,'browser_errors':errors,'failed_assets':failed},indent=2))
    browser.close()
