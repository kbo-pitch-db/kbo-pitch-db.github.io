"""Browser workflow: query → aggregate → pitch evidence, complete CSV, mobile."""
import csv
import io
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

base=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8789').rstrip('/')
chrome=sys.argv[2] if len(sys.argv)>2 else '/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=chrome,headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
    context.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page=context.new_page();errors=[];failures=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failures.append(r.url) if r.status>=400 and base in r.url else None)
    page.goto(base+'/',wait_until='networkidle');page.wait_for_selector('.db-pitch-table tbody tr')
    g=page.locator('#metrics-database-root')
    assert page.locator('.shell').get_attribute('data-route')=='database'
    assert page.locator('#home').is_hidden()
    assert '390' in g.locator('.db-table-actions').inner_text()
    assert g.locator('.db-pitch-table tbody tr').count()==20
    assert g.locator('.db-controls').get_attribute('open') is None
    metric='lead_knee_range_2d'
    g.locator(f'[data-db-sort="{metric}"]').click()
    values=[float(v) for v in g.locator(f'td[data-metric="{metric}"]').evaluate_all('(els)=>els.map(el=>el.dataset.value)')]
    assert values==sorted(values,reverse=True)
    g.locator(f'[data-db-sort="{metric}"]').click()
    values=[float(v) for v in g.locator(f'td[data-metric="{metric}"]').evaluate_all('(els)=>els.map(el=>el.dataset.value)')]
    assert values==sorted(values)
    g.locator('[data-db-page="next"]').click();assert '2 /' in g.locator('.db-pagination').inner_text()
    g.locator('[data-db-filter="scope"]').select_option('records')
    assert '9,754' in g.locator('.db-table-actions').inner_text()
    with page.expect_download() as info:g.locator('[data-db-csv]').click()
    exported=list(csv.DictReader(io.StringIO(Path(info.value.path()).read_text(encoding='utf-8-sig'))))
    assert len(exported)==9754
    missing=next(r for r in exported if r[metric+'__status']=='no_motion')
    assert missing[metric+'__value']==''
    assert missing[metric+'__unit']=='°'
    assert missing[metric+'__calculation_version']=='mechanics-db-20261028-v1'
    assert all(r['unit']=='pitch_record' for r in exported)
    g.locator('.db-controls>summary').click()
    g.locator('[data-db-filter="status"]').select_option('missing')
    assert '9,364' in g.locator('.db-table-actions').inner_text()
    assert all(v=='' for v in g.locator(f'td[data-metric="{metric}"]').evaluate_all('(els)=>els.map(el=>el.dataset.value)'))
    g.locator('[data-db-reset]').click()
    g.locator('[data-db-filter="scope"]').select_option('unlinked')
    assert '423' in g.locator('.db-table-actions').inner_text()
    assert '고유 투구 수' in g.locator('.db-scope-note').inner_text()
    assert g.locator('[data-db-filter="game"] option').count()>1
    assert all(v.startswith('clip:') for v in g.locator('.db-pitch-table tbody tr').evaluate_all('(els)=>els.map(el=>el.dataset.rowId)'))
    g.locator('[data-db-reset]').click()
    g.locator('[data-db-filter="player"]').select_option('65320')
    g.locator('[data-db-filter="pitch"]').select_option('직구')
    assert '4' in g.locator('.db-table-actions').inner_text()
    g.locator('[data-db-filter="pitch"]').select_option('')
    g.locator('[data-db-view="aggregate"]').click()
    assert g.locator('.db-aggregate-table tbody tr').count()==5
    assert g.locator('.db-spark').count()==15
    g.locator('[data-db-preset="pelvis_trunk"]').click()
    assert all('n=0' in s for s in g.locator('.db-summary-value>span').all_text_contents())
    g.locator('[data-db-review]').check()
    assert any('n=0' not in s for s in g.locator('.db-summary-value>span').all_text_contents())
    assert g.locator('.db-review-active').is_visible()
    with page.expect_download() as info:g.locator('[data-db-csv]').click()
    aggregated=list(csv.DictReader(io.StringIO(Path(info.value.path()).read_text(encoding='utf-8-sig'))))
    assert len(aggregated)==5
    assert all(r['include_review_3d']=='true' for r in aggregated)
    assert any(int(r['separation_p50_3d__valid_n'])>0 for r in aggregated)
    g.locator('[data-db-review]').uncheck()
    g.locator('[data-db-preset="default"]').click()
    g.locator('[data-db-add-choice]').select_option('knee_extension_2d');g.locator('[data-db-add]').click()
    assert g.locator('[data-db-remove="knee_extension_2d"]').count()==1
    g.locator('[data-db-remove="throw_elbow_range_2d"]').click()
    assert g.locator('[data-db-remove="throw_elbow_range_2d"]').count()==0
    g.locator('[data-db-view="pitches"]').click()
    g.locator('[data-db-distribution="knee_extension_2d"]').click()
    assert g.locator('.db-data-point').count()==6
    assert '6 / 17' in g.locator('.db-chart-caption').inner_text()
    g.locator('.db-data-point').first.focus();page.keyboard.press('Enter')
    assert page.locator('.db-dialog').is_visible()
    page.locator('.db-dialog [data-db-definition="knee_extension_2d"]').click()
    assert '최대 굴곡' in page.locator('.db-dialog .db-definition').inner_text()
    assert 'lead-knee-pilot-20261028-v1' in page.locator('.db-dialog .db-definition').inner_text()
    page.locator('.db-dialog [data-db-close]').click()
    g.locator('.db-data-point').first.click()
    page.locator('.db-dialog [data-db-open-clip]').click()
    page.wait_for_function('!!app.poses[3]')
    assert page.locator('#explore').is_visible()
    assert page.locator('#detail-panel .guided-angle-chart').count()==1
    page.locator('.main-nav a[href="#database"]').click();page.wait_for_selector('#metrics-database-root .db-data-point')
    assert g.locator('[data-db-filter="player"]').input_value()=='65320'
    g.locator('[data-db-plot="scatter"]').click()
    g.locator('[data-db-axis="x"]').select_option('speed')
    g.locator('[data-db-axis="y"]').select_option('trunk_tilt_p90_2d')
    assert g.locator('.db-data-point').count()==17
    g.locator('[data-db-axis="x"]').select_option('landing_release_time_ms')
    assert g.locator('.db-data-point').count()==0
    assert g.locator('.db-empty-plot').is_visible()
    g.locator('[data-db-axis="x"]').select_option('lead_knee_range_2d')
    assert g.locator('.db-data-point').count()==17
    g.locator('.db-controls>summary').click()
    page.screenshot(path='/tmp/kbo-db-browser-desktop.png')
    for width in [1440,1280,1180,1120,1101,1100,1024,800,600,390,320]:
        page.set_viewport_size({'width':width,'height':1000});page.wait_for_timeout(80)
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,('page overflow',width)
        chart_font=g.locator('.db-data-plot text').first.evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)*el.getScreenCTM().a')
        assert chart_font>=14,('plot label size',width,chart_font)
        if page.locator('.main-nav').is_visible():
            nav=page.locator('.main-nav').bounding_box();actions=page.locator('.top-actions').bounding_box()
            assert nav['x']+nav['width']<=actions['x']+1,('nav overlap',width)
    page.set_viewport_size({'width':390,'height':844});g.locator('.db-heading').scroll_into_view_if_needed();page.screenshot(path='/tmp/kbo-db-browser-mobile.png')
    page.locator('#lang-toggle').click();page.wait_for_selector('#metrics-database-root .db-pitch-table')
    assert g.locator('[data-db-filter="player"]').input_value()=='65320'
    assert 'Pitch mechanics metrics' in g.locator('h1').inner_text()
    shared=page.url;page.goto(shared,wait_until='networkidle');page.wait_for_selector('#metrics-database-root .db-pitch-table')
    assert g.locator('[data-db-filter="player"]').input_value()=='65320'
    assert g.locator('[data-db-axis="x"]').input_value()=='lead_knee_range_2d'
    g.locator('[data-db-view="dictionary"]').click()
    assert g.locator('.db-dictionary>details').count()==24
    g.locator('.db-dictionary>details').first.locator('summary').click()
    assert 'mechanics-db-20261028-v1' in g.locator('.db-dictionary>details').first.inner_text()
    page.goto(base+'/?pitcher=65320#pitchers',wait_until='networkidle');page.wait_for_selector('#player-metrics-root .db-aggregate-table')
    assert page.locator('#report-tab-metrics').get_attribute('aria-selected')=='true'
    assert page.locator('#player-metrics-root .db-aggregate-table tbody tr').count()==5
    page.locator('#player-metrics-root [data-db-filter="game"]').select_option('20240530LGSK02024')
    assert '13' in page.locator('#player-metrics-root .db-table-actions').inner_text()
    page.goto(base+'/?pitcher=50030#pitchers',wait_until='networkidle');page.wait_for_selector('#player-metrics-root .db-aggregate-table')
    assert page.locator('#player-metrics-root [data-db-filter="scope"]').input_value()=='records'
    assert '27' in page.locator('#player-metrics-root .db-table-actions').inner_text()
    assert all('n=0' in s for s in page.locator('#player-metrics-root .db-summary-value>span').all_text_contents())
    assert not errors,errors;assert not failures,failures
    print(json.dumps({'ok':True,'pitchRecords':9754,'linkedPitches':390,'unlinkedClips':423,'metrics':24,'exportedFilteredRows':len(exported),'defaultPlayerAggregation':True,'perMetricQuality':True,'columnSelectionSorting':True,'unitSeparation':True,'existingDetailPreserved':True,'urlAndLanguagePersistence':True,'viewportWidths':[1440,1280,1180,1120,1101,1100,1024,800,600,390,320],'pageErrors':errors,'failedResponses':failures},ensure_ascii=False))
    browser.close()
