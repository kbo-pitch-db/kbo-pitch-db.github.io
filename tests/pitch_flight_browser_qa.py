"""Real-browser validation of ball flight, redesigned exploration and full exports."""
import csv
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs,urlparse
from playwright.sync_api import sync_playwright

BASE=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:50093').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
DATA=json.loads((Path(__file__).resolve().parents[1]/'data/portal/pitch_flight.json').read_text())
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=CHROME,args=['--no-sandbox']);page=b.new_page(viewport={'width':1500,'height':1100});errors=[];failed=[]
 page.on('pageerror',lambda e:errors.append(e.stack));page.on('response',lambda r:failed.append(r.url) if r.status>=400 and r.url.startswith(BASE) else None);page.add_init_script("localStorage.setItem('kbo-lang','ko')")
 page.goto(BASE+'/?pitcher=65320#pitchers',wait_until='networkidle');root=page.locator('#pitcher-profile');page.wait_for_selector('#visitor-flight .flight-chart')
 assert root.locator('.flight-point').count()==10
 assert 'n=356' in root.locator('#visitor-flight').inner_text()
 # Clicking either flight chart uses the same single pitch selection as the motion.
 root.locator('.flight-vaa [data-flight-pitch="커브"]').click();assert page.locator('#visitor-pitch').input_value()=='커브'
 page.wait_for_selector('#visitor-form[data-ready=true]');assert '커브' in root.locator('.visitor-delivery-meta').inner_text()
 root.locator('.flight-point[data-flight-pitch="슬라이더"]').first.focus();page.keyboard.press('Enter');assert page.locator('#visitor-pitch').input_value()=='슬라이더'
 page.screenshot(path='/tmp/kbo-flight-profile-desktop.png',full_page=True)
 page.goto(BASE+'/#database',wait_until='networkidle');g=page.locator('#metrics-database-root');page.wait_for_selector('.db-pitch-table')
 assert g.locator('[data-db-family=flight]').get_attribute('aria-pressed')=='true'
 assert g.locator('.db-result-summary').inner_text().startswith(f"{DATA['recordCount']:,}")
 assert g.locator('.db-controls').is_visible() and g.locator('.db-filter-grid').is_visible()
 assert g.locator('details.db-controls,details.db-plots,.db-release-links').count()==0
 assert f"n={DATA['statusCounts']['derived_pts']}" in g.locator('.db-flight-overview').inner_text()
 assert g.locator('.db-pitch-table td[data-metric=hb_cm]').count()==20
 first=g.locator('.db-pitch-table tbody tr').first;identity=first.get_attribute('data-row-id')
 assert float(first.locator('td[data-metric=hb_cm]').get_attribute('data-value'))==DATA['rows'][identity]['hb_cm']
 assert first.locator('td[data-metric=hb_cm]').inner_text()!='—'
 # Export includes every filtered row, zeros and nulls, not only the visible page.
 with page.expect_download() as download:g.locator('[data-db-csv]').click()
 rows=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
 assert len(rows)==DATA['recordCount'] and len([r for r in rows if r['hb_cm__value']==''])==7
 assert all(r['hb_cm__calculation_version']=='naver-pts-flight-20261029-v1' for r in rows)
 g.locator('[data-db-filter=player]').select_option('65320');assert '356' in g.locator('.db-result-summary').inner_text()
 with page.expect_download() as download:g.locator('[data-db-json]').click()
 payload=json.loads(Path(download.value.path()).read_text());assert len(payload['rows'])==356 and payload['flightVersion']==DATA['version']
 g.locator('[data-db-view=aggregate]').click();assert g.locator('.db-aggregate-table tbody tr').count()==5
 # Flight aggregation does not split the same player's repertoire by video/model availability.
 with page.expect_download() as download:g.locator('[data-db-csv]').click()
 aggregated=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))));assert len(aggregated)==5
 assert sum(int(r['rows']) for r in aggregated)==356
 # Adding a mechanics column preserves capture/model groups rather than pooling them.
 g.locator('[data-db-add-choice]').select_option('lead_knee_net_extension');g.locator('[data-db-add]').click()
 with page.expect_download() as download:g.locator('[data-db-csv]').click()
 mixed=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
 assert len(mixed)>5 and sum(int(r['rows']) for r in mixed)==356
 assert all(r['capture_condition']!='naver_pts' for r in mixed)
 g.locator('[data-db-remove=lead_knee_net_extension]').click()
 assert g.locator('.db-aggregate-table tbody tr').count()==5

 g.locator('[data-db-view=charts]').click();assert g.locator('.db-data-point').count()==356
 g.locator('[data-db-plot=scatter]').click();g.locator('[data-db-axis=x]').select_option('ivb_cm');g.locator('[data-db-axis=y]').select_option('vaa_deg')
 assert g.locator('.db-data-point').count()==356
 share=page.url;page.reload(wait_until='networkidle');g=page.locator('#metrics-database-root')
 assert g.locator('[data-db-view=charts]').get_attribute('aria-selected')=='true'
 assert g.locator('[data-db-axis=x]').input_value()=='ivb_cm'
 assert g.locator('[data-db-axis=y]').input_value()=='vaa_deg'
 # Motion metrics retain unavailable values and do not invent physical peak timing.
 g.locator('[data-db-family=mechanics]').click();assert g.locator('[data-db-filter=condition]').is_visible()
 g.locator('[data-db-filter=metric]').select_option('lead_knee_net_extension')
 assert g.locator('.db-data-point').count()==6
 g.locator('[data-db-plot=scatter]').click();g.locator('[data-db-axis=x]').select_option('fc_br_time_ms');assert g.locator('.db-data-point').count()==0
 assert g.locator('.db-empty-plot').is_visible()
 g.locator('[data-db-view=pitches]').click();g.locator('[data-db-preset=timing]').click()
 assert all(v=='' for v in g.locator('td[data-metric=pelvis_trunk_peak_gap_ms]').evaluate_all('(els)=>els.map(e=>e.dataset.value)'))
 # The old sharing contract still restores columns, filters and plots.
 page.goto(BASE+'/?pitcher=65320&pm_scope=linked&pm_view=aggregate&pm_pitch=슬라이더&pm_columns=lead_knee_net_extension,thorax_pelvis_axial_fc&pm_metric=lead_knee_net_extension&pm_plots=yes#pitchers',wait_until='networkidle')
 page.wait_for_selector('#player-metrics-root .db-aggregate-table');assert urlparse(page.url).fragment=='analysis'
 assert page.locator('#player-metrics-root [data-db-filter=pitch]').input_value()=='슬라이더'
 assert page.locator('#player-metrics-root .db-selected-columns button').count()==2
 for width in [768,390,320]:
  page.set_viewport_size({'width':width,'height':844});page.goto(BASE+'/#database',wait_until='networkidle');page.wait_for_selector('.db-pitch-table')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  if width<760:
   assert page.locator('.db-pitch-table thead').is_hidden()
   assert page.locator('.db-pitch-table tbody tr').first.locator('[data-label]').count()==8
   assert page.locator('[data-db-order]').is_visible()
  page.screenshot(path=f'/tmp/kbo-flight-explorer-{width}.png',full_page=True)
 # English, explanations and archive navigation remain functional.
 page.locator('#lang-toggle').click();page.wait_for_selector('.db-pitch-table');assert 'Explore data' in page.locator('.db-heading').inner_text()
 page.locator('.db-reference-actions [data-db-view=dictionary]').click();assert page.locator('.db-dictionary').is_visible()
 page.locator('[data-db-family=flight]').click();assert page.locator('.db-pitch-table').is_visible()
 assert not errors,errors
 assert not failed,failed
 print(json.dumps({'base':BASE,'passed':True,'records':DATA['recordCount'],'withMovement':DATA['statusCounts']['derived_pts'],'missingTracking':7,'exports':'all filtered rows','mobile':[768,390,320],'errors':errors,'failed':failed}))
 b.close()
