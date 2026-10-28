"""End-to-end database, export, evidence and responsive-language checks."""
import csv,io,json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8765'
CHROME=sys.argv[2] if len(sys.argv)>2 else '/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=CHROME,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(e.stack))
 page.add_init_script("localStorage.setItem('kbo-lang','ko')");page.goto(BASE+'/index.html#database');page.wait_for_selector('.db-pitch-table tbody tr',timeout=60000)
 g=page.locator('#metrics-database-root');assert g.locator('[data-db-remove="lead_knee_flexion_fc"]').count()==1
 assert g.locator('td[data-metric="lead_knee_range_2d"]').count()==0
 g.locator('[data-db-filter="scope"]').select_option('records');assert '9,754' in g.locator('.db-table-actions').inner_text()
 with page.expect_download() as info:g.locator('[data-db-csv]').click()
 rows=list(csv.DictReader(io.StringIO(Path(info.value.path()).read_text(encoding='utf-8-sig'))));assert len(rows)==9754;assert any(r['lead_knee_flexion_fc__status']=='no_motion' and r['lead_knee_flexion_fc__value']=='' for r in rows)
 with page.expect_download() as info:g.locator('[data-db-json]').click()
 payload=json.loads(Path(info.value.path()).read_text());assert len(payload['rows'])==9754;assert payload['version']=='biomechanics-db-20261029-v2';assert payload['dictionary'][0]['basis']=='3d'
 g.locator('[data-db-filter="minSpeed"]').fill('150');g.locator('[data-db-filter="minSpeed"]').dispatch_event('change');assert all(float(s)>=150 for s in g.locator('.db-pitch-table tbody tr td:nth-child(3)').all_text_contents())
 g.locator('[data-db-reset]').click();g.locator('[data-db-filter="player"]').select_option('65320');g.locator('[data-db-view="aggregate"]').click();assert g.locator('.db-aggregate-table tbody tr').count()>0
 with page.expect_download() as info:g.locator('[data-db-csv]').click()
 agg=list(csv.DictReader(io.StringIO(Path(info.value.path()).read_text(encoding='utf-8-sig'))));assert all('capture_condition' in r and 'lead_knee_flexion_fc__std' in r and 'lead_knee_flexion_fc__valid_games' in r for r in agg)
 assert any(int(r['lead_knee_flexion_fc__valid_n'])>0 for r in agg)
 g.locator('[data-db-view="pitches"]').click();g.locator('[data-db-distribution="lead_knee_net_extension"]').click();assert g.locator('.db-data-point').count()==6
 g.locator('.db-data-point').first.click();assert page.locator('.db-dialog').is_visible();assert page.locator('.db-dialog a[href*="biomechanics_detail.html"]').count()==1;page.locator('.db-dialog [data-db-close]').click()
 g.locator('[data-db-plot="scatter"]').click();g.locator('[data-db-axis="x"]').select_option('fc_br_time_ms');assert g.locator('.db-data-point').count()==0;assert g.locator('.db-empty-plot').is_visible()
 share=page.url;page.reload();page.wait_for_selector('#metrics-database-root .db-pitch-table',timeout=60000);g=page.locator('#metrics-database-root');assert g.locator('[data-db-filter="player"]').input_value()=='65320';assert g.locator('[data-db-axis="x"]').input_value()=='fc_br_time_ms'
 g.locator('[data-db-view="dictionary"]').click();g.locator('.db-dictionary details').first.locator('summary').click();assert '원문과의 차이' in g.inner_text();assert g.locator('.db-dictionary a[href*="39157018"]').count()>0
 g.locator('[data-db-view="quality"]').click();assert '품질 확인용' in g.inner_text();assert g.locator('a[href*="metrics-20261028-v1"]').count()==1
 # Event-directed numerical source comparison, full provenance and smooth viewer.
 page.goto(BASE+'/biomechanics_detail.html?clip=0d84991ce9f6_c0017&metric=lead_knee_net_extension');page.wait_for_selector('#pose3d',timeout=60000);assert page.locator('.quality-axes>div').count()==3
 page.locator('[data-event="fc"]').click();assert page.locator('#frame').input_value()=='82';assert 't=' in page.locator('#source-event').get_attribute('href')
 assert page.locator('#speed').get_attribute('aria-label')=='재생 속도';assert page.locator('a[href*="39157018"]').count()>0
 page.locator('[data-event="br"]').click();assert page.locator('#frame').input_value()=='102'
 page.locator('#metric').select_option('shoulder_external_rotation_max');assert '—' in page.locator('.metric-value').inner_text();assert 'humeral_axial_rotation_unidentifiable' in page.locator('.quality-axes').inner_text()
 page.locator('#language').click();assert page.locator('#speed').get_attribute('aria-label')=='Playback speed';assert 'External GT accuracy' in page.locator('.quality-axes').inner_text()
 page.locator('#metric').select_option('lead_knee_flexion_fc');page.locator('#frame').fill('80');page.locator('#play').click();page.wait_for_timeout(350);assert int(page.locator('#frame').input_value())>80;page.locator('#play').click()
 page.screenshot(path='/tmp/kbo-bio-evidence-desktop.png',full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path='/tmp/kbo-bio-evidence-mobile.png',full_page=True)
 page.goto(BASE+'/index.html#database');page.wait_for_selector('.db-pitch-table tbody tr',timeout=60000);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path='/tmp/kbo-bio-db-mobile.png',full_page=True)
 assert not errors,errors;browser.close();print('PASS: table filters, full filtered CSV/JSON, grouped mean/SD/n/games, dictionary sources, legacy quality tab, event frame links, 2D/3D viewer, playback, KO/EN and responsive layouts; no page errors')
