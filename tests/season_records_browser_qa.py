"""Season changes across profiles, similarity, full exports and 2024 motion."""
import csv,json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=(sys.argv[1]if len(sys.argv)>1 else'http://127.0.0.1:50100').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-season-records-qa');OUT.mkdir(exist_ok=True)
with sync_playwright()as p:
 b=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
 page=ctx.new_page();errors,failed=[],[];page.on('pageerror',lambda e:errors.append(str(e)));page.on('response',lambda r:failed.append(r.url)if r.status>=400 and r.url.startswith(BASE)else None);page.add_init_script("localStorage.setItem('kbo-lang','ko')")
 page.goto(BASE+'/',wait_until='networkidle',timeout=60000)
 assert page.locator('#home').is_visible()
 assert '585,299'in page.locator('#overview-stats').inner_text()
 assert '동작 영상 2024'in page.locator('#release-line').inner_text()
 page.locator('.main-nav a[href="#pitchers"]').click();page.wait_for_selector('.pitcher-card')
 assert page.locator('#pitcher-season').count()==1
 page.locator('#pitcher-season').select_option('2025');assert 'season=2025'in page.url
 assert page.locator('.pitcher-card .micro-label').first.inner_text().startswith('2025')
 page.goto(BASE+'/?pitcher=65320&season=2026#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]')
 assert page.locator('#visitor-record-season').input_value()=='2026'
 assert '2024'in page.locator('#visitor-form-title').inner_text()
 assert '2024'in page.locator('#visitor-mechanics-title').inner_text()
 assert '2026-'in page.locator('.visitor-section').nth(1).inner_text()
 for year in ['2026','2025','2024']:
  page.locator('#visitor-record-season').select_option(year);page.wait_for_function('(y)=>document.querySelector("#pitcher-profile").dataset.recordSeason===y',arg=year);page.wait_for_selector('#visitor-form[data-ready=true]')
  page.wait_for_function('(y)=>new URL(location.href).searchParams.get("season")===y',arg=year)
  expected=page.evaluate('(y)=>SeasonRecords.summary(y).then(d=>d.players["65320"].recordCount)',year)
  assert f'n={expected}'in page.locator('#visitor-bars').inner_text()
  assert page.locator('#visitor-form').get_attribute('data-clip')
  page.screenshot(path=str(OUT/f'player-{year}.png'),full_page=True)
 page.go_back();page.wait_for_selector('#visitor-record-season');assert page.locator('#visitor-record-season').input_value()=='2025'
 # A new record-only pitcher does not inherit another player's old video.
 newcode=page.evaluate('Object.keys(SeasonRecords.info(2026).players).find(c=>!PitchPortal.state.summary.players.find(p=>p.code===c)?.detailPath)')
 page.goto(BASE+f'/?pitcher={newcode}&season=2026#pitchers',wait_until='networkidle');page.wait_for_selector('.visitor-empty')
 assert page.locator('#visitor-form canvas').count()==0
 assert '동작 영상'in page.locator('#visitor-form').inner_text()
 assert page.locator('#visitor-cards .visitor-metric-value strong').count()==0
 page.locator('[data-visitor-explore]').click();page.wait_for_selector('#player-metrics-root .db-pitch-table')
 for tab in ['arsenal','overview','mechanics','metrics']:
  page.locator('#report-tab-'+tab).click();page.wait_for_timeout(50)
 assert not errors,errors
 # Similarity compares only the chosen year, defaulting to latest available.
 page.goto(BASE+'/#visuals/affinity',wait_until='networkidle');page.wait_for_selector('#affinity-shapes svg')
 assert page.locator('#affinity-season').input_value()=='2026'
 for year,n in [('2025',146),('2024',154),('2026',125)]:
  page.locator('#affinity-season').select_option(year);page.wait_for_function('(y)=>KBOVisuals.state.data.season===Number(y)',arg=year);page.wait_for_selector('#affinity-shapes svg')
  assert page.evaluate('KBOVisuals.state.data.players.length')==n
  assert all(date.startswith(year)for date in page.evaluate('KBOVisuals.state.data.players.map(p=>p.dateStart)'))
  assert f'year={year}'in page.url
 assert '2026.08.04'in page.locator('.affinity-scope').inner_text()
 # Every CSV row is exported, with actual physical values and no future motion.
 page.goto(BASE+'/?db_family=flight&db_season=2026#database',wait_until='networkidle',timeout=60000);page.wait_for_selector('.db-pitch-table')
 assert '153,820'in page.locator('.db-result-summary').inner_text()
 assert '2026-08-04'in page.locator('.db-season-scope').inner_text()
 with page.expect_download()as dl:page.locator('[data-db-csv]').click()
 file=OUT/'2026.csv';dl.value.save_as(file)
 count=0
 with file.open(encoding='utf-8-sig',newline='')as stream:
  for row in csv.DictReader(stream):
   count+=1;assert row['season']=='2026';assert row['game_id'].startswith('2026');assert not row['primary_clip_id'];assert row['hb_cm__status']=='derived_pts';float(row['hb_cm'])
 assert count==153820
 page.locator('[data-db-view=charts]').click();page.wait_for_selector('.db-data-point')
 assert 0<page.locator('.db-data-point').count()<=2000
 assert '153820'in page.locator('.db-chart-caption').inner_text().replace(',','')
 page.locator('[data-db-filter=season]').select_option('2025');page.wait_for_selector('.db-data-point');assert '215,788'in page.locator('.db-result-summary').inner_text()
 page.locator('[data-db-filter=player]').select_option('65320');page.locator('[data-db-view=pitches]').click();page.wait_for_selector('.db-pitch-table')
 expected=page.evaluate('SeasonRecords.info(2025).players["65320"].recordCount')
 with page.expect_download()as dl:page.locator('[data-db-json]').click()
 file=OUT/'2025-player.json';dl.value.save_as(file);export=json.loads(file.read_text());assert export['season']=='2025';assert len(export['rows'])==expected
 page.locator('[data-db-family=mechanics]').click();page.wait_for_selector('.db-pitch-table')
 assert page.locator('[data-db-filter=season]').input_value()=='2024'
 # Real mobile layouts and share reloads keep years separate.
 for width in [768,390,320]:
  page.set_viewport_size({'width':width,'height':844});page.goto(BASE+'/?pitcher=65320&season=2026#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  assert page.locator('#visitor-record-season').bounding_box()['height']>=44
  page.screenshot(path=str(OUT/f'mobile-player-{width}.png'),full_page=True)
  page.goto(BASE+'/?db_family=flight&db_season=2026&db_player=65320#database',wait_until='networkidle');page.wait_for_selector('.db-pitch-table')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
  assert page.locator('[data-db-filter=season]').input_value()=='2026'
  assert all('2026-'in text for text in page.locator('.db-pitch-table tbody tr').all_inner_texts())
  page.screenshot(path=str(OUT/f'mobile-explorer-{width}.png'),full_page=True)
  page.goto(BASE+'/?season=2026#pitchers',wait_until='networkidle');page.wait_for_selector('.pitcher-card')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),('directory',width)
  assert page.locator('#pitcher-season').count()==1
 page.goto(BASE+'/#guide',wait_until='networkidle');assert page.locator('#season-downloads a').count()==3
 assert all('/202'in a.get_attribute('href')for a in page.locator('#season-downloads a').all())
 assert not errors,errors;assert not failed,failed
 print(json.dumps({'passed':True,'base':BASE,'fullCsvRows':count,'newRecordOnlyPlayer':newcode,'mobile':[768,390,320],'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False));b.close()
