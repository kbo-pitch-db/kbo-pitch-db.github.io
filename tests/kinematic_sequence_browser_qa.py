"""Actual player/style switching, native-frame seeking, advanced shares and phones."""
import json,sys
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from playwright.sync_api import sync_playwright
BASE=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:50094').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-sequence-qa');OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=CHROME,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1000});errors=[];failed=[]
 page.on('pageerror',lambda err:errors.append(str(err)));page.on('response',lambda r:failed.append(r.url) if r.status>=400 and r.url.startswith(BASE) else None);page.add_init_script("localStorage.setItem('kbo-lang','ko')")
 page.goto(BASE+'/?pitcher=77637&delivery=windup#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]');page.wait_for_selector('#visitor-sequence .sequence-row')
 root=page.locator('#pitcher-profile');assert root.locator('.visitor-section').count()==3
 assert root.locator('[data-sequence-stance=windup]').get_attribute('aria-pressed')=='true'
 assert '키네마틱 시퀀스' in root.inner_text();assert '일반 재생속도' in root.locator('#visitor-sequence').inner_text()
 def check_clip(stance):
  cid=page.locator('#visitor-form').get_attribute('data-clip');assert page.locator('#visitor-sequence').get_attribute('data-clip')==cid
  assert page.evaluate('(id)=>KinematicSequence.data.clips[id].stance',cid)==stance
  assert set(parse_qs(urlparse(page.url).query))<= {'pitcher','pitch','delivery'}
  assert parse_qs(urlparse(page.url).query)['delivery']==[stance]
  return cid
 first=check_clip('windup')
 assert root.locator('[data-mechanism=pelvis_trunk_peak_gap_ms] .visitor-metric-value strong').count()==0
 # Peak buttons select the exact native frame on both canvases and the sequence.
 def peak_seek():
  button=root.locator('[data-sequence-frame]').first;frame=int(button.get_attribute('data-sequence-frame'));button.click()
  assert root.locator('#visitor-frame').input_value()==str(frame)
  assert root.locator('canvas').evaluate_all('(els)=>els.map(e=>Number(e.dataset.frame))')==[frame,frame]
  assert all(v==frame for v in root.locator('.sequence-cursor').evaluate_all('(els)=>els.map(e=>Number(e.dataset.frame))'))
 peak_seek();root.locator('#visitor-sequence').screenshot(path=str(OUT/'windup.png'))
 root.locator('[data-sequence-stance=set]').click();page.wait_for_selector('#visitor-form[data-ready=true]');second=check_clip('set');assert first!=second;peak_seek()
 page.reload(wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]');assert check_clip('set')==second
 page.locator('#visitor-pitch').select_option('직구');page.wait_for_selector('#visitor-form[data-ready=true]');check_clip('set')
 # Eligible headline values require at least three distinct pitches in this stance.
 counts=page.evaluate("KinematicSequence.summarize('77637','직구','set')");card=root.locator('[data-mechanism=pelvis_trunk_peak_gap_ms]')
 if counts['n']<3:assert card.locator('.visitor-metric-value strong').count()==0
 else:assert card.locator('.visitor-metric-value strong').count()==1
 # Typical mobile widths preserve readable HTML labels and no horizontal overflow.
 for width in [768,390,320]:
  page.set_viewport_size({'width':width,'height':844});page.goto(BASE+'/?pitcher=77637&delivery=windup#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]');page.wait_for_selector('.sequence-row')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  assert page.locator('.sequence-row>span').first.evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)')>=17
  assert page.locator('[data-sequence-stance=set]').bounding_box()['height']>=44
  peak_seek();page.screenshot(path=str(OUT/f'player-{width}.png'),full_page=True)
 # Advanced shares retain sequence field and set/windup filtering, including CSV.
 page.set_viewport_size({'width':1440,'height':1000});metric='sequence_pelvis_trunk_gap_assumed_ms';share=f'/?db_family=mechanics&db_scope=linked&db_player=77637&db_delivery=windup&db_view=aggregate&db_columns={metric}&db_metric={metric}#database'
 page.goto(BASE+share,wait_until='networkidle');page.wait_for_selector('.db-aggregate-table');assert page.locator('[data-db-filter=delivery]').input_value()=='windup';assert '와인드업' in page.locator('.db-aggregate-table').inner_text()
 with page.expect_download() as download:page.locator('[data-db-csv]').click()
 text=Path(download.value.path()).read_text(encoding='utf-8-sig');assert 'delivery' in text and metric+'__median' in text and 'windup' in text
 page.reload(wait_until='networkidle');page.wait_for_selector('.db-aggregate-table');assert page.locator('[data-db-filter=delivery]').input_value()=='windup'
 page.goto(BASE+'/?pitcher=65320#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]');page.wait_for_selector('.sequence-axis');assert '무릎 최고점' in page.locator('.sequence-axis').inner_text()
 page.goto(BASE+'/?pitcher=54612#pitchers',wait_until='networkidle');page.wait_for_selector('#visitor-form[data-ready=true]');page.wait_for_selector('.sequence-axis');assert '클립 시간' in root.locator('.sequence-axis').inner_text();assert '릴리스' not in root.locator('.sequence-axis').inner_text();assert page.evaluate('KinematicSequence.data.clips[document.querySelector("#visitor-form").dataset.clip].events.br') is None;peak_seek()
 # Old advanced links and the home entry point still resolve independently.
 page.goto(BASE+'/?pitcher=65320&pm_scope=linked&pm_columns=lead_knee_net_extension&pm_view=aggregate#pitchers',wait_until='networkidle');page.wait_for_selector('#player-metrics-root .db-aggregate-table');assert urlparse(page.url).fragment=='analysis'
 page.goto(BASE+'/',wait_until='networkidle');assert page.locator('#home').is_visible()
 assert not errors,errors;assert not failed,failed
 b.close();print(json.dumps({'result':'PASS','styleClips':[first,second],'mobileWidths':[768,390,320],'errors':errors,'failed':failed},ensure_ascii=False))
