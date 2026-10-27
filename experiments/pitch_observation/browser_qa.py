"""Browser verification of fixed axes, source synchronization and native modes."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parents[3]
errors=[]
with sync_playwright() as p:
 browser=p.chromium.connect_over_cdp('http://127.0.0.1:9223');context=browser.new_context(viewport={'width':1440,'height':1100});page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)));page.goto('http://127.0.0.1:8775/comparison.html?v=range1');page.wait_for_function('data !== null');page.wait_for_function('document.querySelector("#video").readyState>=2');page.evaluate('goto(24)');page.wait_for_function('!document.querySelector("#video").seeking && Math.abs(document.querySelector("#video")._decodedTime-data.times[index]) < .002')
 report=page.evaluate('({clips:document.querySelector("#clip").options.length,frames:data.times.length,extent:data.extent,mode:document.querySelector("#mode").value,videoWidth:document.querySelector("#video").videoWidth,canvases:document.querySelectorAll("canvas").length,index,videoTime:document.querySelector("#video").currentTime,expectedTime:data.times[index],decodedTime:document.querySelector("#video")._decodedTime})');print(report,flush=True);assert report['mode']=='global_fixed';assert report['videoWidth']>0;assert abs(report['videoTime']-report['expectedTime'])<.08;assert abs(report['decodedTime']-report['expectedTime'])<.002
 page.screenshot(path=str(OUT/'comparison_screenshot.png'),full_page=True)
 extent=report['extent'];page.evaluate('goto(Math.min(70,data.times.length-1))');assert page.evaluate('data.extent')==extent
 page.select_option('#candidate','MotionBERT last block native');page.select_option('#mode','relative');page.screenshot(path=str(OUT/'relative_comparison_screenshot.png'),full_page=True);page.select_option('#candidate','NLF raw native joint centers (SMPL24 mapping)');page.wait_for_timeout(200);page.screenshot(path=str(OUT/'nlf_comparison_screenshot.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth')<=391;page.screenshot(path=str(OUT/'comparison_mobile.png'),full_page=True);report['errors']=errors;report['fixed_extent_across_frames']=True;report['mobile_no_horizontal_overflow']=True;report['native_relative_modes_checked']=True;assert not errors,errors
 (OUT/'browser_qa.json').write_text(json.dumps(report,indent=2));context.close();print(report)
