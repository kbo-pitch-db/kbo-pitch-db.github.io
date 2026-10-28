"""Regression checks for reading sizes, visible definitions and linked joint emphasis.
Requires Playwright and Chromium. Usage: python3 tests/readability_browser_qa.py URL CHROME
"""
import json
import sys
from playwright.sync_api import sync_playwright

base=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8789').rstrip('/')
chrome=sys.argv[2] if len(sys.argv)>2 else None
contrast_script='''selector=>{
 const el=document.querySelector(selector),style=getComputedStyle(el);
 const rgb=s=>s.match(/[\\d.]+/g).slice(0,3).map(Number);
 let parent=el,background='rgb(255,255,255)';
 while(parent){const value=getComputedStyle(parent).backgroundColor;if(value!=='rgba(0, 0, 0, 0)'&&value!=='transparent'){background=value;break}parent=parent.parentElement}
 const luminance=color=>rgb(color).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
 const foreground=el instanceof SVGElement&&style.fill!=='none'?style.fill:style.color;const a=luminance(foreground),b=luminance(background);const line=parseFloat(style.lineHeight)/parseFloat(style.fontSize);
 return {font:parseFloat(style.fontSize),line:Number.isFinite(line)?line:null,contrast:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};
}'''
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=chrome,headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(base+'/?pitcher=65320&v=reading1#pitchers',wait_until='networkidle')
    page.wait_for_selector('.stat-definition')
    assert page.evaluate('visualViewport.scale')==1
    assert page.locator('.stat-definition').count()==4
    for definition in page.locator('.stat-definition').all():assert definition.is_visible()
    assert page.locator('.report-quick-stats details[open]').count()==0
    whiff=page.locator('[data-stat="whiffRate"]')
    assert '스윙 152회 중 헛스윙 33회' in whiff.locator('.stat-denominator').inner_text()
    assert '타자가 스윙했을 때 공을 맞히지 못한 비율' in whiff.locator('.stat-definition').inner_text()
    whiff.get_by_text('계산 방법 보기',exact=True).click()
    assert whiff.locator('.stat-formula').is_visible()
    whiff.get_by_text('계산 방법 보기',exact=True).click()
    checks={}
    for selector,minimum in [('.stat-definition',17),('.metric-label',16),('.metric-number small',16),('.stat-denominator',14),('.report-sample-note',17),('.report-coverage>small',14),('.report-tabs>button',16)]:
        result=page.evaluate(contrast_script,selector);checks[selector]=result
        assert result['font']>=minimum,(selector,result)
        assert result['contrast']>=4.5,(selector,result)
    assert abs(checks['.stat-definition']['line']-1.6)<.05
    page.screenshot(path='/tmp/kbo-reading-overview.png',full_page=True)
    page.locator('#report-tab-mechanics').click();page.locator('#single-pitch-inspector>summary').click();page.wait_for_selector('.guided-angle-chart')
    assert page.locator('.motion-demo-label h3').inner_text()=='2D 동작'
    assert page.locator('.curve-source-title').inner_text()=='3D 모델에서 계산한 각도'
    assert page.get_by_role('combobox',name='재생 속도',exact=True).count()==1
    for key in ['rightElbow','leftKnee','leftElbow','rightKnee']:
        page.locator('.guided-metric-select').select_option(key)
        assert page.locator('#profile-motion').get_attribute('data-selected-joint')==key
        assert page.locator('.guided-angle-chart').get_attribute('data-selected-joint')==key
        assert all(c.get_attribute('data-selected-joint')==key for c in page.locator('[data-sequence]').all())
    page.locator('.guided-metric-select').select_option('rightElbow')
    page.locator('[data-snapshot="0.5"]').click()
    assert page.locator('#profile-motion').get_attribute('data-highlighted-bones')=='2'
    amber=page.locator('#profile-motion').evaluate('canvas=>{const pixels=canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height).data;let n=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i]===255&&pixels[i+1]===193&&pixels[i+2]===102)n++;return n}')
    assert amber>50,amber
    assert page.locator('.guided-joint-name').inner_text()=='오른팔꿈치'
    def same_frame():
        return page.evaluate('''()=>{const canvas=document.querySelector('#profile-motion'),chart=document.querySelector('.guided-angle-chart');return canvas.dataset.frame===chart.getAttribute('aria-valuenow')&&chart.querySelector('.cursor-value-text').textContent===document.querySelector('.guided-angle-value').textContent&&canvas.dataset.selectedJoint===chart.dataset.selectedJoint}''')
    assert same_frame()
    page.locator('#profile-play').click()
    page.locator('.guided-metric-select').select_option('rightKnee')
    page.wait_for_timeout(200)
    assert same_frame()
    page.locator('#profile-play').click()
    page.locator('.dimension-2').click()
    assert page.locator('.curve-source-title').inner_text()=='2D 영상에서 계산한 각도'
    assert '동작과 각도 모두' in page.locator('.motion-space-note').inner_text()
    page.locator('.dimension-3').click()
    assert '각도는 같은 프레임의 3D 모델' in page.locator('.motion-space-note').inner_text()
    for direction in ['Home','End']:
        page.locator('.guided-angle-chart').focus();page.keyboard.press(direction)
        assert same_frame()
        assert page.locator('.curve-cursor-label').evaluate('e=>{const b=e.getBoundingClientRect(),s=e.ownerSVGElement.getBoundingClientRect();return b.left>=s.left&&b.right<=s.right+1}')
    for selector,minimum in [('.metric-purpose p',17),('.guided-joint-name',16),('.curve-reading-legend',14),('.curve-reading-legend>small',14),('.guided-angle-chart text',14),('.delivery-source-actions .text-link',16),('.motion-space-note',14)]:
        result=page.evaluate(contrast_script,selector);checks[selector]=result
        assert result['font']>=minimum,(selector,result)
        assert result['contrast']>=4.5,(selector,result)
    page.screenshot(path='/tmp/kbo-reading-form.png',full_page=True)
    for width in [1440,1280,1100,1024,800,580,390,320]:
        page.set_viewport_size({'width':width,'height':1000});page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth')<=width+1,width
        if width<=1200:assert page.locator('.delivery-workspace').evaluate('el=>getComputedStyle(el).gridTemplateColumns.split(" ").length')==1
    page.set_viewport_size({'width':390,'height':844});page.screenshot(path='/tmp/kbo-reading-mobile-form.png',full_page=True)
    page.locator('#profile-open-3d').click();page.wait_for_function('!!app.poses[3]')
    page.wait_for_selector('#detail-panel .guided-angle-chart')
    page.locator('#detail-panel .guided-metric-select').select_option('leftElbow')
    assert page.locator('.motion-canvas[data-dimension="2"]').get_attribute('data-selected-joint')=='leftElbow'
    assert page.locator('#speed-select').get_attribute('aria-label')=='재생 속도'
    assert page.evaluate('document.documentElement.scrollWidth')<=391
    # Language changes preserve the selected joint and localize the controls.
    page.locator('#lang-toggle').click();page.wait_for_selector('#detail-panel .guided-angle-chart')
    assert page.locator('#detail-panel .guided-joint-name').inner_text()=='Left elbow'
    assert page.locator('#speed-select').get_attribute('aria-label')=='Playback speed'
    assert not errors,errors
    print(json.dumps({'desktop_zoom':1,'visible_definitions_and_denominators':True,'font_and_contrast_checks':checks,'selected_joint_and_two_bones':True,'cursor_angle_and_joint_in_sync':True,'explicit_2d_and_3d_labels':True,'localized_playback_controls':True,'browser_errors':errors},ensure_ascii=False,indent=2))
    browser.close()
