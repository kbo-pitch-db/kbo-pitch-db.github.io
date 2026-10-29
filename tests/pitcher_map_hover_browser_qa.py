"""Hover previews reveal actual neighbours without changing a pinned selection."""
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

BASE=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:50102').rstrip('/')
CHROME='/home/eis/.cache/ms-playwright/chromium-1129/chrome-linux/chrome'
OUT=Path('/tmp/kbo-pitcher-map-hover-qa');OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    b=p.chromium.launch(executable_path=CHROME,headless=True,args=['--no-sandbox'])
    ctx=b.new_context(viewport={'width':1440,'height':1100})
    page=ctx.new_page();errors=[];failed=[]
    page.add_init_script("localStorage.setItem('kbo-lang','ko')")
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failed.append(r.url) if r.status>=400 and r.url.startswith(BASE) else None)
    page.goto(BASE+'/#visuals/map?year=2024',wait_until='networkidle',timeout=60000);page.wait_for_selector('.pitcher-map-node')
    assert page.locator('.pitcher-map-edges-background line').count()>154
    original=page.evaluate('JSON.stringify(PitcherMap.state.positions)');url=page.url
    page.screenshot(path=str(OUT/'network.png'),full_page=True);page.wait_for_timeout(200)
    node=page.locator('[data-map-player="65320"]');before=node.locator('.pitcher-map-mark').bounding_box()['width']
    node.hover(force=True);page.wait_for_timeout(240)
    assert page.evaluate('PitcherMap.state.hovered')=='65320'
    assert not page.evaluate('PitcherMap.state.selected')
    assert page.url==url and page.evaluate('JSON.stringify(PitcherMap.state.positions)')==original
    assert node.locator('.pitcher-map-mark').bounding_box()['width']>before*1.5
    assert page.locator('#pitcher-map-tooltip').is_visible()
    assert '최원태' in page.locator('#pitcher-map-tooltip').inner_text()
    assert page.locator('.pitcher-map-edge').count()==5
    expected=page.evaluate('AffinityCore.rank(PitcherMap.state.data.players,PitcherMap.state.data.players.find(p=>p.key==="65320")).slice(0,5).map(n=>({key:n.player.key,score:n.score}))')
    for actual,want in zip(page.locator('.pitcher-map-edge').all(),expected):
        assert actual.get_attribute('data-edge-target')==want['key']
        assert abs(float(actual.get_attribute('data-edge-score'))-want['score'])<1e-8
    assert page.locator('.pitcher-map-node.is-context').count()==6
    assert page.locator('.pitcher-map-node:not(.is-context)').first.evaluate('(e)=>parseFloat(getComputedStyle(e).opacity)')<.3
    assert page.locator('.pitcher-map-edge-line').first.evaluate('(e)=>parseFloat(getComputedStyle(e).strokeWidth)')>2
    tip=page.locator('#pitcher-map-tooltip').bounding_box();plot=page.locator('#pitcher-map-plot').bounding_box()
    assert tip['x']>=plot['x'] and tip['x']+tip['width']<=plot['x']+plot['width']+1
    page.screenshot(path=str(OUT/'hover.png'),full_page=True);page.wait_for_timeout(200)
    page.mouse.move(10,10,steps=3);page.wait_for_function('PitcherMap.state.hovered===""')
    assert page.locator('#pitcher-map-tooltip').is_hidden()
    assert page.locator('.pitcher-map-edge').count()==0
    node.click(force=True);assert page.evaluate('PitcherMap.state.selected')=='65320'
    pinned=page.url
    # Preview another pitcher, then return to the pinned graph on mouse leave.
    other=page.locator('[data-map-player="54640"]');other.hover(force=True)
    assert page.evaluate('PitcherMap.state.hovered')=='54640'
    assert page.evaluate('PitcherMap.state.selected')=='65320' and page.url==pinned
    page.mouse.move(10,10,steps=3);assert page.locator('.pitcher-map-node.is-reference').get_attribute('data-map-player')=='65320'
    # Sidebar hover emphasizes the existing pair, with the same exact score.
    row=page.locator('[data-map-neighbor]').first;key=row.get_attribute('data-map-neighbor');row.hover()
    assert page.evaluate('PitcherMap.state.hoverSource')=='list'
    assert page.locator('.pitcher-map-edge.is-pair').get_attribute('data-edge-target')==key
    assert page.locator('.pitcher-map-node.is-reference').get_attribute('data-map-player')=='65320'
    assert '유사도' in page.locator('#pitcher-map-tooltip').inner_text()
    page.screenshot(path=str(OUT/'list-hover.png'),full_page=True);page.wait_for_timeout(200)
    # Keyboard users receive the same preview and can dismiss it with Escape.
    other.focus();assert page.evaluate('PitcherMap.state.hovered')=='54640'
    assert page.locator('#pitcher-map-tooltip').is_visible() and page.url==pinned
    page.keyboard.press('Escape');assert page.locator('#pitcher-map-tooltip').is_hidden()
    assert page.evaluate('PitcherMap.state.selected')=='65320'
    assert page.evaluate('JSON.stringify(PitcherMap.state.positions)')==original
    # Repeated previews cannot replace nodes, navigate or accumulate graph edges.
    for key in ['65320','54640','77829','65320','54640']:
        page.locator(f'[data-map-player="{key}"]').hover(force=True)
        assert page.locator('.pitcher-map-node').count()==154
        assert page.locator('.pitcher-map-edge').count()==5
        assert page.url==pinned
    page.mouse.move(10,10,steps=3)
    assert not errors,errors;assert not failed,failed
    reduced=b.new_context(viewport={'width':1440,'height':1100},reduced_motion='reduce')
    r=reduced.new_page();r.add_init_script("localStorage.setItem('kbo-lang','ko')")
    r.goto(BASE+'/#visuals/map?year=2024',wait_until='networkidle');r.wait_for_selector('.pitcher-map-node');r.locator('[data-map-player="65320"]').hover(force=True)
    assert r.locator('.is-hovered .pitcher-map-mark').evaluate('(e)=>getComputedStyle(e).transitionDuration')=='0s'
    assert r.locator('.pitcher-map-edge-line').first.evaluate('(e)=>getComputedStyle(e).animationName')=='none';reduced.close()
    mobile=b.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True)
    touch=mobile.new_page();touch.add_init_script("localStorage.setItem('kbo-lang','ko')")
    touch.goto(BASE+'/#visuals/map?year=2024',wait_until='networkidle');touch.wait_for_selector('.pitcher-map-node')
    touch.locator('[data-map-player="65320"]').tap(force=True)
    assert touch.evaluate('PitcherMap.state.selected')=='65320'
    assert touch.locator('.pitcher-map-edge').count()==5
    assert touch.locator('#pitcher-map-tooltip').is_hidden()
    for width in [390,320]:
        touch.set_viewport_size({'width':width,'height':844});touch.wait_for_timeout(180)
        assert touch.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        touch.screenshot(path=str(OUT/f'mobile-{width}.png'),full_page=True)
    mobile.close()
    print(json.dumps({'passed':True,'base':BASE,'hoverDoesNotNavigate':True,'fixedCoordinates':True,'exactEdges':True,'keyboardPreview':True,'touch':True,'reducedMotion':True,'errors':errors,'failed':failed,'screenshots':str(OUT)},ensure_ascii=False))
    b.close()
