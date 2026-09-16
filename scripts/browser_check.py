"""Real Chromium checks under the same repository subpath as GitHub Pages."""
import argparse
import json
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        if path.startswith('/PL-predictions/'): path=path[len('/PL-predictions'):]
        return super().translate_path(path)
    def log_message(self,*args): pass

def run(output):
    output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(ROOT/'site')))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/PL-predictions/'
    payload=json.loads((ROOT/'site/data/predictions.json').read_text(encoding='utf8'))
    result={'browser':'Chromium','url_path':'/PL-predictions/','viewports':[], 'scenarios':[]}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch()
            result['browser_version']=browser.version
            for name,width,height in [('desktop',1440,1000),('tablet',834,1112),('mobile',390,844)]:
                page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
                errors=[];failed=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('console',lambda msg:errors.append(msg.text) if msg.type=='error' else None)
                page.on('requestfailed',lambda r:failed.append(r.url))
                page.on('response',lambda r:failed.append(f'{r.status} {r.url}') if r.status>=400 else None)
                started=time.perf_counter();page.goto(url)
                expect(page.locator('#status')).to_contain_text('estimates')
                load_ms=round((time.perf_counter()-started)*1000)
                assert page.locator('.lineup-table').count()==0,'Squads were eagerly rendered'
                assert page.locator('.match-details').count()==0,'Details were eagerly rendered'
                page.screenshot(path=str(output/f'{name}.png'),full_page=True)
                def no_overflow():
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),f'Page overflow at {width}px'
                def accessibility(state):
                    if not page.evaluate('Boolean(window.axe)'):
                        page.add_script_tag(path=str(ROOT/'node_modules/axe-core/axe.min.js'))
                    audit=page.evaluate("async () => await axe.run(document, {runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}})")
                    summary={'violations':audit['violations'],'incomplete':audit['incomplete'],'passed_rules':len(audit['passes'])}
                    (output/f'{name}-{state}-accessibility.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
                    assert not audit['violations'], [(v['id'],len(v['nodes'])) for v in audit['violations']]
                no_overflow()
                page.keyboard.press('Tab');expect(page.locator('.skip')).to_be_focused()
                page.keyboard.press('Enter');expect(page.locator('#forecast-content')).to_be_focused()
                page.locator('[data-league="premier-league"]').focus()
                assert page.locator('[data-league="premier-league"]').evaluate("e => getComputedStyle(e).outlineStyle")!='none'
                if page.locator('.fixture').count():
                    summary=page.locator('.fixture summary').first
                    summary.focus();page.keyboard.press('Enter')
                    expect(page.locator('.fixture[open] .match-details').first).to_be_visible()
                    table_count=page.locator('.lineup-table').count()
                    no_overflow();page.locator('.fixture[open]').screenshot(path=str(output/f'{name}-squad.png'))
                    accessibility('squad')
                    summary.press('Enter');summary.press('Enter')
                    assert page.locator('.lineup-table').count()==table_count,'Repeated opens duplicated tables'
                    if table_count:
                        page.locator('.table-wrap').first.focus()
                        expect(page.locator('.table-wrap').first).to_be_focused()
                        if name=='mobile':
                            page.keyboard.press('ArrowRight');page.wait_for_timeout(100)
                            assert page.locator('.table-wrap').first.evaluate('e => e.scrollLeft')>0
                if page.get_by_label('Next gameweek').is_visible() and not page.get_by_label('Next gameweek').is_disabled():
                    before=page.locator('#gameweek-title').inner_text();page.get_by_label('Next gameweek').click()
                    assert page.locator('#gameweek-title').inner_text()!=before
                    page.get_by_label('Previous gameweek').click();expect(page.locator('#gameweek-title')).to_have_text(before)
                started=time.perf_counter();page.get_by_label('Find a club').fill('Arsenal')
                expect(page.locator('#status')).to_contain_text('all gameweeks')
                expected=sum('arsenal' in (f['home_team']+' '+f['away_team']).lower() for f in payload['leagues']['premier-league'])
                assert page.locator('.fixture').count()==expected
                search_ms=round((time.perf_counter()-started)*1000)
                assert page.locator('.lineup-table').count()==0
                page.get_by_label('Find a club').fill('no such club');expect(page.locator('#fixtures')).to_contain_text('No fixtures found')
                page.get_by_label('Find a club').fill('')
                page.get_by_role('button',name='Season outlook',exact=True).click()
                outlook=payload.get('season_outlook',{}).get('premier-league')
                if outlook: expect(page.locator('.league-table tbody tr')).to_have_count(len(outlook['table']))
                else: expect(page.locator('#outlook')).to_contain_text('Season outlook is unavailable')
                no_overflow();page.locator('#outlook').screenshot(path=str(output/f'{name}-season.png'))
                accessibility('season')
                page.get_by_role('button',name='Championship',exact=True).click()
                expect(page.locator('#metric-league')).to_have_text('Championship')
                notice=payload.get('coverage',{}).get('championship',{}).get('notice')
                if notice: expect(page.locator('#coverage-notice')).to_have_text(notice)
                else: expect(page.locator('#coverage-notice')).to_be_hidden()
                page.get_by_role('button',name='Season outlook',exact=True).click()
                outlook=payload.get('season_outlook',{}).get('championship')
                if outlook: expect(page.locator('.league-table tbody tr')).to_have_count(len(outlook['table']))
                else: expect(page.locator('#outlook')).to_contain_text('Season outlook is unavailable')
                page.get_by_role('button',name='League One',exact=True).click()
                notice=payload.get('coverage',{}).get('league-one',{}).get('notice')
                if notice: expect(page.locator('#coverage-notice')).to_have_text(notice)
                if not payload['leagues'].get('league-one'): expect(page.locator('#fixtures')).to_contain_text('No fixtures found')
                no_overflow()
                # Automated WCAG A/AA audit, including color contrast, plus manual screenshots.
                page.get_by_role('button',name='Premier League',exact=True).click()
                accessibility('matches')
                assert not errors,errors
                assert not failed,failed
                result['viewports'].append(dict(name=name,width=width,height=height,load_ms=load_ms,search_ms=search_ms,console_errors=errors,failed_requests=failed,axe_violations=0))
                page.close()

            def scenario(name,transform=None,optional_fail=False,fail_first=False,hold=False):
                page=browser.new_page(viewport={'width':390,'height':844})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                requests=[]
                count=0
                def route_forecasts(route):
                    nonlocal count
                    count+=1
                    if hold: requests.append(route);return
                    if fail_first and count==1: route.fulfill(status=503,body='Unavailable');return
                    changed=json.loads(json.dumps(payload))
                    if transform: transform(changed)
                    route.fulfill(status=200,content_type='application/json',body=json.dumps(changed))
                page.route('**/data/predictions.json',route_forecasts)
                if optional_fail:
                    page.route('**/data/team_colours.json',lambda r:r.fulfill(status=404,body='Missing'))
                    page.route('**/data/evaluation.json',lambda r:r.fulfill(status=404,body='Missing'))
                page.goto(url,wait_until='domcontentloaded')
                if hold:
                    expect(page.locator('#status')).to_have_text('Loading forecasts…')
                    expect(page.get_by_role('button',name='Premier League',exact=True)).to_be_disabled()
                    page.wait_for_timeout(100)
                    assert requests
                    requests[0].fulfill(status=200,content_type='application/json',body=json.dumps(payload))
                if fail_first:
                    expect(page.get_by_role('button',name='Try again')).to_be_visible()
                    page.get_by_role('button',name='Try again').click()
                expect(page.locator('#status')).to_contain_text('estimates')
                if name=='stale': expect(page.locator('#data-notice')).to_contain_text('48 hours')
                if optional_fail: expect(page.locator('#evaluation')).to_have_text('Evaluation unavailable.')
                if name=='missing-league':
                    page.get_by_role('button',name='League One',exact=True).click();expect(page.locator('#fixtures')).to_contain_text('No fixtures found')
                if name=='unassigned-round':
                    while not page.get_by_label('Next gameweek').is_disabled(): page.get_by_label('Next gameweek').click()
                    expect(page.locator('#gameweek-title')).to_have_text('Round unassigned')
                    expect(page.locator('.fixture')).to_have_count(1)
                assert not errors,errors
                result['scenarios'].append(name)
                page.close()
            scenario('loading',hold=True)
            scenario('retry',fail_first=True)
            scenario('optional-missing',optional_fail=True)
            scenario('stale',transform=lambda p:p.update(generated_at='2020-01-01T00:00:00Z'))
            scenario('missing-league',transform=lambda p:p['leagues'].update({'league-one':[]}))
            if any(isinstance(f.get('matchweek'),int) for f in payload['leagues'].get('premier-league',[])[1:]):
                scenario('unassigned-round',transform=lambda p:p['leagues']['premier-league'][0].update(matchweek=None))
            browser.close()
    finally: server.shutdown();server.server_close()
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/browser');args=parser.parse_args()
    run(args.output)
