"""Read only KBO official throwing-side metadata; never use listed height as scale."""
from common import *
import requests,re,concurrent.futures

def main():
 records=sample('all',False);players={r['player'].split(':')[-1]:r['player_name'] for r in records if r['player'].startswith('naver:')};root=OUT/'metadata/kbo_hands';root.mkdir(parents=True,exist_ok=True)
 def fetch(item):
  code,name=item;dest=root/f'{code}.json'
  if dest.exists() and 'pitching_style' in json.loads(dest.read_text()):return json.loads(dest.read_text())
  url='https://www.koreabaseball.com/Record/Player/PitcherDetail/Basic.aspx?playerId='+code
  try:
   cached=root/f'{code}.html'
   if cached.exists():html=cached.read_text()
   else:
    response=requests.get(url,timeout=20);html=response.text;cached.write_text(html)
   cleanhtml=re.sub('<[^>]+>',' ',html);cleanhtml=re.sub(r'\s+',' ',cleanhtml);m=re.search(r'선수명:\s*([^<]*?)\s+등번호:',cleanhtml);listed=m.group(1).strip() if m else None;throw=re.search(r'투수\(([우좌])([투언사])',cleanhtml);hand='right' if throw and throw[1]=='우' else 'left' if throw and throw[1]=='좌' else 'unknown';style='underhand_or_sidearm' if throw and throw[2] in ['언','사'] else 'unspecified' if throw else 'unknown';verified=listed==name
   if not verified:hand='unknown'
   result=dict(player_code=code,expected_name=name,listed_name=listed,identity_matched=verified,throwing_hand=hand,pitching_style=style,url=url,height_used=False,source='KBO official biography; stable throwing-side metadata, not source motion/3D GT')
  except Exception as e:result=dict(player_code=code,expected_name=name,throwing_hand='unknown',error=type(e).__name__,url=url,height_used=False)
  save(dest,result);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,players.items()))
 save(OUT/'metadata/player_hands.json',{r['player_code']:r for r in results});print('Official names/throwing sides resolved',sum(r['throwing_hand']!='unknown' for r in results),'/',len(results))
if __name__=='__main__':main()
