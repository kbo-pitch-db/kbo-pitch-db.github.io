"""Check deployed canonical index and every GVHMR payload by content hash.

Read-only HTTP; never pushes or creates a deployment-success guard. A live
browser check is also required before the arm experiment may begin.
"""
from common import *
from all813_release import ROOT
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='https://kbo-pitch-db.github.io/');a=ap.parse_args();base=a.base.rstrip('/')+'/';stamp=str(int(time.time()))
 def fetch(path):
  for attempt in range(3):
   try:
    with urlopen(Request(base+path+('?v='+stamp),headers={'Cache-Control':'no-cache','User-Agent':'KBO-Pitch-Lab-release-verification'}),timeout=35) as r:return r.read()
   except Exception:
    if attempt==2:raise
    time.sleep(1+attempt)
 raw=fetch('data/clips.json');expected=(CODE/'data/clips.json').read_bytes();assert raw==expected,'Live canonical index is not the intended release';index=json.loads(raw);records={r['id']:r for r in json.loads((ROOT/'release_records.json').read_text())};assert len(index['clips'])==813
 def verify(c):
  expected=records[c['id']]['gzip_sha256']
  if c.get('supportAudit'):
   audit=json.loads((CODE/'data'/c['supportAudit']).read_text())
   assert audit['selected_pose_sha256']==c['sceneSignature']
   assert audit['status']=='support_consistency_passed'
   expected=audit['public_payload_sha256']
  data=fetch('data/'+c['pose3d']);actual=hashlib.sha256(data).hexdigest();assert actual==expected,c['id'];return dict(id=c['id'],sha256_gzip=actual,bytes=len(data),method=c['motionMethod'])
 results=[]
 with ThreadPoolExecutor(max_workers=8) as executor:
  futures=[executor.submit(verify,c) for c in index['clips']]
  for f in as_completed(futures):
   results.append(f.result())
   if len(results)%100==0:print('LIVE VERIFIED',len(results),'/ 813',flush=True)
 report=dict(base=base,clips=len(results),players=len({c['playerKey'] for c in index['clips']}),index_sha256=hashlib.sha256(raw).hexdigest(),all_pose_hashes_match=True,bytes=sum(r['bytes'] for r in results),verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),results=sorted(results,key=lambda x:x['id']),browser_verification_still_required=True)
 save(ROOT/'live_asset_verification.json',report);print('ALL 813 LIVE GVHMR ASSETS VERIFIED',report['bytes'],'bytes',flush=True)

if __name__=='__main__':main()
