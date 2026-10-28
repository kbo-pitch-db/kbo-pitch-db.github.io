"""Read-only deployed-index and all actual combined pose/quality/final-2D hashes."""
from common import *
from combined_population import ROOT,SITE
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='https://kbo-pitch-db.github.io/');a=p.parse_args();base=a.base.rstrip('/')+'/';stamp=str(int(time.time()))
    def fetch(path):
        for attempt in range(3):
            try:
                with urlopen(Request(base+path+'?v='+stamp,headers={'Cache-Control':'no-cache','User-Agent':'KBO-Pitch-Lab-release-verification'}),timeout=40) as r:return r.read()
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
    raw=fetch('data/clips.json');assert raw==(SITE/'data/clips.json').read_bytes(),'Unexpected deployed index';index=json.loads(raw)
    assert len(index['clips'])==813
    def verify(c):
        assert c['motionMethod']=='gvhmr-ground-motionbert-v1'
        results={}
        for key in ['pose3d','motionQuality','poseRefined']:
            data=fetch('data/'+c[key]);expected=sha(SITE/'data'/c[key]);actual=hashlib.sha256(data).hexdigest()
            assert actual==expected,(c['id'],key);results[key]=actual
        return dict(id=c['id'],hashes=results)
    results=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(verify,c) for c in index['clips']]):
            results.append(f.result())
            if len(results)%100==0:print('LIVE VERIFIED',len(results),'/ 813',flush=True)
    report=dict(base=base,clips=len(results),players=len({c['playerKey'] for c in index['clips']}),
        index_sha256=hashlib.sha256(raw).hexdigest(),all_pose_quality_and_final_2d_hashes_match=True,
        verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),results=sorted(results,key=lambda r:r['id']),browser_verification_still_required=True)
    save(ROOT/'live_asset_verification.json',report);print('ALL 813 COMBINED LIVE ASSETS VERIFIED',flush=True)

if __name__=='__main__':main()
