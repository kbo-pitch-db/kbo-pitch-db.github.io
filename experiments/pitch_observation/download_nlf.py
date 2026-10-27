"""Bounded resumable public release download, exact official SHA-256 check."""
from common import *
import requests,concurrent.futures

def main():
 url='https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_s_multi_0.2.2.patch2.torchscript';size=352815353;expected='61a2c684eba6b12bd5eb70560855e37af2d1c6322b8fcb4a9f5e096932591b92';root=OUT/'downloads/nlf_chunks';root.mkdir(parents=True,exist_ok=True);start=time.monotonic();chunk=16*1024*1024
 response=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(15,30));resolved=response.url;status=response.status_code;response.close()
 if status!=206:raise RuntimeError('Range download unavailable; bounded attempt ends: HTTP '+str(status))
 def fetch(i):
  a=i*chunk;b=min(size,(i+1)*chunk)-1;p=root/f'{i:03d}.bin'
  if p.exists() and p.stat().st_size==b-a+1:return i
  tmp=p.with_suffix('.part');resume=tmp.stat().st_size if tmp.exists() else 0
  if resume>b-a+1:tmp.unlink();resume=0
  if resume==b-a+1:tmp.replace(p);return i
  if time.monotonic()-start>1200:raise TimeoutError('background download budget exhausted')
  response=requests.get(resolved,headers={'Range':f'bytes={a+resume}-{b}'},stream=True,timeout=(15,90))
  if response.status_code!=206:raise RuntimeError('Server ignored byte range '+str(i))
  with tmp.open('ab' if resume else 'wb') as f:
   for value in response.iter_content(1024*1024):
    if time.monotonic()-start>1200:raise TimeoutError('NLF background-only attempt twenty minutes')
    f.write(value)
  response.close()
  if tmp.stat().st_size!=b-a+1:raise RuntimeError('incomplete range '+str(i))
  tmp.replace(p);print('NLF CHUNK',i,flush=True);return i
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(fetch,range((size+chunk-1)//chunk)))
 dest=WORKSPACE/'models/nlf_s_multi_0.2.2.patch2.torchscript';tmp=dest.with_suffix('.verified.part')
 with tmp.open('wb') as f:
  for i in range((size+chunk-1)//chunk):
   with (root/f'{i:03d}.bin').open('rb') as g:
    for value in iter(lambda:g.read(4*1024*1024),b''):f.write(value)
 assert sha(tmp)==expected,'Official GitHub release digest mismatch';tmp.replace(dest);save(OUT/'downloads/nlf_provenance.json',dict(url=url,official_release_sha256=expected,size=size,seconds=time.monotonic()-start));print('VERIFIED NLF',size,flush=True)
if __name__=='__main__':main()
