"""Local-only static research server with byte ranges for precise video seeking."""
from common import OUT
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
import argparse,os,re
class Handler(SimpleHTTPRequestHandler):
 def send_head(self):
  self.byte_range=None;path=self.translate_path(self.path)
  if not os.path.isfile(path):return super().send_head()
  f=open(path,'rb');size=os.fstat(f.fileno()).st_size;value=self.headers.get('Range','');match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
  if match:
   a,b=match.groups();start=int(a) if a else max(0,size-int(b));end=min(size-1,int(b) if b and a else size-1)
   if start>end or start>=size:self.send_error(416);f.close();return None
   self.byte_range=(start,end);f.seek(start);self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{end}/{size}');length=end-start+1
  else:self.send_response(200);length=size
  self.send_header('Content-Type',self.guess_type(path));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(length));self.end_headers();return f
 def copyfile(self,source,outputfile):
  if self.byte_range is None:return super().copyfile(source,outputfile)
  count=self.byte_range[1]-self.byte_range[0]+1
  while count>0:
   chunk=source.read(min(count,1024*1024))
   if not chunk:break
   outputfile.write(chunk);count-=len(chunk)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8775);args=ap.parse_args();ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(OUT))).serve_forever()
