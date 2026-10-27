"""Versioned observation-driven study. Dataset and prior study are read-only."""
import os,sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np
CODE=Path(__file__).resolve().parents[2]
OUT=CODE.parent
WORKSPACE=OUT.parent.parent
DATA=WORKSPACE/'dataset'
PREVIOUS=WORKSPACE/'research/pitch_reconstruction_20261027'
sys.path.insert(0,str(CODE/'scripts'))

def rows(name):
 with (DATA/name).open() as f:return {x['dataset_clip_id']:x for x in csv.DictReader(f)}
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2));tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def atomic_npz(path,**values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**values);tmp.replace(path)
def csvwrite(path,records):
 keys=list(dict.fromkeys(k for r in records for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list)) else clean(v) for k,v in r.items()} for r in records)
def scores(h):
 s=h['scores'].copy();bad=h['missing']|~np.isfinite(h['keypoints']).all(-1)
 for k in ['interpolated','outlier']:bad|=h.get(k,np.zeros_like(bad))
 s[bad]=0;return s
def sample(split='tune',small=True):
 r=json.loads((OUT/('subset.json' if small else 'sample48.json')).read_text());return [x for x in r if split=='all' or x['split']==split]
def init_inputs(cid):
 r=rows('motion_wham.csv')[cid];c=rows('clips.csv')[cid]
 source=next(s for s in csv.DictReader((DATA/'sources.csv').open()) if s['source_id']==c['source_id'])
 return r,c,source,load(DATA/r['pose_2d_refined']),load(DATA/r['prior_3d'])
