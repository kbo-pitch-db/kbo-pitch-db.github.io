import sys,json,gzip,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parent));from pitch_events.contact_model import features,probabilities,decode_contact
R=Path(__file__).resolve().parents[1];P=R.parent/'research/event_detector_20261029';reviews=json.load(open(R/'data/metrics/source_reviews_20261029.json'))['clips'];samples=[]
for r in reviews:
 if not r.get('fc') or not r.get('br'):continue
 z=np.load(P/'foot_features'/f"{r['clipId']}.npz");u=z['frames'];lo=r['pkh'][0] if r.get('pkh') else max(0,r['fc'][0]-18);hi=r['br'][0];sel=(u>=lo)&(u<=hi)&((u<r['fc'][0])|(u>r['fc'][1]));x=features(z['features'][sel]);y=(u[sel]>r['fc'][1]).astype(int)
 if len(np.unique(y))<2:continue
 samples.append(dict(id=r['clipId'],group=r['clipId'].split('_')[0],x=x,y=y,review=r))
def train(s,C):
 x=np.concatenate([v['x'] for v in s]);y=np.concatenate([v['y'] for v in s]);w=np.concatenate([np.full(len(v['y']),len(x)/(len(s)*len(v['y']))) for v in s]);clf=LogisticRegression(C=C,max_iter=2000,class_weight='balanced',solver='liblinear').fit(x,y,sample_weight=w);return dict(coef=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]))
def predict(model,r):
 cid=r['clipId'];z=np.load(P/'foot_features'/f'{cid}.npz');u=z['frames'];a=json.load(gzip.open(R/'data/biomechanics/analysis'/f'{cid}.json.gz'));t=np.array(a['frameTimes']);t-=t[0];return decode_contact(u,probabilities(z['features'],model),t,r['br'],r.get('pkh'))[0]
for C in [.1,1,10]:
 results=[]
 for group in sorted(set(v['group'] for v in samples)):
  model=train([v for v in samples if v['group']!=group],C)
  for v in samples:
   if v['group']!=group:continue
   pred=predict(model,v['review']);fps=json.load(gzip.open(R/'data/biomechanics/analysis'/f"{v['id']}.json.gz"))['cadence']['fileFps'];error=abs(np.mean(pred)-np.mean(v['review']['fc']))/fps*1000 if pred else None;results.append(error)
 print('LOSO C',C,'detected',sum(v is not None for v in results),'of',len(results),'median',np.median([v for v in results if v is not None]),'errors',results,flush=True)
model=train(samples,1);model.update(encoder='DINOv2 ViT-S/14',encoderRevision='e1277af2ba9496fbadf7aec6eba56e8d882d1e35',version='visual-foot-contact-20261029-v1',trainingClipIds=[v['id'] for v in samples],trainingSourceGroups=sorted(set(v['group'] for v in samples)),reference='Provisional source-image brackets; not independent human or force-plate ground truth.',featureDefinition='mean of individual shoe encodings, absolute difference, union foot-context encoding; multiplied by 10')
(P/'contact_model.json').write_text(json.dumps(model));print('model frozen',hashlib.sha256((P/'contact_model.json').read_bytes()).hexdigest())
