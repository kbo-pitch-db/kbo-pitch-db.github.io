"""Equal-clip robust shape sharing within tune players, never pose sharing."""
from common import *

def main():
 from collections import defaultdict
 groups=defaultdict(list)
 for rec in sample('tune',False):groups[rec['player']].append(rec['id'])
 by_clip={};used=[]
 for player,ids in groups.items():
  if len(ids)<2 or player=='unknown':continue
  betas=[]
  for cid in ids:
   w=init_inputs(cid)[4];betas.append(np.nanmedian(w['smpl_betas'],0))
  shared=np.median(betas,axis=0)
  for cid in ids:by_clip[cid]=shared
  used.append(dict(player=player,clips=ids,beta=shared,per_clip_beta=betas,maximum_beta_delta=float(np.max(np.linalg.norm(np.array(betas)-shared,axis=-1)))))
 save(OUT/'shared_shape.json',dict(by_clip=by_clip,groups=used,semantics='Equal-weight median of per-sequence WHAM raw beta, tune athletes only. No known height, no shared pitch movement; clothing/model bias remains.',test_used=False))
if __name__=='__main__':main()
