"""Initializer-pool versus selection-rule ablation, fixed constants from seal."""
from common import *
from pool import POLICY

def choose(rows,field):
 old=next(r for r in rows if r['candidate']=='current_actual');eligible=[r for r in rows if r['finite_ratio']>=.95 and r['arm_guard_pass'] and r['wrist_p95_px'] is not None and r['wrist_p95_px']<=old['wrist_p95_px']+2 and r['selection_mask_iou'] is not None];best=max(eligible,key=lambda r:r['selection_score']) if eligible else old
 if best['candidate']!='current_actual' and best['selection_mask_iou']<old['selection_mask_iou']+.005:best=old
 if not field:return best
 oo=old['field_order']
 if oo.get('selection_frames',0)>=2 and (oo.get('selection_robust_reverse_fraction') or 0)>=.5:
  consistent=[q for q in eligible if q['candidate']!='current_actual' and q['field_order'].get('selection_frames',0)>=2 and q['field_order'].get('selection_robust_reverse_count',1)==0 and q['field_order'].get('selection_ambiguous_count',1)==0 and q['selection_mask_iou']>=old['selection_mask_iou']-.01]
  if consistent:best=max(consistent,key=lambda q:q['selection_score'])
 bo=best['field_order']
 if best['candidate']!='current_actual' and bo.get('selection_frames',0)>=2 and (bo.get('selection_robust_reverse_fraction') or 0)>=.5 and (oo.get('selection_robust_reverse_fraction') or 0)<.5:best=old
 return best

def main():
 output=[]
 for split in ['tune','test']:
  path=OUT/f'candidate_scores_strong_{split}.json'
  if not path.exists():continue
  rows_=json.loads(path.read_text())
  for cid in dict.fromkeys(r['id'] for r in rows_):
   allrows=[r for r in rows_ if r['id']==cid];old=next(r for r in allrows if r['candidate']=='current_actual')
   for pool in ['WHAM','WHAM_GV','WHAM_GV_head','WHAM_GV_head_block']:
    rr=[r for r in allrows if r['candidate']=='current_actual' or r['candidate'].startswith(('WHAM_','wham_')) or ('GV' in pool and r['candidate'].startswith('gvhmr')) or ('head' in pool and r['candidate'].startswith('mb_obp')) or ('block' in pool and r['candidate'].startswith('mb_block'))]
    for field in [False,True]:
     best=choose(rr,field);output.append(dict(id=cid,split=split,pool=pool,field_selector=field,selected=best['candidate'],old_withheld_mask=old['withheld_mask_iou'],new_withheld_mask=best['withheld_mask_iou'],withheld_mask_delta=best['withheld_mask_iou']-old['withheld_mask_iou'],old_withheld_reverse=old['field_order'].get('withheld_robust_reverse_count'),new_withheld_reverse=best['field_order'].get('withheld_robust_reverse_count')))
 save(OUT/'selector_ablation.json',output);csvwrite(OUT/'selector_ablation.csv',output)
if __name__=='__main__':main()
