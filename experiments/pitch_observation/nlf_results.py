"""Model/shape/fitting ablation, including a fresh four-clip confirmation."""
from common import *
from candidates import CC,norm
from evaluate import evaluate_record,current_projection
from events import infer
from axis_order import setup,metrics

def main():
 source=[];body=[];mesh=[]
 for filename,label in [('nlf_mesh_rgb_tune_offset0.json','tune12'),('nlf_mesh_v2_test.json','diagnostic_test_first4'),('nlf_mesh_rgb_test_offset4.json','fresh_confirmation_next4'),('nlf_mesh_hybrid_tune_offset0.json','hybrid_tune12'),('nlf_mesh_hybrid_test_offset12.json','hybrid_confirmation_next4')]:
  path=OUT/filename
  if not path.exists():continue
  rr=json.loads(path.read_text());mesh+=rr
  for model in dict.fromkeys(r['candidate'] for r in rr):
   clips=[];deltas=[]
   for cid in dict.fromkeys(r['id'] for r in rr):
    a=[r['silhouette_iou'] for r in rr if r['id']==cid and r['candidate']==model and r['observation_split']=='withheld'];old=[r['silhouette_iou'] for r in rr if r['id']==cid and r['candidate']=='current_actual' and r['observation_split']=='withheld']
    if a and len(a)==len(old):clips.append(float(np.mean(a)));deltas.append(float(np.mean(a)-np.mean(old)))
   body.append(dict(evaluation_group=label,candidate=model,clips=len(clips),withheld_mask_iou_clip_mean=float(np.mean(clips)) if clips else None,delta_vs_legacy_clip_mean=float(np.mean(deltas)) if deltas else None,clips_delta_above_01=sum(d>.01 for d in deltas),clips_delta_below_minus01=sum(d<-.01 for d in deltas),claim='Source appearance agreement to SAM masks, not true joint depth accuracy'))
 for rec in sample('all',False):
  cid=rec['id'];n=OUT/'nlf_native_fixed'/f'{cid}.npz'
  if not n.exists():continue
  r,c,s,h,w=init_inputs(cid);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];ev=infer(h,bg);old=load(DATA/rows('motion_wham_final_scene.csv')[cid]['pose_3d_scene']);p,leg=norm((old['keypoints']@old['scene_rotation'])[:,CC]);entries=[('current_actual',dict(common=p,uv=current_projection(old,c)[:,CC],leg=leg,global_root=old['root_translation']@old['scene_rotation'],native=old['anatomical_keypoints'],z=old,kind='legacy_actual'))]
  for tag,path in [('NLF_native_fixed',n),('NLF_WHAM_shape_fitted',OUT/('fits_nlf' if rec['split']=='tune' else 'fits_nlf_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'),('NLF_native_shape_fitted',OUT/('fits_nlf_native_shape' if rec['split']=='tune' else 'fits_nlf_native_shape_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz'),('NLF_hybrid_fitted',OUT/('fits_nlf_hybrid' if rec['split']=='tune' else 'fits_nlf_hybrid_test')/'nlf_mb_hybrid_temporal_f4'/f'{cid}.npz'),('NLF_RGB_contour_fitted',OUT/('fits_nlf_rgb' if rec['split']=='tune' else 'fits_nlf_rgb_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz')]:
   if not path.exists():continue
   z=load(path);p,leg=norm(z['keypoints'][:,CC]);entries.append((tag,dict(common=p,uv=z['projected'][:,CC],leg=leg,global_root=z['translation'],native=z['anatomical_keypoints'],z=z,kind='fit')))
  for tag,v in entries:
   v['name']=tag;q=evaluate_record(v,h,cid,rec['split'],[],ev);q['field_order']=metrics(v['common'],setup(cid,h));source.append(q)
 save(OUT/'nlf_ablation_summary.json',body);csvwrite(OUT/'nlf_ablation_summary.csv',body);save(OUT/'nlf_pose_source_evaluation.json',source);csvwrite(OUT/'nlf_pose_source_evaluation.csv',source);save(OUT/'nlf_all_mesh_observations.json',mesh)
 print(json.dumps(clean(body),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
