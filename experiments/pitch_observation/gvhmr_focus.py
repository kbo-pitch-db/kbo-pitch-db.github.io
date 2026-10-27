"""Post-hoc user-feedback review: preserve sealed outputs, expose GVHMR stages.

No new fit, no promotion to validated/public status, no change to the prior
selector or evaluation split. A visual preference is a new research hypothesis.
"""
from common import *
from candidates import CC,norm
from build_viewer import HTML

def model(z):
 xyz=z['keypoints'][:,CC];relative,leg=norm(xyz);g=xyz+z['translation'][:,None];g=(g-g[0,:2].mean(0))/leg
 return dict(relative=relative,global_fixed=g,projected=z['projected'][:,CC],global_validated=False)

def main():
 folder=OUT/'gvhmr_review';folder.mkdir(exist_ok=True);catalog=json.loads((OUT/'viewer_catalog.json').read_text());lineages=[]
 for rec in catalog:
  cid=rec['id'];payload=json.loads((OUT/'viewer_data'/f'{cid}.json').read_text());directory='fits_strong' if rec['split']=='tune' else 'fits_final';models={'current_actual':payload['models']['current_actual']};lineage=[]
  for f in [4,1,12]:
   path=OUT/directory/f'gvhmr_temporal_f{f}'/f'{cid}.npz';z=load(path);models[f'gvhmr_f{f}']=model(z);lineage.append(dict(label=f'gvhmr_f{f}',path=str(path.relative_to(OUT)),stage='camera60+pose160+temporal80',focal_seed_factor=f,fitted_focal_px=float(z['focal']),absolute_camera_calibrated=False))
  for label,stage in [('GVHMR f4 · camera only','camera'),('GVHMR f4 · pose fit','pose')]:
   path=OUT/directory/f'gvhmr_{stage}_f4'/f'{cid}.npz';z=load(path);models[label]=model(z);lineage.append(dict(label=label,path=str(path.relative_to(OUT)),stage=stage))
  raw=load(OUT/'adapters/gvhmr_f4'/f'{cid}.npz');rawp=norm(raw['keypoints'][:,CC])[0];models['GVHMR raw f4 · SMPL adapter']=dict(relative=rawp,global_fixed=None,projected=None,global_validated=False)
  camera=load(OUT/directory/'gvhmr_camera_f4'/f'{cid}.npz');difference=float(np.max(abs(rawp-norm(camera['keypoints'][:,CC])[0])));assert difference<2e-4,(cid,difference)
  weak=OUT/'fits_preserve/gvhmr_temporal_f4'/f'{cid}.npz'
  if weak.exists():models['GVHMR f4 · weak depth prior .002']=model(load(weak))
  for label in ['WHAM_raw_f4','NLF RGB body + temporal MotionBERT arms','NLF RGB + fitting (exploratory)','NLF fixed native shape · raw RGB','NLF native shape + 2D fitting','NLF native shape + RGB contour fitting','NLF raw native joint centers (SMPL24 mapping)']:
   if label in payload['models']:models[label]=payload['models'][label]
  payload['models']=models;payload['extent']=max([3.2]+[float(np.nanmax(abs(np.asarray(m['global_fixed'],dtype=float)))) for m in models.values() if m['global_fixed'] is not None])*1.08
  payload['focus_lineage']=dict(displayed_gvhmr_f4='Official GVHMR learned network -> approximate SMPL shared-rotation adapter -> camera/pose/part-temporal fitting; not full official native SMPL-X output',f4='Focal seed 4 x full-image diagonal, not a camera-view category or measured intrinsics',smpl_shape='Sequence-fixed median WHAM SMPL betas for controlled initializer comparison; GVHMR SMPL-X betas not copied into SMPL',raw_pose_vs_camera_only_max_difference_leg=difference,candidates=lineage,scope='Post-hoc feedback analysis; previous frozen evaluation and selection unchanged')
  save(folder/'data'/f'{cid}.json',payload);lineages.append(dict(id=cid,**payload['focus_lineage']))
 save(folder/'catalog.json',catalog);save(folder/'lineage.json',lineages)
 # Compare only the same clips, without mixing 36-clip and 16-clip summaries.
 rr=json.loads((OUT/'evaluation_component_tune.json').read_text())+json.loads((OUT/'evaluation_final_test.json').read_text());gv={r['id']:r for r in rr if r['candidate'] in ['fits_strong/gvhmr_temporal_f4','fits_final/gvhmr_temporal_f4']};hy={r['id']:r for r in json.loads((OUT/'nlf_pose_source_evaluation.json').read_text()) if r['candidate']=='NLF_hybrid_fitted'};ids=sorted(set(gv)&set(hy));paired=[];summary=[]
 for key in ['common12_source_reprojection_p50_px','wrist_source_reprojection_p95_px','elbow_source_reprojection_p95_px','limb_reversal_count','uncalibrated_root_speed_p95_leg_s','arm0_projection_speed_ratio','arm1_projection_speed_ratio','arm0_projection_peak_delay_s','arm1_projection_peak_delay_s']:
  values=[(i,gv[i].get(key),hy[i].get(key)) for i in ids if gv[i].get(key) is not None and hy[i].get(key) is not None];summary.append(dict(metric=key,n=len(values),gvhmr_f4_clip_median=float(np.median([a for i,a,b in values])),hybrid_clip_median=float(np.median([b for i,a,b in values])),scope='Post-hoc observation/consistency metric; not KBO measured 3D accuracy'))
 for cid in ids:paired.append(dict(id=cid,gvhmr=gv[cid],hybrid=hy[cid]))
 save(folder/'paired.json',paired);save(folder/'summary.json',summary);csvwrite(folder/'summary.csv',summary)
 save(folder/'feedback.json',dict(user_statement='눈으로 보기에는 GVHMR_F4 결과가 제일 좋아보임',received_date='2026-10-28',scope='Qualitative model preference; no explicit clip/frame or full validation scope provided',research_priority='GVHMR f4 as reference candidate for next experiments',existing_human_review_status_changed=False,previous_frozen_selection_changed=False,public_deployment=False))
 html=HTML.replace('KBO 원본과 3D 후보 비교 — 로컬 연구 결과','GVHMR f4 중심 비교 — 원본 · raw · fitting · 카메라 가설')
 html=html.replace('viewer_catalog.json','gvhmr_review/catalog.json').replace('viewer_data/${id}.json','gvhmr_review/data/${id}.json')
 html=html.replace('<label>비교 후보 <select id="candidate">','<label>기준 후보 <select id="reference"></select></label><label>비교 후보 <select id="candidate">')
 html=html.replace('old=data.models.current_actual','old=data.models[$(\'reference\').value||\'current_actual\']')
 html=html.replace("$('candidate').replaceChildren();", "$('reference').replaceChildren();for(let name of Object.keys(data.models))$('reference').add(new Option(name,name));$('reference').value='current_actual';$('candidate').replaceChildren();")
 html=html.replace('requested:data.selection.reference','requested:data.selection.reference')
 html=html.replace('requested:data.selection.reference;',"requested:'gvhmr_f4';")
 html=html.replace("['mode','candidate','obs'", "['reference','mode','candidate','obs'")
 html=html.replace('기존 결과 <span','기준 후보 <span').replace('기존 투영 주황','기준 투영 주황')
 html=html.replace('selection:data.selection,source_note:', 'selection:data.selection,focus_lineage:data.focus_lineage,source_note:')
 html=html.replace('`${data.player} · ${data.split} · 기존 ', '`${data.player} · ${data.split} · 원래 동결 선택 기록: 기존 ')
 html=html.replace('<div class="notice">공개 3D GT', '<div class="notice"><b>GVHMR f4를 우선 검토하는 화면입니다.</b> F4는 원본 영상 대각선 4배의 초점거리 초기값입니다. 현재 gvhmr_f4는 SMPL adapter와 단계별 fitting을 거친 결과입니다. raw와 camera-only를 기준 후보에서 비교할 수 있습니다. 사용자의 모델 선호를 기록했으며 이전 동결 판정은 유지합니다.<br>공개 3D GT')
 (OUT/'gvhmr_review.html').write_text(html)
 save(folder/'manifest.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),clips=len(catalog),paired_clips=len(ids),input_pose_files_modified=False,original_viewer_modified=False,summary='summary.csv',lineage='lineage.json',page='../gvhmr_review.html'))
 print('GVHMR FOCUS',len(catalog),'clips;',len(ids),'paired clips; no fitting or frozen result changes')
if __name__=='__main__':main()
