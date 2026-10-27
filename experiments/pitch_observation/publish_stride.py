"""Package only numerical stride-review assets for the existing static site.

No original broadcast pixels, thumbnails, video, masks, learned weights, licensed
body-model assets or local paths are copied. Original 813 public poses stay intact.
"""
from common import *
import argparse,copy

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(value),ensure_ascii=False,separators=(',',':')))

def rounded(x):
 if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
 if isinstance(x,list):return [rounded(v) for v in x]
 if isinstance(x,float):return round(x,4) if np.isfinite(x) else None
 return x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--site',default=str(CODE));a=ap.parse_args();site=Path(a.site);dest=site/'studies/stride-20261028';dest.mkdir(parents=True,exist_ok=True);source=OUT/'stride_review';catalog=json.loads((source/'catalog.json').read_text());index=[]
 for rec in catalog:
  cid=rec['id'];data=json.loads((source/'data'/f'{cid}.json').read_text());r,c,s,h,w=init_inputs(cid);url='https://www.youtube.com/watch?v='+s['provider_video_id']+'&t='+str(int(float(h['source_time'][0])));models=copy.deepcopy(data['models'])
  for obsolete in ['WHAM raw + fitting','NLF WHAM-shape fitting']:models.pop(obsolete,None)
  for name,m in models.items():m.pop('source',None)
  public=dict(id=cid,player=data['player'],split=data['split'],times=data['times'],source_times=data['source_times'],observations=data['halpe'],scores=data['scores'],crop_box=data['crop_box'],models=models,extent=data['extent'],phase=data['stride_phase'],sourceUrl=url,default_model='gvhmr_f4',legacy_status=data['legacy_status'],legacy_reasons=data['legacy_reasons'],media_policy='numerical_pose_only',human_review_complete=False,absolute_stride_validated=False)
  dump(dest/'viewer'/f'{cid}.json',rounded(public));index.append(rec)
 dump(dest/'viewer/catalog.json',index)
 rows_=json.loads((source/'metrics.json').read_text());dump(dest/'stride_metrics.json',rows_);csvwrite(dest/'stride_metrics.csv',rows_)
 # All models in each aggregate share exactly the same clips and phase windows.
 required=['current_actual','WHAM raw camera only','GVHMR raw f4 SMPL adapter','gvhmr_f4'];by={(r['id'],r['candidate']):r for r in rows_};ids=[r['id'] for r in catalog if all(by[(r['id'],n)]['plant_interankle_3d_leg_range'] is not None for n in required)];summary=[]
 for name in required+['WHAM raw + fitting','GVHMR f4 pose fit']:
  values=[np.mean(by[(cid,name)]['plant_interankle_3d_leg_range']) for cid in ids];summary.append(dict(candidate=name,clips=len(values),plant_native_interankle_interval_midpoint_leg_clip_median=float(np.median(values)),semantics='Same-frame native ankle separation, full 3D, modeled-leg normalized. NOT measured pitching stride. Landing interval inferred from 2D.'))
 nlf_names=['current_actual','gvhmr_f4','NLF native shape + fitting'];nlfids=[cid for cid in ids if all((cid,n) in by and by[(cid,n)]['plant_interankle_3d_leg_range'] is not None for n in nlf_names)]
 nlf_summary=[dict(candidate=n,clips=len(nlfids),plant_native_interankle_interval_midpoint_leg_clip_median=float(np.median([np.mean(by[(cid,n)]['plant_interankle_3d_leg_range']) for cid in nlfids]))) for n in nlf_names]
 dump(dest/'summary.json',dict(default_model='gvhmr_f4',clips=len(catalog),nlf_available=sum('NLF native shape + fitting' in r['models'] for r in catalog),same_clips_main=summary,same_clips_nlf=nlf_summary,classification_only_recoveries=0,measured_kbo_3d_accuracy=None,production_813_replaced=0))
 dump(dest/'decision.json',json.loads((source/'decision.json').read_text()));dump(dest/'sample48.json',json.loads((OUT/'sample48.json').read_text()));dump(dest/'method_lineage.json',json.loads((source/'lineage.json').read_text()));dump(dest/'validation.json',json.loads((OUT/'validation.json').read_text()));dump(dest/'input_integrity.json',json.loads((OUT/'input_integrity.json').read_text()))
 # Numeric provenance only; no datasets, checkpoints or licensed mesh files.
 provenance=json.loads((OUT/'model_provenance.json').read_text());provenance.pop('platform',None);provenance.pop('python',None);dump(dest/'model_provenance.json',provenance)
 from html import escape
 table=''.join('<tr><td>'+escape(r['candidate'])+'</td><td>'+str(r['clips'])+'</td><td>'+format(r['plant_native_interankle_interval_midpoint_leg_clip_median'],'.3f')+'</td></tr>' for r in summary)
 ntable=''.join('<tr><td>'+escape(r['candidate'])+'</td><td>'+str(r['clips'])+'</td><td>'+format(r['plant_native_interankle_interval_midpoint_leg_clip_median'],'.3f')+'</td></tr>' for r in nlf_summary)
 report='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>보폭 중심 모델 선택 근거</title><style>body{font:16px/1.8 system-ui;max-width:950px;margin:32px auto;padding:0 18px;color:#172b3d}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbc8d5}a{color:#006095}</style><h1>보폭 중심 선택: GVHMR f4 기본 · NLF 비교 유지</h1><p><a href="comparison.html">비교기 열기</a> · <a href="../../">전체 아카이브</a></p><p>기존 모션에서 주로 문제가 된 짧은 보폭을 기준으로 독립 초기 추정과 fitting을 분리했다. 원본에 더 부합한다는 검토 의견과 초기값 보존 결과를 반영해 GVHMR f4를 기본 후보로 골랐다. NLF도 합리적인 대안으로 남겼다. 이 선택은 사후 연구 판단이며 새로운 블라인드 3D 정확도 검증을 뜻하지 않는다.</p><h2>초기값부터 달랐던 발 벌어짐</h2><p>같은 20개 클립, 같은 2D 기반 착지 구간에서 native 발목 사이 전체 3D 거리의 구간 최소·최대 중간값을 구하고 클립 중앙값을 집계했다. 단위는 모델 다리 길이다. 실제 투구판–착지 보폭과 구분한다.</p><table><tr><th>단계</th><th>클립</th><th>발목 간격 / leg</th></tr>'''+table+'''</table><p>GVHMR의 발 벌어짐은 초기 추정에 이미 존재하고 fitting 뒤에도 대부분 유지됐다. 평균 보폭에 맞춘 강제 확대는 하지 않았다. WHAM 계열 초기값이 짧은 사례에서는 접지 고정만 푸는 것으로 충분하지 않았다. 큰 거리 자체를 정답으로 선택하는 규칙은 사용하지 않는다.</p><h2>NLF도 같은 클립에서 비교</h2><p>NLF가 있고 착지 구간이 있는 동일한 14개만 다시 묶어 비교했다. 위 20개 집계와 표본을 혼용하지 않는다.</p><table><tr><th>후보</th><th>클립</th><th>발목 간격 / leg</th></tr>'''+ntable+'''</table><p>NLF의 체형 고정 raw와 fitting, 팔 초기값 결합은 서로 다른 후보로 표시한다. 더 큰 수치가 실제 보폭 복원을 의미하지 않는다. 카메라·깊이·가림·접촉 시점에 대한 여러 해가 남는다.</p><h2>유지한 구분</h2><p>① 동일 프레임 발목 간격 ② 착지 구간의 간격 범위 ③ 레그리프트 부근 뒷발 발목 가설에서 앞발까지의 3D 변위 ④ 추정 투구 축 방향 가설별 변위 범위를 따로 저장했다. 초기 뒷발 지지 가설이 없거나 좌우투와 이벤트가 충돌하면 해당 값을 unknown으로 둔다. 공 관측 없이 릴리스를 확정하지 않는다.</p><p>기존 813개 결과와 최종 Halpe-26은 보존하고 36개 실험 표본을 별도 제공한다. 처리하지 않은 12개 튜닝 클립도 고정 48개 목록에 그대로 남겼다. 판정만 바꾼 복구는 0이며 사람 수동 검수 완료로 승격하지 않았다. 실제 KBO 3D GT, 절대 보폭, 전역 이동·접지 정확도는 미확정이다.</p><p><a href="stride_metrics.csv">클립·단계별 CSV</a> · <a href="decision.json">선택 근거 JSON</a> · <a href="model_provenance.json">공식 실행·자체 변형·미실행 사유</a> · <a href="REPRODUCE.txt">재현 명령</a> · <a href="NOTICES.txt">고지</a></p></html>'''
 (dest/'report.html').write_text(report)
 (dest/'REPRODUCE.txt').write_text('Run from the preserved research worktree, whose source files are committed in this repository.\ncd /home/eis/Workspace/Kichang/kbo-pitchers\nSTUDY=research/pitch_observation_20261028\nPY=$STUDY/env/bin/python\nEXP=$STUDY/code/experiments/pitch_observation\n$PY $EXP/stride_review.py\n$PY $EXP/publish_stride.py --site $STUDY/code\n\nRaw inference/fitting and fixed-split provenance: experiments/pitch_observation/README.txt and the local study REPRODUCE.txt.\nGVHMR network uses RTMPose/HMR2a inputs and an approximate SMPL shared-rotation adapter, followed by custom fitting. Full official SMPL-X postprocessing was not run.\nDefault focal seed f4 = 4 times full-frame image diagonal; final focal is optimized, not calibrated.\nNo stride stretching, population target, framewise floor correction or clipping-based recovery.\nOnly numerical poses/observations and source YouTube links are published. Original source media, pseudo-depth/masks, model weights and body-model assets remain local.\n')
 (dest/'NOTICES.txt').write_text('Research estimates, not measured KBO 3D GT or validated biomechanics.\nGVHMR, WHAM, MotionBERT and NLF retain their official model licenses.\nCustom MotionBERT adaptation used public OpenBiomechanics joint centers; its CC BY-NC-SA 4.0 plus professional-sports/financial-firm exclusions remain relevant.\nNo model weights, licensed SMPL assets, raw public mocap or original broadcast pixels are included here.\nSource video is linked externally, not self-hosted.\n')
 template=Path(__file__).parent/'stride_public.html';js=Path(__file__).parent/'stride_public.js';(dest/'index.html').write_text(template.read_text());(dest/'comparison.html').write_text(template.read_text());(dest/'comparison.js').write_text(js.read_text())
 files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='publication.json'}
 dump(dest/'publication.json',dict(media_policy='numerical_pose_only',original_broadcast_media_included=False,clips=len(catalog),default_model='gvhmr_f4',nlf_comparison_preserved=True,previous_production_poses_replaced=0,previous_quality_labels_changed=0,rendering_precision_decimals=4,files=files))
 print('PUBLIC STRIDE BUNDLE',len(catalog),'clips',sum((dest/p).stat().st_size for p in files),'bytes')
 assert sum((dest/p).stat().st_size for p in files)<55_000_000,'Static-hosting bundle budget exceeded'
if __name__=='__main__':main()
