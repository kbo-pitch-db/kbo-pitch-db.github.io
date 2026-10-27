"""Write observation-based ablation and per-clip outcomes; no invented 3D GT."""
from common import *
from arm_cocking_refine import FOLDER,deployment_guard
import html,argparse

def main():
 global FOLDER
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2','v3'],default='v3');args=ap.parse_args();version=args.version
 if version!='v1':FOLDER=__import__('arm_cocking_'+version).FOLDER
 choice=json.loads((FOLDER/'selection_rule.json').read_text())['selected_variant'];site=CODE/'studies'/dict(v1='arm-cocking-v1-20261028',v2='arm-cocking-v2-20261028',v3='arm-cocking-20261028')[version];all_rows=[]
 for cohort in ['tune','confirmation']:all_rows+=json.loads((FOLDER/f'evaluation_{cohort}.json').read_text())
 def delta(r,key):
  b=next(v for v in all_rows if v['id']==r['id'] and v['variant']=='current_GVHMR');return None if r[key] is None or b[key] is None else r[key]-b[key]
 summary=[]
 for cohort in ['tune','confirmation']:
  for variant in dict.fromkeys(r['variant'] for r in all_rows):
   rr=[r for r in all_rows if r['split']==cohort and r['variant']==variant];record=dict(split=cohort,variant=variant,clips=len(rr),source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr))
   for key in ['all_pitch_wrist_p95_px','all_pitch_elbow_p95_px','holdout_pitch_wrist_p95_px','separate_rtmw_pitch_wrist_p95_px','model_forearm_posterior_peak_leg','arm_projected_speed_ratio','arm_change_p95_leg']:
    values=[r[key] for r in rr if r[key] is not None];diff=[delta(r,key) for r in rr if delta(r,key) is not None];record['median_'+key]=float(np.median(values)) if values else None;record['median_delta_'+key]=float(np.median(diff)) if diff else None
   summary.append(record)
 chosen=[r for r in all_rows if r['variant']==choice];per=[]
 for r in chosen:
  wrist=delta(r,'all_pitch_wrist_p95_px');alt=delta(r,'separate_rtmw_pitch_wrist_p95_px');elbow=delta(r,'all_pitch_elbow_p95_px')
  control_name=('raw_ray_gv' if choice.startswith('raw_ray') else 'gv_ray_perframe' if 'perframe' in choice else 'gv_ray_spline') if version=='v3' else ('gv_local_perframe' if 'perframe' in choice else 'gv_local_spline') if version=='v2' else ('gv_axial_source_refit' if 'axial' in choice else 'gv_forearm_source_refit' if 'forearm' in choice else 'gv_source_refit');control=next(q for q in all_rows if q['id']==r['id'] and q['variant']==control_name)
  per.append(dict(image_only_control_variant=control_name,wrist_p95_vs_image_only_control_px=None if r['all_pitch_wrist_p95_px'] is None or control['all_pitch_wrist_p95_px'] is None else r['all_pitch_wrist_p95_px']-control['all_pitch_wrist_p95_px'],id=r['id'],split=r['split'],variant=choice,source_gate=r['source_and_preservation_gate'],gate_reasons=r['gate_reasons'],wrist_p95_delta_px=wrist,elbow_p95_delta_px=elbow,separate_rtmw_wrist_p95_delta_px=alt,posterior_proxy_delta_leg=delta(r,'model_forearm_posterior_peak_leg'),arm_change_p95_leg=r['arm_change_p95_leg'],lower_body_max_change_model=r['lower_body_max_change_model'],native_stride_change_max_leg=r['native_stride_change_max_leg'],observed_wrist_p95_lower_by_2px=wrist is not None and wrist<=-2,separate_detector_wrist_p95_lower_by_2px=alt is not None and alt<=-2,actual_3d_accuracy='unvalidated'))
 report=dict(experiment_version=version,selected_variant=choice,main_release_changed=False,cohorts={c:dict(clips=sum(r['split']==c for r in chosen),source_and_preservation_pass=sum(r['split']==c and r['source_and_preservation_gate'] for r in chosen),original_wrist_p95_lower_by_2px=sum(r['split']==c and r['observed_wrist_p95_lower_by_2px'] for r in per),separate_detector_wrist_p95_lower_by_2px=sum(r['split']==c and r['separate_detector_wrist_p95_lower_by_2px'] for r in per)) for c in ['tune','confirmation']},ablations=summary,clips=per,actual_3d_accuracy_validated=False,interpretation='2D correspondence and motion preservation, not measured 3D cocking/stride accuracy. Held-out observations were excluded only from arm refinement; initial models saw them. RTMW is another model on the same RGB, not GT.',solver_seconds=sum(json.loads(p.read_text())['seconds'] for p in (FOLDER/'reports').glob('*.json')))
 save(FOLDER/'analysis.json',report);save(site/'analysis.json',report);csvwrite(site/'ablation_summary.csv',summary);csvwrite(site/'selected_clip_results.csv',per)
 def value(v):return 'unknown' if v is None else f'{v:.3f}' if isinstance(v,(float,np.floating)) else html.escape(str(v))
 heads=['split','variant','clips','source_and_preservation_pass','median_delta_all_pitch_wrist_p95_px','median_delta_all_pitch_elbow_p95_px','median_delta_separate_rtmw_pitch_wrist_p95_px','median_delta_model_forearm_posterior_peak_leg']
 table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in heads)+'</tr>'+''.join('<tr>'+''.join('<td>'+value(r[k])+'</td>' for k in heads)+'</tr>' for r in summary)+'</table>'
 links=''.join(f'<li><a href="./?clip={html.escape(r["id"])}">{html.escape(r["id"])} · {r["split"]}</a> — 손목 p95 변화 {value(r["wrist_p95_delta_px"])} px / 별도 RTMW {value(r["separate_rtmw_wrist_p95_delta_px"])} px / 보존 검사 {r["source_gate"]}</li>' for r in per)
 text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow,noarchive"><title>팔 결합 실험 분석</title><style>body{{font:15px/1.65 system-ui;margin:30px;color:#243a2b;max-width:1450px}}table{{border-collapse:collapse;display:block;overflow:auto;font-size:12px}}td,th{{padding:7px;border:1px solid #ccc}}a{{color:#25643b}}code{{background:#edf0e8}}</style><h1>GVHMR + MotionBERT 팔 결합 실험 분석 · {version}</h1><p>튜닝 6개에서 고정한 후보: <code>{html.escape(choice)}</code>. 선수·원본 영상이 다른 6개에 같은 선택을 적용했습니다. 메인 813개 결과에는 결합 실험을 적용하지 않았습니다.</p><p>메인 GVHMR의 체형·하체 로컬 회전·카메라·골반 전역 궤적과 보폭은 고정했습니다. 상완·팔꿈치만 수정하며 손목 로컬 회전은 유지했습니다. MotionBERT의 17관절 lifting으로 관측되지 않는 실제 손목/손바닥 회전의 개선은 주장하지 않습니다.</p><p>원본 손목/팔꿈치 재투영과 이번 fitting에서 제외한 매 5번째 프레임, fitting에 넣지 않은 RTMW384의 별도 관측을 비교했습니다. 초기 추정기는 전체 프레임을 보았습니다. 모든 수치는 관측 정합성 또는 모델 진단이며 KBO 3D GT가 아닙니다. 음의 px 변화는 2D 오차 감소입니다. 후방 위치 증가는 정확도 개선 건수에 포함하지 않습니다.</p><p>선택 기준은 손을 가장 뒤로 보내는 후보가 아닙니다. 기본 3D와 함께 MotionBERT 없이 같은 팔 자유도만 2D 재적합한 대조군도 포함합니다. 선택 후보는 MotionBERT 설정들 중 골랐으며, 대조군보다 우수하다는 뜻이 아닙니다. 원본 정합·동작 보존 조건을 적용하고 튜닝의 제외 프레임 손목/팔꿈치 오차와 관측 속도 타이밍 차이로 설정을 고정했습니다. 확인 클립은 선택에 쓰지 않았으며, 이전 연구에 쓰인 자료라는 한계가 있습니다.</p><h2>후보별 효과</h2>{table}<h2>동일 클립 전후 비교</h2><ul>{links}</ul><p><a href="./">동기 비교 프리뷰</a> · <a href="results.csv">전체 CSV</a> · <a href="ablation_summary.csv">요약 CSV</a> · <a href="analysis.json">분석 JSON</a> · <a href="selection.json">동결 선택 규칙</a></p><p>같은 RGB에 대한 두 검출 모델의 일치는 실제 깊이 정답이 아닙니다. 원본에서 가려진 팔의 앞뒤, 외회전, 손목 각도는 여전히 불확실합니다. 육안 검수 완료로 표시하지 않습니다.</p></html>'''
 (site/'report.html').write_text(text);print(json.dumps({k:v for k,v in report.items() if k not in ['ablations','clips']},indent=2))

if __name__=='__main__':main()
