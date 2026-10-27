"""Human-readable interpretation of actual iterations, controls and counterexamples."""
from common import *
from arm_cocking_refine import deployment_guard
from arm_cocking_v3 import FOLDER
import shutil,html

def main():
 deployment_guard();site=CODE/'studies/arm-cocking-20261028';all_versions={};flat=[]
 for v in ['v1','v2','v3']:
  folder=OUT/('arm_cocking_'+v);choice=json.loads((folder/'selection_rule.json').read_text())['selected_variant'];rows_=[]
  for split in ['tune','confirmation']:rows_+=json.loads((folder/f'evaluation_{split}.json').read_text())
  summaries=[]
  for split in ['tune','confirmation']:
   rr=[r for r in rows_ if r['split']==split and r['variant']==choice];record=dict(version=v,split=split,selected_variant=choice,clips=6,source_and_preservation_pass=sum(r['source_and_preservation_gate'] for r in rr))
   for k in ['all_pitch_wrist_p95_px','all_pitch_elbow_p95_px','holdout_pitch_wrist_p95_px','separate_rtmw_pitch_wrist_p95_px','arm_change_p95_leg']:
    vals=[r[k] for r in rr if r[k] is not None];record['median_'+k]=float(np.median(vals)) if vals else None
   summaries.append(record);flat.append(record)
  all_versions[v]=dict(selected=choice,rows=rows_,summary=summaries)
  d=site/'iterations'/v;d.mkdir(parents=True,exist_ok=True)
  for name in ['protocol.json','selection_rule.json','confirmation_summary.json','evaluation_tune.csv','evaluation_confirmation.csv']:
   shutil.copy2(folder/name,d/name)
 save(site/'iteration_summary.json',flat);csvwrite(site/'iteration_summary.csv',flat)
 rows_=all_versions['v3']['rows'];selected=all_versions['v3']['selected'];by={(r['id'],r['variant']):r for r in rows_};cases=[];sensitivity=[]
 for split in ['tune','confirmation']:
  for cid in json.loads((FOLDER/'protocol.json').read_text())[split]:
   b=by[cid,'current_GVHMR'];n=by[cid,selected];g=by[cid,'raw_ray_gv'];cases.append(dict(id=cid,split=split,wrist_before=b['all_pitch_wrist_p95_px'],wrist_after=n['all_pitch_wrist_p95_px'],wrist_gv_depth_control=g['all_pitch_wrist_p95_px'],holdout_before=b['holdout_pitch_wrist_p95_px'],holdout_after=n['holdout_pitch_wrist_p95_px'],posterior_proxy_before=b['model_forearm_posterior_peak_leg'],posterior_proxy_after=n['model_forearm_posterior_peak_leg'],arm_change_p95_leg=n['arm_change_p95_leg'],pass_frozen=n['source_and_preservation_gate'],reasons=n['gate_reasons']))
   report=json.loads((FOLDER/'reports'/f'{cid}.json').read_text());side=0 if report['throwing_side']=='left' else 1;w=20+side;e=18+side;window=report['window']['frame_interval'];leg=native_leg_from_fit(FOLDER/'fits/current_GVHMR'/f'{cid}.npz');accepted=[r['variant'] for r in rows_ if r['id']==cid and r['source_and_preservation_gate']]
   if len(accepted)>=2:
    curves=[]
    for variant in accepted:
     z=load(FOLDER/'fits'/variant/f'{cid}.npz');curves.append((z['anatomical_keypoints'][:,w,2]-z['anatomical_keypoints'][:,e,2])/leg)
    curves=np.stack(curves);span=curves.max(0)-curves.min(0);span=span[window[0]:window[1]+1];sensitivity.append(dict(id=cid,source_gated_candidates=len(accepted),max_forearm_camera_depth_span_same_frame_leg=float(span.max()),not_actual_3d_error=True))
 save(site/'depth_hypothesis_sensitivity.json',sensitivity);save(site/'interpreted_clip_results.json',cases)
 for name in ['native_fk_audit.json','detector_score_sensitivity.json','detector_score_sensitivity.csv']:
  shutil.copy2(FOLDER/name,site/name)
 # Preserve the verified main-release evidence; it describes the earlier
 # actual deployment, not an unperformed arm publication.
 shutil.copy2(OUT/'all813_gvhmr_v1/deployment.json',site/'main_release_deployment.json')
 def med(split,variant,key):return float(np.median([r[key] for r in rows_ if r['split']==split and r['variant']==variant and r[key] is not None]))
 stats=dict(confirmation=dict(wrist_p95_median_before=med('confirmation','current_GVHMR','all_pitch_wrist_p95_px'),wrist_p95_median_after=med('confirmation',selected,'all_pitch_wrist_p95_px'),holdout_wrist_p95_before=med('confirmation','current_GVHMR','holdout_pitch_wrist_p95_px'),holdout_wrist_p95_after=med('confirmation',selected,'holdout_pitch_wrist_p95_px'),gv_depth_control_wrist_p95=med('confirmation','raw_ray_gv','all_pitch_wrist_p95_px'),rtmw_wrist_p95_before=med('confirmation','current_GVHMR','separate_rtmw_pitch_wrist_p95_px'),rtmw_wrist_p95_after=med('confirmation',selected,'separate_rtmw_pitch_wrist_p95_px')),actual_3d_accuracy_validated=False,unique_depth_identified=False,main_arm_experiment_applied=False)
 save(site/'interpretation.json',stats)
 c=stats['confirmation'];table='<table><tr><th>버전</th><th>분할</th><th>고정 선택</th><th>정합·보존 검사</th><th>손목 p95 중앙값 px</th><th>팔 재적합 제외 프레임 p95 px</th><th>팔 3D 변화 p95 / 다리 길이</th></tr>'+''.join(f'<tr><td>{r["version"]}</td><td>{r["split"]}</td><td>{r["selected_variant"]}</td><td>{r["source_and_preservation_pass"]}/6</td><td>{r["median_all_pitch_wrist_p95_px"]:.3f}</td><td>{r["median_holdout_pitch_wrist_p95_px"]:.3f}</td><td>{r["median_arm_change_p95_leg"]:.4f}</td></tr>' for r in flat)+'</table>'
 narrative=f'''<section id="interpretation"><h2>이번에 확인된 점</h2><p>메인은 이미 813개 / 123명 모두 GVHMR 접지 fitting으로 배포했고, 여기의 팔 실험은 메인에 적용하지 않았습니다. 이 프리뷰는 관측에 맞는 새로운 깊이 후보를 만드는 연구 결과이며 실제 코킹 3D 정답을 검증한 결과가 아닙니다.</p><p>고정 설정 <code>{selected}</code>는 튜닝 5/6, 다른 선수·원본 영상의 최종 비교 6/6에서 원본 정합·동작 보존 기준을 만족했습니다. 최종 비교의 손목 p95 중앙값은 {c['wrist_p95_median_before']:.2f} → {c['wrist_p95_median_after']:.2f} px, 이번 팔 재적합에서 직접 제외한 프레임은 {c['holdout_wrist_p95_before']:.2f} → {c['holdout_wrist_p95_after']:.2f} px였습니다. 초기 모델과 메인 fitting은 전체 입력을 보았으므로 완전히 미관측한 2D 검증이나 KBO 3D GT가 아닙니다.</p><h3>무엇이 기여했는가</h3><ol><li>V1: MotionBERT 팔을 그대로 이식하면 원본 재투영이 크게 틀어졌고, SMPL 전신 pose corrective가 발 표면 마커까지 움직였습니다. 가장 보수적인 축 회전 후보는 원본을 덜 망가뜨렸지만 팔 변화가 매우 작아 목표한 개선으로 채택하지 않았습니다.</li><li>V2: 국소 팔 마커와 시간에 연결된 보정만으로는 원본과 멀리 떨어진 초기 팔 자세를 충분히 고치지 못했습니다. 발 고정은 이 실험 범위의 정의이며 접지 정확도 개선 증거가 아닙니다.</li><li>V3: 원본의 팔꿈치·손목 광선과 고정된 상완·전완 길이로 네 가지 앞뒤 깊이 해를 만들었습니다. MotionBERT는 XY 위치를 이식하지 않고 상대 깊이 선택에만 썼습니다. 추가 fitting보다 이 초기 후보의 제외 프레임 정합성이 좋아, 기본 프리뷰는 추가 fitting 전 후보입니다. 모든 다른 해도 비교할 수 있습니다.</li></ol><p><strong>픽셀 오차 감소의 대부분은 원본 조건부 기하에서 왔습니다.</strong> MotionBERT 없이 기존 GV 깊이를 고르는 대조군도 최종 비교 손목 p95 {c['gv_depth_control_wrist_p95']:.2f} px를 얻었습니다. MotionBERT가 바꾸는 것은 주로 팔의 앞뒤 해석이며 그 깊이가 정답이라는 증거는 아직 없습니다. 같은 2D를 설명하는 다른 깊이 후보를 평균하지 않았습니다.</p>{table}<h3>대표 사례와 반례</h3><ul><li><a href="./?clip=e1cae96b4d83_c0027&amp;frame=47&amp;view=side">더거 · 동일 프레임 측면 비교</a>: 손목 p95 3.21 → 0.88 px. 모델 기준 후방 전완 위치 최대는 −0.113 → 0.299 다리 길이로 달라집니다. 이는 새로운 후방 깊이 후보이며 실제 손목 회전각이 아닙니다.</li><li><a href="./?clip=b96e8ea44448_c0044&amp;frame=67&amp;view=side">와이스 · 최종 비교 클립</a>: 손목 p95 4.43 → 1.78 px, 후방 위치 모델 지표 0.063 → 0.215. 하체·골반·발·카메라는 메인 값과 같습니다.</li><li><a href="./?clip=e56945a0fd80_c0028&amp;view=side">양현종 · 실패 보존</a>: 손목 p95 6.30 → 11.77 px로 악화되어 원본 정합 기준을 통과하지 못했습니다. 현재 고정한 몸통·카메라·팔 길이·2D의 불일치를 팔 깊이만으로 해결하지 못합니다.</li><li><a href="./?clip=0d84991ce9f6_c0008&amp;view=side">최원태 · 재투영만으로 판단하면 안 되는 반례</a>: 손목 p95는 6.21 → 2.32 px로 줄지만 후방 위치 모델 지표는 0.282 → −0.005로 줄어듭니다. 사용자가 원하는 뒤손 동작 개선이 항상 따라오는 것은 아닙니다.</li></ul><h3>확인하지 못한 것</h3><p>별도 RTMW 관측에 대한 손목 p95 중앙값은 {c['rtmw_wrist_p95_before']:.2f} → {c['rtmw_wrist_p95_after']:.2f} px로 거의 그대로이며 추가 정확도 증거가 강하지 않습니다. 이 모델의 native 점수는 실제로 1.02–7.06 범위인 decoder 값이어서 확률이 아닙니다. 기존 동결 기준은 유지하고 점수 순위별 민감도를 사후 진단으로만 별도 저장했습니다. RGB blur·가림에서 검출기들이 크게 다릅니다. 원본 대조와 source detector 정합을 실제 3D 각도·거리 정확도로 바꾸어 해석하지 않습니다.</p><p>발과 하체 보존은 의도한 실험 범위입니다. 카메라·지면·축척은 기존 단안 추정에 조건부이며, 손목/손바닥 로컬 회전은 바꾸지 않았습니다. 17관절 MotionBERT로 실제 손목 굽힘을 관측했다고 주장하지 않습니다. 사람의 수동 검수 완료나 실제 MPJPE를 부여하지 않았습니다. 다음 우선순위는 같은 투구의 확인된 리플레이, 원본 RGB의 팔–몸통 가림 순서처럼 네 깊이 후보를 구별하는 추가 관측입니다. 단순히 손을 더 뒤로 보내는 설정이나 재투영만 낮추는 설정을 메인으로 채택하지 않습니다.</p><p><a href="iteration_summary.csv">세 버전 요약 CSV</a> · <a href="iterations/v1/evaluation_confirmation.csv">V1 최종 비교 전체 CSV</a> · <a href="iterations/v2/evaluation_confirmation.csv">V2 최종 비교 전체 CSV</a> · <a href="depth_hypothesis_sensitivity.json">깊이 가설 민감도</a> · <a href="detector_score_sensitivity.csv">검출 점수 사후 진단</a> · <a href="native_fk_audit.json">관절 FK 일치 검사</a></p></section>'''
 report=site/'report.html';text=report.read_text();import re;text=re.sub(r'<section id="interpretation">.*?</section>','',text,flags=re.S);text=text.replace('<h2>후보별 효과</h2>',narrative+'<h2>후보별 전체 수치</h2>');report.write_text(text)
 # Keep own unused V1/V2 UI drafts outside the public dataset tree. All
 # actual numerical runs/checkpoints remain in their versioned result dirs.
 for version in ['v1','v2']:
  draft=CODE/'studies'/f'arm-cocking-{version}-20261028'
  if draft.exists():
   destination=OUT/('arm_cocking_'+version)/'site_draft'
   if destination.exists():destination=destination.with_name('site_draft_'+time.strftime('%H%M%S',time.gmtime()))
   shutil.move(str(draft),str(destination))
 print(json.dumps(stats,indent=2))

def native_leg_from_fit(path):
 from stride_review import native_leg
 return native_leg(load(path)['anatomical_keypoints'])

if __name__=='__main__':main()
