"""Generate auditable completion artifacts from the actual public release."""
import gzip,hashlib,json,subprocess
from pathlib import Path
from .dictionary import VERSION
REPO=Path(__file__).resolve().parents[2]

def main():
 cov=json.loads((REPO/'data/metrics/coverage.json').read_text());d=json.loads((REPO/'data/metrics/dictionary.json').read_text());migration=json.loads((REPO/'data/metrics/migration.json').read_text());ref=json.loads((REPO/'data/metrics/reference_check.json').read_text())
 names={m['id']:m['name']['ko'] for m in d['metrics']}
 candidates=[
  dict(id='ground_reaction_force_and_braking_impulse',name='지면반력·제동 충격량',required=['Synchronized calibrated force plates or validated instrumented mound','Independent physical clock','Foot contact attribution'],excludedProxy='Knee extension is not braking force.'),
  dict(id='whole_body_com_deceleration',name='전신 무게중심 감속',required=['Validated whole-body segment kinematics including relevant segments','Subject mass distribution/inertial model','Calibrated length and real time'],excludedProxy='Pelvis center is not whole-body COM.'),
  dict(id='joint_moments_power_energy_flow',name='관절 모멘트·파워·에너지 흐름',required=['Validated segment coordinate frames and rotations','Subject segment masses/COM/inertia','External forces/contact','Inverse dynamics and power/energy accounting','Synchronized physical time and calibrated length'],excludedProxy='Angular-velocity ratios are not energy efficiency.'),
  dict(id='elbow_shoulder_load',name='팔꿈치·어깨 부하',required=['Validated high-rate upper-limb orientation and anatomical joint axes','External forces and subject-specific inertial assumptions','Inverse dynamics compared against external ground truth'],excludedProxy='Pose angle alone is not torque, tissue stress or injury risk.'),
  dict(id='energy_efficiency_or_kinetic_chain_score',name='에너지 효율·운동사슬 종합 점수',required=['Public reproducible physical definition','Validated energy/power quantities and accounting boundary','External construct validation'],excludedProxy='Peak order alone is not energy-transfer efficiency.'),
  dict(id='mvpi',name='MVPI',required=['Verifiable public formal definition and reproducible implementation','Measurement conditions and independent validation'],excludedProxy='No invented expansion, formula or score is implemented.')]
 (REPO/'data/metrics/research_candidates.json').write_text(json.dumps(dict(version=VERSION,items=candidates,sources=[d['sources']['obp']['url'],d['sources']['block']['url'],'https://pubmed.ncbi.nlm.nih.gov/32635991/']),ensure_ascii=False,indent=2)+'\n')
 files={}
 for folder in ['data/biomechanics/analysis','data/biomechanics/poses']:
  for p in sorted((REPO/folder).glob('*.json.gz')):files[str(p.relative_to(REPO))]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 for p in sorted((REPO/'scripts/biomechanics').glob('*.py')):files[str(p.relative_to(REPO))]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 for name in ['scripts/build_biomechanics_database.py','metrics_database.js','biomechanics_detail.js','data/metrics/database.json.gz','data/metrics/dictionary.json','data/metrics/coverage.json','data/metrics/migration.json','data/metrics/player_summary.json.gz','data/metrics/source_reviews_20261029.json','data/metrics/duplicate_audit.json','data/metrics/reference_check.json']:
  p=REPO/name;files[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 manifest=dict(version=VERSION,analysisClips=len(list((REPO/'data/biomechanics/analysis').glob('*.json.gz'))),poseVariants=len(list((REPO/'data/biomechanics/poses').glob('*.json.gz'))),canonicalPoseReleaseUnchanged='gvhmr-ground-motionbert-20261028-v1',files=files)
 (REPO/'data/biomechanics/manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
 lines=['# 3D 야구 메커니즘 데이터베이스 v2','',f'계산 버전: `{VERSION}`. 지표 데이터·계산 코드·홈페이지가 실제로 교체된 릴리스다. 기존 canonical 3D 뷰어와 원본 2D 바이트는 보존하고, 원본 관측과 일치도가 개선된 17개 분석용 3D 변형을 별도 경로로 추가했다.','',
 '## 범위와 해석','',
 '813개 클립 / 123명에 대해 계산을 적용했다. 경기 기록은 32경기, 9,754투구이며, 390투구에 동작이 자동 연결되어 있다. 나머지 423개 미연결 클립은 별도 단위다. 경기 기록 자동 연결도 사람 검수 전이다. 시즌 전체나 813개 고유 투구로 해석하지 않는다.','',
 f'지표 사전 {len(d["metrics"])}개 중 {sum(m["values"]>0 for m in cov["metrics"])}개 필드에 3D 수치가 있다. **이 수치에는 미채택 자동 이벤트 후보가 포함되므로 산출 수를 유효 연구 표본 수로 읽으면 안 된다.** 원본 프레임을 확인한 클립은 19개이며 사람의 독립 판정은 아니다. 핵심 앞무릎 FC·BR 차이의 원본 확인 표본은 17개다. 17개 모두 설정/모델 민감도 표시가 필요하다.','',
 '기본 화면은 원본 확인된 잠정 3D를 먼저 보여준다. 자동 이벤트 후보는 상태 필터를 바꿔 확인할 수 있으며 집계에는 들어가지 않는다. 모델 민감 3D는 값을 삭제하지 않고 상태·범위를 제공하며 기본 집계에 포함한다. 선택 해제하면 테스트 설정 내 변동이 작은 값만 남는다. 민감도는 정확도 검증이 아니다.','',
 '## 연구 정의와 모델 계산','',
 '- 무릎·팔꿈치 굴곡은 native 3D 관절 중심의 내각을 180°에서 뺀 값이다. 0°가 신전이다. 앞무릎 순신전량은 FC 굴곡−BR 굴곡이며 최대굴곡−BR과 다르다.',
 '- 영상 FC는 처음 보이는 발 접촉의 범위다. OpenBiomechanics의 10% 체중 contact 및 100% plant 힘 이벤트와 동일하지 않다. 영상 BR도 손목이 팔꿈치를 지난 뒤 0.01초라는 실험실 알고리즘과 다르다.',
 '- 골반은 SMPL root, 흉곽은 spine3 누적 회전과 native rest-frame 보정이다. ASIS/PSIS/C7으로 보정한 해부학적 프레임이 아니다. 상대 회전은 intrinsic ZYX의 yaw를 좌우투 역할로 통일한다. 양수는 골반 선행이다.',
 '- 골반선–어깨선 수평 분리각은 별도 필드다. 상대 흉곽–골반 회전으로 섞지 않는다. 전역 상방은 모델/지면에서 추정한 값이며 실측 중력이 아니다.',
 '- 전경·측굴은 몸통 상방 축의 서로 다른 전역 수직 평면에서 atan2로 산출한다. 축회전은 명시한 ZYX yaw다. 평면 경사는 Visual3D Cardan pitch/roll과 동일하지 않으며 하나의 무부호 수직 기울기가 아니다.',
 '- 상완 거상은 몸통 아래 방향에 대한 벡터각이다. 해부학적 어깨 외전·외회전으로 이름을 바꾸지 않는다. 수평외전도 명시한 기하각으로 분리한다.',
 '- 상완 장축 회전은 관절 중심만으로 식별되지 않는다. 상완 twist와 팔꿈치 보상을 함께 바꾸어 관절 중심을 유지하는 합성 반례를 검사했다. SMPL rotation이 존재해도 prior 영향을 배제할 증거가 없어 어깨 ER/MER/IR 속도를 결측으로 둔다.',
 '- 실측 스케일과 신장이 없어 cm 보폭·신장 대비 보폭은 결측이다. 발목 간격 P90으로 채우지 않는다. 모델 다리 길이 대비 좌우 위치는 이름부터 프로젝트 파생 변수이며 보폭이 아니다. 발 방향에는 발가락·뒤꿈치 원본 관측이 모두 필요하다.',
 '- 공간 지표는 관측 원본 프레임의 평활화 없는 값이다. 이벤트 범위의 양 끝·중앙, 독립 이미지 기준 Gaussian sigma 0.5/1, 대안 GVHMR 모델, 투구축 ±10°/20°를 비교한다. 임계 15°/구간 20%는 운영상 민감도 표시이며 정상 범위가 아니다.',
 '- 구간은 각 지표 사전에 고정했다. 최대값의 기본 FC→BR 구간은 그 이전 피크를 포함하지 않을 수 있어 전체 투구 최대값과 비교하면 안 된다.',
 '- 실제 시간 증거가 있는 입력은 native 5/7/9 이미지 Savitzky–Golay 3차와 물리 timestamp 미분으로 속도를 계산할 수 있다. 합성 240 Hz 입력으로 알려진 100°/s와 FC→BR 시간을 검사했다. 이는 연구의 13.4/20 Hz Butterworth와 다른 운영 정의로 기록한다. 현재 중계 표본의 실제 시간 복원은 0개이므로 deg/s·ms를 공개하지 않는다.','',
 '## 실제 수정과 안정성 진단','',
 '최종 공개 pose의 `source_fit_sha256`와 `selection.json` 피팅 파일을 매번 대조한다. 중간 `selection_frozen.json`을 참조하지 않는다. 한 투구에서 323개 피팅 샘플을 324개 영상 프레임에 밀어 넣을 수 있는 불일치도 발견하여 원본 frame ID로 정렬하고 누락을 보간하지 않도록 했다.','',
 '오승환의 교정된 원본 pose는 전체 화면 좌표인데 legacy 크롭 원점을 다시 더하면 잘못된다. 각 pose의 crop-box 좌표를 우선 사용하도록 고쳤다. native joint-center 재투영 실험은 root/camera/shape/상체를 고정하고 엉덩관절·무릎을 조정했다. 19개 중 17개가 사전 정의한 이미지 오차·골 길이·발목 바닥 침범 기준을 만족했다. 나머지 2개는 침범 증가로 채택하지 않았다. 이미지 중앙오차의 클립 간 중앙값은 약 9.45→0.63px이지만 **3D GT 정확도 개선을 뜻하지 않는다.**','',
 f'프레임 디코딩 결과 near-duplicate 후보 쌍은 {cov["nearDuplicateFramePairs"]:,}개다. 중복 제거 후보와 재생 cadence를 보존했으나 슬로모션 배율·물리 clock은 복원하지 못했다. 업샘플링으로 독립 해상도가 높아졌다고 주장하지 않는다.','',
 '자동 이벤트 개발 표본 대조에서 착지 범위 겹침은 5/12, 가장 높은 공 추적 후보의 BR 범위 겹침은 8/15였다. 이는 개발 표본·에이전트 범위 대조이며 정확도나 held-out 성능이 아니다. 발 감속과 흰 물체 추적은 독립 접촉/릴리스 증거로 부족하므로 자동 결과는 미채택 후보로만 남긴다. 손목 속도 피크는 검색 구간에만 쓰며 BR로 채우지 않는다. MER로 ER를 검증하는 순환은 없다.','',
 '## 지표별 실제 대상·산출·결측','',
 '대상은 각각 813개 동작 클립이다. “원본 확인”에도 민감도 큰 추정이 포함된다. 후보는 기본 선수 통계에서 제외된다. 동작 없는 경기 기록 9,364행은 이 표의 813개 대상과 별도로 `no_motion`이다.','',
 '| 지표 | 수치 전체 | 원본 확인 | 자동 후보 | 결측 이유 |','|---|---:|---:|---:|---|']
 for m in cov['metrics']:lines.append(f'| {names[m["metricId"]]} | {m["values"]} | {m["sourceReviewed"]} | {m["automaticCandidates"]} | '+('; '.join(f'{k}: {v}' for k,v in m['missingReasons'].items()) or '—')+' |')
 lines+=['','## 기존→신규 대응','', '| 기존 필드 | 신규 필드 / 처리 | 변경 이유 |','|---|---|---|']
 for m in migration['fields']:lines.append(f'| `{m["old"]}` | '+(f'`{m["new"]}`' if m['new'] else m['action'])+' | '+m['reason']+' |')
 lines+=['','v1 JSON·CSV·사전·원래 계산 코드·UI는 `data/releases/metrics-20261028-v1/`에 보존했다. manifest의 SHA256은 압축 전 원본 바이트와 원래 gzip을 각각 검증한다. 2D 필드는 기본 지표에서 제거하고 품질 보조 탭에 둔다.','',
 '## 대표 원본 대조 사례','',
 '모든 사례의 원본 주요 프레임은 로컬에서 직접 확인했다. 원본 RGB는 프로젝트의 pose-only 정책에 따라 재배포하지 않는다. 공개 근거 화면은 원본 중계의 시점 링크, 원본 2D 검출과 native 3D 재투영, 실제 계산용 3D, 이벤트 범위와 지표 값을 함께 제공한다. 이벤트 원본 프레임 구간은 다음과 같다.','',
 '| 선수 / 투구 | 손 | FC | BR | 3D 계산·원본 링크 |','|---|---|---|---|---|']
 for c in cov['cases']:
  interval=lambda e:'—' if c['events'][e] is None else '–'.join(map(str,c['events'][e]))
  lines.append(f'| {c["player"]} / {c["pitch"]} | {c["hand"]} | {interval("fc")} | {interval("br")} | [근거 화면](https://kbo-pitch-db.github.io/{c["evidenceUrl"]}) · [중계]({c["sourceUrl"]}) |')
 lines+=['','## 검증의 세 층','',
 '1. **계산 검증:** 알려진 굴곡, 좌표 회전, 좌우 부호, wrapping, gimbal singularity, 전경/측굴 분리, 상완 twist 반례, 실제 clock 유무, 순신전 식, 결측/가림, 원본 frame 정렬을 합성·공개 수치로 검사한다.',
 '2. **안정성 진단:** 이벤트 범위, 평활화, 투구 축, 모델 대안을 바꾼 범위를 저장한다. 에이전트 영상 확인과 재투영 개선도 이 층이다. 모두 GT 정확도와 구분한다.',
 '3. **외부 정확도 검증:** 현재 중계 3D에 대응하는 외부 marker/동기 다중 카메라 GT는 없다. 모든 값은 `not_validated_against_external_GT`다. 다른 markerless 시스템의 논문 성능을 전이하지 않는다.','',
 'OpenBiomechanics 공개 12투구에서 3관절 기하 굴곡과 공개 knee angle의 구간 RMSE 중앙값은 약 0.84°였다. 다만 공개 POI 순신전량은 10%·100% 이벤트 어느 한쪽의 최근접 각도 차이와 일괄 일치하지 않았다(각 최대 차이 약 7.59°/2.70°). 따라서 **POI 숫자를 정확히 재현했다고 주장하지 않는다.** 이벤트/표본/처리 차이를 남겨 후속 확인 대상으로 둔다. 이 비교는 실험실 신호의 정의·구현 대조이며 중계 3D 정확도 검증이 아니다. 소스 ZIP은 SHA256을 기록하고 재배포하지 않았다.','',
 '## 재사용 파일과 집계','',
 '- `data/metrics/database.json(.gz)`: 10,177행, 버전·사전·지표별 수치/상태. 동작 없는 행은 빈 metrics map이 모든 지표의 null/no_motion을 뜻한다. CSV는 이를 명시적으로 펼친다.',
 '- `dictionary.json`, `migration.json`, `coverage.json`, `clip_provenance.json`, `reference_check.json`, `research_candidates.json`.',
 '- `player_summary.json(.gz)`: 선수×구종×자료 단위×촬영 조건×분석 모델×계산 버전별 평균, 표본 SD, 중앙값, Q25/Q75/IQR, 유효 n·경기 수·상태 수. SD는 n≥2, IQR은 n≥3. 민감 3D를 포함하고 상태를 유지한다.',
 '- `data/biomechanics/analysis/*.json.gz`: 이벤트·원본 frame/time·cadence·원본 2D/3D 경로·사용 프레임·부호/좌표·프레임별 골반/흉곽 방향행렬·계산 파형·재투영·필터/모델/축 민감도. 수치 재사용에 모델 가중치가 필요하지 않는다.',
 '- `data/biomechanics/poses/*.json.gz`: 채택한 17개 별도 분석 포즈. canonical 원본 3D는 덮어쓰지 않았다.',
 '- `manifest.json`: 813개 분석 파일과 17개 변형, 계산 모듈 SHA256. 원본 2D/3D 자산의 Git 변경은 없다.',
 '- `duplicate_audit.json`: 135,741개 원본 이미지를 디코딩해 같은 선수·경기에서 정확한 축소 이미지 해시 3개 이상 공유 또는 원본 구간 80% 이상 겹침을 검사했다. 후보 0건, 중복 accepted pitch ID 0건이다. 재인코딩·재크롭 리플레이는 이 검사로 배제할 수 없다.',
 '- 같은 accepted pitch ID는 한 번만 집계한다. 확인되지 않은 리플레이의 동일성은 임의 병합하지 않으며, 423개 미연결 클립은 고유 투구 통계에 섞지 않는다.','',
 '촬영 조건이 미확인인 클립은 기본적으로 서로 다른 그룹이다. 확인한 조건도 같은 source·centerfield/side·계산 모델 수준의 거친 분류이며 엄밀히 동일한 카메라 보정 조건이라고 주장하지 않는다. 반복성에는 동작 변화와 추정 오차가 함께 들어 있다.','',
 '## 남은 차단 원인과 다음 작업','',
 '- 자동 접촉·공 분리 후보의 신뢰도가 낮다. 사람의 프레임 판정과 ball/foot segmentation·occlusion 처리를 추가하고 선수/경기 분리 검증 표본에서 다시 평가한 뒤 후보를 승격한다.',
 '- 794개 클립의 투구 축이 미확인이다. 원본 카메라/필드 특징과 실제 투구 방향을 보정하고, 분절 축까지 외부 대조해야 한다. 축 불필요 지표는 이미 남겨두었다.',
 '- 813개 모두 실제 시간 변환 증거가 없다. 슬로모션 편집 기록이나 동기 고속 원본을 확보한다. 독립 이미지 cadence는 참고일 뿐이다. 고속 팔 회전 피크에는 높은 native rate와 별도 안정성 검사가 필요하다.',
 '- 상완 장축 회전은 현재 관측에서 식별되지 않는다. 전완 대체각을 만들지 않고 다중 시점·표면/분절 방향 관측과 외부 GT를 요구한다.',
 '- cm 보폭·신장 정규화에는 독립 스케일과 실측 신장이 필요하다. GRF·COM 감속·모멘트·파워·에너지·부하는 별도 research_candidates 파일의 관측/역동역학 요건을 갖추어 연구한다.',
 '- 원본 일치도만으로 깊이/회전 정확도를 보장할 수 없다. 이번 17개 변형도 측정 정확도 검증을 통과한 결과가 아니며, 더 많은 지표 숫자를 채우기 위해 요건을 낮추지 않는다.','',
 '## 재현','',
 '```sh','python3 scripts/biomechanics/observations.py --workers 4', '../research/pitch_observation_20261028/env/bin/python scripts/biomechanics/native_refit.py', 'python3 scripts/build_biomechanics_database.py --workers 6','PYTHONPATH=scripts python3 -m biomechanics.reference_check --input-dir /tmp/kbo-obp-reference','PYTHONPATH=scripts python3 -m biomechanics.release_report','python3 -m unittest discover -s tests -p "test_*.py"','node tests/metrics_database.test.cjs','```','',
 '전체 재처리는 기존 로컬 dataset/clips.csv·비공개 원본 중계·source pose·최종 피팅 NPZ 및 라이선스가 필요한 SMPL 자산을 요구한다. 원본/가중치는 새로 공개하지 않는다. 공개 릴리스는 계산용 관절·방향행렬·파형·출처·사용 프레임을 포함한다. 외부 데이터 파일과 모델 가중치는 Git에 넣지 않는다.','',
 '## 확인한 연구·현장 출처','']
 for s in d['sources'].values():lines+= [f'- [{s["title"]}]({s["url"]})'+(f' · DOI `{s["doi"]}`' if s.get('doi') else '')+f' · 접근 범위: `{s["access"]}`. {s["notes"]}']
 lines+=['','OpenBiomechanics README·데이터 사전·모델 코드·공개 신호를 읽고 대조했다. 원문 접근이 되지 않은 논문은 abstract-only로 표시했으며, 그 논문의 상세 계산법까지 확인했다고 주장하지 않는다. 논문/현장 연관성을 인과관계나 정상/비정상 KBO 기준으로 바꾸지 않는다.']
 (REPO/'docs/metrics-database.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(manifestFiles=len(files),reportLines=len(lines))))

if __name__=='__main__':main()
