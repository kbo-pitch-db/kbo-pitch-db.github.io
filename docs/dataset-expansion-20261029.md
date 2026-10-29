# 2026-10-29 추가 수집

영상 5개에서 기존 수집 기준을 통과한 클립 130개를 얻었다. 전광판 자막을 여러 프레임에서 대조해 선수 이름을 확인한 79개 중 78개를 공개했다. 선수 확인이 안 된 51개와 팔 동작 검사에 걸린 1개는 원본과 함께 로컬에 보관한다.

공개 데이터는 891개 클립, 동작이 있는 투수 126명, 원본 프레임 143,009개다. 기존 813개 클립의 항목과 2D·3D 파일 2,439개는 바꾸지 않았다. 새 클립에도 RTMPose Halpe-26, GVHMR, 지면·발 접촉 보정, MotionBERT 팔 보정, 고정된 뼈 길이와 신발 형태, 프레임 연속성 검사를 썼다. 검사 기준은 기존 `combined-population-full-support-qc-v5`와 같다. 초기 GVHMR 추정은 최근 선수 추적 복구에 쓰던 경로를 따른다.

5개 영상에서 11개 클립을 골라 원본과 추정 자세를 5프레임씩 비교했다. 이 비교와 수치 검사는 실측 3D 정확도 검증을 대신하지 않는다. 원본 중계 영상과 검토 이미지는 공개 저장소에 넣지 않았다.

## 투구 기록

네이버 API 요청이 시간 초과되어 [기존에 안내한 저장소의 2024년 PTS 파일](https://github.com/leekichang/where-do-the-robot-umpires-see/blob/37cf62be0604aa8a7f077759eebf7a6eda002edb/data/2024.csv)을 사용했다. 기존 기록과 겹치는 9,588개 투구에서 투수·타자 식별자, 이름, 타석 내 투구 번호와 결과가 모두 일치했다. 궤적 계수도 일치했으며 교차 지점 좌표에는 반올림 수준의 차이가 있었다.

추가한 1,407개 기록은 파일에 실제로 있는 추적 투구다. 해당 경기의 모든 투구를 수집했다는 뜻은 아니다. 누적 투구 수, 초·말, 타석 식별자와 투구 전 카운트는 파일에 없어 비워 두었다. CSV 순서로 이 값을 만들거나 같은 선수라는 이유로 동작 클립을 특정 투구에 연결하지 않는다. 구종별 무브먼트와 구속 통계는 수집한 기록만 집계한다.

현재 기록은 37경기, 투수 159명의 11,161개 투구다. 그중 11,154개에서 기존 계산법으로 HB·IVB·VAA를 구할 수 있다. 나머지 7개는 이전부터 추적 정보가 없던 기록이다. 선수의 투구 손 정보가 필요한 새 선수는 KBO 선수 페이지에서 확인했다. 현재 소속팀이나 신장 정보는 2024년 기록에 옮기지 않았다.

## 파일과 재현

- [추가 영상의 해시와 원본 링크](../data/releases/combined-20261029-increment-v1/sources.json)
- [클립별 검사 결과](../data/releases/combined-20261029-increment-v1/quality_records.json)
- [기록 대조 결과](../data/releases/combined-20261029-increment-v1/external_pts_validation.json)
- [이전 공개 인덱스](../data/releases/combined-20261029-increment-v1/previous_index.json.gz)
- [이번 실행에 사용한 작업 스크립트](https://github.com/kbo-pitch-db/kbo-pitch-db.github.io/tree/main/data/releases/combined-20261029-increment-v1/sources)

로컬의 `dataset/motion_gvhmr_combined.csv`에는 현재 공개한 최종 2D·3D 파일 경로와 해시가 있다. 이전 WHAM 실험 테이블은 그대로 남겼다. 새 클립의 원본 프레임 번호와 시각을 유지했으며, 이벤트 자동 후보는 선수 대표 통계에 포함하지 않는다. 기존 이벤트 모델과 12개 클립의 비교 조건도 바꾸지 않았다.

현재 작업 디렉터리가 있는 서버에서 인덱스를 다시 만들 때는 다음 명령을 사용한다. 추가 연구 디렉터리에는 이번 실행의 원본 추정치와 `combined_population_v1/selection.json`이 필요하다. 모델 파일과 원본 영상은 별도로 있어야 한다.

```bash
python scripts/build_portal_data.py
python scripts/build_pitch_flight.py
python scripts/build_biomechanics_database.py \
  --additional-study ../research/dataset_expansion_20261029 \
  --reuse-existing --refresh-records
python scripts/publish_pitch_events.py
python scripts/build_kinematic_sequence.py
python scripts/build_player_overview.py
```
