오승환 Delivery 3: 잘못 선택된 대상 복구
=======================================

클립 `4e09dd724a8c_c0011` (2024-07-19, 원본 614.8142–617.2833초)은 기존 bbox와 Halpe-26 2D부터 타자를 추적했습니다. 선수 이름 메타데이터는 투수를 가리키지만 추적 대상의 정확성을 보증하지 않았습니다. 접지 fitting이나 뷰어의 문제로 분류하지 않습니다.

같은 원본 75프레임에서 마운드 앞쪽 투수 영역과 시간 연속성으로 YOLOv8n 검출을 연결했습니다. 75프레임 모두 직접 검출이 있었고, 같은 RTMPose-m Halpe-26 모델로 새 2D를 추출했습니다. 기존과 같은 2D refinement 설정(.45, 최대 3프레임 보간)을 사용했습니다. 기존 NPZ·공개 2D·3D를 덮어쓰지 않고 새 경로로 저장했습니다.

HMR2a RGB → GVHMR f1/f4/f12 → 단계별 fitting → 기존 8개 접지/지면 후보 → 기존 선택 규칙을 실행했습니다. 타자의 WHAM 자세·체형은 사용하지 않았고, 수정한 투수 RGB의 HMR2a 체형 중앙값을 시퀀스 내 고정했습니다. 실제 native SMPL FK를 75프레임 모두 대조했습니다. 원본 픽셀은 로컬 진단에서만 사용합니다. 공개 썸네일은 숫자 관절점으로만 렌더링합니다.

이 구간은 스트라이드 이후부터 팔 동작·팔로스루를 담습니다. 레그리프트·착지 이벤트는 unknown/구간 밖이고 접지 가중치는 0입니다. 바닥은 단안 추정 기준면이며 실측 마운드가 아닙니다. 대상 선택 오류의 복구이지, 절대 보폭이나 코킹 각도의 실제 정확도 개선 검증이 아닙니다. 사람이 수동 검수했다고 표시하지 않습니다.

실행 환경은 기존 `research/pitch_observation_20261028/env`, 별도 worktree는 `research/oh_delivery3_fix_20261028/code`입니다. 이 worktree에서 다음을 실행합니다.

```bash
PY=/home/eis/Workspace/Kichang/kbo-pitchers/research/pitch_observation_20261028/env/bin/python
EXP=experiments/pitch_observation
CUDA_VISIBLE_DEVICES=1 $PY $EXP/correct_oh_subject.py observe
CUDA_VISIBLE_DEVICES=1 $PY $EXP/correct_oh_subject.py features
CUDA_VISIBLE_DEVICES=1 $PY $EXP/correct_oh_subject.py network
CUDA_VISIBLE_DEVICES=1 $PY $EXP/correct_oh_subject.py base
CUDA_VISIBLE_DEVICES=1 $PY $EXP/correct_oh_subject.py contact
$PY $EXP/correct_oh_subject.py select
$PY $EXP/publish_oh_subject.py
# stage의 audit와 원본 대조 후 로컬 active pointer만 변경:
$PY $EXP/publish_oh_subject.py --promote
$PY -m unittest discover -s tests
```

완료 파일을 재사용합니다. 관측/설정 변경은 새 버전으로 실행해야 합니다. 로컬 진단과 원본 전체 프레임은 사이트 밖 `../subject_correction_v1/`에 있습니다. 모델·입력·과거 자산·새 자산 해시, 선택 근거와 전체 후보는 `data/subject_corrections/4e09dd724a8c_c0011-v1.json`에 기록합니다. 이 복구 스크립트는 자동 push/배포를 하지 않습니다.
