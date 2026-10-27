#!/usr/bin/env python3
"""Publish numerical study evidence only; never copy original broadcast media."""
import argparse
import hashlib
import json
from pathlib import Path

MODELS = ('current_final', 'H_local_temporal', 'WHAM_raw_articulation', 'MotionBERT_articulation')
TABLES = ('ablation.csv', 'ablation.json', 'ablation_summary.csv', 'ablation_summary.json',
          'status_comparison.csv', 'status_comparison.json', 'final_summary.json',
          'selection_policy.json', 'sample.json', 'sample_summary.json',
          'geometry_summary.json', 'model_comparison.json', 'model_comparison.csv',
          'independent_depth_diagnostics.csv', 'model_manifest.json', 'visual_notes.json',
          'optimizer_equivalence.json', 'overall_metrics.json')


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return round(value, 3) if isinstance(value, float) else value


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def package(study, site):
    target = site / 'studies' / 'pitch-20261027'
    target.mkdir(parents=True, exist_ok=True)
    viewer = target / 'viewer'
    viewer.mkdir(exist_ok=True)
    samples = json.loads((study / 'sample.json').read_text())
    statuses = {row['id']: row for row in json.loads((study / 'status_comparison.json').read_text())}
    clips = {row['id']: row for row in json.loads((site / 'data/clips.json').read_text())['clips']}
    for sample in samples:
        cid = sample['id']
        data = json.loads((study / 'viewer' / f'{cid}.json').read_text())
        # Explicit schema allowlist prevents source paths or image references leaking into the bundle.
        payload = {key: data[key] for key in ('id', 'times', 'observations', 'imageScale', 'edges', 'bounds')}
        payload['models'] = {name: data['models'][name] for name in MODELS}
        payload['status'] = {key: statuses[cid][key] for key in (
            'old_status', 'old_reasons', 'new_primary', 'selection_guard_reasons',
            'root_relative_articulation', 'global_translation', 'depth', 'ground_contact',
            'stride_absolute_distance', 'new_validation')}
        payload['player'] = clips[cid].get('player', '')
        payload['sourceUrl'] = clips[cid]['sourceUrl']
        payload['display_precision'] = '3 decimals for visualization only; evaluation CSV/JSON keeps full precision'
        write_json(viewer / f'{cid}.json', compact(payload))
    write_json(viewer / 'manifest.json', [{key: row[key] for key in ('id', 'split', 'old_status', 'player_name')} for row in samples])
    for name in TABLES:
        if name.endswith('.json'):
            write_json(target / name, json.loads((study / name).read_text()))
        else:
            (target / name).write_text((study / name).read_text())
    # Report is numerical prose/tables. Adapt its local-only viewer descriptions to the public bundle.
    report = (study / 'REPORT_KO.html').read_text()
    report = report.replace('<meta charset="utf-8">', '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')
    report = report.replace('<h1>', '<p><a href="../../">← KBO Pitch Lab</a> · <a href="comparison.html">48개 실험 비교 열기 →</a></p><h1>', 1)
    report = report.replace('자동 push·배포 없음.', '실험 당시 자동 push·배포하지 않았으며, 이후 사용자 요청으로 이 수치 보고서와 별도 비교 화면을 공개했습니다.')
    report = report.replace('원본·2D·고정 축 3D 비교', '수치 2D·고정 축 3D 비교')
    report = report.replace('exact source montage 및 3D 정면/측면/상단은 comparison.html에서 같은 범위로 비교할 수 있습니다. 원본 MP4 코덱이 없는 브라우저에서도 프레임 이미지로 동작합니다.',
                            '공개 비교 화면은 최종 2D·재투영과 3D 정면/측면/상단을 같은 범위로 표시합니다. 원본 영상은 각 클립의 YouTube 출처 링크에서 확인합니다. 원본 프레임·몽타주는 로컬 진단 자료에만 남겼습니다.')
    report = report.replace('원본 영상이 있는 이 진단 폴더는 로컬 전용이며 기존 공개 pose-only 정책은 유지합니다.',
                            '원본 영상이 있는 진단 폴더는 로컬에 유지했습니다. 이 공개 페이지에는 수치·포즈 비교만 있으며, 기존 공개 pose-only 정책을 유지합니다.')
    report = report.replace('새 공개 수 0, 새 3D/보폭 검증 수 0입니다.',
                            '메인 813개 클립의 추가 공개 수 0, 새 3D/보폭 검증 수 0입니다. 이번 배포는 별도 48개 표본의 실험 비교를 추가합니다.')
    (target / 'index.html').write_text(report)
    (target / 'comparison.html').write_text(Path(__file__).with_name('reconstruction_comparison.html').read_text())
    (target / 'comparison.js').write_text(Path(__file__).with_name('reconstruction_comparison.js').read_text())
    commands = (study / 'REPRODUCE.txt').read_text().replace(str(study.parents[1]), '<workspace>')
    (target / 'REPRODUCE.txt').write_text('Public numerical comparison: original broadcast media is not included.\n\n' + commands)
    files = [path for path in target.rglob('*') if path.is_file() and path.name != 'publication.json']
    assert len(list(viewer.glob('*.json'))) == 49
    assert all(path.suffix in ('.json', '.csv', '.html', '.js', '.txt') for path in files)
    assert not any(b'source_frames/' in path.read_bytes() or b'source_clips/' in path.read_bytes()
                   for path in viewer.glob('*.json'))
    manifest = dict(media_policy='numerical_pose_only', original_broadcast_media_included=False,
                    study_clips=48, total_alternatives_completed=813, public_models=list(MODELS),
                    production_poses_replaced=0, production_quality_labels_changed=0,
                    rendering_precision_decimals=3,
                    files={str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    write_json(target / 'publication.json', manifest)
    print(json.dumps(dict(path=str(target), files=len(files)+1, bytes=sum(path.stat().st_size for path in files), clips=48)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package(args.study.resolve(), args.site.resolve())
