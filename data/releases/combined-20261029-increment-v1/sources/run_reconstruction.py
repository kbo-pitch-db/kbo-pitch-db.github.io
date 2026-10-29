"""Isolated incremental execution of the current production solvers.

No original public poses, frozen study inputs, or public indexes are changed.
Only completed full-support geometry passes are staged for a later merge.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parent.parent
SITE = WORKSPACE/'kbo-pitch-db.github.io'
CODE = OUT/'code'
OBSERVATION = CODE/'experiments/pitch_observation'
sys.path[:0] = [str(OBSERVATION), str(CODE/'scripts')]
os.environ['KBO_COMBINED_SITE'] = str(CODE)


def run(command, cwd=WORKSPACE):
    print('RUN', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def bootstrap():
    # The raw ingest must be finished before its selected clip index is frozen.
    assert (OUT/'ingest/batch_summary.csv').exists()
    ingest = WORKSPACE/'kbo_pitch_motion_pilot'
    python = WORKSPACE/'.venv/bin/python'
    run([python, SITE/'scripts/import_published_pts.py', '--dataset', OUT/'dataset',
         '--csv', OUT/'external_records/2024.csv'])
    # CSV pitch rows lack cumulative counts and cannot enter the relay matcher.
    # Read scoreboards independently to select the correct broadcast OCR layout.
    run([python, OUT/'observe_scoreboards.py', OUT/'dataset'])
    run([python, SITE/'scripts/recover_player_identity.py', '--dataset', OUT/'dataset', '--workers', '4'])
    run([WORKSPACE/'.wham-env/bin/python', CODE/'scripts/wham_pipeline.py',
         '--dataset', OUT/'dataset', '--wham-root', WORKSPACE/'external/WHAM',
         '--pose-model', WORKSPACE/'models/rtmpose_halpe26_m.onnx',
         '--device', 'cuda:0', '--continue-on-error'])
    run([python, CODE/'scripts/build_site_data.py', '--dataset', OUT/'dataset',
         '--output', CODE/'data', '--limit', '0', '--max-video-mb', '0', '--numeric-only'])
    run([python, CODE/'scripts/package_motion.py', '--dataset', OUT/'dataset',
         '--output', CODE/'data', '--backend', 'wham'])
    # These are joint definitions only; no existing player's coordinates enter.
    index = json.loads((SITE/'data/clips.json').read_text())
    template = json.loads(gzip.decompress((SITE/'data'/index['clips'][0]['pose3d']).read_bytes()))
    private = json.loads((CODE/'data/clips.json').read_text())
    completed = []
    for clip in private['clips']:
        if (clip.get('pose3dBackend') != 'wham' or not clip.get('poseRefined')
                or not clip.get('playerKnown')):
            continue
        path = CODE/'data'/clip['pose3d']
        pose = json.loads(path.read_text())
        for key in ['joints', 'primary_representation', 'keypoints_semantics',
                    'anatomy_joints', 'anatomy_parents', 'foot_joints']:
            pose[key] = copy.deepcopy(template[key])
        path.write_text(json.dumps(pose, ensure_ascii=False, separators=(',', ':')))
        completed.append(clip)
    private['clips'] = completed
    (CODE/'data/clips.json').write_text(json.dumps(private, ensure_ascii=False, indent=2))
    (OUT/'initial_index.json').write_text(json.dumps(private, ensure_ascii=False, indent=2))


def upstream():
    import common
    motion = common.rows('motion_wham.csv')
    for metadata in (OUT/'raw/hmr2a').glob('*.json'):
        cached = json.loads(metadata.read_text())
        cid = cached['id']
        assert cached['source_2d_sha256'] == common.sha(OUT/'dataset'/motion[cid]['pose_2d_refined']), cid
    for stage in ['prepare', 'features', 'network']:
        run([sys.executable, OBSERVATION/'all813_release.py', stage, '--workers', '1'])
    run([sys.executable, OBSERVATION/'all813_background.py', '--workers', '1'])
    run([sys.executable, OBSERVATION/'all813_fit.py', 'base', '--workers', '1'])
    # Use the same direct GVHMR base path as the latest source-recovered clips.
    # The final support/IK/rigid-shoe solvers below provide the contact fit.
    import common
    from all813_release import ROOT, population
    from stride_review import phase_info
    from package_explorer_3d import build_pose
    initial = json.loads((OUT/'initial_index.json').read_text())
    by_id = {c['id']: c for c in initial['clips']}
    complete, failures = [], {}
    for rec in population():
        cid = rec['id']
        source = ROOT/'base/gvhmr_temporal_f4'/f'{cid}.npz'
        if not source.exists():
            failures[cid] = 'GVHMR base fit unavailable'
            continue
        z = common.load(source)
        _, _, _, h, _ = common.init_inputs(cid)
        normal = z['normal']
        offset = float(common.np.median(z['foot_signed_height'] -
                       ((z['foot_keypoints']+z['translation'][:, None])*normal).sum(-1)))
        z['pilot_support_offset'] = common.np.array(offset)
        selected = ROOT/'selected'/f'{cid}.npz'
        common.atomic_npz(selected, **z)
        phase, _ = phase_info(rec, h, z['background_transforms'])
        common.save(selected.with_suffix('.json'), dict(id=cid, phase=phase,
                    source_base_sha256=common.sha(source), actual_gvhmr_inference=True,
                    initialization='Same direct GVHMR base path as source-recovered production clips'))
        clip = by_id[cid]
        two = json.loads((CODE/'data'/clip['poseRefined']).read_text())
        old = json.loads((CODE/'data'/clip['pose3d']).read_text())
        pose, _, _ = build_pose(cid, 'GVHMR initialization', 'gvhmr_f4', selected, old, two)
        relative = f'reconstructions/increment-initial/{cid}.json.gz'
        path = CODE/'data'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(common.clean(pose), separators=(',', ':')).encode(), mtime=0))
        clip.update(pose3d=relative, sceneSignature=common.sha(selected), pose3dBackend='gvhmr')
        for key in ['pose3dRaw', 'whamSignature', 'legacyPose3d']:
            clip.pop(key, None)
        complete.append(clip)
    common.save(CODE/'data/clips.json', dict(initial, clips=complete))
    common.save(OUT/'upstream_failures.json', failures)
    run([sys.executable, OBSERVATION/'combined_population.py', 'prepare'])
    run([sys.executable, OBSERVATION/'combined_population.py', 'motionbert'])


def combine():
    import common
    from combined_population import ROOT, manifest
    from pitch_study_fit import StudyFitter
    import combined_arms as arms
    import combined_contact_ik as ik
    import combined_foot_pose_temporal as shoe
    import combined_support as ground
    import combined_foot_pose_retry as retry
    import combined_root_support as roots
    import audit_combined_supported as audit
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    records = manifest()['records']
    gpu = StudyFitter(WORKSPACE/'external/WHAM', 'cuda:0', use_cuda_graph=False)
    cpu = StudyFitter(WORKSPACE/'external/WHAM', 'cpu', use_cuda_graph=False)
    failures, selected = {}, {}
    for number, rec in enumerate(records, 1):
        cid = rec['id']
        try:
            ground.fit(rec, gpu)
            arms.fit(rec)
            ik.process(rec, ROOT)
            choice = dict(fit_path=f'polished_v1/base/combined/{cid}.npz')
            shoe.fit(rec, choice)
            audit.ROOT = ROOT/'foot_pose_v2'
            result = audit.audit(rec, cpu)
            path = ROOT/'foot_pose_v2/combined'/f'{cid}.npz'
            if not result['geometry_pass']:
                retry.fit(rec, choice)
                audit.ROOT = ROOT/'foot_pose_v3'
                alternative = audit.audit(rec, cpu)
                if alternative['geometry_pass']:
                    result, path = alternative, ROOT/'foot_pose_v3/combined'/f'{cid}.npz'
                elif sum(not v for k,v in alternative['checks'].items() if k!='root_step_diagnostic') < sum(not v for k,v in result['checks'].items() if k!='root_step_diagnostic'):
                    result, path = alternative, ROOT/'foot_pose_v3/combined'/f'{cid}.npz'
            if not result['geometry_pass']:
                roots.fit(rec, dict(fit_path=str(path.relative_to(ROOT))))
                audit.ROOT = ROOT/'foot_pose_v4'
                result = audit.audit(rec, cpu)
                path = ROOT/'foot_pose_v4/combined'/f'{cid}.npz'
            if not result['geometry_pass']:
                failures[cid] = dict(stage='full_support_audit', checks=result['checks'])
            else:
                selected[cid] = dict(folder=str(path.parents[1].relative_to(ROOT)),
                                    fit_path=str(path.relative_to(ROOT)),
                                    audit_path=str((path.parents[1]/'audit'/f'{cid}.json').relative_to(ROOT)),
                                    geometry_pass=True, source_pass=result['source_pass'],
                                    source_checks=result['source_checks'], before=result['before'], after=result['after'])
        except Exception as error:
            import traceback
            failures[cid] = dict(error=str(error), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        common.save(OUT/'reconstruction_progress.json', dict(total=len(records), processed=number,
                    accepted=len(selected), selected=selected, failures=failures))
        print('INCREMENT', number, '/', len(records), cid, 'accepted', cid in selected, flush=True)
        torch.cuda.empty_cache()
    common.save(ROOT/'selection.json', dict(selected=selected, summary=dict(total=len(records),
                selected_geometry_pass=len(selected), actual_3d_accuracy_validated=False)))


def package():
    import common
    from combined_population import ROOT, manifest
    import package_combined_population as packager
    selection = json.loads((ROOT/'selection.json').read_text())
    index = json.loads((ROOT/'original_index.json').read_text())
    by_id = {c['id']: c for c in index['clips']}
    stage = OUT/'site_stage'
    packager.FOLDER = 'combined-increment-20261029-v1'
    packager.VERSION = 'gvhmr-ground-motionbert-20261029-increment-v1'
    clips, reports = [], []
    for rec in manifest()['records']:
        cid = rec['id']
        if cid not in selection['selected']:
            continue
        clip, report = packager.pack(rec, by_id[cid], selection['selected'][cid], stage)
        for key in ['pose', 'poseRefined', 'poseThumbnail']:
            if not clip.get(key):
                continue
            relative = clip[key]
            target = stage/'data'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE/'data'/relative, target)
        clips.append(clip)
        reports.append(report)
    common.save(stage/'data/clips.json', dict(index, clips=clips,
                packageVersion=packager.VERSION, packageStats=dict(clipsWith3d=len(clips),
                clipsWithGvhmr=len(clips), clipsWithWham=0, clipsWithRawWham=0)))
    common.save(OUT/'release_records.json', reports)
    print('STAGED', len(clips), 'fully reconstructed clips', flush=True)


if __name__ == '__main__':
    from runtime import ensure_runtime
    ensure_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['bootstrap', 'upstream', 'combine', 'package'])
    args = parser.parse_args()
    globals()[args.stage]()
