"""Audit a targeted support repair, stage it, and optionally update its public pointer.

The original all813 results and original 2D remain immutable. A patch records
its own source, contact and FK checks; it does not relabel the frozen audit.
"""
from common import *
from all813_release import ROOT, enable_primary_shot_policy
from support_transfer_root import save_geometry
from independent_support import episode_metrics
from audit_rear_pivot import source_detail
from scipy.spatial.transform import Rotation
import argparse, copy, gzip, shutil

RULE = dict(primary_toe_extent_leg=.03, other_toe_extent_leg=.05,
            floor_penetration_p95_leg=.04, floor_penetration_max_leg=.08,
            stride_change_p95_leg=.05, source_regression_limit_px=2.)


def select_and_audit(cid):
    import torch
    from pitch_study_fit import StudyFitter
    torch.set_num_threads(1)
    parent = OUT / 'independent_support_v7' / (cid + '.npz')
    data, report = load(parent), json.loads(parent.with_suffix('.json').read_text())
    baseline = load(ROOT / 'selected' / (cid + '.npz'))
    assert sha(ROOT / 'selected' / (cid + '.npz')) == report['source_fit_sha256']
    notes = report['marker_states']
    assert notes
    leg = float(baseline['model_leg'])
    candidates = []
    for row in report['hypotheses']:
        i = row['index']
        candidate = {**baseline, 'keypoints': data['all_keypoints'][i],
                     'anatomical_keypoints': data['all_anatomical_keypoints'][i],
                     'foot_keypoints': data['all_foot_keypoints'][i],
                     'smpl_rotations': data['all_smpl_rotations'][i],
                     'translation': data['root_hypotheses_translation'][i],
                     'focal': data['all_focal'][i]}
        episodes = episode_metrics(candidate, notes)
        first = set()
        support_ok = True
        for e in episodes:
            primary = e['marker'] not in first
            first.add(e['marker'])
            support_ok &= e['extent_from_start_leg'] <= RULE[
                'primary_toe_extent_leg' if primary else 'other_toe_extent_leg']
        f = candidate['foot_keypoints'] + candidate['translation'][:, None]
        signed = (f * data['all_plane_normals'][i]).sum(-1) + data['all_plane_offsets'][i]
        penetration = np.maximum(-signed, 0) / leg
        knee = Rotation.from_matrix(candidate['smpl_rotations'][:, [4, 5]].reshape(-1, 3, 3)).as_rotvec()
        joint_ok = bool(knee[:, 0].min() >= -.0801 and knee[:, 0].max() <= 2.8001
                        and abs(knee[:, 1:]).max() <= .1801)
        checks = dict(source=bool(row['source_gate']), support=bool(support_ok), joints=joint_ok,
            stride=row['stride_change_p95_leg'] <= RULE['stride_change_p95_leg'],
            floor_p95=np.percentile(penetration, 95) <= RULE['floor_penetration_p95_leg'],
            floor_max=penetration.max() <= RULE['floor_penetration_max_leg'])
        candidates.append({**row, 'episodes': episodes, 'checks': checks,
            'floor_penetration_p95_leg': float(np.percentile(penetration, 95)),
            'floor_penetration_max_leg': float(penetration.max()),
            'support_consistency_passed': bool(all(checks.values()))})
    eligible = [r for r in candidates if r['support_consistency_passed']]
    dest = OUT / 'independent_support_selected_v7' / (cid + '.npz')
    if not eligible:
        save(dest.with_suffix('.json'), dict(id=cid, status='no_acceptable_repair',
                                          rule=RULE, candidates=candidates))
        raise ValueError('No candidate passes the unchanged source and physical screens')
    chosen = min(eligible, key=lambda r: r['source_p50_px'])
    i = chosen['index']
    z = {**baseline, **{key: data[source][i] for key, source in [
        ('keypoints', 'all_keypoints'), ('anatomical_keypoints', 'all_anatomical_keypoints'),
        ('foot_keypoints', 'all_foot_keypoints'), ('smpl_rotations', 'all_smpl_rotations'),
        ('translation', 'root_hypotheses_translation'), ('focal', 'all_focal')]}}
    _, _, _, h, _ = init_inputs(cid)
    immutable = next(r for r in json.loads((ROOT / 'input_manifest.json').read_text())['inputs'] if r['id'] == cid)
    assert sha(DATA / immutable['final_2d_path']) == immutable['final_2d_sha256']
    before, _ = source_detail(baseline, h, data['contact_weight'])
    after, uv = source_detail(z, h, data['contact_weight'])
    assert np.array_equal(z['smpl_betas'], baseline['smpl_betas'])
    assert np.array_equal(z['smpl_rotations'][:, 16:], baseline['smpl_rotations'][:, 16:])
    assert np.array_equal(z['source_time'], h['source_time'])
    assert np.array_equal(z['frame_index'], h['frame_index'])
    assert all(np.isfinite(z[k]).all() for k in ['keypoints', 'anatomical_keypoints',
               'foot_keypoints', 'smpl_rotations', 'translation'])
    assert (z['keypoints'] + z['translation'][:, None])[..., 2].min() > .2
    parents = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
    bones = np.linalg.norm(z['anatomical_keypoints'][:, 1:] - z['anatomical_keypoints'][:, parents[1:]], axis=-1)
    bone_error = float(abs(bones - np.median(bones, axis=0)).max())
    assert bone_error < .00002
    body = StudyFitter(WORKSPACE / 'external/WHAM', 'cpu', use_cuda_graph=False)
    beta = z['smpl_betas']; body.prepare_shape(beta)
    with torch.no_grad():
        p, a, f = body.model(torch.tensor(z['smpl_rotations'], dtype=torch.float32),
            torch.tensor(np.repeat(beta[None], len(z['frame_index']), axis=0), dtype=torch.float32))
    fk = max(float(abs(v.numpy() - z[k]).max()) for k, v in
             [('keypoints', p), ('anatomical_keypoints', a), ('foot_keypoints', f)])
    assert fk < .00001
    save_geometry(dest, z, z['translation'], uv, data['contact_weight'],
                  data['all_plane_normals'][i], float(data['all_plane_offsets'][i]))
    audit = dict(id=cid, version='independent-support-v7', selected=i, rule=RULE,
        normalization_leg_model=leg,
        normalization_semantics='Frozen model_leg reference carried from the population fit, not measured leg length',
        candidates=candidates, selected_metrics=chosen, before=before, after=after,
        before_episodes=episode_metrics(baseline, notes), after_episodes=chosen['episodes'],
        marker_states=notes, phase=report['phase'], fit_config=report['config'],
        selected_pose_sha256=sha(dest), fit_parent_sha256=sha(parent),
        source_fit_sha256=report['source_fit_sha256'],
        original_2d_sha256=sha(DATA / rows('motion_wham.csv')[cid]['pose_2d_refined']),
        fk_checked_frames=len(z['frame_index']), fk_max_error_model=fk,
        bone_length_variation_max_model=bone_error, shape_unchanged=True,
        local_upper_limb_rotations_unchanged=True,
        status='support_consistency_passed', human_review_complete=False,
        source_visual_comparison='Codex inspected local source frames; not human annotation',
        actual_3d_accuracy_validated=False,
        parameter_search='This reported failure was used to tune v1-v7; original attempts remain in private versioned directories',
        limitations=['One user-reported clip used for fitting and parameter selection; no held-out accuracy evidence',
            'Contact intervals are image-supported hypotheses, not manually labeled force/contact data',
            'Source errors and worst-frame regressions are retained; monocular depth, scale and stride unvalidated'])
    save(dest.with_suffix('.json'), audit)
    return dest, z, audit


def package(site, path, z, audit, promote):
    cid = audit['id']; original_index = (site / 'data/clips.json').read_bytes()
    index = json.loads(original_index)
    clip = next(c for c in index['clips'] if c['id'] == cid)
    assert clip['sceneSignature'] == audit['source_fit_sha256'], 'Public parent changed; do not apply a patch twice'
    original = (site / 'data' / clip['pose3d']).read_bytes()
    pose = json.loads(gzip.decompress(original))
    two_bytes = (site / 'data' / clip['poseRefined']).read_bytes()
    two = json.loads(two_bytes)
    assert len(pose['frames']) == len(two['frames']) == len(z['frame_index'])
    normal = z['normal'].astype(float); normal /= np.linalg.norm(normal)
    x = np.array([1.,0.,0.]); x -= (x @ normal) * normal; x /= np.linalg.norm(x)
    rotation = np.stack([x, normal, np.cross(x, normal)])
    assert np.linalg.det(rotation) > .99999
    first = audit['marker_states'][0]
    point = z['foot_keypoints'][first['start'], first['marker']] + z['translation'][first['start']]
    origin = point - (point @ normal + float(z['pilot_support_offset'])) * normal
    for key, field in [('keypoints', 'keypoints'), ('anatomical_keypoints', 'anatomy'), ('foot_keypoints', 'feet')]:
        values = (z[key].astype(float) + z['translation'][:,None] - origin) @ rotation.T
        for i, frame in enumerate(pose['frames']):
            points = np.round(values[i], 6).tolist()
            frame[field] = ([p + [old[3]] for p, old in zip(points, frame[field])]
                            if field == 'keypoints' else points)
    version = 'gvhmr-contact-20261028-v1-support7'
    relative = f'reconstructions/gvhmr-release-v1/support-v7/{cid}.json.gz'
    quality = f'quality/gvhmr-release-v1/support-v7/{cid}.json'
    reasons = ['monocular_global_depth_and_absolute_stride_unvalidated']
    pose.update(source_fit_sha256=sha(path), scene_signature=sha(path),
                postprocessing_version=audit['version'])
    pose['scene'].update(rotation_camera_to_scene=rotation.tolist(), origin_camera=origin.tolist(),
        origin_source_frame=int(z['frame_index'][first['start']]),
        focal_pixels_estimate=float(z['focal']), per_frame_recentering=False)
    pose['scene'].pop('comparison_bounds', None)
    pose['quality'].update(contact='hypothesis_unvalidated', rear_contact_evaluable=True,
        support_consistency_passed=True, reasons=reasons, human_review_complete=False)
    pose['fit_provenance'] = dict(selected_fit_sha256=sha(path), base_fit_sha256=audit['source_fit_sha256'],
        normalization_leg_model=audit['normalization_leg_model'],
        postprocessing_version=audit['version'], contact_episodes=audit['marker_states'],
        phase=audit['phase'], metrics=audit['selected_metrics'],
        frozen_population_consistency_screen_recomputed=False)
    stage = path.parent / 'site_stage'
    dest = stage / 'data' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.compress(json.dumps(clean(pose), ensure_ascii=False, separators=(',',':')).encode(), compresslevel=9, mtime=0))
    audit['public_payload_sha256'] = sha(dest)
    save(stage / 'data' / quality, {**pose['quality'], 'support_audit':audit})
    report_relative = f'data/support_repairs/{cid}-v7.json'
    save(stage / report_relative, audit)
    clip.update(pose3d=relative, motionQuality=quality, sceneSignature=sha(path),
                pose3dReasons=reasons, reconstructionVersion=version,
                supportAudit=report_relative.removeprefix('data/'),
                postprocessingVersion=audit['version'])
    index['packageVersion'] = version
    index['packageStats']['clipsWithContactFit'] += 1
    save(stage / 'data/clips.json', index)
    summary = json.loads((site / 'data/gvhmr_release_summary.json').read_text())
    for key in ['contact_available', 'contact_hypotheses_available', 'rear_contact_evaluable']: summary[key] += 1
    for key in ['contact_unknown', 'contact_hypotheses_unknown']: summary[key] -= 1
    summary.update(patch_version=audit['version'], support_repaired_clips=[cid],
        frozen_consistency_screen_counts_preserved=True)
    save(stage / 'data/gvhmr_release_summary.json', summary)
    assert (site / 'data' / clip['poseRefined']).read_bytes() == two_bytes
    if promote:
        archived_index = site / 'data/releases/gvhmr-20261028/original_index.json.gz'
        if not archived_index.exists():
            archived_index.parent.mkdir(parents=True, exist_ok=True)
            archived_index.write_bytes(gzip.compress(original_index, compresslevel=9, mtime=0))
        for relative_path in [relative, quality, report_relative.removeprefix('data/'), 'gvhmr_release_summary.json']:
            target=site / 'data' / relative_path; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / 'data' / relative_path, target)
        save(site / 'data/clips.json', index)
    print(json.dumps(dict(id=cid, selected=audit['selected'], promoted=promote,
        support_before=audit['before_episodes'], support_after=audit['after_episodes'],
        source_before=audit['before'], source_after=audit['after']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    from runtime import ensure_runtime
    ensure_runtime(); enable_primary_shot_policy()
    parser=argparse.ArgumentParser(); parser.add_argument('--id', required=True)
    parser.add_argument('--site', type=Path, required=True); parser.add_argument('--promote', action='store_true')
    args=parser.parse_args(); path,z,audit=select_and_audit(args.id)
    package(args.site, path, z, audit, args.promote)
