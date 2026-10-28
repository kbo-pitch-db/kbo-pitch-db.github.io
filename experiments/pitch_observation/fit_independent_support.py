"""Refit explicitly requested clips without changing the frozen population fits."""
from common import *
import argparse
from all813_release import ROOT, population, enable_primary_shot_policy
import support_episode_solver as solver
from rear_pivot_bounded import settings
from independent_support import contact_state, episode_metrics, VERSION


def main():
    import torch
    from pitch_study_fit import StudyFitter
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    enable_primary_shot_policy()
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.15, 0)
    solver.CONFIG = {**settings(), 'version': 'independent-support-fit-v7',
        'detector_version': VERSION,
        'detector_sha256': sha(Path(__file__).with_name('independent_support.py')),
        'rear_pivot_only': False, 'cuda_graph': args.device.startswith('cuda'),
        'pivot_weights': [4.5, 6., 7.5], 'plane_angles': [-.12, -.10, -.08],
        'foot_observation_weight': 30., 'penetration_weight': 20., 'steps': 2200,
        'foot_rotation_limit': .7, 'arm_observation_weight': 6.,
        'pivot_normal_weight': .05,
        'solver_sha256': sha(solver.__file__)}
    solver.contact_state = contact_state
    solver.SOURCE_FOLDER = ROOT / 'selected'
    solver.REFERENCE_FOLDER = OUT / 'no_archived_independent_contact_reference'
    solver.OUTPUT_NAME = 'independent_support_v7'
    dest = OUT / solver.OUTPUT_NAME
    dest.mkdir(exist_ok=True)
    save(dest / 'protocol.json', dict(config=solver.CONFIG,
        detector_sha256=sha(Path(__file__).with_name('independent_support.py')),
        solver_sha256=sha(solver.__file__), population_release_changed=False))
    body = StudyFitter(WORKSPACE / 'external/WHAM', args.device, use_cuda_graph=False)
    rec = next(r for r in population() if r['id'] == args.id)
    report = solver.fit(rec, body, solver.CONFIG['steps'])
    path = dest / (args.id + '.npz')
    if path.exists():
        before, after = load(solver.SOURCE_FOLDER / path.name), load(path)
        report.update(before_episodes=episode_metrics(before, report['marker_states']),
                      after_episodes=episode_metrics(after, report['marker_states']))
        save(path.with_suffix('.json'), report)


if __name__ == '__main__':
    from runtime import ensure_runtime
    ensure_runtime()
    main()
