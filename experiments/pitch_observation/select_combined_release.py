"""Freeze the complete audited rigid-shoe population, never a label-only fallback."""
from common import *
from combined_population import ROOT,manifest

def main():
    selected={};reports=[]
    for rec in manifest()['records']:
        cid=rec['id'];candidates=[]
        override=ROOT/'subject_corrected_combined_v1/ready.json'
        subject=json.loads(override.read_text()) if override.exists() else None
        tail=ROOT/'tail_recovery_v1/combined/ready'/f'{cid}.json'
        if tail.exists():subject=json.loads(tail.read_text())
        paths=[ROOT/subject['fit_path']] if subject and subject['id']==cid else [ROOT/name/'combined'/f'{cid}.npz' for name in ['foot_pose_v2','foot_pose_v3','foot_pose_v4']]
        for path in paths:
            ap=path.parents[1]/'audit'/f'{cid}.json'
            if not ap.exists():continue
            audit=json.loads(ap.read_text());meta=json.loads(path.with_suffix('.json').read_text())
            assert audit['hashes']['combined']==sha(path),cid
            if not audit['geometry_pass'] or audit['rule']['version']!='combined-population-full-support-qc-v5':continue
            assert meta['motionbert_applied'] and meta['rigid_sole_refit']['constant_shape'],cid
            candidates.append((path,ap,audit))
        assert candidates,('No actual combined geometry pass',cid)
        path,ap,audit=min(candidates,key=lambda x:(not x[2]['source_checks']['source_p50_px'],not x[2]['source_pass'],x[2]['after']['feet_p95_px']))
        selected[cid]=dict(folder=str(path.parents[1].relative_to(ROOT)),fit_path=str(path.relative_to(ROOT)),
            audit_path=str(ap.relative_to(ROOT)),geometry_pass=True,source_pass=audit['source_pass'],
            source_checks=audit['source_checks'],before=audit['before'],after=audit['after'])
        reports.append(audit)
    assert len(selected)==813
    summary=dict(total=813,selected_geometry_pass=813,strict_source_pass=sum(r['source_pass'] for r in reports),
        strict_source_regression_ids=[r['id'] for r in reports if not r['source_pass']],
        important_source_regression_ids=[r['id'] for r in reports if not r['source_checks']['source_p50_px'] or not r['source_checks']['wrist_p95_px']],
        actual_3d_accuracy_validated=False,auto_publish=False,
        selection_policy='All actual rigid-shoe fits must pass unchanged contact/FK/bone/floor/root-continuity screens; strict source regressions retained')
    save(ROOT/'selection.json',dict(summary=summary,selected=selected));print('FROZEN',summary['selected_geometry_pass'],'SOURCE',summary['strict_source_pass'])

if __name__=='__main__':main()
