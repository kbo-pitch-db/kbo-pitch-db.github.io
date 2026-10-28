"""Stream final FK/contact audits and select only geometrically passing outputs."""
from common import *
from combined_population import ROOT,manifest
import audit_combined_polished as auditor
import argparse,time,traceback


def main(once=False):
    import torch
    from pitch_study_fit import StudyFitter
    torch.set_num_threads(1);body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    records=manifest()['records'];failures={}
    while True:
        candidates={r['id']:[] for r in records}
        for folder in sorted((ROOT/'polished_v1').glob('*')):
            if not folder.is_dir():continue
            auditor.ROOT=folder
            for rec in records:
                try:
                    result=auditor.audit(rec,body)
                    if result is not None:
                        candidates[rec['id']].append(dict(folder=str(folder.relative_to(ROOT)),
                            geometry_pass=result['geometry_pass'],source_pass=result['source_pass'],
                            source_checks=result['source_checks'],after=result['after'],before=result['before'],
                            audit_path=str((folder/'audit'/f'{rec["id"]}.json').relative_to(ROOT)),
                            fit_path=str((folder/'combined'/f'{rec["id"]}.npz').relative_to(ROOT))))
                except Exception as e:
                    failures[folder.name+':'+rec['id']]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        selected={};missing=[];source_regressions=[]
        for rec in records:
            cid=rec['id'];valid=[c for c in candidates[cid] if c['geometry_pass']]
            if not valid:missing.append(cid);continue
            # Keep the strict source screen visible even when contact fitting
            # trades away a few pixels of shoe-marker agreement. Never relabel
            # a failed source check as a pass or call this measured 3D accuracy.
            def rank(c):
                important=sum(not c['source_checks'][k] for k in ['source_p50_px','wrist_p95_px'])
                return (important,not c['source_pass'],c['after']['feet_p95_px'] or 100,
                        c['after']['source_p50_px'] or 100)
            choice=min(valid,key=rank);selected[cid]=choice
            if not choice['source_pass']:source_regressions.append(cid)
        report=dict(total=len(records),selected_geometry_pass=len(selected),missing_geometry=missing,
            strict_source_pass=sum(c['source_pass'] for c in selected.values()),
            strict_source_regression_ids=source_regressions,
            important_source_regression_ids=[cid for cid,c in selected.items() if not c['source_checks']['source_p50_px'] or not c['source_checks']['wrist_p95_px']],
            candidate_counts={cid:len(c) for cid,c in candidates.items()},failures=failures,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            actual_3d_accuracy_validated=False,auto_publish=False)
        save(ROOT/'selection.json',dict(summary=report,selected=selected,candidates=candidates))
        print('FINAL SELECTION',len(selected),'/',len(records),'SOURCE',report['strict_source_pass'],flush=True)
        if once:return
        time.sleep(25)


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();main(a.once)
