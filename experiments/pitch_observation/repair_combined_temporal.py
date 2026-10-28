"""Retry failed temporal screens with wider ankle-depth search; retain all failures."""
from common import *
from combined_population import ROOT,manifest
import argparse,traceback

def run(once=False):
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.18,0)
    import combined_foot_pose_retry as solver,audit_combined_temporal as auditor
    from pitch_study_fit import StudyFitter
    auditor.ROOT=ROOT/'foot_pose_v3';body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    records=manifest()['records'];selected=json.loads((ROOT/'selection_frozen.json').read_text())['selected'];failures={}
    while True:
        audited=0;failed=[]
        for rec in records:
            cid=rec['id'];ap=ROOT/'foot_pose_v2/audit'/f'{cid}.json'
            if not ap.exists():continue
            audited+=1
            if json.loads(ap.read_text())['geometry_pass']:continue
            if cid=='4e09dd724a8c_c0011':continue # separate corrected-subject reconstruction
            dest=auditor.ROOT/'audit'/f'{cid}.json'
            try:
                if not dest.exists():solver.fit(rec,selected[cid]);auditor.audit(rec,body)
                r=json.loads(dest.read_text())
                if not r['geometry_pass']:failed.append(cid)
            except Exception as e:
                failures[cid]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        save(ROOT/'foot_pose_v3/repair_progress.json',dict(base_audited=audited,base_total=len(records),retry_failed=failed,failures=failures))
        if once or audited==len(records):return
        time.sleep(20)

if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();run(a.once)
