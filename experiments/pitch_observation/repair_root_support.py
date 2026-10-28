"""Audit every frame for unsupported whole-body flight, then refit actual root/feet."""
from common import *
from combined_population import ROOT,manifest
import argparse,traceback
# These image-confirmed identity transitions require new observations/GVHMR,
# rather than forcing the old batter observation onto the pitcher's ground.
RECOVER=['0b70417b94ab_c0010','3ef5720dccec_c0010','782c3ca51bd5_c0004',
 '1161553ed7d7_c0056','3ef5720dccec_c0021','e72b3a782760_c0009',
 'b54e0e701d77_c0006','02dad52fd781_c0049','4e09dd724a8c_c0011']

def run(worker,workers):
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.22,0)
    import audit_combined_supported as audit,combined_root_support as root
    from pitch_study_fit import StudyFitter
    body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False);failures={};done=[]
    for rec in manifest()['records'][worker::workers]:
        cid=rec['id']
        if cid in RECOVER:continue
        try:
            candidates=[]
            for name in ['foot_pose_v2','foot_pose_v3']:
                folder=ROOT/name;path=folder/'combined'/f'{cid}.npz'
                if not path.exists():continue
                audit.ROOT=folder;r=audit.audit(rec,body)
                candidates.append((path,r))
            accepted=[(p,r) for p,r in candidates if r['geometry_pass']]
            if not accepted:
                source=min(candidates,key=lambda x:sum(not v for k,v in x[1]['checks'].items() if k!='root_step_diagnostic'))[0]
                root.fit(rec,dict(fit_path=str(source.relative_to(ROOT))));audit.ROOT=ROOT/'foot_pose_v4'
                result=audit.audit(rec,body)
                if not result['geometry_pass']:failures[cid]=dict(checks=result['checks'])
            done.append(cid)
        except Exception as e:failures[cid]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        save(ROOT/f'root_support_worker{worker}.json',dict(done=len(done),failures=failures))

if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args();run(a.worker,a.workers)
