"""Incrementally repair failed production screens; original candidates immutable."""
from common import *
from combined_population import ROOT,manifest
import importlib
ground=importlib.import_module(os.environ.get('KBO_COMBINED_RETRY_SOLVER','combined_support_retry'))
import combined_arms as arms
import audit_combined_population as audit
import argparse,time,traceback


def main():
    import torch
    from pitch_study_fit import StudyFitter
    p=argparse.ArgumentParser();p.add_argument('--id');p.add_argument('--once',action='store_true');p.add_argument('--max-frames',type=int,default=10000)
    p.add_argument('--geometry-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('KBO_COMBINED_RETRY_MEMORY','.48')),0)
    gpu=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False)
    cpu=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    folder=ground.ROOT
    if not (folder/'motionbert').exists():(folder/'motionbert').symlink_to(ROOT/'motionbert',target_is_directory=True)
    arms.ROOT=folder;audit.ROOT=folder
    records=manifest()['records'];failures={}
    while True:
        processed=[];pending=[]
        for rec in records:
            cid=rec['id']
            if a.id and cid!=a.id or rec['valid_frames']>a.max_frames:continue
            current=ROOT/'audit'/f'{cid}.json'
            if not current.exists():pending.append(cid);continue
            first=json.loads(current.read_text())
            if a.geometry_only and first['geometry_pass']:continue
            if first['geometry_pass'] and first['source_pass']:continue
            accepted=False
            for other in ROOT.glob('retry_*/audit/'+cid+'.json'):
                result=json.loads(other.read_text())
                if result['geometry_pass'] and result['source_pass']:accepted=True;break
            if accepted:continue
            done=folder/'audit'/f'{cid}.json'
            if done.exists():continue
            try:
                ground.fit(rec,gpu);arms.fit(rec);report=audit.audit(rec,cpu)
                processed.append(cid);print('RETRY',cid,'geometry',report['geometry_pass'],'source',report['source_pass'],flush=True)
            except Exception as e:
                failures[cid]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
            torch.cuda.empty_cache()
        save(folder/'progress.json',dict(pending=pending,processed_last_pass=processed,failures=failures))
        if a.once or not pending:return
        time.sleep(30)


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();main()
