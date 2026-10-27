"""Measure isolated representative fits without modifying frozen algorithms."""
from common import *
import argparse,resource,importlib
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();import torch
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['main','rgb'],default='main');a=ap.parse_args();torch.set_num_threads(1);torch.cuda.init();torch.cuda.reset_peak_memory_stats(0);start=time.monotonic();module=importlib.import_module('optimize' if a.kind=='main' else 'optimize_nlf_rgb');sys.argv=[module.__file__,'--id','4e09dd724a8c_c0031','--fit-dir','resource_probe_'+a.kind,'--initializers','WHAM_raw' if a.kind=='main' else 'nlf_rgb','--branches','temporal','--full-2d','--depth-preserve','2'];module.main();save(OUT/f'resource_probe_{a.kind}.json',dict(id='4e09dd724a8c_c0031',frames=103,focal_candidates=3,optimization_steps=[60,160,80],seconds=time.monotonic()-start,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(0),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device=torch.cuda.get_device_name(0),scope='Single isolated measured fit, not whole-population peak'))
