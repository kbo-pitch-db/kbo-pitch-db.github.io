"""Extend the existing full-frame, people/graphics-masked warp to all clips.

The already inspected implementation is reused unchanged in isolated worker
directories. Only its records/output path and score helper are injected.
"""
from common import *
from all813_release import ROOT,population
import argparse,importlib.util,shutil

def main():
 import torch
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15,0)
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 records=population()[a.worker::a.workers];worker=ROOT/f'background_worker{a.worker}';worker.mkdir(exist_ok=True);save(worker/'sample.json',records)
 for r in records:
  for suffix in ['npz','json']:
   source=PREVIOUS/'background'/f'{r["id"]}.{suffix}';dest=worker/'background'/source.name
   if source.exists() and not dest.exists():dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
 source=PREVIOUS/'code/experiments/pitch20261027/background.py';spec=importlib.util.spec_from_file_location('archived_masked_background',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.OUT=worker;module.observed_scores=scores
 save(worker/'provenance.json',dict(implementation_sha256=sha(source),records=len(records),full_original_frames=True,people_and_broadcast_border_masks=True,metric_slam=False,original_inputs_modified=False))
 module.main()

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
