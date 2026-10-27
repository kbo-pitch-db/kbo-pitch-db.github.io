"""Opt-in inference population runner; never edits the fixed48 or dataset.

A single stage per process keeps upstream packages' `lib` namespaces isolated.
All813 inference requires all stage-specific cached inputs/assets. Failures stay
visible and can be resumed with the exact same command and output directory.
"""
import argparse,importlib,sys
from pathlib import Path
import common

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--population',choices=['fixed48','all813'],default='fixed48');ap.add_argument('--stage',required=True,choices=['rgb_initializers','gvhmr_network','motionbert_unmerged','public_temporal_adaptation','motionbert_smpl','wham_unmerged','rgb_observations','nlf_rgb','optimize']);ap.add_argument('stage_arguments',nargs=argparse.REMAINDER);a=ap.parse_args()
 if a.population=='all813':
  records=common.json.loads((common.PREVIOUS/'audit_all.json').read_text());records=[dict(r,split='batch') for r in records];common.save(common.OUT/'population_all813.json',records)
  def population(split='all',small=False):return records if split in ['all','batch'] else [r for r in records if r['split']==split]
  common.sample=population
 module=importlib.import_module(a.stage);arguments=a.stage_arguments
 if arguments and arguments[0]=='--':arguments=arguments[1:]
 sys.argv=[str(Path(module.__file__)),*arguments];module.main()
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
