"""Additional detector observations after main deployment; never fitting inputs."""
from common import *
from arm_cocking_refine import FOLDER,deployment_guard
import argparse

def main():
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['tune','confirmation','all'],default='all');a=ap.parse_args();protocol=json.loads((FOLDER/'protocol.json').read_text());ids=protocol['tune']+protocol['confirmation'] if a.cohort=='all' else protocol[a.cohort]
 import alternative_2d
 records={r['id']:r for r in sample('all',False)};alternative_2d.sample=lambda split='all',small=True:[records[cid] for cid in ids];sys.argv=[alternative_2d.__file__];alternative_2d.main()

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
