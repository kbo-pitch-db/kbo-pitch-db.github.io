"""Stronger consistent pivot likelihood; same source gates and F4 stride prior."""
from common import *
import rear_pivot_consistent as solver

def main():
 solver.CONFIG={**solver.CONFIG,'version':'rear-pivot-v9','steps':1000,'pivot_weights':[10.,30.]}
 solver.OUTPUT_NAME='rear_pivot_v9'
 folder=OUT/solver.OUTPUT_NAME;folder.mkdir(exist_ok=True)
 save(folder/'solver_provenance.json',dict(wrapper_sha256=sha(__file__),solver_sha256=sha(solver.__file__),config=solver.CONFIG,source_gate_relaxed=False))
 solver.main()

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
