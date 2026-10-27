"""Bounded knee FK, rear toe closure and source-weighted joint fitting."""
from common import *
import rear_pivot_consistent as solver

def main():
 solver.CONFIG={**solver.CONFIG,'version':'rear-pivot-bounded-v13','steps':1100,
  'pivot_weights':[10.,30.],'plane_angles':[-.20,-.10,0.,.10],
  'foot_observation_weight':2.,'pivot_normal_weight':1.,'rear_pivot_only':True,
  'max_feet_p95_increase_px':2.,'max_rear_point_extent_3d_leg':.05,
  'bounded_knees':True,'stride_vector_weight':.10,'stride_distance_weight':4.,
  'knee_limits_radians':[-.08,2.8],'knee_side_limit_radians':.18}
 solver.OUTPUT_NAME='rear_pivot_v13';folder=OUT/solver.OUTPUT_NAME;folder.mkdir(exist_ok=True)
 save(folder/'solver_provenance.json',dict(wrapper_sha256=sha(__file__),solver_sha256=sha(solver.__file__),config=solver.CONFIG,source_gate_relaxed=False,previous_candidate_not_adopted='v11 stationarity could use excessive knee extension'))
 solver.main()

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
