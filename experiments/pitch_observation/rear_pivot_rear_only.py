"""Pivot closure in normal+tangent directions and original foot observations.

Negative and positive camera-space floor inclines are hypotheses: mound slope
and camera pitch are unmeasured. One plane per candidate, never per frame.
Raw foot reprojection is screened too; reducing skating must not hide a worse
source match or replace it with vertical floating.
"""
from common import *
import rear_pivot_consistent as solver

def main():
 solver.CONFIG={**solver.CONFIG,'version':'rear-pivot-v11','steps':1000,
  'pivot_weights':[10.,30.],'plane_angles':[-.20,-.10,0.,.10],
  'foot_observation_weight':2.,'pivot_normal_weight':1.,
  'max_feet_p95_increase_px':2.,'rear_pivot_only':True,'max_rear_point_extent_3d_leg':.05}
 solver.OUTPUT_NAME='rear_pivot_v11';folder=OUT/solver.OUTPUT_NAME;folder.mkdir(exist_ok=True)
 save(folder/'solver_provenance.json',dict(wrapper_sha256=sha(__file__),solver_sha256=sha(solver.__file__),config=solver.CONFIG,source_gate_relaxed=False,raw_feet_gate_added=True))
 solver.main()

if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
