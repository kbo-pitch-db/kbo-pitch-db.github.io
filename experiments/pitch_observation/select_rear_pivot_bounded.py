"""Apply the unchanged source screens plus pivot, floor and knee consistency."""
from common import *
import select_rear_pivot as selector

def main():
 selector.PARENT_NAME='rear_pivot_v13';selector.SELECTED_NAME='rear_pivot_selected_v14'
 selector.RULE={**selector.RULE,'version':'rear-pivot-selected-v14','bounded_knees':True,
  'knee_x_limits_rad':[-.08,2.8],'knee_side_limit_rad':.18}
 folder=OUT/selector.SELECTED_NAME;folder.mkdir(exist_ok=True)
 save(folder/'selector_provenance.json',dict(wrapper_sha256=sha(__file__),selector_sha256=sha(selector.__file__),rule=selector.RULE,source_gate_relaxed=False))
 selector.main()

if __name__=='__main__':main()
