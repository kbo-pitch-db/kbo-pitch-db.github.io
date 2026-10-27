#!/usr/bin/env python3
"""Reconstruct sampled saved fits with official full-mesh SMPL (internal check only)."""
import argparse,json
from pathlib import Path
import numpy as np
from package_scene import records

def main():
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=root/'dataset');p.add_argument('--wham-root',type=Path,default=root/'external/WHAM');p.add_argument('--device',default='cuda:1');p.add_argument('--ids',default='');a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.10,torch.device(a.device))
    from scene_fit import SceneFitter
    from package_pitch_scene import scene_records
    model=SceneFitter(a.wham_root,a.device);index=scene_records(a.dataset)
    ids=a.ids.split(',') if a.ids else [sorted(index)[i] for i in np.linspace(0,len(index)-1,8,dtype=int)]
    report=[]
    with torch.no_grad():
        for cid in ids:
            with np.load(a.dataset/index[cid]['pose_3d_scene']) as z:
                valid=np.flatnonzero(np.isfinite(z['anatomical_keypoints']).all((1,2)));frames=valid[np.linspace(0,len(valid)-1,min(5,len(valid)),dtype=int)]
                rotations=np.concatenate([z['smpl_global_orient'][frames],z['smpl_body_pose'][frames]],1)
                tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=a.device)
                body,anatomy,feet=model.model(tensor(rotations),tensor(np.repeat(z['smpl_betas'][None],len(frames),0)))
                errors={}
                for key,values in [('keypoints',body),('anatomical_keypoints',anatomy),('foot_keypoints',feet)]:
                    world=(values.cpu().numpy()+z['camera_translation'][frames,None]-z['scene_origin'])@z['scene_rotation'].T
                    error=float(np.max(np.abs(world-z[key][frames])));errors[key]=error
                    np.testing.assert_allclose(world,z[key][frames],atol=6e-5,rtol=0)
                report.append(dict(id=cid,source_frames=z['frame_index'][frames].tolist(),max_absolute_error_model_m=errors))
    result=dict(checked_clips=len(report),records=report,interpretation='Reconstruction consistency against official full-mesh SMPL, not measured pose or stride accuracy')
    (a.dataset/'wham_smpl_reconstruction_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    from wham_runtime import ensure_runtime
    ensure_runtime();main()
