"""Independent sparse-SMPL FK check of actual arm outputs; no KBO GT claim."""
from common import *
from arm_cocking_refine import deployment_guard
import argparse

def main():
 deployment_guard()
 import torch
 from pitch_study_fit import StudyFitter
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2','v3'],default='v3');a=ap.parse_args();folder=OUT/('arm_cocking_'+a.version);protocol=json.loads((folder/'protocol.json').read_text());body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False);records=[]
 for cid in protocol['tune']+protocol['confirmation']:
  report=json.loads((folder/'reports'/f'{cid}.json').read_text());base=load(folder/'fits/current_GVHMR'/f'{cid}.npz');beta=base['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;body.prepare_shape(beta)
  for variant in report['variants']:
   z=load(folder/'fits'/variant/f'{cid}.npz');frames=np.unique([0,len(z['frame_index'])//2,len(z['frame_index'])-1])
   with torch.no_grad():_,an,_=body.model(torch.tensor(z['smpl_rotations'][frames],dtype=torch.float32),torch.tensor(np.repeat(beta[None],len(frames),0),dtype=torch.float32))
   model=an.numpy().astype(float);actual=z['anatomical_keypoints'][frames].astype(float);error=float(abs((model-model[:,0,None])-(actual-actual[:,0,None])).max());assert error<.00001,(cid,variant,error);records.append(dict(id=cid,variant=variant,fk_frames=frames.tolist(),native_fk_max_error_model=error,full_mesh_surface_claim=a.version=='v1',actual_3d_accuracy_claim=False))
 summary=dict(version=a.version,clips=12,models=len(records),fk_frames=sum(len(r['fk_frames']) for r in records),native_fk_error_max_model=max(r['native_fk_max_error_model'] for r in records),scope='Exact model-kinematic integrity, not measured KBO 3D accuracy',foot_scope='Baseline frozen for V2/V3; no new contact evidence');save(folder/'native_fk_audit.json',dict(summary=summary,results=records));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
