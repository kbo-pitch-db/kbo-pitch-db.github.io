"""Actual arm-preview payload audit against immutable main data, not 3D GT."""
from common import *
from arm_cocking_refine import FOLDER,deployment_guard
import gzip,argparse

def main():
 global FOLDER
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2','v3'],default='v3');args=ap.parse_args();version=args.version
 if version!='v1':FOLDER=__import__('arm_cocking_'+version).FOLDER
 expected={'v1':27,'v2':12,'v3':16}[version];site=CODE/'studies'/dict(v1='arm-cocking-v1-20261028',v2='arm-cocking-v2-20261028',v3='arm-cocking-20261028')[version];catalog=json.loads((site/'catalog.json').read_text());provenance=json.loads((site/'provenance.json').read_text());assert len(catalog)==12;assert provenance['main_index_sha256']==sha(CODE/'data/clips.json');mainclips={c['id']:c for c in json.loads((CODE/'data/clips.json').read_text())['clips']};protocol=json.loads((site/'protocol.json').read_text());assert not set(protocol['tune'])&set(protocol['confirmation']);assert not {mainclips[c]['playerKey'] for c in protocol['tune']}&{mainclips[c]['playerKey'] for c in protocol['confirmation']};assert not {c.split('_c')[0] for c in protocol['tune']}&{c.split('_c')[0] for c in protocol['confirmation']};results=[]
 for rec in catalog:
  path=site/rec['file'];assert sha(path)==rec['sha256_gzip'];data=json.loads(gzip.decompress(path.read_bytes()));two=json.loads((CODE/'data'/mainclips[rec['id']]['poseRefined']).read_text());released=json.loads(gzip.decompress((CODE/'data'/mainclips[rec['id']]['pose3d']).read_bytes()));assert data['source_frames']==[f['source_frame'] for f in two['frames']];assert data['source_times']==[f['source_time'] for f in two['frames']];assert data['times']==[f['time'] for f in two['frames']];assert len(data['models'])==expected
  base=data['models']['current_GVHMR'];a=np.array([f['anatomy'] for f in base['frames']]);leg=data['display_transform']['leg_divisor'];public=np.array([f['anatomy'] for f in released['frames']]);baseline_error=float(abs(a-public/leg).max());assert baseline_error<.00003
  actual_raw_arm_difference=0.;max_lower=0.
  for name,m in data['models'].items():
   assert len(m['frames'])==len(two['frames']);assert m['scene']['comparison_bounds']==base['scene']['comparison_bounds'];assert not m['scene']['per_frame_recentering'];assert m['scene']['floor']['height']==0
   z=np.array([f['anatomy'] for f in m['frames']]);feet=np.array([f['feet'] for f in m['frames']]);uv=np.asarray(m['projected_crop']);assert np.isfinite(z).all() and np.isfinite(feet).all() and np.isfinite(uv).all();lower=float(abs(z[:,:16]-a[:,:16]).max());max_lower=max(lower,max_lower);assert lower<.00003
   if name.startswith('raw_graft_') or name in ['raw_ray_mb','raw_ray_flip'] or version=='v2' and name.startswith('mb_native'):actual_raw_arm_difference=max(actual_raw_arm_difference,float(np.linalg.norm(z[:,[18,19,20,21]]-a[:,[18,19,20,21]],axis=-1).max()))
  assert actual_raw_arm_difference>.001,'No distinct MotionBERT arm initialization: '+rec['id'];results.append(dict(id=rec['id'],models=len(data['models']),frames=len(two['frames']),baseline_vs_main_error_max_leg=baseline_error,lower_body_change_max_leg=max_lower,distinct_raw_arm_change_max_leg=actual_raw_arm_difference))
 assert not any(p.suffix.lower() in ['.mp4','.jpg','.png','.jpeg','.npz','.pkl','.pt','.bin'] for p in site.rglob('*'))
 assert 'arm-cocking' not in (CODE/'index.html').read_text();assert 'noindex,nofollow,noarchive' in (site/'index.html').read_text();record=dict(clips=12,variants=expected,clip_variant_payloads=12*expected,exact_source_timing=True,main_index_unchanged=True,all_lower_body_preserved=True,shared_fixed_scene=True,original_pixels_published=False,actual_3d_accuracy_claim=False,results=results);save(FOLDER/'package_audit.json',record);save(site/'package_audit.json',record);print(json.dumps({k:v for k,v in record.items() if k!='results'},indent=2))

if __name__=='__main__':main()
