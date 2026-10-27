"""Unlisted synchronized NLF+MotionBERT / latest GVHMR comparison.

No inference/refitting: export the actual cached poses with one constant
similarity per model. Source pixels and licensed meshes never enter Pages.
"""
from common import *
from stride_review import native_leg
import gzip

def main():
 site=CODE/'studies/arm-comparison-20261028';site.mkdir(parents=True,exist_ok=True)
 clips={c['id']:c for c in json.loads((CODE/'data/clips.json').read_text())['clips']}
 registry=json.loads((CODE/'data/preview_rear_pivot_registry.json').read_text());catalog=[]
 for cid in registry['clips']:
  nlf=OUT/'fits_nlf_hybrid/nlf_mb_hybrid_temporal_f4'/f'{cid}.npz';gv=OUT/'rear_pivot_selected_v14'/f'{cid}.npz'
  if not nlf.exists():continue
  g=load(gv);n=load(nlf);meta=json.loads(gv.with_suffix('.json').read_text());clip=clips[cid]
  two=json.loads((CODE/'data'/clip['poseRefined']).read_text());times=np.array([f['time'] for f in two['frames']]);source_times=np.array([f['source_time'] for f in two['frames']]);source_frames=np.array([f['source_frame'] for f in two['frames']])
  normal=g['normal'].astype(float);normal/=np.linalg.norm(normal);x=np.array([1.,0.,0.]);x-=x@normal*normal;x/=np.linalg.norm(x);rotation=np.stack([x,normal,np.cross(x,normal)])
  assert np.linalg.det(rotation)>.99999
  rear=meta['selected_metrics']['rear_marker'];first=meta['selected_metrics']['rear_interval'][0];models={};cloud=[]
  for key,label,z,path in [('nlf_mb','NLF + MotionBERT · 팔 보정',n,nlf),('gvhmr','GVHMR · 최신 접지 보정',g,gv)]:
   assert np.array_equal(z['frame_index'],source_frames)
   assert np.max(abs(z['source_time']-source_times))<1e-6
   leg=native_leg(z['anatomical_keypoints']);origin=z['foot_keypoints'][first,rear].astype(float)+z['translation'][first]
   transform=lambda k:(z[k].astype(float)+z['translation'][:,None]-origin)@rotation.T/leg
   anatomy=transform('anatomical_keypoints');feet=transform('foot_keypoints');cloud.extend([anatomy.reshape(-1,3),feet.reshape(-1,3)])
   frames=[dict(anatomy=np.round(a,6).tolist(),feet=np.round(f,6).tolist(),source_frame=int(sf),source_time=float(st),time=float(t)) for a,f,sf,st,t in zip(anatomy,feet,source_frames,source_times,times)]
   models[key]=dict(label=label,frames=frames,source_fit_sha256=sha(path),display_transform=dict(rotation_camera_to_scene=rotation,constant_origin_camera=origin,constant_model_leg_divisor=leg,per_frame_recentering=False),method='nlf_mb_hybrid_temporal_f4' if key=='nlf_mb' else 'gvhmr-rear-pivot-v14')
  points=np.concatenate(cloud);bounds=dict(min=points.min(0).tolist(),max=points.max(0).tolist())
  for m in models.values():m['scene']=dict(frame='shared-camera-basis-initial-rear-toe',show_floor=True,floor=dict(height=0.,estimated=True,semantics='Display reference through initial rear toe, not a newly fitted or measured floor'),comparison_bounds=bounds,per_frame_recentering=False)
  payload=dict(id=cid,player=clip['player'],fps=two['fps'],times=times,source_times=source_times,source_frames=source_frames,pose2d='../../data/'+clip['poseRefined'],source_url=clip['sourceUrl'],models=models,comparison_bounds=bounds,rear_marker=rear,rear_interval=meta['selected_metrics']['rear_interval'],human_review_complete=False,absolute_stride_validated=False,coordinate_policy='One constant initial rear-toe origin and own fixed modeled leg divisor per model; common GVHMR estimated support normal in camera basis; no per-frame shifts, scaling, floor correction or new fitting',comparison_scope='Existing NLF+MotionBERT custom arm-transplant plus temporal fitting versus latest GVHMR joint/contact fitting; estimator and postprocessing differ, not an isolated estimator ablation')
  relative=f'{cid}.json.gz';raw=json.dumps(clean(payload),ensure_ascii=False,separators=(',',':')).encode();dest=site/relative;dest.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
  catalog.append(dict(id=cid,player=clip['player'],date=clip['date'],file=relative,frames=len(times),sha256_gzip=sha(dest)))
 save(site/'catalog.json',catalog)
 save(site/'provenance.json',dict(packager_sha256=sha(__file__),clips=len(catalog),published_default_changed=False,original_broadcast_media_included=False,new_inference_or_fitting=False,models='Actual cached NLF+MotionBERT custom arm-transplant fitting and GVHMR rear-pivot-v14',canonical_index_sha256=sha(CODE/'data/clips.json')))
 print('ARM COMPARISON',len(catalog),'clips')

if __name__=='__main__':main()
