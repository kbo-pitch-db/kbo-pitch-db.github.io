"""Connect Explorer to actual latest fits, preserving canonical 813/2D assets."""
from common import *
import argparse,gzip,copy

def build_pose(cid,label,method,path,base,pose2d):
 z=load(path);N=len(z['keypoints']);assert N==len(pose2d['frames']);assert np.max(abs(z['source_time']-np.array([f['source_time'] for f in pose2d['frames']])))<1e-6
 assert np.array_equal(z['frame_index'],[f['source_frame'] for f in pose2d['frames']]);rotation=np.diag([1.,-1.,-1.]);native=z['anatomical_keypoints'].astype(np.float64)+z['translation'].astype(np.float64)[:,None];good=np.flatnonzero(np.isfinite(native[:,0]).all(-1));anchor=native[good[0],0].copy()
 transform=lambda p:(p.astype(np.float64)+z['translation'].astype(np.float64)[:,None]-anchor)@rotation.T
 kp=transform(z['keypoints']);an=transform(z['anatomical_keypoints']);feet=transform(z['foot_keypoints']);r,c,s,h,w=init_inputs(cid);sc=scores(h)
 point=lambda p:np.round(p,6).tolist() if np.isfinite(p).all() else None
 frames=[]
 for i,f in enumerate(pose2d['frames']):
  points=[]
  for j,p in enumerate(kp[i]):
   pp=point(p);points.append([*pp,round(float(sc[i,j]),4)] if pp is not None else None)
  frames.append(dict(frame=f['frame'],source_frame=f['source_frame'],time=f['time'],source_time=f['source_time'],keypoints=points,anatomy=[point(p) for p in an[i]],feet=[point(p) for p in feet[i]],interpolated=h.get('interpolated',np.zeros_like(h['scores'],bool))[i,:17].tolist(),outlier=h.get('outlier',np.zeros_like(h['scores'],bool))[i,:17].tolist()))
 keys=['joints','primary_representation','keypoints_semantics','anatomy_joints','anatomy_parents','foot_joints'];pose={k:copy.deepcopy(base[k]) for k in keys};pose.update(schema_version=3,dimension=3,fps=pose2d['fps'],model=label,method=method,model_sha256='4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91' if method=='gvhmr_f4' else '61a2c684eba6b12bd5eb70560855e37af2d1c6322b8fcb4a9f5e096932591b92',source_fit_sha256=sha(path),score_semantics='2D observation scores only, never calibrated 3D confidence; native centers are model predictions',coordinate_system='Clip-fixed camera basis; X right, Y up, Z toward camera; estimated model meters, uncalibrated global trajectory',scene_signature=sha(path),scene=dict(frame='clip-fixed-camera-estimate',floor=None,show_floor=False,rotation_camera_to_scene=rotation.tolist(),origin_camera=anchor.tolist(),origin_source_frame=int(z['frame_index'][good[0]]),anchor_policy='One constant first-valid SMPL pelvis camera position; no framewise recentering or floor lift',calibration='Uncalibrated monocular; scale, root trajectory, contact and absolute stride not validated',per_frame_recentering=False,focal_pixels_estimate=float(z['focal']),contacts_semantics='No measured contact or calibrated floor'),limitations=['Research candidate, not KBO 3D ground truth','GVHMR uses modified inputs and approximate SMPL adapter; not full official SMPL-X postprocessing' if method=='gvhmr_f4' else 'NLF native shape retained, then custom confidence fitting','Source time is broadcast playback time; physical slow-motion timing unknown'],quality=dict(status='research_candidate_unvalidated',numeric_finite=bool(np.isfinite(an).all()),human_review_complete=False,stride_validated=False,global_trajectory_validated=False),frames=frames)
 return pose,an,feet

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--site',default=str(CODE));a=ap.parse_args();site=Path(a.site);public=site/'data';index=json.loads((public/'clips.json').read_text());baseby={r['id']:r for r in index['clips']};records=json.loads((OUT/'viewer_catalog.json').read_text());registry=dict(schema_version=1,version='20261028-explorer-gvhmr-f4-v1',preferred_method='gvhmr_f4',coordinate_policy='Fixed camera basis, first-valid pelvis origin; no inferred floor',coverage=dict(total=813,latest=36,legacy_only=777),clips={});total=0
 for rec in records:
  cid=rec['id'];clip=baseby[cid];legacy=json.loads((public/clip['pose3d']).read_text());two=json.loads((public/(clip.get('poseRefined') or clip['pose'])).read_text());directory='fits_strong' if rec['split']=='tune' else 'fits_final';variants={};outputs=[];points=[]
  for method,label,path in [('gvhmr_f4','GVHMR f4',OUT/directory/'gvhmr_temporal_f4'/f'{cid}.npz'),('nlf_native','NLF native shape + fitting',OUT/('fits_nlf_native_shape' if rec['split']=='tune' else 'fits_nlf_native_shape_test')/'nlf_rgb_temporal_f4'/f'{cid}.npz')]:
   if not path.exists():continue
   pose,native,feet=build_pose(cid,label,method,path,legacy,two);points.extend([native.reshape(-1,3),feet.reshape(-1,3)]);relative=f'reconstructions/stride-20261028/{cid}/{method}.json.gz';outputs.append((relative,pose));variants[method]=dict(id=method,label=label,pose3d=relative,model3d=pose['model'],method=method,sceneFrame=pose['scene']['frame'],sourceFitSha256=pose['source_fit_sha256'],qualityUrl='../studies/stride-20261028/report.html',role='preferred' if method=='gvhmr_f4' else 'comparison')
  cloud=np.concatenate(points);cloud=cloud[np.isfinite(cloud).all(1)];bounds=dict(min=cloud.min(0).tolist(),max=cloud.max(0).tolist())
  for relative,pose in outputs:
   pose['scene']['comparison_bounds']=bounds;raw=json.dumps(clean(pose),ensure_ascii=False,separators=(',',':')).encode();compressed=gzip.compress(raw,compresslevel=9,mtime=0);dest=public/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(compressed);total+=len(compressed)
   variants[pose['method']]['sha256_gzip']=sha(dest)
  registry['clips'][cid]=dict(preferred='gvhmr_f4',variants=variants,legacy_pose3d=clip['pose3d'])
 registry['coverage']=dict(total=len(index['clips']),latest=len(registry['clips']),nlf=sum('nlf_native' in r['variants'] for r in registry['clips'].values()),legacy_only=len(index['clips'])-len(registry['clips']))
 save(public/'reconstruction_registry.json',registry);save(OUT/'explorer_package.json',dict(latest_clips=len(registry['clips']),nlf_clips=sum('nlf_native' in r['variants'] for r in registry['clips'].values()),gzip_bytes=total,canonical_index_sha256=sha(public/'clips.json'),source_2d_assets_changed=False,coordinate_transform_only='constant origin and proper rigid rotation; no stride scaling or pose change'))
 print('EXPLORER ASSETS',len(registry['clips']),total,'bytes')
if __name__=='__main__':main()
