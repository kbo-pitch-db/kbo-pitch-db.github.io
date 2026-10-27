"""Direct-link arm experiment viewer; never change canonical main assets."""
from common import *
from all813_release import ROOT
from arm_cocking_refine import FOLDER,deployment_guard
from stride_review import native_leg
import gzip,argparse

LABELS={'current_GVHMR':'현재 GVHMR',
 'raw_graft_native_single':'Native MotionBERT 팔 방향 · SMPL 전신 IK 없이','raw_graft_native_flip':'Native MotionBERT 팔 방향 · flip · 전신 IK 없이',
 'mb_native_phase_002':'Native MB 팔 방향 · 구간 prior 약','mb_native_flip_phase_002':'Native MB 팔 방향 · flip prior 약',
 'gv_forearm_source_refit':'대조군 · GVHMR 전완만 2D 재적합 (MB 없음)',
 'gv_source_refit':'대조군 · GVHMR 팔만 2D 재적합 (MB 없음)','gv_axial_source_refit':'대조군 · 상완 축만 2D 재적합 (MB 없음)',
 'raw_axial_single':'상완 축 회전 · 초기 후보','raw_axial_flip':'상완 축 회전 · flip 초기 후보',
 'mb_axial_source_refit':'상완 축 회전 · 원본 2D 재적합','mb_axial_phase_002':'상완 축 회전 · 구간 prior 약','mb_axial_phase_010':'상완 축 회전 · 구간 prior 강','mb_flip_axial_phase_002':'상완 축 회전 · flip prior 약',
 'raw_graft_single':'MotionBERT 팔 전체 · 초기 후보','raw_graft_flip':'MotionBERT 팔 전체 · flip 복원 초기',
 'raw_forearm_single':'MotionBERT 전완만 · 초기 후보','raw_forearm_flip':'MotionBERT 전완만 · flip 복원 초기',
 'mb_single_source_refit':'팔 전체 · 원본 2D 재적합',
 'mb_single_phase_002':'팔 전체 · 구간별 깊이 prior 약',
 'mb_single_phase_010':'팔 전체 · 구간별 깊이 prior 강',
 'mb_flip_phase_002':'팔 전체 · flip 복원 prior 약',
 'mb_flip_phase_010':'팔 전체 · flip 복원 prior 강',
 'mb_forearm_source_refit':'전완만 · 원본 2D 재적합',
 'mb_forearm_phase_002':'전완만 · 구간별 깊이 prior 약',
 'mb_forearm_phase_010':'전완만 · 구간별 깊이 prior 강',
 'mb_flip_forearm_phase_002':'전완만 · flip 복원 prior 약'}

def main():
 global FOLDER,LABELS
 deployment_guard();ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['v1','v2','v3'],default='v3');args=ap.parse_args();version=args.version
 if version!='v1':
  module=__import__('arm_cocking_'+version);FOLDER=module.FOLDER;LABELS=module.LABELS
 site_name={'v1':'arm-cocking-v1-20261028','v2':'arm-cocking-v2-20261028','v3':'arm-cocking-20261028'}[version]
 protocol=json.loads((FOLDER/'protocol.json').read_text());selection=json.loads((FOLDER/'selection_rule.json').read_text());site=CODE/'studies'/site_name;site.mkdir(exist_ok=True);clips={c['id']:c for c in json.loads((CODE/'data/clips.json').read_text())['clips']};catalog=[]
 for split in ['tune','confirmation']:
  for cid in protocol[split]:
   report=json.loads((FOLDER/'reports'/f'{cid}.json').read_text());evaluation=json.loads((FOLDER/'evaluation'/f'{cid}.json').read_text());by={r['variant']:r for r in evaluation};clip=clips[cid]
   base=load(FOLDER/'fits/current_GVHMR'/f'{cid}.npz');current=json.loads(gzip.decompress((CODE/'data'/clip['pose3d']).read_bytes()));rotation=np.array(current['scene']['rotation_camera_to_scene']);origin=np.array(current['scene']['origin_camera']);leg=native_leg(base['anatomical_keypoints']);models={};cloud=[];crop=init_inputs(cid)[1];crop_origin=np.array([float(crop['crop_x1']),float(crop['crop_y1'])])
   for variant in report['variants']:
    path=FOLDER/'fits'/variant/f'{cid}.npz';z=load(path);assert np.array_equal(z['frame_index'],base['frame_index']) and np.array_equal(z['source_time'],base['source_time']);an=(z['anatomical_keypoints'].astype(np.float64)+z['translation'][:,None].astype(np.float64)-origin)@rotation.T/leg;feet=(z['foot_keypoints'].astype(np.float64)+z['translation'][:,None].astype(np.float64)-origin)@rotation.T/leg;cloud.extend([an.reshape(-1,3),feet.reshape(-1,3)])
    frames=[dict(anatomy=np.round(a,6).tolist(),feet=np.round(f,6).tolist()) for a,f in zip(an,feet)]
    models[variant]=dict(label=LABELS[variant],frames=frames,projected_crop=np.round(z['projected'][:,:17]-crop_origin,4).tolist(),source_fit_sha256=sha(path),metrics=by[variant])
   points=np.concatenate(cloud);bounds=dict(min=points.min(0).tolist(),max=points.max(0).tolist())
   for m in models.values():m['scene']=dict(frame='released-scene-fixed',show_floor=True,floor=dict(height=0.,estimated=True),comparison_bounds=bounds,per_frame_recentering=False)
   two=json.loads((CODE/'data'/clip['poseRefined']).read_text());payload=dict(id=cid,player=clip['player'],split=split,models=models,default_variant=selection['selected_variant'],times=[f['time'] for f in two['frames']],source_times=[f['source_time'] for f in two['frames']],source_frames=[f['source_frame'] for f in two['frames']],pose2d='../../data/'+clip['poseRefined'],source_url=clip['sourceUrl'],fps=two['fps'],side=report['throwing_side'],window=report['window'],main_release_changed=False,experiment_version=version,method_summary={'v1':'팔 회전을 이식하고 전체 SMPL 표면 모델로 재적합한 첫 실험입니다. 원본 정합·비국소 발 마커 변화가 문제로 남았습니다.','v2':'팔 관측 마커를 국소화하고 보정량의 시간 연결을 비교했습니다. 초기 자세의 원본 불일치가 충분히 해결되지 않았습니다.','v3':'원본 2D와 고정된 팔 길이에 맞는 네 가지 깊이 후보를 만들고, MotionBERT는 상대 깊이 선택에만 사용합니다. 하체·골반·카메라·발은 메인과 같이 유지합니다. 실제 깊이 정답을 확정한 결과는 아닙니다.'}[version],marker_scope=report.get('marker_scope','Full SMPL surface/pose-corrective model'),ray_hypotheses=report.get('ray_hypothesis_diagnostics'),wrist_rotation_status=report['wrist_rotation_status'],human_review_complete=False,absolute_stride_validated=False,display_transform=dict(origin_camera=origin.tolist(),rotation_camera_to_scene=rotation.tolist(),leg_divisor=float(leg),per_frame_recentering=False),coordinate_policy='Same released constant origin/rotation/leg divisor for all variants; no per-frame centering or scaling',source_preservation_gate=by[selection['selected_variant']]['source_and_preservation_gate'])
   dest=site/f'{cid}.json.gz';dest.write_bytes(gzip.compress(json.dumps(clean(payload),ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0));catalog.append(dict(id=cid,player=clip['player'],split=split,file=dest.name,sha256_gzip=sha(dest)))
 adapter_provenance={cid:{tag:json.loads((OUT/'adapters'/tag/f'{cid}.json').read_text()) for tag in ['mb_block_single_smpl','mb_block_flip_smpl']} for cid in protocol['tune']+protocol['confirmation']}
 save(site/'adapter_provenance.json',adapter_provenance)
 save(site/'native_direction_adapter_provenance.json',{cid:json.loads((FOLDER/'reports'/f'{cid}.json').read_text()).get('native_direction_adapter',{}) for cid in protocol['tune']+protocol['confirmation']})
 save(site/'catalog.json',catalog);save(site/'protocol.json',protocol);save(site/'selection.json',selection);save(site/'confirmation_summary.json',json.loads((FOLDER/'confirmation_summary.json').read_text()));shorthand=[]
 for cohort in ['tune','confirmation']:
  rr=json.loads((FOLDER/f'evaluation_{cohort}.json').read_text());shorthand.extend({k:v for k,v in r.items() if k!='model_forearm_posterior_curve_leg'} for r in rr)
 csvwrite(site/'results.csv',shorthand);save(site/'provenance.json',dict(experiment_version=version,full_smpl_surface_pose_correctives=version=='v1',native_joint_fk_preserved=True,foot_markers_frozen_by_design=version!='v1',main_release_commit=json.loads((ROOT/'deployment.json').read_text())['commit'],packager_sha256=sha(__file__),main_release_changed=False,main_index_sha256=sha(CODE/'data/clips.json'),clips=len(catalog),original_broadcast_media_included=False,body_root_contact_from_release=True,wrist_flexion_angle_not_observed=True,motionbert=dict(base_checkpoint_sha256=sha(WORKSPACE/'models/motionbert_h36m.bin'),public_last_block_checkpoint_sha256=sha(OUT/'temporal_adaptation/best.pt'),public_last_block_training_provenance=json.loads((OUT/'temporal_adaptation/provenance.json').read_text()),training_used_kbo_labels=False,new_training_this_experiment=False,independent_lifting_cache_reused=True,adapter='Two distinct paths: whole-body strong 3D IK to SMPL (WHAM RAW solver seed and fixed shape; conversion errors per clip), or direct native H36M17 upper-arm/forearm directions on the released GVHMR body, without that whole-body IK',hypotheses='Single and flip-restored predictions kept distinct; no coordinate averaging'),solver_files_sha256={name:sha(Path(__file__).with_name(name)) for name in ['arm_cocking_refine.py','arm_cocking_axial.py','arm_cocking_native.py','arm_cocking_evaluate.py','arm_cocking_observations.py']+(['arm_cocking_'+version+'.py','evaluate_arm_cocking_'+version+'.py'] if version!='v1' else [])+(['arm_depth_rays.py'] if version=='v3' else [])}))
 print('ARM COCKING PREVIEW',len(catalog),'clips; main unchanged')

if __name__=='__main__':main()
