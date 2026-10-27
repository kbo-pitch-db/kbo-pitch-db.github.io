"""Source/geometry audit and fixed-axis local diagnostics; no broadcast export.

Run after select_rear_pivot_bounded.py. Licensed SMPL files stay local. PNGs
with source pixels are ONLY written outside the Pages checkout.
"""
from common import *
from support_transfer_root import metrics
from rear_pivot_consistent import rear_diagnostics
from stride_review import native_leg
from scipy.spatial.transform import Rotation
import argparse,html

NAMES={'rear_pivot_v5':'Root-only grip: rejected, wrist source regression',
 'rear_pivot_rigid_v6':'Root plus rigid orientation: rejected, wrist regression',
 'rear_pivot_joint_v7':'Joint fitting: rejected, endpoint slip and stride change',
 'rear_pivot_consistent_v8':'Persistent contact: insufficient stationarity',
 'rear_pivot_v9':'Stronger tangent closure: rejected, vertical escape/source feet',
 'rear_pivot_v10':'Normal closure and feet observations: rejected, front toe overconstraint',
 'rear_pivot_v11':'Rear-only closure: rejected, knee hyperextension',
 'rear_pivot_v13':'Bounded knees, rear-only closure, native stride: selected candidates only'}

def source_detail(z,h,cw):
 m,uv=metrics(z,h,z['background_transforms'],z['translation'],cw)
 target=np.concatenate([h['keypoints'][:,:17],h['keypoints'][:,20:26]],1)
 sc=np.concatenate([scores(h)[:,:17],scores(h)[:,20:26]],1)
 err=np.linalg.norm(uv-target,axis=-1)
 for name,ids in [('source',list(range(23))),('elbow',[7,8]),('wrist',[9,10]),('feet',list(range(17,23)))]:
  e=err[:,ids][(sc[:,ids]>.5)&np.isfinite(err[:,ids])]
  for p in [50,95,99,100]:m[f'{name}_p{p}_px']=float(np.percentile(e,p)) if len(e) else None
 foot_error=err[:,17:].copy();foot_error[sc[:,17:]<=.5]=np.nan
 i,j=np.unravel_index(np.nanargmax(foot_error),foot_error.shape)
 m['feet_worst_frame_index']=int(i);m['feet_worst_halpe_joint']=int(j+20)
 m['feet_worst_source_frame']=int(h['frame_index'][i])
 return m,uv

def audit(geometry=False,plots=False):
 folder=OUT/'rear_pivot_selected_v14';reports=[];ablation=[]
 if geometry:
  import torch
  from pitch_study_fit import StudyFitter
  torch.set_num_threads(1);body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
 for rec in sample('tune',True):
  cid=rec['id'];meta_path=folder/f'{cid}.json'
  if not meta_path.exists():continue
  selected=json.loads(meta_path.read_text())
  if selected['status']!='eligible_research_preview':continue
  z=load(meta_path.with_suffix('.npz'));old=load(OUT/'support_transfer_arm_v4'/f'{cid}.npz');base=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz');oldmeta=json.loads((OUT/'support_transfer_arm_v4'/f'{cid}.json').read_text());_,_,_,h,_=init_inputs(cid);leg=native_leg(base['anatomical_keypoints']);cw=z['contact_weight'];r=dict(id=cid,player=rec['player_name'],split='previously used tuning12',selected=selected['selected'],selected_metrics=selected['selected_metrics'],human_review_complete=False,absolute_stride_validated=False)
  for name,state in [('f4',base),('v4',old),('v14',z)]:
   detail,uv=source_detail(state,h,cw);detail.update(rear_diagnostics(state,old,oldmeta));sep=np.linalg.norm(state['anatomical_keypoints'][:,7]-state['anatomical_keypoints'][:,8],axis=-1)/leg;detail['peak_interankle_leg']=float(sep.max());detail['root_end_minus_start_model']=((state['anatomical_keypoints'][-1,0]+state['translation'][-1])-(state['anatomical_keypoints'][0,0]+state['translation'][0])).tolist()
   # Camera-relative projected throwing-wrist motion. Not ball release or
   # true 3D arm speed. Confidence gaps split derivatives, never interpolate.
   side=0 if selected['phase']['throwing_hand']=='left' else 1;wrist=9+side;shoulder=5+side;target=h['keypoints'];dt=np.diff(h['source_time']);valid=(scores(h)[:,[wrist,shoulder]]>.5).all(1);links=valid[1:]&valid[:-1]&(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1)
   observed=np.linalg.norm(np.diff(target[:,wrist]-target[:,shoulder],axis=0)/dt[:,None],axis=-1);predicted=np.linalg.norm(np.diff(uv[:,wrist]-uv[:,shoulder],axis=0)/dt[:,None],axis=-1);ix=np.flatnonzero(links)
   if len(ix):detail.update(arm_projected_speed_p95_ratio=float(np.percentile(predicted[ix],95)/np.percentile(observed[ix],95)),arm_projected_peak_delay_broadcast_s=float(h['source_time'][ix[np.argmax(predicted[ix])]+1]-h['source_time'][ix[np.argmax(observed[ix])]+1]))
   r[name]=detail
  assert np.array_equal(z['frame_index'],base['frame_index']) and np.array_equal(z['source_time'],base['source_time'])
  assert np.array_equal(z['smpl_betas'],base['smpl_betas'])
  assert np.array_equal(z['smpl_rotations'][:,16:24],base['smpl_rotations'][:,16:24])
  parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]);bl=lambda a:np.linalg.norm(a[:,1:]-a[:,parents[1:]],axis=-1)
  r['native_bone_length_max_change_model']=float(abs(bl(z['anatomical_keypoints'])-bl(base['anatomical_keypoints'])).max());assert r['native_bone_length_max_change_model']<2e-5
  r['f4_local_joint_change_degrees_max']=np.rad2deg(Rotation.from_matrix((np.swapaxes(base['smpl_rotations'],-1,-2)@z['smpl_rotations']).reshape(-1,3,3)).magnitude().reshape(-1,24).max(0)).tolist()
  r['v13_seconds']=json.loads((OUT/'rear_pivot_v13'/f'{cid}.json').read_text())['seconds']
  r['source_fit_sha256']=sha(meta_path.with_suffix('.npz'));r['source_2d_sha256']=sha(DATA/rows('motion_wham.csv')[cid]['pose_2d_refined'])
  r['arm_proxy_semantics']='Global raw projected-speed peak, confidence gated; not release. Existing f4 timing mismatch may persist; unchanged peak delay does not validate the motion.'
  if geometry:
   beta=z['smpl_betas'];beta=np.median(beta,0) if beta.ndim==2 else beta;body.prepare_shape(beta)
   with torch.no_grad():p,a,f=body.model(torch.tensor(z['smpl_rotations'],dtype=torch.float32),torch.tensor(np.repeat(beta[None],len(z['keypoints']),0),dtype=torch.float32))
   r['fk_max_error_model']={k:float(abs(actual.numpy()-z[k]).max()) for k,actual in [('keypoints',p),('anatomical_keypoints',a),('foot_keypoints',f)]}
   assert max(r['fk_max_error_model'].values())<2e-5
  reports.append(r)
  for dirname,decision in NAMES.items():
   mp=OUT/dirname/f'{cid}.json'
   if not mp.exists():continue
   q=json.loads(mp.read_text())
   for candidate in q.get('hypotheses',[]):ablation.append(dict(id=cid,player=rec['player_name'],experiment=dirname,decision=decision,**candidate))
  if plots:draw_local(cid,z,old,h,oldmeta,leg,folder)
 summary=dict(clips=len(reports),test24_used=False,public_813_release_changed=False,human_review_complete=False,accuracy_claim='No GT accuracy or absolute stride claim; support residual is directly optimized',clip_median_rear_extent_before=float(np.median([r['v4']['rear_point_extent_3d_leg'] for r in reports])),clip_median_rear_extent_after=float(np.median([r['v14']['rear_point_extent_3d_leg'] for r in reports])),source_foot_p100_worse_clips=[r['id'] for r in reports if r['v14']['feet_p100_px']>r['v4']['feet_p100_px']],results=reports)
 save(folder/'audit.json',summary);csvwrite(folder/'ablation.csv',ablation);save(folder/'ablation.json',ablation)
 flat=[]
 for r in reports:
  for name in ['f4','v4','v14']:flat.append(dict(id=r['id'],player=r['player'],method=name,**r[name],stride_change_p95_leg=r['selected_metrics']['stride_change_p95_leg'] if name=='v14' else None))
 csvwrite(folder/'comparison.csv',flat)
 # Only numerical records and method explanation enter the unlisted Pages
 # report. No source images, videos, weights, licensed meshes or local paths.
 site=CODE/'studies/rear-pivot-20261028';site.mkdir(parents=True,exist_ok=True)
 save(site/'results.json',summary);csvwrite(site/'comparison.csv',flat);csvwrite(site/'ablation.csv',ablation)
 table=''.join(f'<tr><td>{html.escape(r["player"])}</td><td>{r["v4"]["rear_point_extent_3d_leg"]*100:.2f} → {r["v14"]["rear_point_extent_3d_leg"]*100:.2f}%</td><td>{r["selected_metrics"]["stride_change_p95_leg"]*100:.2f}%</td><td>{r["v4"]["feet_p95_px"]:.2f} → {r["v14"]["feet_p95_px"]:.2f}</td><td>{r["v4"]["feet_p100_px"]:.2f} → {r["v14"]["feet_p100_px"]:.2f}</td><td><a href="/?preview=rear-pivot&amp;clip={r["id"]}#explore">새 결과</a> · <a href="/?preview=support-v4&amp;clip={r["id"]}#explore">이전</a></td></tr>' for r in reports)
 text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="robots" content="noindex,nofollow,noarchive"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rear support fitting notes</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:3rem auto;padding:1rem;color:#192635}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.6rem;border:1px solid #bbc9d4;text-align:left}code,pre{background:#edf2f5}a{color:#17618a}</style><h1>뒷발 지지 중 역방향 미끄러짐 수정</h1><p>공개 데이터셋 기본 결과는 그대로다. 직접 링크에서만 동일 Explorer로 후보를 확인한다. 이 페이지는 검색·내비게이션에서 제외했으며 접근 통제된 비공개 페이지는 아니다.</p><p>원인: v4는 전 시퀀스 평균 robust contact가 초기 뒷발 지지의 큰 잔차를 약하게 다뤘고, 고정된 상대 자세와 root 보정만으로는 원본의 팔·발 관측을 동시에 보존하기 어려웠다. 접촉을 강화한 중간 해에서는 수직 이탈과 무릎 과신전이 발견돼 기각했다.</p><p>최종 후보: 기존 GVHMR f4의 큰 보폭 가설, 고정 체형과 로컬 팔 회전을 유지한다. root·하체·몸통·공통 focal을 공동 fitting한다. 관측 기반 초기 뒷발 발끝 한 점에 유한한 confidence 가중 접선·법선 closure를 주고 뒤꿈치 들림, 앞발 피벗, 지지 해제를 허용한다. 무릎은 넓은 모델 회전 범위(-0.08…2.8 rad, 비굴곡 성분 ±0.18 rad) 안에서 FK로 생성한다. 사람의 실제 해부학적 관절각 측정은 아니다. 평균 보폭·신장·투구판 거리에 맞춘 확대나 프레임별 바닥/원점 보정은 없다.</p><p>8개 후보(두 closure 강도 × 네 지면/카메라 기울기)를 보존한다. 기존 f4 대비 원본 중앙값·손목 p95·발 p95의 +2px 화면을 유지하고, 뒷발 이동·관통·무릎·보폭 일관성을 모두 만족한 후보 중 원본 중앙 오차가 가장 작은 것을 선택한다. 평면은 한 클립에 하나이며 실제 마운드를 측량한 값이 아니다.</p><table><tr><th>선수</th><th>뒷발 발끝 최대 이동 / 모델 다리 길이</th><th>f4 대비 발목 간 거리 변화 p95</th><th>원본 발 오차 p95 (px)</th><th>원본 발 최대 오차 (px)</th><th>동일 Explorer</th></tr>TABLE</table><p>위 뒷발 이동은 이전 v4에서 정한 동일 지지 구간에서 시작점 대비 3D 최대 이동이다. 스트라이드 항목은 프레임별 발목 간 거리 변화이며 실제 투구판–착지 거리나 mm 정확도가 아니다. 접촉 잔차는 목적함수에 들어가므로 독립적인 정확도 증거가 아니다. 발 p95가 줄어도 최악 프레임이 나빠지는 사례가 있다. 원본 RTMPose 관측의 가림·겹침 자체도 불확실하다.</p><p>6개 모두 이미 사용한 튜닝 표본이다. 두 문제 사례 후 같은 설정으로 추가 4개를 확인했지만 새 held-out 정확도 검증은 아니다. 기존 최종 test24는 사용하지 않았다. 813개 전수 결과 개선이나 수동 검수 완료를 주장하지 않는다. 시퀀스 끝 root는 저장된 이동을 유지하며 원점으로 되돌리는 코드가 없다. 원본 대조 이미지는 로컬 연구 결과 폴더에만 저장했다.</p><p><a href="results.json">수치·FK 검증 JSON</a> · <a href="comparison.csv">전후 CSV</a> · <a href="ablation.csv">기각 실험 포함 후보 CSV</a></p><p>재현: 저장소 experiments/pitch_observation/rear_pivot_bounded.py → select_rear_pivot_bounded.py → audit_rear_pivot.py --geometry --plots → package_rear_pivot_preview.py. 준비된 로컬 데이터와 모델 파일이 필요하다. 재실행은 완료된 클립을 입력 SHA와 설정 검사 후 재사용한다.</p></html>'''.replace('TABLE',table)
 (site/'index.html').write_text(text)
 print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))

def draw_local(cid,z,old,h,meta,leg,folder):
 import cv2,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rear=1 if meta['phase']['lead_side']=='left' else 0;ep=next(q for q in meta['marker_states'] if q['marker']==rear)
 frames=sorted(set([ep['start'],(ep['start']+ep['end'])//2,meta['phase']['plant_interval'][0],ep['end']]))
 normal=z['normal'].astype(float);normal/=np.linalg.norm(normal);x=np.array([1.,0.,0.]);x-=x@normal*normal;x/=np.linalg.norm(x);rot=np.stack([x,np.cross(x,normal),normal]);origin=z['foot_keypoints'][0,rear]+z['translation'][0];origin-= (origin@normal+float(z['pilot_support_offset']))*normal
 transform=lambda p,tr:(p+tr[:,None]-origin)@rot.T/leg
 data=[(transform(q['anatomical_keypoints'],q['translation']),transform(q['foot_keypoints'],q['translation'])) for q in [old,z]]
 cloud=np.concatenate([a.reshape(-1,3) for pair in data for a in pair]);lo=cloud.min(0)-.1;hi=cloud.max(0)+.1
 fig=plt.figure(figsize=(19,4*len(frames)));cap=cv2.VideoCapture(str(OUT/'original_clips'/f'{cid}.mp4'));parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
 for row,i in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();ax=fig.add_subplot(len(frames),5,row*5+1)
  if ok:
   ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));obs=h['keypoints'][i];sc=scores(h)[i];g=sc>.5;ax.scatter(obs[g,0],obs[g,1],s=7,c='yellow');uv=z['projected'][i];ax.scatter(uv[:,0],uv[:,1],s=9,c='cyan',marker='+')
  ax.set_title(f'Local source + 2D / fitted projection | frame {i}');ax.axis('off')
  for col,(version,view) in enumerate([(0,(8,-5)),(1,(8,-5)),(0,(90,-90)),(1,(90,-90))],1):
   ax=fig.add_subplot(len(frames),5,row*5+col+1,projection='3d');a,f=data[version]
   for j,p in enumerate(parents[1:],1):ax.plot(*a[i,[p,j]].T,c='#28618d',lw=1.6)
   ax.scatter(*f[i].T,c=['orange' if j==rear else '#279574' for j in range(6)],s=16)
   ax.scatter(*f[ep['start']:ep['end']+1,rear].T,s=2,c='#db6834',alpha=.5)
   xx,yy=np.meshgrid([lo[0],hi[0]],[lo[1],hi[1]]);ax.plot_surface(xx,yy,np.zeros_like(xx),color='grey',alpha=.12)
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(*view);ax.set_title(('v4' if version==0 else 'v14')+(' side' if col<3 else ' top'));ax.set_xlabel('camera X / leg');ax.set_ylabel('camera depth / leg')
 cap.release();fig.tight_layout();fig.savefig(folder/f'{cid}_local_source_fixed_axes.png',dpi=125);plt.close(fig)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--geometry',action='store_true');ap.add_argument('--plots',action='store_true');a=ap.parse_args();audit(a.geometry,a.plots)
