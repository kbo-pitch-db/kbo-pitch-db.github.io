"""Local numerical-only, fixed-axis before/after viewer for the root pilot."""
from common import *
from candidates import CC,norm
from stride_review import metric
import re,copy

def main():
 folder=OUT/'support_transfer_root_v1';dest=folder/'comparison';dest.mkdir(exist_ok=True);catalog=[]
 for rec in sample('tune',True):
  cid=rec['id'];baseline=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz')
  paths=[folder/m/f'{cid}.npz' for m in ['root_temporal_only','root_support_transfer']]
  if not all(p.exists() for p in paths):continue
  payload=json.loads((CODE/'studies/stride-20261028/viewer'/f'{cid}.json').read_text());phase=payload['phase'];relative,leg=norm(baseline['keypoints'][:,CC]);anchor=(baseline['keypoints'][:,CC]+baseline['translation'][:,None])[0,:2].mean(0);models={}
  for method,z in [('gvhmr_f4',baseline)]+[(p.parent.name,load(p)) for p in paths]:
   native=z['anatomical_keypoints'];global_native=native+z['translation'][:,None]
   models[method]=dict(relative=relative,global_fixed=(z['keypoints'][:,CC]+z['translation'][:,None]-anchor)/leg,projected=z['projected'][:,CC],stride=metric(native,global_native,phase,None),native_lower_hip_relative_leg=(native[:,[1,2,4,5,7,8]]-native[:,[1,2]].mean(1,keepdims=True))/leg)
  payload.update(models=models,extent=max(3.2,max(float(np.nanmax(abs(m['global_fixed']))) for m in models.values()))*1.08,default_model='root_support_transfer',root_pilot='Exact unchanged articulation; modified root only; inferred contact/plane, not GT')
  save(dest/'viewer'/f'{cid}.json',payload);catalog.append(dict(id=cid,player=rec['player_name'],split=rec['cohort'] if 'cohort' in rec else 'tune'))
 save(dest/'viewer/catalog.json',catalog)
 template=(Path(__file__).parent/'stride_public.html').read_text()
 template=template.replace('보폭 3D 비교 · GVHMR f4 기본 / NLF 비교','지지 전환 실험 · GVHMR f4 자세 고정')
 template=re.sub(r'<div class="notice">.*?</div>','<div class="notice"><b>연구용 root-only 비교, 배포 설정 미변경.</b> 기존 GVHMR f4의 관절 자세·방향·동일 프레임 양발 간격을 고정한 채 전역 이동만 보정했습니다. 시간 제약만 적용한 후보와 축발→앞발 soft contact를 추가한 후보를 분리했습니다. 접촉과 단일 지면은 가설이며 실측이 아닙니다. 접촉 자체의 개선은 독립 정확도 증거가 아닙니다. 원본 2D 오차도 확인하세요. 기존 최종 비교 24개는 이 실험에 쓰지 않았고, 이번 12개는 모두 기존 튜닝 영역입니다.</div>',template,count=1,flags=re.S)
 template=re.sub(r'<p><a href="../../">.*?</p>','<p><a href="https://kbo-pitch-db.github.io/?new3d=1#explore">현재 공개 Explorer</a> · <a href="../pilot_metrics.csv">파일럿 CSV</a> · <a href="../confirmation_metrics.csv">나머지 튜닝 6개 CSV</a> · <a href="../protocol.json">고정 설정</a></p>',template,count=1,flags=re.S)
 template=re.sub(r'<details><summary>GVHMR f4와 NLF 결과의 생성 경로</summary>.*?</details>','<details><summary>이번 실험의 범위</summary><p>새 추론기가 아니라 기존 GVHMR f4의 root trajectory만 수정한 실험입니다. 관절·체형·원래 f4 카메라는 고정했습니다. 단일 지면의 세 법선 가설을 남겼고 목적함수로 고른 가설이 실제 지면임을 뜻하지 않습니다. 물리적 체중·힘의 전달을 관측한 결과가 아닙니다.</p></details>',template,count=1,flags=re.S)
 (dest/'index.html').write_text(template)
 js=(Path(__file__).parent/'stride_public.js').read_text().replace("options('baseline',names,'current_actual')","options('baseline',names,'gvhmr_f4')").replace("options('candidate',names,'gvhmr_f4')","options('candidate',names,'root_support_transfer')")
 (dest/'comparison.js').write_text(js)
 print('ROOT COMPARISON',len(catalog),'clips',dest)
if __name__=='__main__':main()
