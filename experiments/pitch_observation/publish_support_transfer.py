"""Publish the numerical pilot separately; never switch Explorer reconstructions."""
from common import *
from publish_stride import rounded
import gzip,re,shutil

def main():
 source=OUT/'support_transfer_root_v1';dest=CODE/'studies/support-transfer-20261028';dest.mkdir(exist_ok=True)
 catalog=json.loads((source/'comparison/viewer/catalog.json').read_text());protocol=json.loads((source/'protocol.json').read_text())
 for rec in catalog:
  rec['split']='pilot6' if rec['id'] in protocol['pilot'] else 'confirmation6'
  data=json.loads((source/'comparison/viewer'/f'{rec["id"]}.json').read_text())
  raw=json.dumps(rounded(data),ensure_ascii=False,separators=(',',':')).encode()
  path=dest/'viewer'/f'{rec["id"]}.json.gz';path.parent.mkdir(exist_ok=True);path.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
 save(dest/'viewer/catalog.json',catalog)
 for name in ['pilot_metrics.csv','confirmation_metrics.csv','summary.json','protocol.json','execution.json','integrity.json']:
  shutil.copy2(source/name,dest/name)
 summary=json.loads((dest/'summary.json').read_text());summary.update(main_site_changed=True,explorer_default_changed=False,experimental_comparison_published=True,publication_note='Only this separate numerical comparison page was published; Explorer reconstruction selection is unchanged.');save(dest/'summary.json',summary)
 html=(source/'comparison/index.html').read_text().replace('<title>KBO 보폭 3D 비교 · GVHMR f4 / NLF</title>','<title>지지 전환 3D 비교 · GVHMR f4 전후</title>')
 html=html.replace('연구용 root-only 비교, 배포 설정 미변경.','별도 실험 비교 · Explorer 기본 결과는 유지.')
 html=html.replace('href="../pilot_metrics.csv"','href="pilot_metrics.csv"').replace('href="../confirmation_metrics.csv"','href="confirmation_metrics.csv"').replace('href="../protocol.json"','href="protocol.json"')
 html=html.replace('고정 설정</a></p>','고정 설정</a> · <a href="report.html">결과 요약</a></p>')
 html=html.replace('현재 클립 수치 JSON','현재 클립 수치 JSON.gz')
 html=html.replace('<button id="prev">','<button id="plant">추정 착지 구간으로</button><button id="prev">')
 html=html.replace('<h2>최종 Halpe-26', '<details id="source-video"><summary>원본 영상 펼치기 · YouTube</summary><p class="small">원본은 기록된 방송 시각에서 재생됩니다. 아래 3D 재생과는 별도로 조작합니다. 재생이 제한되면 위 원본 링크를 여세요.</p><div id="source-player"></div></details><h2>최종 Halpe-26')
 html=html.replace('<script src="comparison.js"></script>', '<script src="../../reconstruction_registry.js?v=20261028-explorer"></script><script src="comparison.js?v=support-transfer-v1"></script>')
 if 'name="robots"' not in html:html=html.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta name="robots" content="noindex,nofollow,noarchive">',1)
 (dest/'index.html').write_text(html)
 js=(source/'comparison/comparison.js').read_text()
 js=js.replace('return r.json();','return window.ReconstructionRegistry.decode(r,url);')
 js=js.replace("'use strict';", "'use strict';\nconst modelLabel=n=>({'gvhmr_f4':'기존 GVHMR f4','root_temporal_only':'전역 이동 시간 제약','root_support_transfer':'시간 제약 + 축발→앞발 지지 전환'}[n]||n);")
 js=js.replace('new Option(n,n)','new Option(modelLabel(n),n)')
 js=js.replace("${$('baseline').value}","${modelLabel($('baseline').value)}").replace("${$('candidate').value}","${modelLabel($('candidate').value)}")
 js=js.replace('viewer/${encodeURIComponent($(\'clip\').value)}.json','viewer/${encodeURIComponent($(\'clip\').value)}.json.gz').replace('viewer/${encodeURIComponent(data.id)}.json','viewer/${encodeURIComponent(data.id)}.json.gz')
 js=js.replace("$('source').href=source.href;", "$('source').href=source.href;renderSourcePlayer();")
 js=js.replace("data=next;frame=0;", "data=next;const params=new URLSearchParams(location.search),requestedFrame=Number(params.get('frame'));frame=params.get('clip')===data.id&&Number.isInteger(requestedFrame)?Math.min(data.times.length-1,Math.max(0,requestedFrame)):0;$('plant').disabled=!data.phase.plant_interval;")
 js+='''
function renderSourcePlayer(){
 const host=$('source-player');host.replaceChildren();if(!$('source-video').open||!data)return;
 const link=new URL(data.sourceUrl),id=link.searchParams.get('v');if(!/^[A-Za-z0-9_-]{11}$/.test(id||''))return;
 const embed=document.createElement('iframe');embed.src=`https://www.youtube-nocookie.com/embed/${id}?start=${Math.floor(data.source_times[frame])}&end=${Math.ceil(data.source_times.at(-1))}&playsinline=1&rel=0`;
 embed.title=`원본 영상 · ${data.player}`;embed.allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';embed.allowFullscreen=true;embed.referrerPolicy='strict-origin-when-cross-origin';embed.style.cssText='width:100%;max-width:960px;aspect-ratio:16/9;border:0';host.append(embed);
}
$('source-video').addEventListener('toggle',renderSourcePlayer);
$('plant').onclick=()=>{const plant=data?.phase.plant_interval;if(plant)step(Math.round((plant[0]+plant[1])/2));};
'''
 (dest/'comparison.js').write_text(js)
 report=(source/'REPORT_KO.html').read_text().replace('href="comparison/"','href="./"')
 report=report.replace('기존 main·공개 데이터·최종 Halpe-26은 변경하지 않았습니다.','Explorer 기본 결과와 최종 Halpe-26은 유지하며 이 비교 페이지만 별도로 공개합니다.')
 if 'name="robots"' not in report:report=report.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta name="robots" content="noindex,nofollow,noarchive">',1)
 (dest/'report.html').write_text(report)
 (dest/'REPRODUCE.txt').write_text('From the preserved research worktree:\n../env/bin/python experiments/pitch_observation/support_transfer_root.py --cohort pilot\n../env/bin/python experiments/pitch_observation/support_transfer_root.py --cohort confirmation\n../env/bin/python experiments/pitch_observation/view_support_transfer.py\n../env/bin/python experiments/pitch_observation/publish_support_transfer.py\nCached fits resume; original Halpe-26 and canonical Explorer registry are unchanged.\nOnly numerical outputs and external YouTube references are published. No source broadcast pixels, videos, checkpoints or licensed meshes are included.\n')
 files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='publication.json'}
 save(dest/'publication.json',dict(version='support-transfer-v1',clips=len(catalog),models=['gvhmr_f4','root_temporal_only','root_support_transfer'],media_policy='numerical_pose_only',original_broadcast_media_included=False,rendering_precision_decimals=4,source_youtube_embed='Only after user opens the original-video panel; separate playback',explorer_default_changed=False,full813_processed=False,human_review_complete=False,files=files))
 total=sum((dest/p).stat().st_size for p in files);assert total<5_000_000,total
 print('PUBLIC SUPPORT COMPARISON',len(catalog),'clips',total,'bytes')
if __name__=='__main__':main()
