"""Propose clip-to-relay pitch links from the visible broadcast score bug.

Only exact, unique agreements of pitcher, batter, inning and pitcher cumulative
pitch count become proposed links. They remain unverified until human review.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import re
import subprocess

import av
import cv2
import numpy as np

from .__main__ import frame_at, read_csv
from .dataset import CLIP_FIELDS, LINK_FIELDS, atomic_csv, write_dataset_summary


OCR_VERSION='2024_layouts_v3'
OBS_FIELDS=['dataset_clip_id','source_id','naver_game_id','layout','ocr_version','source_time','scoreboard_image',
            'pitcher_ocr','batter_ocr','inning_ocr','pitch_count_ocr',
            'balls_on_screen','strikes_on_screen','outs_on_screen','half_on_screen','count_pair_ocr',
            'pitcher_name','batter_name','inning','inning_alt_ocr','inning_resolution','pitches_completed_on_screen',
            'pitcher_match','batter_match','ocr_status','frame_attempts']
CAND_FIELDS=['dataset_clip_id','naver_pitch_id','rank','match_status','match_basis','match_score',
             'pitcher_name','batter_name','inning','half_inning','pitcher_pitch_number',
             'pitch_number','pitch_type_raw','speed_kmh','count_state_match','evidence','scoreboard_image']


def ocr_paths():
    bundled=Path(__file__).resolve().parent.parent/'tools/ocr/usr'
    executable=Path(os.environ['KBO_TESSERACT']) if os.environ.get('KBO_TESSERACT') else bundled/'bin/tesseract'
    if not executable.exists():
        from shutil import which
        found=which('tesseract')
        if not found:raise RuntimeError('Tesseract missing; set KBO_TESSERACT and TESSDATA_PREFIX')
        executable=Path(found)
    tessdata=Path(os.environ['TESSDATA_PREFIX']) if os.environ.get('TESSDATA_PREFIX') else bundled/'share/tesseract-ocr/4.00/tessdata'
    return executable,tessdata


def ocr(image, language, psm, digits=False):
    executable,tessdata=ocr_paths()
    ok,encoded=cv2.imencode('.png',image)
    if not ok:return ''
    command=[str(executable),'stdin','stdout','-l',language,'--psm',str(psm)]
    if digits:command+=['-c','tessedit_char_whitelist=0123456789']
    env=os.environ.copy()
    if tessdata.exists():env['TESSDATA_PREFIX']=str(tessdata)
    result=subprocess.run(command,input=encoded.tobytes(),capture_output=True,env=env,timeout=10)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[:400])
    return result.stdout.decode('utf-8',errors='replace').replace('\x0c','').strip()


def scoreboard_crop(frame, box):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=box
    return frame[round(y1*h/1080):round(y2*h/1080),round(x1*w/1920):round(x2*w/1920)]


LAYOUTS={
    'spotv':{'pitcher':(135,822,281,856),'batter':(135,858,260,892),
             'pitch_count':(310,826,364,857),'inning':(156,949,179,981),
             'thumb':(127,816,372,1037)},
    'sbs':{'pitcher':(315,112,420,140),'batter':(95,112,205,140),
           'pitch_count':(477,112,516,142),'inning':(568,112,588,142),
           'thumb':(80,49,684,174)},
    'sbs_left':{'pitcher':(95,112,205,140),'batter':(315,112,420,140),
           'pitch_count':(271,112,304,142),'inning':(568,112,588,142),
           'thumb':(80,49,684,174)},
    'kbs':{'pitcher':(1534,826,1637,859),'batter':(1534,783,1640,818),
           'pitch_count':(1777,826,1815,860),'inning':(1677,879,1705,912),
           'thumb':(1493,777,1825,971)},
    'mbc':{'pitcher':(1315,1004,1421,1047),'batter':(1657,1004,1775,1047),
           'pitch_count':(1157,1007,1192,1039),'inning':(327,1006,352,1040),
           'thumb':(310,995,1830,1054)},
}


def read_scoreboard(frame,layout,fields=('pitcher','batter','pitch_count','inning')):
    regions=LAYOUTS[layout]
    values={}
    for field in fields:
        box=regions[field]
        roi=scoreboard_crop(frame,box)
        if roi.size==0:values[field]='';continue
        if field in ('pitcher','batter'):
            scaled=cv2.resize(roi,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
            if (layout.startswith('sbs') and field=='pitcher') or (layout=='kbs' and field=='batter'):
                scaled=255-cv2.cvtColor(scaled,cv2.COLOR_BGR2GRAY)
            if layout=='mbc' and field=='batter':
                gray=cv2.cvtColor(scaled,cv2.COLOR_BGR2GRAY)
                scaled=255-cv2.threshold(gray,130,255,cv2.THRESH_BINARY)[1]
            values[field]=ocr(scaled,'kor+eng',7)
        else:
            gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            if layout=='mbc' or layout.startswith('sbs') and field=='pitch_count' or layout=='spotv' and np.median(gray)<115:
                gray=255-gray
            scaled=cv2.resize(gray,None,fx=5,fy=5,interpolation=cv2.INTER_CUBIC)
            psm=13 if layout.startswith('sbs') and field=='inning' else 7 if field=='pitch_count' or layout=='mbc' else 10
            values[field]=ocr(scaled,'eng',psm,True)
            if field=='pitch_count' and not values[field] and layout=='spotv':
                values[field]=ocr(scaled,'eng',13,True)
            if field=='inning' and not values[field] and layout=='kbs':
                values[field]=ocr(255-scaled,'eng',7,True)
    return values


def clean_name(value):
    value=re.sub(r'^[0-9]+','',value or '')
    value=re.sub(r'^\s*[Pp][.\s]+','',value)
    return ''.join(re.findall(r'[가-힣A-Za-z]+',value)).lower()


def name_match(value, known):
    name=clean_name(value)
    if not name:return '', 'missing'
    exact=[item for item in known if clean_name(item)==name]
    if len(exact)==1:return exact[0],'exact'
    # One-character OCR errors are retained only as review hints, never as link evidence.
    near=[item for item in known if abs(len(clean_name(item))-len(name))<=1
          and edit_distance_le_one(clean_name(item),name)]
    return (near[0],'fuzzy_review') if len(near)==1 else ('','unmatched')


def edit_distance_le_one(a,b):
    if a==b:return True
    if abs(len(a)-len(b))>1:return False
    if len(a)==len(b):return sum(x!=y for x,y in zip(a,b))<=1
    if len(a)>len(b):a,b=b,a
    return any(a==b[:i]+b[i+1:] for i in range(len(b)))


def parse_number(text, minimum, maximum):
    stripped=re.sub(r'\s','',text or '')
    if not stripped.isdecimal():return None
    value=int(stripped)
    return value if minimum<=value<=maximum else None


def spotv_count_lights(scoreboard_image):
    """Read the colored B/S/O dots in the fixed 2024 SPOTV score bug crop."""
    if scoreboard_image is None or scoreboard_image.shape[0]<190 or scoreboard_image.shape[1]<235:
        return {}
    hsv=cv2.cvtColor(scoreboard_image,cv2.COLOR_BGR2HSV)
    def lit(xs,low,high):
        total=0
        for source_x in xs:
            x=source_x-127
            patch=hsv[170:183,x-5:x+6]
            colored=(patch[:,:,1]>65)&(patch[:,:,0]>=low)&(patch[:,:,0]<=high)
            total+=float(colored.mean())>.55
        return int(total)
    arrow=(hsv[:,:,0]<=45)&(hsv[:,:,1]>90)&(hsv[:,:,2]>90)
    histogram=arrow[140:160,47:66].sum(axis=1)
    visible=np.flatnonzero(histogram);half=''
    if len(visible)>=8 and histogram.sum()>=50:
        upper=histogram[visible[:3]].sum();lower=histogram[visible[-3:]].sum()
        if lower>upper*1.7:half='top'
        elif upper>lower*1.7:half='bottom'
    return dict(balls_on_screen=lit((165,184,204),40,95),
                strikes_on_screen=lit((260,278),0,30),
                outs_on_screen=lit((331,348),145,179),half_on_screen=half)


def mbc_scoreboard_state(scoreboard_image):
    """Read fixed B/S/O lamps and the inning arrow from the 2024 MBC score bug."""
    if scoreboard_image is None or scoreboard_image.shape[:2] != (59,1520):
        return {}
    hsv=cv2.cvtColor(scoreboard_image,cv2.COLOR_BGR2HSV)
    def lit(xs,low,high):
        total=0
        for x in xs:
            patch=hsv[22:37,x-5:x+6]
            colored=(patch[:,:,1]>70)&(patch[:,:,2]>75)&(patch[:,:,0]>=low)&(patch[:,:,0]<=high)
            total+=float(colored.mean())>.5
        return int(total)
    green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>70)&(hsv[:,:,2]>90)
    ys,_=np.where(green[16:42,48:72]);half=''
    if len(ys)>35:
        top,bottom=int(ys.min())+16,int(ys.max())+16
        if top<=22 and bottom<=33:half='top'
        elif top>=24 and bottom>=35:half='bottom'
    return dict(balls_on_screen=lit((547,573,599),35,95),
                strikes_on_screen=lit((660,686),15,40),
                outs_on_screen=lit((750,776),170,179),half_on_screen=half)


def sbs_scoreboard_state(scoreboard_image):
    if scoreboard_image is None or scoreboard_image.shape[:2] != (125,604):return {}
    gray=cv2.cvtColor(scoreboard_image,cv2.COLOR_BGR2GRAY)
    count_roi=cv2.resize(gray[0:54,530:604],None,fx=5,fy=5,interpolation=cv2.INTER_CUBIC)
    pair=ocr(count_roi,'eng',13,False)
    parsed=re.search(r'([0-3])\s*-\s*([0-2])',pair)
    histogram=(gray[69:84,468:488]<65).sum(axis=1)
    visible=np.flatnonzero(histogram);half=''
    if len(visible)>=7:
        upper=histogram[visible[:3]].sum();lower=histogram[visible[-3:]].sum()
        if lower>upper*1.4:half='top'
        elif upper>lower*1.4:half='bottom'
    hsv=cv2.cvtColor(scoreboard_image,cv2.COLOR_BGR2HSV)
    outs=0
    for x in (553,577):
        patch=hsv[62:78,x-5:x+6]
        red=(patch[:,:,1]>70)&(patch[:,:,2]>65)&((patch[:,:,0]<15)|(patch[:,:,0]>165))
        outs+=float(red.mean())>.4
    return dict(balls_on_screen=parsed.group(1) if parsed else '',
                strikes_on_screen=parsed.group(2) if parsed else '',
                outs_on_screen=int(outs),half_on_screen=half,count_pair_ocr=pair)


def mbc_alternative_inning(scoreboard_image):
    if scoreboard_image is None or scoreboard_image.shape[:2] != (59,1520):return ''
    gray=cv2.cvtColor(scoreboard_image[9:47,12:46],cv2.COLOR_BGR2GRAY)
    enlarged=255-cv2.resize(gray,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    padded=cv2.copyMakeBorder(enlarged,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
    return ocr(padded,'eng',7,True)


def resolve_mbc_inning(observation,scoreboard_image,rows):
    """Accept an alternative OCR only when the other score-bug fields identify one relay pitch."""
    if observation.get('inning_alt_ocr','')=='':
        observation['inning_alt_ocr']=mbc_alternative_inning(scoreboard_image)
    alternative=parse_number(observation['inning_alt_ocr'],1,15)
    if alternative is None or str(alternative)==str(observation.get('inning','')):return
    count=observation.get('pitches_completed_on_screen','')
    half=observation.get('half_on_screen','')
    state=[observation.get(field,'') for field in ('balls_on_screen','strikes_on_screen','outs_on_screen')]
    if count=='' or half not in ('top','bottom') or any(value=='' for value in state):return
    def hits(inning):
        return [row for row in rows if str(row['inning'])==str(inning) and row['half_inning']==half
                and row['pitcher_pitch_count']==int(count)+1
                and [str(row[field]) for field in ('balls_before','strikes_before','outs_before')]==[str(v) for v in state]]
    if len(hits(observation.get('inning','')))==0 and len(hits(alternative))==1:
        observation['inning']=alternative
        observation['inning_resolution']='alt_ocr_relay_consistent'


def pitch_index(pitches):
    by_game=defaultdict(list)
    for row in pitches:by_game[row['naver_game_id']].append(dict(row))
    for rows in by_game.values():
        rows.sort(key=lambda r:(int(r['inning']),0 if r['half_inning']=='top' else 1,
                                int(r['relay_seqno']) if str(r['relay_seqno']).isdigit() else 0))
        counts=Counter()
        for row in rows:
            # Published tracking CSVs omit untracked pitches and relay order.
            # Their row positions cannot establish cumulative pitcher counts.
            if row.get('extraction_method')=='published_naver_pts_csv':
                row['pitcher_pitch_count']=None
                continue
            key=row['pitcher_code'] or row['pitcher_name']
            counts[key]+=1
            saved=row.get('pitcher_pitch_number','')
            if saved!='' and int(saved)!=counts[key]:
                raise ValueError(f'inconsistent pitcher pitch order: {row["naver_pitch_id"]}: {saved} != {counts[key]}')
            row['pitcher_pitch_count']=counts[key]
    return by_game


def observe_clip(root, clip, source, game_rows, layout='auto'):
    out=root/'naver_sports';thumb_rel=f'evidence/scoreboards/{clip["dataset_clip_id"]}.jpg'
    names_pitcher={r['pitcher_name'] for r in game_rows if r['pitcher_name']}
    names_batter={r['batter_name'] for r in game_rows if r['batter_name']}
    candidates=[]
    start=float(clip['start_time']);end=float(clip['end_time'])
    times=[(start+end)/2, start+min(.35,max(0,end-start)/4)]
    with av.open(source['source_path']) as container:
        stream=container.streams.video[0]
        for attempt,sec in enumerate(times,1):
            frame,actual=frame_at(container,stream,sec)
            if frame is None:continue
            layouts=['sbs','sbs_left'] if layout in ('sbs','sbs_left') else [layout] if layout in LAYOUTS else LAYOUTS
            for chosen_layout in layouts:
                values=read_scoreboard(frame,chosen_layout)
                pitcher,pm=name_match(values['pitcher'],names_pitcher)
                batter,bm=name_match(values['batter'],names_batter)
                inning=parse_number(values['inning'],1,15)
                count=parse_number(values['pitch_count'],0,250)
                strength=2*(pm=='exact')+2*(bm=='exact')+(pm=='fuzzy_review')+(bm=='fuzzy_review')+bool(inning)+(count is not None)
                candidates.append((strength,frame,actual,values,pitcher,batter,pm,bm,inning,count,attempt,chosen_layout))
                if strength==6:break
            if max(c[0] for c in candidates)==6:break
    if not candidates:
        return dict(dataset_clip_id=clip['dataset_clip_id'],source_id=clip['source_id'],
                    naver_game_id=clip['naver_game_id'],ocr_status='frame_unavailable',frame_attempts=0)
    _,frame,actual,values,pitcher,batter,pm,bm,inning,count,attempt,chosen_layout=max(candidates,key=lambda x:x[0])
    thumb=scoreboard_crop(frame,LAYOUTS[chosen_layout]['thumb'])
    destination=out/thumb_rel;destination.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(destination),thumb,[cv2.IMWRITE_JPEG_QUALITY,88])
    lights=(spotv_count_lights(thumb) if chosen_layout=='spotv' else
            mbc_scoreboard_state(thumb) if chosen_layout=='mbc' else
            sbs_scoreboard_state(thumb) if chosen_layout in ('sbs','sbs_left') else {})
    status='four_fields_exact' if pm=='exact' and bm=='exact' and inning is not None and count is not None else 'needs_review'
    return dict(dataset_clip_id=clip['dataset_clip_id'],source_id=clip['source_id'],
        naver_game_id=clip['naver_game_id'],layout=chosen_layout,ocr_version=OCR_VERSION,
        source_time=round(actual,4),scoreboard_image=thumb_rel,
        pitcher_ocr=values['pitcher'],batter_ocr=values['batter'],inning_ocr=values['inning'],
        pitch_count_ocr=values['pitch_count'],pitcher_name=pitcher,batter_name=batter,
        inning=inning or '',pitches_completed_on_screen=count if count is not None else '',pitcher_match=pm,batter_match=bm,
        ocr_status=status,frame_attempts=attempt,**lights)


def candidates_for(clip,observation,rows):
    rows=[r for r in rows if r.get('pitcher_pitch_count') is not None
          and r.get('extraction_method')!='published_naver_pts_csv']
    if not rows:return []
    pitcher=observation.get('pitcher_name','');batter=observation.get('batter_name','')
    inning=observation.get('inning','');count=observation.get('pitches_completed_on_screen','')
    if not inning or count=='':return []
    # The score bug's P is the number already thrown. The on-screen delivery is P+1.
    target=int(count)+1
    def state_match(row):
        checks=(('balls_on_screen','balls_before'),('strikes_on_screen','strikes_before'),
                ('outs_on_screen','outs_before'))
        available=[(observation.get(screen,''),row.get(record,'')) for screen,record in checks]
        if any(screen=='' or record=='' for screen,record in available):return 'unavailable'
        return 'pre_pitch_match' if all(int(screen)==int(record) for screen,record in available) else 'mismatch_review'
    fixed_state=(observation.get('layout') in ('mbc','spotv','sbs','sbs_left')
                 and observation.get('half_on_screen') in ('top','bottom')
                 and all(observation.get(field,'')!='' for field in ('balls_on_screen','strikes_on_screen','outs_on_screen')))
    if fixed_state:
        # The fixed score bug can identify a pitch even when player-name OCR fails.
        matched=[r for r in rows if str(r['inning'])==str(inning)
                 and r['half_inning']==observation['half_on_screen']]
        exact=[r for r in matched if r['pitcher_pitch_count']==target and state_match(r)=='pre_pitch_match']
    else:
        if not pitcher and not batter:return []
        matched=[r for r in rows if (not pitcher or r['pitcher_name']==pitcher)
                 and (not batter or r['batter_name']==batter) and str(r['inning'])==str(inning)]
        exact=[r for r in matched if r['pitcher_pitch_count']==target]
    if not matched:return []
    name_conflict=(len(exact)==1 and observation.get('layout') in ('spotv','sbs','sbs_left') and
                   ((observation.get('pitcher_match')=='exact' and pitcher!=exact[0]['pitcher_name']) or
                    (observation.get('batter_match')=='exact' and batter!=exact[0]['batter_name'])))
    unique_strong=(len(exact)==1 and not name_conflict and (fixed_state or (pitcher and batter
                   and observation['pitcher_match']=='exact' and observation['batter_match']=='exact'
                   and observation['ocr_status']=='four_fields_exact')))
    ordered=sorted(matched,key=lambda r:(0 if r in exact else 1,
                         abs(r['pitcher_pitch_count']-target),
                         r['pitcher_pitch_count']))[:5]
    evidence=f'OCR source_time={observation.get("source_time","")}; pitcher={observation.get("pitcher_ocr","")}; batter={observation.get("batter_ocr","")}; inning={observation.get("inning","")}; inning_source={observation.get("inning_resolution","primary")}; half={observation.get("half_on_screen","")}; displayed_P_completed={observation.get("pitch_count_ocr","")}; target_pitcher_pitch_number={target}; screen_BSO={observation.get("balls_on_screen","")}/{observation.get("strikes_on_screen","")}/{observation.get("outs_on_screen","")}'
    result=[]
    for rank,row in enumerate(ordered,1):
        consistency=state_match(row)
        status='proposed' if unique_strong and row in exact and consistency!='mismatch_review' else 'needs_review'
        result.append(dict(dataset_clip_id=clip['dataset_clip_id'],naver_pitch_id=row['naver_pitch_id'],
            rank=rank,match_status=status,match_basis='scorebug_P_half_BSO' if fixed_state else 'ocr_names_inning_P',
            match_score=sum([row in exact,
                observation['pitcher_match']=='exact',observation['batter_match']=='exact'])+
                (2 if row in exact else 0),pitcher_name=row['pitcher_name'],batter_name=row['batter_name'],
            inning=row['inning'],half_inning=row['half_inning'],pitcher_pitch_number=row['pitcher_pitch_count'],
            pitch_number=row['pitch_number'],pitch_type_raw=row['pitch_type_raw'],speed_kmh=row['speed_kmh'],
            count_state_match=consistency,evidence=evidence,scoreboard_image=observation.get('scoreboard_image','')))
    return result


def write_review(root,clips,observations,candidates,links):
    naver=root/'naver_sports';obs={r['dataset_clip_id']:r for r in observations};by_clip=defaultdict(list)
    link_by={(r['dataset_clip_id'],r['naver_pitch_id']):r for r in links}
    for row in candidates:by_clip[row['dataset_clip_id']].append(row)
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>네이버 투구 연결 검토</title>',
           '<style>body{font:14px sans-serif;margin:2rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:5px;vertical-align:top}img{width:210px}video{width:300px}small{color:#666}</style>',
           '<h1>네이버 투구 연결 후보</h1><p>화면 P는 이미 던진 공 수로 읽으며 현재 투구는 P+1번째 공으로 대조합니다. accepted는 자동 근거 검사를 통과한 연결이고, limited는 일부 화면 근거가 없는 연결입니다. 수동 정답 판정은 아닙니다.</p>',
           '<table><tr><th>클립</th><th>화면 정보</th><th>후보</th><th>영상</th></tr>']
    for clip in sorted(clips,key=lambda r:(r['source_id'],float(r['start_time']))):
        cid=clip['dataset_clip_id'];ob=obs.get(cid,{})
        image=f'<img src="{html.escape(ob["scoreboard_image"])}">' if ob.get('scoreboard_image') else '없음'
        details=f'{html.escape(str(ob.get("pitcher_ocr","")))} / {html.escape(str(ob.get("batter_ocr","")))}<br>inning {html.escape(str(ob.get("inning","")))} {html.escape(str(ob.get("half_on_screen","")))} · P {html.escape(str(ob.get("pitch_count_ocr","")))}<br>B/S/O {html.escape(str(ob.get("balls_on_screen","?")))}/{html.escape(str(ob.get("strikes_on_screen","?")))}/{html.escape(str(ob.get("outs_on_screen","?")))}<br><small>{html.escape(ob.get("ocr_status","not_processed"))} · {html.escape(ob.get("inning_resolution","primary"))}</small>'
        hits='<br>'.join(f'{html.escape(x["match_status"])} / {html.escape(link_by.get((cid,x["naver_pitch_id"]),{}).get("auto_decision",""))}: {html.escape(x["naver_pitch_id"])} · 투수 {html.escape(str(x["pitcher_pitch_number"]))}번째 공 · {html.escape(x["pitch_type_raw"])} · {html.escape(str(x["speed_kmh"]))} km/h · {html.escape(x.get("count_state_match",""))} · {html.escape(x.get("match_basis",""))}' for x in by_clip[cid])
        if not hits:
            if ob.get('pitches_completed_on_screen','')=='':
                reason='화면 투구수 P를 읽지 못함'
            elif ob.get('inning','')=='':
                reason='화면 이닝을 읽지 못함'
            elif ob.get('layout') in ('spotv','mbc','sbs','sbs_left') and ob.get('half_on_screen','')=='':
                reason='이닝 방향을 판독하지 못함'
            elif ob.get('layout') in ('spotv','mbc','sbs','sbs_left') and ob.get('balls_on_screen','')=='':
                reason='볼·스트라이크 표시를 판독하지 못함'
            elif not ob.get('pitcher_name') and not ob.get('batter_name') and ob.get('layout')=='kbs':
                reason='투수·타자 이름을 문자중계와 연결하지 못함'
            else:
                reason='화면 이닝·투구수·카운트에 맞는 문자중계 기록이 없음'
            hits='후보 없음<br><small>'+reason+'</small>'
        path='../'+clip['crop_video']
        video=f'<video controls preload="none" src="{html.escape(path)}"></video>'
        lines.append(f'<tr><td>{html.escape(cid)}<br>{html.escape(clip["naver_game_id"])}<br>{clip["start_time"]}–{clip["end_time"]}s</td><td>{image}<br>{details}</td><td>{hits}</td><td>{video}</td></tr>')
    lines.append('</table></html>')
    (naver/'match_review.html').write_text('\n'.join(lines),encoding='utf-8')


def match(dataset_root,source_ids=None,refresh=False,workers=3):
    root=Path(dataset_root).resolve();naver=root/'naver_sports'
    sources={r['source_id']:r for r in read_csv(root/'sources.csv')}
    clips=read_csv(root/'clips.csv');pitches=pitch_index(read_csv(naver/'pitches.csv'))
    selected=[c for c in clips if (not source_ids or c['source_id'] in source_ids)
              and c.get('naver_game_id') in pitches]
    active_ids={c['dataset_clip_id'] for c in clips}
    if source_ids and not set(source_ids)<=set(sources):raise ValueError('unknown source ID')
    previous={r['dataset_clip_id']:r for r in read_csv(naver/'scoreboard_observations.csv')}
    observations={key:value for key,value in previous.items() if key in active_ids}
    for observation in observations.values():
        if observation.get('pitches_completed_on_screen','')=='':
            old=observation.get('pitcher_pitch_count','')
            parsed=parse_number(observation.get('pitch_count_ocr',''),0,250)
            observation['pitches_completed_on_screen']=old if old!='' else parsed if parsed is not None else ''
        if (observation.get('pitches_completed_on_screen','')!=''
                and observation.get('pitcher_match')=='exact'
                and observation.get('batter_match')=='exact'
                and observation.get('inning','')!=''):
            observation['ocr_status']='four_fields_exact'
    layouts={r['provider_video_id']:r['layout'] for r in read_csv(naver/'scoreboard_layouts.csv')
             if r.get('provider_video_id') and r.get('layout') in LAYOUTS}
    work=[c for c in selected if refresh or c['dataset_clip_id'] not in observations
          or observations[c['dataset_clip_id']].get('ocr_version')!=OCR_VERSION]
    print(f'MATCH OCR {len(work)} clips; {len(selected)-len(work)} cached',flush=True)
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        jobs={pool.submit(observe_clip,root,c,sources[c['source_id']],pitches[c['naver_game_id']],
                          layouts.get(sources[c['source_id']].get('provider_video_id'),'auto')):c for c in work}
        for index,future in enumerate(as_completed(jobs),1):
            clip=jobs[future]
            try:observations[clip['dataset_clip_id']]=future.result()
            except Exception as exc:
                observations[clip['dataset_clip_id']]=dict(dataset_clip_id=clip['dataset_clip_id'],
                    source_id=clip['source_id'],naver_game_id=clip['naver_game_id'],
                    ocr_status=f'failed:{type(exc).__name__}:{exc}',frame_attempts=0)
            if index%25==0 or index==len(work):print(f'MATCH OCR {index}/{len(work)}',flush=True)
    for clip in selected:
        observation=observations.get(clip['dataset_clip_id'],{})
        if observation.get('layout') in ('spotv','mbc','sbs','sbs_left') and (observation.get('balls_on_screen','')=='' or
                observation.get('half_on_screen','')==''):
            image_path=observation.get('scoreboard_image','')
            if image_path:
                image=cv2.imread(str(naver/image_path))
                observation.update(spotv_count_lights(image) if observation['layout']=='spotv' else
                                   mbc_scoreboard_state(image) if observation['layout']=='mbc' else
                                   sbs_scoreboard_state(image))
        if observation.get('layout')=='mbc':
            image_path=observation.get('scoreboard_image','')
            if image_path:
                image=cv2.imread(str(naver/image_path))
                resolve_mbc_inning(observation,image,pitches[clip['naver_game_id']])
    old_links=read_csv(naver/'clip_pitch_links.csv')
    kept=[x for x in old_links if x['dataset_clip_id'] in active_ids and
          (x['link_status']=='verified' or x['dataset_clip_id'] not in {c['dataset_clip_id'] for c in selected})]
    verified={x['dataset_clip_id'] for x in kept if x['link_status']=='verified'}
    selected_ids={c['dataset_clip_id'] for c in selected}
    all_candidates=[r for r in read_csv(naver/'match_candidates.csv')
                    if r['dataset_clip_id'] in active_ids and r['dataset_clip_id'] not in selected_ids]
    proposed=0
    for clip in selected:
        cid=clip['dataset_clip_id'];ob=observations.get(cid,{})
        choices=candidates_for(clip,ob,pitches[clip['naver_game_id']]);all_candidates.extend(choices)
        for choice in choices:
            if choice['match_status']=='proposed' and cid not in verified:
                kept.append(dict(dataset_clip_id=cid,naver_pitch_id=choice['naver_pitch_id'],
                    link_status='proposed',evidence=choice['evidence']+'; count_state='+choice['count_state_match']+'; scoreboard='+choice['scoreboard_image'],
                    video_to_pitch_time_offset_seconds='',reviewer=''))
                proposed+=1
    from .audit_naver import audit_links
    kept=audit_links(root,kept,clips,list(observations.values()),pitches,sources,selected_ids)
    link_status=defaultdict(set)
    for row in kept:link_status[row['dataset_clip_id']].add(row['link_status'])
    for clip in clips:
        statuses=link_status[clip['dataset_clip_id']]
        clip['naver_match_status']='matched_verified' if 'verified' in statuses else 'candidate' if 'proposed' in statuses else 'unmatched'
    atomic_csv(naver/'scoreboard_observations.csv',sorted(observations.values(),key=lambda r:r['dataset_clip_id']),OBS_FIELDS)
    atomic_csv(naver/'match_candidates.csv',sorted(all_candidates,key=lambda r:(r['dataset_clip_id'],r['rank'])),CAND_FIELDS)
    atomic_csv(naver/'clip_pitch_links.csv',sorted(kept,key=lambda r:(r['dataset_clip_id'],r['naver_pitch_id'])),LINK_FIELDS)
    atomic_csv(root/'clips.csv',clips,CLIP_FIELDS)
    write_review(root,clips,list(observations.values()),all_candidates,kept)
    write_dataset_summary(root)
    statuses=Counter(observations[c['dataset_clip_id']]['ocr_status'] for c in selected)
    decisions=Counter(row.get('auto_decision','') for row in kept if row['dataset_clip_id'] in selected_ids)
    print(f'MATCH TOTAL selected={len(selected)} proposed={proposed} auto={dict(decisions)} candidates={len(all_candidates)} OCR={dict(statuses)}',flush=True)
    return 1 if any(key.startswith('failed:') or key=='frame_unavailable' for key in statuses) else 0
