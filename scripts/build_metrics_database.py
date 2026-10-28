"""Build a pitch-indexed mechanics database without changing any source poses.

Record pitches and unresolved motion clips are distinct units. Event-free
statistics describe the observed clip, never an invented pitching phase.
Native 3D geometry is retained as needs_review_3d, excluded from default
aggregates. All uncomputable values are explicit nulls with metric-level reasons.
"""
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
OUT = DATA/'metrics'
VERSION = 'mechanics-db-20261028-v1'
MIN_FRAMES = 10
MIN_FRACTION = .5


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    a = (len(values)-1)*p
    i = int(a)
    return values[i]+(values[min(len(values)-1,i+1)]-values[i])*(a-i)


def vector(a,b):
    return [x-y for x,y in zip(a[:3],b[:3])]


def angle(a,b):
    den = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if den < 1e-9:
        return None
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))/den))))


def bend(a,b,c,dim):
    included = angle([a[i]-b[i] for i in range(dim)],[c[i]-b[i] for i in range(dim)])
    return None if included is None else 180-included


def good2(frame,ids):
    for i in ids:
        p = frame['keypoints'][i]
        if not p or len(p)<3 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p[:3]) or p[2]<.5:
            return False
        for key in ['interpolated','outlier']:
            flag=frame.get(key,False)
            if flag[i] if isinstance(flag,list) else flag:
                return False
    return True


def native(frame,ids):
    if frame.get('available') is False:
        return False
    return all(i<len(frame.get('anatomy',[])) and len(frame['anatomy'][i])>=3 and all(math.isfinite(x) for x in frame['anatomy'][i][:3]) for i in ids)


def metric(key,ko,en,group,basis,definition,joints,formula,unit='°',event='observed_clip',conditions=None):
    return dict(id=key,name={'ko':ko,'en':en},group=group,unit=unit,basis=basis,
                definition={'ko':definition[0],'en':definition[1]},formula=formula,
                coordinateSystem=('Halpe-26 crop pixels; image x right, y down' if basis=='2d' else 'SMPL-24 joint centers; clip-fixed inferred support; Y up; uncalibrated scale'),
                referenceEvent=event,requiredJoints=joints,calculationVersion=VERSION,
                conditions=conditions or 'At least 10 finite supported frames and >=50% of the clip; original 2D score >=0.5 at required observations; no interpolated/outlier joints. No gap filling or phase inference.',
                limitations=('Image projection depends on view and clip boundaries; not anatomical 3D flexion.' if basis=='2d' else 'Unvalidated monocular 3D geometry; excluded from default aggregate statistics. Not a laboratory measurement.'))


def dictionary():
    result=[]
    for stem,ko,en,group,joints in [
        ('lead_knee','앞무릎 투영 굴곡','Lead-knee projected flexion','lead_leg','R:11-13-15; L:12-14-16'),
        ('throw_elbow','투구팔꿈치 투영 굴곡','Throwing-elbow projected flexion','arm','R:6-8-10; L:5-7-9')]:
        for suffix,label,english,formula in [('p10','P10','P10','Q0.10(flexion)'),('p90','P90','P90','Q0.90(flexion)'),('range','변화폭 P90−P10','range P90−P10','Q0.90(flexion)-Q0.10(flexion)')]:
            result.append(metric(f'{stem}_{suffix}_2d',f'{ko} {label}',f'{en} {english}',group,'2d',
                (f'관측된 클립 구간에서 {ko}의 {label}. 0°는 영상에서 펴진 관절입니다. 착지–릴리스 구간의 신전량과 다릅니다.',f'{english} of projected flexion across the observed clip. 0° means projected straight. Distinct from event-aligned extension.'),
                joints,'flexion=180-acos((a-b)·(c-b)/(|a-b||c-b|)); '+formula))
    result.extend([
        metric('trunk_tilt_p90_2d','몸통 영상 수직선 기울기 P90','Image-vertical trunk tilt P90','posture','2d',
               ('골반 중점→어깨 중점 벡터와 영상 위쪽 수직선 사이 각도의 P90입니다. 해부학적 전경·측굴과 구분합니다.','P90 angle between hip-midpoint→shoulder-midpoint and image up. Does not identify anatomical forward or lateral lean.'),
               '5,6,11,12','Q0.90(angle(shoulder_mid-hip_mid,[0,-1]))'),
        metric('arm_elevation_p90_2d','투구 상완–몸통 투영각 P90','Projected upper-arm / trunk angle P90','arm','2d',
               ('어깨→투구 팔꿈치와 어깨 중점→골반 중점 사이 투영각의 P90입니다. 팔이 몸통 아래 방향이면 0°입니다.','P90 projected angle of throwing upper arm relative to downward trunk. Arm along downward trunk is 0°.'),
               'R:6,8; L:5,7; torso:5,6,11,12','Q0.90(angle(elbow-shoulder,hip_mid-shoulder_mid))'),
        metric('separation_p50_3d','어깨선–골반선 수평 분리각 P50','Shoulder–pelvis horizontal separation P50','pelvis_trunk','3d',
               ('모델의 좌→우 어깨선과 좌→우 골반선을 추정 지면에 투영한 무부호 각도의 P50입니다. 해부학적 흉곽 회전이 아닙니다.','P50 unsigned angle between left-to-right shoulder and hip lines projected onto inferred ground. Not anatomical thorax rotation.'),
               'SMPL:1,2,16,17','Q0.50(angle(shoulder_line_XZ,hip_line_XZ))'),
        metric('separation_p90_3d','어깨선–골반선 수평 분리각 P90','Shoulder–pelvis horizontal separation P90','pelvis_trunk','3d',
               ('관측 클립의 모델 수평 분리각 P90입니다. 최대 분리각이나 착지 시 값과 다릅니다.','P90 horizontal separation across the observed clip. Not maximum separation or the landing value.'),
               'SMPL:1,2,16,17','Q0.90(angle(shoulder_line_XZ,hip_line_XZ))'),
        metric('pelvis_rotation_range_3d','골반선 관측 회전폭 P90−P10','Observed pelvis-line rotation range P90−P10','pelvis_trunk','3d',
               ('가장 긴 연속 유효 구간에서 수평 골반선 방향을 연속화한 후 P90−P10을 계산합니다. 투구 단계나 회전 속도를 뜻하지 않습니다.','Unwrapped horizontal hip-line yaw P90−P10 in the longest contiguous supported run. Not a pitching-phase excursion or angular velocity.'),
               'SMPL:1,2; Halpe:11,12','unwrap(atan2(hip_line.z,hip_line.x)); Q0.90-Q0.10'),
        metric('trunk_inclination_p90_3d','몸통 추정 수직선 기울기 P90','Estimated vertical trunk inclination P90','posture','3d',
               ('모델 골반→목 벡터와 추정 장면의 수직선 사이 각도의 P90입니다. 투구 방향 기준 전경·측굴로 해석하지 않습니다.','P90 angle of model pelvis→neck against estimated scene vertical. Not forward/lateral lean relative to pitching direction.'),
               'SMPL:0,12; Halpe:5,6,11,12','Q0.90(angle(neck-pelvis,[0,1,0]))'),
        metric('arm_elevation_p90_3d','투구 상완–몸통 추정각 P90','Estimated upper-arm / trunk angle P90','arm','3d',
               ('투구측 어깨→팔꿈치와 모델 목→골반 사이 각도의 P90입니다. 해부학적 어깨 외전이나 공의 진입각이 아닙니다.','P90 model upper-arm angle against downward trunk. Not anatomical shoulder abduction or ball approach angle.'),
               'R SMPL:17,19; L:16,18; trunk:0,12','Q0.90(angle(elbow-shoulder,pelvis-neck))'),
        metric('ankle_spread_p90_3d','정규화 발목 수평 간격 P90','Normalized horizontal ankle separation P90','space','3d',
               ('발목 사이 XZ 거리 / 양쪽 고관절–무릎–발목 길이 평균의 P90입니다. 착지 보폭이나 실측 거리가 아닙니다.','P90 ankle XZ separation divided by mean hip–knee–ankle segment length. Not landing stride or measured distance.'),
               'SMPL:1,2,4,5,7,8','Q0.90(norm((ankle_L-ankle_R)_XZ)/mean(leg_segment_lengths))',unit='ratio'),
    ])
    event_specs=[
        ('landing_knee_flexion_2d','착지 시 앞무릎 투영 굴곡','Lead-knee projected flexion at landing','landingFlexion','착지 후보 범위 중앙에서의 투영 굴곡입니다.','Projected flexion at landing-bracket midpoint.','°'),
        ('extra_knee_flexion_2d','착지 후 앞무릎 추가 굴곡','Additional projected knee bending after landing','extraFlexion','착지–릴리스 최대 굴곡 − 착지 굴곡입니다.','Maximum aligned flexion minus landing flexion.','°'),
        ('knee_extension_2d','최대 굴곡 후 앞무릎 신전량','Projected knee straightening after maximum bend','extension','착지–릴리스 최대 굴곡 − 릴리스 굴곡입니다. 착지 기준 순 신전량과 다릅니다.','Maximum aligned flexion minus release flexion. Distinct from net extension from landing.','°'),
        ('net_knee_extension_2d','착지→릴리스 앞무릎 순 신전량','Net projected knee straightening landing→release','netExtension','착지 굴곡 − 릴리스 굴곡입니다. 음수는 착지 때보다 더 굽혀진 상태를 뜻합니다.','Landing flexion minus release flexion. Negative means more bent at release than landing.','°'),
        ('max_knee_flexion_progress','앞무릎 최대 굴곡 시점','Timing of maximum projected knee bend','maximumFlexionProgress','착지 0%, 릴리스 100% 정렬 구간의 첫 최대 굴곡 시점입니다. 최대 신전 속도 시점이 아닙니다.','First maximum bend in landing=0%, release=100% alignment. Not the time of peak extension speed.','%'),
    ]
    for key,ko,en,source,definition,eng,unit in event_specs:
        m=metric(key,ko,en,'lead_leg','2d',(definition,eng),'R:11,13,15; L:12,14,16',f'lead-knee-pilot-20261028-v1: {source}; 3-frame mean',unit,'provisional_landing_release',
                 'Requires published provisional visual event brackets; original joint scores >=0.4 throughout; no filled observations; midpoint alignment; 3-frame box mean. Separate strict >=0.5 screen and sensitivity remain in source pilot.')
        m['calculationVersion']='lead-knee-pilot-20261028-v1';result.append(m)
    pending=[
        ('landing_release_time_ms','착지→릴리스 실제 시간','Physical landing→release time','timing','ms','physical_time_unverified','Physical source clock with duplicate/slow-motion calibration and validated event brackets'),
        ('pelvis_trunk_peak_gap_ms','골반–몸통 회전 피크 시간 차이','Pelvis–trunk rotation peak-time offset','timing','ms','physical_time_unverified','Validated physical clock, segment coordinates, filtering and stable rotation peaks'),
        ('trunk_forward_lean_release','릴리스 몸통 전경','Forward trunk lean at release','posture','°','pitch_direction_unverified','Calibrated pitching-direction frame plus audited release'),
        ('trunk_lateral_lean_release','릴리스 몸통 측굴','Lateral trunk lean at release','posture','°','pitch_direction_unverified','Calibrated pitching-direction frame plus audited release'),
        ('normalized_stride_landing','착지 시 정규화 보폭','Normalized stride at landing','space','ratio','pitch_direction_unverified','Audited landing plus validated stride direction, contact and model geometry'),
    ]
    for key,ko,en,group,unit,reason,condition in pending:
        m=metric(key,ko,en,group,'3d',('필요 조건이 검증되지 않아 현재 결측입니다. 다른 지표나 클립 진행률로 대체하지 않습니다.','Currently missing because required conditions are unvalidated. Never substituted with another metric or clip progress.'),
                 'See required conditions','Not computed',unit,'audited_landing_release',condition)
        m['pendingReason']=reason;result.append(m)
    return result


def null_cell(reason,total=0,valid=0):
    return dict(value=None,status=reason,validFrames=valid,totalFrames=total)


def summary_cell(values,total,stat,basis):
    supported=[(i,v) for i,v in enumerate(values) if v is not None and math.isfinite(v)]
    valid=len(supported)
    if valid<MIN_FRAMES or valid/max(1,total)<MIN_FRACTION:
        return null_cell('insufficient_observations',total,valid)
    nums=[v for _,v in supported]
    value=quantile(nums,.9)-quantile(nums,.1) if stat=='range' else quantile(nums,float(stat[1:])/100)
    return dict(value=round(value,4),status='observed_2d' if basis=='2d' else 'needs_review_3d',validFrames=valid,totalFrames=total,frameRange=[supported[0][0],supported[-1][0]])


def unwrap_longest(values):
    runs=[];run=[]
    for i,v in enumerate(values):
        if v is None:
            if run:runs.append(run);run=[]
        else:run.append((i,v))
    if run:runs.append(run)
    run=max(runs,key=len,default=[])
    result=[None]*len(values);last=None;current=None
    for i,v in run:
        current=v if last is None else current+(v-last+180)%360-180
        result[i]=current;last=v
    return result


def clip_metrics(clip,hand,defs,pilot):
    pose_path=DATA/clip['pose'];pose=read(pose_path);frames=pose['frames'];total=len(frames)
    ids={'lead_knee':[11,13,15] if hand=='R' else [12,14,16], 'throw_elbow':[6,8,10] if hand=='R' else [5,7,9]}
    values={key:[] for key in ['lead_knee','throw_elbow','trunk_tilt','arm_elevation']}
    for frame in frames:
        points=frame['keypoints']
        for key in ['lead_knee','throw_elbow']:
            values[key].append(bend(*(points[i] for i in ids[key]),2) if hand in ['R','L'] and good2(frame,ids[key]) else None)
        torso_ids=[5,6,11,12]
        good=good2(frame,torso_ids)
        shoulder=[(points[5][i]+points[6][i])/2 for i in range(2)] if good else None
        hip=[(points[11][i]+points[12][i])/2 for i in range(2)] if good else None
        values['trunk_tilt'].append(angle([shoulder[i]-hip[i] for i in range(2)],[0,-1]) if good else None)
        arm_ids=[6,8] if hand=='R' else [5,7]
        values['arm_elevation'].append(angle([points[arm_ids[1]][i]-points[arm_ids[0]][i] for i in range(2)],[hip[i]-shoulder[i] for i in range(2)]) if hand in ['R','L'] and good and good2(frame,arm_ids) else None)
    cells={}
    for key in ['lead_knee','throw_elbow']:
        for stat in ['p10','p90','range']:
            cells[f'{key}_{stat}_2d']=summary_cell(values[key],total,stat,'2d') if hand in ['R','L'] else null_cell('throwing_side_unknown',total)
    for key in ['trunk_tilt','arm_elevation']:
        cells[f'{key}_p90_2d']=summary_cell(values[key],total,'p90','2d') if key=='trunk_tilt' or hand in ['R','L'] else null_cell('throwing_side_unknown',total)
    model_path=DATA/clip['pose3d'];model=read(model_path)
    aligned=(len(model['frames'])==total and abs(model['fps']-pose['fps'])<1e-8 and all(f.get('source_frame')==g.get('source_frame') and f.get('source_time')==g.get('source_time') for f,g in zip(frames,model['frames'])))
    features={key:[] for key in ['separation','pelvis_rotation','trunk_inclination','arm_elevation','ankle_spread']}
    for i,frame in enumerate(model['frames'] if aligned else []):
        observed=frames[i];a=frame.get('anatomy',[])
        good=native(frame,[1,2,16,17]) and good2(observed,[5,6,11,12])
        hipline=vector(a[2],a[1]) if good else None
        shoulders=vector(a[17],a[16]) if good else None
        # Reject almost vertical lines whose horizontal orientation is ill-defined.
        separation=angle([hipline[0],hipline[2]],[shoulders[0],shoulders[2]]) if good and math.hypot(hipline[0],hipline[2])>.2*math.sqrt(sum(v*v for v in hipline)) and math.hypot(shoulders[0],shoulders[2])>.2*math.sqrt(sum(v*v for v in shoulders)) else None
        features['separation'].append(separation)
        hipgood=native(frame,[1,2]) and good2(observed,[11,12])
        line=vector(a[2],a[1]) if hipgood else None
        features['pelvis_rotation'].append(math.degrees(math.atan2(line[2],line[0])) if hipgood and math.hypot(line[0],line[2])>.2*math.sqrt(sum(v*v for v in line)) else None)
        torsogood=native(frame,[0,12]) and good2(observed,[5,6,11,12])
        features['trunk_inclination'].append(angle(vector(a[12],a[0]),[0,1,0]) if torsogood else None)
        arm=[17,19] if hand=='R' else [16,18];arm2=[6,8] if hand=='R' else [5,7]
        features['arm_elevation'].append(angle(vector(a[arm[1]],a[arm[0]]),vector(a[0],a[12])) if hand in ['R','L'] and torsogood and native(frame,arm) and good2(observed,arm2) else None)
        footgood=native(frame,[1,2,4,5,7,8]) and good2(observed,[11,12,13,14,15,16])
        leg=sum(math.dist(a[h][:3],a[k][:3])+math.dist(a[k][:3],a[n][:3]) for h,k,n in [(1,4,7),(2,5,8)])/2 if footgood else 0
        features['ankle_spread'].append(math.hypot(a[7][0]-a[8][0],a[7][2]-a[8][2])/leg if footgood and leg>1e-8 else None)
    for name,feature,stat in [('separation_p50_3d','separation','p50'),('separation_p90_3d','separation','p90'),('pelvis_rotation_range_3d','pelvis_rotation','range'),('trunk_inclination_p90_3d','trunk_inclination','p90'),('arm_elevation_p90_3d','arm_elevation','p90'),('ankle_spread_p90_3d','ankle_spread','p90')]:
        series=unwrap_longest(features[feature]) if feature=='pelvis_rotation' else features[feature]
        cells[name]=summary_cell(series,total,stat,'3d') if aligned else null_cell('model_alignment_mismatch',total)
        if name=='arm_elevation_p90_3d' and hand not in ['R','L']:cells[name]=null_cell('throwing_side_unknown',total)
    for m in defs:
        if m.get('pendingReason'):
            cells[m['id']]=null_cell(m['pendingReason'],total)
        elif m['referenceEvent']=='provisional_landing_release':
            sample=pilot.get(clip['id'])
            if sample:
                key=m['formula'].split(': ')[1].split(';')[0]
                cells[m['id']]=dict(value=sample['metrics'][key],status='provisional_event',validFrames=len(sample['rawFrames']),totalFrames=len(sample['rawFrames']),frameRange=[sample['rawFrames'][0],sample['rawFrames'][-1]],sensitivity=sample['sensitivity'].get(key),strictObservationEligible=sample['strictObservationEligible'])
            else:cells[m['id']]=null_cell('event_not_audited',total)
    return dict(clipId=clip['id'],hand=hand,totalFrames=total,fps=pose['fps'],sourceId=clip['sourceId'],
                posePath=clip['pose'],poseSha256=sha(pose_path),modelPath=clip['pose3d'],modelSha256=sha(model_path),
                modelVersion=clip['reconstructionVersion'],qualityFlags=model.get('quality',{}).get('source_regression_flags',[]),
                sourceUrl=clip['sourceUrl'],metrics=cells)


def main():
    OUT.mkdir(exist_ok=True)
    defs=dictionary();index=read(DATA/'clips.json');clips={c['id']:c for c in index['clips']}
    summary=read(DATA/'portal/summary.json');profiles={p['code']:p for p in summary['players']}
    pilot={s['clipId']:s for s in read(DATA/'mechanics/65320.json')['samples']}
    records=[];links={};pitch_clips=defaultdict(list)
    source_file=ROOT.parent/'dataset/sources.csv'
    with open(source_file) as stream:
        source_games={r['source_id']:r['naver_game_id'] for r in csv.DictReader(stream) if r['naver_game_id']}
    for code in profiles:
        detail=read(DATA/f'portal/players/{code}.json')
        records.extend((code,r) for r in detail['records'])
        for cid,link in detail['clips'].items():
            links[cid]=link
            if link['status']=='auto_accepted':pitch_clips[link['pitchId']].append(cid)
    extracted={}
    for n,(cid,c) in enumerate(clips.items(),1):
        extracted[cid]=clip_metrics(c,profiles.get(c.get('playerCode'),{}).get('hand',''),defs,pilot)
        if n%100==0:print(f'Extracted {n}/{len(clips)} clips',flush=True)
    # One representative observation interval per unique provider pitch. Do not
    # mix different replay windows within a row. Prefer audited event evidence,
    # then supported 2D frames; preserve every alternate clip ID.
    def primary(ids):
        return max(ids,key=lambda cid:(cid in pilot,sum(cell['validFrames'] for k,cell in extracted[cid]['metrics'].items() if k.endswith('_2d') and cell['value'] is not None),-len(cid),cid))
    rows=[]
    for code,r in records:
        ids=sorted(pitch_clips.get(r['id'],[]));chosen=primary(ids) if ids else None
        rows.append(dict(id=r['id'],pitchId=r['id'],unit='pitch_record',playerCode=code,player=profiles[code]['name'],hand=profiles[code].get('hand',''),season='2024',gameId=r['gameId'],date=r['date'],pitch=r['pitch'] or None,speed=r['speed'],sourceId=clips[chosen]['sourceId'] if chosen else None,sourceUrl=clips[chosen]['sourceUrl'] if chosen else None,recordSourceUrl=r['sourceUrl'],clipIds=ids,primaryClipId=chosen,metrics=extracted[chosen]['metrics'] if chosen else None,linkStatus='auto_accepted' if ids else 'no_motion',qualityFlags=extracted[chosen]['qualityFlags'] if chosen else []))
    for cid,c in clips.items():
        if links.get(cid,{}).get('status')=='auto_accepted':continue
        rows.append(dict(id='clip:'+cid,pitchId=None,unit='unlinked_clip',playerCode=c.get('playerCode') or None,player=c.get('player') or None,hand=extracted[cid]['hand'],season='2024',gameId=source_games.get(c['sourceId']),date=c.get('date'),pitch=c.get('pitch') if c.get('pitch') not in ['',None,'unknown'] else None,speed=float(c['speed']) if c.get('speed') else None,sourceId=c['sourceId'],sourceUrl=c['sourceUrl'],recordSourceUrl=None,clipIds=[cid],primaryClipId=cid,metrics=extracted[cid]['metrics'],linkStatus=links.get(cid,{}).get('status','unresolved'),qualityFlags=extracted[cid]['qualityFlags']))
    rows.sort(key=lambda r:(r['date'] or '',r['id']))
    status_counts=defaultdict(Counter)
    for row in rows:
        for m in defs:status_counts[m['id']][row['metrics'][m['id']]['status'] if row['metrics'] else 'no_motion']+=1
    status={
        'observed_2d':{'ko':'2D 관측 통계','en':'2D observation statistic'},
        'needs_review_3d':{'ko':'3D 추정·검토 필요','en':'3D estimate · review required'},
        'provisional_event':{'ko':'잠정 이벤트','en':'Provisional events'},
        'no_motion':{'ko':'연결 동작 없음','en':'No linked motion'},
        'event_not_audited':{'ko':'이벤트 미검수','en':'Events not audited'},
        'insufficient_observations':{'ko':'유효 관측 부족','en':'Insufficient supported observations'},
        'throwing_side_unknown':{'ko':'투구측 미확인','en':'Throwing side unknown'},
        'physical_time_unverified':{'ko':'실제 시간 미검증','en':'Physical time unverified'},
        'pitch_direction_unverified':{'ko':'투구 기준축 미검증','en':'Pitching frame unverified'},
        'model_alignment_mismatch':{'ko':'모델 정렬 불일치','en':'Model alignment mismatch'}}
    payload=dict(schemaVersion=1,version=VERSION,poseVersion=index['packageVersion'],
        dictionary=defs,statusCodes=status,
        defaults=dict(include3dInAggregates=False,scope='linked',minimumIqrSamples=3),
        missingSemantics='metrics:null means every dictionary metric has value:null, status:no_motion, validFrames:0,totalFrames:0. Other cells always store explicit null and a reason; never impute zero.',
        primaryClipPolicy='Prefer provisionally event-audited observation; otherwise greatest sum of supported 2D frames. One whole observation interval per pitch. All alternate clips retained; no averaging across replays.',
        scope=dict(recordPitches=len(records),motionClips=len(clips),linkedPitches=len(pitch_clips),linkedClips=sum(map(len,pitch_clips.values())),unlinkedClips=sum(r['unit']=='unlinked_clip' for r in rows),rows=len(rows),metricCount=len(defs),motionPlayers=len({r['playerCode'] for r in rows if r['primaryClipId']})),
        statusCounts=dict(status_counts),rows=rows,
        provenance=dict(builderSha256=sha(Path(__file__)),clipIndexSha256=sha(DATA/'clips.json'),portalSummarySha256=sha(DATA/'portal/summary.json'),pilotSha256=sha(DATA/'mechanics/65320.json'),sourceGameMapSha256=sha(source_file),sourceClock='Broadcast playback; duplicate frames and unknown slow motion are not calibrated into physical time. Frame counts are not independent observations.'))
    (OUT/'database.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    (OUT/'database.json.gz').write_bytes(gzip.compress((OUT/'database.json').read_bytes(),compresslevel=6,mtime=0))
    (OUT/'dictionary.json').write_text(json.dumps(dict(schemaVersion=1,version=VERSION,metrics=defs,statusCodes=status),ensure_ascii=False,indent=2)+'\n')
    (OUT/'clip_provenance.json').write_text(json.dumps(dict(version=VERSION,clips={cid:{k:v for k,v in c.items() if k!='metrics'} for cid,c in extracted.items()}),ensure_ascii=False,separators=(',',':'))+'\n')
    # Full export expands absent motion into explicit missing cells. The UI
    # additionally supports exports of any selected columns and filtered rows.
    output=io.StringIO(newline='');columns=['row_id','unit','pitch_id','player_code','player','throwing_hand','game_id','date','pitch_type','speed_kmh','primary_clip_id','all_clip_ids','pitch_link_status','database_version']
    suffixes=['value','unit','status','valid_frames','total_frames','calculation_version','reference_event','source_clip_id']
    for m in defs:columns.extend(m['id']+'__'+suffix for suffix in suffixes)
    writer=csv.DictWriter(output,fieldnames=columns);writer.writeheader()
    for row in rows:
        values=[row['id'],row['unit'],row['pitchId'],row['playerCode'],row['player'],row['hand'],row['gameId'],row['date'],row['pitch'],row['speed'],row['primaryClipId'],'|'.join(row['clipIds']),row['linkStatus'],VERSION]
        out=dict(zip(columns[:14],values))
        for m in defs:
            cell=row['metrics'][m['id']] if row['metrics'] else null_cell('no_motion')
            fields=[cell['value'],m['unit'],cell['status'],cell['validFrames'],cell['totalFrames'],m['calculationVersion'],m['referenceEvent'],row['primaryClipId']]
            out.update({m['id']+'__'+suffix:value for suffix,value in zip(suffixes,fields)})
        writer.writerow(out)
    with open(OUT/'pitches.csv.gz','wb') as stream:stream.write(gzip.compress(output.getvalue().encode('utf-8-sig'),mtime=0))
    print(json.dumps(payload['scope'],ensure_ascii=False));print(f'Database JSON: {(OUT/"database.json").stat().st_size/1e6:.2f} MB')


if __name__=='__main__':
    raise SystemExit('Archived v1 builder. Use scripts/build_biomechanics_database.py for current research 3D data; the original executable is losslessly preserved in data/releases/metrics-20261028-v1/.')
