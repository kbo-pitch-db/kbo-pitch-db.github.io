#!/usr/bin/env python3
"""Build v2 from immutable v1 pitch identity and canonical 3D fit lineage.

Run observations.py first (private RGB input stays local). No network needed.
Numerical poses and derived metrics only are packaged for GitHub Pages.
"""
import argparse,collections,csv,gzip,hashlib,io,itertools,json,math,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from biomechanics.dictionary import dictionary,VERSION,SOURCES
from biomechanics.calculation import make_series,measurement,support_masks,sequence_diagnostic

REPO=Path(__file__).resolve().parents[1]
STUDY=REPO.parent/'research/pitch_observation_20261028'
WORK=REPO.parent/'research/biomechanics_20261029'
DATA=REPO/'data/metrics';ANALYSIS=REPO/'data/biomechanics/analysis';POSES=REPO/'data/biomechanics/poses'


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,(np.floating,float)):return round(float(value),6) if math.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def atomic_bytes(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);tmp.replace(path)


def write(path,data,compress=False):
    b=(json.dumps(clean(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.gz':atomic_bytes(path,gzip.compress(b,mtime=0))
    else:
        atomic_bytes(path,b)
        if compress:atomic_bytes(path.with_name(path.name+'.gz'),gzip.compress(b,mtime=0))


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def align_fit(fit,source_frames):
    # Missing source frames stay missing; never stretch 323 fitted samples over
    # a 324-frame video or invent a final frame by interpolation.
    old=fit['frame_index'];lookup={int(f):i for i,f in enumerate(old)}
    indices=[lookup.get(int(f)) for f in source_frames]
    for key in ['anatomical_keypoints','foot_keypoints','smpl_rotations','translation','source_time']:
        if key not in fit:continue
        original=fit[key];out=np.full((len(indices),)+original.shape[1:],np.nan)
        for i,j in enumerate(indices):
            if j is not None:out[i]=original[j]
        fit[key]=out
    fit['frame_index']=np.asarray(source_frames)
    return fit


def scene_data(fit,R,origin):
    a=fit['anatomical_keypoints'];translation=fit.get('translation',np.zeros((len(a),3)))
    return np.einsum('ij,tkj->tki',R,a+translation[:,None]-origin)


def calculate_clip(job):
    c,hand,selection,metrics=job;cid=c['id'];original_path=REPO/'data'/c['pose3d'];model=json.load(gzip.open(original_path,'rt'))
    path=STUDY/'combined_population_v1'/selection['fit_path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==model['source_fit_sha256'],('Pose lineage mismatch',cid)
    fit=align_fit(load_npz(path),[f['source_frame'] for f in model['frames']]);R=np.asarray(model['scene']['rotation_camera_to_scene']);origin=np.asarray(model['scene']['origin_camera']);original=scene_data(fit,R,origin)
    published=np.asarray([[p[:3] if p else [np.nan]*3 for p in f['anatomy']] for f in model['frames']]);finite=np.isfinite(published)
    assert np.max(abs(original[finite]-published[finite]))<1e-4,('Published scene mismatch',cid)
    pose=json.loads((REPO/'data'/c['pose']).read_text());observation=json.loads((WORK/'observations'/(cid+'.json')).read_text());masks=support_masks(pose,hand)
    assert len(pose['frames'])==len(original)==len(observation['frameTimes'])
    assert np.allclose(fit['source_time'][np.isfinite(fit['source_time'])],np.asarray(observation['frameTimes'])[np.isfinite(fit['source_time'])],atol=2e-5),('Timestamp lineage mismatch',cid)
    feet_original=np.asarray([[p[:3] if p else [np.nan]*3 for p in f['feet']] for f in model['frames']])
    series_original,cal=make_series(original,fit['smpl_rotations'],fit['anatomical_keypoints'],R,hand,observation,feet_original)
    a=original;rot=fit['smpl_rotations'];camera=fit['anatomical_keypoints'];feet=feet_original;analysis_pose=c['pose3d'];model_name='canonical_native_smpl_adapter';refit_report=None
    refit=WORK/'native_refit'/(cid+'.npz')
    if refit.exists():
        refit_report=json.loads(refit.with_suffix('.json').read_text());assert refit_report['sourceFitSha256']==sha
        refit_report['usedPoseCropOrigin']=True
        # The corrected subject uses full-image pixels; ordinary crops do not.
        import csv as csv_module
        catalog=next(r for r in csv_module.DictReader(open(REPO.parent/'dataset/clips.csv')) if r['dataset_clip_id']==cid)
        old_origin=[float(catalog['crop_x1']),float(catalog['crop_y1'])]
        refit_report['coordinateOriginCorrected']=bool(np.any(np.asarray(refit_report['cropOrigin'])!=old_origin))
        if refit_report['adoptedForAnalysis']:
            newfit=align_fit(load_npz(refit),observation['sourceFrames']);a=scene_data(newfit,R,origin);rot=newfit['smpl_rotations'];camera=newfit['anatomical_keypoints']
            feet=np.einsum('ij,tkj->tki',R,newfit['foot_keypoints']+newfit['translation'][:,None]-origin)
            analysis_pose='biomechanics/poses/'+cid+'.json.gz';variant=dict(model);variant['version']=VERSION;variant['analysisVariant']='native-joint-refit-20261029-v1';variant['source_pose']=c['pose3d'];variant['source_fit_sha256']=sha
            variant['frames']=[dict(f,anatomy=clean(a[i]),feet=clean(feet[i])) for i,f in enumerate(model['frames'])]
            write(REPO/'data'/analysis_pose,variant);model_name='native_joint_reprojection_variant'
    masks['source_pose_available']=np.isfinite(a).all(axis=(1,2))
    series,cal=make_series(a,rot,camera,R,hand,observation,feet)
    alternatives={'pre_native_refit':series_original} if model_name!='canonical_native_smpl_adapter' else {}
    alt_path=STUDY/'all813_gvhmr_v1/base/gvhmr_camera_f12'/(cid+'.npz')
    alt_meta=None
    if alt_path.exists():
        alt=align_fit(load_npz(alt_path),observation['sourceFrames'])
        if len(alt['anatomical_keypoints'])==len(a):
            b=scene_data(alt,R,origin);alt_feet=None
            if 'foot_keypoints' in alt:alt_feet=np.einsum('ij,tkj->tki',R,alt['foot_keypoints']+alt['translation'][:,None]-origin)
            alt_series,_=make_series(b,alt['smpl_rotations'],alt['anatomical_keypoints'],R,hand,observation,alt_feet)
            alternatives['gvhmr_camera_f12']=alt_series;alt_meta=dict(path=str(alt_path.relative_to(STUDY)),sha256=hashlib.sha256(alt_path.read_bytes()).hexdigest(),comparison='model sensitivity, not ground truth')
    axis_variants=[]
    if observation.get('pitchAxisCamera') is not None:
        direction=R@np.asarray(observation['pitchAxisCamera']);angle=observation['axisUncertaintyDegrees']
        for offset in [-angle,angle]:
            trial=dict(observation);trial['pitchAxisCamera']=(R.T@Rotation.from_rotvec(np.array([0,1,0])*np.radians(offset)).apply(direction)).tolist()
            s,_=make_series(a,rot,camera,R,hand,trial,feet);axis_variants.append(s)
    values={m['id']:measurement(m,series,observation,masks,alternatives,axis_variants) for m in metrics}
    view=observation.get('view') or 'unclassified'
    # Unknown or merely approximate views cannot be pooled across clips.
    condition='source-reviewed:'+c['sourceId']+':'+view if observation['inspected'] else 'unclassified:'+cid
    valid_frames=[i for i in observation['cadence']['uniqueImageCandidates'] if i<len(a)]
    geometry=clean({k:[v[i] if masks.get(k,np.ones(len(v),bool))[i] else None for i in range(len(v))] for k,v in series.items()})
    # Native-center source consistency, independently from 3D model agreement.
    legacy=observation['legacyCropOrigin'];crop_origin=observation['cropOrigin'];focal=float(fit['focal']);principal=fit['principal'];world=camera+fit['translation'][:,None];uv=world[...,:2]/world[...,2:]*focal+principal
    source_errors={}
    for label,js,ps in [('lead_leg',[1,4,7] if hand=='R' else [2,5,8],[11,13,15] if hand=='R' else [12,14,16]),('throw_arm',[17,19,21] if hand=='R' else [16,18,20],[6,8,10] if hand=='R' else [5,7,9])]:
        errs=[]
        from biomechanics.calculation import supported
        for i in valid_frames:
            for j,p in zip(js,ps):
                if supported(pose['frames'][i],p):errs.append(np.linalg.norm(uv[i,j]-(np.asarray(pose['frames'][i]['keypoints'][p][:2])+crop_origin)))
        source_errors[label]=dict(medianPixels=float(np.median(errs)) if errs else None,p95Pixels=float(np.quantile(errs,.95)) if errs else None,observations=len(errs),groundTruthAccuracy=False)
    evidence=dict(version=VERSION,clipId=cid,throwingHand=hand,roleMap=dict(leadSide='left' if hand=='R' else 'right',throwingSide='right' if hand=='R' else 'left'),originalHand=c.get('hand'),
        posePath=analysis_pose,originalPosePath=c['pose3d'],source2dPath=c['pose'],sourceUrl=c['sourceUrl'],sourceFitSha256=sha,sourceFitRelativePath=selection['fit_path'],analysisModel=model_name,
        eventCalculationVersion=observation['version'],eventInputSignature=observation.get('inputSignature'),frameTimes=observation['frameTimes'],sourceFrames=observation['sourceFrames'],events=observation['events'],cadence=observation['cadence'],
        coordinate=dict(scene=model['scene'],segment=cal,pitchAxisCamera=observation.get('pitchAxisCamera'),axisUncertaintyDegrees=observation.get('axisUncertaintyDegrees'),axisEvidence='coarse source-image hypothesis; not calibrated field geometry' if observation.get('pitchAxisCamera') else 'unknown'),
        series=geometry,projectedNativeJoints=uv-np.asarray(crop_origin),metrics=values,sourceReview=observation['inspected'],captureCondition=condition,sourceConsistency=source_errors,refit=refit_report,alternative=alt_meta,
        automaticEventProposals=dict(image=observation['imageProposal'],ballTracks=observation['ballTracks'],accepted=False,reason='Development-sample source comparison found early contact / false white-object tracks; candidates require adjudication.'),
        sequenceDiagnostic=sequence_diagnostic(series,observation,masks),
        filtering=dict(eventMidpoint='floor(mean of inclusive bracket endpoints + 0.5); nearest existing independent source image',publishedSpatialValues='unsmoothed native observations; no interpolation',sensitivityGaussianSigmaIndependentImages=[.5,1.],differentiation='central np.gradient only in playback-index diagnostic; no physical velocities published',operationalSensitivityTolerance='15 degrees or 20 interval-percent; not clinical normality'),
        accuracyValidation='not_validated_against_external_GT',copyright='Numerical estimated poses only. Original broadcast RGB remains private; source links provided.')
    write(ANALYSIS/(cid+'.json.gz'),evidence)
    return cid,dict(metrics=values,analysisPath='biomechanics/analysis/'+cid+'.json.gz',analysisPose=analysis_pose,captureCondition=condition,view=view,sourceReview=observation['inspected'],events={k:v['interval'] for k,v in observation['events'].items()},cadence=observation['cadence'],sourceConsistency=source_errors,refit=refit_report,model=model_name)


def aggregate(rows,metrics):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['primaryClipId']:groups[(r['unit'],r['playerCode'],r['pitch'],r['captureCondition'],r['analysisModel'])].append(r)
    result=[]
    for (unit,code,pitch,condition,model),rs in sorted(groups.items(),key=lambda p:str(p[0])):
        g=dict(unit=unit,playerCode=code,player=rs[0]['player'],pitch=pitch,captureCondition=condition,analysisModel=model,calculationVersion=VERSION,rows=len(rs),metrics={})
        for m in metrics:
            cells=[(r,r['metrics'][m['id']]) for r in rs];accepted=[(r,v) for r,v in cells if v['value'] is not None and v['status'] in ['provisional_3d','sensitive_3d']];values=[v['value'] for r,v in accepted];a=np.asarray(values)
            g['metrics'][m['id']]=dict(n=len(a),games=len(set(r['gameId'] for r,v in accepted if r['gameId'])),mean=float(a.mean()) if len(a) else None,std=float(a.std(ddof=1)) if len(a)>1 else None,median=float(np.median(a)) if len(a) else None,q25=float(np.quantile(a,.25)) if len(a)>=3 else None,q75=float(np.quantile(a,.75)) if len(a)>=3 else None,iqr=float(np.ptp(np.quantile(a,[.25,.75]))) if len(a)>=3 else None,statusCounts=dict(collections.Counter(v['status'] for r,v in cells)))
        result.append(g)
    return dict(version=VERSION,grouping=['unit','player','pitch','source/capture condition','analysis model','calculation version'],eligibility='source-reviewed provisional 3D including separately flagged sensitivity; candidate automatic events excluded; not external GT validation',repeatability='Observed between-pitch variability also contains estimation error; not measurement reliability.',groups=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    metrics=dictionary();old=json.load(gzip.open(REPO/'data/releases/metrics-20261028-v1/database.json.gz','rt'));clips=json.loads((REPO/'data/clips.json').read_text())['clips'];profiles={p['code']:p for p in json.loads((REPO/'data/portal/summary.json').read_text())['players']};selection=json.loads((STUDY/'combined_population_v1/selection.json').read_text())['selected']
    if args.limit:clips=clips[:args.limit]
    ANALYSIS.mkdir(parents=True,exist_ok=True);POSES.mkdir(parents=True,exist_ok=True)
    jobs=[(c,profiles[c['playerCode']]['hand'],selection[c['id']],metrics) for c in clips];computed={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(cid,result) in enumerate(pool.map(calculate_clip,jobs)):
            computed[cid]=result
            if (i+1)%100==0:print('Computed',i+1,'/',len(jobs),flush=True)
    if args.limit:print('Pilot computed',len(computed));return
    rows=old['rows'];counts={m['id']:collections.Counter() for m in metrics};stability_counts={m['id']:collections.Counter() for m in metrics}
    for r in rows:
        r['auxiliaryMetrics']={k:v for k,v in (r.get('metrics') or {}).items() if k in {m['id'] for m in old['dictionary'] if m['basis']=='2d'}};cid=r['primaryClipId'];r['metrics']={}
        r.update(calculationVersion=VERSION,timeStatus='no_motion',captureCondition='no_motion',analysisModel=None)
        if cid:
            x=computed[cid];r.update(metrics=x['metrics'],analysisPath=x['analysisPath'],analysisPose=x['analysisPose'],captureCondition=x['captureCondition'],analysisModel=x['model'],view=x['view'],timeStatus=x['cadence']['physicalTimeStatus'],sourceReviewed=x['sourceReview'],events=x['events'])
        for m in metrics:
            v=r['metrics'].get(m['id']);counts[m['id']][v['status'] if v else 'no_motion']+=1
            if v and v['value'] is not None:stability_counts[m['id']][v['stability']]+=1
    # One row per accepted pitch identity; unlinked clips remain separately labelled units.
    linked=[r for r in rows if r['unit']=='pitch_record' and r['primaryClipId']];pitchids=[r['pitchId'] for r in linked];assert len(set(pitchids))==len(pitchids)
    defaults=['lead_knee_flexion_fc','lead_knee_net_extension','thorax_pelvis_axial_fc','elbow_flexion_fc']
    status_codes={
     'provisional_3d':['원본 확인 · 잠정 3D','Source-reviewed provisional 3D'], 'sensitive_3d':['설정·모델 민감 3D','Sensitive 3D'], 'event_candidate':['자동 이벤트 후보 · 미채택','Unaccepted automatic event candidate'],
     'no_motion':['연결 동작 없음','No linked motion'],'event_not_identified':['이벤트 미확인','Event unresolved'],'required_joints_occluded':['필요 관절 가림·낮은 신뢰도','Required joints unsupported'],
     'physical_time_unverified':['실제 시간 변환 미확인','Physical clock unverified'],'pitch_axis_unavailable':['투구 축 미확인','Pitch axis unknown'],'humeral_axial_rotation_unidentifiable':['상완 축회전 식별 불가','Humeral axial rotation unidentifiable'],
     'anatomical_shoulder_frame_unavailable':['해부학적 어깨 좌표계 부재','Anatomical shoulder frame unavailable'],'metric_scale_unverified':['절대 길이 스케일 미확인','Metric length scale unknown'],
     'measured_stature_unavailable':['실측 신장 부재','Measured stature unavailable'],'event_order_ambiguous':['이벤트 순서 불확실','Event order ambiguous'],'extension_transition_unresolved':['신전 전환 미확인','Extension transition unresolved'],
     'source_pose_unavailable':['원본 포즈 부재','Source pose unavailable'],'insufficient_native_samples':['독립 원본 표본 부족','Insufficient native samples'],'irregular_physical_sampling':['실제 표본 간격 불균일','Irregular physical sampling'],'implementation_incomplete':['계산 구현 미완료','Implementation incomplete'],'degenerate_coordinate_geometry':['각도 분해·투영 방향 불안정','Degenerate angle/coordinate geometry']}
    codes={k:dict(zip(['ko','en'],v)) for k,v in status_codes.items()}
    db=dict(schemaVersion=2,version=VERSION,poseVersion=old['poseVersion'],dictionary=metrics,auxiliaryDictionary=[m for m in old['dictionary'] if m['basis']=='2d'],statusCodes=codes,defaults=defaults,
      missingSemantics='null, never zero-imputed. Three separate axes: definition correspondence; calculability/stability; external GT accuracy.',
      primaryClipPolicy=old['primaryClipPolicy'],scope=dict(old['scope'],metricCount=len(metrics),sourceReviewedClips=sum(x['sourceReview'] for x in computed.values())),statusCounts={k:dict(v) for k,v in counts.items()},rows=rows,
      provenance=dict(sources=SOURCES,originalRelease='releases/metrics-20261028-v1/manifest.json',calculation='scripts/build_biomechanics_database.py',sourceReview='metrics/source_reviews_20261029.json',physicalClockPolicy='No physical deg/s or ms without a verified playback-to-real-time mapping. Image cadence diagnostics are not that mapping.'))
    write(DATA/'database.json',db,True);write(DATA/'dictionary.json',dict(version=VERSION,metrics=metrics,sources=SOURCES),True)
    write(DATA/'player_summary.json',aggregate(rows,metrics),True)
    coverage=[]
    for m in metrics:
        c=counts[m['id']];coverage.append(dict(metricId=m['id'],targets=len(clips),values=sum(n for s,n in c.items() if s in ['provisional_3d','sensitive_3d','event_candidate']),sourceReviewed=sum(c.get(s,0) for s in ['provisional_3d','sensitive_3d']),acceptedForDefaultAggregation=c.get('provisional_3d',0)+c.get('sensitive_3d',0),withinTestedSensitivity=c.get('provisional_3d',0),automaticCandidates=c.get('event_candidate',0),missingReasons={s:n for s,n in c.items() if s not in ['no_motion','provisional_3d','sensitive_3d','event_candidate']},missingRequirements=dict(collections.Counter(reason for r in rows if r['primaryClipId'] for reason in r['metrics'][m['id']].get('missingRequirements',[]))),unlinkedRecordRows=c.get('no_motion',0),stability=dict(stability_counts[m['id']])) )
    review_clips=[c for c in clips if computed[c['id']]['sourceReview']]
    cases=[];fc_overlaps=[];br_overlaps=[]
    for c in review_clips:
        cid=c['id'];x=computed[cid];o=json.loads((WORK/'observations'/(cid+'.json')).read_text());proposal=o['imageProposal'];true=x['events']['fc'];auto=proposal.get('fc')
        if true and auto:fc_overlaps.append(max(true[0],auto[0])<=min(true[1],auto[1]))
        tracks=o['ballTracks'];truebr=x['events']['br'];autobr=None
        if tracks:
            f=tracks[0]['frames'];u=o['cadence']['uniqueImageCandidates'];autobr=[u[max(0,u.index(f[0])-1)],f[1]]
        if truebr and autobr:br_overlaps.append(max(truebr[0],autobr[0])<=min(truebr[1],autobr[1]))
        cases.append(dict(clipId=cid,player=c['player'],hand=profiles[c['playerCode']]['hand'],pitch=c['pitch'],sourceUrl=c['sourceUrl'],events=x['events'],automaticContact=auto,automaticBallCandidate=autobr,analysisPath=x['analysisPath'],sourceConsistency=x['sourceConsistency'],refit=x['refit'],metrics={k:x['metrics'][k] for k in defaults},evidenceUrl='biomechanics_detail.html?clip='+cid))
    write(DATA/'coverage.json',dict(version=VERSION,clips=len(clips),metrics=coverage,eventCounts={e:sum(bool(x['events'][e]) for x in computed.values()) for e in ['pkh','fc','br','mer']},sourceReviewed=len(review_clips),timeRecovered=0,nearDuplicateFramePairs=sum(x['cadence']['nearDuplicatePairs'] for x in computed.values()),
        automaticEventDevelopmentCheck=dict(heldOut=False,reference='19 agent source-frame reviews; no human/forceplate GT',contactIntervalOverlap=[sum(fc_overlaps),len(fc_overlaps)],ballCandidateIntervalOverlap=[sum(br_overlaps),len(br_overlaps)],decision='Automatic proposals retained as candidates; not accepted for default aggregation.'),
        duplicatePolicy=dict(acceptedPitchRows=len(linked),duplicateAcceptedPitchIds=0,unresolvedClipRows=old['scope']['unlinkedClips'],unresolvedClipPolicy='Not merged into pitch-record summaries; exact physical-pitch identity not established.'),
        nativeRefit=dict(attempted=sum(x['refit'] is not None for x in computed.values()),adopted=sum(x['model']=='native_joint_reprojection_variant' for x in computed.values()),externalGTValidated=False),cases=cases),True)
    write(DATA/'clip_provenance.json',dict(version=VERSION,clips={k:{a:b for a,b in v.items() if a not in ['metrics','cadence']} for k,v in computed.items()}),True)
    # Full long-form export: metadata columns retain per-cell provenance without ambiguous missing numbers.
    out=io.StringIO();writer=csv.writer(out);header=['row_id','unit','pitch_id','player_code','player','throwing_hand','game_id','date','pitch_type','speed_kmh','primary_clip_id','all_clip_ids','pitch_link_status','database_version','capture_condition','analysis_model']
    for m in metrics:
        for suffix in ['value','unit','status','valid_frames','total_frames','calculation_version','reference_event','source_clip_id','event_source','physical_time_status','used_frames','definition_correspondence','stability','external_accuracy_validation']:header.append(m['id']+'__'+suffix)
    writer.writerow(header)
    for r in rows:
        values=[r['id'],r['unit'],r['pitchId'],r['playerCode'],r['player'],r['hand'],r['gameId'],r['date'],r['pitch'],r['speed'],r['primaryClipId'],'|'.join(r['clipIds']),r['linkStatus'],VERSION,r['captureCondition'],r['analysisModel']]
        for m in metrics:
            v=r['metrics'].get(m['id'],{});values.extend([v.get('value'),m['unit'],v.get('status','no_motion'),v.get('validFrames',0),v.get('totalFrames',0),VERSION,m['referenceEvent'],r['primaryClipId'],json.dumps(v.get('eventSource')),v.get('timeStatus','no_motion'),json.dumps(v.get('usedFrames',[])),m['definitionCorrespondence'],v.get('stability','not_evaluable'),'not_validated_against_external_GT'])
        writer.writerow(["'"+v if isinstance(v,str) and v.startswith(('=','+','@','-')) else v for v in values])
    (DATA/'pitches.csv.gz').write_bytes(gzip.compress(out.getvalue().encode('utf-8-sig'),mtime=0))
    migrations=[]
    mapping={'landing_knee_flexion_2d':'lead_knee_flexion_fc','extra_knee_flexion_2d':'lead_knee_extra_flexion','net_knee_extension_2d':'lead_knee_net_extension','landing_release_time_ms':'fc_br_time_ms','pelvis_trunk_peak_gap_ms':'pelvis_trunk_peak_gap_ms','trunk_forward_lean_release':'trunk_forward_lean_br','trunk_lateral_lean_release':'trunk_lateral_lean_br','normalized_stride_landing':'stride_height_pct'}
    for m in old['dictionary']:
        migrations.append(dict(old=m['id'],new=mapping.get(m['id']),action='auxiliary_quality_only' if m['basis']=='2d' else 'replaced_definition' if m['id'] in mapping else 'retired',reason='2D projections do not replace event-aligned 3D biomechanics.' if m['basis']=='2d' else 'Whole-clip percentiles, unsigned separation, total tilt and ankle spread are not the intended event/segment measurements; v1 preserved losslessly.' if m['id'] not in mapping else 'Explicit event/axis/time/stature requirements; no silent numerical carry-over.'))
    write(DATA/'migration.json',dict(fromVersion=old['version'],toVersion=VERSION,fields=migrations),True)
    print(json.dumps(dict(version=VERSION,clips=len(clips),metrics=len(metrics),rows=len(rows),sourceReviewed=len(review_clips),nativeRefitAdopted=sum(x['model']=='native_joint_reprojection_variant' for x in computed.values())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
