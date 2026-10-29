"""Review/apply an additive release while preserving every existing pose byte."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

OUT=Path(__file__).resolve().parent
WORKSPACE=OUT.parent.parent
SITE=WORKSPACE/'kbo-pitch-db.github.io'
CANONICAL=WORKSPACE/'dataset'
VERSION='gvhmr-ground-motionbert-20261029-expand-v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def table(path):
    if not path.exists():return [],[]
    with path.open() as stream:
        reader=csv.DictReader(stream);return list(reader),reader.fieldnames

def write_table(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.csv')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)

def merge_table(source,destination):
    added,afields=table(source);old,fields=table(destination)
    fields=list(dict.fromkeys((fields or [])+(afields or [])))
    primary=('source_id' if source.name in ['sources.csv','splits.csv'] else
             'naver_pitch_id' if source.name=='pitches.csv' else
             'dataset_clip_id' if source.name in ['clips.csv','motion_wham.csv',
                 'scoreboard_observations.csv'] else None)
    key=lambda r:r[primary] if primary else tuple(r.get(f,'') for f in fields)
    records={key(row):row for row in old}
    for row in added:
        ident=key(row)
        if ident in records:
            if records[ident]!=row:raise ValueError('Existing dataset row would change: '+str(destination)+':'+str(ident))
        else:records[ident]=row
    write_table(destination,records.values(),fields)

def link_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('Existing dataset asset would change: '+str(target))
        return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)

def main(apply=False):
    protected=json.loads((OUT/'baseline_protection.json').read_text())
    base=protected['index'];stage=OUT/'site_stage/data';addition=json.loads((stage/'clips.json').read_text())
    old_ids={c['id'] for c in base['clips']};clips=addition['clips']
    assert clips and not old_ids&{c['id'] for c in clips}
    assert len({c['id'] for c in clips})==len(clips)
    assert sha(SITE/'data/clips.json')==protected['indexSha256'],'Public index changed while the batch was running'
    for relative,expected in protected['files'].items():
        assert sha(SITE/'data'/relative)==expected,relative
    for c in clips:
        assert c['motionMethod']=='gvhmr-ground-motionbert-v1'
        assert c['playerKnown'] and c['playerCode']
        assert not c.get('pose3dRaw') and not c.get('poseVideo')
        quality=json.loads((stage/c['motionQuality']).read_text())
        assert all(v for k,v in quality['physical_checks'].items() if k!='root_step_diagnostic')
        assert quality['physical_rule']['version']=='combined-population-full-support-qc-v5'
    source_rows,_=table(OUT/'dataset/sources.csv')
    new_pitches,_=table(OUT/'dataset/naver_sports/pitches.csv')
    plan=dict(addedSources=len(source_rows),addedClips=len(clips),newSourceRecords=len(new_pitches),
              finalClips=len(base['clips'])+len(clips),allAddedClipsPassedUnchangedPhysicalRules=True,
              existingPosePayloadsUnchanged=len(protected['files']),version=VERSION)
    save(OUT/'merge_plan.json',plan);print(json.dumps(plan),flush=True)
    if not apply:return
    # The staging dataset is private. Copy only new assets and merge its indexes.
    for source in (OUT/'dataset').rglob('*'):
        if not source.is_file():continue
        relative=source.relative_to(OUT/'dataset');target=CANONICAL/relative
        if relative.suffix=='.csv' and len(relative.parts)<=2:
            merge_table(source,target)
        elif str(relative)=='player_identity.json':
            original=json.loads(target.read_text()) if target.exists() else dict(clips={})
            new=json.loads(source.read_text())
            assert not set(original['clips'])&set(new['clips'])
            original['clips'].update(new['clips']);save(target,original)
        elif not target.exists():link_copy(source,target)
    for source in stage.rglob('*'):
        if source.is_file() and source.relative_to(stage).parts[0] in ['assets','quality','reconstructions']:
            assert source.suffix.lower() in ['.json','.gz','.jpg']
            link_copy(source,SITE/'data'/source.relative_to(stage))
    current={**base,'packageVersion':VERSION,'clips':base['clips']+clips}
    stats=current['packageStats'];total=len(current['clips'])
    for key in ['clipsWith3d','clipsWithGvhmr','clipsWithSceneFit','clipsWithContactFit','reviewRequiredPreviews','clipsWithMotionBert']:
        stats[key]=total
    stats.update(clipsWithWham=0,clipsWithRawWham=0)
    save(SITE/'data/clips.json',current)
    evidence=SITE/'data/player_identity_evidence.json'
    old_evidence=json.loads(evidence.read_text())
    new_evidence=json.loads((OUT/'code/data/player_identity_evidence.json').read_text())
    assert not set(old_evidence['clips'])&set(new_evidence['clips'])
    old_evidence['clips'].update(new_evidence['clips']);save(evidence,old_evidence)
    summary=json.loads((SITE/'data/combined_release_summary.json').read_text())
    reports=json.loads((OUT/'release_records.json').read_text())
    qualities=[json.loads((stage/c['motionQuality']).read_text()) for c in clips]
    previous=summary['release_id']
    for key in ['clips','motionbert_clips','gvhmr_clips','contact_consistency_passed','constant_shoe_shape_clips']:
        summary[key]+=len(clips)
    summary['frames']+=sum(c['frames'] for c in clips)
    summary['valid_3d_frames']+=sum(r['valid_3d_frames'] for r in reports)
    passed=sum(r['strict_source_pass'] for r in reports)
    summary['strict_source_pass']+=passed;summary['strict_source_regression_clips']+=len(clips)-passed
    for key in ['contact_diameter_max_leg','sole_length_variation_max_model','sole_fk_error_max_model']:
        summary[key]=max([summary[key]]+[r[key] for r in reports])
    for key in summary['source_regression_counts']:
        summary['source_regression_counts'][key]+=sum(not r['source_checks'][key] for r in qualities)
    summary.update(release_id=VERSION,previous_release_id=previous,added_clips=len(clips),
                   players=len({c['playerKey'] for c in current['clips']}),
                   baseline_clips_and_pose_payloads_unchanged=True)
    for name in ['combined_release_summary.json','motion_summary.json']:save(SITE/'data'/name,summary)
    registry=SITE/'data/reconstruction_registry.json'
    reg=json.loads(registry.read_text());reg['coverage']['total']=total
    reg['coverage']['legacy_only']=total-reg['coverage']['latest'];save(registry,reg)
    release=SITE/'data/releases/combined-20261029-increment-v1'
    collection=json.loads((WORKSPACE/'videos/remote_2024_increment_20261029/collection_manifest.json').read_text())
    save(release/'sources.json',dict(sources=[dict(providerVideoId=r['provider_video_id'],bytes=r['bytes'],
         sha256=r['sha256'],verified=r['verified'],sourceUrl='https://www.youtube.com/watch?v='+r['provider_video_id']) for r in collection['files']]))
    save(release/'summary.json',plan);save(release/'clips.json',addition)
    save(release/'quality_records.json',reports)
    save(release/'external_pts_validation.json',json.loads((OUT/'external_records/validation.json').read_text()))
    save(release/'record_provenance.json',json.loads((OUT/'dataset/naver_sports/published_csv_provenance.json').read_text()))
    # A portable final-pose index makes current coordinates available locally
    # without replacing the archived WHAM experiment tables.
    numeric=[]
    for c in current['clips']:
        source,clip=c['id'].rsplit('_',1)
        p3=Path('poses/3d_final_gvhmr')/source/(clip+'.json.gz')
        p2=Path('poses/2d_final_halpe26')/source/(clip+'.json')
        link_copy(SITE/'data'/c['pose3d'],CANONICAL/p3)
        link_copy(SITE/'data'/c['poseRefined'],CANONICAL/p2)
        numeric.append(dict(dataset_clip_id=c['id'],method=c['motionMethod'],release_id=c['reconstructionVersion'],
            pose_2d_final=str(p2),pose_3d_final=str(p3),pose_2d_sha256=sha(CANONICAL/p2),
            pose_3d_sha256=sha(CANONICAL/p3),frames=c['frames'],fps=c['fps'],status=c['pose3dStatus']))
    write_table(CANONICAL/'motion_gvhmr_combined.csv',numeric,list(numeric[0]))
    for relative,expected in protected['files'].items():assert sha(SITE/'data'/relative)==expected
    save(OUT/'merge_applied.json',plan)
    print('Merged; derived indexes, event inference, browser QA and deployment remain.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args();main(args.apply)
