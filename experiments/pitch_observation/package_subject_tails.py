"""Version corrected tail observations; preserve good frames and every old asset."""
from common import *
from combined_population import ROOT,SITE
from package_motion import export_pose
from build_site_data import render_pose_media
import shutil,copy,gzip
TAIL=ROOT/'tail_recovery_v1'

def main():
    records=[];registry=json.loads((SITE/'data/subject_corrections.json').read_text())
    ready=list((TAIL/'combined/ready').glob('*.json'));assert len(ready)==8
    for p in ready:
        item=json.loads(p.read_text());cid=item['id'];rec=item['record'];folder=TAIL/'inputs'/cid
        prov=json.loads((folder/'provenance.json').read_text());old=json.loads((folder/'old_clip.json').read_text());c=rows('clips.csv')[cid]
        asset=f'assets/{cid}/subject-tail-v1';dest=SITE/'data'/asset;dest.mkdir(parents=True,exist_ok=True)
        outputs=[]
        for kind,filename,original in [('raw','pose_halpe26.json',old['pose']),('refined','pose_halpe26_refined.json',old['poseRefined'])]:
            two=export_pose(folder/f'{kind}.npz',c,2,old['fps']);previous=json.loads((SITE/'data'/original).read_text());start=prov['recovery_start']
            assert len(two['frames'])==len(previous['frames'])
            assert two['frames'][:start]==previous['frames'][:start],(cid,kind,'good original public frames changed')
            for a,b in zip(two['frames'],previous['frames']):
                assert [a[k] for k in ['frame','source_frame','time','source_time']]==[b[k] for k in ['frame','source_frame','time','source_time']]
            two.update(subject='pitcher',subject_correction='subject-tail-v1',recovery_start_frame=start,
                crop_box=[int(c['crop_x1']),int(c['crop_y1']),int(c['crop_x2']),int(c['crop_y2'])],
                source_image_size=[1920,1080])
            (dest/filename).write_text(json.dumps(clean(two),ensure_ascii=False,separators=(',',':'),allow_nan=False));outputs.append(two)
        render_pose_media(outputs[1],dest,'pose.mp4')
        report=dict(id=cid,version='subject-tail-v1',cause=prov['cause'],old_clip=old,
            original_2d_sha256=sha(SITE/'data'/old['poseRefined']),original_raw_2d_sha256=sha(SITE/'data'/old['pose']),
            original_3d_sha256=sha(SITE/'data'/old['pose3d']),new_2d=f'{asset}/pose_halpe26_refined.json',
            new_2d_sha256=sha(dest/'pose_halpe26_refined.json'),new_raw_2d=f'{asset}/pose_halpe26.json',new_raw_2d_sha256=sha(dest/'pose_halpe26.json'),
            new_3d=f'reconstructions/combined-v1/{cid}.json.gz',selected_fit_sha256=sha(ROOT/item['fit_path']),
            recovery_start_frame=prov['recovery_start'],frames=len(outputs[1]['frames']),original_good_frames_preserved=True,
            frame_and_timestamp_preserved=True,original_assets_preserved=True,original_batter_shape_or_pose_used=False,
            actual_rgb_features_and_gvhmr_reinferred=True,actual_motionbert_reinferred=True,original_pixels_published=False,
            human_review_complete=False,stride_validated=False,input_provenance=prov)
        path=f'subject_corrections/{cid}-tail-v1.json';save(SITE/'data'/path,report)
        rec.update(public_2d=report['new_2d'],public_2d_sha256=report['new_2d_sha256'],original_public_2d=old['poseRefined'],
            original_public_2d_sha256=report['original_2d_sha256'],subject_correction=path,
            clip_update=dict(pose=report['new_raw_2d'],poseRefined=report['new_2d'],poseThumbnail=f'{asset}/pose_thumbnail.jpg',
                poseVideo='',subjectCorrection=path,poseStatus='available' if all(all(p is not None for p in f['keypoints']) for f in outputs[1]['frames']) else 'missing frames present'))
        item['record']=rec;save(p,item);records.append(rec)
        registry['clips'][cid]=dict(manifest=path,old_2d=old['poseRefined'],new_2d=report['new_2d'],old_3d=old['pose3d'],new_3d=report['new_3d'])
    save(TAIL/'subject_corrections.json',registry);save(TAIL/'publication_ready.json',dict(records=records,clips=8,original_files_overwritten=False))
    print('VERSIONED TAIL OBSERVATIONS READY',len(records),flush=True)

if __name__=='__main__':main()
