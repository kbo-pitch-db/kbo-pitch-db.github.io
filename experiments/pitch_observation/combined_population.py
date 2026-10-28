"""Resumable full-population GVHMR/support/MotionBERT production inputs.

The archived research fits and final 2D are immutable. Every stage writes to
its own production directory; preparation/inference never publish anything.
"""
from common import *
from all813_release import ROOT as GV_ROOT
from independent_support import contact_state
import argparse,copy,traceback

ROOT=OUT/'combined_population_v1'
SITE=Path(os.environ.get('KBO_COMBINED_SITE',str(WORKSPACE/'kbo-pitch-db.github.io')))


def manifest():return json.loads((ROOT/'inputs.json').read_text())


def inputs(rec):
    h=load(DATA/rec['pose_2d_refined'])
    n=rec['valid_frames']
    for key in ['bbox','frame_index','pts','source_time','track_id','keypoints','scores','missing','interpolated','outlier']:
        if key in h:h[key]=h[key][:n].copy()
    return h


def episodes(h,phase,bg):
    # Marker-wise image evidence remains usable when metadata hand/lead-side
    # inference conflicts. This does not invent a ball release or plant time.
    p=copy.deepcopy(phase);p['phase_side_conflict']=False
    return contact_state(h,p,bg)


def prepare():
    ROOT.mkdir(exist_ok=True)
    dest=ROOT/'inputs.json'
    if dest.exists():print('FROZEN INPUTS',len(manifest()['records']));return
    index=json.loads((SITE/'data/clips.json').read_text())
    rows2=rows('motion_wham.csv');records=[];counts={}
    for c in index['clips']:
        cid=c['id'];source=GV_ROOT/'selected'/f'{cid}.npz'
        meta=json.loads(source.with_suffix('.json').read_text())
        if c.get('postprocessingVersion')=='independent-support-v7':
            source=OUT/'independent_support_selected_v7'/f'{cid}.npz'
        assert sha(source)==c['sceneSignature'],cid
        z=load(source);h=load(DATA/rows2[cid]['pose_2d_refined']);n=len(z['frame_index'])
        rec=dict(id=cid,player=c['player'],player_key=c['playerKey'],source=str(source.relative_to(OUT)),
            source_sha256=sha(source),pose_2d_refined=rows2[cid]['pose_2d_refined'],
            pose_2d_sha256=sha(DATA/rows2[cid]['pose_2d_refined']),public_2d_sha256=sha(SITE/'data'/c['poseRefined']),
            source_frames=len(h['frame_index']),valid_frames=n,phase=meta['phase'],
            public_pose=c['pose3d'],public_2d=c['poseRefined'])
        hh={k:(v[:n] if np.ndim(v) and len(v)==len(h['frame_index']) and k not in ['joints','edges'] else v) for k,v in h.items()}
        _,_,notes=episodes(hh,rec['phase'],z['background_transforms'])
        rec['initial_contact_episodes']=notes
        counts['with_image_support']=counts.get('with_image_support',0)+bool(notes)
        records.append(rec)
    save(ROOT/'original_index.json',index)
    save(dest,dict(version='combined-population-v1',records=records,
        canonical_index_sha256=sha(SITE/'data/clips.json'),clips=len(records),
        original_2d_immutable=True,public_source_media_included=False,
        motionbert_checkpoint_sha256=sha(OUT/'temporal_adaptation/best.pt'),
        base_motionbert_checkpoint_sha256=sha(WORKSPACE/'models/motionbert_h36m.bin'),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),contact_counts=counts))
    print('PREPARED',len(records),'CONTACT COVERAGE',counts,flush=True)


def infer(records,worker):
    import torch
    from public_adaptation import setup_model
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30,0)
    model=setup_model();checkpoint=torch.load(OUT/'temporal_adaptation/best.pt',map_location='cpu')
    model.load_state_dict(checkpoint['trainable'],strict=False);model.eval()
    from lib.data.dataset_wild import halpe2h36m
    from lib.utils.utils_data import crop_scale,flip_data
    failures={}
    for number,rec in enumerate(records):
        cid=rec['id'];dest=ROOT/'motionbert'/f'{cid}.npz';meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():continue
        try:
            h=inputs(rec);sc=scores(h);inp=np.concatenate([h['keypoints'].copy(),sc[...,None]],-1)
            for j in range(26):
                good=np.isfinite(inp[:,j,:2]).all(-1)&(sc[:,j]>0)
                for k in range(2):inp[:,j,k]=np.interp(np.arange(len(inp)),np.flatnonzero(good),inp[good,j,k]) if good.any() else 0
            normalized=crop_scale(halpe2h36m(inp).astype(np.float32),[1,1]).astype(np.float32)
            n=len(inp);pred=np.empty((2,n,17,3),np.float32)
            edges=np.flatnonzero((np.diff(h['frame_index'])!=1)|(np.diff(h['track_id'])!=0))+1
            for segment in np.split(np.arange(n),edges):
                # Overlapping context windows; use each window's central
                # predictions without averaging different depth hypotheses.
                for start in range(0,len(segment),121):
                    core=segment[start:start+121];lo=max(0,min(start-61,len(segment)-243));ids=segment[lo:lo+243]
                    x=torch.tensor(normalized[ids][None],device='cuda:0')
                    with torch.inference_mode():outputs=[model(x),flip_data(model(flip_data(x)))]
                    for i,p in enumerate(outputs):
                        a=p[0].cpu().numpy();a-=a[:,0:1];pred[i,core]=a[core-ids[0]]
            assert np.isfinite(pred).all()
            atomic_npz(dest,single=pred[0],flip=pred[1],frame_index=h['frame_index'],source_time=h['source_time'])
            save(meta,dict(id=cid,source_2d_sha256=rec['pose_2d_sha256'],frames=n,
                checkpoint_sha256=manifest()['motionbert_checkpoint_sha256'],
                model='MotionBERT + existing public-mocap last-block adaptation',
                inference='actual checkpoint inference; original and flip-restored hypotheses separate',
                source_time='Broadcast timing; physical slow-motion factor unknown'))
            print('MB',worker,number+1,'/',len(records),cid,flush=True)
        except Exception as e:
            failures[cid]=dict(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        save(ROOT/f'mb_worker{worker}.json',dict(done=number+1,total=len(records),failures=failures))


if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','motionbert'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        rr=manifest()['records'][a.worker::a.workers]
        infer(rr[:a.limit or None],a.worker)
