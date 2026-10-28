"""Detect release from learned ball trajectories and landing from stabilized RGB observations."""
import argparse,csv,gzip,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from pitch_events.contact_model import probabilities,decode_contact
from pitch_events.detector import VERSION,release_tracks,knee_peak,landing,associate_feet
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT.parent/'research/event_detector_20261029'
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return round(float(v),6) if np.isfinite(v) else None
    return v

def detect(clip,row):
    cid=clip['id'];p=json.load(open(ROOT/'data'/clip['pose']));a=json.load(gzip.open(ROOT/'data/biomechanics/analysis'/f'{cid}.json.gz'));ball_path=PRIVATE/'ball'/f'{cid}.json'
    if not ball_path.exists():return None
    ball=json.load(open(ball_path));xy=np.array([[v[:2] if v else [np.nan,np.nan] for v in f['keypoints']] for f in p['frames']]);score=np.array([[v[2] if v else 0 for v in f['keypoints']] for f in p['frames']]);times=np.array([f['time'] for f in p['frames']]);u=np.array(a['cadence']['uniqueImageCandidates']);video_origin=np.array([float(row['crop_x1']),float(row['crop_y1'])]);origin=np.array(p.get('crop_box',video_origin.tolist())[:2]);xy+=origin
    study=ROOT.parent/'research/pitch_observation_20261028/all813_gvhmr_v1';paths=[study/'background'/f'{cid}.npz',study/'background_worker0/background'/f'{cid}.npz',study/'background_worker1/background'/f'{cid}.npz'];bg=next((p for p in paths if p.exists()),paths[0]);matrices=None;cuts=[]
    if bg.exists():
        with np.load(bg) as data:
            if len(data['full_transforms'])==len(times):matrices=data['full_transforms'];cuts=data['cuts'].tolist()
    br,tracks=release_tracks(ball['detections'],xy,score,u,times,ball['height'],a['throwingHand'],matrices,video_origin)
    if br and any(br['track'][0]['frame']<=cut<=br['track'][-1]['frame'] for cut in cuts):br=None
    release=br['interval'] if br else None;xy,score,swapped=associate_feet(xy,score,u,times,ball['height']);pkh=knee_peak(xy,score,u,times,ball['height'],a['throwingHand'],release);fc,diagnostic=landing(xy,score,u,times,ball['height'],a['throwingHand'],release,pkh,matrices)
    trajectory_fc=fc;trajectory_diagnostic=diagnostic;model_path=ROOT/'data/kinematics/models/foot_contact_linear.json';model_path=model_path if model_path.exists() else PRIVATE/'contact_model.json';feature_path=PRIVATE/'foot_features'/f'{cid}.npz'
    if model_path.exists() and feature_path.exists():
        model=json.load(open(model_path))
        with np.load(feature_path) as feature:fc,diagnostic=decode_contact(feature['frames'],probabilities(feature['features'],model),times,release)
    else:fc=None;diagnostic={'reason':'visual_contact_model_unavailable'}
    if pkh and fc and pkh[1]>=fc[0]:pkh=None
    return clean(dict(clipId=cid,version=VERSION,poseSha256=hashlib.sha256((ROOT/'data'/clip['pose']).read_bytes()).hexdigest(),modelSha256=ball['modelSha256'],events=dict(pkh=pkh,fc=fc,br=release),eventEvidence=dict(trajectoryContact=trajectory_fc,trajectoryContactDiagnostic=trajectory_diagnostic,footLabelSwapFrames=swapped,br=br,fc=diagnostic,pkh='Observed supported lead-knee height maximum with before/after prominence'),alternativeBallTracks=tracks[:2],cameraStabilized=matrices is not None,sourceUrl=clip['sourceUrl'],fps=p['fps'],uniqueImageCount=len(u),model='BaseballCV ball_trackingv4 YOLO11x',eventAcceptance='automatic_candidate_pending_validation'))

def job(arguments):
    clip,row=arguments;return detect(clip,row)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();ids=set(json.load(open(args.ids))) if args.ids else None
    clips=json.load(open(ROOT/'data/clips.json'))['clips'];rows={r['dataset_clip_id']:r for r in csv.DictReader(open(ROOT.parent/'dataset/clips.csv'))};dest=PRIVATE/'candidates';dest.mkdir(exist_ok=True);results={}
    jobs=[(c,rows[c['id']]) for c in clips if ids is None or c['id'] in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(job,jobs)):
            if result:results[result['clipId']]=result;(dest/f"{result['clipId']}.json").write_text(json.dumps(result,separators=(',',':'))+'\n')
            if (i+1)%100==0:print('Detected',i+1,flush=True)
    print('Candidates',len(results),{k:sum(bool(c['events'][k]) for c in results.values()) for k in ['pkh','fc','br']})
if __name__=='__main__':main()
