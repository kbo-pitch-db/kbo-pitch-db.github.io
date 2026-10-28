"""Small reproducible OpenBiomechanics definition comparison; never broadcast GT.

Obtain release ZIPs using the upstream scripts/release_checksums.sha256.
Only derived error summaries are published (CC BY-NC-SA attribution retained).
"""
import argparse,csv,gzip,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from .geometry import flexion

REPO=Path(__file__).resolve().parents[2]


def first_pitches(path,limit=12,ids=None):
    out={}
    with zipfile.ZipFile(path) as z:
        rows=csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0])))
        for row in rows:
            key=row['session_pitch']
            if key not in out:
                if ids is not None and key not in ids:
                    if len(out)==len(ids):break
                    continue
                if ids is None and len(out)>=limit:break
                out[key]=[]
            out[key].append(row)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',default='/tmp/kbo-obp-reference');p.add_argument('--limit',type=int,default=12);args=p.parse_args();root=Path(args.input_dir)
    angles=first_pitches(root/'pitching_joint_angles.zip',args.limit);ids=set(angles)
    landmarks=first_pitches(root/'pitching_landmarks.zip',ids=ids);velos=first_pitches(root/'pitching_joint_velos.zip',ids=ids)
    poi={r['session_pitch']:r for r in csv.DictReader(open(REPO.parent/'external/openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv'))};results=[]
    for key,rows in angles.items():
        t=np.asarray([float(r['time']) for r in rows]);a=np.asarray([float(r['lead_knee_angle_x']) for r in rows]);r=rows[0];fc10=float(r['fp_10_time']);fc100=float(r['fp_100_time']);br=float(r['BR_time']);p=poi[key]
        nearest=lambda time:int(np.argmin(abs(t-time)));i,j,k=map(nearest,[fc10,fc100,br]);net10=a[i]-a[k];net100=a[j]-a[k];published=float(p['lead_knee_extension_from_fp_to_br'])
        lm=landmarks[key];lt=np.asarray([float(r['time']) for r in lm]);pts=lambda name:np.asarray([[float(r[name+'_'+axis]) for axis in 'xyz'] for r in lm]);geometric=flexion(pts('lead_hip'),pts('lead_knee_jc'),pts('lead_ankle_jc'))
        mask=(lt>=fc10)&(lt<=br);ref=np.interp(lt[mask],t,a);difference=geometric[mask]-ref
        vr=velos[key];vt=np.asarray([float(r['time']) for r in vr]);vv=np.asarray([float(r['lead_knee_velo_x'] or 'nan') for r in vr]);window=(vt>=fc10)&(vt<=br);knee_peak=float(np.nanmax(vv[window]));published_peak=float(p['lead_knee_extension_angular_velo_max'])
        pv=np.asarray([float(r['pelvis_velo_z'] or 'nan') for r in vr]);tv=np.asarray([float(r['torso_velo_z'] or 'nan') for r in vr]);wide=(vt>=float(r['pkh_time']))&(vt<=br);indices=np.where(wide)[0];gap=float(vt[indices[np.nanargmax(tv[wide])]]-vt[indices[np.nanargmax(pv[wide])]])
        result=dict(sessionPitch=key,sourceContactTimes=dict(fp10=fc10,fp100=fc100,br=br),kneeGeometryVersusPublishedAngle=dict(rmseDegrees=float(np.sqrt(np.mean(difference**2))),maxAbsoluteDegrees=float(np.max(abs(difference)))),netExtension=dict(recomputedFC10=net10,recomputedFP100=net100,publishedPOI=published,errorFC10=net10-published,errorFP100=net100-published),kneePeakExtensionVelocity=dict(recomputedFC10BR=knee_peak,publishedPOI=published_peak,difference=knee_peak-published_peak),pelvisTrunkPeakGap=dict(recomputedPKHBRSeconds=gap,publishedPOISeconds=float(p['timing_peak_torso_to_peak_pelvis_rot_velo'])))
        results.append(result)
    summary=dict(kind='definition_and_implementation_comparison',broadcastAccuracyValidation=False,source='https://github.com/drivelineresearch/openbiomechanics',license='Derived summaries from OpenBiomechanics; CC BY-NC-SA 4.0; attribution Driveline Baseball. Source ZIPs not redistributed.',
        sourceHashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.zip')},pitches=len(results),results=results,
        findings=['Three-point knee flexion compared to published laboratory knee channel; differences are retained, not forced to zero.','Both force-event thresholds are compared; POI wording alone does not establish whether 10% or 100% BW was used.','These are laboratory signal/definition checks, not a validation of reconstructed broadcast poses.'])
    (REPO/'data/metrics/reference_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(pitches=len(results),medianKneeRMSE=np.median([r['kneeGeometryVersusPublishedAngle']['rmseDegrees'] for r in results]),maxFC10NetError=max(abs(r['netExtension']['errorFC10']) for r in results),maxFP100NetError=max(abs(r['netExtension']['errorFP100']) for r in results))),flush=True)


if __name__=='__main__':main()
