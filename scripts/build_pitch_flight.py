"""Publish ball-flight metrics from the cached Naver relay's nine-parameter fits.

Movement: Nathan (2012) gravity/drag-corrected acceleration, y=40 ft to the
front edge of the plate (17/12 ft). Positive HB follows Naver/PITCHf/x +x,
positive IVB is up. VAA is atan2(vz, -vy) at the front edge, negative down.
These are derived PTS values, not provider-reported Statcast measurements.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'naver-pts-flight-20261029-v1'
G = 32.174
START = 40.0
PLATE = 17 / 12
FIELDS = ('x0_ft', 'y0_ft', 'z0_ft', 'vx0_ft_s', 'vy0_ft_s', 'vz0_ft_s',
          'ax_ft_s2', 'ay_ft_s2', 'az_ft_s2')


def time_at_y(y0, vy, ay, target):
    if abs(ay) < 1e-10:
        roots = [(target-y0)/vy] if vy else []
    else:
        discriminant = vy*vy-2*ay*(y0-target)
        if discriminant < 0:
            return None
        roots = [(-vy-math.sqrt(discriminant))/ay, (-vy+math.sqrt(discriminant))/ay]
    roots = [r for r in roots if 0 <= r < 2 and vy+ay*r < 0]
    return min(roots) if roots else None


def calculate(row):
    if row.get('tracking_status') != 'derived_validated_x':
        return dict(hb_cm=None, ivb_cm=None, vaa_deg=None, status='no_tracking')
    try:
        x0,y0,z0,vx,vy,vz,ax,ay,az = [float(row[k]) for k in FIELDS]
        if not all(math.isfinite(v) for v in (x0,y0,z0,vx,vy,vz,ax,ay,az)):
            raise ValueError('nonfinite input')
        if not y0 >= START > PLATE:
            raise ValueError('fit does not span movement interval')
        a,b = time_at_y(y0,vy,ay,START),time_at_y(y0,vy,ay,PLATE)
        if a is None or b is None or not 0 < b-a < 1:
            raise ValueError('invalid flight time')
        # Project the gravity-removed mean acceleration perpendicular to mean
        # velocity; a pure drag trajectory then has zero induced movement.
        dt=b-a
        mean_v=[vx+ax*(a+b)/2,vy+ay*(a+b)/2,vz+az*(a+b)/2]
        norm=math.hypot(*mean_v)
        if norm < 1:
            raise ValueError('invalid mean velocity')
        direction=[v/norm for v in mean_v]
        non_gravity=[ax,ay,az+G]
        projection=sum(x*y for x,y in zip(non_gravity,direction))
        induced=[v-projection*d for v,d in zip(non_gravity,direction)]
        hb=.5*induced[0]*dt*dt*30.48
        ivb=.5*induced[2]*dt*dt*30.48
        vaa=math.degrees(math.atan2(vz+az*b,-(vy+ay*b)))
        return dict(hb_cm=round(hb,5),ivb_cm=round(ivb,5),vaa_deg=round(vaa,5),status='derived_pts')
    except (KeyError,TypeError,ValueError,OverflowError):
        return dict(hb_cm=None,ivb_cm=None,vaa_deg=None,status='invalid_tracking')


def dictionary():
    metrics=[]
    for key,ko,en,unit,definition,formula in [
        ('hb_cm','가로 움직임 (HB)','Horizontal break (HB)','cm','포수 시점의 좌우 움직임입니다. 양수는 오른쪽입니다.','0.5 * a_induced_x * dt^2 * 30.48'),
        ('ivb_cm','수직 움직임 (IVB)','Induced vertical break (IVB)','cm','중력에 의한 낙하를 제외한 수직 움직임입니다. 양수는 위쪽입니다.','0.5 * a_induced_z * dt^2 * 30.48'),
        ('vaa_deg','진입각 (VAA)','Vertical approach angle (VAA)','°','공이 홈플레이트 앞면에 들어올 때의 수직 각도입니다. 음수는 아래로 향합니다.','degrees(atan2(vz0 + az*t_plate, -(vy0 + ay*t_plate)))'),
    ]:
        metrics.append(dict(id=key,name=dict(ko=ko,en=en),unit=unit,basis='pts',group='ball_flight',
            definition=dict(ko=definition,en=en+' derived from the Naver nine-parameter trajectory fit.'),
            formula=formula,calculationVersion=VERSION,referenceEvent='plate_front',
            analysisInterval='y=40 ft to y=17/12 ft; VAA evaluated at y=17/12 ft',
            coordinateSystem='Naver PTS / PITCHf/x feet; x catcher right, y toward pitcher, z up',
            requiredJoints=None,requiredInputs=list(FIELDS),
            sign='HB +x catcher right; IVB positive up; VAA negative downward',
            implementation='scripts/build_pitch_flight.py',definitionCorrespondence='trajectory_derived',
            externalAccuracyValidation='not_independently_calibrated',
            differenceFromSource='Derived from Naver PTS fits using Nathan mean-velocity drag correction. Not a direct Naver or Statcast reported movement field. Fixed 40 ft reference, not release-to-plate.',
            conditions='Finite, validated PTS fit spanning y=40 ft to the plate front; no imputation.',
            sources=[dict(title='Alan M. Nathan — Determining Movement (2012)',url='https://baseball.physics.illinois.edu/Movement.pdf',access='public'),dict(title='Naver Sports relay PTS',url='https://m.sports.naver.com/baseball/kbo/schedule/index',access='public game relay')]))
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT.parent/'dataset')
    args=parser.parse_args()
    source=args.dataset/'naver_sports/pitches.csv'
    records=list(csv.DictReader(source.open()))
    rows={r['naver_pitch_id']:calculate(r) for r in records}
    if len(rows)!=len(records):
        raise ValueError('duplicate pitch IDs')
    status=Counter(r['status'] for r in rows.values())
    data=dict(schemaVersion=1,version=VERSION,sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        recordCount=len(rows),statusCounts=dict(status),dictionary=dictionary(),
        reference=dict(startYFt=START,plateYFt=PLATE,gravityFtS2=G,method='Nathan 2012 mean-velocity gravity/drag correction',movementUnit='cm',vaaUnit='degree'),rows=rows)
    base=ROOT/'data/portal/pitch_flight'
    encoded=(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    base.with_suffix('.json').write_bytes(encoded)
    base.with_suffix('.json.gz').write_bytes(gzip.compress(encoded,mtime=0))
    out=io.StringIO();writer=csv.writer(out);writer.writerow(['pitch_id','player_code','player','game_id','date','pitch','speed_kmh','hb_cm','ivb_cm','vaa_deg','status','source_url','calculation_version',*FIELDS])
    for r in records:
        v=rows[r['naver_pitch_id']]
        writer.writerow([r['naver_pitch_id'],r['pitcher_code'],r['pitcher_name'],r['naver_game_id'],r['game_date'],r['pitch_type_raw'],r['speed_kmh'],v['hb_cm'],v['ivb_cm'],v['vaa_deg'],v['status'],r['source_url'],VERSION,*[r[k] for k in FIELDS]])
    base.with_suffix('.csv.gz').write_bytes(gzip.compress(out.getvalue().encode('utf-8-sig'),mtime=0))
    print(json.dumps(dict(records=len(rows),status=status,bytes=len(encoded))))
    valid=[v for v in rows.values() if v['status']=='derived_pts']
    print({k:[min(v[k] for v in valid),max(v[k] for v in valid)] for k in ['hb_cm','ivb_cm','vaa_deg']})


if __name__=='__main__':
    main()
