"""Publish compact, independent seasonal PTS records; preserve all motion assets."""
import argparse
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math
from pathlib import Path

import build_affinity_data as affinity
from build_affinity_data import pack_projections
from build_portal_data import pitch_statistics
from import_published_pts import RESULTS

ROOT=Path(__file__).resolve().parents[1]
VERSION='kbo-pts-seasons-20261030-v1'
BUCKETS=16
COLS=['nativePitchId','gameIndex','pitchTypeIndex','speedKmh','hbMilliCm','ivbMilliCm','vaaMilliDeg','plateXTenThousandthFt','plateZTenThousandthFt','zoneBottomTenThousandthFt','zoneTopTenThousandthFt','result','batterHand','inning','atBatPitch','legacySupplement']


def integer(value,scale=1):
    try:
        x=float(value)
        return round(x*scale)if math.isfinite(x)else None
    except(TypeError,ValueError):return None


def packed(record,index):
    return [record['id'].split(':',1)[1],index,record['type'],record['speed'],
        *[integer(record[k],1000)for k in ['hb','ivb','vaa']],
        *[integer(record[k],10000)for k in ['plateX','plateZ','zoneBottom','zoneTop']],
        record['result'],record.get('batterHand'),record.get('inning'),record.get('atBatPitch'),record.get('legacy',0)]


def write_gzip(path,data):
    raw=(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True);encoded=gzip.compress(raw,compresslevel=9,mtime=0);path.write_bytes(encoded)
    return dict(path=str(path.relative_to(ROOT/'data')),sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded))


def canonical():
    records={}
    flight=json.loads((ROOT/'data/portal/pitch_flight.json').read_text())['rows']
    for path in (ROOT/'data/portal/players').glob('*.json'):
        p=json.loads(path.read_text());code=p['profile']['code']
        for r in p['records']:
            f=flight.get(r['id'],{})
            records[r['id']]=dict(id=r['id'],code=code,name=p['profile']['name'],gameId=r['gameId'],date=r['date'],
                type=next((i for i,t in enumerate(affinity.TYPE_INFO)if t[1]==r['pitch']),-1),pitch=r['pitch'],speed=r['speed'],
                hb=f.get('hb_cm'),ivb=f.get('ivb_cm'),vaa=f.get('vaa_deg'),plateX=r['plateX'],plateZ=r['plateZ'],
                zoneBottom=r['zoneBottom'],zoneTop=r['zoneTop'],result=r['result'],batterHand=r.get('batterHand'),
                inning=r.get('inning'),atBatPitch=r.get('atBatPitch'),legacy=1)
    return records


def build(source_paths,output=ROOT/'data/seasons'):
    config=json.loads((ROOT/'scripts/pts_sources.json').read_text());old=canonical();manifest=dict(schemaVersion=1,version=VERSION,
        latestSeason=max(source_paths),motionSeasons=[2024],columns=COLS,buckets=BUCKETS,
        pitchTypes=[dict(id=i,name=dict(ko=ko,en=en),color=color)for i,ko,en,color in affinity.TYPE_INFO],seasons={})
    index_affinity=dict(schemaVersion=3,version=VERSION,latestSeason=max(source_paths),seasons={})
    for year,path in sorted(source_paths.items()):
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();spec=config['seasons'][str(year)]
        if sha!=spec['sha256']:raise ValueError('Source SHA256 mismatch: '+str(year))
        source_url=f"https://github.com/{config['repository']}/blob/{config['commit']}/data/{year}.csv"
        with path.open(encoding='cp949',newline='')as stream:rows=list(csv.DictReader(stream))
        data=affinity.aggregate(rows,season=year)
        data['source'].update(url=source_url,sha256=sha)
        index_affinity['seasons'][str(year)]=dict(season=year,dateStart=data['dateStart'],dateEnd=data['dateEnd'],sourcePitches=data['sourcePitches'],pitchers=len(data['players']),**write_gzip(ROOT/f'data/visuals/affinity-{year}.json.gz',pack_projections(data)))
        records={};names={}
        for r in rows:
            game=r['game_id'];pid=game+':'+r['pitchId']
            if pid in records:raise ValueError('Duplicate pitch')
            values,error=affinity.observation(r,with_vaa=True)
            if r['result']not in RESULTS:raise ValueError('Unknown result')
            code=r['pitcher'];name=r['picher_name']
            if code in names and names[code]!=name:raise ValueError('Conflicting pitcher name')
            names[code]=name
            records[pid]=dict(id=pid,code=code,name=name,gameId=game,date=f'{game[:4]}-{game[4:6]}-{game[6:8]}',
                type=affinity.TYPE_IDS.index(r['stuff'])if r['stuff']in affinity.TYPE_IDS else -1,
                pitch=next((t[1]for t in affinity.TYPE_INFO if t[0]==r['stuff']),r['stuff']),speed=float(r['speed'])if r['speed']else None,
                hb=values[1]if values else None,ivb=values[2]if values else None,vaa=values[3]if values else None,
                plateX=float(r['x'])if not error else None,plateZ=float(r['y'])if not error else None,
                zoneBottom=float(r['bottomSz'])if not error else None,zoneTop=float(r['topSz'])if not error else None,
                result=RESULTS[r['result']],batterHand=r['stance']if r['stance']in['L','R']else None,
                inning=integer(r['inning']),atBatPitch=integer(r['pitch_num']),legacy=0)
        additions=0
        for key,r in old.items():
            if r['date'][:4]==str(year)and key not in records:records[key]=r;names[r['code']]=r['name'];additions+=1
        games=sorted({r['gameId']for r in records.values()});game_indices={g:i for i,g in enumerate(games)}
        by_player=defaultdict(list)
        for r in records.values():by_player[r['code']].append(r)
        buckets=[{}for _ in range(BUCKETS)];profiles={}
        for code,rs in sorted(by_player.items()):
            rs.sort(key=lambda r:(r['gameId'],r['id']))
            profiles[code]=dict(code=code,name=names[code],season=year,recordCount=len(rs),recordGames=len({r['gameId']for r in rs}),recordStart=min(r['date']for r in rs),recordEnd=max(r['date']for r in rs),statistics=pitch_statistics(rs))
            buckets[int(code)%BUCKETS][code]=[packed(r,game_indices[r['gameId']])for r in rs]
        meta=dict(season=year,recordCount=len(records),sourcePitches=len(rows),legacySupplement=additions,trackedPitches=sum(r['hb']is not None for r in records.values()),games=games,dateStart=min(r['date']for r in records.values()),dateEnd=max(r['date']for r in records.values()),
            source=dict(url=source_url,sha256=sha,gitBlob=spec['blob'],lastSourceUpdate='2026-08-10'),players=profiles,
            summary=write_gzip(output/f'{year}/profiles.json.gz',dict(schemaVersion=1,season=year,players=profiles)),
            shards=[write_gzip(output/f'{year}/bucket-{i:02d}.json.gz',dict(schemaVersion=1,season=year,players=b))for i,b in enumerate(buckets)])
        # Full statistics live in a compressed summary. The small index gives
        # directory labels/counts without eagerly decoding every pitch.
        meta['players']={code:{k:p[k]for k in['code','name','season','recordCount','recordGames','recordStart','recordEnd']}|{'fastballVelocity':p['statistics']['fastballVelocity'],'archetype':p['statistics']['archetype']}for code,p in profiles.items()}
        manifest['seasons'][str(year)]=meta
        print(json.dumps(dict(season=year,records=len(records),source=len(rows),supplement=additions,players=len(profiles),bytes=sum(p['bytes']for p in meta['shards'])+meta['summary']['bytes'],affinityBytes=index_affinity['seasons'][str(year)]['bytes']),ensure_ascii=False),flush=True)
    output.mkdir(parents=True,exist_ok=True);(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    (ROOT/'data/visuals/affinity.json').write_text(json.dumps(index_affinity,separators=(',',':'))+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',action='append',required=True,help='YEAR=/path/to/CSV');a=p.parse_args()
    build({int(v.split('=',1)[0]):Path(v.split('=',1)[1])for v in a.source})
if __name__=='__main__':main()
