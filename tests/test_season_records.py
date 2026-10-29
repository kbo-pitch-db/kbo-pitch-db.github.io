"""Season isolation, immutable motion linkage, compact precision and coverage."""
from collections import Counter
import gzip,hashlib,json
from pathlib import Path
import sys,unittest
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_affinity_data as affinity
from build_season_records import pack_projections


class SeasonRecordsTests(unittest.TestCase):
    def test_static_controls_have_unique_ids(self):
        class Ids(HTMLParser):
            def __init__(self):super().__init__();self.ids=[]
            def handle_starttag(self,tag,attrs):self.ids.extend(v for k,v in attrs if k=='id')
        ids=Ids();ids.feed((ROOT/'index.html').read_text())
        self.assertEqual(len(ids.ids),len(set(ids.ids)))

    def test_all_shards_match_manifest_and_original_records_survive(self):
        index=json.loads((ROOT/'data/seasons/manifest.json').read_text())
        self.assertEqual(index['motionSeasons'],[2024])
        self.assertEqual(index['latestSeason'],2026)
        expected={2024:215691,2025:215788,2026:153820}
        original={}
        for path in (ROOT/'data/portal/players').glob('*.json'):
            player=json.loads(path.read_text())
            for r in player['records']:original[r['id']]={**r,'_code':player['profile']['code']}
        flight=json.loads((ROOT/'data/portal/pitch_flight.json').read_text())['rows']
        preserved=set();total=0
        for year in ['2024','2025','2026']:
            meta=index['seasons'][year];seen=set();counts=Counter();tracked=0
            for shard in meta['shards']:
                raw=(ROOT/'data'/shard['path']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),shard['sha256'])
                payload=json.loads(gzip.decompress(raw));self.assertEqual(payload['season'],int(year))
                for code,rows in payload['players'].items():
                    self.assertIn(code,meta['players'])
                    for r in rows:
                        self.assertEqual(len(r),len(index['columns']))
                        game=meta['games'][r[1]];pid=game+':'+r[0]
                        self.assertTrue(game.startswith(year));self.assertNotIn(pid,seen);seen.add(pid);counts[code]+=1
                        self.assertTrue(-1<=r[2]<len(index['pitchTypes']))
                        self.assertTrue(all(v is None or isinstance(v,int)for v in r[4:11]))
                        tracked+=r[4]is not None
                        if pid in original:
                            old=original[pid];preserved.add(pid)
                            self.assertEqual(old['gameId'],game);self.assertEqual(old['_code'],code)
                            self.assertEqual(old['speed'],r[3])
                            for slot,key in [(4,'hb_cm'),(5,'ivb_cm'),(6,'vaa_deg')]:
                                previous=flight[pid][key]
                                if previous is None:self.assertIsNone(r[slot])
                                else:self.assertLessEqual(abs(previous-r[slot]/1000),.00051)
                            self.assertEqual(year,'2024')
            self.assertEqual(len(seen),meta['recordCount']);self.assertEqual(len(seen),expected[int(year)])
            self.assertEqual(tracked,meta['trackedPitches'])
            self.assertEqual(dict(counts),{code:p['recordCount']for code,p in meta['players'].items()})
            total+=len(seen)
        self.assertEqual(total,585299)
        self.assertEqual(preserved,set(original))

    def test_comparisons_use_separate_years_and_actual_cutoffs(self):
        manifest=json.loads((ROOT/'data/visuals/affinity.json').read_text())
        self.assertEqual(manifest['schemaVersion'],3)
        sources=json.loads((ROOT/'scripts/pts_sources.json').read_text())
        for year,entry in manifest['seasons'].items():
            raw=(ROOT/'data'/entry['path']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['sha256'])
            d=json.loads(gzip.decompress(raw));self.assertEqual(str(d['season']),year)
            self.assertEqual(d['source']['sha256'],sources['seasons'][year]['sha256'])
            self.assertEqual(d['similarity']['projectionEncoding'],'delta-int-1e4')
            self.assertTrue(d['dateStart'].startswith(year));self.assertTrue(d['dateEnd'].startswith(year))
            self.assertTrue(all(p['dateStart'].startswith(year)and p['dateEnd'].startswith(year)for p in d['players']))
            self.assertNotIn('groups',d)
        self.assertEqual(manifest['seasons']['2026']['dateEnd'],'20260804')

    def test_delta_quantization_bound_and_no_year_pooling(self):
        source={'players':[{'types':[{'projections':[[-1.234567,-.123456,.123456,2.345678]]}]}],'similarity':{}}
        original=source['players'][0]['types'][0]['projections'][0][:]
        packed=pack_projections(source)['players'][0]['types'][0]['projections'][0]
        value=0
        for expected,delta in zip(original,packed):
            value+=delta;self.assertLessEqual(abs(value/10000-expected),.000051)
        with self.assertRaises(ValueError):affinity.aggregate([{'game_id':'20250322HHLG02025','pitchId':'x'}])


if __name__=='__main__':unittest.main()
