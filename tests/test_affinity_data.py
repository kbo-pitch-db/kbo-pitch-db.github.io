"""Repertoire distributions keep true counts, physics and missingness."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_affinity_data import aggregate,observation,directions,TYPE_IDS


def pitch(identity,stuff='4-Seam',speed=140):
    return dict(game_id='20240401HHLG02024',pitchId=identity,pitcher='200',picher_name='투수',
                stuff=stuff,speed=str(speed),x='0',crossPlateY='.7083',
                x0='0',y0='50',z0='6',vx0='0',vy0='-130',vz0='0',
                ax='0',ay='0',az='-32.174')


class AffinityDataTests(unittest.TestCase):
    def test_real_pitch_types_and_distributions_without_outcome_inputs(self):
        rows=[pitch('1',speed=130),pitch('2',speed=150),pitch('3','Slider',120)]
        data=aggregate(rows,minimum=1)
        p=data['players'][0]
        self.assertEqual(p['counts'][:4],[2,0,0,1])
        self.assertEqual(p['pitches'],3)
        self.assertEqual(p['types'][0]['histograms'][0],[[130,1],[150,1]])
        self.assertEqual(p['types'][0]['quantiles'][0][2],140)
        self.assertEqual(p['types'][0]['histograms'][1],[[0,2]])
        self.assertIsNone(p['types'][1])
        # The source has no result or batter columns: they are not features.
        changed=copy.deepcopy(rows)
        for r in changed:r.update(result='anything',stance='anything')
        self.assertEqual(aggregate(changed,minimum=1)['players'],data['players'])
        self.assertEqual(aggregate(rows,minimum=4)['players'],[])
        self.assertEqual(set(data)&{'groups','outcomeOrder','outcomeCounts'},set())

    def test_movement_sign_and_tracking_gate(self):
        row=pitch('1'); values,error=observation(row)
        self.assertIsNone(error);self.assertEqual(values,[140,0,0])
        row.update(ax='10')
        duration=(50-.7083)/130;row['x']=str(.5*10*duration*duration)
        values,error=observation(row);self.assertIsNone(error);self.assertGreater(values[1],0)
        row['ax']='-10';row['x']=str(-.5*10*duration*duration)
        reflected,error=observation(row);self.assertIsNone(error)
        self.assertAlmostEqual(values[1],-reflected[1]);self.assertEqual(values[2],reflected[2])
        for changes in [dict(x='10'),dict(ax='nan'),dict(vy0='130'),dict(speed=''),dict(y0='30')]:
            invalid=pitch('1');invalid.update(changes)
            self.assertIsNone(observation(invalid)[0])

    def test_duplicate_unknown_and_missing_records(self):
        with self.assertRaises(ValueError):aggregate([pitch('1'),pitch('1')],minimum=1)
        bad=pitch('2');bad['speed']=''
        data=aggregate([pitch('1'),bad],minimum=1,minimum_coverage=.5)
        p=data['players'][0]
        self.assertEqual(p['observedPitches'],2);self.assertEqual(p['pitches'],1)
        self.assertEqual(sum(data['excluded'].values()),1)
        self.assertEqual(aggregate([pitch('1'),bad],minimum=1)['players'],[])
        unknown=pitch('2','Unlabelled')
        data=aggregate([pitch('1'),unknown],minimum=1,minimum_coverage=.5)
        self.assertEqual(data['excluded'],{'unknown_pitch_type':1})
        self.assertEqual(data['players'][0]['pitches'],1)

    def test_published_count_projection_and_histogram_contracts(self):
        data=json.loads((ROOT/'data/visuals/affinity.json').read_text())
        self.assertEqual(data['sourcePitches'],data['usablePitches']+sum(data['excluded'].values()))
        self.assertEqual(data['sourcePitches'],215525)
        self.assertEqual(data['games'],708)
        self.assertEqual(data['eligiblePitches'],sum(p['pitches']for p in data['players']))
        self.assertEqual(data['source']['sha256'],'b4fc2497f579cc60c714b54c1fe91e9046c905cccc4dfae6eae6f85a2a6d21f4')
        self.assertEqual(len(directions()),13)
        for d in directions():self.assertAlmostEqual(sum(v*v for v in d),1)
        self.assertEqual(len({p['key']for p in data['players']}),len(data['players']))
        for p in data['players']:
            self.assertEqual(p['pitches'],sum(p['counts']))
            self.assertGreaterEqual(p['pitches'],500)
            self.assertGreaterEqual(p['pitches']/p['observedPitches'],.9)
            self.assertEqual(p['key'],p['code'])
            self.assertTrue(data['dateStart']<=p['dateStart']<=p['dateEnd']<=data['dateEnd'])
            for i,t in enumerate(p['types']):
                if t is None:self.assertEqual(p['counts'][i],0);continue
                self.assertEqual(t['n'],p['counts'][i])
                for h in t['histograms']:self.assertEqual(sum(n for _,n in h),t['n'])
                for projection in t['projections']:
                    self.assertEqual(len(projection),data['similarity']['quantileCount'])
                    self.assertEqual(projection,sorted(projection))
                    self.assertTrue(all(math.isfinite(v)for v in projection))


if __name__=='__main__':unittest.main()
