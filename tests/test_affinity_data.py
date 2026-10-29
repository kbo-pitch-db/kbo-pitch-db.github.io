"""Keep pitch observations separate from unobserved PA/contact outcomes."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_affinity_data import aggregate


def pitch(identity, result='Hit', side='L', hitter='100', pitcher='200'):
    return dict(game_id='20240401HHLG02024', pitchId=identity, result=result,
                stance=side, batter=hitter, batter_name='타자'+hitter,
                pitcher=pitcher, picher_name='투수'+pitcher)


class AffinityDataTests(unittest.TestCase):
    def test_in_play_is_not_hits_and_batting_sides_remain_separate(self):
        rows=[pitch('1'),pitch('2','Ball'),pitch('3','Swing','R'),
              pitch('4','Strike'),pitch('5','Foul'),pitch('6','BuntFoul')]
        data=aggregate(rows,minimum=1)
        hitters={p['key']:p for p in data['groups']['hitters']}
        self.assertEqual(hitters['100-L']['counts'],[1,1,0,1,1])
        self.assertEqual(hitters['100-R']['counts'],[0,0,1,0,0])
        self.assertEqual(data['groups']['pitchers'][0]['counts'],[1,1,1,1,1])
        self.assertEqual(data['sourcePitches'],6)
        self.assertEqual(data['usablePitches'],5)
        self.assertEqual(data['excludedBunts'],{'BuntFoul':1})
        self.assertNotIn('battingAverage',str(data))
        self.assertNotIn('strikeouts',str(data))
        self.assertEqual(aggregate(rows,minimum=5)['groups']['hitters'],[])

    def test_ambiguous_or_duplicate_observations_fail(self):
        for rows in [[pitch('1'),pitch('1')],[pitch('1','Unknown')],
                     [pitch('1',side='?')]]:
            with self.assertRaises(ValueError):aggregate(rows,minimum=1)

    def test_published_denominators_and_source_are_explicit(self):
        data=json.loads((ROOT/'data/visuals/affinity.json').read_text())
        self.assertEqual(data['usablePitches'],sum(data['outcomeCounts']))
        self.assertEqual(data['sourcePitches'],data['usablePitches']+sum(data['excludedBunts'].values()))
        self.assertEqual(data['sourcePitches'],215525)
        self.assertEqual(data['games'],708)
        self.assertEqual(data['source']['sha256'],'b4fc2497f579cc60c714b54c1fe91e9046c905cccc4dfae6eae6f85a2a6d21f4')
        for role,players in data['groups'].items():
            self.assertEqual(len({p['key'] for p in players}),len(players))
            for p in players:
                self.assertEqual(sum(p['counts']),p['pitches'])
                self.assertGreaterEqual(p['pitches'],500)
                self.assertEqual(len(p['counts']),5)
                self.assertTrue(all(isinstance(n,int)and n>=0 for n in p['counts']))
                self.assertTrue(data['dateStart']<=p['dateStart']<=p['dateEnd']<=data['dateEnd'])
                self.assertEqual(p['key'],p['code']+('-'+p['side']if role=='hitters'else''))


if __name__=='__main__':unittest.main()
