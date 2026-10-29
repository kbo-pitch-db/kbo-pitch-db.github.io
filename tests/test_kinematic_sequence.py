"""Sequence math, release coverage and protection of the verified-clock dataset."""
import csv,gzip,importlib.util,io,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('sequence',ROOT/'scripts/build_kinematic_sequence.py');build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
class SequenceTests(unittest.TestCase):
    def test_native_nonuniform_clock_derivative_and_offset_invariance(self):
        t=np.cumsum(np.resize([.033,.034,.066],40));angle=35*t*t+60*t
        a=build.local_velocity(angle,t,np.ones(40,bool));b=build.local_velocity(angle,t+35,np.ones(40,bool))
        for w in a:
            np.testing.assert_allclose(a[w],70*t+60,atol=1e-8)
            np.testing.assert_allclose(a[w],b[w],atol=1e-8)
    def test_yaw_wrap_and_clock_scaling(self):
        t=np.arange(40)/30;yaw=(150+120*t+180)%360-180
        for v in build.local_velocity(yaw,t,np.ones(40,bool)).values():np.testing.assert_allclose(v,120,atol=1e-8)
        for v in build.local_velocity(yaw,t*2,np.ones(40,bool)).values():np.testing.assert_allclose(v,60,atol=1e-8)
    def test_missing_observations_are_not_upsampled_or_bridged(self):
        t=np.arange(30)/30;mask=np.ones(30,bool);mask[8:20]=False
        for v in build.local_velocity(t*30,t,mask).values():
            self.assertTrue(np.isnan(v[8:20]).all());np.testing.assert_allclose(v[np.isfinite(v)],30)
    def test_event_alignment_does_not_use_clip_length(self):
        np.testing.assert_allclose(build.phase([1.,1.5,2.],1,1.5,2),[-100,0,100])
        np.testing.assert_allclose(build.phase([20.,20.5,21.],20,20.5,21),[-100,0,100])
        self.assertEqual(build.relation([1,1.1],[1.09,1.2],1/30),'overlap')
        self.assertEqual(build.relation([1,1.05],[1.2,1.3],1/30),'before')
    def test_motion_proxy_and_unreviewed_ball_tracks_are_not_release(self):
        u=np.array([0,2,4,6,8,10])
        ev={'br':{'source':'rgb_ball_departure_hypothesis','interval':[4,8]},'fc':{'source':'image_keypoint_hypothesis','interval':[0,2]}}
        points,intervals=build.reviewed_events(ev,u);self.assertIsNone(points['br']);self.assertIsNone(points['fc']);self.assertEqual(intervals,{})
        ev['br']={'source':'agent_visual_source_bracket','interval':[5,8]}
        points,intervals=build.reviewed_events(ev,u);self.assertEqual(points['br'],6);self.assertEqual(intervals['br'],[5,8])
        ev['br']['interval']=[1,1];self.assertIsNone(build.reviewed_events(ev,u)[0]['br'])
    def test_public_coverage_exclusions_and_separate_time_basis(self):
        data=json.loads((ROOT/'data/kinematics/sequences.json').read_text());index=json.loads((ROOT/'data/clips.json').read_text());db=json.loads((ROOT/'data/metrics/database.json').read_text())
        self.assertEqual(data['poseVersion'],index['packageVersion']);self.assertEqual(set(data['clips']),{c['id'] for c in index['clips']})
        self.assertTrue(all(r['metrics'].get('pelvis_trunk_peak_gap_ms',{}).get('value') is None for r in db['rows']))
        for cid in ['8eeef70dce48_c0050','7ef66f7a16b0_c0043']:
            c=data['clips'][cid];self.assertEqual(c['state'],'not_pitch_delivery');self.assertNotIn('peaks',c);self.assertEqual(c['slowMotion'],'not_flagged')
        for c in data['clips'].values():
            if 'events' not in c:continue
            ev=c['events']
            if c['state']=='estimated':
                self.assertTrue(all(ev[k] is not None for k in ['fc','br']))
                if ev['pkh'] is not None:self.assertLess(ev['pkh'],ev['fc'])
                self.assertLess(ev['fc'],ev['br'])
                self.assertEqual(c['eventAcceptance'],'source_reviewed_provisional')
            else:self.assertIsNone(c.get('gapAssumedMs'))
            if c['eventBasis']['br']=='unconfirmed':self.assertIsNone(ev['br'])
            self.assertFalse(c['diagnosticWindow']['releaseProxyUsed'])
            for p in c.get('peaks',{}).values():self.assertLessEqual(p['range'][0],p['time']);self.assertGreaterEqual(p['range'][1],p['time'])
            if c.get('gapAssumedMs') is not None:
                self.assertEqual(c['timeBasis'],'normal_speed_assumed_by_user');self.assertEqual(c['slowMotion'],'not_flagged')
                self.assertFalse(c['peaks']['pelvis']['boundary'] or c['peaks']['trunk']['boundary'])
                self.assertLessEqual(c['gapRangePlaybackMs'][1]-c['gapRangePlaybackMs'][0],160.001)
        text=gzip.decompress((ROOT/'data/kinematics/sequences.json.gz').read_bytes()).decode();self.assertEqual(json.loads(text),data)
        rows=list(csv.DictReader(io.StringIO(gzip.decompress((ROOT/'data/kinematics/sequences.csv.gz').read_bytes()).decode('utf-8-sig'))));self.assertEqual(len(rows),len(index['clips']));self.assertIn('gap_range_min_playback_ms',rows[0])
if __name__=='__main__':unittest.main()
