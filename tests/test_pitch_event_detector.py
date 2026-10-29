"""Visible-ball evidence, symmetric foot features and independent evaluation contracts."""
import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from pitch_events.detector import release_tracks,associate_feet
from pitch_events.contact_model import features,decode_contact
class EventDetectorTests(unittest.TestCase):
    def fixtures(self):
        u=np.arange(0,22,2);t=np.arange(22)/60;xy=np.zeros((22,26,2));xy[:,10]=[30,20];score=np.ones((22,26));detections={str(i):[[30+i*5-3,17,30+i*5+3,23,.9,0]] for i in u};return u,t,xy,score,detections
    def test_release_requires_detected_ball_not_wrist_speed(self):
        u,t,xy,score,boxes=self.fixtures();xy[:,10,0]=np.arange(22)*20
        self.assertIsNone(release_tracks({},xy,score,u,t,100,'R')[0])
    def test_ball_departure_tracks_native_images(self):
        u,t,xy,score,boxes=self.fixtures();event,tracks=release_tracks(boxes,xy,score,u,t,100,'R');self.assertIsNotNone(event);self.assertTrue(all(v['frame'] in u for v in event['track']));self.assertLessEqual(event['interval'][0],2);self.assertGreaterEqual(event['interval'][1],2)
    def test_stationary_ball_is_not_release(self):
        u,t,xy,score,boxes=self.fixtures();boxes={str(i):[[27,17,33,23,.9,0]] for i in u};self.assertIsNone(release_tracks(boxes,xy,score,u,t,100,'R')[0])
    def test_shoe_feature_order_is_invariant(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(10,3,384));np.testing.assert_array_equal(features(x),features(x[:,[1,0,2]]))
    def test_contact_needs_air_to_ground_transition(self):
        u=np.arange(30);t=u/30
        for p in [.1,.5,.9]:self.assertIsNone(decode_contact(u,np.full(30,p),t,[25,27])[0])
        p=np.r_[np.full(15,.1),np.full(15,.9)];event,_=decode_contact(u,p,t,[25,27]);self.assertLessEqual(event[0],15);self.assertGreaterEqual(event[1],15)
    def test_publication_keeps_automatic_events_out_of_statistics(self):
        data=json.loads((ROOT/'data/kinematics/sequences.json').read_text());registry=json.loads((ROOT/'data/kinematics/events_v3.json').read_text());report=json.loads((ROOT/'data/kinematics/event_validation_v3.json').read_text());model=(ROOT/'data/kinematics/models/foot_contact_linear.json').read_bytes()
        index=json.loads((ROOT/'data/clips.json').read_text());self.assertEqual(set(data['clips']),{c['id'] for c in index['clips']});self.assertEqual(set(data['clips']),set(registry['clips']));self.assertEqual(data['eventVersion'],registry['version']);self.assertEqual(hashlib.sha256(model).hexdigest(),report['modelSha256']);self.assertEqual(report['modelSha256'],report['freeze']['modelSha256']);self.assertFalse(report['independentHumanGroundTruth'])
        for path in ['scripts/pitch_events/detector.py','scripts/pitch_events/contact_model.py']:self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),report['freeze']['files'][path])
        training=set(json.loads(model)['trainingClipIds']);test={c['clipId'] for c in report['clips']};self.assertFalse(training&test);self.assertEqual(len(test),12)
        for c in data['clips'].values():
            if c['state']=='automatic_preview':self.assertIsNone(c['gapAssumedMs']);self.assertEqual(c['eventAcceptance'],'automatic_preview');self.assertIsNone(c['events']['br'])
        self.assertEqual(report['summary']['br']['new']['detectedN'],10)
if __name__=='__main__':unittest.main()
