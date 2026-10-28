"""Scientific definition checks and release invariants, not UI snapshots."""
import gzip,hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from biomechanics.geometry import flexion,wrap,pitch_frame,euler_zyx,anatomical_frame,signed_horizontal,global_rotations,PARENTS,plane_inclinations
from biomechanics.calculation import evaluate,measurement
from biomechanics.dictionary import dictionary,VERSION

class GeometryTests(unittest.TestCase):
 def test_flexion_zero_and_known_angle(self):
  self.assertAlmostEqual(float(flexion(np.array([0,1,0]),np.zeros(3),np.array([0,-1,0]))),0)
  self.assertAlmostEqual(float(flexion(np.array([1,0,0]),np.zeros(3),np.array([0,1,0]))),90)
 def test_right_handed_pitch_frame_and_coordinate_invariance(self):
  B=pitch_frame([0,0,1]);np.testing.assert_allclose(B.T@B,np.eye(3));self.assertAlmostEqual(np.linalg.det(B),1)
  # World rotation must not change the local decomposition.
  W=Rotation.from_euler('xyz',[25,-72,43],degrees=True).as_matrix();Q=Rotation.from_euler('ZYX',[35,20,-15],degrees=True).as_matrix()
  np.testing.assert_allclose(euler_zyx((W@B).T@(W@B@Q)),[35,20,-15],atol=1e-9)
 def test_wrapping_and_signed_mirror(self):
  self.assertEqual(wrap(181),-179);np.testing.assert_allclose(np.degrees(np.unwrap(np.radians([179,-179,-177]))),[179,181,183])
  a=np.array([[1.,0,0]]);b=np.array([[0.,0,1]]);self.assertAlmostEqual(float(signed_horizontal(a,b)[0]), -90)
  reflected=np.array([-1,1,1]);self.assertAlmostEqual(float(signed_horizontal(a*reflected,b*reflected)[0]),90)
 def test_forward_and_lateral_are_distinct_planes(self):
  B=np.eye(3);q=Rotation.from_euler('ZYX',[90,30,0],degrees=True).as_matrix();f,l=plane_inclinations(q[:,2],B,'R');self.assertAlmostEqual(float(f),0,places=9);self.assertAlmostEqual(float(l),30)
  f,l=plane_inclinations(q[:,2],B,'L');self.assertAlmostEqual(float(l),-30)
 def test_gimbal_yaw_is_not_fabricated(self):
  q=Rotation.from_euler('ZYX',[40,90,20],degrees=True).as_matrix();e=euler_zyx(q);self.assertTrue(np.isnan(e[0]));self.assertAlmostEqual(e[1],90)
 def test_segment_frame_orthonormal(self):
  A=anatomical_frame(np.array([.2,1.,.1]),np.array([0.,.1,1.]));np.testing.assert_allclose(A.T@A,np.eye(3),atol=1e-14);self.assertAlmostEqual(np.linalg.det(A),1)
 def test_humeral_twist_is_not_identified_by_centers(self):
  # A local humerus twist, compensated at the elbow, leaves every observed
  # shoulder/elbow/wrist center identical. Thus center coordinates do not
  # identify anatomical external rotation, even if a prior supplies rotations.
  shoulder=np.zeros(3);elbow=np.array([0.,-1.,0.]);forearm=np.array([.4,-.8,.2]);axis=elbow/np.linalg.norm(elbow)
  twist=Rotation.from_rotvec(axis*np.radians(75)).as_matrix();comp=twist.T
  np.testing.assert_allclose(twist@elbow,elbow,atol=1e-15)
  np.testing.assert_allclose(elbow+twist@comp@forearm,elbow+forearm,atol=1e-15)
  self.assertGreater(np.linalg.norm(twist-np.eye(3)),1)

class CalculationTests(unittest.TestCase):
 def setUp(self):
  self.m=next(m for m in dictionary() if m['id']=='lead_knee_net_extension')
  self.s={'lead_knee_flexion':np.array([50.,55,60,40,20,10])};self.mask={'lead_knee_flexion':np.ones(6,bool)}
  self.o={'events':{'fc':{'interval':[1,1],'source':'agent_visual_source_bracket'},'br':{'interval':[5,5],'source':'agent_visual_source_bracket'}},'cadence':{'uniqueImageCandidates':list(range(6)),'physicalTimeStatus':'unverified_playback_rate'},'frameTimes':list(range(6))}
 def test_net_extension_is_fc_minus_br_not_max_minus_br(self):
  v,frames,r=evaluate(self.m,self.s,self.o['events'],range(6),np.arange(6),self.mask);self.assertEqual(v,45);self.assertIsNone(r)
 def test_missing_event_is_null_not_zero(self):
  self.o['events']['fc']['interval']=None;v=measurement(self.m,self.s,self.o,self.mask);self.assertIsNone(v['value']);self.assertEqual(v['missingReason'],'event_not_identified')
 def test_unknown_clock_blocks_velocity_not_geometry(self):
  spatial=measurement(self.m,self.s,self.o,self.mask);self.assertEqual(spatial['value'],45)
  m=next(m for m in dictionary() if m['id']=='lead_knee_peak_velocity');v=measurement(m,self.s,self.o,self.mask);self.assertIsNone(v['value']);self.assertEqual(v['status'],'physical_time_unverified')
 def test_verified_clock_computes_known_physical_velocity(self):
  n=61;times=np.arange(n)/240;series={'lead_knee_flexion':70-100*times};mask={'lead_knee_flexion':np.ones(n,bool)};o={'events':{'fc':{'interval':[10,10],'source':'agent_visual_source_bracket'},'br':{'interval':[50,50],'source':'agent_visual_source_bracket'}},'cadence':{'uniqueImageCandidates':list(range(n)),'physicalTimeStatus':'verified','physicalSecondsPerPlaybackSecond':1,'physicalClockEvidence':'synthetic clock 240 Hz'},'frameTimes':times.tolist()}
  m=next(m for m in dictionary() if m['id']=='lead_knee_peak_velocity');v=measurement(m,series,o,mask);self.assertAlmostEqual(v['value'],100,places=4)
  m=next(m for m in dictionary() if m['id']=='fc_br_time_ms');v=measurement(m,series,o,mask);self.assertAlmostEqual(v['value'],1000/6,places=4)
  o['cadence'].pop('physicalClockEvidence');v=measurement(m,series,o,mask);self.assertIsNone(v['value'])
 def test_candidate_events_do_not_claim_accepted(self):
  self.o['events']['fc']['source']='image_keypoint_hypothesis';v=measurement(self.m,self.s,self.o,self.mask);self.assertEqual(v['status'],'event_candidate');self.assertEqual(v['externalAccuracyValidation'],'not_validated_against_external_GT')
 def test_occluded_endpoint_is_not_interpolated(self):
  self.mask['lead_knee_flexion'][1]=False;v=measurement(self.m,self.s,self.o,self.mask);self.assertIsNone(v['value']);self.assertEqual(v['status'],'required_joints_occluded')
 def test_event_perturbation_is_recorded(self):
  self.o['events']['fc']['interval']=[0,2];v=measurement(self.m,self.s,self.o,self.mask);self.assertEqual(v['sensitivity']['eventFrames'],[40,50]);self.assertIn('definitionCorrespondence',v);self.assertIn('stability',v)

class ReleaseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.db=json.loads((ROOT/'data/metrics/database.json').read_text());cls.cov=json.loads((ROOT/'data/metrics/coverage.json').read_text())
 def test_schema_three_axes_and_no_primary_2d(self):
  self.assertEqual(self.db['schemaVersion'],2);self.assertEqual(self.db['version'],VERSION)
  for m in self.db['dictionary']:
   self.assertEqual(m['basis'],'3d');self.assertTrue(m['sources']);self.assertIn('differenceFromSource',m);self.assertIn('sign',m);self.assertIn('implementation',m)
  for r in self.db['rows']:
   for v in r['metrics'].values():
    self.assertIn('definitionCorrespondence',v);self.assertIn('stability',v);self.assertEqual(v['externalAccuracyValidation'],'not_validated_against_external_GT')
    self.assertTrue(v['value'] is None or isinstance(v['value'],(int,float)))
 def test_preserved_release_hashes(self):
  d=ROOT/'data/releases/metrics-20261028-v1';m=json.loads((d/'manifest.json').read_text())
  # The archived uncompressed JSON is the exact old identity/metric source.
  old=json.load(gzip.open(d/'database.json.gz','rt'));self.assertEqual(old['schemaVersion'],1);self.assertEqual(len(old['rows']),len(self.db['rows']))
  self.assertEqual({r['id'] for r in old['rows']},{r['id'] for r in self.db['rows']})
  for name,entry in m['files'].items():
   raw=(d/entry['archive']).read_bytes();raw=raw if name.endswith('.gz') else gzip.decompress(raw);self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['sha256'],name)
 def test_undefined_axial_and_clock_not_fabricated(self):
  for r in self.db['rows']:
   for key,v in r['metrics'].items():
    if key in ['shoulder_external_rotation_fc','shoulder_external_rotation_max','shoulder_internal_peak_velocity','fc_br_time_ms','pelvis_trunk_peak_gap_ms']:self.assertIsNone(v['value'])
 def test_event_evidence_and_duplicate_identity(self):
  linked=[r for r in self.db['rows'] if r['unit']=='pitch_record' and r['primaryClipId']];self.assertEqual(len(linked),len({r['pitchId'] for r in linked}))
  self.assertEqual(self.cov['sourceReviewed'],19);self.assertEqual(self.cov['nativeRefit']['adopted'],17)
  for case in self.cov['cases']:
   d=json.load(gzip.open(ROOT/'data'/case['analysisPath'],'rt'));self.assertTrue(d['sourceReview']);self.assertFalse(d['sequenceDiagnostic']['physicalSequencePublished']);self.assertFalse(d['automaticEventProposals']['accepted'])
 def test_scalars_match_downloaded_analysis_pose_and_segment_frames(self):
  for case in self.cov['cases']:
   d=json.loads(gzip.decompress((ROOT/'data'/case['analysisPath']).read_bytes()));p=json.loads(gzip.decompress((ROOT/'data'/d['posePath']).read_bytes()));a=np.asarray([[q[:3] if q else [np.nan]*3 for q in f['anatomy']] for f in p['frames']]);h,k,ankle=(1,4,7) if case['hand']=='R' else (2,5,8);knee=flexion(a[:,h],a[:,k],a[:,ankle])
   for id in ['lead_knee_flexion_fc','lead_knee_flexion_br','lead_knee_net_extension']:
    v=d['metrics'][id]
    if v['value'] is None:continue
    frames=v['usedFrames'];expected=knee[frames[0]]-knee[frames[-1]] if id.endswith('net_extension') else knee[frames[0]];self.assertAlmostEqual(expected,v['value'],places=3)
   v=d['metrics']['thorax_pelvis_axial_fc']
   if v['value'] is not None:
    i=v['usedFrames'][0];seg=d['coordinate']['segment'];Q=np.array(seg['pelvisFramesScene'][i]).T@np.array(seg['thoraxFramesScene'][i]);expected=-(1 if case['hand']=='R' else -1)*np.degrees(np.arctan2(Q[1,0],Q[0,0]));self.assertAlmostEqual(expected,v['value'],places=3)
 def test_reference_is_not_broadcast_accuracy(self):
  d=json.loads((ROOT/'data/metrics/reference_check.json').read_text());self.assertEqual(d['pitches'],12);self.assertFalse(d['broadcastAccuracyValidation']);self.assertTrue(all(r['kneeGeometryVersusPublishedAngle']['rmseDegrees']<2 for r in d['results']))

if __name__=='__main__':unittest.main()
