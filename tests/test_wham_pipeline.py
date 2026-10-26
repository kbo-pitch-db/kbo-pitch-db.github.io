"""Mapping, track-boundary and publication tests with explicit synthetic fixtures.

These are not model inference or accuracy tests. They never enter public data/.
"""
import csv
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wham_pipeline import (HALPE26,HALPE_EDGES,COCO17,coco_input,camera_coordinates,validate_halpe_model,
    build_tracking,refine_clip,paths_for,write_index)
from package_motion import export_pose,package
from setup_wham import extract_body_assets
from wham_runtime import runtime_environment


class WhamTests(unittest.TestCase):
    def test_bundled_cudnn_precedes_old_system_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix=Path(temp)
            torch_lib=prefix/'lib/python3.9/site-packages/torch/lib'
            torch_lib.mkdir(parents=True)
            environment=runtime_environment(prefix,{'LD_LIBRARY_PATH':'/usr/local/cuda-11.6/lib64:'})
            self.assertEqual(environment['LD_LIBRARY_PATH'].split(':')[0],str(torch_lib))
            self.assertIn('/usr/local/cuda-11.6/lib64',environment['LD_LIBRARY_PATH'])
            self.assertEqual(runtime_environment(prefix,environment),environment)

    def test_runtime_leaves_non_torch_environments_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(runtime_environment(Path(temp),{'OTHER':'value'}),{'OTHER':'value'})

    def test_runtime_joint_validation_accepts_symbolic_onnx_metadata(self):
        class Pose:
            model_input_size=(192,256)
            session=SimpleNamespace(get_outputs=lambda:[SimpleNamespace(shape=['batch','symbolic_joint_count',384])])
            def __call__(self,image,bboxes):
                return np.zeros((1,26,2)),np.zeros((1,26))
        validate_halpe_model(Pose())

    def test_runtime_joint_validation_rejects_actual_coco17_output(self):
        class Pose:
            model_input_size=(192,256)
            def __call__(self,image,bboxes):
                return np.zeros((1,17,2)),np.zeros((1,17))
        with self.assertRaisesRegex(ValueError,'decoded Halpe-26'):
            validate_halpe_model(Pose())

    def halpe(self, count=12):
        points = np.zeros((count,26,2),np.float32)
        points[:,:,0] = np.linspace(100,220,26)
        points[:,:,1] = np.linspace(80,400,26)
        points += np.arange(count)[:,None,None]
        return dict(keypoints=points,scores=np.ones((count,26),np.float32),
            missing=np.zeros((count,26),bool),bbox=np.tile([80,50,260,460],(count,1)),
            source_time=10+np.arange(count)/30,frame_index=300+np.arange(count),
            track_id=np.zeros(count,int),fps=np.array(30),joints=np.array(HALPE26),
            edges=np.array(HALPE_EDGES),model=np.array('RTMPose-m Halpe-26'),
            model_sha256=np.array('synthetic-2d-test-only'),input_signature=np.array('fixture'))

    def test_halpe_mapping_preserves_input_and_masks_missing(self):
        raw = self.halpe()
        before = raw['keypoints'].copy()
        raw['missing'][3,9] = True
        output = coco_input(raw['keypoints'],raw['scores'],raw['missing'],[10,20])
        self.assertEqual(output.shape,(12,17,3))
        np.testing.assert_array_equal(output[0,:,:2],before[0,:17]-[10,20])
        self.assertEqual(output[3,9,2],0)
        np.testing.assert_array_equal(raw['keypoints'],before)
        self.assertEqual(HALPE26[20:],[ 'left_big_toe','right_big_toe','left_small_toe','right_small_toe','left_heel','right_heel'])
        with self.assertRaises(ValueError):
            coco_input(before[:,:17],raw['scores'][:,:17],raw['missing'][:,:17],[0,0])

    def test_track_state_never_crosses_missing_frames_or_identity(self):
        raw = self.halpe()
        raw['track_id'][6:] = 1
        raw['scores'][3] = 0
        tracks = build_tracking(raw,np.array([0,0]))
        runs = [r['frame_id'].tolist() for r in tracks.values()]
        self.assertEqual(runs,[[0,1,2],[4,5],[6,7,8,9,10,11]])
        for track in tracks.values():
            self.assertTrue((track['bbox'][:,2]>0).all())

    def test_camera_axes_and_hip_origin(self):
        raw = np.arange(2*23*3,dtype=np.float32).reshape(2,23,3)
        result = camera_coordinates(raw)
        self.assertEqual(result.shape,(2,17,3))
        np.testing.assert_allclose((result[:,11]+result[:,12])/2,0)
        np.testing.assert_allclose(result[:,1]-result[:,0],[ [3,-3,-3],[3,-3,-3]])

    def fixture(self,root,model='WHAM synthetic test fixture'):
        dataset,site = root/'dataset',root/'site'
        dataset.mkdir();site.mkdir()
        row = dict(dataset_clip_id='source_clip',source_id='source',source_clip_id='clip',
                   crop_x1='10',crop_y1='20')
        raw = self.halpe()
        xyz = np.c_[raw['keypoints'][0,:17],np.linspace(-.1,.1,17)]
        xyz = np.tile(xyz[None],(12,1,1)).astype(np.float32)/300
        prior = dict(keypoints=xyz,keypoints_2d=raw['keypoints'][:,:17],
            scores=raw['scores'][:,:17],missing=np.zeros((12,17),bool),
            source_time=raw['source_time'],frame_index=raw['frame_index'],fps=raw['fps'],
            joints=np.array(COCO17),model=np.array(model),model_sha256=np.array('synthetic-3d-test-only'),
            source_2d_signature=np.array('fixture'),input_signature=np.array('3d-fixture'))
        paths = paths_for(row)
        for path,values in [(paths['pose_2d'],raw),(paths['prior_3d'],prior)]:
            (dataset/path).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dataset/path,**values)
        args = SimpleNamespace(dataset=dataset,strength=.45,max_gap=3)
        record = refine_clip(args,row,raw,prior)
        write_index(dataset,record)
        with (dataset/'clips.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        index={'schemaVersion':4,'packageStats':{},'clips':[dict(id='source_clip',fps=30,frames=12,
            mediaPolicy='pose-only',pose='baseline.json',poseVideo='baseline.mp4')]}
        (site/'clips.json').write_text(json.dumps(index))
        return dataset,site,row,raw,prior

    def test_refinement_and_publication_keep_26_in_2d_and_17_in_3d(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,site,row,raw,prior = self.fixture(Path(temp))
            exported=export_pose(dataset/paths_for(row)['pose_2d'],row,2,30)
            self.assertEqual(len(exported['joints']),26)
            self.assertEqual(len(exported['frames'][0]['keypoints']),26)
            self.assertIn([15,24],exported['edges'])
            package(dataset,site,backend='wham')
            index=json.loads((site/'clips.json').read_text())
            clip=index['clips'][0]
            self.assertEqual(clip['pose3dBackend'],'wham')
            self.assertEqual(clip['poseVideo'],'')
            self.assertEqual(index['packageStats']['clipsWithWham'],1)
            self.assertEqual(len(json.loads((site/clip['pose']).read_text())['joints']),26)
            self.assertEqual(len(json.loads((site/clip['pose3d']).read_text())['joints']),17)
            # A repeated publish reuses the same numerical assets.
            old=(site/clip['pose3d']).stat().st_mtime_ns
            package(dataset,site,backend='wham')
            self.assertEqual(old,(site/clip['pose3d']).stat().st_mtime_ns)
            # A generic site rebuild must not leave a cached 17-joint pose in a WHAM clip.
            index=json.loads((site/'clips.json').read_text())
            index['clips'][0]['pose']='rebuilt-coco17.json'
            (site/'clips.json').write_text(json.dumps(index))
            package(dataset,site,backend='wham')
            self.assertTrue(json.loads((site/'clips.json').read_text())['clips'][0]['pose'].endswith('pose_halpe26.json'))

    def test_partial_migration_keeps_unprocessed_model_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,site,row,raw,prior = self.fixture(Path(temp))
            index=json.loads((site/'clips.json').read_text())
            legacy=dict(id='not_processed',fps=30,frames=12,model='RTMPose COCO-17',
                pose='legacy.json',pose3d='blazepose.json',poseVideo='',pose3dStatus='limited')
            index['clips'].append(legacy)
            (site/'clips.json').write_text(json.dumps(index))
            package(dataset,site,backend='wham')
            result=json.loads((site/'clips.json').read_text())
            self.assertEqual(result['clips'][1],legacy)
            progress=json.loads((site/'wham_progress.json').read_text())
            self.assertEqual(progress['completed_wham'],1)
            self.assertEqual(progress['remaining_baseline'],1)

    def test_mislabeled_baseline_never_becomes_public_wham(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,site,row,raw,prior = self.fixture(Path(temp))
            prior['model']=np.array('MediaPipe BlazePose')
            np.savez_compressed(dataset/paths_for(row)['prior_3d'],**prior)
            before=(site/'clips.json').read_bytes()
            with self.assertRaises(ValueError):
                package(dataset,site,backend='wham')
            self.assertEqual(before,(site/'clips.json').read_bytes())

    def test_source_time_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,site,row,raw,prior = self.fixture(Path(temp))
            prior['frame_index']=prior['frame_index']+1
            with self.assertRaises(ValueError):
                refine_clip(SimpleNamespace(dataset=dataset,strength=.45,max_gap=3),row,raw,prior)

    def test_archive_cannot_write_outside_private_model_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as stream:
                entry=tarfile.TarInfo('../outside');entry.size=1
                stream.addfile(entry,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                extract_body_assets(archive,root/'model')


if __name__=='__main__':
    unittest.main()
