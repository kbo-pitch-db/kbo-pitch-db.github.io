"""Check the actual public bundle: media policy, complete sample, unchanged production."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import unittest

SITE = Path(__file__).resolve().parents[1]
STUDY = SITE / 'studies/pitch-20261027'


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key,value in attrs if key in ('href','src') and value)


class PublicStudyTests(unittest.TestCase):
    def test_publication_hashes_and_no_source_media(self):
        publication=json.loads((STUDY/'publication.json').read_text())
        self.assertFalse(publication['original_broadcast_media_included'])
        self.assertEqual(publication['production_poses_replaced'],0)
        actual={str(p.relative_to(STUDY)) for p in STUDY.rglob('*') if p.is_file()}
        self.assertEqual(actual,set(publication['files'])|{'publication.json'})
        for name,digest in publication['files'].items():
            path=STUDY/name
            self.assertIn(path.suffix,('.json','.csv','.html','.js','.txt'))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name)
        for forbidden in ('source_frames/','source_clips/','<video','<img'):
            self.assertNotIn(forbidden,(STUDY/'comparison.html').read_text())
            self.assertNotIn(forbidden,(STUDY/'comparison.js').read_text())

    def test_all_48_candidates_preserve_frame_alignment_and_labels(self):
        samples=json.loads((STUDY/'viewer/manifest.json').read_text())
        self.assertEqual(len(samples),48)
        self.assertEqual(sum(r['split']=='test' for r in samples),24)
        self.assertEqual(sum(r['split']=='tune' for r in samples),24)
        for row in samples:
            data=json.loads((STUDY/'viewer'/f'{row["id"]}.json').read_text())
            count=len(data['times'])
            self.assertTrue(all(a<b for a,b in zip(data['times'],data['times'][1:])))
            self.assertEqual(len(data['observations']),count)
            self.assertEqual(set(data['models']),{'current_final','H_local_temporal','WHAM_raw_articulation','MotionBERT_articulation'})
            for name,model in data['models'].items():
                self.assertEqual(len(model['points']),count)
                if 'projection' in model:self.assertEqual(len(model['projection']),count)
            self.assertFalse(data['status']['new_validation'])
            self.assertEqual(data['status']['old_status'],row['old_status'])
            self.assertTrue(data['sourceUrl'].startswith('https://www.youtube.com/'))

    def test_generated_report_links_resolve(self):
        for name in ('index.html','comparison.html'):
            parser=Links();parser.feed((STUDY/name).read_text())
            for link in parser.links:
                if link.startswith(('https:','http:','#')):continue
                target=(STUDY/link.split('#')[0]).resolve()
                self.assertTrue(target.exists(),(name,link))

    def test_production_index_and_final_2d_are_not_replaced(self):
        previous=subprocess.check_output(['git','show','HEAD:data/clips.json'],cwd=SITE)
        self.assertEqual(previous,(SITE/'data/clips.json').read_bytes())
        clips=json.loads(previous)['clips']
        self.assertEqual(len(clips),813)
        self.assertEqual(sum(c['pose3dStatus']=='screened_estimate' for c in clips),44)
        self.assertEqual(sum(c['pose3dStatus']=='review_required_preview' for c in clips),769)


if __name__=='__main__':unittest.main()
