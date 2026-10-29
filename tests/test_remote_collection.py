"""Source deduplication, bounded downloads and changed-source protection."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('collector', ROOT/'scripts/collect_remote_videos.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class RemoteCollectionTests(unittest.TestCase):
    def test_deduplicates_provider_ids_across_renamed_paths_and_honors_budget(self):
        inventory = [dict(name='known [abcdefghijk].mp4', bytes=20),
                     dict(name='new [ABCDEFGHIJK].mp4', bytes=50),
                     dict(name='renamed [ABCDEFGHIJK].mp4', bytes=60),
                     dict(name='large [12345678901].mp4', bytes=100),
                     dict(name='small [12345678902].mp4', bytes=30)]
        rows = collector.select_batch(inventory, {'abcdefghijk'}, 5, 90)
        self.assertEqual([r['provider_video_id'] for r in rows], ['12345678902', 'ABCDEFGHIJK'])
        self.assertEqual(sum(r['bytes'] for r in rows), 80)
        self.assertEqual(len(collector.select_batch(inventory, set(), 1, 1000)), 1)

    def test_rejects_escaping_paths_and_line_based_manifest_injection(self):
        for name in ['../video.mp4', '/tmp/video.mp4', 'a\\b.mp4', 'a\nb.mp4', '.', '..']:
            with self.assertRaises(ValueError):
                collector.checked_name(name)
        with self.assertRaises(ValueError):
            collector.validate_destination(ROOT/'data/new-originals')

    def test_fetch_hash_verification_and_resume_rejects_changed_remote(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_index = root/'sources.csv'
            source_index.write_text('provider_video_id\nabcdefghijk\n')
            filename = '새 경기 [ABCDEFGHIJK].mp4'
            payload = b'new source content'
            source = root/'source.mp4'
            source.write_bytes(payload)
            expected = collector.digest(source)
            destination = root/'batch'
            remote_hash = {filename: expected}
            transfers = []

            def run(command, **kwargs):
                if command[0] == 'ssh':
                    data = ([dict(name=filename, bytes=len(payload))]
                            if 'root.iterdir()' in command[-1] else remote_hash)
                    return subprocess.CompletedProcess(command, 0, json.dumps(data), '')
                self.assertEqual(command[0], 'rsync')
                transfers.append(command)
                (destination/filename).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0)

            args = ['--remote', 'test-host', '--remote-dir', '/data/2024',
                    '--sources', str(source_index), '--destination', str(destination),
                    '--reserve-gib', '0', '--fetch']
            with patch.object(collector.subprocess, 'run', side_effect=run):
                collector.main(args)
                manifest = json.loads((destination/'collection_manifest.json').read_text())
                self.assertTrue(manifest['files'][0]['verified'])
                self.assertFalse(manifest['published'])
                self.assertEqual((destination/filename).read_bytes(), payload)
                self.assertEqual(len(transfers), 1)
                # A same-size server replacement must fail before another rsync.
                remote_hash[filename] = '0'*64
                with self.assertRaisesRegex(ValueError, 'content changed'):
                    collector.main(args)
                self.assertEqual(len(transfers), 1)

    def test_corrupted_download_never_marked_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root/'sources.csv'
            index.write_text('provider_video_id\n')
            destination = root/'batch'
            name = 'video [ABCDEFGHIJK].mp4'

            def run(command, **kwargs):
                if command[0] == 'ssh':
                    data = ([dict(name=name, bytes=3)] if 'root.iterdir()' in command[-1]
                            else {name: '0'*64})
                    return subprocess.CompletedProcess(command, 0, json.dumps(data), '')
                (destination/name).write_bytes(b'bad')
                return subprocess.CompletedProcess(command, 0)

            with patch.object(collector.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'verification'):
                    collector.main(['--remote', 'test-host', '--remote-dir', '/data',
                                    '--sources', str(index), '--destination', str(destination),
                                    '--reserve-gib', '0', '--fetch'])
            manifest = json.loads((destination/'collection_manifest.json').read_text())
            self.assertNotIn('verified', manifest['files'][0])


if __name__ == '__main__':
    unittest.main()
