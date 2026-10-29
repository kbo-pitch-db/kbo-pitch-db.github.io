"""Plan and fetch an incremental source batch; never publish intermediate poses.

Uses the caller's SSH configuration/agent. No credentials are stored. Downloaded
videos stay outside the public repository; successful files are SHA-256 verified
against the server before becoming inputs to the existing reconstruction pipeline.
"""
import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

WORKSPACE = Path(__file__).resolve().parents[2]
VIDEO_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]\.mp4$", re.I)
INVENTORY_CODE = """import json,sys
from pathlib import Path
root=Path(sys.argv[1])
if not root.is_dir(): raise SystemExit('Remote source directory not found')
print(json.dumps([{'name':p.name,'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file() and p.suffix.lower()=='.mp4'],ensure_ascii=False))
"""
HASH_CODE = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);result={}
for name in json.loads(sys.argv[2]):
 p=root/name;h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''): h.update(block)
 result[name]=h.hexdigest()
print(json.dumps(result))
"""


def checked_name(name):
    if (not isinstance(name, str) or not name or name in {'.', '..'}
            or PurePosixPath(name).name != name or '/' in name or '\\' in name
            or any(c in name for c in '\n\r\t\0')):
        raise ValueError('Unsafe remote filename')
    return name


def select_batch(inventory, known_ids, limit, budget):
    """Stable, size-bounded selection, deduplicated by provider ID, not path."""
    candidates = {}
    for item in inventory:
        name = checked_name(item['name'])
        match = VIDEO_ID.search(name)
        if not match or match[1] in known_ids:
            continue
        size = item['bytes']
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError('Invalid source byte count')
        record = dict(name=name, bytes=size, provider_video_id=match[1])
        previous = candidates.get(match[1])
        if previous is None or (size, name) < (previous['bytes'], previous['name']):
            candidates[match[1]] = record
    selected = []
    for item in sorted(candidates.values(), key=lambda x: (x['bytes'], x['name'])):
        if len(selected) == limit:
            break
        if item['bytes'] <= budget:
            selected.append(item)
            budget -= item['bytes']
    return selected


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def validate_destination(destination):
    public = Path(__file__).resolve().parents[1]
    if destination == public or public in destination.parents:
        raise ValueError('Source videos must remain outside the public repository')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True, help='SSH host alias or user@host')
    parser.add_argument('--remote-dir', required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--control-path', type=Path, help='Reuse an authenticated SSH connection')
    parser.add_argument('--sources', type=Path, default=WORKSPACE/'dataset/sources.csv')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--max-gib', type=float, default=2)
    parser.add_argument('--reserve-gib', type=float, default=8,
                        help='Free space retained for reconstruction/intermediates')
    parser.add_argument('--fetch', action='store_true', help='Otherwise only save a plan')
    args = parser.parse_args(argv)
    if (not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.remote)
            or not args.remote_dir.startswith('/') or any(c in args.remote_dir for c in '\n\r\0')
            or args.limit < 1 or args.max_gib <= 0 or args.reserve_gib < 0
            or (args.port is not None and not 1 <= args.port <= 65535)):
        parser.error('Invalid SSH target, path, limit or disk budget')
    destination = args.destination.resolve()
    validate_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    if args.port:
        ssh += ['-p', str(args.port)]
    if args.identity:
        ssh += ['-i', str(args.identity.resolve()), '-o', 'IdentitiesOnly=yes']
    if args.control_path:
        ssh += ['-S', str(args.control_path.resolve())]

    def remote_json(code, *arguments):
        command = shlex.join(['python3', '-c', code, args.remote_dir, *arguments])
        result = subprocess.run([*ssh, args.remote, command], check=True,
                                capture_output=True, text=True, timeout=180)
        return json.loads(result.stdout)

    inventory = remote_json(INVENTORY_CODE)
    with args.sources.open() as f:
        known_ids = {r['provider_video_id'] for r in csv.DictReader(f)}
    manifest_path = destination/'collection_manifest.json'
    # Keep the same selected batch on retry; newly published sources are then
    # still verifiable without silently downloading a different batch.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['remote'], manifest['remoteDirectory']) != (args.remote, args.remote_dir):
            raise ValueError('Destination belongs to another remote batch')
        selected = manifest['files']
        available = {checked_name(r['name']): r['bytes'] for r in inventory}
        for item in selected:
            if available.get(checked_name(item['name'])) != item['bytes']:
                raise ValueError('Remote source changed since batch was planned')
    else:
        budget = min(int(args.max_gib * 2**30),
                     shutil.disk_usage(destination).free - int(args.reserve_gib * 2**30))
        selected = select_batch(inventory, known_ids, args.limit, budget)
        manifest = dict(schemaVersion=1, remote=args.remote, remoteDirectory=args.remote_dir,
                        sourceIndexSha256=digest(args.sources), files=selected,
                        sourceCountBefore=len(known_ids), published=False)
    if not selected:
        raise RuntimeError('No new videos fit the available disk budget')
    hashes = remote_json(HASH_CODE, json.dumps([r['name'] for r in selected]))
    for item in selected:
        expected = hashes[item['name']]
        if not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Invalid remote SHA-256')
        if item.get('sha256', expected) != expected:
            raise ValueError('Remote video content changed since batch was planned')
        item['sha256'] = expected
    write_json(manifest_path, manifest)
    print(json.dumps(dict(selected=len(selected), bytes=sum(r['bytes'] for r in selected),
                          manifest=str(manifest_path), fetch=args.fetch)))
    if not args.fetch:
        return
    needed = sum(r['bytes'] for r in selected if not (destination/r['name']).exists()
                 or (destination/r['name']).stat().st_size != r['bytes'])
    if shutil.disk_usage(destination).free < needed + int(args.reserve_gib*2**30):
        raise RuntimeError('Insufficient free space; reserved reconstruction space would be used')
    file_list = destination/'collection_files.txt'
    file_list.write_text(''.join(r['name']+'\n' for r in selected))
    subprocess.run(['rsync', '-a', '--partial', '--protect-args', '--checksum',
                    '--files-from='+str(file_list), '-e', shlex.join(ssh), '--',
                    args.remote+':'+args.remote_dir.rstrip('/')+'/', str(destination)+'/'], check=True)
    for item in selected:
        path = destination/item['name']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise RuntimeError('Downloaded video failed size/SHA-256 verification: '+item['name'])
        item['verified'] = True
    write_json(manifest_path, manifest)
    print('Verified %d source videos; reconstruction/publication has not run.' % len(selected))


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print((error.stderr or str(error)).strip(), file=sys.stderr)
        raise SystemExit(error.returncode)
