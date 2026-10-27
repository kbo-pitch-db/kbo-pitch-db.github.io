"""Immutable original release fixture after the explicit production migration."""
from pathlib import Path
import gzip,json
SITE=Path(__file__).resolve().parents[1]
def original_index_bytes():
 p=SITE/'data/releases/wham-20261028/original_index.json.gz'
 return gzip.decompress(p.read_bytes()) if p.exists() else (SITE/'data/clips.json').read_bytes()
def original_clips():return json.loads(original_index_bytes())['clips']
