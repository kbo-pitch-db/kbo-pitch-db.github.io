"""Adopt the already-public corrected input only after its combined audit passes."""
from common import *
from combined_population import ROOT,SITE,manifest
import copy

def main():
    ready=json.loads((ROOT/'subject_corrected_combined_v1/ready.json').read_text());cid=ready['id']
    audit=json.loads((ROOT/ready['audit_path']).read_text());assert audit['geometry_pass']
    current=(SITE/'data/clips.json').read_bytes();now=json.loads(current);before=json.loads((ROOT/'original_index.json').read_text())
    previous={c['id']:c for c in before['clips']};updated={c['id']:c for c in now['clips']}
    assert set(previous)==set(updated)
    assert all(previous[k]==updated[k] for k in previous if k!=cid),'Review unrelated canonical input changes explicitly'
    rec=ready['record'];assert sha(SITE/'data'/updated[cid]['poseRefined'])==rec['public_2d_sha256']
    manifest_path=ROOT/'inputs.json';m=manifest()
    if m.get('subject_input_updates'):return
    (ROOT/'inputs_before_subject_update.json').write_bytes(manifest_path.read_bytes())
    (ROOT/'original_index_before_subject_update.json').write_bytes((ROOT/'original_index.json').read_bytes())
    m['records']=[rec if r['id']==cid else r for r in m['records']]
    m['subject_input_updates']=[dict(id=cid,manifest=rec['subject_correction'],cause='Preserve separately published corrected pitcher input',
        prior_canonical_index_sha256=m['canonical_index_sha256'])]
    m['canonical_index_sha256']=hashlib.sha256(current).hexdigest()
    save(manifest_path,m);(ROOT/'original_index.json').write_bytes(current)
    print('ADOPTED CURRENT PITCHER INPUT',cid,flush=True)

if __name__=='__main__':main()
