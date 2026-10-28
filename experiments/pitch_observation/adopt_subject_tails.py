"""Explicitly adopt eight newly inferred pitcher tracks, never overwrite originals."""
from common import *
from combined_population import ROOT,manifest

def main():
    update=json.loads((ROOT/'tail_recovery_v1/publication_ready.json').read_text());m=manifest()
    assert len(update['records'])==8
    if m.get('recovered_final_2d_clips'):return
    (ROOT/'inputs_before_tail_recovery.json').write_bytes((ROOT/'inputs.json').read_bytes())
    replacements={r['id']:r for r in update['records']}
    m['records']=[replacements.get(r['id'],r) for r in m['records']]
    m['recovered_final_2d_clips']=sorted(replacements)
    m['final_2d_policy']='Original input bytes immutable; eight versioned tail recoveries preserve every original good frame and all timing'
    save(ROOT/'inputs.json',m);print('ADOPTED RECOVERED PITCHER TAILS',len(replacements),flush=True)

if __name__=='__main__':main()
