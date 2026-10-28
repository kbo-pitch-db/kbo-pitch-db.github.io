"""Combine a concurrently published pitcher-track correction without reverting it.

The original frozen population and wrong-person assets stay immutable. Run in
an isolated production subfolder; final assembly explicitly adopts this input.
"""
from common import *
import combined_population as population
import copy,gzip
CID='4e09dd724a8c_c0011'

def main():
    import torch
    original_root=population.ROOT;folder=original_root/'subject_corrected_combined_v1'
    current=json.loads((population.SITE/'data/clips.json').read_text())
    if current.get('packageVersion')=='gvhmr-ground-motionbert-20261028-v1':
        current=json.loads(gzip.decompress((population.SITE/'data/releases/support-20261028/original_index.json.gz').read_bytes()))
    c=next(c for c in current['clips'] if c['id']==CID)
    assert c.get('subjectCorrection')
    correction=json.loads((population.SITE/'data'/c['subjectCorrection']).read_text())
    recovery=WORKSPACE/'research/oh_delivery3_fix_20261028';source=recovery/'all813_gvhmr_v1/selected'/f'{CID}.npz'
    two=recovery/'subject_correction_v1/halpe26_refined.npz';z=load(source);meta=json.loads(source.with_suffix('.json').read_text())
    assert sha(source)==c['sceneSignature']==correction['selected_fit_sha256']
    assert sha(two)==correction['input_provenance']['corrected_2d_sha256']
    assert sha(population.SITE/'data'/c['poseRefined'])==correction['new_2d_sha256']
    m=population.manifest();rec=copy.deepcopy(next(r for r in m['records'] if r['id']==CID))
    rec.update(source=os.path.relpath(source,OUT),source_sha256=sha(source),pose_2d_refined=os.path.relpath(two,DATA),
        pose_2d_sha256=sha(two),public_2d_sha256=sha(population.SITE/'data'/c['poseRefined']),
        phase=meta['phase'],public_pose=c['pose3d'],public_2d=c['poseRefined'],
        subject_correction=c['subjectCorrection'],valid_frames=len(z['frame_index']))
    h=population.inputs(rec);_,_,rec['initial_contact_episodes']=population.episodes(h,rec['phase'],z['background_transforms'])
    assert rec['initial_contact_episodes'],'Do not invent contact when the corrected input has no support evidence'
    m['records']=[rec if r['id']==CID else r for r in m['records']]
    save(folder/'inputs.json',m);population.ROOT=folder
    torch.set_num_threads(1);population.infer([rec],'corrected-subject')
    import combined_support as ground,combined_arms as arms,combined_contact_ik as ik
    from pitch_study_fit import StudyFitter
    body=StudyFitter(WORKSPACE/'external/WHAM','cuda:0',use_cuda_graph=False)
    ground.fit(rec,body);del body;torch.cuda.empty_cache();arms.fit(rec);ik.process(rec,folder)
    import combined_foot_pose_temporal as shoe
    choice=dict(fit_path=f'polished_v1/base/combined/{CID}.npz');shoe.fit(rec,choice)
    import audit_combined_temporal as audit
    audit.ROOT=folder/'foot_pose_v2';body=StudyFitter(WORKSPACE/'external/WHAM','cpu',use_cuda_graph=False)
    report=audit.audit(rec,body);assert report['geometry_pass'],report['checks']
    save(folder/'ready.json',dict(id=CID,record=rec,fit_path=str((audit.ROOT/'combined'/f'{CID}.npz').relative_to(original_root)),
        audit_path=str((audit.ROOT/'audit'/f'{CID}.json').relative_to(original_root)),source_recovery_preserved=True))
    print('CORRECTED SUBJECT COMBINED READY',CID,flush=True)

if __name__=='__main__':
    from runtime import ensure_runtime
    ensure_runtime();main()
