/* Explicit model provenance; the immutable legacy index is never relabeled. */
(function(root){
  function published(base){return {...base,reconstructionId:'published',reconstructionState:'published',reconstructionVersion:base.reconstructionVersion||'dataset-20261028'};}
  function preview(base,registry){
    if(registry?.clips?.[base.id])return resolve(base,registry);
    return {...base,pose3d:null,pose3dStatus:'preview_unavailable',reconstructionId:'preview_unavailable',model3d:'Preview unavailable'};
  }
  function resolve(base,registry,requested){
    const entry=registry?.clips?.[base.id],method=requested||entry?.preferred;
    const variant=method!=='legacy'&&entry?.variants?.[method];
    if(!variant)return {...base,reconstructionId:'legacy',reconstructionState:entry?'legacy_preserved':registry?.unavailable?'registry_unavailable':'legacy_unprocessed',reconstructionLabel:'WHAM v2/v3',reconstructionVersion:'legacy',legacyPose3d:base.pose3d};
    return {...base,pose3d:variant.pose3d,pose3dBackend:variant.id,model3d:variant.model3d,motionMethod:variant.method,sceneFrame:variant.sceneFrame,motionQuality:variant.qualityUrl,pose3dStatus:'review_required_preview',pose3dReasons:['new_method_unvalidated'],reconstructionId:variant.id,reconstructionState:variant.role,reconstructionLabel:variant.label,reconstructionVersion:registry.version,legacyPose3d:base.pose3d,legacyPose3dStatus:base.pose3dStatus,legacyPose3dReasons:base.pose3dReasons,legacySceneSignature:base.sceneSignature,sceneSignature:variant.sourceFitSha256,whamSignature:undefined,pose3dRaw:undefined,reconstructionCoordinates:variant.coordinates?'2D: crop pixels · 3D: '+variant.coordinates:'2D: crop pixels · 3D: fixed camera basis, model meters; no calibrated floor'};
  }
  async function decode(response,path){
    if(!response.ok)throw Error(`Pose request failed: ${response.status}`);
    if(!/\.gz(?:[?#]|$)/.test(path))return response.json();
    const bytes=new Uint8Array(await response.arrayBuffer());
    const raw=bytes[0]===31&&bytes[1]===139?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text():new TextDecoder().decode(bytes);
    return JSON.parse(raw);
  }
  const api={resolve,published,preview,decode};root.ReconstructionRegistry=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
