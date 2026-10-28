/* Procedural bone illustration driven by inferred SMPL joint centers.
 * Geometry is illustrative; it adds no measured anatomy or OpenSim simulation.
 */
import {interpolateAnatomy,interpolateRigidFeet} from './bone_interpolation.js?v=20261028-combined-v1';
export {interpolateAnatomy};
import * as THREE from './third_party/three.module.min.js';

const V=(p)=>new THREE.Vector3(...p);
const up=new THREE.Vector3(0,1,0);
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21];

function interpolate(a,b,t){return a.map((p,i)=>p&&b?.[i]?p.slice(0,3).map((x,k)=>x+(b[i][k]-x)*t):p?.slice(0,3)||null);}
function fallbackAnatomy(p){
  const mix=(a,b,t=.5)=>a&&b?a.slice(0,3).map((x,k)=>x*(1-t)+b[k]*t):null;
  const pelvis=mix(p[11],p[12]),neck=mix(p[5],p[6]);
  return [pelvis,p[11],p[12],mix(pelvis,neck,.25),p[13],p[14],mix(pelvis,neck,.5),p[15],p[16],mix(pelvis,neck,.8),p[15],p[16],neck,mix(neck,p[5]),mix(neck,p[6]),p[0],p[5],p[6],p[7],p[8],p[9],p[10],p[9],p[10]];
}

export function createBoneScene(canvas,data){
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:true});
  renderer.setSize(720,720,false);renderer.setPixelRatio(1);
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.05;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#1b3028');
  const material=new THREE.MeshStandardMaterial({color:'#eee5cc',roughness:.48,metalness:.05});
  const jointMaterial=new THREE.MeshStandardMaterial({color:'#d5d9cd',roughness:.6});
  const cavityMaterial=new THREE.MeshStandardMaterial({color:'#4e594e',roughness:1});
  const sphereGeometry=new THREE.SphereGeometry(1,16,12);
  const shaftGeometry=new THREE.LatheGeometry([
    new THREE.Vector2(.28,-.5),new THREE.Vector2(.48,-.46),new THREE.Vector2(.4,-.36),
    new THREE.Vector2(.21,-.22),new THREE.Vector2(.18,0),new THREE.Vector2(.22,.23),
    new THREE.Vector2(.39,.38),new THREE.Vector2(.48,.46),new THREE.Vector2(.28,.5)],12);
  const rig=new THREE.Group();scene.add(rig);
  const boneParts=[];
  function mesh(geometry,mat=material,parent=rig){const item=new THREE.Mesh(geometry,mat);item.castShadow=true;item.receiveShadow=true;parent.add(item);return item;}
  function segment(radius=.025){return {shaft:mesh(shaftGeometry),a:mesh(sphereGeometry),b:mesh(sphereGeometry),radius};}
  function placeSegment(bone,a,b,scale=1){
    const valid=a&&b;for(const m of [bone.shaft,bone.a,bone.b])m.visible=!!valid;
    if(!valid)return;
    const va=V(a),vb=V(b),direction=vb.clone().sub(va),length=direction.length();
    bone.shaft.position.copy(va).add(vb).multiplyScalar(.5);
    bone.shaft.quaternion.setFromUnitVectors(up,direction.normalize());
    const r=bone.radius*scale;bone.shaft.scale.set(2*r,length,2*r);
    bone.a.position.copy(va);bone.b.position.copy(vb);
    bone.a.scale.setScalar(r*.8);bone.b.scale.setScalar(r*.9);
  }
  for(let j=1;j<24;j++){
    if([3,6,9,12,13,14,15].includes(j))continue;
    const radius=[4,5].includes(j)?.035:[7,8].includes(j)?.025:[18,19].includes(j)?.025:[20,21].includes(j)?.016:.018;
    boneParts.push({a:parents[j],b:j,part:segment(radius)});
  }
  const paired=[{a:4,b:7},{a:5,b:8},{a:18,b:20},{a:19,b:21}].map(p=>({...p,part:segment(.009)}));
  const clavicles=[segment(.014),segment(.014)];
  const spine=Array.from({length:4},()=>segment(.025));
  const vertebrae=Array.from({length:13},()=>mesh(sphereGeometry,jointMaterial));
  const ribs=new THREE.Group();rig.add(ribs);
  for(let i=0;i<9;i++)for(const side of [-1,1]){
    const width=.48*(1-.035*Math.abs(i-3)),y=.45-i*.105;
    const curve=new THREE.CatmullRomCurve3([
      new THREE.Vector3(side*.06,y,.23),new THREE.Vector3(side*width,y-.04,.14),
      new THREE.Vector3(side*(width+.03),y-.12,-.09),new THREE.Vector3(side*.07,y-.15,-.25)]);
    mesh(new THREE.TubeGeometry(curve,20,.014,6,false),material,ribs);
  }
  const sternum=mesh(new THREE.CapsuleGeometry(.027,.53,4,8),material,ribs);sternum.position.set(0,.05,.24);
  const pelvis=new THREE.Group();rig.add(pelvis);
  for(const side of [-1,1]){
    const wing=mesh(new THREE.SphereGeometry(1,18,12),material,pelvis);
    wing.scale.set(.53,.5,.16);wing.position.set(side*.4,.08,0);wing.rotation.z=side*-.3;
    const ring=mesh(new THREE.TorusGeometry(.24,.055,8,24),material,pelvis);
    ring.position.set(side*.3,-.35,0);ring.scale.set(.8,1.1,.7);
  }
  const sacrum=mesh(new THREE.ConeGeometry(.2,.5,6),material,pelvis);sacrum.rotation.z=Math.PI;sacrum.position.set(0,-.08,-.13);
  const head=new THREE.Group();rig.add(head);
  const cranium=mesh(new THREE.SphereGeometry(1,24,20),material,head);cranium.scale.set(.084,.105,.086);cranium.position.y=.035;
  const face=mesh(new THREE.SphereGeometry(1,18,14),material,head);face.scale.set(.067,.066,.055);face.position.set(0,-.016,.046);
  for(const x of [-.031,.031]){
    const socket=mesh(sphereGeometry,cavityMaterial,head);socket.scale.set(.022,.021,.012);socket.position.set(x,.005,.091);
  }
  const nose=mesh(new THREE.ConeGeometry(.013,.03,3),cavityMaterial,head);nose.rotation.z=Math.PI;nose.position.set(0,-.027,.099);
  const jaw=mesh(new THREE.TorusGeometry(.045,.009,8,18,Math.PI*1.5),material,head);
  jaw.rotation.x=.75;jaw.rotation.z=Math.PI*.75;jaw.position.set(0,-.071,.045);jaw.scale.set(1,.7,1);
  const feet=Array.from({length:6},()=>segment(.009));
  const hands=[];
  for(const side of [0,1])for(let finger=0;finger<5;finger++)hands.push({side,finger,segments:[segment(.006),segment(.0045),segment(.0035)]});

  const bounds=new THREE.Box3();
  for(const f of data.frames)for(const p of (f.anatomy||f.keypoints))if(p&&p.slice(0,3).every(Number.isFinite))bounds.expandByPoint(V(p.slice(0,3)));
  if(bounds.isEmpty())bounds.set(new THREE.Vector3(-1,0,-1),new THREE.Vector3(1,2,1));
  const comparisonBounds=data.scene?.comparison_bounds;
  if(comparisonBounds)bounds.set(V(comparisonBounds.min),V(comparisonBounds.max));
  const showFloor=data.scene?.show_floor!==false;
  const floorY=showFloor?Number(data.scene?.floor?.height??bounds.min.y-.04):0;
  const target=bounds.getCenter(new THREE.Vector3());if(showFloor)target.y=(floorY+bounds.max.y)/2;
  const size=bounds.getSize(new THREE.Vector3());
  let radius=.95;
  for(const f of data.frames)for(const p of (f.anatomy||f.keypoints))if(p&&p.slice(0,3).every(Number.isFinite))radius=Math.max(radius,V(p.slice(0,3)).distanceTo(target));
  if(comparisonBounds)radius=Math.max(.95,size.length()/2);
  radius*=1.08;
  const camera=new THREE.OrthographicCamera(-radius,radius,radius,-radius,.01,100);
  const groundSize=Math.max(4,Math.ceil(Math.max(size.x,size.z)+2)*2);
  const ground=mesh(new THREE.PlaneGeometry(groundSize,groundSize),new THREE.MeshStandardMaterial({color:'#294437',roughness:1}),scene);
  ground.rotation.x=-Math.PI/2;ground.position.set(target.x,floorY,target.z);ground.receiveShadow=true;ground.castShadow=false;
  const grid=new THREE.GridHelper(groundSize,Math.round(groundSize/.25),'#6a9279','#3c6050');grid.position.set(target.x,floorY+.002,target.z);scene.add(grid);
  const axes=new THREE.AxesHelper(.4);axes.position.set(0,floorY+.008,0);scene.add(axes);
  const originDot=mesh(new THREE.RingGeometry(.035,.055,24),new THREE.MeshBasicMaterial({color:'#597767',side:THREE.DoubleSide}),scene);
  originDot.rotation.x=-Math.PI/2;originDot.position.y=floorY+.006;
  ground.visible=grid.visible=originDot.visible=showFloor;
  canvas.dataset.floorVisible=String(showFloor);
  scene.add(new THREE.HemisphereLight('#fffdf3','#8a9e92',1.7));
  const light=new THREE.DirectionalLight('#fff8e6',2.6);light.position.set(-3,6,4);light.castShadow=true;
  light.shadow.mapSize.set(1024,1024);light.shadow.camera.left=light.shadow.camera.bottom=-4;light.shadow.camera.right=light.shadow.camera.top=4;
  light.shadow.normalBias=.025;scene.add(light);
  const fill=new THREE.DirectionalLight('#bddbec',1.4);fill.position.set(3,2,-3);scene.add(fill);
  let zoom=1;
  canvas.addEventListener('wheel',onWheel,{passive:false});
  function onWheel(event){event.preventDefault();zoom=THREE.MathUtils.clamp(zoom*Math.exp(-event.deltaY*.001),.65,2.8);camera.zoom=zoom;camera.updateProjectionMatrix();renderer.render(scene,camera);}

  function basis(points){
    if(!points[16]||!points[17]||!points[0]||!points[12])return new THREE.Quaternion();
    const y=V(points[12]).sub(V(points[0])).normalize();
    const x=V(points[16]).sub(V(points[17]));x.addScaledVector(y,-x.dot(y)).normalize();
    const z=new THREE.Vector3().crossVectors(x,y).normalize();
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z));
  }
  function render(current,next,fraction,yaw,pitch){
    material.color.set(current.interpolated?.some(Boolean)?'#efbd78':'#eee5cc');
    const points=interpolateAnatomy(current.anatomy||fallbackAnatomy(current.keypoints),next?.anatomy||fallbackAnatomy(next?.keypoints||current.keypoints),fraction,parents);
    const visible=points.filter(Boolean).length>12;rig.visible=visible;
    if(visible){
      const q=basis(points),personScale=Math.max(.7,Math.min(1.4,points[4]&&points[1]?V(points[4]).distanceTo(V(points[1]))/.42:1));
      for(const b of boneParts)placeSegment(b.part,points[b.a],points[b.b],personScale);
      for(const b of paired){
        const shift=new THREE.Vector3(.018,0,-.006).applyQuaternion(q).multiplyScalar(personScale);
        placeSegment(b.part,points[b.a]&&V(points[b.a]).add(shift).toArray(),points[b.b]&&V(points[b.b]).add(shift).toArray(),personScale);
      }
      const spineIndices=[0,3,6,9,12];
      for(let i=0;i<4;i++)placeSegment(spine[i],points[spineIndices[i]],points[spineIndices[i+1]],personScale);
      for(let i=0;i<vertebrae.length;i++){
        const t=i/(vertebrae.length-1)*3,index=Math.min(2,Math.floor(t)),a=points[spineIndices[index]],b=points[spineIndices[index+1]];
        vertebrae[i].visible=!!(a&&b);
        if(a&&b){vertebrae[i].position.copy(V(a).lerp(V(b),t-index));vertebrae[i].quaternion.copy(q);vertebrae[i].scale.set(.043,.015,.037).multiplyScalar(personScale);}
      }
      placeSegment(clavicles[0],points[12],points[16],personScale);placeSegment(clavicles[1],points[12],points[17],personScale);
      if(points[6]&&points[9]&&points[16]&&points[17]){
        ribs.visible=true;ribs.position.copy(V(points[6]).lerp(V(points[9]),.65));ribs.quaternion.copy(q);
        const width=V(points[16]).distanceTo(V(points[17]));ribs.scale.set(width*.85,personScale*.34,personScale*.42);
      }else ribs.visible=false;
      if(points[0]&&points[1]&&points[2]){pelvis.visible=true;pelvis.position.copy(V(points[0]));pelvis.quaternion.copy(q);const hip=V(points[1]).distanceTo(V(points[2]));pelvis.scale.set(hip*1.25,personScale*.22,personScale*.24);}else pelvis.visible=false;
      if(points[15]){head.visible=true;head.position.copy(V(points[15]));head.quaternion.copy(q);head.scale.setScalar(personScale);}else head.visible=false;
      const foot=(data.quality?.constant_shoe_shape?interpolateRigidFeet:interpolate)(current.feet||[],next?.feet||current.feet||[],fraction);
      for(let side=0;side<2;side++){
        const ankle=points[7+side],toe=foot[side],small=foot[2+side],heel=foot[4+side];
        placeSegment(feet[side*3],ankle,heel,personScale);placeSegment(feet[side*3+1],heel,toe,personScale);placeSegment(feet[side*3+2],heel,small,personScale);
      }
      for(const hand of hands){
        const wrist=points[20+hand.side],palm=points[22+hand.side],elbow=points[18+hand.side];
        if(!wrist||!palm||!elbow){for(const s of hand.segments)placeSegment(s,null,null);continue;}
        const direction=V(palm).sub(V(wrist)).normalize();
        if(direction.lengthSq()<.5)direction.copy(V(wrist).sub(V(elbow))).normalize();
        const spread=new THREE.Vector3(1,0,0).applyQuaternion(q).multiplyScalar((hand.finger-2)*.014*personScale);
        const start=V(palm).add(spread),length=(hand.finger===0?.05:.075-Math.abs(hand.finger-2)*.009)*personScale;
        let a=start;
        for(let j=0;j<3;j++){const b=a.clone().addScaledVector(direction,length*[.45,.32,.23][j]);placeSegment(hand.segments[j],a.toArray(),b.toArray(),personScale);a=b;}
      }
    }
    const distance=radius*4,elevation=THREE.MathUtils.clamp(pitch+.35,.1,1.3);
    camera.position.set(target.x+Math.sin(yaw)*Math.cos(elevation)*distance,target.y+Math.sin(elevation)*distance,target.z+Math.cos(yaw)*Math.cos(elevation)*distance);
    camera.lookAt(target);renderer.render(scene,camera);
    canvas.dataset.cameraMatrix=camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).elements.map(v=>v.toFixed(8)).join(',');
  }
  canvas.dataset.renderer='bone-mesh';canvas.dataset.coordinateFrame=data.scene?.frame||'legacy-root-relative';
  return {render,resetZoom(){zoom=1;camera.zoom=1;camera.updateProjectionMatrix();},dispose(){
    canvas.removeEventListener('wheel',onWheel);
    const geometries=new Set(),materials=new Set();scene.traverse(item=>{if(item.geometry)geometries.add(item.geometry);if(item.material)materials.add(item.material);});
    geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();renderer.forceContextLoss();
  }};
}
