"""Root-only support-transfer pilot; articulation and per-frame ankle spacing stay exact.

Uses cached GVHMR f4 poses. No new network inference, real force/COM estimate,
known camera/floor, manual review, or deployment. The previously frozen test24
is not used. Six existing tuning clips are pilots and the other six are a
predeclared confirmation of this setting, not a new untouched final test.
"""
from common import *
from events import smooth_xy
from stride_review import native_leg,phase_info
from pitch_study_fit import spline_basis
import argparse

CONFIG=dict(version='root-support-v1',steps=500,spacing_seconds=.08,
 image_weight=1.,root_acc_weight=.01,root_prior_weight=.0001,
 contact_normal_weight=.03,contact_tangent_weight=.01,penetration_weight=.001,
 plane_pitch_candidates_radians=[0.,.10,.25],normal_tolerance_leg=.03,
 max_source_p50_increase_px=2.,max_wrist_p95_increase_px=2.,
 articulation_policy='Exact unchanged camera-relative pose and body orientation',
 support_policy='Independent rear departure and lead arrival, overlap allowed; marker confidence and image speed; no hard position anchors',
 camera_policy='Each existing f4 estimated camera held constant to isolate root-only effects; not measured intrinsics',
 floor_policy='One constant plane per candidate, three uncalibrated normal hypotheses; no per-frame floor or floor lift',
 timing='Broadcast timestamps; slow-motion physical time is unknown')
PILOT=[0,1,2,4,5,7]

def support_weights(h,event,phase,bg):
 xy,s=smooth_xy(h,bg);xy=xy[:,20:26];s=s[:,20:26]
 speed=np.linalg.norm(np.gradient(xy,h['source_time'],axis=0),axis=-1)
 weights=np.exp(-(speed/.35)**2)*s*s
 plant=phase['plant_interval'];confidence=float(event.get('contact_confidence',0.))
 if phase['phase_side_conflict'] or not plant:return weights*0
 t=h['source_time'];t=t-t[0];lead=0 if phase['lead_side']=='left' else 1
 li=[lead,lead+2,lead+4];ri=[1-lead,3-lead,5-lead]
 arrival=np.clip((t-t[plant[0]])/max(.03,t[plant[1]]-t[plant[0]]),0,1)
 # Rear contact does not vanish merely because the lead foot arrives. Look
 # for sustained observed rear-foot departure, independently per marker.
 weights[:,li]*=arrival[:,None]
 for j in ri:
  departure=None
  for i in range(plant[0],len(t)-2):
   if (speed[i:i+3,j]>.55).all() and (s[i:i+3,j]>.45).all():departure=i;break
  if departure is not None:weights[:,j]*=np.clip((t[departure]+.10-t)/.10,0,1)
 # Lead heel/toes can switch off independently on clear departure.
 for j in li:
  departure=None
  for i in range(plant[1],len(t)-2):
   if (speed[i:i+3,j]>.55).all() and (s[i:i+3,j]>.45).all():departure=i;break
  if departure is not None:weights[:,j]*=np.clip((t[departure]+.10-t)/.10,0,1)
 return (confidence*weights).astype(np.float64)

def metrics(z,h,bg,translation,cw):
 p=np.concatenate([z['keypoints'],z['foot_keypoints']],1).astype(float)+translation[:,None]
 uv=p[...,:2]/p[...,2:]*float(z['focal'])+z['principal']
 uv=np.einsum('tij,tkj->tki',np.linalg.inv(bg),np.concatenate([uv,np.ones((*uv.shape[:2],1))],-1))[...,:2]
 target=np.concatenate([h['keypoints'][:,:17],h['keypoints'][:,20:26]],1)
 sc=np.concatenate([scores(h)[:,:17],scores(h)[:,20:26]],1);error=np.linalg.norm(uv-target,axis=-1);valid=(sc>.5)&np.isfinite(error)
 q=lambda a:float(np.nanpercentile(a,95)) if np.isfinite(a).any() else None
 out={}
 for name,ids in [('source',list(range(23))),('wrist',[9,10]),('feet',list(range(17,23)))]:
  e=error[:,ids][valid[:,ids]];out[name+'_p50_px']=float(np.median(e)) if len(e) else None;out[name+'_p95_px']=q(e)
 dt=np.diff(h['source_time']);links=(dt>0)&(dt<.1)&(np.diff(h['frame_index'])==1);triples=links[1:]&links[:-1];leg=native_leg(z['anatomical_keypoints'])
 root=(z['anatomical_keypoints'][:,0]+translation)/leg
 v=np.diff(root,axis=0)/dt[:,None];acc=np.diff(v,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
 out['root_speed_p95_leg_s']=q(np.linalg.norm(v[links],axis=-1));out['root_acc_p95_leg_s2']=q(np.linalg.norm(acc[triples],axis=-1))
 feet=(z['foot_keypoints']+translation[:,None])/leg;fv=np.linalg.norm(np.diff(feet,axis=0)/dt[:,None,None],axis=-1);weights=np.minimum(cw[1:],cw[:-1])*links[:,None]
 out['assumed_contact_speed_weighted_leg_s']=float((fv*weights).sum()/weights.sum()) if weights.sum()>0 else None
 out['root_change_p50_leg']=float(np.median(np.linalg.norm(translation-z['translation'],axis=-1))/leg)
 out['root_change_p95_leg']=q(np.linalg.norm(translation-z['translation'],axis=-1)/leg)
 return out,uv

def save_geometry(dest,z,translation,uv,cw,normal,offset):
 points=np.concatenate([z['keypoints'],z['foot_keypoints']],1).astype(float)+translation[:,None]
 projected=points[...,:2]/points[...,2:]*float(z['focal'])+z['principal']
 signed=((z['foot_keypoints']+translation[:,None])*normal).sum(-1)+offset
 atomic_npz(dest,**{**z,'translation':translation,'projected':uv,'projected_stabilized':projected,'normal':normal,'foot_signed_height':signed,'contact_weight':cw,'pilot_support_weights':cw,'pilot_support_normal':normal,'pilot_support_offset':np.array(offset),'pilot_support_available':np.array(cw.sum()>0),'root_pilot_schema_version':np.array(2)})

def fit(rec,folder,steps):
 import torch
 torch.set_num_threads(1)
 cid=rec['id'];z=load(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz');r,c,s,h,w=init_inputs(cid)
 bg=z['background_transforms'];phase,_=phase_info(rec,h,bg);event=phase['events'];cw=support_weights(h,event,phase,bg)
 leg=native_leg(z['anatomical_keypoints']);N=len(z['keypoints']);tensor=lambda x:torch.tensor(x,dtype=torch.float64)
 basis=tensor(spline_basis(N,float(h['fps']),spacing=CONFIG['spacing_seconds']));initial=z['translation'].astype(float)/leg
 controls0=np.linalg.lstsq(basis.numpy(),initial,rcond=None)[0]
 points=tensor(np.concatenate([z['keypoints'],z['foot_keypoints']],1)/leg);feet=tensor(z['foot_keypoints']/leg)
 target=np.einsum('tij,tkj->tki',bg,np.concatenate([h['keypoints'],np.ones((*h['keypoints'].shape[:2],1))],-1))[...,:2]
 target=tensor(np.nan_to_num(np.concatenate([target[:,:17],target[:,20:26]],1)));sc=scores(h);weight=np.concatenate([sc[:,:17]**2,sc[:,20:26]**2*.35],1);weight[:,:5]*=.2;weight=tensor(weight)
 height=max(100.,float(np.median(h['bbox'][:,3]-h['bbox'][:,1])));focal=float(z['focal']);principal=tensor(z['principal']);dt=tensor(np.diff(h['source_time']))
 links=(dt>0)&(dt<.1)&tensor(np.diff(h['frame_index'])==1).bool();triples=links[1:]&links[:-1];dtmid=(dt[1:]+dt[:-1])/2
 contact=tensor(cw);pair_contact=torch.minimum(contact[1:],contact[:-1])*links[:,None]
 result=[];baseline,uv=metrics(z,h,bg,z['translation'].astype(float),cw)
 result.append(dict(id=cid,player=rec['player_name'],cohort=rec['cohort'],method='current_gvhmr_f4',**baseline))
 for method in ['root_temporal_only','root_support_transfer']:
  dest=folder/method/f'{cid}.npz'
  if dest.exists():
   cached=load(dest);meta=json.loads(dest.with_suffix('.json').read_text())
   if meta['config']!=CONFIG or meta['initial_fit_sha256']!=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz'):raise ValueError('Changed config/input needs a new versioned output directory: '+cid)
   if int(cached.get('root_pilot_schema_version',0))<2:
    selected=meta['hypotheses'][meta['selected']];tr=cached['translation'];_,uv=metrics(z,h,bg,tr,cw)
    save_geometry(dest,z,tr,uv,cw,np.array(selected['plane_normal']),selected['plane_offset_model_m'])
   meta['support_plane_status']='hypothesis_only' if cw.sum()>0 else 'unknown_no_contact'
   save(dest.with_suffix('.json'),meta);result+=meta['rows'];continue
  H=1 if method=='root_temporal_only' else 3;normal=tensor([[0,-np.cos(a),-np.sin(a)] for a in CONFIG['plane_pitch_candidates_radians'][:H]])
  controls=torch.nn.Parameter(tensor(np.repeat(controls0[None],H,0)))
  signed0=((feet[None]+tensor(initial)[None,:,None])*normal[:,None,None]).sum(-1).numpy();offset=[]
  good=cw>.03
  for x in signed0:offset.append(-float(np.median(x[good])) if good.any() else -float(np.quantile(x,.1)))
  floor=torch.nn.Parameter(tensor(offset));optimizer=torch.optim.Adam([controls,floor],lr=.02)
  def evaluate():
   tr=torch.einsum('nk,hkc->hnc',basis,controls);p=points[None]+tr[:,:,None];uv=p[...,:2]/p[...,2:].clamp_min(.2)*focal+principal
   error=(uv-target)/height;image=((torch.sqrt(error.square().sum(-1)+.005**2)-.005)*weight).sum((1,2))/weight.sum().clamp_min(1)
   v=(tr[:,1:]-tr[:,:-1])/dt[None,:,None];acc=(v[:,1:]-v[:,:-1])/dtmid[None,:,None]
   smooth=(acc[:,triples]/30).square().mean((1,2));prior=(tr-tensor(initial)).square().mean((1,2));absolute_feet=feet[None]+tr[:,:,None]
   signed=(absolute_feet*normal[:,None,None]).sum(-1)+floor[:,None,None]
   contact_height=((torch.sqrt(signed.square()+.03**2)-.03)*contact).sum((1,2))/contact.sum().clamp_min(1)
   fv=(absolute_feet[:,1:]-absolute_feet[:,:-1])/dt[None,:,None,None];nv=(fv*normal[:,None,None]).sum(-1);tv=fv-nv[...,None]*normal[:,None,None]
   sliding=((torch.sqrt((tv/3).square().sum(-1)+.03**2)-.03)*pair_contact).sum((1,2))/pair_contact.sum().clamp_min(1)
   penetration=torch.relu(-signed-.03).square().mean((1,2))
   total=image+CONFIG['root_acc_weight']*smooth+CONFIG['root_prior_weight']*prior
   if method=='root_support_transfer' and contact.sum()>0:total=total+CONFIG['contact_normal_weight']*contact_height+CONFIG['contact_tangent_weight']*sliding+CONFIG['penetration_weight']*penetration
   return total,tr,dict(image=image,root_acc=smooth,contact_height=contact_height,sliding=sliding)
  start=time.monotonic()
  for i in range(steps):
   optimizer.param_groups[0]['lr']=.02*(.05+.95*(1-i/max(1,steps-1))**2);optimizer.zero_grad();loss,tr,parts=evaluate()
   if not torch.isfinite(loss).all():raise ValueError('Nonfinite root loss: '+cid)
   loss.sum().backward();torch.nn.utils.clip_grad_norm_([controls,floor],10);optimizer.step()
  with torch.no_grad():loss,tr,parts=evaluate()
  hypotheses=[]
  for j in range(H):
   translation=tr[j].numpy()*leg;m,uv=metrics(z,h,bg,translation,cw);hypotheses.append(dict(index=j,objective=float(loss[j]),plane_normal=normal[j].tolist(),plane_offset_model_m=float(floor[j])*leg,metrics=m))
  selected=int(loss.argmin());translation=tr[selected].numpy()*leg;m,uv=metrics(z,h,bg,translation,cw)
  # Translation cannot change same-frame articulation or interankle distance.
  native=z['anatomical_keypoints'].astype(float);original=native+z['translation'].astype(float)[:,None];changed=native+translation[:,None]
  interankle_error=float(np.max(abs(np.linalg.norm(changed[:,7]-changed[:,8],axis=-1)-np.linalg.norm(original[:,7]-original[:,8],axis=-1)))/leg)
  assert interankle_error<1e-10
  row=dict(id=cid,player=rec['player_name'],cohort=rec['cohort'],method=method,**m,interankle_change_max_leg=interankle_error,articulation_changed=False,source_gate_pass=(m['source_p50_px']<=baseline['source_p50_px']+2 and m['wrist_p95_px']<=baseline['wrist_p95_px']+2),contact_available=bool(cw.sum()>0),seconds=time.monotonic()-start)
  result.append(row);save_geometry(dest,z,translation,uv,cw,normal[selected].numpy(),float(floor[selected])*leg)
  save(dest.with_suffix('.json'),dict(rows=[row],phase=phase,hypotheses=hypotheses,selected=selected,support_plane_status='hypothesis_only' if cw.sum()>0 else 'unknown_no_contact',selection_semantics='Minimum training objective, no claim that plane or depth is true',config=CONFIG,initial_fit_sha256=sha(OUT/'fits_strong/gvhmr_temporal_f4'/f'{cid}.npz')))
  print('ROOT PILOT',cid,method,'source',round(m['source_p50_px'],2),'wrist95',round(m['wrist_p95_px'],2),'acc95',round(m['root_acc_p95_leg_s2'],2),'gate',row['source_gate_pass'],flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['pilot','confirmation','all'],default='pilot');ap.add_argument('--steps',type=int,default=CONFIG['steps']);a=ap.parse_args();folder=OUT/'support_transfer_root_v1';folder.mkdir(exist_ok=True)
 if a.steps!=CONFIG['steps']:raise ValueError('Changed step budget needs a new versioned output directory')
 records=sample('tune',True)
 for i,rec in enumerate(records):rec['cohort']='pilot' if i in PILOT else 'confirmation'
 protocol=folder/'protocol.json'
 if not protocol.exists():save(protocol,dict(config=CONFIG,pilot=[r['id'] for r in records if r['cohort']=='pilot'],confirmation=[r['id'] for r in records if r['cohort']=='confirmation'],frozen_test24_used=False,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),code_sha256=sha(__file__)))
 save(folder/'execution.json',dict(config=CONFIG,code_sha256=sha(__file__),cohort=a.cohort,steps=a.steps,implementation_corrections=['Zero-weight missing 2D targets replaced by finite placeholders to prevent NaN times zero. Abort on any nonfinite loss.'],protocol_sha256=sha(protocol)))
 records=[r for r in records if a.cohort=='all' or r['cohort']==a.cohort];result=[]
 for rec in records:result+=fit(rec,folder,a.steps)
 save(folder/f'{a.cohort}_metrics.json',result);csvwrite(folder/f'{a.cohort}_metrics.csv',result)
 summary=[]
 for method in ['current_gvhmr_f4','root_temporal_only','root_support_transfer']:
  rr=[r for r in result if r['method']==method];row=dict(method=method,clips=len(rr))
  for k in ['source_p50_px','wrist_p95_px','root_acc_p95_leg_s2','root_speed_p95_leg_s','assumed_contact_speed_weighted_leg_s','root_change_p95_leg']:
   v=[r[k] for r in rr if r[k] is not None];row[k+'_clip_median']=float(np.median(v)) if v else None
  if method!='current_gvhmr_f4':row['source_gate_pass_count']=sum(r['source_gate_pass'] for r in rr);row['contact_available_count']=sum(r['contact_available'] for r in rr)
  summary.append(row)
 save(folder/f'{a.cohort}_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 from runtime import ensure_runtime
 ensure_runtime();main()
