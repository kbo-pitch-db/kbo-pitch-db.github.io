"""Weak source-contour constraint, using selection RGB keyframes only.

SAM is a same-RGB model observation, not GT. No depth from silhouette claim.
Exact SMPL LBS at extra mesh vertices; no fake joint/shape displacement.
"""
from common import *
from pitch_study_fit import StudyFitter,static_rigid_transform

class MaskFitter(StudyFitter):
 def configure_rgb(self,cid):
  import torch,cv2
  from scipy.spatial import cKDTree
  r,c,s,h,w=init_inputs(cid);bg=load(PREVIOUS/'background'/f'{cid}.npz')['full_transforms'];ids=[];targets=[];vertices=[];self.rgb_height=max(100,float(np.nanmedian(h['bbox'][:,3]-h['bbox'][:,1])))
  for order,path in enumerate(sorted((OUT/'observations'/cid).glob('*.npz'))):
   if order%3==2:continue
   i=int(path.stem);obs=load(path);contours,_=cv2.findContours(obs['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   if not contours:continue
   contour=max(contours,key=cv2.contourArea).reshape(-1,2);edge=contour[np.linspace(0,len(contour)-1,64).round().astype(int)]+obs['crop_origin'];native=load(OUT/'raw/nlf_frames'/cid/f'{i:05d}_f4.npz')['vertices2d'][0];dist,vi=cKDTree(native).query(edge);vertices.extend(vi[dist<self.rgb_height*.08]);target=(bg[i]@np.c_[edge,np.ones(len(edge))].T).T[:,:2];ids.append(i);targets.append(target)
  extra=np.unique(vertices);extra=extra[np.linspace(0,len(extra)-1,min(192,len(extra))).round().astype(int)] if len(extra) else np.array([],int);base=(self.body.J_regressor_wham[:17].abs().sum(0)>0).nonzero().flatten();keep=torch.unique(torch.cat([base,torch.as_tensor(self.foot_vertices,device=self.device),torch.tensor(extra,device=self.device)]),sorted=True);self.selected_vertices=keep;self.selected_regressor=self.body.J_regressor_wham[:17,keep];self.selected_weights=self.body.lbs_weights[keep];self.selected_posedirs=self.body.posedirs.reshape(207,-1,3)[:,keep].reshape(207,-1);self.selected_feet=torch.as_tensor([int((keep==i).nonzero()[0]) for i in self.foot_vertices],device=self.device);self.rgb_ids=torch.tensor(ids,device=self.device);self.rgb_targets=torch.tensor(np.array(targets),device=self.device,dtype=torch.float32)/self.rgb_height
 def model(self,rotations,betas):
  torch=self.torch;count=len(rotations);feature=(rotations[:,1:]-torch.eye(3,device=self.device)).reshape(count,207);posed=self.rest_vertices[None]+(feature@self.selected_posedirs).reshape(count,-1,3);joints,A=static_rigid_transform(rotations,self.rest_joints[None].expand(count,-1,-1),self.body.parents);transforms=(self.selected_weights[None]@A.reshape(count,24,16)).reshape(count,-1,4,4);homogeneous=torch.cat([posed,torch.ones((*posed.shape[:2],1),device=self.device)],-1);vertices=(transforms@homogeneous[:,:,:,None])[:,:,:3,0];body=self.selected_regressor[None]@vertices;root=(body[:,11:12]+body[:,12:13])*.5;self.rgb_vertices=vertices-root;return body-root,joints-root,vertices[:,self.selected_feet]-root
 def rgb_loss(self,tr,f,principal):
  torch=self.torch;H,N=tr.shape[:2];vv=self.rgb_vertices.reshape(H,N,-1,3)[:,self.rgb_ids]+tr[:,self.rgb_ids,None];uv=vv[...,:2]/vv[...,2:].clamp_min(.5)*f[:,None,None,None]+principal;uv=uv/self.rgb_height;target=self.rgb_targets[None].expand(H,-1,-1,-1);distance=torch.cdist(target.flatten(0,1),uv.flatten(0,1)).min(-1).values.reshape(H,len(self.rgb_ids),-1);return ((distance.square()+.005**2).sqrt()-.005).clamp_max(.08).mean((1,2))
