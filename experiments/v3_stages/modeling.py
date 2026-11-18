"""Invocation-level AFIL adaptation with explicit feature contracts."""
import json
import numpy as np
import torch
from torch import nn
BASE=['service_type','request_bytes','response_bytes','digital_processing_workload','user_ap_id_at_t0','entity_ap_id_at_t0','terminal_bandwidth_limit','user_bandwidth_est_at_t0','entity_bandwidth_est_at_t0']
CAP=['entity_type','device_bandwidth_limit','digital_processing_capacity']
STATE=['digital_queue_length_at_t0','digital_busy_at_t0','digital_activity_elapsed_at_t0','digital_busy_fraction_10s_at_t0']
IDENTITY=['user_identity','service_identity']
CAT=set(IDENTITY)|{'service_type','entity_type','entity_status_at_t0','activity_purpose_at_t0','user_ap_id_at_t0','entity_ap_id_at_t0'}
KEYS={'run_id','world_id','split','invocation_id','user_id','service_id','entity_id'}
MECH={'mechanism_network','mechanism_ready','mechanism_execution','mechanism_fallback'}
TARGETS=['network_time','ready_time','execution_time','completion_time']
def columns(frame,stage,variant):
 if stage=='network':
  ids=IDENTITY if variant.endswith('_id') else []
  core=variant.removesuffix('_id')
  if core=='base':return BASE+ids
  if core=='no_cap':return BASE+STATE+ids
  if core=='no_state':return BASE+CAP+ids
  return BASE+CAP+STATE+ids
 if stage=='execution':return ['base_execution_time','processing_capacity']
 cols=sorted(set(frame.columns)-KEYS)
 if variant=='raw':cols=[c for c in cols if c not in MECH]
 return cols
class Preprocessor:
 def __init__(self,cols):self.cols=cols
 def frame(self,df):
  if set(self.cols)&set(IDENTITY):
   df=df.copy()
   for name,key in zip(IDENTITY,['user_id','service_id']):df[name]=df.world_id.astype(str)+'|'+df[key].astype(str)
  return df
 def fit(self,df):
  df=self.frame(df)
  self.num=[c for c in self.cols if c not in CAT];self.cat=[c for c in self.cols if c in CAT]
  v=df[self.num].to_numpy(float);v[~np.isfinite(v)]=np.nan
  self.med=np.array([np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else 0 for a in v.T]);v=np.where(np.isfinite(v),v,self.med)
  self.mean=v.mean(0);self.scale=v.std(0);self.scale[self.scale<1e-8]=1
  self.vocab={c:{v:i+1 for i,v in enumerate(sorted(df[c].dropna().astype(str).unique()))} for c in self.cat}
  return self
 def transform(self,df):
  df=self.frame(df)
  v=df[self.num].to_numpy(float);missing=~np.isfinite(v);v=(np.where(missing,self.med,v)-self.mean)/self.scale
  n=np.stack([v,missing.astype(float)],axis=-1).astype('float32')
  c=np.column_stack([df[k].astype(str).map(self.vocab[k]).fillna(0).to_numpy('int64') for k in self.cat]) if self.cat else np.zeros((len(df),0),dtype='int64')
  return n,c
 def dense(self,df):
  n,c=self.transform(df);blocks=[n.reshape(len(df),-1)]
  for j,k in enumerate(self.cat):blocks.append(np.eye(len(self.vocab[k])+1,dtype='float32')[c[:,j]])
  return np.concatenate(blocks,1)
class AFIL(nn.Module):
 def __init__(self,prep,dim=32,variant='full'):
  super().__init__();variant=variant.removesuffix('_id');self.variant=variant;self.num=prep.num;self.cat=prep.cat
  self.nproj=nn.Parameter(torch.randn(len(self.num),2,dim)*.05);self.nb=nn.Parameter(torch.zeros(len(self.num),dim))
  self.emb=nn.ModuleList([nn.Embedding(len(prep.vocab[k])+1,dim,padding_idx=0 if k in IDENTITY else None) for k in self.cat]);names=self.num+self.cat
  self.bi=[i for i,k in enumerate(names) if k in BASE+IDENTITY];self.ei=[i for i,k in enumerate(names) if k not in BASE+IDENTITY]
  self.projection=nn.Linear(dim,4*32,bias=False);self.norm=nn.LayerNorm(32)
  count=len(self.bi) if variant=='gated' else len(names)
  self.dnn=nn.Sequential(nn.Linear(count*32,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
  if variant=='gated':
   self.entity_encoder=nn.Sequential(nn.Linear(len(self.ei)*dim,128),nn.ReLU(),nn.LayerNorm(128));self.gate=nn.Linear(256,128);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2)
  self.output=nn.Linear(128,1)
 def forward(self,n,c):
  fields=[torch.einsum('bfi,fid->bfd',n,self.nproj)+self.nb]
  if self.emb:fields.append(torch.stack([e(c[:,i]) for i,e in enumerate(self.emb)],dim=1))
  allfields=torch.cat(fields,1);f=allfields[:,self.bi] if self.variant=='gated' else allfields
  p=self.projection(f).reshape(len(f),-1,4,32).transpose(1,2);a=(p@p.transpose(-1,-2))/(32**.5);ctx=torch.softmax(a,-1)@p
  interaction=.5*(ctx.sum(1).square()-ctx.square().sum(1));h=self.dnn(self.norm(interaction).flatten(1))
  if self.variant=='gated':
   e=self.entity_encoder(allfields[:,self.ei].flatten(1));h=h+torch.sigmoid(self.gate(torch.cat([h,e],1)))*e
  return self.output(h).squeeze(-1)
class MLP(nn.Module):
 def __init__(self,d,width):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Linear(width,width//2),nn.ReLU(),nn.Linear(width//2,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def metrics(y,p):
 e=p-y
 return dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),p95_absolute_error=float(np.quantile(np.abs(e),.95)),bias=float(e.mean()),n=len(y))
