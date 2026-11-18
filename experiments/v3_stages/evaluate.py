"""Frozen-checkpoint evaluation; no fitting and no model selection."""
import argparse,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from modeling import *

def infer(folder,x,device):
 z=json.loads((folder/'result.json').read_text());c=z['config']
 with open(folder/'preprocessor.pkl','rb') as f:p=pickle.load(f)
 if c['kind']=='tree':
  with open(folder/'model.pkl','rb') as f:m=pickle.load(f)
  pred=m.predict(p.dense(x))
 else:
  if c['kind']=='afil':
   m=AFIL(p,c['width'],c['variant'] if c.get('architecture','auto')=='auto' else c['architecture']);n,cat=p.transform(x);inputs=[torch.from_numpy(n),torch.from_numpy(cat)]
  else:
   dense=p.dense(x);m=MLP(dense.shape[1],c['width']);inputs=[torch.from_numpy(dense)]
  m.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True));m.to(device).eval()
  with torch.no_grad():pred=torch.cat([m(*[a[i:i+4096].to(device) for a in inputs]).cpu() for i in range(0,len(x),4096)]).numpy()
 pred=pred*z['target_scale']+z['target_mean']
 if c['kind']=='residual':pred=pred+x.mechanism_ready.to_numpy()
 return np.maximum(0,pred)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root);out=r/'evaluation';out.mkdir(exist_ok=True);sel=json.loads((r/'selection.json').read_text());torch.set_num_threads(4);device='cuda:0' if torch.cuda.is_available() else 'cpu'
 if device!='cpu':assert '5880' in torch.cuda.get_device_name(0);torch.cuda.set_per_process_memory_fraction(.15,0)
 frozen={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for pat in ['code/**/*.py','code/*.sh','*_execution_schedule.json','run_metadata.json','data_source_hashes.json','environment_requirements.txt','cache/*_x.pkl','cache/*_y.pkl','selection.json','results/*/result.json','results/*/model.pt','results/*/model.pkl','results/*/preprocessor.pkl'] for f in r.glob(pat)}
 freeze=r/'evaluation_freeze.json'
 if freeze.exists():assert json.loads(freeze.read_text())==frozen,'Frozen artifacts changed'
 else:freeze.write_text(json.dumps(frozen,indent=2))
 families=dict(sel['best_by_family'])
 for core in ['no_cap','no_state']:
  variant=core+('_id' if sel['network']['variant'].endswith('_id') else '')
  folders={}
  for f in (r/'results').glob(f'network_afil_{variant}_*/result.json'):
   z=json.loads(f.read_text());folders[str(z['config']['seed'])]=f.parent.name
  assert len(folders)==3;families['network_afil_'+variant]=dict(folders=folders,stage='network')
 allrows=[];slices=[];tails=[];worst_cases=[];splits=[p.name[:-6] for p in (r/'cache').glob('*_x.pkl') if p.name!='train_x.pkl'];paired=[]
 for split in sorted(splits):
  x=pd.read_pickle(r/'cache'/(split+'_x.pkl'));y=pd.read_pickle(r/'cache'/(split+'_y.pkl'));store={};dest=out/split;dest.mkdir(exist_ok=True)
  def record(name,seed,stage,p):
   # Preserve exact float32 predictions as float64 decimal values in CSV.
   # Pandas' short float32 formatting otherwise changes low-error metrics on reload.
   p=np.asarray(p,dtype=np.float64)
   actual=y[stage].to_numpy();df=x[['run_id','world_id','invocation_id','entity_id','entity_type']].copy();df['actual']=actual;df['pred']=p;df.to_csv(dest/f'{name}_s{seed}_{stage}.csv.gz',index=False)
   for group in ['all','static','dynamic']:
    mask=np.ones(len(x),bool) if group=='all' else (x.entity_type==group).to_numpy()
    if mask.any():allrows.append(dict(split=split,model=name,seed=seed,stage=stage,group=group,weighting='call',**metrics(actual[mask],p[mask])))
   if name in ['selected_pipeline','mechanistic']:
    ae=np.abs(p-actual)
    tails.append(dict(split=split,model=name,seed=seed,stage=stage,n=len(x),median_absolute_error=float(np.median(ae)),p99_absolute_error=float(np.quantile(ae,.99)),p999_absolute_error=float(np.quantile(ae,.999)),max_absolute_error=float(ae.max()),fraction_over_1s=float((ae>1).mean()),fraction_over_10s=float((ae>10).mean()),fraction_over_60s=float((ae>60).mean())))
    top=np.argsort(ae)[-5:][::-1];cases=df.iloc[top].copy();cases['split']=split;cases['model']=name;cases['seed']=seed;cases['stage']=stage;cases['absolute_error']=ae[top]
    for feature in ['digital_queue_length_at_t0','digital_busy_at_t0','queue_length_at_t0','entity_status_at_t0']:cases[feature]=x.iloc[top][feature].to_numpy()
    worst_cases.append(cases)
   if stage=='network_time':
    for busy in [0,1]:
     mask=(x.digital_busy_at_t0.astype(int)==busy).to_numpy()
     if mask.any():slices.append(dict(split=split,model=name,seed=seed,stage=stage,context='digital_busy_at_t0',value=busy,**metrics(actual[mask],p[mask])))
   if stage=='ready_time':
    for waiting in [0,1]:
     mask=((x.queue_length_at_t0>0).astype(int)==waiting).to_numpy()
     if mask.any():slices.append(dict(split=split,model=name,seed=seed,stage=stage,context='physical_queue_nonempty_at_t0',value=waiting,**metrics(actual[mask],p[mask])))
   df['ae']=np.abs(p-actual);df['se']=(p-actual)**2;by=df.groupby(['world_id','entity_id'])
   allrows.append(dict(split=split,model=name,seed=seed,stage=stage,group='all',weighting='device',mae=float(by.ae.mean().mean()),rmse=float(np.sqrt(by.se.mean().mean())),p95_absolute_error=float(by.ae.quantile(.95).mean()),bias=float(df.assign(error=p-actual).groupby(['world_id','entity_id']).error.mean().mean()),n=len(x)))
  for family,z in families.items():
   stage={'network':'network_time','ready':'ready_time','execution':'execution_time','total':'completion_time'}[z['stage']]
   for seed,folder in z['folders'].items():
    p=infer(r/'results'/folder,x,device);store[(family,int(seed))]=p;record(family,int(seed),stage,p)
  n=x.mechanism_network.to_numpy();rd=x.mechanism_ready.to_numpy();ex=x.mechanism_execution.to_numpy()
  for stage,p in [('network_time',n),('ready_time',rd),('execution_time',ex),('completion_time',n+rd+ex)]:record('mechanistic',0,stage,p)
  nf='network_afil_'+sel['network']['variant'];rf='ready_'+sel['ready']['kind']+'_full'
  for seed in [11,22,33]:
   pn=store[(nf,seed)];pr=rd if sel['ready']['kind']=='mechanistic' else store[(rf,11 if sel['ready']['kind']=='tree' else seed)]
   for stage,p in [('network_time',pn),('ready_time',pr),('execution_time',ex),('completion_time',pn+pr+ex)]:record('selected_pipeline',seed,stage,p)
   pure=store[('network_mlp_full',seed)]+store[('ready_mlp_raw',seed)]+store[('execution_mlp_raw',seed)];record('independent_mlp',seed,'completion_time',pure)
  # Paired bootstrap by complete run; aggregate over training seeds, not best seed.
  actual=y.network_time.to_numpy();baseline_family='network_afil_'+sel['network_baseline']['variant'];be=np.mean([np.abs(store[(baseline_family,s)]-actual) for s in [11,22,33]],axis=0);ee=np.mean([np.abs(store[(nf,s)]-actual) for s in [11,22,33]],axis=0)
  tab=pd.DataFrame(dict(run=x.run_id,b=be,e=ee));g=tab.groupby('run').agg(b=('b','sum'),e=('e','sum'),n=('b','size'));rng=np.random.default_rng(20261118);bs=[];es=[]
  for _ in range(1000):
   q=g.iloc[rng.integers(0,len(g),len(g))];bs.append(q.b.sum()/q.n.sum());es.append(q.e.sum()/q.n.sum())
  improvement=1-ee.mean()/be.mean();ci=np.quantile(1-np.array(es)/np.array(bs),[.025,.975]);paired.append(dict(split=split,baseline=baseline_family,enhanced=nf,base_mae=float(be.mean()),enhanced_mae=float(ee.mean()),relative_improvement=float(improvement),relative_improvement_ci95=ci.tolist(),runs=len(g)))
  pd.DataFrame(tails).to_csv(out/'tail_risk.csv',index=False);pd.concat(worst_cases,ignore_index=True).to_csv(out/'worst_cases.csv',index=False);pd.DataFrame(slices).to_csv(out/'context_slices.csv',index=False);pd.DataFrame(allrows).to_csv(out/'metrics.csv',index=False);(out/'network_paired_comparison.json').write_text(json.dumps(paired,indent=2));print('EVALUATED',split,flush=True)
 print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
