"""Build t0-only feature cache. Labels are separate and never used in forecasts.
Only training targets determine the explicit failure fallback.
"""
import argparse, gzip, hashlib, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from qops_v3.world import World
from qops_v3.mechanistic import predict

TARGETS=['network_time','ready_time','execution_time','completion_time']
def load_world(root,row):
 p=root/'worlds'/row['world_id']; cfg=json.loads((root/row['path']/'metadata/config.json').read_text()); m=json.loads((p/'map.json').read_text())
 tables={n:pd.read_csv(p/(n+'.csv')).to_dict('records') for n in ('users','services','static_entities','dynamic_entities','charge_points')}
 entities=[]
 for typ in ('static','dynamic'):
  for e in tables[typ+'_entities']: e['entity_type']=typ; entities.append(e)
 return World(row['world_id'],0,0,cfg,tables['users'],tables['services'],entities,tables['charge_points'],{tuple(c) for c in m['blocked_cells']},{int(k):v for k,v in m['eligible_users'].items()})
def one(args):
 root,out,row,fallback=args;root=Path(root);out=Path(out);d=root/row['path']; dest=out/'runs'/row['run_id'];dest.mkdir(parents=True,exist_ok=True)
 x=pd.read_csv(d/'data/model_inputs.csv'); y=pd.read_csv(d/'data/labels.csv')
 assert x.invocation_id.equals(y.invocation_id)
 mask=(~y.warmup)&(y.terminal_status=='completed'); x=x.loc[mask].reset_index(drop=True); y=y.loc[mask,TARGETS].reset_index(drop=True)
 if (dest/'done.json').exists():return json.loads((dest/'done.json').read_text())
 with gzip.open(d/'logs/queue_snapshots.jsonl.gz','rt') as f:q={int(z['invocation_id']):z for z in map(json.loads,f)}
 w=load_world(root,row); pp=[]; falls=[]
 for item in x.to_dict('records'):
  p,b=predict(w,item,q[int(item['invocation_id'])],fallback);pp.append(p[:3]);falls.append(int(b))
 pp=np.array(pp);x['mechanism_network']=pp[:,0];x['mechanism_ready']=pp[:,1];x['mechanism_execution']=pp[:,2];x['mechanism_fallback']=falls
 x['charge_power_config']=w.config['energy']['charge_power_w'];x['base_network_config']=w.config['network']['base_s']
 for key in ['run_id','world_id','split']:x[key]=row[key]
 assert np.isfinite(pp).all()
 assert np.allclose(y.completion_time,y.network_time+y.ready_time+y.execution_time,rtol=0,atol=1e-8)
 assert np.allclose(y.execution_time,x.base_execution_time/x.processing_capacity,rtol=0,atol=1e-8)
 x.to_pickle(dest/'x.pkl');y.to_pickle(dest/'y.pkl')
 info=dict(run_id=row['run_id'],split=row['split'],calls=len(x),fallbacks=sum(falls),excluded=int((~mask).sum()))
 (dest/'done.json').write_text(json.dumps(info));return info

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 idx=pd.read_csv(root/'run_index.csv');assert idx.run_id.is_unique
 vals=[]
 for row in idx[idx.split=='train'].itertuples():
  y=pd.read_csv(root/row.path/'data/labels.csv');vals.extend(y.loc[(~y.warmup)&(y.terminal_status=='completed'),'ready_time'].tolist())
 fallback=float(np.mean(vals)); rows=idx.to_dict('records')
 # Prepare train/validation first; evaluation partitions remain inaccessible to training entrypoints.
 rows.sort(key=lambda r:({'train':0,'validation':1}.get(r['split'],2),r['run_id']))
 reports=[]
 with ProcessPoolExecutor(max_workers=2) as ex:
  for i,r in enumerate(ex.map(one,[(str(root),str(out),r,fallback) for r in rows])):
   reports.append(r)
   if i%10==0:print('prepared',i+1,'/',len(rows),r['run_id'],flush=True)
 for split in idx.split.unique():
  rr=idx[idx.split==split];xs=[pd.read_pickle(out/'runs'/r.run_id/'x.pkl') for r in rr.itertuples()];ys=[pd.read_pickle(out/'runs'/r.run_id/'y.pkl') for r in rr.itertuples()]
  pd.concat(xs,ignore_index=True).to_pickle(out/(split+'_x.pkl'));pd.concat(ys,ignore_index=True).to_pickle(out/(split+'_y.pkl'))
 (out/'preparation_report.json').write_text(json.dumps(dict(fallback_train_mean=fallback,runs=reports),indent=2))
 print('PREPARE_COMPLETE',flush=True)
if __name__=='__main__':main()
