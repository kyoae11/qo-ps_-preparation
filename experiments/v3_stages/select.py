"""Select from validation only and record every tested configuration."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from modeling import metrics
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root); rows=[]
required=json.loads((r/'search_jobs.json').read_text())
folders=[f"{j['stage']}_{j['kind']}_{j['variant']}_w{j['width']}_lr{j['lr']}_s{j['seed']}" for j in required]
assert len(set(folders))==len(folders),'Duplicate search candidates'
for folder in folders:
 f=r/'results'/folder/'result.json'
 assert f.exists(),f'Incomplete search candidate: {folder}'
 z=json.loads(f.read_text());c=z['config'];rows.append(dict(stage=c['stage'],kind=c['kind'],variant=c['variant'],width=c['width'],lr=c['lr'],seed=c['seed'],mae=z['metrics']['mae'],folder=f.parent.name))
d=pd.DataFrame(rows);assert len(d)==len(required),f'{len(d)}/{len(required)} complete'
groups=[]
for keys,g in d.groupby(['stage','kind','variant','width','lr']):
 expected=1 if keys[1]=='tree' else 3
 assert len(g)==expected
 groups.append(dict(stage=str(keys[0]),kind=str(keys[1]),variant=str(keys[2]),width=int(keys[3]),lr=float(keys[4]),mae=float(g.mae.mean()),folders=dict(zip(g.seed.astype(str),g.folder))))
summary=pd.DataFrame([{k:v for k,v in z.items() if k!='folders'} for z in groups]);summary.to_csv(r/'validation_search.csv',index=False)
best={}
for z in groups:
 key='_'.join([z['stage'],z['kind'],z['variant']])
 if key not in best or (z['mae'],z['width'],z['lr'])<(best[key]['mae'],best[key]['width'],best[key]['lr']):best[key]=z
net=min([best[k] for k in ['network_afil_full','network_afil_gated','network_afil_full_id','network_afil_gated_id']],key=lambda z:(z['mae'],z['variant']))
x=pd.read_pickle(r/'cache/validation_x.pkl');y=pd.read_pickle(r/'cache/validation_y.pkl')
mech=metrics(y.ready_time.to_numpy(),x.mechanism_ready.to_numpy())['mae'];candidates=[dict(kind='mechanistic',variant='full',stage='ready',mae=mech,folders={})]+[best[k] for k in ['ready_mlp_full','ready_residual_full','ready_tree_full']]
ready=min(candidates,key=lambda z:(round(z['mae'],6),{'mechanistic':0,'tree':1,'mlp':2,'residual':3}[z['kind']]))
base=min([best['network_afil_base'],best['network_afil_base_id']],key=lambda z:z['mae']);gain=1-net['mae']/base['mae']
selection=dict(candidate_runs_evaluated=len(d),network=net,network_baseline=base,ready=ready,execution='base_execution_time / processing_capacity',best_by_family=best,ready_mechanistic_validation_mae=mech,network_validation_relative_improvement=gain,network_goal_10pct_met=gain>=.10,selection_partitions=['train','validation'],test_used_for_selection=False)
(r/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))
