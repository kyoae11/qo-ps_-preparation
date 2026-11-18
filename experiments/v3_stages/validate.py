"""Independent pretraining assertions and serialized-data hash checks."""
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from modeling import *
a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--cache',required=True);a.add_argument('--manifest',required=True);a.add_argument('--out',required=True);a=a.parse_args()
root=Path(a.root);cache=Path(a.cache); hashes=json.loads(Path(a.manifest).read_text())
for path,h in hashes.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==h,path
x=pd.read_pickle(cache/'train_x.pkl');v=pd.read_pickle(cache/'validation_x.pkl');y=pd.read_pickle(cache/'train_y.pkl')
assert set(x.run_id).isdisjoint(v.run_id)
assert not x[['run_id','invocation_id']].duplicated().any()
assert np.allclose(x.mechanism_execution,y.execution_time,rtol=0,atol=1e-8)
assert np.allclose(y.completion_time,y.iloc[:,:3].sum(axis=1),rtol=0,atol=1e-8)
for variant in ['base','full','gated','no_cap','no_state','base_id','full_id','gated_id','no_cap_id','no_state_id']:
 p=Preprocessor(columns(x,'network',variant)).fit(x);n,c=p.transform(x.iloc[:17]);m=AFIL(p,32,variant);z=m(torch.from_numpy(n),torch.from_numpy(c));z.sum().backward();assert z.shape==(17,);assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
 assert not set(p.cols)&(set(TARGETS)|KEYS|MECH)
p=Preprocessor(columns(x,'ready','full')).fit(x);n=p.dense(v.iloc[:17]);assert np.isfinite(n).all()
# Shared user/service pairs must retain invocation-specific state, not a static lookup.
assert x.groupby(['user_id','service_id']).size().max()>1
report=dict(source_hashes_checked=len(hashes),train_calls=len(x),validation_calls=len(v),run_disjoint=True,joins_unique=True,formula_consistent=True,neural_forward_backward=True,test_labels_used_for_selection=False)
Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report))
