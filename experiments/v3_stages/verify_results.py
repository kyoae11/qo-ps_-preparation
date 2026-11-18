"""Recompute saved prediction metrics without importing the trainer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root);records=pd.read_csv(r/'evaluation/metrics.csv');checked=0
for keys, group in records.groupby(['split','model','seed','stage']):
 split,model,seed,stage=keys
 f=r/'evaluation'/split/f'{model}_s{seed}_{stage}.csv.gz';d=pd.read_csv(f)
 assert not d[['run_id','invocation_id']].duplicated().any();assert np.isfinite(d[['pred','actual']]).all().all();assert (d.pred>=0).all()
 for row in group.itertuples():
  z=d if row.group=='all' else d[d.entity_type==row.group]
  err=z.pred-z.actual
  if row.weighting=='call':
   computed=dict(mae=err.abs().mean(),rmse=np.sqrt((err**2).mean()),p95_absolute_error=err.abs().quantile(.95),bias=err.mean())
  else:
   z=z.assign(ae=err.abs(),se=err**2,error=err);devices=z.groupby(['world_id','entity_id'])
   computed=dict(mae=devices.ae.mean().mean(),rmse=np.sqrt(devices.se.mean().mean()),p95_absolute_error=devices.ae.quantile(.95).mean(),bias=devices.error.mean().mean())
  assert row.n==len(z)
  for name,value in computed.items():assert np.isclose(getattr(row,name),value,rtol=1e-7,atol=1e-9),(f,row.group,row.weighting,name)
 checked+=1
contexts=pd.read_csv(r/'evaluation/context_slices.csv')
for split,group in contexts.groupby('split'):
 x=pd.read_pickle(r/'cache'/(split+'_x.pkl'))
 for row in group.itertuples():
  d=pd.read_csv(r/'evaluation'/split/f'{row.model}_s{row.seed}_{row.stage}.csv.gz')
  assert d[['run_id','invocation_id']].equals(x[['run_id','invocation_id']])
  source=x.digital_busy_at_t0.astype(int) if row.context=='digital_busy_at_t0' else (x.queue_length_at_t0>0).astype(int)
  mask=(source==row.value).to_numpy();err=(d.pred-d.actual).to_numpy()[mask]
  assert row.n==len(err)
  for name,value in [('mae',np.abs(err).mean()),('rmse',np.sqrt(np.mean(err**2))),('p95_absolute_error',np.quantile(np.abs(err),.95))]:assert np.isclose(getattr(row,name),value,rtol=1e-7,atol=1e-9)
tails=pd.read_csv(r/'evaluation/tail_risk.csv')
for row in tails.itertuples():
 d=pd.read_csv(r/'evaluation'/row.split/f'{row.model}_s{row.seed}_{row.stage}.csv.gz');ae=(d.pred-d.actual).abs().to_numpy()
 computed=dict(median_absolute_error=np.median(ae),p99_absolute_error=np.quantile(ae,.99),p999_absolute_error=np.quantile(ae,.999),max_absolute_error=ae.max(),fraction_over_1s=(ae>1).mean(),fraction_over_10s=(ae>10).mean(),fraction_over_60s=(ae>60).mean())
 for name,value in computed.items():assert np.isclose(getattr(row,name),value,rtol=1e-7,atol=1e-9)
for folder in (r/'evaluation').iterdir():
 if not folder.is_dir():continue
 for seed in [11,22,33]:
  frames=[pd.read_csv(folder/f'selected_pipeline_s{seed}_{s}.csv.gz') for s in ['network_time','ready_time','execution_time','completion_time']]
  assert all(frames[0][['run_id','invocation_id']].equals(x[['run_id','invocation_id']]) for x in frames[1:])
  assert np.allclose(sum(x.pred.to_numpy() for x in frames[:3]),frames[3].pred,rtol=1e-9,atol=1e-8)
for name,h in json.loads((r/'evaluation_freeze.json').read_text()).items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h
for name,h in json.loads((r/'data_source_hashes.json').read_text()).items():assert hashlib.sha256((r/'data'/name).read_bytes()).hexdigest()==h
report=dict(prediction_tables_recomputed=checked,metric_rows_recomputed=len(records),context_rows_recomputed=len(contexts),tail_rows_recomputed=len(tails),nonnegative=True,stage_sum_consistent=True,source_data_unchanged=True,frozen_artifacts_unchanged=True)
(r/'evaluation/independent_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
