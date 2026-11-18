"""Safe loading: respect run splits and remove warmup before model fitting."""
from pathlib import Path
import pandas as pd
from .schema import INPUT_COLUMNS

TARGET_COLUMNS=['network_time','ready_time','execution_time']
JOIN_ONLY_COLUMNS=['invocation_id','user_id','service_id','entity_id']


def load_split(root,split='train',keep_identifiers=False,return_report=False):
    root=Path(root);index=pd.read_csv(root/'run_index.csv');xs=[];ys=[];keys=[];excluded=[]
    for row in index.loc[index.split==split].itertuples():
        path=root/row.path
        x=pd.read_csv(path/'data/model_inputs.csv');y=pd.read_csv(path/'data/labels.csv')
        assert set(x.columns)==INPUT_COLUMNS
        assert x.invocation_id.equals(y.invocation_id)
        mask=(~y.warmup)&(y.terminal_status=='completed')
        excluded.append(dict(run_id=row.run_id,total=len(y),warmup=int(y.warmup.sum()),infeasible=int((~y.warmup & (y.terminal_status=='infeasible')).sum()),censored=int((~y.warmup & (y.terminal_status=='censored')).sum()),included=int(mask.sum())))
        key=x.loc[mask,['invocation_id','entity_id']].copy();key['run_id']=row.run_id;key['world_id']=row.world_id
        xs.append(x.loc[mask]);ys.append(y.loc[mask,TARGET_COLUMNS+['completion_time']]);keys.append(key)
    if not xs:raise ValueError(f'No runs in split {split}')
    x=pd.concat(xs,ignore_index=True);y=pd.concat(ys,ignore_index=True);keys=pd.concat(keys,ignore_index=True)
    if not keep_identifiers:x=x.drop(columns=JOIN_ONLY_COLUMNS)
    x.attrs['exclusion_report']=excluded
    return (x,y,keys,excluded) if return_report else (x,y,keys)
