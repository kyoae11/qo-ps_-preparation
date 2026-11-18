"""Explicit tables, normalized shared worlds, and per-run simulation records."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import simpy
from .simulator import EVENTS,LABELS
from .schema import INPUT_COLUMNS
from .digital import FIELDS as DIGITAL_FIELDS


def json_write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)


def export_world(world,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    csv_write(path/'users.csv',world.users);csv_write(path/'services.csv',world.services)
    for typ in ('static','dynamic'):
        rows=[{k:v for k,v in e.items() if k!='entity_type'} for e in world.entities if e['entity_type']==typ]
        csv_write(path/f'{typ}_entities.csv',rows)
    csv_write(path/'charge_points.csv',world.chargers)
    csv_write(path/'service_entity_mapping.csv',[dict(service_id=e['entity_id'],entity_id=e['entity_id'],entity_type=e['entity_type']) for e in world.entities])
    json_write(path/'map.json',dict(world_id=world.world_id,width=world.width,height=world.height,resolution=world.resolution,blocked_cells=sorted(world.blocked),eligible_users=world.eligible))
    json_write(path/'world_manifest.json',dict(layout=world.layout,realization=world.realization,initial_state_not_current_state=True,parameter_source='simulation_assumption'))


def describe(sim):
    rows=[r for r in sim.records if not r['warmup']];ok=[r for r in rows if r['terminal_status']=='completed']
    result=dict(total_calls=len(sim.records),warmup_calls=len(sim.records)-len(rows),observed_calls=len(rows),completed=sum(r['terminal_status']=='completed' for r in rows),
                infeasible=sum(r['terminal_status']=='infeasible' for r in rows),incomplete=sum(r['terminal_status']=='censored' for r in rows),
                charge_then_dispatch=sum(r['branch']=='charge_then_dispatch' for r in rows),end_time=sim.end_time,observation_start=sim.warmup,observation_end=sim.horizon)
    result['latencies']={}
    for key in LABELS:
        values=[r[key] for r in rows if r[key] is not None]
        if values:result['latencies'][key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(max(values)))
    result['devices']=[]
    for eid,d in sim.devices.items():
        ledger=d.ledger
        def time_in_window(row):return max(0,min(row['end_time'],sim.horizon)-max(row['start_time'],sim.warmup))
        task=sum(time_in_window(x) for x in ledger if x['purpose'].startswith('task'))
        busy=sum(time_in_window(x) for x in ledger if x['kind']!='idle')
        own=[r for r in ok if r['entity_id']==eid]
        idle_charges=sum(x['kind']=='charge' and x['purpose']=='idle_maintenance' and time_in_window(x)>0 for x in ledger)
        result['devices'].append(dict(entity_id=eid,entity_type=d.e['entity_type'],task_utilization=task/sim.observation,total_utilization=busy/sim.observation,
             mean_task_occupancy=None if not own else float(np.mean([r['t3_task_completed']-r['decision_time'] for r in own])),observed_completed=len(own),idle_charge_sessions=idle_charges))
    return result


def export_run(sim,path,run_id,split,load,rates,world_relative):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    csv_write(path/'data/invocations.csv',sim.records)
    csv_write(path/'data/model_inputs.csv',sim.inputs)
    csv_write(path/'data/labels.csv',[{k:r[k] for k in ['invocation_id','warmup','terminal_status','feasible','failure_reason']+LABELS} for r in sim.records])
    csv_write(path/'logs/events.csv.gz',sim.events)
    csv_write(path/'logs/network_process.csv.gz',sim.network_logs)
    with gzip.open(path/'logs/digital_jobs.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(DIGITAL_FIELDS)
        for server in sim.digital.values():
            if server.log is not None:writer.writerows(server.log)
    csv_write(path/'logs/energy_ledger.csv.gz',sim.ledger)
    with gzip.open(path/'logs/queue_snapshots.jsonl.gz','wt',encoding='utf-8') as f:
        for r in sim.queue_snapshots:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    csv_write(path/'logs/final_states.csv',sim.final_states)
    json_write(path/'metadata/config.json',sim.config)
    json_write(path/'metadata/distribution_report.json',describe(sim))
    json_write(path/'metadata/run_manifest.json',dict(version=sim.config['version'],run_id=run_id,world_id=sim.world.world_id,world_path=world_relative,seed=sim.seed,split=split,load=load,
        rates_per_second=rates,warmup_s=sim.warmup,observation_s=sim.observation,end_time=sim.end_time,network_seed_streams=[200,201,202],demand_seed_streams='[seed,300,service_id]',
        rng='numpy.PCG64 via SeedSequence',python=sys.version,numpy=np.__version__,pandas=pd.__version__,simpy=simpy.__version__,platform=platform.platform(),
        normalization='World tables shared between runs; join by world_id and entity/service/user ID. invocation_id unique inside run only.'))
    hashes={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file()}
    json_write(path/'metadata/file_hashes.json',hashes)


def validate_export(path,world):
    """Read the actual disk artifact. Reject schema/label leakage and bad coordinates."""
    path=Path(path);x=pd.read_csv(path/'data/model_inputs.csv');y=pd.read_csv(path/'data/labels.csv');r=pd.read_csv(path/'data/invocations.csv')
    forbidden=set(EVENTS+LABELS+['decision_time','feasible','failure_reason','random_seed','task_size','network_bandwidth_actual','required_energy_at_dispatch','branch','warmup'])
    assert not (set(x.columns)&forbidden),f'Forbidden inputs: {set(x.columns)&forbidden}'
    assert set(x.columns)==INPUT_COLUMNS,f'Unexpected/missing inputs: {set(x.columns)^INPUT_COLUMNS}'
    assert x.invocation_id.is_unique and y.invocation_id.is_unique and r.invocation_id.is_unique
    assert list(x.invocation_id)==list(y.invocation_id)==list(r.invocation_id)
    assert set(x.service_id)<=set(world.service_by_id)
    assert set(x.user_id)<=set(world.user_by_id)
    assert (x.entity_id==x.service_id).all()
    for col in ['user_x','user_y','entity_x_at_t0','entity_y_at_t0','digital_processing_capacity','base_execution_time']:
        assert x[col].notna().all(),f'Missing {col}'
    assert x.user_x.between(0,world.width*world.resolution).all()
    assert x.user_y.between(0,world.height*world.resolution).all()
    for row in x.itertuples():
        e=world.entity_by_id[row.entity_id];u=world.user_by_id[row.user_id]
        assert abs(row.user_x-u['user_x'])<1e-7 and abs(row.user_y-u['user_y'])<1e-7
        if e['entity_type']=='static':
            assert pd.isna(row.battery_level_at_t0)
            assert abs(row.entity_x_at_t0-e['entity_x'])<1e-7 and abs(row.entity_y_at_t0-e['entity_y'])<1e-7
    completed=r.terminal_status=='completed';bad=r.terminal_status=='infeasible'
    assert r.loc[completed|bad,'network_time'].notna().all()
    assert (r.network_time.dropna()>0).all()
    for key in EVENTS[2:]+LABELS[1:]:
        assert r.loc[completed,key].notna().all()
        assert r.loc[bad,key].isna().all(),f'Infeasible has {key}'
    np.testing.assert_allclose(r.loc[completed,'completion_time'],r.loc[completed,'network_time']+r.loc[completed,'ready_time']+r.loc[completed,'execution_time'],rtol=1e-9,atol=1e-7)
    return dict(passed=True,rows=len(r),checks=['schema_no_future_inputs','unique_keys','foreign_keys','coordinates_present_and_correct','static_battery_null','completed_and_infeasible_null_masks','label_sum'])
