"""Checks computed from events/ledger rather than a generator PASS flag."""
import math
from collections import defaultdict
import numpy as np
from .simulator import EVENTS,LABELS


def validate(sim):
    counts=defaultdict(int)
    for r in sim.records:
        if r['terminal_status']=='censored':
            assert r['t3_task_completed'] is None and r['completion_time'] is None
            counts['censored_masks']+=1;continue
        assert r['network_time']>0
        assert r['t1_network_response_received']>=r['t0_request_sent']
        assert r['decision_time'] is not None and r['decision_time']>=r['t1_network_response_received']
        counts['network_and_decision_order']+=1
        if r['terminal_status']=='infeasible':
            assert r['feasible'] is False and r['failure_reason']
            assert all(r[k] is None for k in EVENTS[2:]+LABELS[1:])
            counts['infeasible_masks']+=1;continue
        assert r['terminal_status']=='completed','Drain ended with incomplete calls'
        assert r['feasible'] is True
        times=[r[k] for k in EVENTS]
        assert times==sorted(times)
        assert r['decision_time']<=r['td_dispatchable']+1e-7
        for a,b in [(r['completion_time'],sum(r[k] for k in ['network_time','ready_time','execution_time'])),
                    (r['t3_task_completed']-r['t0_request_sent'],r['completion_time']),
                    (r['ready_time'],r['dispatch_wait_time']+r['move_time']+r['rotate_time'])]:
            assert math.isclose(a,b,abs_tol=1e-7)
        e=sim.world.entity_by_id[r['entity_id']]
        assert math.isclose(r['execution_time'],e['base_execution_time']/e['processing_capacity'],abs_tol=1e-7)
        counts['five_events_and_stage_sums']+=1
        if r['entity_type']=='static':
            assert r['move_time']==0 and math.hypot(r['user_x']-e['entity_x'],r['user_y']-e['entity_y'])<=e['service_radius']+1e-8
        else:
            assert r['battery_at_dispatch']+1e-6>=r['required_energy_at_dispatch']
            assert r['battery_at_completion']+1e-5>=r['return_energy_budget']+e['reserve_ratio']*e['battery_capacity']
            if r['branch']=='charge_then_dispatch':
                assert r['required_energy_at_charger']<=e['battery_capacity']+1e-6
                assert r['charge_time']>=0
            counts['dispatch_and_return_budget']+=1
    for eid,d in sim.devices.items():
        mine=[r for r in sim.records if r['entity_id']==eid]
        enqueues=[r['invocation_id'] for r in sim.events if r['entity_id']==eid and r['event']=='enqueue']
        decisions=[r['invocation_id'] for r in sim.events if r['entity_id']==eid and r['event']=='head_decision']
        assert enqueues[:len(decisions)]==decisions,'FIFO mismatch'
        previous=0
        for r in sorted([x for x in mine if x['decision_time'] is not None],key=lambda r:r['decision_time']):
            assert r['decision_time']>=previous-1e-7
            previous=r['t3_task_completed'] if r['terminal_status']=='completed' else r['decision_time']
        ledger=d.ledger;last=0.;battery=d.e.get('initial_battery_level')
        lastpos=(d.e.get('initial_x',d.e.get('entity_x')),d.e.get('initial_y',d.e.get('entity_y')))
        for row in ledger:
            assert math.isclose(row['start_time'],last,abs_tol=1e-7),'Device timeline gap/overlap'
            assert math.dist(lastpos,(row['start_x'],row['start_y']))<1e-7
            dt=row['end_time']-row['start_time'];assert dt>0
            assert row['kind'] in ('idle','move','rotate','execute','charge')
            if row['kind']!='move':assert math.dist((row['start_x'],row['start_y']),(row['end_x'],row['end_y']))<1e-7
            if d.dynamic:
                assert math.isclose(row['battery_before'],battery,abs_tol=1e-5)
                expected=battery+row['charge_energy']-row['standby_energy']-row['extra_energy']
                assert math.isclose(expected,row['battery_after'],abs_tol=1e-5)
                assert -1e-5<=expected<=d.e['battery_capacity']+1e-5
                assert math.isclose(row['standby_energy'],d.standby*dt,abs_tol=1e-5)
                if row['kind']=='move':
                    distance=math.dist((row['start_x'],row['start_y']),(row['end_x'],row['end_y']))
                    assert math.isclose(distance,dt*d.e['mobility_speed'],abs_tol=1e-5)
                    assert math.isclose(row['extra_energy'],distance*d.e['energy_per_meter'],abs_tol=1e-5)
                    # Validate every crossed grid cell independently of A*.
                    res=sim.world.resolution
                    steps=max(1,int(math.ceil(distance/res)))
                    for step in range(steps+1):
                        frac=step/steps
                        cell=(round((row['start_x']+frac*(row['end_x']-row['start_x']))/res),round((row['start_y']+frac*(row['end_y']-row['start_y']))/res))
                        assert cell not in sim.world.blocked,'Movement crosses obstacle'
                if row['kind']=='rotate':
                    angle=abs((row['end_heading']-row['start_heading']+180)%360-180)
                    assert math.isclose(dt,angle*d.e['rotation_time_per_degree'],abs_tol=1e-6)
                    assert math.isclose(row['extra_energy'],angle*d.e['energy_per_degree'],abs_tol=1e-5)
                if row['kind']=='execute':assert math.isclose(row['extra_power'],d.e['base_task_energy']/(d.e['base_execution_time']/d.e['processing_capacity']),abs_tol=1e-7)
                if row['kind']=='charge':
                    assert math.isclose(row['charge_energy'],sim.config['energy']['charge_power_w']*dt,abs_tol=1e-5)
                    assert any(math.dist((row['start_x'],row['start_y']),(c['charge_x'],c['charge_y']))<1e-6 for c in sim.world.chargers)
                battery=row['battery_after']
            last=row['end_time'];lastpos=(row['end_x'],row['end_y'])
        assert math.isclose(last,sim.end_time,abs_tol=1e-7)
        if d.dynamic:
            assert math.isclose(sum(r['standby_energy'] for r in ledger),d.standby*sim.end_time,abs_tol=1e-4)
        counts['fifo_and_common_clock_energy_continuity']+=1
    # t0 snapshots independently reconstructed from exported piecewise ledger.
    for f,r in zip(sim.inputs,sim.records):
        assert f['invocation_id']==r['invocation_id']
        t=r['t0_request_sent'];d=sim.devices[f['entity_id']]
        p=next((x for x in d.ledger if x['start_time']-1e-10<=t<x['end_time']-1e-10),None)
        if p is None:continue
        dt=t-p['start_time'];ratio=dt/p['duration']
        x=p['start_x']+ratio*(p['end_x']-p['start_x']);y=p['start_y']+ratio*(p['end_y']-p['start_y'])
        assert math.isclose(f['entity_x_at_t0'],x,abs_tol=1e-6) and math.isclose(f['entity_y_at_t0'],y,abs_tol=1e-6)
        if d.dynamic:
            expected=p['battery_before']+dt*(p['charge_power']-p['extra_power']-d.standby)
            assert math.isclose(f['battery_level_at_t0'],expected,abs_tol=1e-5)
        counts['continuous_t0_snapshots']+=1
    return dict(passed=True,checked_counts=dict(counts))
