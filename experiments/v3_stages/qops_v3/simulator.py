"""One serial physical worker per device. Digital acknowledgements are independent.

All energy/position snapshots are continuous evaluations of the current segment.
No writes to SimPy private queues, event success flags, or timeout internals.
"""
from collections import deque
import math
import numpy as np
import simpy
from .world import delta_heading
from .network import Network
from .digital import DigitalServer
from .features import observe

LABELS=['network_time','dispatch_wait_time','ready_time','execution_time','completion_time']
EVENTS=['t0_request_sent','t1_network_response_received','td_dispatchable','t2_entity_ready','t3_task_completed']


class Device:
    def __init__(self,sim,entity):
        self.sim=sim;self.env=sim.env;self.e=entity;self.eid=entity['entity_id']
        self.dynamic=entity['entity_type']=='dynamic'
        self.position=(entity.get('initial_x',entity.get('entity_x')),entity.get('initial_y',entity.get('entity_y')))
        self.heading=entity['initial_heading'];self.battery=entity.get('initial_battery_level')
        self.charge_target=None;self.route_destination=None
        self.last=0.;self.phase=None;self.queue=deque();self.active=None
        self.wake=self.env.event();self.ledger=[]
        self.proc=self.env.process(self.worker())

    @property
    def standby(self): return self.e.get('standby_power',0)
    @property
    def reserve(self): return self.e.get('battery_capacity',0)*self.e.get('reserve_ratio',0)

    def snapshot(self,t):
        p=self.phase
        if p is None:
            return dict(x=self.position[0],y=self.position[1],heading=self.heading,battery=None if self.battery is None else self.battery-self.standby*(t-self.last),status='idle',elapsed=t-self.last,target_x=self.position[0],target_y=self.position[1])
        elapsed=min(max(t-p['start_time'],0),p['planned_duration'])
        f=elapsed/p['planned_duration'] if p['planned_duration'] else 1
        battery=None if p['battery_before'] is None else p['battery_before']+(p['charge_power']-self.standby-p['extra_power'])*elapsed
        return dict(x=p['start_x']+f*(p['end_x']-p['start_x']),y=p['start_y']+f*(p['end_y']-p['start_y']),heading=(p['start_heading']+f*delta_heading(p['start_heading'],p['end_heading']))%360,
                    battery=battery,status=p['kind'],elapsed=elapsed,target_x=p['end_x'],target_y=p['end_y'])

    def settle(self,t):
        s=self.snapshot(t)
        if self.phase:
            row=dict(self.phase);dt=t-row['start_time']
            row.update(end_time=t,duration=dt,end_x=s['x'],end_y=s['y'],end_heading=s['heading'],battery_after=s['battery'],
                       standby_energy=self.standby*dt,extra_energy=row['extra_power']*dt,charge_energy=row['charge_power']*dt,partial=dt<row['planned_duration']-1e-8)
            if dt>1e-10:self.ledger.append(row)
        elif t>self.last+1e-10:
            dt=t-self.last
            self.ledger.append(dict(entity_id=self.eid,invocation_id=None,purpose='idle',kind='idle',start_time=self.last,end_time=t,planned_duration=dt,duration=dt,
                start_x=self.position[0],start_y=self.position[1],end_x=self.position[0],end_y=self.position[1],start_heading=self.heading,end_heading=self.heading,
                battery_before=self.battery,battery_after=s['battery'],extra_power=0,charge_power=0,extra_energy=0,standby_energy=self.standby*dt,charge_energy=0,partial=False))
        self.position=(s['x'],s['y']);self.heading=s['heading'];self.battery=s['battery'];self.last=t;self.phase=None
        if self.dynamic and not (-1e-5<=self.battery<=self.e['battery_capacity']+1e-5):
            raise AssertionError(f'Energy boundary: device {self.eid} time {t}: {self.battery}')

    def segment(self,kind,duration,end=None,end_heading=None,extra=0,charge_power=0,purpose='task',iid=None):
        if duration<=1e-12:return
        self.settle(self.env.now)
        end=self.position if end is None else end;end_heading=self.heading if end_heading is None else end_heading
        self.phase=dict(entity_id=self.eid,invocation_id=iid,purpose=purpose,kind=kind,start_time=self.env.now,planned_duration=duration,
            start_x=self.position[0],start_y=self.position[1],end_x=end[0],end_y=end[1],start_heading=self.heading,end_heading=end_heading,
            battery_before=self.battery,extra_power=extra/duration,charge_power=charge_power)
        yield self.env.timeout(duration)
        self.settle(self.env.now)

    def travel(self,route,purpose,iid):
        self.route_destination=route[-1]['end'] if route else self.position
        for seg in route:
            yield from self.segment(seg['kind'],seg['duration'],seg['end'],seg['end_heading'],seg['extra'],purpose=purpose,iid=iid)
        self.route_destination=None

    def route_energy(self,route):
        return sum(s['extra']+self.standby*s['duration'] for s in route)

    def required(self,start,heading,target):
        route=self.sim.world.route(start,heading,target,self.e)
        if route is None:return None
        arrival_heading=route[-1]['end_heading'] if route else heading
        c=self.sim.world.nearest(target)
        if c is None:return None
        back=self.sim.world.route(target,arrival_heading,(c['charge_x'],c['charge_y']),self.e)
        exec_time=self.e['base_execution_time']/self.e['processing_capacity']
        energy=self.route_energy(route)+self.e['base_task_energy']+self.standby*exec_time+self.route_energy(back)+self.reserve
        return dict(energy=energy,route=route,return_energy=self.route_energy(back),return_charge_point_id=c['charge_point_id'])

    def charge_to(self,target,purpose,iid):
        self.settle(self.env.now)
        net=self.sim.config['energy']['charge_power_w']-self.standby
        if net<=0:raise ValueError('Charging power must exceed standby power')
        if target>self.e['battery_capacity']+1e-8:raise ValueError('Charge target exceeds capacity')
        self.charge_target=target
        if self.battery<target-1e-8:
            yield from self.segment('charge',(target-self.battery)/net,charge_power=self.sim.config['energy']['charge_power_w'],purpose=purpose,iid=iid)
        self.charge_target=None

    def maintenance_plan(self):
        c=self.sim.world.nearest(self.position)
        if c is None:return None
        route=self.sim.world.route(self.position,self.heading,(c['charge_x'],c['charge_y']),self.e)
        trip=self.route_energy(route)
        trigger=max(2*self.reserve,self.reserve+trip)
        return route,trip,trigger,min(self.e['battery_capacity'],trigger+self.reserve)

    def worker(self):
        while True:
            if not self.queue:
                # An unconsumed wake from busy time is harmless and replaced here.
                self.wake=self.env.event()
                if not self.dynamic:
                    yield self.wake;continue
                self.settle(self.env.now)
                plan=self.maintenance_plan()
                if plan is None:
                    yield self.wake;continue
                route,trip,trigger,target=plan
                delay=max(0,(self.battery-trigger)/self.standby) if self.standby else math.inf
                if math.isinf(delay):yield self.wake;continue
                if delay>1e-8:
                    yield self.wake | self.env.timeout(delay)
                    if self.queue:continue
                self.settle(self.env.now)
                if self.battery+1e-6<trip+self.reserve:
                    raise RuntimeError('Idle maintenance cannot reach charger safely')
                if not route and target<=self.battery+1e-8:
                    raise RuntimeError('Idle charging policy would spin without advancing time')
                self.charge_target=target
                yield from self.travel(route,'idle_maintenance',None)
                yield from self.charge_to(target,'idle_maintenance',None)
                continue
            r=self.queue.popleft();self.active=r
            r['decision_time']=self.env.now
            self.sim.events.append(dict(time=self.env.now,event='head_decision',entity_id=self.eid,invocation_id=r['invocation_id']))
            self.settle(self.env.now)
            target=(r['user_x'],r['user_y'])
            if not self.dynamic:
                if math.dist(self.position,target)>self.e['service_radius']+1e-8:
                    self.fail(r,'outside_static_coverage');continue
                heading=math.degrees(math.atan2(target[1]-self.position[1],target[0]-self.position[0]))%360 if target!=self.position else self.heading
                duration=abs(delta_heading(self.heading,heading))*self.e['rotation_time_per_degree']
                r.update(feasible=True,branch='direct',td_dispatchable=self.env.now,rotate_time=duration,move_time=0.)
                yield from self.segment('rotate',duration,end_heading=heading,iid=r['invocation_id'])
            else:
                req=self.required(self.position,self.heading,target)
                r['battery_at_decision']=self.battery
                r['required_energy_at_current']=None if req is None else req['energy']
                if req is None:
                    self.fail(r,'no_task_or_return_path');continue
                if self.battery+1e-7>=req['energy']:
                    r['branch']='direct'
                else:
                    c=self.sim.world.nearest(self.position)
                    if c is None:self.fail(r,'no_reachable_charger');continue
                    cp=(c['charge_x'],c['charge_y'])
                    to_c=self.sim.world.route(self.position,self.heading,cp,self.e)
                    trip=self.route_energy(to_c)
                    if self.battery+1e-7<trip+self.reserve:
                        self.fail(r,'insufficient_energy_to_charger');continue
                    h=to_c[-1]['end_heading'] if to_c else self.heading
                    from_c=self.required(cp,h,target)
                    r.update(charge_point_id=c['charge_point_id'],required_energy_at_charger=None if from_c is None else from_c['energy'])
                    if from_c is None or from_c['energy']>self.e['battery_capacity']+1e-7:
                        self.fail(r,'capacity_insufficient_from_nearest_charger');continue
                    r.update(feasible=True,branch='charge_then_dispatch')
                    began=self.env.now
                    self.charge_target=from_c['energy']
                    yield from self.travel(to_c,'task_charge_route',r['invocation_id'])
                    r['charge_route_time']=self.env.now-began
                    began=self.env.now
                    yield from self.charge_to(from_c['energy'],'task_charge',r['invocation_id'])
                    r['charge_time']=self.env.now-began
                    req=from_c
                r.update(feasible=True,td_dispatchable=self.env.now,battery_at_dispatch=self.battery,required_energy_at_dispatch=req['energy'],
                    return_energy_budget=req['return_energy'],return_charge_point_id=req['return_charge_point_id'],
                    rotate_time=sum(x['duration'] for x in req['route'] if x['kind']=='rotate'),move_time=sum(x['duration'] for x in req['route'] if x['kind']=='move'))
                yield from self.travel(req['route'],'task_positioning',r['invocation_id'])
            r['t2_entity_ready']=self.env.now
            duration=self.e['base_execution_time']/self.e['processing_capacity']
            yield from self.segment('execute',duration,extra=self.e.get('base_task_energy',0),iid=r['invocation_id'])
            r.update(t3_task_completed=self.env.now,terminal_status='completed',battery_at_completion=self.battery)
            r.update(dispatch_wait_time=r['td_dispatchable']-r['t1_network_response_received'],ready_time=r['t2_entity_ready']-r['t1_network_response_received'],execution_time=duration,completion_time=self.env.now-r['t0_request_sent'])
            self.sim.events.append(dict(time=self.env.now,event='completed',entity_id=self.eid,invocation_id=r['invocation_id']))
            self.active=None;self.sim.mark_finished()

    def fail(self,r,reason):
        r.update(feasible=False,failure_reason=reason,branch='rejected',terminal_status='infeasible',terminal_time=self.env.now)
        self.sim.events.append(dict(time=self.env.now,event='infeasible',entity_id=self.eid,invocation_id=r['invocation_id']))
        self.active=None;self.sim.mark_finished()


class Simulation:
    def __init__(self,world,rates,seed,warmup=3600,observation=14400,arrivals=None,record_digital=True):
        self.world=world;self.config=world.config;self.env=simpy.Environment();self.seed=seed
        self.warmup=warmup;self.horizon=warmup+observation;self.observation=observation
        self.network=Network(world.config,seed)
        self.digital={e['entity_id']:DigitalServer(e,world.config,seed,record_digital) for e in world.entities}
        self.network_logs=[]
        self.records=[];self.inputs=[];self.queue_snapshots=[];self.events=[]
        self.done=self.env.event();self.finished=0
        self.devices={e['entity_id']:Device(self,e) for e in world.entities}
        if arrivals is None:
            arrivals=[]
            for s in world.services:
                sid=s['service_id'];rng=np.random.default_rng(np.random.SeedSequence([seed,300,sid]));t=0.
                while True:
                    t+=float(rng.exponential(1/rates[sid]))
                    if t>=self.horizon:break
                    arrivals.append((t,int(rng.choice(world.eligible[sid])),sid))
        self.arrivals=sorted(arrivals,key=lambda a:(a[0],a[2],a[1]))
        for i,(t,u,s) in enumerate(self.arrivals,1):self.env.process(self.request(i,t,u,s))
        if not self.arrivals:self.done.succeed()

    def mark_finished(self):
        self.finished+=1
        if self.finished==len(self.arrivals) and not self.done.triggered:self.done.succeed()

    def request(self,i,t,uid,sid):
        yield self.env.timeout(t)
        user=self.world.user_by_id[uid];service=self.world.service_by_id[sid];device=self.devices[sid];entity=device.e
        rt,net_input,net_actual=self.network.request(t,user,service,entity)
        snap=device.snapshot(t)
        f=dict(invocation_id=i,user_id=uid,service_id=sid,entity_id=sid,entity_type=entity['entity_type'],user_x=user['user_x'],user_y=user['user_y'],
               user_ap_id_at_t0=user['access_point_id'],entity_ap_id_at_t0=entity['access_point_id'],terminal_bandwidth_limit=user['terminal_bandwidth_limit'],
               queue_length_at_t0=len(device.queue),entity_status_at_t0=snap['status'],entity_x_at_t0=snap['x'],entity_y_at_t0=snap['y'],entity_heading_at_t0=snap['heading'],
               battery_level_at_t0=snap['battery'],activity_elapsed_at_t0=snap['elapsed'],current_target_x_at_t0=snap['target_x'],current_target_y_at_t0=snap['target_y'])
        f.update(net_input)
        f.update({k:v for k,v in service.items() if k!='service_id'})
        for k in ('device_bandwidth_limit','digital_processing_capacity','rotation_time_per_degree','processing_capacity','base_execution_time','service_radius','mobility_speed','battery_capacity','reserve_ratio','energy_per_meter','energy_per_degree','base_task_energy','standby_power'):
            f[k]=entity.get(k)
        # User targets already in the queue are known at t0; no future actual path.
        q=[dict(invocation_id=x['invocation_id'],user_x=x['user_x'],user_y=x['user_y']) for x in device.queue]
        self.queue_snapshots.append(dict(invocation_id=i,t0=t,active_invocation_id=None if device.active is None else device.active['invocation_id'],waiting=q))
        f['distance_to_user_at_t0']=math.hypot(snap['x']-user['user_x'],snap['y']-user['user_y'])
        previous=(snap['target_x'],snap['target_y']);distance=0.
        for item in q:
            goal=(item['user_x'],item['user_y'])
            path=self.world.path(previous,goal)
            distance+=(len(path)-1)*self.world.resolution if path else math.dist(previous,goal)
            previous=goal
        f['queued_target_distance_sum_at_t0']=distance
        f['distance_from_last_known_target_at_t0']=math.dist(previous,(user['user_x'],user['user_y']))
        extra,known=observe(device,user)
        f.update(extra);f.update(self.digital[sid].snapshot(t))
        known['invocation_id']=i
        self.queue_snapshots[-1]=known
        self.inputs.append(f)
        r=dict(f);r.update({k:None for k in LABELS+EVENTS});r.update(net_actual)
        r.update(t0_request_sent=t,network_time=None,feasible=None,failure_reason=None,decision_time=None,terminal_status='pending',branch=None,
                 warmup=t<self.warmup,charge_route_time=0.,charge_time=0.,rotate_time=None,move_time=None,battery_at_decision=None,battery_at_dispatch=None,battery_at_completion=None,
                 required_energy_at_current=None,required_energy_at_charger=None,required_energy_at_dispatch=None,return_energy_budget=None,return_charge_point_id=None,charge_point_id=None,terminal_time=None)
        self.records.append(r)
        self.events.append(dict(time=t,event='request',entity_id=sid,invocation_id=i))
        # Transmission and digital resource events share the physical clock.
        capacity=min(user['terminal_bandwidth_limit'],entity['device_bandwidth_limit'])
        up=8*service['request_bytes']/(1e6*min(capacity,self.network.bandwidth(self.env.now)))
        yield self.env.timeout(up)
        arrived=self.env.now
        job=self.digital[sid].submit(arrived,service['digital_processing_workload'],i)
        yield self.env.timeout(job[5]-self.env.now)
        down=8*service['response_bytes']/(1e6*min(capacity,self.network.bandwidth(self.env.now)))
        overhead=self.config['network']['base_s']+net_actual['network_noise']
        yield self.env.timeout(down+overhead)
        r.update(network_time=self.env.now-t,request_transfer_time=up,
                 digital_wait_time=job[4]-arrived,digital_time=job[5]-job[4],
                 response_transfer_time=down,network_overhead_time=overhead,
                 digital_arrival_time=arrived,digital_start_time=job[4],digital_end_time=job[5])
        self.network_logs.append(dict(invocation_id=i,entity_id=sid,t0=t,request_arrived=arrived,
            digital_start=job[4],digital_end=job[5],response_transfer=down,overhead=overhead,t1=self.env.now))
        r['t1_network_response_received']=self.env.now
        device.queue.append(r)
        self.events.append(dict(time=self.env.now,event='enqueue',entity_id=sid,invocation_id=i))
        if not device.wake.triggered:device.wake.succeed()

    def run(self):
        def termination():
            yield self.env.timeout(self.horizon)
            yield self.done | self.env.timeout(self.config['run']['max_drain_s'])
        stop=self.env.process(termination())
        self.env.run(until=stop)
        self.end_time=self.env.now
        self.final_states=[dict(entity_id=eid,time=self.end_time,**d.snapshot(self.end_time)) for eid,d in self.devices.items()]
        for d in self.devices.values():d.settle(self.end_time)
        for r in self.records:
            if r['terminal_status']=='pending':
                r['terminal_status']='censored';r['terminal_time']=self.end_time
        for server in self.digital.values():server.advance(self.end_time)
        self.records.sort(key=lambda r:r['invocation_id']);self.inputs.sort(key=lambda r:r['invocation_id'])
        self.events.sort(key=lambda r:r['time'])
        self.ledger=[r for d in self.devices.values() for r in d.ledger]
        self.ledger.sort(key=lambda r:(r['entity_id'],r['start_time']))
        return self
