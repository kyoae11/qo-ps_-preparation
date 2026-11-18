"""A t0-only physical replay baseline; no simulator objects or future event access."""
import math
from .world import delta_heading


class Forecast:
    def __init__(self,world,entity,features):
        self.w,self.e,self.f=world,entity,features
        self.dynamic=entity['entity_type']=='dynamic'
        self.pos=(features['entity_x_at_t0'],features['entity_y_at_t0'])
        self.h=features['entity_heading_at_t0']
        self.b=features['battery_level_at_t0'] if self.dynamic else None
        self.t=0.
        self.standby=entity.get('standby_power',0.)
        self.reserve=entity.get('battery_capacity',0)*entity.get('reserve_ratio',0)
        self.exe=entity['base_execution_time']/entity['processing_capacity']

    def elapsed(self,dt,extra=0):
        if dt< -1e-8:raise ValueError('Negative forecast interval')
        dt=max(0,dt)
        if self.dynamic:self.b-=self.standby*dt+extra
        self.t+=dt

    def rotation(self,heading):
        angle=abs(delta_heading(self.h,heading))
        self.elapsed(angle*self.e['rotation_time_per_degree'],angle*self.e.get('energy_per_degree',0))
        self.h=heading

    def travel(self,target):
        route=self.w.route(self.pos,self.h,target,self.e)
        if route is None:return False
        for seg in route:
            self.elapsed(seg['duration'],seg['extra'])
            self.pos=seg['end'];self.h=seg['end_heading']
        return True

    def energy(self,route):
        return sum(s['extra']+self.standby*s['duration'] for s in route)

    def budget(self,pos,h,target):
        route=self.w.route(pos,h,target,self.e)
        cp=self.w.nearest(target)
        if route is None or cp is None:return None
        ah=route[-1]['end_heading'] if route else h
        back=self.w.route(target,ah,(cp['charge_x'],cp['charge_y']),self.e)
        if back is None:return None
        return self.energy(route)+self.e['base_task_energy']+self.standby*self.exe+self.energy(back)+self.reserve

    def charge(self,target):
        if target is None or not math.isfinite(target):return
        dt=max(0,target-self.b)/(self.w.config['energy']['charge_power_w']-self.standby)
        self.t+=dt;self.b=max(self.b,target)

    def execute(self,remaining=None):
        dt=self.exe if remaining is None else max(0,remaining)
        self.elapsed(dt,self.e.get('base_task_energy',0)*dt/self.exe)

    def finish_active(self):
        f=self.f;status=f['entity_status_at_t0'];purpose=f['activity_purpose_at_t0']
        active=bool(f['active_task_present_at_t0'])
        user=(f['active_task_target_x_at_t0'],f['active_task_target_y_at_t0'])
        if status=='idle':return
        if status=='execute':
            self.execute(self.exe-f['activity_elapsed_at_t0']);return
        if status=='move':
            end=(f['current_target_x_at_t0'],f['current_target_y_at_t0'])
            d=math.dist(self.pos,end)
            self.elapsed(d/self.e['mobility_speed'],d*self.e['energy_per_meter'])
            self.pos=end;self.h=f['segment_target_heading_at_t0']
        elif status=='rotate':
            self.rotation(f['segment_target_heading_at_t0'])
        if self.dynamic:
            if purpose in ('task_charge_route','idle_maintenance') and status in ('move','rotate'):
                self.travel((f['command_route_target_x_at_t0'],f['command_route_target_y_at_t0']))
            if purpose in ('task_charge_route','task_charge','idle_maintenance'):
                self.charge(f['charge_target_at_t0'])
            if active:
                self.travel(user);self.execute()
        elif active:
            # A static active task can only be in rotation before execution.
            self.execute()

    def idle_until(self,arrival):
        while self.t<arrival-1e-9:
            if not self.dynamic or self.standby==0:
                self.elapsed(arrival-self.t);return
            c=self.w.nearest(self.pos)
            if c is None:self.elapsed(arrival-self.t);return
            cp=(c['charge_x'],c['charge_y'])
            route=self.w.route(self.pos,self.h,cp,self.e)
            trip=self.energy(route);trigger=max(2*self.reserve,self.reserve+trip)
            wait=max(0,(self.b-trigger)/self.standby)
            if self.t+wait>=arrival:
                self.elapsed(arrival-self.t);return
            self.elapsed(wait)
            if self.b+1e-6<trip+self.reserve:return
            target=min(self.e['battery_capacity'],trigger+self.reserve)
            self.travel(cp);self.charge(target)

    def task(self,target):
        if self.dynamic:
            need=self.budget(self.pos,self.h,target)
            if need is None:return False
            if self.b<need:
                cp=self.w.nearest(self.pos)
                if cp is None:return False
                dest=(cp['charge_x'],cp['charge_y'])
                route=self.w.route(self.pos,self.h,dest,self.e)
                if self.b+1e-6<self.energy(route)+self.reserve:return False
                heading=route[-1]['end_heading'] if route else self.h
                need=self.budget(dest,heading,target)
                if need is None or need>self.e['battery_capacity']:return False
                self.travel(dest);self.charge(need)
            self.travel(target)
        else:
            if math.dist(self.pos,target)>self.e['service_radius']:return False
            heading=math.degrees(math.atan2(target[1]-self.pos[1],target[0]-self.pos[0]))%360 if target!=self.pos else self.h
            self.rotation(heading)
        self.execute();return True


def predict(world,features,queue,ready_fallback):
    """Network estimates use public job-distribution assumptions, never future jobs."""
    f=features;e=world.entity_by_id[int(f['entity_id'])]
    cap=e['digital_processing_capacity']
    elapsed=f['digital_activity_elapsed_at_t0']
    lo=world.config['network']['background_work_min']/cap
    hi=world.config['network']['background_work_max']/cap
    # Remaining time for a uniform-length background job given it is still active.
    residual=max(0,(max(lo,elapsed)+hi)/2-elapsed) if elapsed<hi else 0.
    wait=f['digital_queue_length_at_t0']*(lo+hi)/2+f['digital_busy_at_t0']*residual
    bw=min(f['user_bandwidth_est_at_t0'],f['entity_bandwidth_est_at_t0'])
    net=8*(f['request_bytes']+f['response_bytes'])/(bw*1e6)+wait+f['digital_processing_workload']/cap+world.config['network']['base_s']
    sim=Forecast(world,e,f);sim.finish_active()
    for job in queue['waiting']:sim.task((job['user_x'],job['user_y']))
    sim.idle_until(net)
    ok=sim.task((f['user_x'],f['user_y']))
    ready=max(0,sim.t-net-sim.exe) if ok else ready_fallback
    return [net,ready,sim.exe,net+ready+sim.exe],not ok
