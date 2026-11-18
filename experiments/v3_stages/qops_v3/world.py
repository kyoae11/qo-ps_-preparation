"""Fixed worlds; deterministic A*; straight segments retain all turns."""
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import numpy as np


def delta_heading(a, b):
    return (b - a + 180) % 360 - 180


@dataclass
class World:
    world_id: str
    layout: int
    realization: int
    config: dict
    users: list
    services: list
    entities: list
    chargers: list
    blocked: set
    eligible: dict

    def __post_init__(self):
        self.user_by_id = {u['user_id']: u for u in self.users}
        self.entity_by_id = {e['entity_id']: e for e in self.entities}
        self.service_by_id = {s['service_id']: s for s in self.services}
        self.width = self.config['world']['width']
        self.height = self.config['world']['height']
        self.resolution = self.config['world']['resolution']
        self.path = lru_cache(maxsize=50000)(self._path)

    def _path(self, start, goal):
        r = self.resolution
        a = tuple(int(round(v/r)) for v in start)
        b = tuple(int(round(v/r)) for v in goal)
        if a in self.blocked or b in self.blocked:
            return None
        if not all(0 <= p[0] < self.width and 0 <= p[1] < self.height for p in (a,b)):
            return None
        heap = [(abs(a[0]-b[0])+abs(a[1]-b[1]), 0, a)]
        distances, prev = {a: 0}, {}
        while heap:
            _, g, p = heapq.heappop(heap)
            if g != distances[p]:
                continue
            if p == b:
                cells = [p]
                while p != a:
                    p = prev[p]; cells.append(p)
                return tuple((x*r,y*r) for x,y in reversed(cells))
            for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
                q = (p[0]+dx,p[1]+dy)
                if 0<=q[0]<self.width and 0<=q[1]<self.height and q not in self.blocked:
                    if g+1 < distances.get(q, math.inf):
                        distances[q],prev[q] = g+1,p
                        heapq.heappush(heap,(g+1+abs(q[0]-b[0])+abs(q[1]-b[1]),g+1,q))
        return None

    def route(self, start, heading, goal, entity):
        path = self.path(tuple(start),tuple(goal))
        if path is None:
            return None
        result, last, h = [], tuple(start), heading
        for i in range(1,len(path)):
            p,q = path[i-1],path[i]
            nh = math.degrees(math.atan2(q[1]-p[1],q[0]-p[0])) % 360
            dh = delta_heading(h,nh)
            if dh:
                result.append(dict(kind='rotate',duration=abs(dh)*entity['rotation_time_per_degree'],end=last,end_heading=nh,extra=abs(dh)*entity.get('energy_per_degree',0),distance=0,angle=abs(dh)))
            dist = math.dist(p,q)
            if result and result[-1]['kind']=='move' and abs(delta_heading(result[-1]['end_heading'],nh))<1e-9:
                seg=result[-1]; seg['duration']+=dist/entity['mobility_speed'];seg['distance']+=dist;seg['extra']+=dist*entity['energy_per_meter'];seg['end']=q
            else:
                result.append(dict(kind='move',duration=dist/entity['mobility_speed'],end=q,end_heading=nh,extra=dist*entity['energy_per_meter'],distance=dist,angle=0))
            last,h=q,nh
        return result

    def nearest(self, start):
        candidates=[]
        for c in self.chargers:
            p=self.path(tuple(start),(c['charge_x'],c['charge_y']))
            if p is not None:
                candidates.append(((len(p)-1)*self.resolution,c['charge_point_id'],c))
        return min(candidates,key=lambda x:x[:2])[2] if candidates else None


def build_world(config, layout=0, realization=0):
    # Device realizations are independent of maps and user terminal parameters.
    # This makes unseen-layout and unseen-device tests interpretable separately.
    rng=np.random.default_rng(np.random.SeedSequence([20261018,100,realization]))
    user_rng=np.random.default_rng(np.random.SeedSequence([20261018,101]))
    r=config['world']['resolution']
    # Six rooms separated by walls, connected by doors into a central corridor.
    blocked=set()
    for x in (20,40):
        for y in range(40):
            if y not in (18,19,20,21): blocked.add((x,y))
    for y in (17,22):
        for x in range(60):
            doorcenters=([10,30,50],[5,35,45],[15,25,55],[7,27,47],[12,32,52])[layout%5]
            if not any(abs(x-d)<=1 for d in doorcenters): blocked.add((x,y))
    users=[];entities=[];services=[];eligible={}
    energy=config['energy']
    for zone in range(6):
        col,row=zone%3,zone//3
        ybase=3 if row==0 else 26
        positions=[((col*20+4+3*(i%5))*r,(ybase+6*(i//5))*r) for i in range(config['world']['users_per_zone'])]
        ids=[]
        for x,y in positions:
            uid=len(users)+1;ids.append(uid)
            users.append(dict(user_id=uid,user_x=x,user_y=y,access_point_id=zone%4+1,terminal_bandwidth_limit=float(user_rng.choice([20,50,100]))))
        for j in range(2):
            eid=len(entities)+1
            e=dict(entity_id=eid,entity_type='static',entity_x=(col*20+10)*r,entity_y=(ybase+3+j)*r,service_radius=12*r,
                   access_point_id=zone%4+1,device_bandwidth_limit=float(rng.choice([50,100])),digital_processing_capacity=float(rng.choice([1000,2000,4000])),
                   initial_heading=float(rng.choice([0,90,180,270])),rotation_time_per_degree=float(rng.choice([.01,.02,.03])),
                   processing_capacity=float(rng.choice([.8,1,1.2])),base_execution_time=float([60,120][j]))
            entities.append(e);eligible[eid]=ids
            services.append(dict(service_id=eid,service_type=['fixed_scan','fixed_test'][j],request_bytes=[512,1024][j],response_bytes=128,digital_processing_workload=[10,20][j]))
    for i in range(6):
        eid=len(entities)+1
        e=dict(entity_id=eid,entity_type='dynamic',initial_x=float((5+i*9)*r),initial_y=19*r,
            access_point_id=i%4+1,device_bandwidth_limit=float(rng.choice([50,100])),digital_processing_capacity=float(rng.choice([1000,2000,4000])),
            mobility_speed=float(rng.choice([.8,1,1.2])),initial_heading=float(rng.choice([0,90,180,270])),rotation_time_per_degree=float(rng.choice([.01,.02,.03])),
            processing_capacity=float(rng.choice([.8,1,1.2])),base_execution_time=float([30,90][i%2]),battery_capacity=float(energy['capacity_j']),
            initial_battery_level=float(energy['capacity_j']*rng.uniform(energy['initial_ratio_min'],energy['initial_ratio_max'])),reserve_ratio=energy['reserve_ratio'],
            energy_per_meter=energy['move_j_per_m'],energy_per_degree=energy['rotate_j_per_degree'],base_task_energy=energy['task_j'],standby_power=float(rng.choice([10,20,30])))
        entities.append(e);eligible[eid]=[u['user_id'] for u in users]
        services.append(dict(service_id=eid,service_type=['mobile_test','delivery'][i%2],request_bytes=[1024,2048][i%2],response_bytes=128,digital_processing_workload=[20,30][i%2]))
    # Independent Latin-hypercube draws across entities for each configuration.
    ranges={'digital_processing_capacity':(1000,4000),'processing_capacity':(.8,1.2),
            'rotation_time_per_degree':(.01,.03),'mobility_speed':(.6,1.2),
            'battery_capacity':(288000,432000),'energy_per_meter':(120,180),
            'energy_per_degree':(1.6,2.4),'base_task_energy':(16000,24000),'standby_power':(10,30)}
    for key,(lo,hi) in ranges.items():
        applicable=[e for e in entities if key in e]
        n=len(applicable);values=lo+(hi-lo)*(rng.permutation(n)+rng.random(n))/n
        for e,value in zip(applicable,values):e[key]=float(value)
    for e in entities:
        if e['entity_type']=='dynamic':
            e['initial_battery_level']=float(e['battery_capacity']*rng.uniform(.7,1.0))
    # Additional wall islands, separated from installed devices, users and chargers.
    if layout>=3:
        for x in ([7,27,47] if layout==3 else [12,32,52]):
            for y in (11,12,29,30):blocked.add((x,y))
    chargers=[dict(charge_point_id=i+1,charge_x=x*r,charge_y=20*r) for i,x in enumerate((8,30,52))]
    return World(f'L{layout+1}_D{realization+1}',layout,realization,config,users,services,entities,chargers,blocked,eligible)
