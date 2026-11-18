"""Lazy one-second background process, without a finite precomputed horizon."""
import math
import numpy as np


class Network:
    def __init__(self, config, seed):
        self.c=config['network']
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,200]))
        self.obs=np.random.default_rng(np.random.SeedSequence([seed,201]))
        self.noise=np.random.default_rng(np.random.SeedSequence([seed,202]))
        self.busy=False;self.next_switch=float(self.rng.exponential(self.c['normal_dwell_s']))
        self.b=[float(self.c['normal_mbps'])]

    def bandwidth(self,t):
        index=int(math.floor(t))
        while len(self.b)<=index:
            time=len(self.b)
            while time>=self.next_switch:
                self.busy=not self.busy
                self.next_switch+=float(self.rng.exponential(self.c['busy_dwell_s'] if self.busy else self.c['normal_dwell_s']))
            target=self.c['busy_mbps'] if self.busy else self.c['normal_mbps']
            value=target+self.c['ar']*(self.b[-1]-target)+self.rng.normal(0,self.c['innovation_sd'])
            self.b.append(float(np.clip(value,self.c['min_mbps'],self.c['max_mbps'])))
        return self.b[index]

    def request(self,t,user,service,entity):
        b=self.bandwidth(t);frac=self.c['estimate_noise_fraction']
        ue=min(user['terminal_bandwidth_limit'],b)*self.obs.uniform(1-frac,1+frac)
        ee=min(entity['device_bandwidth_limit'],b)*self.obs.uniform(1-frac,1+frac)
        ue=min(user['terminal_bandwidth_limit'],ue);ee=min(entity['device_bandwidth_limit'],ee)
        transfer=8*(service['request_bytes']+service['response_bytes'])/(1e6*min(b,user['terminal_bandwidth_limit'],entity['device_bandwidth_limit']))
        digital=service['digital_processing_workload']/entity['digital_processing_capacity']
        eps=float(self.noise.uniform(-self.c['noise_halfwidth_s'],self.c['noise_halfwidth_s']))
        rt=transfer+digital+self.c['base_s']+eps
        assert rt>0
        return rt,dict(user_bandwidth_est_at_t0=float(ue),entity_bandwidth_est_at_t0=float(ee)),dict(network_bandwidth_actual=b,transfer_time=transfer,digital_time=digital,network_noise=eps)
