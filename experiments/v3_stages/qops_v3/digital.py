"""Exact lazy FCFS server. Background arrivals have independent random streams.

No SimPy event is required for each background completion: an arrival appends a
non-preemptive service interval. Snapshots use only arrivals no later than t.
"""
from collections import deque
import math
import numpy as np


class DigitalServer:
    def __init__(self, entity, config, seed, record=True):
        self.eid = entity['entity_id']
        self.capacity = entity['digital_processing_capacity']
        self.c = config['network']
        levels = self.c['background_levels']
        self.rho = self.c.get('background_rho_override',
                             levels[(self.eid + seed % 100) % len(levels)])
        self.rho *= self.c.get('background_rate_multiplier', 1.)
        self.mean_work = (self.c['background_work_min']+self.c['background_work_max'])/2
        self.rate = self.rho*self.capacity/self.mean_work
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 410, self.eid]))
        self.next_arrival = self.gap()
        self.next_free = 0.
        self.last_query = 0.
        self.recent = deque()
        self.log = [] if record else None
        self.counter = 0

    def gap(self):
        return float(self.rng.exponential(1/self.rate)) if self.rate else math.inf

    def append(self, t, work, invocation_id=0):
        start = max(t, self.next_free)
        end = start+work/self.capacity
        self.counter += 1
        job = (self.eid, self.counter, invocation_id, t, start, end, work)
        self.recent.append(job)
        if self.log is not None:
            self.log.append(job)
        self.next_free = end
        return job

    def advance(self, t):
        assert t >= self.last_query - 1e-8, "Digital server queried in the past"
        while self.next_arrival <= t:
            arrival = self.next_arrival
            work = float(self.rng.uniform(self.c['background_work_min'],
                                         self.c['background_work_max']))
            self.append(arrival, work)
            self.next_arrival += self.gap()
        self.last_query = t
        cutoff = max(0., t-self.c['history_window_s'])
        while self.recent and self.recent[0][5] < cutoff:
            self.recent.popleft()

    def snapshot(self, t):
        self.advance(t)
        active = next((j for j in self.recent if j[4] <= t < j[5]), None)
        waiting = sum(j[4] > t for j in self.recent)
        window = min(t, self.c['history_window_s'])
        busy = sum(max(0., min(t, j[5])-max(t-window, j[4])) for j in self.recent)
        return dict(digital_queue_length_at_t0=waiting, digital_busy_at_t0=int(active is not None),
                    digital_activity_elapsed_at_t0=0. if active is None else t-active[4],
                    digital_busy_fraction_10s_at_t0=busy/window if window else 0.)

    def submit(self, t, work, invocation_id):
        self.advance(t)
        return self.append(t, work, invocation_id)


FIELDS = ['entity_id','digital_job_id','invocation_id','arrival_time','start_time',
          'end_time','digital_work']
