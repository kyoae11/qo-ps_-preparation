"""Digital and prediction-boundary checks independent of generator state flags."""
import numpy as np
from .validate import validate


def audit(sim):
    result=validate(sim)
    jobs=0
    for eid,server in sim.digital.items():
        if server.log is None:continue
        a=np.asarray(server.log,dtype=float)
        if not len(a):continue
        arrival,start,end,work=a[:,3],a[:,4],a[:,5],a[:,6]
        assert (np.diff(arrival)>=-1e-8).all()
        previous=np.concatenate([[0.],end[:-1]])
        np.testing.assert_allclose(start,np.maximum(arrival,previous),atol=1e-7)
        np.testing.assert_allclose(end-start,work/server.capacity,atol=1e-7)
        assert (start>=arrival-1e-8).all()
        cumulative=np.concatenate([[0.],np.cumsum(end-start)])
        def busy_to(t):
            completed=np.searchsorted(end,t,side='right')
            used=cumulative[completed]
            if completed<len(start):used+=max(0,t-start[completed])
            return used
        for f,r in zip(sim.inputs,sim.records):
            if f['entity_id']!=eid:continue
            t=r['t0_request_sent']
            # Only arrivals strictly before this t0, or background arrivals at t0.
            n_arr=np.searchsorted(arrival,t,side='right');n_start=np.searchsorted(start,t,side='right');n_end=np.searchsorted(end,t,side='right')
            assert f['digital_queue_length_at_t0']==n_arr-n_start
            assert f['digital_busy_at_t0']==int(n_start>n_end)
            elapsed=t-start[n_end] if n_start>n_end else 0.
            assert abs(f['digital_activity_elapsed_at_t0']-elapsed)<1e-7
            window=min(t,10.)
            expected=(busy_to(t)-busy_to(t-window))/window if window else 0.
            assert abs(f['digital_busy_fraction_10s_at_t0']-expected)<1e-6
        jobs+=len(a)
    for r in sim.records:
        if r['network_time'] is None:continue
        total=sum(r[k] for k in ['request_transfer_time','digital_wait_time','digital_time','response_transfer_time','network_overhead_time'])
        assert abs(total-r['network_time'])<1e-7
        assert r['digital_arrival_time']<=r['digital_start_time']<=r['digital_end_time']<=r['t1_network_response_received']
    result['digital_jobs_checked']=jobs
    result['network_components_checked']=sum(r['network_time'] is not None for r in sim.records)
    return result
