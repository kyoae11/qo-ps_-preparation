"""Features reconstructed solely from the current observable command and queue."""
import math
from .world import delta_heading


def route_summary(world, start, heading, target, entity):
    route = world.route(start, heading, target, entity)
    if route is None:
        return None
    return (sum(s['distance'] for s in route), sum(s['angle'] for s in route),
            route[-1]['end_heading'] if route else heading)


def observe(device, user):
    w = device.sim.world
    t = device.env.now
    s = device.snapshot(t)
    phase = device.phase
    active = device.active
    q = [dict(invocation_id=r['invocation_id'], user_x=r['user_x'],
              user_y=r['user_y']) for r in device.queue]
    purpose = phase['purpose'] if phase else 'idle'
    goal = ((active['user_x'], active['user_y']) if active else
            (device.route_destination or (s['target_x'], s['target_y'])))
    f = dict(activity_purpose_at_t0=purpose,
             activity_start_time_at_t0=phase['start_time'] if phase else device.last,
             active_task_target_x_at_t0=goal[0], active_task_target_y_at_t0=goal[1],
             segment_target_heading_at_t0=phase['end_heading'] if phase else s['heading'],
             charge_target_at_t0=device.charge_target,
             active_task_present_at_t0=int(active is not None))
    # Continue an issued movement/rotation to its known endpoint. Replanning
    # a fractional cell by rounding would lose the unfinished segment.
    start = (s['x'], s['y']); h = s['heading']
    distance = angle = 0.
    if phase and phase['kind'] in ('move', 'rotate'):
        endpoint = (phase['end_x'], phase['end_y'])
        distance += math.dist(start, endpoint) if phase['kind'] == 'move' else 0.
        angle += abs(delta_heading(h, phase['end_heading'])) if phase['kind'] == 'rotate' else 0.
        start, h = endpoint, phase['end_heading']
    route_goal=device.route_destination or start
    f.update(command_route_target_x_at_t0=route_goal[0],command_route_target_y_at_t0=route_goal[1])
    reachable = True
    if device.dynamic:
        if purpose in ('task_charge_route','idle_maintenance') and phase and phase['kind'] in ('move','rotate'):
            leg=route_summary(w,start,h,route_goal,device.e)
            if leg is not None:
                distance+=leg[0];angle+=leg[1];start=route_goal;h=leg[2]
            else:reachable=False
        summary = route_summary(w, start, h, goal, device.e)
        if summary is None:
            reachable = False
        else:
            distance += summary[0]; angle += summary[1]; h = summary[2]
    else:
        if active and s['status'] != 'execute':
            final_h = math.degrees(math.atan2(goal[1]-start[1], goal[0]-start[0])) % 360
            angle += abs(delta_heading(h, final_h)); h = final_h
    f.update(active_route_distance_at_t0=distance if reachable else None,
             active_route_angle_at_t0=angle if reachable else None,
             active_route_reachable_at_t0=int(reachable))
    previous = goal
    qdistance = qangle = 0.
    qreachable = reachable
    for item in q:
        target = (item['user_x'], item['user_y'])
        if device.dynamic:
            summary = route_summary(w, previous, h, target, device.e)
            if summary is None:
                qreachable = False
                break
            qdistance += summary[0]; qangle += summary[1]; h = summary[2]
            previous = target
        else:
            target_h = math.degrees(math.atan2(target[1]-device.position[1],
                                               target[0]-device.position[0])) % 360
            qangle += abs(delta_heading(h, target_h)); h = target_h
    target = (user['user_x'], user['user_y'])
    last = route_summary(w, previous, h, target, device.e) if device.dynamic and qreachable else None
    f.update(queued_target_distance_sum_at_t0=qdistance if qreachable else None,
             queued_route_angle_sum_at_t0=qangle if qreachable else None,
             queued_route_reachable_at_t0=int(qreachable),
             distance_from_last_known_target_at_t0=last[0] if last else (0. if not device.dynamic else None),
             angle_from_last_known_target_at_t0=last[1] if last else (0. if not device.dynamic else None),
             last_route_reachable_at_t0=int(last is not None or not device.dynamic))
    return f, dict(invocation_id=None, t0=t,
                   active_invocation_id=None if active is None else active['invocation_id'],
                   active_target=list(goal), observable_charge_target=device.charge_target,
                   status=s['status'], purpose=purpose, route_destination=list(route_goal), waiting=q)
