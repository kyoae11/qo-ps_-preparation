"""Prediction inputs allowed by the t0 information contract (not inferred from data)."""
INPUT_COLUMNS = {
    'invocation_id','user_id','service_id','entity_id','entity_type','user_x','user_y',
    'user_ap_id_at_t0','entity_ap_id_at_t0','terminal_bandwidth_limit',
    'queue_length_at_t0','entity_status_at_t0','entity_x_at_t0','entity_y_at_t0',
    'entity_heading_at_t0','battery_level_at_t0','activity_elapsed_at_t0',
    'current_target_x_at_t0','current_target_y_at_t0',
    'user_bandwidth_est_at_t0','entity_bandwidth_est_at_t0','service_type',
    'request_bytes','response_bytes','digital_processing_workload',
    'device_bandwidth_limit','digital_processing_capacity','rotation_time_per_degree',
    'processing_capacity','base_execution_time','service_radius','mobility_speed',
    'battery_capacity','reserve_ratio','energy_per_meter','energy_per_degree',
    'base_task_energy','standby_power','distance_to_user_at_t0',
    'queued_target_distance_sum_at_t0','distance_from_last_known_target_at_t0',
}

INPUT_COLUMNS.update({
 'digital_queue_length_at_t0','digital_busy_at_t0','digital_activity_elapsed_at_t0','digital_busy_fraction_10s_at_t0',
 'activity_purpose_at_t0','activity_start_time_at_t0','active_task_target_x_at_t0','active_task_target_y_at_t0',
 'segment_target_heading_at_t0','charge_target_at_t0','active_task_present_at_t0',
 'active_route_distance_at_t0','active_route_angle_at_t0','active_route_reachable_at_t0',
 'queued_route_angle_sum_at_t0','queued_route_reachable_at_t0','angle_from_last_known_target_at_t0','last_route_reachable_at_t0'
})

INPUT_COLUMNS.update({'command_route_target_x_at_t0','command_route_target_y_at_t0'})
