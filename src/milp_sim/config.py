from dataclasses import dataclass


@dataclass
class SimulationConfig:
    seed: int = 7

    map_width: float = 100.0
    map_height: float = 100.0

    depot_min_x: float = 4.0
    depot_min_y: float = 4.0
    depot_max_x: float = 20.0
    depot_max_y: float = 20.0

    num_obstacles: int = 9
    obstacle_vertex_min: int = 5
    obstacle_vertex_max: int = 9
    obstacle_radius_min: float = 4.0
    obstacle_radius_max: float = 10.0
    obstacle_margin_from_border: float = 5.0
    obstacle_max_retries: int = 600

    num_vehicles: int = 4
    num_tasks: int = 18

    vehicle_radius: float = 1.0
    safety_margin: float = 0.8

    vehicle_speed_min: float = 3.5
    vehicle_speed_max: float = 6.0
    vehicle_omega_min: float = 3.5
    vehicle_omega_max: float = 6.5
    vehicle_capacity_min: int = 14
    vehicle_capacity_max: int = 20

    task_demand_min: int = 1
    task_demand_max: int = 3

    lambda_psi: float = 0.05
    lambda_rho: float = 0.20
    corridor_width: float = 8.0
    verify_epsilon: float = 0.25

    astar_resolution: float = 1.0
    astar_connect_diagonal: bool = True
    # Pre-smooth A* polyline by line-of-sight shortcut before Dubins/fillet.
    astar_smooth_before_dubins: bool = True

    # Hybrid trajectory: A* skeleton + Dubins segments
    use_dubins_hybrid: bool = True
    dubins_sample_step: float = 0.5
    dubins_collision_margin: float = 0.8
    # Safety-first defaults: if local fillet smoothing is unsafe, keep A* geometry.
    dubins_fallback_to_astar: bool = True
    # Debug-only switch; when True, it may generate paths that clip obstacles.
    dubins_force_mode: bool = False
    # Blend terminal heading between "face current task" and "face next task".
    # Smaller value reduces local looping near close targets.
    goal_heading_blend_turn_radius_factor: float = 4.0
    # Soft terminal-heading window around blended heading (radians).
    goal_heading_tolerance_rad: float = 0.9
    # Number of heading samples in the soft window (>=1).
    goal_heading_num_samples: int = 5
    # Candidate score penalty on initial heading change (rad-weighted by turn radius).
    goal_heading_turn_penalty: float = 0.35
    # Hard limit on initial heading change when selecting goal-heading candidates.
    goal_heading_max_dpsi_rad: float = 1.05
    # Small slack above dpsi hard limit to avoid rejecting near-threshold good candidates.
    goal_heading_max_dpsi_slack_rad: float = 0.12

    # Lightweight neighborhood coordination
    comm_radius: float = 38.0
    sync_stable_h: int = 2
    sync_rmax: int = 5

    # Round-2 dynamic events
    dynamic_new_tasks: int = 3
    dynamic_cancel_tasks: int = 2

    # Online runtime simulation
    online_dt: float = 0.5
    online_replan_period_s: float = 2.5
    online_new_task_replan_batch_size: int = 3
    preempt_gain_threshold: float = 0.20
    # When False, in-progress tasks are never released by soft preempt.
    online_allow_active_task_preempt: bool = False
    # Weight applied to already-committed prefix execution time during bidding/verification.
    # Set to 0 to ignore prefix cost entirely and skip prefix estimation calls.
    committed_prefix_time_weight: float = 0.0
    online_task_reach_tolerance: float = 0.25
    # Number of future tasks kept/visualized beyond the current in-progress task.
    online_future_task_horizon: int = 3
    # Planner debug: dump heading-candidate scores into event logs.
    plan_debug_enabled: bool = True
    plan_debug_vehicle_id: int = 2
    plan_debug_task_id: int = 7
    plan_debug_top_k: int = 8

    figure_dpi: int = 130
    figure_size: tuple = (11, 10)

    def __post_init__(self) -> None:
        if int(self.online_new_task_replan_batch_size) < 1:
            raise ValueError("online_new_task_replan_batch_size must be >= 1")
        if float(self.committed_prefix_time_weight) < 0.0:
            raise ValueError("committed_prefix_time_weight must be >= 0")


DEFAULT_CONFIG = SimulationConfig()
