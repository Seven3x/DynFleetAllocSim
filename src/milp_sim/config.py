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

    vehicle_speed_min: float = 7.0
    vehicle_speed_max: float = 12.0
    vehicle_omega_min: float = 1.4
    vehicle_omega_max: float = 2.8
    vehicle_capacity_min: int = 14
    vehicle_capacity_max: int = 20

    task_demand_min: int = 1
    task_demand_max: int = 3

    lambda_psi: float = 0.15
    lambda_rho: float = 1.20
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

    # Lightweight neighborhood coordination
    comm_radius: float = 38.0
    sync_stable_h: int = 2
    sync_rmax: int = 5

    # Round-2 dynamic events
    dynamic_new_tasks: int = 3
    dynamic_cancel_tasks: int = 2

    # Online runtime simulation
    online_dt: float = 0.5
    online_replan_period_s: float = 2.0
    preempt_gain_threshold: float = 0.10

    figure_dpi: int = 130
    figure_size: tuple = (11, 10)


DEFAULT_CONFIG = SimulationConfig()
