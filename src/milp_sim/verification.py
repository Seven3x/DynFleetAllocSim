from __future__ import annotations

from dataclasses import dataclass

from .config import SimulationConfig
from .cost_estimator import heading_to_point, wrap_to_pi
from .dubins_path import build_dubins_hybrid_path
from .entities import Task, Vehicle
from .planner_astar import AStarPlanner


@dataclass
class VerificationResult:
    passed: bool
    path_length: float
    c_tilde: float
    e_under: float


def verify_bid(
    vehicle: Vehicle,
    task: Task,
    c_hat: float,
    cfg: SimulationConfig,
    planner: AStarPlanner,
) -> VerificationResult:
    path, astar_len = planner.plan(vehicle.current_pos, task.position)
    if not path or astar_len == float("inf"):
        return VerificationResult(
            passed=False,
            path_length=float("inf"),
            c_tilde=float("inf"),
            e_under=1.0,
        )

    turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)
    target_heading = heading_to_point(vehicle.current_pos, task.position)
    if cfg.use_dubins_hybrid:
        hybrid_path, hybrid_len, _ = build_dubins_hybrid_path(
            world=planner.world,
            cfg=cfg,
            start_pose=(vehicle.current_pos[0], vehicle.current_pos[1], vehicle.current_heading),
            goal_pose=(task.position[0], task.position[1], target_heading),
            astar_planner=planner,
            turn_radius=turn_radius,
            astar_path=path,
            astar_length=astar_len,
        )
        path_length = hybrid_len if hybrid_path and hybrid_len != float("inf") else astar_len
    else:
        path_length = astar_len

    delta_heading = abs(wrap_to_pi(target_heading - vehicle.current_heading))
    corrected_length = path_length + cfg.lambda_psi * turn_radius * delta_heading
    c_tilde = corrected_length / vehicle.speed
    if c_tilde <= 1e-12:
        return VerificationResult(passed=True, path_length=path_length, c_tilde=c_tilde, e_under=0.0)

    e_under = (c_tilde - c_hat) / c_tilde
    return VerificationResult(
        passed=e_under <= cfg.verify_epsilon,
        path_length=path_length,
        c_tilde=c_tilde,
        e_under=e_under,
    )
