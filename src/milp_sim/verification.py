from __future__ import annotations

from dataclasses import dataclass

from .config import SimulationConfig
from .cost_estimator import heading_to_point, wrap_to_pi
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

    tgt_heading = heading_to_point(vehicle.current_pos, task.position)
    delta = abs(wrap_to_pi(tgt_heading - vehicle.current_heading))
    turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)

    c_tilde = (astar_len + cfg.lambda_psi * turn_radius * delta) / vehicle.speed
    if c_tilde <= 1e-12:
        return VerificationResult(passed=True, path_length=astar_len, c_tilde=c_tilde, e_under=0.0)

    e_under = (c_tilde - c_hat) / c_tilde
    return VerificationResult(
        passed=e_under <= cfg.verify_epsilon,
        path_length=astar_len,
        c_tilde=c_tilde,
        e_under=e_under,
    )
