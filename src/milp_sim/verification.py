from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _committed_prefix_task_ids(
    vehicle: Vehicle,
    tasks_by_id: Dict[int, Task],
    exclude_task_id: int | None = None,
) -> List[int]:
    if vehicle.active_task_id is None:
        return []

    out: List[int] = []
    seen: set[int] = set()

    active_tid = vehicle.active_task_id
    if active_tid is not None and active_tid != exclude_task_id:
        active_task = tasks_by_id.get(active_tid)
        if active_task is not None and active_task.status in {"locked", "in_progress"}:
            out.append(active_tid)
            seen.add(active_tid)

    for tid in vehicle.task_sequence:
        if tid in seen or tid == exclude_task_id:
            continue
        task = tasks_by_id.get(tid)
        if task is None:
            continue
        if task.status not in {"locked", "in_progress"}:
            continue
        out.append(tid)
        seen.add(tid)

    return out


def _segment_corrected_time(
    vehicle: Vehicle,
    task: Task,
    start_pos: Tuple[float, float],
    start_heading: float,
    cfg: SimulationConfig,
    planner: AStarPlanner,
) -> Tuple[float, float, float]:
    path, astar_len = planner.plan(start_pos, task.position)
    if not path or astar_len == float("inf"):
        return float("inf"), float("inf"), start_heading

    turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)
    target_heading = heading_to_point(start_pos, task.position)
    if cfg.use_dubins_hybrid:
        hybrid_path, hybrid_len, _ = build_dubins_hybrid_path(
            world=planner.world,
            cfg=cfg,
            start_pose=(start_pos[0], start_pos[1], start_heading),
            goal_pose=(task.position[0], task.position[1], target_heading),
            astar_planner=planner,
            turn_radius=turn_radius,
            astar_path=path,
            astar_length=astar_len,
        )
        path_length = hybrid_len if hybrid_path and hybrid_len != float("inf") else astar_len
    else:
        path_length = astar_len

    delta_heading = abs(wrap_to_pi(target_heading - start_heading))
    corrected_length = path_length + cfg.lambda_psi * turn_radius * delta_heading
    corrected_time = corrected_length / max(vehicle.speed, 1e-9)
    return path_length, corrected_time, target_heading


def verify_bid(
    vehicle: Vehicle,
    task: Task,
    c_hat: float,
    cfg: SimulationConfig,
    planner: AStarPlanner,
    tasks_by_id: Dict[int, Task] | None = None,
) -> VerificationResult:
    path_length_total = 0.0
    c_tilde_total = 0.0
    cur_pos = vehicle.current_pos
    cur_heading = vehicle.current_heading

    prefix_ids: List[int] = []
    if tasks_by_id is not None:
        prefix_ids = _committed_prefix_task_ids(
            vehicle=vehicle,
            tasks_by_id=tasks_by_id,
            exclude_task_id=task.id,
        )

    task_chain: List[Task] = []
    for tid in prefix_ids:
        prefix_task = tasks_by_id.get(tid) if tasks_by_id is not None else None
        if prefix_task is None:
            continue
        task_chain.append(prefix_task)
    task_chain.append(task)

    for seg_task in task_chain:
        seg_len, seg_time, next_heading = _segment_corrected_time(
            vehicle=vehicle,
            task=seg_task,
            start_pos=cur_pos,
            start_heading=cur_heading,
            cfg=cfg,
            planner=planner,
        )
        if seg_time == float("inf"):
            return VerificationResult(
                passed=False,
                path_length=float("inf"),
                c_tilde=float("inf"),
                e_under=1.0,
            )
        path_length_total += seg_len
        c_tilde_total += seg_time
        cur_pos = seg_task.position
        cur_heading = next_heading

    if c_tilde_total <= 1e-12:
        return VerificationResult(passed=True, path_length=path_length_total, c_tilde=c_tilde_total, e_under=0.0)

    e_under = (c_tilde_total - c_hat) / c_tilde_total
    return VerificationResult(
        passed=e_under <= cfg.verify_epsilon,
        path_length=path_length_total,
        c_tilde=c_tilde_total,
        e_under=e_under,
    )
