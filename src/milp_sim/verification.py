from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .cost_estimator import goal_heading_candidates, heading_to_point, wrap_to_pi
from .dubins_path import DubinsHybridMeta, build_dubins_hybrid_path  # backward-compatible import for tests/patching
from .hybrid_heading_selector import select_best_heading_path
from .entities import Task, Vehicle
from .path_postprocess import maybe_buffer_initial_turn_path
from .planner_astar import AStarPlanner


@dataclass
class VerificationResult:
    passed: bool
    path_length: float
    c_tilde: float
    e_under: float
    dubins_used_fallback: bool = False
    dubins_fallback_details: str = ""
    debug_message: str = ""


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
    next_task_pos: Tuple[float, float] | None,
    cfg: SimulationConfig,
    planner: AStarPlanner,
) -> Tuple[float, float, float, DubinsHybridMeta | None, str]:
    path, astar_len = planner.plan(start_pos, task.position)
    if not path or astar_len == float("inf"):
        return float("inf"), float("inf"), start_heading, None, ""

    turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)
    target_heading = heading_to_point(start_pos, task.position)
    if not cfg.use_dubins_hybrid:
        delta_heading = abs(wrap_to_pi(target_heading - start_heading))
        corrected_length = astar_len + cfg.lambda_psi * turn_radius * delta_heading
        corrected_time = corrected_length / max(vehicle.speed, 1e-9)
        return astar_len, corrected_time, target_heading, None, ""

    all_headings = goal_heading_candidates(
        current_pos=start_pos,
        task_pos=task.position,
        next_task_pos=next_task_pos,
        turn_radius=turn_radius,
        blend_turn_radius_factor=cfg.goal_heading_blend_turn_radius_factor,
        tolerance_rad=cfg.goal_heading_tolerance_rad,
        num_samples=cfg.goal_heading_num_samples,
    )
    heading_sel = select_best_heading_path(
        cfg=cfg,
        world=planner.world,
        planner=planner,
        start_pos=start_pos,
        start_heading=start_heading,
        task_pos=task.position,
        target_heading=target_heading,
        turn_radius=turn_radius,
        all_headings=all_headings,
        astar_path=path,
        astar_length=astar_len,
        build_path_fn=build_dubins_hybrid_path,
    )
    chosen_heading = heading_sel.chosen_heading
    path_length = heading_sel.chosen_length
    hybrid_meta = heading_sel.chosen_meta
    chosen_path = heading_sel.chosen_path
    cand_debug = heading_sel.cand_debug
    if not math.isfinite(path_length):
        return float("inf"), float("inf"), start_heading, None, ""

    if chosen_path:
        _, path_length = maybe_buffer_initial_turn_path(
            world=planner.world,
            cfg=cfg,
            planner=planner,
            start_pos=start_pos,
            start_heading=start_heading,
            task_pos=task.position,
            path=chosen_path,
            length=path_length,
            goal_heading=chosen_heading,
            turn_radius=turn_radius,
        )

    delta_heading = abs(wrap_to_pi(chosen_heading - start_heading))
    corrected_length = path_length + cfg.lambda_psi * turn_radius * delta_heading
    corrected_time = corrected_length / max(vehicle.speed, 1e-9)
    debug_message = ""
    debug_enabled = bool(getattr(cfg, "plan_debug_enabled", False))
    target_vid = int(getattr(cfg, "plan_debug_vehicle_id", -1))
    target_tid = int(getattr(cfg, "plan_debug_task_id", -1))
    should_emit_debug = debug_enabled and (target_vid < 0 or target_vid == vehicle.id) and (
        target_tid < 0 or target_tid == task.id
    )
    if should_emit_debug and hybrid_meta is not None:
        ranked = sorted(
            cand_debug,
            key=lambda x: (0 if x.ok else 1, x.score if math.isfinite(x.score) else float("inf")),
        )[: max(1, int(getattr(cfg, "plan_debug_top_k", 5)))]
        parts: List[str] = []
        for cand in ranked:
            h = cand.heading
            l = cand.length
            ok = cand.ok
            score = cand.score
            dpsi = cand.dpsi
            fallback_used = cand.fallback_used
            detail = cand.detail
            trace = cand.trace
            ls = f"{l:.3f}" if math.isfinite(l) else "inf"
            ss = f"{score:.3f}" if math.isfinite(score) else "inf"
            dpsi_deg = math.degrees(dpsi) if math.isfinite(dpsi) else float("inf")
            fb = detail if detail else "-"
            tr = trace if trace else "-"
            parts.append(
                f"h={math.degrees(h):.1f} len={ls} score={ss} "
                f"dpsi={dpsi_deg:.1f} ok={ok} fallback={fallback_used} detail={fb} trace={tr}"
            )
        chosen_detail = str(
            getattr(hybrid_meta, "fallback_details", "")
            or getattr(hybrid_meta, "fallback_reason", "")
        )
        chosen_trace = str(getattr(hybrid_meta, "debug_trace", ""))
        connector_type = str(getattr(hybrid_meta, "connector_type", "") or "-")
        connector_summary = str(getattr(hybrid_meta, "connector_summary", "") or "-")
        debug_message = (
            f"verify_plan_debug task=T{task.id} "
            f"from=({start_pos[0]:.3f},{start_pos[1]:.3f},{math.degrees(start_heading):.1f}deg) "
            f"to=({task.position[0]:.3f},{task.position[1]:.3f}) "
            f"choose_h={math.degrees(chosen_heading):.1f} len={path_length:.3f} "
            f"connector={connector_type} connector_summary={connector_summary} "
            f"fallback_detail={chosen_detail or '-'} trace={chosen_trace or '-'} "
            f"cand[{len(cand_debug)}]: " + " | ".join(parts)
        )
    return path_length, corrected_time, chosen_heading, hybrid_meta, debug_message


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
    fallback_parts: List[str] = []
    debug_parts: List[str] = []
    prefix_weight = max(0.0, float(getattr(cfg, "committed_prefix_time_weight", 1.0)))

    prefix_ids: List[int] = []
    if tasks_by_id is not None and prefix_weight > 1e-12:
        prefix_ids = _committed_prefix_task_ids(
            vehicle=vehicle,
            tasks_by_id=tasks_by_id,
            exclude_task_id=task.id,
        )
    task_chain: List[Task] = []
    for tid in prefix_ids:
        prefix_task = tasks_by_id.get(tid) if tasks_by_id is not None else None
        if prefix_task is not None:
            task_chain.append(prefix_task)
    task_chain.append(task)

    for idx, chain_task in enumerate(task_chain):
        next_task_pos = None
        if idx + 1 < len(task_chain):
            next_task_pos = task_chain[idx + 1].position
        seg_len, seg_time, next_heading, hybrid_meta, debug_message = _segment_corrected_time(
            vehicle=vehicle,
            task=chain_task,
            start_pos=cur_pos,
            start_heading=cur_heading,
            next_task_pos=next_task_pos,
            cfg=cfg,
            planner=planner,
        )
        if seg_time == float("inf"):
            return VerificationResult(
                passed=False,
                path_length=float("inf"),
                c_tilde=float("inf"),
                e_under=1.0,
                dubins_used_fallback=bool(fallback_parts),
                dubins_fallback_details=";".join(fallback_parts),
                debug_message="\n".join(debug_parts),
            )
        path_length_total += seg_len
        if idx + 1 < len(task_chain):
            c_tilde_total += prefix_weight * seg_time
        else:
            c_tilde_total += seg_time
        if hybrid_meta is not None and getattr(hybrid_meta, "used_fallback", False):
            detail = str(
                getattr(hybrid_meta, "fallback_details", "")
                or getattr(hybrid_meta, "fallback_reason", "fallback")
            )
            fallback_parts.append(f"T{chain_task.id}:{detail}")
        if debug_message:
            debug_parts.append(debug_message)
        cur_pos = chain_task.position
        cur_heading = next_heading

    if c_tilde_total <= 1e-12:
        return VerificationResult(
            passed=True,
            path_length=path_length_total,
            c_tilde=c_tilde_total,
            e_under=0.0,
            dubins_used_fallback=bool(fallback_parts),
            dubins_fallback_details=";".join(fallback_parts),
            debug_message="\n".join(debug_parts),
        )

    e_under = (c_tilde_total - c_hat) / c_tilde_total
    return VerificationResult(
        passed=e_under <= cfg.verify_epsilon,
        path_length=path_length_total,
        c_tilde=c_tilde_total,
        e_under=e_under,
        dubins_used_fallback=bool(fallback_parts),
        dubins_fallback_details=";".join(fallback_parts),
        debug_message="\n".join(debug_parts),
    )
