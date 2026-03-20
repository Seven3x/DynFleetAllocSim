from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .cost_estimator import goal_heading_candidates, heading_to_point, prioritize_goal_heading_candidates, wrap_to_pi
from .dubins_path import DubinsHybridMeta, build_dubins_hybrid_path
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
    use_hybrid_mode = bool(getattr(cfg, "use_hybrid_astar", False))
    stop_on_first_non_fallback = bool(getattr(cfg, "hybrid_astar_stop_on_first_non_fallback", True))
    primary_disable_retry = bool(getattr(cfg, "hybrid_astar_primary_disable_retry", True))
    cfg_no_hybrid_retry = cfg
    if use_hybrid_mode and primary_disable_retry and bool(getattr(cfg, "hybrid_astar_retry_on_fail", True)):
        cfg_no_hybrid_retry = replace(cfg, hybrid_astar_retry_on_fail=False)
    headings = list(all_headings)
    retry_headings: List[float] = []
    retry_cap = 0
    if use_hybrid_mode:
        headings = prioritize_goal_heading_candidates(
            headings=all_headings,
            current_heading=start_heading,
            target_heading=target_heading,
            limit=int(getattr(cfg, "hybrid_astar_heading_candidate_limit", 2)),
        )
        retry_cap = max(0, int(getattr(cfg, "hybrid_astar_heading_candidate_retry_limit", 0)))
        retry_headings = [
            h
            for h in all_headings
            if all(abs(wrap_to_pi(h - k)) > 1e-4 for k in headings)
        ]
    turn_penalty = max(0.0, float(getattr(cfg, "goal_heading_turn_penalty", 0.0)))
    fallback_penalty = (
        max(0.0, float(getattr(cfg, "hybrid_astar_fallback_penalty", 0.0))) if use_hybrid_mode else 0.0
    )
    dpsi_limit = max(0.0, float(getattr(cfg, "goal_heading_max_dpsi_rad", 0.0)))
    dpsi_slack = max(0.0, float(getattr(cfg, "goal_heading_max_dpsi_slack_rad", 0.0)))
    dpsi_limit_eff = dpsi_limit + dpsi_slack

    best_heading = target_heading
    best_length = float("inf")
    best_score = float("inf")
    best_meta: DubinsHybridMeta | None = None
    best_path: List[Tuple[float, float]] = []

    fallback_heading = target_heading
    fallback_length = float("inf")
    fallback_score = float("inf")
    fallback_meta: DubinsHybridMeta | None = None
    fallback_path: List[Tuple[float, float]] = []
    cand_debug: List[Tuple[float, float, bool, float, float, bool, str, str]] = []
    found_non_fallback = False

    def _evaluate_heading(goal_heading: float, *, allow_hybrid_retry: bool = False) -> None:
        nonlocal fallback_heading
        nonlocal fallback_length
        nonlocal fallback_score
        nonlocal fallback_meta
        nonlocal fallback_path
        nonlocal best_heading
        nonlocal best_length
        nonlocal best_score
        nonlocal best_meta
        nonlocal best_path
        nonlocal found_non_fallback

        planner_cfg = cfg if allow_hybrid_retry else cfg_no_hybrid_retry
        hybrid_path, hybrid_len, hybrid_meta = build_dubins_hybrid_path(
            world=planner.world,
            cfg=planner_cfg,
            start_pose=(start_pos[0], start_pos[1], start_heading),
            goal_pose=(task.position[0], task.position[1], goal_heading),
            astar_planner=planner,
            turn_radius=turn_radius,
            astar_path=path,
            astar_length=astar_len,
        )
        ok = bool(hybrid_path) and hybrid_len != float("inf")
        if not ok:
            detail = str(
                getattr(hybrid_meta, "fallback_details", "")
                or getattr(hybrid_meta, "fallback_reason", "")
            )
            trace = str(getattr(hybrid_meta, "debug_trace", ""))
            cand_debug.append((goal_heading, hybrid_len, False, float("inf"), float("inf"), True, detail, trace))
            return

        fallback_used = bool(getattr(hybrid_meta, "used_fallback", False))
        dpsi = abs(wrap_to_pi(goal_heading - start_heading))
        score = hybrid_len + turn_penalty * turn_radius * dpsi + (fallback_penalty if fallback_used else 0.0)
        detail = str(
            getattr(hybrid_meta, "fallback_details", "")
            or getattr(hybrid_meta, "fallback_reason", "")
        )
        trace = str(getattr(hybrid_meta, "debug_trace", ""))
        cand_debug.append((goal_heading, hybrid_len, True, score, dpsi, fallback_used, detail, trace))
        if score < fallback_score:
            fallback_heading = goal_heading
            fallback_length = hybrid_len
            fallback_score = score
            fallback_meta = hybrid_meta
            fallback_path = hybrid_path
        if not fallback_used:
            found_non_fallback = True
        if dpsi > dpsi_limit_eff + 1e-9:
            return
        if score < best_score:
            best_heading = goal_heading
            best_length = hybrid_len
            best_score = score
            best_meta = hybrid_meta
            best_path = hybrid_path

    for goal_heading in headings:
        _evaluate_heading(goal_heading)
        if stop_on_first_non_fallback and found_non_fallback and math.isfinite(best_score):
            break
    if use_hybrid_mode and (not found_non_fallback) and retry_cap > 0:
        for goal_heading in retry_headings[:retry_cap]:
            _evaluate_heading(goal_heading)
            if stop_on_first_non_fallback and found_non_fallback and math.isfinite(best_score):
                break

    if (
        use_hybrid_mode
        and primary_disable_retry
        and (not found_non_fallback)
        and bool(getattr(cfg, "hybrid_astar_retry_on_fail", True))
        and math.isfinite(fallback_score)
    ):
        robust_cap = max(1, int(getattr(cfg, "hybrid_astar_robust_retry_headings", 1)))
        robust_headings: List[float] = [fallback_heading]
        for h in headings + retry_headings:
            if all(abs(wrap_to_pi(h - k)) > 1e-4 for k in robust_headings):
                robust_headings.append(h)
        for goal_heading in robust_headings[:robust_cap]:
            _evaluate_heading(goal_heading, allow_hybrid_retry=True)
            if found_non_fallback and math.isfinite(best_score):
                break

    chosen_heading = best_heading
    path_length = best_length
    hybrid_meta = best_meta
    chosen_path = best_path
    if not math.isfinite(path_length):
        chosen_heading = fallback_heading
        path_length = fallback_length
        hybrid_meta = fallback_meta
        chosen_path = fallback_path
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
    if should_emit_debug and hybrid_meta is not None and bool(getattr(hybrid_meta, "used_fallback", False)):
        ranked = sorted(
            cand_debug,
            key=lambda x: (0 if x[2] else 1, x[3] if math.isfinite(x[3]) else float("inf")),
        )[: max(1, int(getattr(cfg, "plan_debug_top_k", 5)))]
        parts: List[str] = []
        for h, l, ok, score, dpsi, fallback_used, detail, trace in ranked:
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
        debug_message = (
            f"verify_plan_debug task=T{task.id} "
            f"from=({start_pos[0]:.3f},{start_pos[1]:.3f},{math.degrees(start_heading):.1f}deg) "
            f"to=({task.position[0]:.3f},{task.position[1]:.3f}) "
            f"choose_h={math.degrees(chosen_heading):.1f} len={path_length:.3f} "
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
