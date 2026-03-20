from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from .cost_estimator import prioritize_goal_heading_candidates, wrap_to_pi
from .dubins_path import DubinsHybridMeta, build_dubins_hybrid_path
from .map_utils import WorldMap
from .planner_astar import AStarPlanner

Point2D = Tuple[float, float]


def _append_meta_reason(meta: DubinsHybridMeta | None, reason: str) -> None:
    if meta is None or not reason:
        return

    detail = str(getattr(meta, "fallback_details", "") or "")
    parts = [part.strip() for part in detail.split("|") if part.strip()]
    if reason not in parts:
        parts.append(reason)
        setattr(meta, "fallback_details", "|".join(parts))

    trace = str(getattr(meta, "debug_trace", "") or "")
    trace_token = f"selector:{reason}"
    if trace_token not in trace:
        setattr(meta, "debug_trace", f"{trace}|{trace_token}" if trace else trace_token)


@dataclass
class HeadingCandidateEval:
    heading: float
    length: float
    ok: bool
    score: float
    dpsi: float
    fallback_used: bool
    detail: str
    trace: str


@dataclass
class HeadingSelectionResult:
    chosen_heading: float
    chosen_path: List[Point2D]
    chosen_length: float
    chosen_meta: DubinsHybridMeta | None
    cand_debug: List[HeadingCandidateEval]
    found_non_fallback: bool
    selection_reason_tags: Tuple[str, ...] = ()


def select_best_heading_path(
    *,
    cfg,
    world: WorldMap,
    planner: AStarPlanner,
    start_pos: Point2D,
    start_heading: float,
    task_pos: Point2D,
    target_heading: float,
    turn_radius: float,
    all_headings: Sequence[float],
    astar_path: Optional[List[Point2D]] = None,
    astar_length: Optional[float] = None,
    build_path_fn: Callable[..., Tuple[List[Point2D], float, DubinsHybridMeta]] = build_dubins_hybrid_path,
) -> HeadingSelectionResult:
    use_hybrid_mode = bool(getattr(cfg, "use_hybrid_astar", False))
    stop_on_first_non_fallback = bool(getattr(cfg, "hybrid_astar_stop_on_first_non_fallback", True))
    primary_disable_retry = bool(getattr(cfg, "hybrid_astar_primary_disable_retry", True))
    cfg_no_hybrid_retry = cfg
    if use_hybrid_mode and primary_disable_retry and bool(getattr(cfg, "hybrid_astar_retry_on_fail", True)):
        cfg_no_hybrid_retry = replace(cfg, hybrid_astar_retry_on_fail=False)

    all_headings_list = list(all_headings)
    headings = list(all_headings_list)
    retry_headings: List[float] = []
    retry_cap = 0
    if use_hybrid_mode:
        headings = prioritize_goal_heading_candidates(
            headings=list(all_headings_list),
            current_heading=start_heading,
            target_heading=target_heading,
            limit=int(getattr(cfg, "hybrid_astar_heading_candidate_limit", 2)),
        )
        retry_cap = max(0, int(getattr(cfg, "hybrid_astar_heading_candidate_retry_limit", 0)))
        retry_headings = [h for h in all_headings_list if all(abs(wrap_to_pi(h - k)) > 1e-4 for k in headings)]

    turn_penalty = max(0.0, float(getattr(cfg, "goal_heading_turn_penalty", 0.0)))
    fallback_penalty = max(0.0, float(getattr(cfg, "hybrid_astar_fallback_penalty", 0.0))) if use_hybrid_mode else 0.0
    dpsi_limit = max(0.0, float(getattr(cfg, "goal_heading_max_dpsi_rad", 0.0)))
    dpsi_slack = max(0.0, float(getattr(cfg, "goal_heading_max_dpsi_slack_rad", 0.0)))
    dpsi_limit_eff = dpsi_limit + dpsi_slack

    best_heading = target_heading
    best_length = float("inf")
    best_score = float("inf")
    best_meta: DubinsHybridMeta | None = None
    best_path: List[Point2D] = []

    fallback_heading = target_heading
    fallback_length = float("inf")
    fallback_score = float("inf")
    fallback_meta: DubinsHybridMeta | None = None
    fallback_path: List[Point2D] = []
    soft_heading = target_heading
    soft_length = float("inf")
    soft_score = float("inf")
    soft_meta: DubinsHybridMeta | None = None
    soft_path: List[Point2D] = []

    cand_debug: List[HeadingCandidateEval] = []
    found_non_fallback = False
    pruned_by_dpsi = False

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
        nonlocal soft_heading
        nonlocal soft_length
        nonlocal soft_score
        nonlocal soft_meta
        nonlocal soft_path
        nonlocal found_non_fallback
        nonlocal pruned_by_dpsi

        planner_cfg = cfg if allow_hybrid_retry else cfg_no_hybrid_retry
        hybrid_path, hybrid_len, hybrid_meta = build_path_fn(
            world=world,
            cfg=planner_cfg,
            start_pose=(start_pos[0], start_pos[1], start_heading),
            goal_pose=(task_pos[0], task_pos[1], goal_heading),
            astar_planner=planner,
            turn_radius=turn_radius,
            astar_path=astar_path,
            astar_length=astar_length,
        )
        ok = bool(hybrid_path) and hybrid_len != float("inf")
        if not ok:
            detail = str(getattr(hybrid_meta, "fallback_details", "") or getattr(hybrid_meta, "fallback_reason", ""))
            trace = str(getattr(hybrid_meta, "debug_trace", ""))
            cand_debug.append(
                HeadingCandidateEval(
                    heading=goal_heading,
                    length=hybrid_len,
                    ok=False,
                    score=float("inf"),
                    dpsi=float("inf"),
                    fallback_used=True,
                    detail=detail,
                    trace=trace,
                )
            )
            return

        fallback_used = bool(getattr(hybrid_meta, "used_fallback", False))
        dpsi = abs(wrap_to_pi(goal_heading - start_heading))
        score = hybrid_len + turn_penalty * turn_radius * dpsi + (fallback_penalty if fallback_used else 0.0)
        detail = str(getattr(hybrid_meta, "fallback_details", "") or getattr(hybrid_meta, "fallback_reason", ""))
        trace = str(getattr(hybrid_meta, "debug_trace", ""))
        cand_debug.append(
            HeadingCandidateEval(
                heading=goal_heading,
                length=hybrid_len,
                ok=True,
                score=score,
                dpsi=dpsi,
                fallback_used=fallback_used,
                detail=detail,
                trace=trace,
            )
        )
        if score < fallback_score:
            fallback_heading = goal_heading
            fallback_length = hybrid_len
            fallback_score = score
            fallback_meta = hybrid_meta
            fallback_path = hybrid_path
        if not fallback_used:
            found_non_fallback = True
        if dpsi > dpsi_limit_eff + 1e-9:
            pruned_by_dpsi = True
            if not fallback_used:
                soft_penalty = turn_penalty * turn_radius * max(0.0, dpsi - dpsi_limit_eff) * 1.5
                soft_candidate_score = score + soft_penalty
                if soft_candidate_score < soft_score:
                    soft_heading = goal_heading
                    soft_length = hybrid_len
                    soft_score = soft_candidate_score
                    soft_meta = hybrid_meta
                    soft_path = hybrid_path
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
    chosen_length = best_length
    chosen_meta = best_meta
    chosen_path = best_path
    if not math.isfinite(chosen_length):
        if math.isfinite(soft_length):
            chosen_heading = soft_heading
            chosen_length = soft_length
            chosen_meta = soft_meta
            chosen_path = soft_path
        if not math.isfinite(chosen_length):
            chosen_heading = fallback_heading
            chosen_length = fallback_length
            chosen_meta = fallback_meta
            chosen_path = fallback_path

    selection_reason_tags: List[str] = []
    if pruned_by_dpsi or (use_hybrid_mode and len(headings) < len(all_headings_list)):
        selection_reason_tags.append("goal_heading_pruned")
    if selection_reason_tags and bool(getattr(chosen_meta, "used_fallback", False)):
        for reason in selection_reason_tags:
            _append_meta_reason(chosen_meta, reason)

    return HeadingSelectionResult(
        chosen_heading=chosen_heading,
        chosen_path=chosen_path,
        chosen_length=chosen_length,
        chosen_meta=chosen_meta,
        cand_debug=cand_debug,
        found_non_fallback=found_non_fallback,
        selection_reason_tags=tuple(selection_reason_tags),
    )
