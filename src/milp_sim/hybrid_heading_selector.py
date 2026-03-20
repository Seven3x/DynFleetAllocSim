from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from .cost_estimator import prioritize_goal_heading_candidates, wrap_to_pi
from .dubins_path import DubinsHybridMeta, build_dubins_hybrid_path
from .map_utils import WorldMap
from .planner_astar import AStarPlanner

Point2D = Tuple[float, float]


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

    headings = list(all_headings)
    retry_headings: List[float] = []
    retry_cap = 0
    if use_hybrid_mode:
        headings = prioritize_goal_heading_candidates(
            headings=list(all_headings),
            current_heading=start_heading,
            target_heading=target_heading,
            limit=int(getattr(cfg, "hybrid_astar_heading_candidate_limit", 2)),
        )
        retry_cap = max(0, int(getattr(cfg, "hybrid_astar_heading_candidate_retry_limit", 0)))
        retry_headings = [h for h in all_headings if all(abs(wrap_to_pi(h - k)) > 1e-4 for k in headings)]

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

    cand_debug: List[HeadingCandidateEval] = []
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
        chosen_heading = fallback_heading
        chosen_length = fallback_length
        chosen_meta = fallback_meta
        chosen_path = fallback_path

    return HeadingSelectionResult(
        chosen_heading=chosen_heading,
        chosen_path=chosen_path,
        chosen_length=chosen_length,
        chosen_meta=chosen_meta,
        cand_debug=cand_debug,
        found_non_fallback=found_non_fallback,
    )
