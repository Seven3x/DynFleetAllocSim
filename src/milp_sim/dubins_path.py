from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from shapely.geometry import LineString

from .cost_estimator import heading_to_point, wrap_to_pi
from .map_utils import WorldMap
from .path_postprocess import simplify_reference_polyline
from .planner_astar import AStarPlanner, HybridPlanDiagnostics

try:
    import rsplan
except ImportError:  # pragma: no cover - validated by behavior when dependency is missing
    rsplan = None  # type: ignore[assignment]


Point2D = Tuple[float, float]
Pose2D = Tuple[float, float, float]

DUBINS_WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


@dataclass
class DubinsHybridMeta:
    used_fallback: bool
    fallback_segments: int
    dubins_segments: int
    sample_count: int
    dubins_ratio: float
    fallback_reason: str = ""
    fallback_details: str = ""
    debug_trace: str = ""
    connector_type: str = ""
    connector_summary: str = ""


def _mod2pi(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _lsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p2 = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    if p2 < 0.0:
        return None
    tmp = math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) - math.sin(beta))
    t = _mod2pi(-alpha + tmp)
    p = math.sqrt(p2)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _rsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p2 = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(beta) - math.sin(alpha))
    if p2 < 0.0:
        return None
    tmp = math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    t = _mod2pi(alpha - tmp)
    p = math.sqrt(p2)
    q = _mod2pi(-beta + tmp)
    return t, p, q


def _lsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p2 = -2.0 + d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p2 < 0.0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(
        -2.0, p
    )
    t = _mod2pi(-alpha + tmp)
    q = _mod2pi(-beta + tmp)
    return t, p, q


def _rsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p2 = -2.0 + d * d + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p2 < 0.0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _rlr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(
        alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + 0.5 * p
    )
    q = _mod2pi(alpha - beta - t + p)
    return t, p, q


def _lrl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(
        -alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + 0.5 * p
    )
    q = _mod2pi(beta - alpha - t + p)
    return t, p, q


_WORD_SOLVERS: Dict[str, Callable[[float, float, float], Optional[Tuple[float, float, float]]]] = {
    "LSL": _lsl,
    "RSR": _rsr,
    "LSR": _lsr,
    "RSL": _rsl,
    "RLR": _rlr,
    "LRL": _lrl,
}


def solve_dubins_word(word: str, alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    solver = _WORD_SOLVERS[word]
    return solver(alpha, beta, d)


_DUBINSMANEUVER_CTRL_WORDS: Dict[str, Tuple[str, ...]] = {
    "L": ("LSL", "LSR", "LRL"),
    "LS": ("LSL", "LSR"),
    "R": ("RSR", "RSL", "RLR"),
    "RS": ("RSR", "RSL"),
    "*-L": ("LSL", "RSR", "LSR", "RSL", "LRL"),
    "*-R": ("LSL", "RSR", "LSR", "RSL", "RLR"),
    "S": ("LSL", "RSR", "LSR", "RSL"),
    "*": DUBINS_WORDS,
}


def _dubinsmaneuver_words(ctrl_dir: str) -> Tuple[str, ...]:
    token = str(ctrl_dir).strip().upper() if ctrl_dir is not None else "*"
    if not token:
        token = "*"
    return _DUBINSMANEUVER_CTRL_WORDS.get(token, DUBINS_WORDS)


def _generate_dubinsmaneuver_local_course(
    params: Tuple[float, float, float],
    word: str,
    turn_radius: float,
) -> List[Point2D]:
    x = 0.0
    y = 0.0
    yaw = 0.0
    points: List[Point2D] = [(0.0, 0.0)]
    step = math.radians(6.0)

    for mode, seg_len_norm in zip(word, params):
        seg_len = abs(seg_len_norm)
        progressed = 0.0
        while progressed < max(0.0, seg_len - step):
            dl = step
            x += dl * turn_radius * math.cos(yaw)
            y += dl * turn_radius * math.sin(yaw)
            if mode == "L":
                yaw += dl
            elif mode == "R":
                yaw -= dl
            points.append((x, y))
            progressed += dl

        dl = max(0.0, seg_len - progressed)
        x += dl * turn_radius * math.cos(yaw)
        y += dl * turn_radius * math.sin(yaw)
        if mode == "L":
            yaw += dl
        elif mode == "R":
            yaw -= dl
        points.append((x, y))

    return points


def _build_dubinsmaneuver2d_candidate(
    start_pose: Pose2D,
    goal_pose: Pose2D,
    turn_radius: float,
    ctrl_dir: str,
) -> Optional[Tuple[str, Tuple[float, float, float], List[Point2D]]]:
    sx, sy, syaw = start_pose
    gx, gy, gyaw = goal_pose
    dx = gx - sx
    dy = gy - sy

    radius = max(float(turn_radius), 1e-9)
    d = math.hypot(dx, dy) / radius
    theta = _mod2pi(math.atan2(dy, dx))
    alpha = _mod2pi(syaw - theta)
    beta = _mod2pi(gyaw - theta)

    best_word: Optional[str] = None
    best_params: Optional[Tuple[float, float, float]] = None
    best_cost = float("inf")
    for word in _dubinsmaneuver_words(ctrl_dir):
        out = solve_dubins_word(word, alpha=alpha, beta=beta, d=d)
        if out is None:
            continue
        t, p, q = out
        cost = radius * (abs(t) + abs(p) + abs(q))
        if cost < best_cost:
            best_cost = cost
            best_word = word
            best_params = (t, p, q)

    if best_word is None or best_params is None:
        return None

    local_points = _generate_dubinsmaneuver_local_course(best_params, best_word, radius)
    c = math.cos(syaw)
    s = math.sin(syaw)
    global_points: List[Point2D] = []
    for lx, ly in local_points:
        gx = c * lx - s * ly + sx
        gy = s * lx + c * ly + sy
        global_points.append((gx, gy))

    points = _dedupe_points(global_points)
    if len(points) < 2:
        return None
    points[0] = (float(sx), float(sy))
    points[-1] = (float(goal_pose[0]), float(goal_pose[1]))
    return best_word, best_params, points


def _build_shortest_dubins_local(
    start_pose: Pose2D,
    goal_pose: Pose2D,
    turn_radius: float,
) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    sx, sy, syaw = start_pose
    gx, gy, gyaw = goal_pose
    dx = gx - sx
    dy = gy - sy
    c = math.cos(syaw)
    s = math.sin(syaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    lyaw = wrap_to_pi(gyaw - syaw)

    d = math.hypot(lx, ly) / max(turn_radius, 1e-9)
    theta = _mod2pi(math.atan2(ly, lx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(lyaw - theta)

    best_word: Optional[str] = None
    best_params: Optional[Tuple[float, float, float]] = None
    best_cost = float("inf")

    for word in DUBINS_WORDS:
        out = solve_dubins_word(word, alpha, beta, d)
        if out is None:
            continue
        t, p, q = out
        cost = t + p + q
        if cost < best_cost:
            best_cost = cost
            best_word = word
            best_params = (t, p, q)

    if best_word is None or best_params is None:
        return None
    return best_word, best_params


def _sample_dubins_segment(
    start_pose: Pose2D,
    word: str,
    params: Tuple[float, float, float],
    turn_radius: float,
    sample_step: float,
) -> List[Point2D]:
    x = 0.0
    y = 0.0
    yaw = 0.0
    points: List[Point2D] = [(0.0, 0.0)]
    inv_radius = 1.0 / max(turn_radius, 1e-9)
    step_norm = max(sample_step * inv_radius, 1e-4)

    for seg_len_norm, mode in zip(params, word):
        remain = seg_len_norm
        while remain > 1e-10:
            dl = min(step_norm, remain)
            if mode == "S":
                ds = dl / inv_radius
                x = x + ds * math.cos(yaw)
                y = y + ds * math.sin(yaw)
            else:
                if mode == "L":
                    center_x = x - turn_radius * math.sin(yaw)
                    center_y = y + turn_radius * math.cos(yaw)
                    yaw2 = yaw + dl
                    x = center_x + turn_radius * math.sin(yaw2)
                    y = center_y - turn_radius * math.cos(yaw2)
                    yaw = yaw2
                else:
                    center_x = x + turn_radius * math.sin(yaw)
                    center_y = y - turn_radius * math.cos(yaw)
                    yaw2 = yaw - dl
                    x = center_x - turn_radius * math.sin(yaw2)
                    y = center_y + turn_radius * math.cos(yaw2)
                    yaw = yaw2
            points.append((x, y))
            remain -= dl

    sx, sy, syaw = start_pose
    c = math.cos(syaw)
    s = math.sin(syaw)
    global_points: List[Point2D] = []
    for px, py in points:
        gx = c * px - s * py + sx
        gy = s * px + c * py + sy
        global_points.append((gx, gy))
    return global_points


def _collision_free(points: List[Point2D], world: WorldMap, margin: float) -> bool:
    for p in points:
        if not world.point_is_free(p, clearance=margin):
            return False
    return True


def _polyline_length(points: List[Point2D]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    return total


def _dedupe_points(points: List[Point2D]) -> List[Point2D]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def _segment_collision_free(a: Point2D, b: Point2D, world: WorldMap, margin: float) -> bool:
    line = LineString([a, b])
    if margin > 0.0:
        swept = line.buffer(margin, cap_style=2, join_style=2)
        return not swept.intersects(world.obstacle_union)
    return not line.intersects(world.obstacle_union)


def _polyline_collision_free(points: List[Point2D], world: WorldMap, margin: float) -> bool:
    if len(points) < 2:
        return True
    line = LineString(points)
    if margin > 0.0:
        swept = line.buffer(margin, cap_style=2, join_style=2)
        return not swept.intersects(world.obstacle_union)
    return not line.intersects(world.obstacle_union)


def _format_reason_counts(reason_counts: Dict[str, int]) -> str:
    if not reason_counts:
        return ""
    return "|".join(f"{reason}:{reason_counts[reason]}" for reason in sorted(reason_counts))


def _primary_reason(reason_counts: Dict[str, int]) -> str:
    if not reason_counts:
        return ""
    return sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _bump_reason(reason_counts: Dict[str, int], reason: str) -> None:
    if not reason:
        return
    reason_counts[reason] = reason_counts.get(reason, 0) + 1


def _reason_tokens(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.split("|"):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            head, tail = token.rsplit(":", 1)
            if tail.isdigit():
                token = head.strip()
        if token and token not in out:
            out.append(token)
    return out


def _append_reason_tokens(text: str, *tokens: str) -> str:
    out = _reason_tokens(text)
    for token in tokens:
        tok = str(token).strip()
        if not tok or tok in out:
            continue
        out.append(tok)
    return "|".join(out)


def _append_trace_tokens(text: str, *tokens: str) -> str:
    out: List[str] = []
    for raw in str(text).split(";"):
        token = raw.strip()
        if token and token not in out:
            out.append(token)
    for token in tokens:
        tok = str(token).strip()
        if tok and tok not in out:
            out.append(tok)
    return ";".join(out)


def _primary_reason_token(text: str, default: str = "") -> str:
    tokens = _reason_tokens(text)
    return tokens[0] if tokens else default


def _stage_reason_tags(diag: HybridPlanDiagnostics, stage: str) -> List[str]:
    tags: List[str] = []
    if stage == "primary":
        tags.append("unreachable_main_search")
    elif stage == "retry":
        tags.append("unreachable_after_retry")
    for tag in getattr(diag, "reason_tags", ()) or ():
        if tag not in tags:
            tags.append(tag)
    if not tags and getattr(diag, "reason", ""):
        tags.append(str(diag.reason))
    return tags


def _bbox_from_points(points: List[Point2D], padding: float) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = max(0.0, float(padding))
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _path_terminal_heading(points: List[Point2D], default_heading: float) -> float:
    for i in range(len(points) - 1, 0, -1):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        if math.hypot(dx, dy) > 1e-9:
            return math.atan2(dy, dx)
    return default_heading


def _heading_error(a: float, b: float) -> float:
    return abs(wrap_to_pi(a - b))


def _blend_heading_soft(current_heading: float, next_heading: float) -> float:
    return wrap_to_pi(current_heading + 0.5 * wrap_to_pi(next_heading - current_heading))


def _connector_detour_too_large(
    *,
    candidate_length: float,
    reference_length: float,
    cfg,
) -> bool:
    return _detour_too_large(
        candidate_length=candidate_length,
        reference_length=reference_length,
        ratio_limit=max(1.0, float(getattr(cfg, "connector_max_detour_ratio_vs_reference", 1.35))),
        abs_limit=max(0.0, float(getattr(cfg, "connector_max_detour_abs_vs_reference", 12.0))),
    )


def _detour_too_large(
    *,
    candidate_length: float,
    reference_length: float,
    ratio_limit: float,
    abs_limit: float,
) -> bool:
    if not math.isfinite(candidate_length) or not math.isfinite(reference_length):
        return False
    if reference_length <= 1e-9:
        return False
    return candidate_length > reference_length * ratio_limit + abs_limit


def _try_straight_connector(
    *,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    world: WorldMap,
    margin: float,
    heading_tolerance: float,
    enforce_goal_heading: bool = True,
) -> Tuple[List[Point2D], float, str]:
    line_heading = heading_to_point((start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1]))
    start_err = _heading_error(line_heading, start_pose[2])
    goal_err = _heading_error(goal_pose[2], line_heading)
    if enforce_goal_heading:
        if max(start_err, goal_err) > max(0.0, float(heading_tolerance)) + 1e-9:
            return [], float("inf"), "connector_heading_reject"
    elif start_err > max(0.0, float(heading_tolerance)) + 1e-9:
        return [], float("inf"), "connector_heading_reject"

    points = [(start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1])]
    if not _polyline_collision_free(points, world=world, margin=margin):
        return [], float("inf"), "connector_collision_reject"
    return points, _polyline_length(points), "connector_straight_success"


def _resample_polyline_max_step(points: List[Point2D], max_step: float) -> List[Point2D]:
    if len(points) < 2 or max_step <= 1e-9:
        return list(points)
    out: List[Point2D] = [points[0]]
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg_len <= 1e-9:
            continue
        n = max(1, int(math.ceil(seg_len / max_step)))
        for k in range(1, n + 1):
            t = k / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
                out.append(p)
    return out


def build_rsplan_connector(
    *,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    turn_radius: float,
    world: WorldMap,
    cfg,
    margin: float,
    require_terminal_heading: bool,
) -> Tuple[List[Point2D], float, str]:
    if rsplan is None:
        return [], float("inf"), "rsplan_import_unavailable"

    radius_cfg = float(getattr(cfg, "rsplan_turn_radius", 0.0))
    radius = max(1e-6, radius_cfg if radius_cfg > 1e-9 else float(turn_radius))
    step_size = max(0.02, float(getattr(cfg, "rsplan_step_size", max(0.1, getattr(cfg, "dubins_sample_step", 0.5)))))
    runway_length = float(getattr(cfg, "rsplan_runway_length", 0.0))
    length_tolerance = max(0.0, float(getattr(cfg, "rsplan_length_tolerance", 2.0)))
    collision_step = max(
        0.02,
        float(
            getattr(
                cfg,
                "rsplan_collision_sample_step",
                min(step_size, max(0.1, getattr(cfg, "dubins_sample_step", 0.5))),
            )
        ),
    )

    try:
        rs_path = rsplan.path(
            start_pose=(float(start_pose[0]), float(start_pose[1]), float(start_pose[2])),
            end_pose=(float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])),
            turn_radius=radius,
            runway_length=runway_length,
            step_size=step_size,
            length_tolerance=length_tolerance,
        )
        rs_waypoints = list(rs_path.waypoints())
    except Exception:
        return [], float("inf"), "rsplan_generation_failed"

    if len(rs_waypoints) < 2:
        return [], float("inf"), "rsplan_generation_failed"

    points = _dedupe_points([(float(wp.x), float(wp.y)) for wp in rs_waypoints])
    if len(points) < 2:
        return [], float("inf"), "rsplan_generation_failed"

    points[0] = (float(start_pose[0]), float(start_pose[1]))
    points[-1] = (float(goal_pose[0]), float(goal_pose[1]))
    sampled_points = _resample_polyline_max_step(points, max_step=collision_step)
    if (not _collision_free(sampled_points, world=world, margin=margin)) or (
        not _polyline_collision_free(points, world=world, margin=margin)
    ):
        return [], float("inf"), "rsplan_collision_reject"
    return points, _polyline_length(points), "connector_rsplan_success"


def _try_dubins_connector(
    *,
    cfg,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    turn_radius: float,
    sample_step: float,
    world: WorldMap,
    margin: float,
) -> Tuple[List[Point2D], float, str]:
    if bool(getattr(cfg, "prioritize_dubinsmaneuver2d", True)):
        ctrl_dir = str(getattr(cfg, "dubinsmaneuver_ctrl_dir", "*"))
        legacy_candidate = _build_dubinsmaneuver2d_candidate(
            start_pose=start_pose,
            goal_pose=goal_pose,
            turn_radius=turn_radius,
            ctrl_dir=ctrl_dir,
        )
        if legacy_candidate is not None:
            _, _, legacy_points = legacy_candidate
            if _polyline_collision_free(legacy_points, world=world, margin=margin):
                return legacy_points, _polyline_length(legacy_points), "connector_dubinsmaneuver2d_success"

    dubins_sol = _build_shortest_dubins_local(start_pose=start_pose, goal_pose=goal_pose, turn_radius=turn_radius)
    if dubins_sol is not None:
        word, params = dubins_sol
        dubins_points = _sample_dubins_segment(
            start_pose=start_pose,
            word=word,
            params=params,
            turn_radius=turn_radius,
            sample_step=sample_step,
        )
        dubins_points[0] = (start_pose[0], start_pose[1])
        dubins_points[-1] = (goal_pose[0], goal_pose[1])
        if _polyline_collision_free(dubins_points, world=world, margin=margin):
            return dubins_points, _polyline_length(dubins_points), "connector_dubins_success"

    recovered_points, recovered_len, _ = _try_direct_recovery_path(
        start_pose=start_pose,
        goal_pose=goal_pose,
        turn_radius=turn_radius,
        sample_step=sample_step,
        world=world,
        margin=margin,
    )
    if recovered_points and math.isfinite(recovered_len):
        return recovered_points, recovered_len, "connector_dubins_success"
    return [], float("inf"), "connector_collision_reject"


def _empty_connector_meta(reason: str, debug_trace: str = "") -> DubinsHybridMeta:
    return DubinsHybridMeta(
        used_fallback=True,
        fallback_segments=1,
        dubins_segments=0,
        sample_count=0,
        dubins_ratio=0.0,
        fallback_reason=reason,
        fallback_details=reason,
        debug_trace=debug_trace,
    )


def _finalize_connector_meta(
    *,
    points: List[Point2D],
    length: float,
    success_counts: Dict[str, int],
    event_counts: Dict[str, int],
    plain_astar_count: int,
    debug_parts: List[str],
) -> DubinsHybridMeta:
    connector_type = "mixed"
    if plain_astar_count > 0:
        connector_type = "plain_astar"
    else:
        nonzero = [name for name, count in success_counts.items() if count > 0]
        if len(nonzero) == 1:
            connector_type = nonzero[0]
        elif len(nonzero) > 1:
            connector_type = "mixed"
    connector_summary = "|".join(
        [
            f"straight:{success_counts.get('straight', 0)}",
            f"rsplan:{success_counts.get('rsplan', 0)}",
            f"reeds_shepp_like:{success_counts.get('reeds_shepp_like', 0)}",
            f"dubins_like:{success_counts.get('dubins_like', 0)}",
            f"hybrid_local:{success_counts.get('hybrid_local', 0)}",
            f"plain_astar:{plain_astar_count}",
        ]
    )
    event_summary = "|".join(f"{name}:{event_counts[name]}" for name in sorted(event_counts) if event_counts[name] > 0)
    debug_trace = ";".join(part for part in debug_parts if part)
    if event_summary:
        debug_trace = f"{debug_trace};events:{event_summary}" if debug_trace else f"events:{event_summary}"

    straight_len = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) if len(points) >= 2 else 0.0
    added_curve = max(0.0, length - straight_len)
    ratio = 0.0 if length <= 1e-9 else max(0.0, min(1.0, added_curve / length))
    used_fallback = plain_astar_count > 0
    fallback_reason = "connector_plain_astar_fallback" if used_fallback else ""
    fallback_details = event_summary if used_fallback else ""
    return DubinsHybridMeta(
        used_fallback=used_fallback,
        fallback_segments=plain_astar_count,
        dubins_segments=success_counts.get("rsplan", 0)
        + success_counts.get("reeds_shepp_like", 0)
        + success_counts.get("dubins_like", 0),
        sample_count=len(points),
        dubins_ratio=ratio,
        fallback_reason=fallback_reason,
        fallback_details=fallback_details,
        debug_trace=debug_trace,
        connector_type=connector_type,
        connector_summary=connector_summary,
    )


def build_segment_connector_path(
    world: WorldMap,
    cfg,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    astar_planner: AStarPlanner,
    turn_radius: float,
    astar_path: Optional[List[Point2D]] = None,
    astar_length: Optional[float] = None,
) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
    if astar_path is None or astar_length is None:
        astar_path, astar_length = astar_planner.plan((start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1]))
    if not astar_path or astar_length == float("inf"):
        return [], float("inf"), _empty_connector_meta("astar_unreachable", "connector_plain_astar_fallback")

    collision_margin = float(getattr(cfg, "dubins_collision_margin", 0.0))
    smoothing_margin = max(
        collision_margin,
        float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
    )
    reference_path = simplify_reference_polyline(
        world,
        astar_path,
        margin=smoothing_margin,
        enable_shortcut=bool(getattr(cfg, "connector_shortcut_enable", True)),
        enable_string_pull=bool(getattr(cfg, "connector_string_pull_enable", True)),
        string_pull_passes=int(getattr(cfg, "connector_string_pull_passes", 2)),
        split_turn_angle_threshold=float(getattr(cfg, "connector_split_turn_angle_threshold", 0.0)),
    )
    if len(reference_path) < 2:
        reference_path = [(start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1])]
    reference_path[0] = (start_pose[0], start_pose[1])
    reference_path[-1] = (goal_pose[0], goal_pose[1])
    reference_length = _polyline_length(reference_path)

    straight_tol = float(getattr(cfg, "connector_max_heading_error_for_straight", 0.35))
    short_span_tol = float(getattr(cfg, "connector_short_span_heading_error_for_straight", straight_tol))
    rs_max_expansions = int(getattr(cfg, "connector_rs_max_expansions", 180))
    rs_max_depth = int(getattr(cfg, "connector_rs_max_depth", 6))
    hybrid_max_expansions = int(getattr(cfg, "connector_max_local_hybrid_expansions", 2600))
    hybrid_heading_bins = int(getattr(cfg, "connector_local_hybrid_heading_bins", 48))
    hybrid_goal_pos_tol = float(getattr(cfg, "connector_local_hybrid_goal_pos_tolerance", 1.8))
    hybrid_goal_yaw_tol = float(getattr(cfg, "connector_local_hybrid_goal_heading_tolerance_rad", 0.95))
    corridor_radius = float(getattr(cfg, "connector_corridor_radius", 6.0))
    bbox_padding = float(getattr(cfg, "connector_local_bbox_padding", 4.0))
    allow_reverse = bool(getattr(cfg, "hybrid_astar_allow_reverse", False))
    internal_position_only = bool(getattr(cfg, "connector_internal_waypoints_position_only", True))
    prefer_legacy_dubins = bool(getattr(cfg, "prioritize_dubinsmaneuver2d", True))
    plain_astar_ok = bool(getattr(cfg, "connector_use_plain_astar_fallback", True)) and bool(
        getattr(cfg, "dubins_fallback_to_astar", True)
    )
    use_rsplan_connector = bool(getattr(cfg, "use_rsplan_connector", False))
    rsplan_fallback_to_custom = bool(getattr(cfg, "rsplan_fallback_to_custom_connector", True))
    rsplan_internal_soft = bool(getattr(cfg, "rsplan_enable_internal_waypoint_soft_heading", True))

    success_counts = {
        "straight": 0,
        "rsplan": 0,
        "reeds_shepp_like": 0,
        "dubins_like": 0,
        "hybrid_local": 0,
    }
    event_counts = {
        "connector_straight_success": 0,
        "connector_rsplan_success": 0,
        "connector_rs_success": 0,
        "connector_dubins_success": 0,
        "connector_hybrid_local_success": 0,
        "connector_plain_astar_fallback": 0,
        "connector_collision_reject": 0,
        "connector_heading_reject": 0,
        "connector_stitch_failure": 0,
        "rsplan_import_unavailable": 0,
        "rsplan_generation_failed": 0,
        "rsplan_collision_reject": 0,
        "rsplan_terminal_heading_only": 0,
        "rsplan_internal_waypoint_position_only": 0,
        "rsplan_fallback_custom_connector": 0,
        "rsplan_fallback_plain_astar": 0,
    }
    debug_parts = [f"connector_reference_points:{len(reference_path)}"]

    if bool(getattr(cfg, "connector_direct_whole_segment_first", True)):
        whole_bbox = _bbox_from_points(reference_path, corridor_radius + bbox_padding)
        whole_dist = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
        whole_straight_tol = short_span_tol if whole_dist <= 2.0 * turn_radius else straight_tol
        if bool(getattr(cfg, "connector_use_straight_first", True)):
            points, length, tag = _try_straight_connector(
                start_pose=start_pose,
                goal_pose=goal_pose,
                world=world,
                margin=collision_margin,
                heading_tolerance=whole_straight_tol,
            )
            if points and math.isfinite(length):
                event_counts["connector_straight_success"] += 1
                success_counts["straight"] += 1
                return points, length, _finalize_connector_meta(
                    points=points,
                    length=length,
                    success_counts=success_counts,
                    event_counts=event_counts,
                    plain_astar_count=0,
                    debug_parts=debug_parts + [tag],
                )
            event_counts["connector_heading_reject" if tag == "connector_heading_reject" else "connector_collision_reject"] += 1

        if prefer_legacy_dubins and bool(getattr(cfg, "connector_use_dubins", True)):
            points, length, tag = _try_dubins_connector(
                cfg=cfg,
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                sample_step=float(getattr(cfg, "dubins_sample_step", 0.5)),
                world=world,
                margin=collision_margin,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append("connector_dubins_detour_reject")
                else:
                    event_counts["connector_dubins_success"] += 1
                    success_counts["dubins_like"] += 1
                    return points, length, _finalize_connector_meta(
                        points=points,
                        length=length,
                        success_counts=success_counts,
                        event_counts=event_counts,
                        plain_astar_count=0,
                        debug_parts=debug_parts + [tag],
                    )
            event_counts["connector_collision_reject"] += 1

        rsplan_failed = False
        if allow_reverse and use_rsplan_connector:
            event_counts["rsplan_terminal_heading_only"] += 1
            points, length, tag = build_rsplan_connector(
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                world=world,
                cfg=cfg,
                margin=collision_margin,
                require_terminal_heading=True,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append("connector_rsplan_detour_reject")
                else:
                    event_counts["connector_rsplan_success"] += 1
                    success_counts["rsplan"] += 1
                    return points, length, _finalize_connector_meta(
                        points=points,
                        length=length,
                        success_counts=success_counts,
                        event_counts=event_counts,
                        plain_astar_count=0,
                        debug_parts=debug_parts + ["connector_rsplan_success"],
                    )
            else:
                rsplan_failed = True
                if tag in event_counts:
                    event_counts[tag] += 1
                if tag == "rsplan_collision_reject":
                    event_counts["connector_collision_reject"] += 1
                debug_parts.append(tag)

        use_custom_connector = (not use_rsplan_connector) or (not rsplan_failed) or rsplan_fallback_to_custom
        if rsplan_failed and use_custom_connector:
            event_counts["rsplan_fallback_custom_connector"] += 1

        if use_custom_connector and allow_reverse and bool(getattr(cfg, "connector_use_reeds_shepp", True)):
            points, length, diag = astar_planner.plan_local_connector(
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                step_size=float(getattr(cfg, "hybrid_astar_step_size", 1.0)),
                heading_bins=hybrid_heading_bins,
                goal_pos_tolerance=hybrid_goal_pos_tol,
                goal_heading_tolerance=hybrid_goal_yaw_tol,
                allow_reverse=True,
                max_expansions=rs_max_expansions,
                max_depth=rs_max_depth,
                connector_radius=max(corridor_radius, 1.2 * whole_dist),
                search_bbox=whole_bbox,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append("connector_rs_detour_reject")
                else:
                    event_counts["connector_rs_success"] += 1
                    success_counts["reeds_shepp_like"] += 1
                    return points, length, _finalize_connector_meta(
                        points=points,
                        length=length,
                        success_counts=success_counts,
                        event_counts=event_counts,
                        plain_astar_count=0,
                        debug_parts=debug_parts + ["connector_rs_success"],
                    )
            if getattr(diag, "reason", "") == "collision_on_connector":
                event_counts["connector_collision_reject"] += 1

        if use_custom_connector and bool(getattr(cfg, "connector_use_dubins", True)) and (not prefer_legacy_dubins):
            points, length, tag = _try_dubins_connector(
                cfg=cfg,
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                sample_step=float(getattr(cfg, "dubins_sample_step", 0.5)),
                world=world,
                margin=collision_margin,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append("connector_dubins_detour_reject")
                else:
                    event_counts["connector_dubins_success"] += 1
                    success_counts["dubins_like"] += 1
                    return points, length, _finalize_connector_meta(
                        points=points,
                        length=length,
                        success_counts=success_counts,
                        event_counts=event_counts,
                        plain_astar_count=0,
                        debug_parts=debug_parts + [tag],
                    )
            event_counts["connector_collision_reject"] += 1

    out_points: List[Point2D] = [(start_pose[0], start_pose[1])]
    current_pose = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
    plain_astar_count = 0

    for idx in range(1, len(reference_path)):
        target_point = reference_path[idx]
        is_terminal_waypoint = idx == len(reference_path) - 1
        span_position_only = internal_position_only and (not is_terminal_waypoint)
        if is_terminal_waypoint:
            target_heading = float(goal_pose[2])
        elif span_position_only:
            incoming_heading = heading_to_point(current_pose[:2], target_point)
            outgoing_heading = heading_to_point(target_point, reference_path[idx + 1])
            target_heading = _blend_heading_soft(incoming_heading, outgoing_heading)
        else:
            target_heading = heading_to_point(target_point, reference_path[idx + 1])
        span_goal = (float(target_point[0]), float(target_point[1]), float(target_heading))
        span_points_for_bbox = [current_pose[:2], target_point]
        if idx < len(reference_path) - 1:
            span_points_for_bbox.append(reference_path[idx + 1])
        span_bbox = _bbox_from_points(span_points_for_bbox, corridor_radius + bbox_padding)
        span_dist = math.hypot(target_point[0] - current_pose[0], target_point[1] - current_pose[1])
        span_reference_length = span_dist
        span_straight_tol = short_span_tol if span_dist <= 2.0 * turn_radius else straight_tol

        accepted_points: List[Point2D] = []
        accepted_length = float("inf")
        accepted_heading = current_pose[2]
        if span_position_only:
            debug_parts.append(f"span{idx}:position_only")
            if use_rsplan_connector and not rsplan_internal_soft:
                event_counts["rsplan_internal_waypoint_position_only"] += 1
                debug_parts.append(f"span{idx}:rsplan_internal_waypoint_position_only")
        else:
            debug_parts.append(f"span{idx}:terminal_pose_goal")

        if bool(getattr(cfg, "connector_use_straight_first", True)):
            points, length, tag = _try_straight_connector(
                start_pose=current_pose,
                goal_pose=span_goal,
                world=world,
                margin=collision_margin,
                heading_tolerance=span_straight_tol,
                enforce_goal_heading=not span_position_only,
            )
            if points and math.isfinite(length):
                accepted_points = points
                accepted_length = length
                accepted_heading = _path_terminal_heading(points, span_goal[2])
                success_counts["straight"] += 1
                event_counts["connector_straight_success"] += 1
                debug_parts.append(f"span{idx}:{tag}")
            else:
                event_counts["connector_heading_reject" if tag == "connector_heading_reject" else "connector_collision_reject"] += 1

        if (
            (not accepted_points)
            and (not span_position_only)
            and prefer_legacy_dubins
            and bool(getattr(cfg, "connector_use_dubins", True))
        ):
            points, length, tag = _try_dubins_connector(
                cfg=cfg,
                start_pose=current_pose,
                goal_pose=span_goal,
                turn_radius=turn_radius,
                sample_step=float(getattr(cfg, "dubins_sample_step", 0.5)),
                world=world,
                margin=collision_margin,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=span_reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append(f"span{idx}:connector_dubins_detour_reject")
                else:
                    accepted_points = points
                    accepted_length = length
                    accepted_heading = span_goal[2]
                    success_counts["dubins_like"] += 1
                    event_counts["connector_dubins_success"] += 1
                    debug_parts.append(f"span{idx}:{tag}")
            else:
                event_counts["connector_collision_reject"] += 1

        rsplan_failed = False
        if (
            (not accepted_points)
            and allow_reverse
            and use_rsplan_connector
            and ((not span_position_only) or rsplan_internal_soft)
        ):
            if span_position_only:
                event_counts["rsplan_internal_waypoint_position_only"] += 1
                debug_parts.append(f"span{idx}:rsplan_internal_waypoint_position_only")
            else:
                event_counts["rsplan_terminal_heading_only"] += 1
            points, length, tag = build_rsplan_connector(
                start_pose=current_pose,
                goal_pose=span_goal,
                turn_radius=turn_radius,
                world=world,
                cfg=cfg,
                margin=collision_margin,
                require_terminal_heading=not span_position_only,
            )
            if points and math.isfinite(length):
                internal_detour_reject = span_position_only and _detour_too_large(
                    candidate_length=length,
                    reference_length=span_reference_length,
                    ratio_limit=max(
                        1.0,
                        float(getattr(cfg, "rsplan_internal_max_detour_ratio_vs_reference", 1.20)),
                    ),
                    abs_limit=max(
                        0.0,
                        float(getattr(cfg, "rsplan_internal_max_detour_abs_vs_reference", 2.0)),
                    ),
                )
                if internal_detour_reject or _connector_detour_too_large(
                    candidate_length=length,
                    reference_length=span_reference_length,
                    cfg=cfg,
                ):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append(f"span{idx}:connector_rsplan_detour_reject")
                else:
                    accepted_points = points
                    accepted_length = length
                    accepted_heading = span_goal[2]
                    success_counts["rsplan"] += 1
                    event_counts["connector_rsplan_success"] += 1
                    debug_parts.append(f"span{idx}:connector_rsplan_success")
            else:
                rsplan_failed = True
                if tag in event_counts:
                    event_counts[tag] += 1
                if tag == "rsplan_collision_reject":
                    event_counts["connector_collision_reject"] += 1
                debug_parts.append(f"span{idx}:{tag}")

        use_custom_connector = (not use_rsplan_connector) or (not rsplan_failed) or rsplan_fallback_to_custom
        if rsplan_failed and use_custom_connector:
            event_counts["rsplan_fallback_custom_connector"] += 1

        if (
            (not accepted_points)
            and (not span_position_only)
            and use_custom_connector
            and allow_reverse
            and bool(getattr(cfg, "connector_use_reeds_shepp", True))
        ):
            points, length, diag = astar_planner.plan_local_connector(
                start_pose=current_pose,
                goal_pose=span_goal,
                turn_radius=turn_radius,
                step_size=float(getattr(cfg, "hybrid_astar_step_size", 1.0)),
                heading_bins=hybrid_heading_bins,
                goal_pos_tolerance=hybrid_goal_pos_tol,
                goal_heading_tolerance=hybrid_goal_yaw_tol,
                allow_reverse=True,
                max_expansions=rs_max_expansions,
                max_depth=rs_max_depth,
                connector_radius=max(corridor_radius, 1.25 * span_dist),
                search_bbox=span_bbox,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=span_reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append(f"span{idx}:connector_rs_detour_reject")
                else:
                    accepted_points = points
                    accepted_length = length
                    accepted_heading = span_goal[2]
                    success_counts["reeds_shepp_like"] += 1
                    event_counts["connector_rs_success"] += 1
                    debug_parts.append(f"span{idx}:connector_rs_success")
            elif getattr(diag, "reason", "") == "collision_on_connector":
                event_counts["connector_collision_reject"] += 1

        if (
            (not accepted_points)
            and (not span_position_only)
            and use_custom_connector
            and bool(getattr(cfg, "connector_use_dubins", True))
            and (not prefer_legacy_dubins)
        ):
            points, length, tag = _try_dubins_connector(
                cfg=cfg,
                start_pose=current_pose,
                goal_pose=span_goal,
                turn_radius=turn_radius,
                sample_step=float(getattr(cfg, "dubins_sample_step", 0.5)),
                world=world,
                margin=collision_margin,
            )
            if points and math.isfinite(length):
                if _connector_detour_too_large(candidate_length=length, reference_length=span_reference_length, cfg=cfg):
                    event_counts["connector_heading_reject"] += 1
                    debug_parts.append(f"span{idx}:connector_dubins_detour_reject")
                else:
                    accepted_points = points
                    accepted_length = length
                    accepted_heading = span_goal[2]
                    success_counts["dubins_like"] += 1
                    event_counts["connector_dubins_success"] += 1
                    debug_parts.append(f"span{idx}:{tag}")
            else:
                event_counts["connector_collision_reject"] += 1

        if (
            (not accepted_points)
            and (not span_position_only)
            and use_custom_connector
            and bool(getattr(cfg, "use_hybrid_astar", False))
            and bool(getattr(cfg, "connector_use_hybrid_local_rescue", True))
        ):
            points, length, diag = astar_planner.plan_hybrid_detailed(
                start_pose=current_pose,
                goal_pose=span_goal,
                turn_radius=turn_radius,
                step_size=float(getattr(cfg, "hybrid_astar_step_size", 1.0)),
                heading_bins=hybrid_heading_bins,
                max_expansions=hybrid_max_expansions,
                goal_pos_tolerance=hybrid_goal_pos_tol,
                goal_heading_tolerance=hybrid_goal_yaw_tol,
                allow_reverse=allow_reverse,
                reverse_penalty=float(getattr(cfg, "hybrid_astar_reverse_penalty", 1.22)),
                heuristic_weight=float(getattr(cfg, "hybrid_astar_heuristic_weight", 1.05)),
                near_goal_connector_expansions=rs_max_expansions,
                near_goal_connector_depth=rs_max_depth,
                near_goal_connector_radius_factor=max(1.5, corridor_radius / max(turn_radius, 1e-6)),
                search_bbox=span_bbox,
            )
            if points and math.isfinite(length):
                accepted_points = points
                accepted_length = length
                accepted_heading = span_goal[2]
                success_counts["hybrid_local"] += 1
                event_counts["connector_hybrid_local_success"] += 1
                debug_parts.append(f"span{idx}:connector_hybrid_local_success")
            elif getattr(diag, "reason", "") == "collision_on_connector":
                event_counts["connector_collision_reject"] += 1

        if not accepted_points and (plain_astar_ok or span_position_only):
            span_astar_path, span_astar_len = astar_planner.plan(current_pose[:2], target_point)
            if span_astar_path and math.isfinite(span_astar_len):
                accepted_points = span_astar_path
                accepted_length = span_astar_len
                accepted_heading = _path_terminal_heading(span_astar_path, span_goal[2])
                plain_astar_count += 1
                event_counts["connector_plain_astar_fallback"] += 1
                if use_rsplan_connector:
                    event_counts["rsplan_fallback_plain_astar"] += 1
                if span_position_only:
                    debug_parts.append(f"span{idx}:skipped_heading_for_internal_waypoint")
                debug_parts.append(f"span{idx}:connector_plain_astar_fallback")

        if not accepted_points or (not math.isfinite(accepted_length)):
            return [], float("inf"), _empty_connector_meta(
                "connector_stitch_failure",
                ";".join(debug_parts + ["connector_stitch_failure"]),
            )

        stitched = _dedupe_points(out_points + accepted_points[1:])
        if not _polyline_collision_free(stitched, world=world, margin=collision_margin):
            event_counts["connector_stitch_failure"] += 1
            if plain_astar_ok:
                if use_rsplan_connector:
                    event_counts["rsplan_fallback_plain_astar"] += 1
                return astar_path, astar_length, _finalize_connector_meta(
                    points=astar_path,
                    length=astar_length,
                    success_counts=success_counts,
                    event_counts=event_counts,
                    plain_astar_count=max(1, plain_astar_count),
                    debug_parts=debug_parts + ["connector_stitch_failure", "connector_plain_astar_fallback"],
                )
            return [], float("inf"), _empty_connector_meta(
                "connector_stitch_failure",
                ";".join(debug_parts + ["connector_stitch_failure"]),
            )

        out_points = stitched
        current_pose = (out_points[-1][0], out_points[-1][1], accepted_heading)

    total_len = _polyline_length(out_points)
    return out_points, total_len, _finalize_connector_meta(
        points=out_points,
        length=total_len,
        success_counts=success_counts,
        event_counts=event_counts,
        plain_astar_count=plain_astar_count,
        debug_parts=debug_parts,
    )


def _sample_recovery_arc(
    start_pose: Pose2D,
    turn_radius: float,
    delta_heading: float,
    sample_step: float,
) -> Tuple[List[Point2D], Pose2D]:
    x0, y0, yaw0 = start_pose
    if turn_radius <= 1e-9 or abs(delta_heading) <= 1e-9:
        return [(x0, y0)], (x0, y0, yaw0)

    left_turn = delta_heading > 0.0
    if left_turn:
        cx = x0 - turn_radius * math.sin(yaw0)
        cy = y0 + turn_radius * math.cos(yaw0)
    else:
        cx = x0 + turn_radius * math.sin(yaw0)
        cy = y0 - turn_radius * math.cos(yaw0)

    arc_len = turn_radius * abs(delta_heading)
    n = max(1, int(math.ceil(arc_len / max(sample_step, 1e-6))))
    points: List[Point2D] = [(x0, y0)]
    for k in range(1, n + 1):
        yaw = yaw0 + delta_heading * (k / n)
        if left_turn:
            px = cx + turn_radius * math.sin(yaw)
            py = cy - turn_radius * math.cos(yaw)
        else:
            px = cx - turn_radius * math.sin(yaw)
            py = cy + turn_radius * math.cos(yaw)
        points.append((px, py))

    end_pose = (points[-1][0], points[-1][1], wrap_to_pi(yaw0 + delta_heading))
    return points, end_pose


def _try_direct_recovery_path(
    start_pose: Pose2D,
    goal_pose: Pose2D,
    turn_radius: float,
    sample_step: float,
    world: WorldMap,
    margin: float,
) -> Tuple[List[Point2D], float, str]:
    desired_heading = heading_to_point((start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1]))
    preferred_sign = 1.0 if wrap_to_pi(desired_heading - start_pose[2]) >= 0.0 else -1.0
    sign_order = (preferred_sign, -preferred_sign)
    step_candidates = (
        math.radians(10.0),
        math.radians(15.0),
        math.radians(20.0),
        math.radians(30.0),
        math.radians(45.0),
        math.radians(60.0),
        math.radians(75.0),
    )

    best_path: List[Point2D] = []
    best_len = float("inf")
    trace_parts: List[str] = []
    for step in step_candidates:
        for sign in sign_order:
            label = f"{'L' if sign > 0 else 'R'}{math.degrees(step):.0f}"
            arc_points, arc_end_pose = _sample_recovery_arc(
                start_pose=start_pose,
                turn_radius=turn_radius,
                delta_heading=sign * step,
                sample_step=sample_step,
            )
            if not _polyline_collision_free(arc_points, world=world, margin=margin):
                trace_parts.append(f"{label}:arc_collision")
                continue

            dubins_sol = _build_shortest_dubins_local(
                start_pose=arc_end_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
            )
            if dubins_sol is None:
                trace_parts.append(f"{label}:no_solution")
                continue

            word, params = dubins_sol
            tail_points = _sample_dubins_segment(
                start_pose=arc_end_pose,
                word=word,
                params=params,
                turn_radius=turn_radius,
                sample_step=sample_step,
            )
            tail_points[0] = (arc_end_pose[0], arc_end_pose[1])
            tail_points[-1] = (goal_pose[0], goal_pose[1])
            full_points = arc_points + tail_points[1:]
            if not _polyline_collision_free(full_points, world=world, margin=margin):
                trace_parts.append(f"{label}:{word}:tail_collision")
                continue

            full_len = _polyline_length(full_points)
            if full_len < best_len:
                best_path = full_points
                best_len = full_len
            trace_parts.append(f"{label}:{word}:ok:{full_len:.3f}")

    return best_path, best_len, ";".join(trace_parts)


def _shortcut_smooth_path(path: List[Point2D], world: WorldMap, margin: float) -> List[Point2D]:
    if len(path) <= 2:
        return path

    out: List[Point2D] = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        next_idx = i + 1
        for j in range(n - 1, i + 1, -1):
            if _segment_collision_free(path[i], path[j], world=world, margin=margin):
                next_idx = j
                break
        out.append(path[next_idx])
        i = next_idx

    return out


def _add(a: Point2D, b: Point2D) -> Point2D:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Point2D, b: Point2D) -> Point2D:
    return a[0] - b[0], a[1] - b[1]


def _scale(a: Point2D, s: float) -> Point2D:
    return a[0] * s, a[1] * s


def _norm(a: Point2D) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Point2D) -> Optional[Point2D]:
    n = _norm(a)
    if n <= 1e-9:
        return None
    return a[0] / n, a[1] / n


def _left_normal(d: Point2D) -> Point2D:
    return -d[1], d[0]


def _right_normal(d: Point2D) -> Point2D:
    return d[1], -d[0]


def _sample_arc(
    center: Point2D,
    radius: float,
    start: Point2D,
    end: Point2D,
    left_turn: bool,
    step: float,
) -> List[Point2D]:
    a1 = math.atan2(start[1] - center[1], start[0] - center[0])
    a2 = math.atan2(end[1] - center[1], end[0] - center[0])
    points: List[Point2D] = [start]

    if left_turn:
        delta = _mod2pi(a2 - a1)
        n = max(1, int(math.ceil((radius * delta) / max(step, 1e-6))))
        for k in range(1, n):
            ang = a1 + delta * (k / n)
            points.append((center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)))
    else:
        delta = _mod2pi(a1 - a2)
        n = max(1, int(math.ceil((radius * delta) / max(step, 1e-6))))
        for k in range(1, n):
            ang = a1 - delta * (k / n)
            points.append((center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)))

    points.append(end)
    return points


@dataclass
class _CornerFillet:
    t_in: Point2D
    t_out: Point2D
    arc_points: List[Point2D]


def _build_corner_fillet(
    prev_pt: Point2D,
    corner_pt: Point2D,
    next_pt: Point2D,
    radius: float,
    step: float,
    world: WorldMap,
    margin: float,
    force_mode: bool,
) -> Tuple[Optional[_CornerFillet], str]:
    vin_raw = _sub(corner_pt, prev_pt)
    vout_raw = _sub(next_pt, corner_pt)
    len_in = _norm(vin_raw)
    len_out = _norm(vout_raw)
    if len_in <= 1e-9 or len_out <= 1e-9:
        return None, "degenerate_segment"

    vin = _unit(vin_raw)
    vout = _unit(vout_raw)
    if vin is None or vout is None:
        return None, "degenerate_segment"

    dot = max(-1.0, min(1.0, vin[0] * vout[0] + vin[1] * vout[1]))
    phi = math.acos(dot)
    if phi <= 1e-3:
        return None, "straight_corner"
    if abs(math.pi - phi) <= 1e-3:
        return None, "uturn_corner"

    tan_half = math.tan(phi / 2.0)
    if abs(tan_half) <= 1e-9:
        return None, "uturn_corner"

    r_max = max(0.0, min(len_in, len_out) * tan_half * 0.98)
    r_eff = min(radius, r_max)
    if r_eff <= 1e-4:
        return None, "radius_too_small"

    t = r_eff / tan_half
    t_in = _sub(corner_pt, _scale(vin, t))
    t_out = _add(corner_pt, _scale(vout, t))

    cross = vin[0] * vout[1] - vin[1] * vout[0]
    if abs(cross) <= 1e-9:
        return None, "straight_corner"
    left_turn = cross > 0.0
    n_in = _left_normal(vin) if left_turn else _right_normal(vin)
    center = _add(t_in, _scale(n_in, r_eff))

    arc = _sample_arc(center=center, radius=r_eff, start=t_in, end=t_out, left_turn=left_turn, step=step)
    if not force_mode and not _collision_free(arc, world=world, margin=margin):
        return None, "corner_collision"
    return _CornerFillet(t_in=t_in, t_out=t_out, arc_points=arc), ""


def _build_fillet_polyline(
    path: List[Point2D],
    turn_radius: float,
    sample_step: float,
    world: WorldMap,
    margin: float,
    force_mode: bool,
) -> Tuple[List[Point2D], int, int, Dict[str, int]]:
    if len(path) < 3:
        return path, 0, 0, {}

    corners: Dict[int, _CornerFillet] = {}
    rejected_reasons: Dict[int, str] = {}
    for i in range(1, len(path) - 1):
        fillet, reject_reason = _build_corner_fillet(
            prev_pt=path[i - 1],
            corner_pt=path[i],
            next_pt=path[i + 1],
            radius=turn_radius,
            step=sample_step,
            world=world,
            margin=margin,
            force_mode=force_mode,
        )
        if fillet is not None:
            corners[i] = fillet
        elif reject_reason:
            rejected_reasons[i] = reject_reason

    # Resolve overlap between adjacent corner fillets on short middle segments.
    # If both sides cut too much on one segment, keep only one corner to avoid
    # reversed ordering (visual "jump lines"/backtracking spikes).
    while True:
        changed = False
        for seg_idx in range(len(path) - 1):
            start_corner = corners.get(seg_idx)
            end_corner = corners.get(seg_idx + 1)
            if start_corner is None and end_corner is None:
                continue

            seg_len = _norm(_sub(path[seg_idx + 1], path[seg_idx]))
            if seg_len <= 1e-9:
                if start_corner is not None:
                    del corners[seg_idx]
                    rejected_reasons[seg_idx] = "adjacent_overlap"
                if end_corner is not None:
                    del corners[seg_idx + 1]
                    rejected_reasons[seg_idx + 1] = "adjacent_overlap"
                changed = True
                break

            start_cut = (
                _norm(_sub(start_corner.t_out, path[seg_idx]))
                if start_corner is not None
                else 0.0
            )
            end_cut = (
                _norm(_sub(path[seg_idx + 1], end_corner.t_in))
                if end_corner is not None
                else 0.0
            )

            if start_cut + end_cut > seg_len - 1e-6:
                if start_corner is not None and end_corner is not None:
                    if start_cut >= end_cut:
                        del corners[seg_idx]
                        rejected_reasons[seg_idx] = "adjacent_overlap"
                    else:
                        del corners[seg_idx + 1]
                        rejected_reasons[seg_idx + 1] = "adjacent_overlap"
                elif start_corner is not None:
                    del corners[seg_idx]
                    rejected_reasons[seg_idx] = "adjacent_overlap"
                else:
                    del corners[seg_idx + 1]
                    rejected_reasons[seg_idx + 1] = "adjacent_overlap"
                changed = True
                break
        if not changed:
            break

    out: List[Point2D] = [path[0]]
    for seg_idx in range(len(path) - 1):
        start_pt = corners[seg_idx].t_out if seg_idx in corners else path[seg_idx]
        end_corner_idx = seg_idx + 1
        end_pt = corners[end_corner_idx].t_in if end_corner_idx in corners else path[seg_idx + 1]

        if _norm(_sub(start_pt, out[-1])) > 1e-8:
            out.append(start_pt)
        if _norm(_sub(end_pt, out[-1])) > 1e-8:
            out.append(end_pt)

        if end_corner_idx in corners:
            arc = corners[end_corner_idx].arc_points
            for p in arc[1:]:
                if _norm(_sub(p, out[-1])) > 1e-8:
                    out.append(p)

    fallback_reason_counts: Dict[str, int] = {}
    fallback_count = 0
    fallback_reasons = {
        "adjacent_overlap",
        "corner_collision",
        "radius_too_small",
        "uturn_corner",
        "missing_corner_status",
    }
    for i in range(1, len(path) - 1):
        if i in corners:
            continue
        reason = rejected_reasons.get(i, "missing_corner_status")
        if reason not in fallback_reasons:
            continue
        fallback_count += 1
        _bump_reason(fallback_reason_counts, reason)

    return out, len(corners), fallback_count, fallback_reason_counts


def _dir8(a: Point2D, b: Point2D) -> Tuple[int, int]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    sx = 0 if abs(dx) <= 1e-9 else (1 if dx > 0 else -1)
    sy = 0 if abs(dy) <= 1e-9 else (1 if dy > 0 else -1)
    return sx, sy


def _compress_astar_path(path: List[Point2D]) -> List[Point2D]:
    # Merge staircase runs with same 8-neighbor direction, so Dubins operates on meaningful corners.
    if len(path) <= 2:
        return path

    out: List[Point2D] = [path[0]]
    prev = path[0]
    prev_dir = _dir8(path[0], path[1])

    for i in range(1, len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]
        d = _dir8(cur, nxt)
        if d != prev_dir:
            if math.hypot(cur[0] - prev[0], cur[1] - prev[1]) > 1e-9:
                out.append(cur)
                prev = cur
            prev_dir = d

    if math.hypot(path[-1][0] - out[-1][0], path[-1][1] - out[-1][1]) > 1e-9:
        out.append(path[-1])
    return out


def _build_fillet_result(
    sparse_path: List[Point2D],
    fallback_path: List[Point2D],
    fallback_len: float,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    world: WorldMap,
    cfg,
    turn_radius: float,
    force_mode: bool,
    debug_trace: str = "",
) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
    fillet_points, fillet_count, fallback_count, fallback_reason_counts = _build_fillet_polyline(
        path=sparse_path,
        turn_radius=turn_radius,
        sample_step=cfg.dubins_sample_step,
        world=world,
        margin=cfg.dubins_collision_margin,
        force_mode=force_mode,
    )
    if fallback_count > 0 and not force_mode and not cfg.dubins_fallback_to_astar:
        return (
            [],
            float("inf"),
            DubinsHybridMeta(
                used_fallback=True,
                fallback_segments=fallback_count,
                dubins_segments=fillet_count,
                sample_count=0,
                dubins_ratio=0.0,
                fallback_reason=_primary_reason(fallback_reason_counts),
                fallback_details=_format_reason_counts(fallback_reason_counts),
                debug_trace=debug_trace,
            ),
        )
    out_points = fillet_points
    out_points[0] = (start_pose[0], start_pose[1])
    out_points[-1] = (goal_pose[0], goal_pose[1])

    if not force_mode and not _polyline_collision_free(
        out_points,
        world=world,
        margin=cfg.dubins_collision_margin,
    ):
        final_reason_counts = dict(fallback_reason_counts)
        _bump_reason(final_reason_counts, "final_collision")
        rejected_segments = max(1, fallback_count)
        if cfg.dubins_fallback_to_astar:
            return (
                fallback_path,
                fallback_len,
                DubinsHybridMeta(
                    used_fallback=True,
                    fallback_segments=rejected_segments,
                    dubins_segments=fillet_count,
                    sample_count=len(fallback_path),
                    dubins_ratio=0.0,
                    fallback_reason="final_collision",
                    fallback_details=_format_reason_counts(final_reason_counts),
                    debug_trace=debug_trace,
                ),
            )
        return (
            [],
            float("inf"),
            DubinsHybridMeta(
                used_fallback=True,
                fallback_segments=rejected_segments,
                dubins_segments=fillet_count,
                sample_count=0,
                dubins_ratio=0.0,
                fallback_reason="final_collision",
                fallback_details=_format_reason_counts(final_reason_counts),
                debug_trace=debug_trace,
            ),
        )

    total_len = 0.0
    for i in range(len(out_points) - 1):
        total_len += math.hypot(out_points[i + 1][0] - out_points[i][0], out_points[i + 1][1] - out_points[i][1])

    straight_len = 0.0
    for i in range(len(sparse_path) - 1):
        straight_len += math.hypot(
            sparse_path[i + 1][0] - sparse_path[i][0],
            sparse_path[i + 1][1] - sparse_path[i][1],
        )
    added_curve = max(0.0, total_len - straight_len)
    ratio = 0.0 if total_len <= 1e-9 else max(0.0, min(1.0, added_curve / total_len))

    used_fallback = fallback_count > 0
    meta = DubinsHybridMeta(
        used_fallback=used_fallback,
        fallback_segments=fallback_count,
        dubins_segments=fillet_count,
        sample_count=len(out_points),
        dubins_ratio=ratio,
        fallback_reason=_primary_reason(fallback_reason_counts),
        fallback_details=_format_reason_counts(fallback_reason_counts),
        debug_trace=debug_trace,
    )
    return out_points, total_len, meta


def _build_bid_verification_cfg(cfg):
    return replace(
        cfg,
        use_dubins_hybrid=bool(getattr(cfg, "bid_verification_use_dubins_hybrid", False)),
        enable_connector_first_planner=bool(getattr(cfg, "bid_verification_enable_connector_first", False)),
        use_rsplan_connector=bool(getattr(cfg, "use_rsplan_connector", False))
        and bool(getattr(cfg, "rsplan_enable_in_verification", False)),
        hybrid_astar_retry_on_fail=False,
        hybrid_astar_relaxed_goal_on_unreachable=False,
        connector_use_hybrid_local_rescue=False,
    )


def _tag_meta_debug(meta: DubinsHybridMeta, *tokens: str) -> DubinsHybridMeta:
    meta.debug_trace = _append_trace_tokens(meta.debug_trace, *tokens)
    if meta.used_fallback:
        meta.fallback_details = _append_reason_tokens(meta.fallback_details, *tokens)
        if not meta.fallback_reason:
            meta.fallback_reason = _primary_reason_token(meta.fallback_details, default="mode_tagged_fallback")
    return meta


def build_dubins_hybrid_path(
    world: WorldMap,
    cfg,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    astar_planner: AStarPlanner,
    turn_radius: float,
    astar_path: Optional[List[Point2D]] = None,
    astar_length: Optional[float] = None,
) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
    force_mode = bool(getattr(cfg, "dubins_force_mode", False))
    force_astar_only = bool(getattr(cfg, "force_astar_only", False))
    use_dubins_hybrid = bool(getattr(cfg, "use_dubins_hybrid", False))
    collision_margin = float(getattr(cfg, "dubins_collision_margin", 0.0))
    legacy_debug_prefix = ""
    legacy_reason_detail = ""

    if force_astar_only:
        if astar_path is None or astar_length is None:
            astar_path, astar_length = astar_planner.plan((start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1]))
        if not astar_path or astar_length == float("inf"):
            return (
                [],
                float("inf"),
                DubinsHybridMeta(
                    used_fallback=True,
                    fallback_segments=0,
                    dubins_segments=0,
                    sample_count=0,
                    dubins_ratio=0.0,
                    fallback_reason="astar_unreachable",
                    fallback_details="astar_unreachable:1",
                    debug_trace="astar_only:unreachable",
                ),
            )

        smoothing_margin = max(
            collision_margin,
            float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
        )
        if bool(getattr(cfg, "astar_smooth_before_dubins", True)):
            base_path = _shortcut_smooth_path(astar_path, world=world, margin=smoothing_margin)
            if not _polyline_collision_free(base_path, world=world, margin=smoothing_margin):
                base_path = astar_path
                trace = "astar_only:raw_astar"
            else:
                trace = "astar_only:shortcut"
        else:
            base_path = astar_path
            trace = "astar_only:raw_astar"

        if base_path:
            base_path[0] = (start_pose[0], start_pose[1])
            base_path[-1] = (goal_pose[0], goal_pose[1])
        base_len = _polyline_length(base_path)
        return (
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
                debug_trace=trace,
            ),
        )

    if use_dubins_hybrid and bool(getattr(cfg, "enable_connector_first_planner", True)) and not force_mode:
        connector_points, connector_len, connector_meta = build_segment_connector_path(
            world=world,
            cfg=cfg,
            start_pose=start_pose,
            goal_pose=goal_pose,
            astar_planner=astar_planner,
            turn_radius=turn_radius,
            astar_path=astar_path,
            astar_length=astar_length,
        )
        if connector_points and math.isfinite(connector_len):
            return connector_points, connector_len, connector_meta
        if not bool(getattr(cfg, "hybrid_astar_fallback_to_legacy", True)):
            return connector_points, connector_len, connector_meta

    def _finalize_legacy_return(
        points: List[Point2D],
        length: float,
        meta: DubinsHybridMeta,
    ) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
        if not legacy_debug_prefix:
            return points, length, meta

        if not meta.debug_trace:
            meta.debug_trace = legacy_debug_prefix
        elif legacy_debug_prefix not in meta.debug_trace:
            meta.debug_trace = f"{legacy_debug_prefix};{meta.debug_trace}"

        if not meta.used_fallback:
            meta.used_fallback = True
            meta.fallback_segments = max(1, int(meta.fallback_segments))
        meta.fallback_details = _append_reason_tokens(meta.fallback_details, *_reason_tokens(legacy_reason_detail))
        if not meta.fallback_reason:
            meta.fallback_reason = _primary_reason_token(meta.fallback_details, default="hybrid_astar_fallback")
        return points, length, meta

    # Primary mode: Hybrid A* on (x, y, yaw) state space.
    if use_dubins_hybrid and bool(getattr(cfg, "use_hybrid_astar", False)):
        planner_clearance = max(0.0, float(getattr(astar_planner, "inflation_radius", 0.0)))

        def _try_hybrid_once(
            *,
            step_size: float,
            heading_bins: int,
            max_expansions: int,
            goal_pos_tolerance: float,
            goal_heading_tolerance: float,
            heuristic_weight: float,
            allow_reverse: bool,
            trace: str,
        ) -> Tuple[Optional[Tuple[List[Point2D], float, DubinsHybridMeta]], HybridPlanDiagnostics]:
            hybrid_points, hybrid_len, hybrid_diag = astar_planner.plan_hybrid_detailed(
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                step_size=step_size,
                heading_bins=heading_bins,
                max_expansions=max_expansions,
                goal_pos_tolerance=goal_pos_tolerance,
                goal_heading_tolerance=goal_heading_tolerance,
                allow_reverse=allow_reverse,
                reverse_penalty=float(getattr(cfg, "hybrid_astar_reverse_penalty", 1.6)),
                heuristic_weight=heuristic_weight,
                near_goal_connector_expansions=int(
                    getattr(cfg, "hybrid_astar_near_goal_connector_expansions", 160)
                ),
                near_goal_connector_depth=int(getattr(cfg, "hybrid_astar_near_goal_connector_depth", 6)),
                near_goal_connector_radius_factor=float(
                    getattr(cfg, "hybrid_astar_near_goal_connector_radius_factor", 2.75)
                ),
            )
            if not hybrid_points or not math.isfinite(hybrid_len):
                return None, hybrid_diag

            need_recheck = collision_margin > planner_clearance + 1e-9
            if not force_mode and need_recheck and not _polyline_collision_free(
                hybrid_points,
                world=world,
                margin=collision_margin,
            ):
                return None, HybridPlanDiagnostics(
                    reason="collision_on_connector",
                    reason_tags=("collision_on_connector",),
                    debug_trace="hybrid_astar:margin_collision",
                    expansions=hybrid_diag.expansions,
                    used_connector=hybrid_diag.used_connector,
                    connector_mode=hybrid_diag.connector_mode,
                )

            return (
                hybrid_points,
                hybrid_len,
                DubinsHybridMeta(
                    used_fallback=False,
                    fallback_segments=0,
                    dubins_segments=0,
                    sample_count=len(hybrid_points),
                    dubins_ratio=0.0,
                    debug_trace=str(getattr(hybrid_diag, "debug_trace", "") or trace),
                ),
            ), hybrid_diag

        primary_out, primary_diag = _try_hybrid_once(
            step_size=float(getattr(cfg, "hybrid_astar_step_size", 0.8)),
            heading_bins=int(getattr(cfg, "hybrid_astar_heading_bins", 72)),
            max_expansions=int(getattr(cfg, "hybrid_astar_max_expansions", 45000)),
            goal_pos_tolerance=float(getattr(cfg, "hybrid_astar_goal_pos_tolerance", 1.5)),
            goal_heading_tolerance=float(getattr(cfg, "hybrid_astar_goal_heading_tolerance_rad", 0.8)),
            heuristic_weight=float(getattr(cfg, "hybrid_astar_heuristic_weight", 1.05)),
            allow_reverse=bool(getattr(cfg, "hybrid_astar_allow_reverse", False)),
            trace="hybrid_astar:ok",
        )
        if primary_out is not None:
            return primary_out

        hybrid_fail_reason = getattr(primary_diag, "reason", "") or "unreachable_main_search"
        primary_tags = _stage_reason_tags(primary_diag, stage="primary")
        debug_trace = "hybrid_astar:primary_" + "|".join(primary_tags)
        legacy_reason_detail = _append_reason_tokens(legacy_reason_detail, *primary_tags)

        if bool(getattr(cfg, "hybrid_astar_retry_on_fail", True)):
            retry_out, retry_diag = _try_hybrid_once(
                step_size=float(getattr(cfg, "hybrid_astar_retry_step_size", 0.8)),
                heading_bins=int(getattr(cfg, "hybrid_astar_retry_heading_bins", 72)),
                max_expansions=int(getattr(cfg, "hybrid_astar_retry_max_expansions", 45000)),
                goal_pos_tolerance=float(getattr(cfg, "hybrid_astar_retry_goal_pos_tolerance", 2.2)),
                goal_heading_tolerance=float(getattr(cfg, "hybrid_astar_retry_goal_heading_tolerance_rad", 1.1)),
                heuristic_weight=float(getattr(cfg, "hybrid_astar_retry_heuristic_weight", 1.02)),
                allow_reverse=bool(
                    getattr(
                        cfg,
                        "hybrid_astar_retry_allow_reverse",
                        bool(getattr(cfg, "hybrid_astar_allow_reverse", False)),
                    )
                ),
                trace="hybrid_astar:retry_ok",
            )
            if retry_out is not None:
                return retry_out
            retry_tags = _stage_reason_tags(retry_diag, stage="retry")
            retry_fail = getattr(retry_diag, "reason", "") or "unreachable_after_retry"
            hybrid_fail_reason = retry_fail
            debug_trace = f"{debug_trace}|retry_" + "|".join(retry_tags)
            legacy_reason_detail = _append_reason_tokens(legacy_reason_detail, *retry_tags)

            # Single rescue pass for unreachable: relax terminal heading constraint.
            if (
                retry_fail in {"unreachable_main_search", "unreachable_after_retry", "exceeded_expansions"}
                and bool(getattr(cfg, "hybrid_astar_relaxed_goal_on_unreachable", True))
            ):
                relaxed_out, relaxed_diag = _try_hybrid_once(
                    step_size=float(getattr(cfg, "hybrid_astar_retry_step_size", 0.8)),
                    heading_bins=int(getattr(cfg, "hybrid_astar_retry_heading_bins", 72)),
                    max_expansions=int(getattr(cfg, "hybrid_astar_relaxed_goal_max_expansions", 12000)),
                    goal_pos_tolerance=float(getattr(cfg, "hybrid_astar_relaxed_goal_pos_tolerance", 3.0)),
                    goal_heading_tolerance=float(
                        getattr(cfg, "hybrid_astar_relaxed_goal_heading_tolerance_rad", math.pi)
                    ),
                    heuristic_weight=float(getattr(cfg, "hybrid_astar_retry_heuristic_weight", 1.02)),
                    allow_reverse=bool(
                        getattr(
                            cfg,
                            "hybrid_astar_retry_allow_reverse",
                            bool(getattr(cfg, "hybrid_astar_allow_reverse", False)),
                        )
                    ),
                    trace="hybrid_astar:relaxed_goal_ok",
                )
                if relaxed_out is not None:
                    return relaxed_out
                relaxed_tags = list(getattr(relaxed_diag, "reason_tags", ()) or ())
                relaxed_fail = getattr(relaxed_diag, "reason", "") or "unreachable_after_retry"
                hybrid_fail_reason = relaxed_fail
                debug_trace = f"{debug_trace}|relaxed_" + ("|".join(relaxed_tags) if relaxed_tags else relaxed_fail)
                legacy_reason_detail = _append_reason_tokens(legacy_reason_detail, *relaxed_tags)

        legacy_debug_prefix = debug_trace
        if not bool(getattr(cfg, "hybrid_astar_fallback_to_legacy", True)):
            return (
                [],
                float("inf"),
                DubinsHybridMeta(
                    used_fallback=True,
                    fallback_segments=1,
                    dubins_segments=0,
                    sample_count=0,
                    dubins_ratio=0.0,
                    fallback_reason=_primary_reason_token(legacy_reason_detail, default=hybrid_fail_reason),
                    fallback_details=legacy_reason_detail or hybrid_fail_reason,
                    debug_trace=debug_trace,
                ),
            )

    if astar_path is None or astar_length is None:
        astar_path, astar_length = astar_planner.plan((start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1]))

    if not astar_path or astar_length == float("inf"):
        return _finalize_legacy_return(
            [],
            float("inf"),
            DubinsHybridMeta(
                used_fallback=True,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=0,
                dubins_ratio=0.0,
                fallback_reason="astar_unreachable",
                fallback_details="astar_unreachable:1",
                debug_trace=legacy_debug_prefix,
            ),
        )

    smoothing_margin = max(
        collision_margin,
        float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
    )
    if getattr(cfg, "astar_smooth_before_dubins", True):
        base_path = _shortcut_smooth_path(astar_path, world=world, margin=smoothing_margin)
        if not _polyline_collision_free(base_path, world=world, margin=smoothing_margin):
            base_path = astar_path
    else:
        base_path = astar_path
    base_len = _polyline_length(base_path)

    # Hybrid failed already: skip expensive legacy fillet reconstruction and
    # directly return smoothed A* fallback to keep runtime bounded.
    if legacy_debug_prefix and bool(getattr(cfg, "hybrid_astar_direct_astar_fallback", True)):
        return _finalize_legacy_return(
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=True,
                fallback_segments=1,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
                fallback_reason="degraded_to_plain_astar",
                fallback_details=_append_reason_tokens(legacy_reason_detail, "degraded_to_plain_astar"),
                debug_trace=legacy_debug_prefix,
            ),
        )

    if not use_dubins_hybrid:
        return _finalize_legacy_return(
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
                debug_trace=legacy_debug_prefix,
            ),
        )

    sparse_path = _compress_astar_path(base_path)
    if len(sparse_path) < 2:
        return _finalize_legacy_return(
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
                debug_trace=legacy_debug_prefix,
            ),
        )

    if len(sparse_path) == 2:
        direct_reason = "direct_no_solution"
        debug_trace = "direct:start"
        if legacy_debug_prefix:
            debug_trace = f"{legacy_debug_prefix};{debug_trace}"
        if bool(getattr(cfg, "prioritize_dubinsmaneuver2d", True)):
            ctrl_dir = str(getattr(cfg, "dubinsmaneuver_ctrl_dir", "*"))
            legacy_candidate = _build_dubinsmaneuver2d_candidate(
                start_pose=start_pose,
                goal_pose=goal_pose,
                turn_radius=turn_radius,
                ctrl_dir=ctrl_dir,
            )
            if legacy_candidate is not None:
                legacy_word, _, legacy_points = legacy_candidate
                if force_mode or _polyline_collision_free(
                    legacy_points,
                    world=world,
                    margin=cfg.dubins_collision_margin,
                ):
                    debug_trace = (
                        f"{legacy_debug_prefix};direct:dubinsmaneuver2d:{legacy_word}:ok"
                        if legacy_debug_prefix
                        else f"direct:dubinsmaneuver2d:{legacy_word}:ok"
                    )
                    dubins_len = _polyline_length(legacy_points)
                    straight_len = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
                    added_curve = max(0.0, dubins_len - straight_len)
                    ratio = 0.0 if dubins_len <= 1e-9 else max(0.0, min(1.0, added_curve / dubins_len))
                    return _finalize_legacy_return(
                        legacy_points,
                        dubins_len,
                        DubinsHybridMeta(
                            used_fallback=False,
                            fallback_segments=0,
                            dubins_segments=1,
                            sample_count=len(legacy_points),
                            dubins_ratio=ratio,
                            debug_trace=debug_trace,
                        ),
                    )
                direct_reason = "direct_collision"
                debug_trace = (
                    f"{legacy_debug_prefix};direct:dubinsmaneuver2d:{legacy_word}:collision"
                    if legacy_debug_prefix
                    else f"direct:dubinsmaneuver2d:{legacy_word}:collision"
                )

        dubins_sol = _build_shortest_dubins_local(start_pose=start_pose, goal_pose=goal_pose, turn_radius=turn_radius)
        if dubins_sol is not None:
            word, params = dubins_sol
            dubins_points = _sample_dubins_segment(
                start_pose=start_pose,
                word=word,
                params=params,
                turn_radius=turn_radius,
                sample_step=cfg.dubins_sample_step,
            )
            dubins_points[0] = (start_pose[0], start_pose[1])
            dubins_points[-1] = (goal_pose[0], goal_pose[1])

            if force_mode or _polyline_collision_free(
                dubins_points,
                world=world,
                margin=cfg.dubins_collision_margin,
                ):
                debug_trace = (
                    f"{legacy_debug_prefix};direct:{word}:ok" if legacy_debug_prefix else f"direct:{word}:ok"
                )
                dubins_len = _polyline_length(dubins_points)
                straight_len = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
                added_curve = max(0.0, dubins_len - straight_len)
                ratio = 0.0 if dubins_len <= 1e-9 else max(0.0, min(1.0, added_curve / dubins_len))
                return _finalize_legacy_return(
                    dubins_points,
                    dubins_len,
                    DubinsHybridMeta(
                        used_fallback=False,
                        fallback_segments=0,
                        dubins_segments=1,
                        sample_count=len(dubins_points),
                        dubins_ratio=ratio,
                        debug_trace=debug_trace,
                    ),
                )
            direct_reason = "direct_collision"
            debug_trace = (
                f"{legacy_debug_prefix};direct:{word}:collision"
                if legacy_debug_prefix
                else f"direct:{word}:collision"
            )

        recovered_points, recovered_len, recovery_trace = _try_direct_recovery_path(
            start_pose=start_pose,
            goal_pose=goal_pose,
            turn_radius=turn_radius,
            sample_step=cfg.dubins_sample_step,
            world=world,
            margin=cfg.dubins_collision_margin,
        )
        if recovered_points and math.isfinite(recovered_len):
            straight_len = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
            added_curve = max(0.0, recovered_len - straight_len)
            ratio = 0.0 if recovered_len <= 1e-9 else max(0.0, min(1.0, added_curve / recovered_len))
            return _finalize_legacy_return(
                recovered_points,
                recovered_len,
                DubinsHybridMeta(
                    used_fallback=False,
                    fallback_segments=0,
                    dubins_segments=2,
                    sample_count=len(recovered_points),
                    dubins_ratio=ratio,
                    debug_trace=f"{debug_trace};recovery:{recovery_trace}",
                ),
            )

        if cfg.dubins_fallback_to_astar:
            return _finalize_legacy_return(
                base_path,
                base_len,
                DubinsHybridMeta(
                    used_fallback=True,
                    fallback_segments=1,
                    dubins_segments=0,
                    sample_count=len(base_path),
                    dubins_ratio=0.0,
                    fallback_reason=direct_reason,
                    fallback_details=f"{direct_reason}:1",
                    debug_trace=f"{debug_trace};recovery:{recovery_trace}",
                ),
            )
        return _finalize_legacy_return(
            [],
            float("inf"),
            DubinsHybridMeta(
                used_fallback=True,
                fallback_segments=1,
                dubins_segments=0,
                sample_count=0,
                dubins_ratio=0.0,
                fallback_reason=direct_reason,
                fallback_details=f"{direct_reason}:1",
                debug_trace=f"{debug_trace};recovery:{recovery_trace}",
            ),
        )

    out_points, out_len, out_meta = _build_fillet_result(
        sparse_path=sparse_path,
        fallback_path=base_path,
        fallback_len=base_len,
        start_pose=start_pose,
        goal_pose=goal_pose,
        world=world,
        cfg=cfg,
        turn_radius=turn_radius,
        force_mode=force_mode,
        debug_trace=legacy_debug_prefix,
    )
    return _finalize_legacy_return(out_points, out_len, out_meta)


def build_bid_verification_path(
    world: WorldMap,
    cfg,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    astar_planner: AStarPlanner,
    turn_radius: float,
    astar_path: Optional[List[Point2D]] = None,
    astar_length: Optional[float] = None,
) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
    verify_cfg = _build_bid_verification_cfg(cfg)
    points, length, meta = build_dubins_hybrid_path(
        world=world,
        cfg=verify_cfg,
        start_pose=start_pose,
        goal_pose=goal_pose,
        astar_planner=astar_planner,
        turn_radius=turn_radius,
        astar_path=astar_path,
        astar_length=astar_length,
    )
    return points, length, _tag_meta_debug(
        meta,
        "mode:bid_verification",
        "verify_mode_position_only",
        "verify_mode_terminal_heading_only",
    )


def build_final_execution_path(
    world: WorldMap,
    cfg,
    start_pose: Pose2D,
    goal_pose: Pose2D,
    astar_planner: AStarPlanner,
    turn_radius: float,
    astar_path: Optional[List[Point2D]] = None,
    astar_length: Optional[float] = None,
) -> Tuple[List[Point2D], float, DubinsHybridMeta]:
    points, length, meta = build_dubins_hybrid_path(
        world=world,
        cfg=cfg,
        start_pose=start_pose,
        goal_pose=goal_pose,
        astar_planner=astar_planner,
        turn_radius=turn_radius,
        astar_path=astar_path,
        astar_length=astar_length,
    )
    return points, length, _tag_meta_debug(meta, "mode:final_execution", "final_mode_heading_refined")
