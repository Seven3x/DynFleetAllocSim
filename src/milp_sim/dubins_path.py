from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from shapely.geometry import LineString

from .cost_estimator import heading_to_point, wrap_to_pi
from .map_utils import WorldMap
from .planner_astar import AStarPlanner


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
            ),
        )

    smoothing_margin = max(
        float(getattr(cfg, "dubins_collision_margin", 0.0)),
        float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
    )
    if getattr(cfg, "astar_smooth_before_dubins", True):
        base_path = _shortcut_smooth_path(astar_path, world=world, margin=smoothing_margin)
        if not _polyline_collision_free(base_path, world=world, margin=smoothing_margin):
            base_path = astar_path
    else:
        base_path = astar_path
    base_len = _polyline_length(base_path)

    if not getattr(cfg, "use_dubins_hybrid", False):
        return (
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
            ),
        )

    sparse_path = _compress_astar_path(base_path)
    if len(sparse_path) < 2:
        return (
            base_path,
            base_len,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(base_path),
                dubins_ratio=0.0,
            ),
        )

    if len(sparse_path) == 2:
        direct_reason = "direct_no_solution"
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
                dubins_len = _polyline_length(dubins_points)
                straight_len = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
                added_curve = max(0.0, dubins_len - straight_len)
                ratio = 0.0 if dubins_len <= 1e-9 else max(0.0, min(1.0, added_curve / dubins_len))
                return (
                    dubins_points,
                    dubins_len,
                    DubinsHybridMeta(
                        used_fallback=False,
                        fallback_segments=0,
                        dubins_segments=1,
                        sample_count=len(dubins_points),
                        dubins_ratio=ratio,
                    ),
                )
            direct_reason = "direct_collision"

        if cfg.dubins_fallback_to_astar:
            return (
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
                ),
            )
        return (
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
            ),
        )

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
                base_path,
                base_len,
                DubinsHybridMeta(
                    used_fallback=True,
                    fallback_segments=rejected_segments,
                    dubins_segments=fillet_count,
                    sample_count=len(base_path),
                    dubins_ratio=0.0,
                    fallback_reason="final_collision",
                    fallback_details=_format_reason_counts(final_reason_counts),
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
    )
    return out_points, total_len, meta
