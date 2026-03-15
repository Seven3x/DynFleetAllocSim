from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
) -> Optional[_CornerFillet]:
    vin_raw = _sub(corner_pt, prev_pt)
    vout_raw = _sub(next_pt, corner_pt)
    len_in = _norm(vin_raw)
    len_out = _norm(vout_raw)
    if len_in <= 1e-9 or len_out <= 1e-9:
        return None

    vin = _unit(vin_raw)
    vout = _unit(vout_raw)
    if vin is None or vout is None:
        return None

    dot = max(-1.0, min(1.0, vin[0] * vout[0] + vin[1] * vout[1]))
    phi = math.acos(dot)
    if phi <= 1e-3 or abs(math.pi - phi) <= 1e-3:
        return None

    tan_half = math.tan(phi / 2.0)
    if abs(tan_half) <= 1e-9:
        return None

    r_max = max(0.0, min(len_in, len_out) * tan_half * 0.98)
    r_eff = min(radius, r_max)
    if r_eff <= 1e-4:
        return None

    t = r_eff / tan_half
    t_in = _sub(corner_pt, _scale(vin, t))
    t_out = _add(corner_pt, _scale(vout, t))

    cross = vin[0] * vout[1] - vin[1] * vout[0]
    if abs(cross) <= 1e-9:
        return None
    left_turn = cross > 0.0
    n_in = _left_normal(vin) if left_turn else _right_normal(vin)
    center = _add(t_in, _scale(n_in, r_eff))

    arc = _sample_arc(center=center, radius=r_eff, start=t_in, end=t_out, left_turn=left_turn, step=step)
    if not force_mode and not _collision_free(arc, world=world, margin=margin):
        return None
    return _CornerFillet(t_in=t_in, t_out=t_out, arc_points=arc)


def _build_fillet_polyline(
    path: List[Point2D],
    turn_radius: float,
    sample_step: float,
    world: WorldMap,
    margin: float,
    force_mode: bool,
) -> Tuple[List[Point2D], int, int]:
    if len(path) < 3:
        return path, 0, 0

    corners: Dict[int, _CornerFillet] = {}
    for i in range(1, len(path) - 1):
        fillet = _build_corner_fillet(
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

    return out, len(corners), (len(path) - 2 - len(corners))


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
            ),
        )

    if not getattr(cfg, "use_dubins_hybrid", False):
        return (
            astar_path,
            astar_length,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(astar_path),
                dubins_ratio=0.0,
            ),
        )

    sparse_path = _compress_astar_path(astar_path)
    if len(sparse_path) < 2:
        return (
            astar_path,
            0.0,
            DubinsHybridMeta(
                used_fallback=False,
                fallback_segments=0,
                dubins_segments=0,
                sample_count=len(astar_path),
                dubins_ratio=0.0,
            ),
        )

    fillet_points, fillet_count, fallback_count = _build_fillet_polyline(
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
            ),
        )
    out_points = fillet_points
    out_points[0] = (start_pose[0], start_pose[1])
    out_points[-1] = (goal_pose[0], goal_pose[1])

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
    )
    return out_points, total_len, meta
