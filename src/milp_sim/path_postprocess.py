from __future__ import annotations

import math
from typing import List, Tuple

from shapely.geometry import LineString, Point

from .config import SimulationConfig
from .cost_estimator import heading_to_point, wrap_to_pi
from .entities import Point2D
from .map_utils import WorldMap
from .planner_astar import AStarPlanner


def path_initial_turn_delta(path: List[Point2D], start_heading: float) -> float:
    if len(path) < 2:
        return 0.0
    return abs(wrap_to_pi(heading_to_point(path[0], path[1]) - start_heading))


def polyline_is_clear(world: WorldMap, points: List[Point2D], margin: float) -> bool:
    for p in points:
        if not world.point_in_bounds(p, margin=margin):
            return False
    if len(points) < 2:
        return True
    line = LineString(points)
    if margin > 0.0:
        swept = line.buffer(margin, cap_style=2, join_style=2)
        return not swept.intersects(world.obstacle_union)
    return not line.intersects(world.obstacle_union)


def _dedupe_points(points: List[Point2D]) -> List[Point2D]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def _unit(vec: Point2D) -> Point2D | None:
    norm = math.hypot(vec[0], vec[1])
    if norm <= 1e-9:
        return None
    return (vec[0] / norm, vec[1] / norm)


def _circle_center_from_three_points(p1: Point2D, p2: Point2D, p3: Point2D) -> tuple[Point2D, float] | None:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) <= 1e-9:
        return None

    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3

    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / det
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / det
    r = math.hypot(x1 - ux, y1 - uy)
    if not math.isfinite(r) or r <= 1e-9:
        return None
    return (ux, uy), r


def _ccw_delta(a: float, b: float) -> float:
    return (b - a) % (2.0 * math.pi)


def _sample_arc(
    center: Point2D,
    radius: float,
    a0: float,
    a1: float,
    *,
    ccw: bool,
    max_step: float,
) -> List[Point2D]:
    total = _ccw_delta(a0, a1) if ccw else _ccw_delta(a1, a0)
    arc_len = radius * total
    n = max(1, int(math.ceil(arc_len / max(max_step, 1e-3))))
    out: List[Point2D] = []
    for k in range(n + 1):
        t = k / n
        ang = a0 + total * t if ccw else a0 - total * t
        out.append((center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)))
    return out


def _sample_arc_through_waypoint(
    start: Point2D,
    waypoint: Point2D,
    end: Point2D,
    *,
    max_step: float,
) -> List[Point2D] | None:
    circle = _circle_center_from_three_points(start, waypoint, end)
    if circle is None:
        return None
    center, radius = circle
    a_start = math.atan2(start[1] - center[1], start[0] - center[0])
    a_mid = math.atan2(waypoint[1] - center[1], waypoint[0] - center[0])
    a_end = math.atan2(end[1] - center[1], end[0] - center[0])

    ccw_ok = _ccw_delta(a_start, a_mid) <= _ccw_delta(a_start, a_end) + 1e-9
    cw_ok = _ccw_delta(a_end, a_mid) <= _ccw_delta(a_end, a_start) + 1e-9
    if not ccw_ok and not cw_ok:
        return None

    if ccw_ok and (not cw_ok or _ccw_delta(a_start, a_end) <= _ccw_delta(a_end, a_start)):
        first = _sample_arc(center, radius, a_start, a_mid, ccw=True, max_step=max_step)
        second = _sample_arc(center, radius, a_mid, a_end, ccw=True, max_step=max_step)
    else:
        first = _sample_arc(center, radius, a_start, a_mid, ccw=False, max_step=max_step)
        second = _sample_arc(center, radius, a_mid, a_end, ccw=False, max_step=max_step)

    if not first or not second:
        return None
    return _dedupe_points(first + second[1:])


def segment_is_clear(world: WorldMap, start: Point2D, end: Point2D, margin: float) -> bool:
    return polyline_is_clear(world, [start, end], margin=margin)


def shortcut_polyline(world: WorldMap, points: List[Point2D], margin: float) -> List[Point2D]:
    path = _dedupe_points(points)
    if len(path) <= 2:
        return path

    out: List[Point2D] = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        next_idx = anchor + 1
        for probe in range(len(path) - 1, anchor + 1, -1):
            if segment_is_clear(world, path[anchor], path[probe], margin=margin):
                next_idx = probe
                break
        out.append(path[next_idx])
        anchor = next_idx
    return _dedupe_points(out)


def string_pull_polyline(
    world: WorldMap,
    points: List[Point2D],
    margin: float,
    passes: int = 2,
) -> List[Point2D]:
    path = _dedupe_points(points)
    if len(path) <= 2:
        return path

    out = list(path)
    for _ in range(max(0, int(passes))):
        updated = shortcut_polyline(world, out, margin=margin)
        updated = list(reversed(shortcut_polyline(world, list(reversed(updated)), margin=margin)))
        if len(updated) == len(out) and all(
            math.hypot(a[0] - b[0], a[1] - b[1]) <= 1e-9 for a, b in zip(updated, out)
        ):
            break
        out = updated
    return _dedupe_points(out)


def simplify_reference_polyline(
    world: WorldMap,
    points: List[Point2D],
    margin: float,
    *,
    enable_shortcut: bool = True,
    enable_string_pull: bool = True,
    string_pull_passes: int = 2,
    split_turn_angle_threshold: float = 0.0,
) -> List[Point2D]:
    path = _dedupe_points(points)
    if len(path) <= 2:
        return path

    if enable_shortcut:
        path = shortcut_polyline(world, path, margin=margin)
    if enable_string_pull:
        path = string_pull_polyline(world, path, margin=margin, passes=string_pull_passes)
    if len(path) <= 2 or split_turn_angle_threshold <= 1e-9:
        return path

    out: List[Point2D] = [path[0]]
    threshold = float(split_turn_angle_threshold)
    for i in range(1, len(path) - 1):
        prev_pt = out[-1]
        cur_pt = path[i]
        next_pt = path[i + 1]
        ax = cur_pt[0] - prev_pt[0]
        ay = cur_pt[1] - prev_pt[1]
        bx = next_pt[0] - cur_pt[0]
        by = next_pt[1] - cur_pt[1]
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 1e-9 or lb <= 1e-9:
            continue
        dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        turn = math.acos(dot)
        if turn + 1e-9 >= threshold:
            out.append(cur_pt)
    out.append(path[-1])
    return _dedupe_points(out)


def _sample_turn_recovery_arc(
    cfg: SimulationConfig,
    start_pose: Tuple[float, float, float],
    turn_radius: float,
    delta_heading: float,
) -> Tuple[List[Point2D], Tuple[float, float, float]]:
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
    step = max(0.1, float(getattr(cfg, "dubins_sample_step", 0.5)))
    n = max(1, int(math.ceil(arc_len / step)))
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


def maybe_buffer_initial_turn_path(
    world: WorldMap,
    cfg: SimulationConfig,
    planner: AStarPlanner,
    start_pos: Point2D,
    start_heading: float,
    task_pos: Point2D,
    path: List[Point2D],
    length: float,
    goal_heading: float,
    turn_radius: float,
) -> Tuple[List[Point2D], float]:
    from .dubins_path import build_final_execution_path

    if len(path) < 2 or not math.isfinite(length):
        return path, length

    max_initial_turn = max(
        0.0,
        float(getattr(cfg, "online_max_initial_turn_rad", math.pi / 4.0)),
    )
    current_delta = path_initial_turn_delta(path, start_heading)
    if current_delta <= max_initial_turn + 1e-9:
        return path, length

    desired_heading = heading_to_point(start_pos, task_pos)
    desired_delta = wrap_to_pi(desired_heading - start_heading)
    if abs(desired_delta) <= max_initial_turn + 1e-9:
        return path, length

    clearance = max(
        float(getattr(cfg, "dubins_collision_margin", 0.0)),
        float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
    )
    needed_turn = max(0.0, abs(desired_delta) - max_initial_turn)
    candidate_steps = [
        min(abs(desired_delta), needed_turn),
        min(abs(desired_delta), math.radians(60.0)),
        min(abs(desired_delta), math.radians(45.0)),
        min(abs(desired_delta), math.radians(30.0)),
    ]

    tried: set[int] = set()
    for step in candidate_steps:
        key = int(round(step * 1_000_000))
        if step <= 1e-6 or key in tried:
            continue
        tried.add(key)

        signed_step = math.copysign(step, desired_delta)
        arc_points, arc_end_pose = _sample_turn_recovery_arc(
            cfg=cfg,
            start_pose=(start_pos[0], start_pos[1], start_heading),
            turn_radius=turn_radius,
            delta_heading=signed_step,
        )
        if not world.point_is_free(arc_end_pose[:2], clearance=clearance):
            continue
        if Point(arc_end_pose[:2]).within(world.depot_polygon):
            continue
        if not polyline_is_clear(world, arc_points, margin=clearance):
            continue

        tail_path, tail_length, _ = build_final_execution_path(
            world=world,
            cfg=cfg,
            start_pose=arc_end_pose,
            goal_pose=(task_pos[0], task_pos[1], goal_heading),
            astar_planner=planner,
            turn_radius=turn_radius,
        )
        if not tail_path or not math.isfinite(tail_length):
            continue

        full_path = arc_points + tail_path[1:]
        if not polyline_is_clear(world, full_path, margin=clearance):
            continue

        join_delta = path_initial_turn_delta(full_path[len(arc_points) - 1 :], arc_end_pose[2])
        if join_delta > max_initial_turn + 1e-9:
            continue

        full_length = turn_radius * abs(signed_step) + tail_length
        return full_path, full_length

    return path, length


def smooth_task_joint_path(
    world: WorldMap,
    cfg: SimulationConfig,
    points: List[Point2D],
    task_waypoint_indices: List[int],
    *,
    turn_radius: float,
) -> List[Point2D]:
    out = _dedupe_points(points)
    if len(out) < 3 or not task_waypoint_indices:
        return out

    max_step = max(
        0.05,
        float(
            getattr(
                cfg,
                "online_path_sample_step",
                min(0.25, float(getattr(cfg, "dubins_sample_step", 0.5))),
            )
        ),
    )
    clearance = max(
        float(getattr(cfg, "dubins_collision_margin", 0.0)),
        float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
    )
    # Keep this local and conservative: smooth only meaningful corners and never clip obstacles.
    min_turn = math.radians(20.0)
    base_cut = max(0.4, min(2.0, 0.9 * max(turn_radius, 1e-6)))

    for idx in sorted(set(task_waypoint_indices), reverse=True):
        if idx <= 0 or idx >= len(out) - 1:
            continue

        prev_pt = out[idx - 1]
        cur_pt = out[idx]
        next_pt = out[idx + 1]

        in_vec = (cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1])
        out_vec = (next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])
        in_len = math.hypot(in_vec[0], in_vec[1])
        out_len = math.hypot(out_vec[0], out_vec[1])
        if in_len <= 1e-6 or out_len <= 1e-6:
            continue

        in_dir = _unit(in_vec)
        out_dir = _unit(out_vec)
        if in_dir is None or out_dir is None:
            continue

        dot = max(-1.0, min(1.0, in_dir[0] * out_dir[0] + in_dir[1] * out_dir[1]))
        turn = math.acos(dot)
        if turn < min_turn:
            continue

        cut = min(base_cut, 0.45 * in_len, 0.45 * out_len)
        if cut <= max_step:
            continue

        arc_start = (cur_pt[0] - in_dir[0] * cut, cur_pt[1] - in_dir[1] * cut)
        arc_end = (cur_pt[0] + out_dir[0] * cut, cur_pt[1] + out_dir[1] * cut)
        arc_pts = _sample_arc_through_waypoint(arc_start, cur_pt, arc_end, max_step=max_step)
        if arc_pts is None or len(arc_pts) < 3:
            continue

        local_candidate = [prev_pt] + arc_pts + [next_pt]
        if not polyline_is_clear(world, local_candidate, margin=clearance):
            continue

        out = out[: idx - 1] + local_candidate + out[idx + 2 :]

    return _dedupe_points(out)


def resample_path(points: List[Point2D], max_step: float) -> List[Point2D]:
    if len(points) < 2 or max_step <= 1e-6:
        return list(points)

    out: List[Point2D] = [points[0]]
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        seg_len = math.hypot(end[0] - start[0], end[1] - start[1])
        if seg_len <= 1e-9:
            continue

        n = max(1, int(math.ceil(seg_len / max_step)))
        for k in range(1, n + 1):
            alpha = k / n
            p = (
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
            )
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
                out.append(p)
    return out
