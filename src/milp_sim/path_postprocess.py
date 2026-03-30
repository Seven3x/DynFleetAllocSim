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


def path_clearance_margin(cfg: SimulationConfig) -> float:
    base_margin = float(getattr(cfg, "dubins_collision_margin", 0.0))
    if bool(getattr(cfg, "trajectory_guard_use_vehicle_footprint", False)):
        return max(
            base_margin,
            float(getattr(cfg, "vehicle_radius", 0.0)) + float(getattr(cfg, "safety_margin", 0.0)),
        )
    return base_margin


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


def _joint_context_index(points: List[Point2D], center_idx: int, *, direction: int, min_span: float) -> int | None:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if center_idx <= 0 or center_idx >= len(points) - 1:
        return None
    traveled = 0.0
    idx = center_idx
    while True:
        nxt = idx + direction
        if nxt < 0 or nxt >= len(points):
            return None
        traveled += math.hypot(points[nxt][0] - points[idx][0], points[nxt][1] - points[idx][1])
        idx = nxt
        if traveled >= min_span - 1e-9:
            return idx


def _heading_between(src: Point2D, dst: Point2D) -> float:
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def _angle_bisector_heading(h1: float, h2: float) -> float:
    sx = math.cos(h1) + math.cos(h2)
    sy = math.sin(h1) + math.sin(h2)
    if abs(sx) <= 1e-9 and abs(sy) <= 1e-9:
        return h2
    return math.atan2(sy, sx)


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

    clearance = path_clearance_margin(cfg)
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


def rebuild_fallback_path_with_heading_release(
    world: WorldMap,
    cfg: SimulationConfig,
    planner: AStarPlanner,
    *,
    start_pos: Point2D,
    start_heading: float,
    goal_pos: Point2D,
    base_path: List[Point2D],
    base_length: float,
) -> Tuple[List[Point2D], float]:
    if not base_path or not math.isfinite(base_length):
        return base_path, base_length

    clearance = path_clearance_margin(cfg)
    initial_turn = path_initial_turn_delta(base_path, start_heading)
    if initial_turn <= math.pi / 2.0 + 1e-9:
        return base_path, base_length

    best_path = list(base_path)
    best_length = float(base_length)
    best_score = float("inf")
    for release_distance in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        release_point = (
            start_pos[0] + math.cos(start_heading) * release_distance,
            start_pos[1] + math.sin(start_heading) * release_distance,
        )
        prefix = [start_pos, release_point]
        if not world.point_is_free(release_point, clearance=clearance):
            continue
        if not polyline_is_clear(world, prefix, margin=clearance):
            continue

        tail_path, tail_length = planner.plan(release_point, goal_pos)
        if not tail_path or not math.isfinite(tail_length):
            continue

        candidate = _dedupe_points(prefix + list(tail_path[1:]))
        if len(candidate) < 2 or not polyline_is_clear(world, candidate, margin=clearance):
            continue

        candidate_initial_turn = path_initial_turn_delta(candidate, start_heading)
        score = candidate_length = 0.0
        for i in range(len(candidate) - 1):
            candidate_length += math.hypot(candidate[i + 1][0] - candidate[i][0], candidate[i + 1][1] - candidate[i][1])
        score = candidate_initial_turn + 0.02 * candidate_length
        if score < best_score:
            best_score = score
            best_path = candidate
            best_length = candidate_length

    return best_path, best_length


def stabilize_terminal_approach_path(
    world: WorldMap,
    cfg: SimulationConfig,
    points: List[Point2D],
) -> Tuple[List[Point2D], float]:
    path = _dedupe_points(points)
    if len(path) < 4:
        return path, sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)
        )

    clearance = path_clearance_margin(cfg)
    goal = path[-1]
    tail_window = min(len(path) - 2, 24)
    best_path = list(path)
    best_score = float("inf")
    best_length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))

    def _tail_score(candidate: List[Point2D]) -> tuple[float, float]:
        tail = candidate[-min(len(candidate), 20) :]
        signed_deltas: List[float] = []
        prev_heading: float | None = None
        for i in range(len(tail) - 1):
            h = math.atan2(tail[i + 1][1] - tail[i][1], tail[i + 1][0] - tail[i][0])
            if prev_heading is not None:
                signed_deltas.append(wrap_to_pi(h - prev_heading))
            prev_heading = h
        wiggle_threshold = math.radians(3.0)
        flips = 0
        prev_sign = 0
        energy = 0.0
        for delta in signed_deltas:
            if abs(delta) <= wiggle_threshold:
                continue
            energy += abs(delta)
            sign = 1 if delta > 0.0 else -1
            if prev_sign != 0 and sign != prev_sign:
                flips += 1
            prev_sign = sign
        return float(flips), energy

    for keep_back in range(2, tail_window + 1):
        anchor_idx = len(path) - 1 - keep_back
        anchor = path[anchor_idx]
        shortcut = path[: anchor_idx + 1] + [goal]
        if not polyline_is_clear(world, shortcut[-2:], margin=clearance):
            continue
        if not polyline_is_clear(world, shortcut, margin=clearance):
            continue
        flips, energy = _tail_score(shortcut)
        candidate_length = sum(
            math.hypot(shortcut[i + 1][0] - shortcut[i][0], shortcut[i + 1][1] - shortcut[i][1])
            for i in range(len(shortcut) - 1)
        )
        score = 4.0 * flips + energy + 0.01 * candidate_length
        if score < best_score - 1e-9:
            best_score = score
            best_path = shortcut
            best_length = candidate_length

    return best_path, best_length


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
    clearance = path_clearance_margin(cfg)
    # Keep this local and conservative, but probe several cut sizes so hard task joints
    # still get a chance to smooth even when the largest arc collides.
    min_turn = math.radians(10.0)
    base_cut = max(0.4, min(2.4, 1.0 * max(turn_radius, 1e-6)))
    context_span = max(0.75, 3.0 * max_step)

    for idx in sorted(set(task_waypoint_indices), reverse=True):
        if idx <= 0 or idx >= len(out) - 1:
            continue

        prev_idx = _joint_context_index(out, idx, direction=-1, min_span=context_span)
        next_idx = _joint_context_index(out, idx, direction=1, min_span=context_span)
        if prev_idx is None or next_idx is None or prev_idx >= idx or next_idx <= idx:
            prev_idx = idx - 1
            next_idx = idx + 1

        prev_pt = out[prev_idx]
        cur_pt = out[idx]
        next_pt = out[next_idx]

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

        base_cut_limit = min(base_cut, 0.48 * in_len, 0.48 * out_len)
        if base_cut_limit <= max_step:
            continue

        improved = False
        for cut_scale in (1.0, 0.8, 0.65, 0.5, 0.35):
            cut = base_cut_limit * cut_scale
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

            out = out[:prev_idx] + local_candidate + out[next_idx + 1 :]
            improved = True
            break

        if improved:
            continue

        h_in = _heading_between(prev_pt, cur_pt)
        h_out = _heading_between(cur_pt, next_pt)
        tangent_heading = _angle_bisector_heading(h_in, h_out)
        tangent_limit = min(0.45 * in_len, 0.45 * out_len)
        for offset_scale in (0.35, 0.25, 0.18):
            offset = min(tangent_limit, base_cut_limit * offset_scale)
            if offset <= max_step:
                continue
            pre_pt = (
                cur_pt[0] - math.cos(tangent_heading) * offset,
                cur_pt[1] - math.sin(tangent_heading) * offset,
            )
            post_pt = (
                cur_pt[0] + math.cos(tangent_heading) * offset,
                cur_pt[1] + math.sin(tangent_heading) * offset,
            )
            bridge_candidate = [prev_pt, pre_pt, cur_pt, post_pt, next_pt]
            if not polyline_is_clear(world, bridge_candidate, margin=clearance):
                relaxed_margin = max(0.0, 0.5 * clearance)
                if relaxed_margin + 1e-9 < clearance and not polyline_is_clear(
                    world, bridge_candidate, margin=relaxed_margin
                ):
                    continue
            out = out[:prev_idx] + bridge_candidate + out[next_idx + 1 :]
            break

    return _dedupe_points(out)


def smooth_task_segment_path(
    world: WorldMap,
    cfg: SimulationConfig,
    points: List[Point2D],
    task_waypoint_indices: List[int],
) -> tuple[List[Point2D], List[int]]:
    out = _dedupe_points(points)
    if len(out) < 3 or not task_waypoint_indices:
        return out, list(task_waypoint_indices)

    clearance = path_clearance_margin(cfg)
    new_points: List[Point2D] = [out[0]]
    new_indices: List[int] = []
    anchor = 0

    for idx in task_waypoint_indices:
        if idx <= anchor or idx >= len(out):
            continue
        segment = out[anchor : idx + 1]
        simplified = simplify_reference_polyline(
            world,
            segment,
            margin=clearance,
            enable_shortcut=bool(getattr(cfg, "connector_shortcut_enable", True)),
            enable_string_pull=bool(getattr(cfg, "connector_string_pull_enable", True)),
            string_pull_passes=int(getattr(cfg, "connector_string_pull_passes", 2)),
            split_turn_angle_threshold=0.0,
        )
        if len(simplified) < 2:
            simplified = list(segment)

        simplified[0] = segment[0]
        simplified[-1] = segment[-1]
        if not polyline_is_clear(world, simplified, margin=clearance):
            simplified = list(segment)
        if len(new_points) > 0:
            new_points.extend(simplified[1:])
        else:
            new_points.extend(simplified)
        new_indices.append(len(new_points) - 1)
        anchor = idx

    return _dedupe_points(new_points), new_indices


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
