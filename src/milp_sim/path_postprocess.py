from __future__ import annotations

import math
from typing import List, Tuple

from shapely.geometry import LineString, Point

from .config import SimulationConfig
from .cost_estimator import heading_to_point, wrap_to_pi
from .dubins_path import build_dubins_hybrid_path
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

        tail_path, tail_length, _ = build_dubins_hybrid_path(
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
