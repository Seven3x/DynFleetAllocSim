from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from shapely.geometry import LineString

from .config import SimulationConfig
from .entities import Task, Vehicle
from .map_utils import WorldMap


@dataclass
class CostDetail:
    distance: float
    delta_heading: float
    obstacle_density: float
    estimated_length: float
    estimated_time: float


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_to_point(src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def blended_goal_heading(
    current_pos: Tuple[float, float],
    task_pos: Tuple[float, float],
    next_task_pos: Tuple[float, float] | None,
    turn_radius: float,
    blend_turn_radius_factor: float,
) -> float:
    direct_heading = heading_to_point(current_pos, task_pos)
    if next_task_pos is None:
        return direct_heading

    next_heading = heading_to_point(task_pos, next_task_pos)
    if turn_radius <= 1e-9 or blend_turn_radius_factor <= 1e-9:
        return next_heading

    dist_to_task = math.hypot(task_pos[0] - current_pos[0], task_pos[1] - current_pos[1])
    blend_distance = max(1e-6, blend_turn_radius_factor * turn_radius)
    w = max(0.0, min(1.0, dist_to_task / blend_distance))
    return wrap_to_pi(direct_heading + w * wrap_to_pi(next_heading - direct_heading))


def corridor_density(
    world: WorldMap,
    src: Tuple[float, float],
    dst: Tuple[float, float],
    corridor_width: float,
) -> float:
    if src == dst:
        return 0.0

    corridor = LineString([src, dst]).buffer(corridor_width / 2.0, cap_style=2, join_style=2)
    area = corridor.area
    if area <= 1e-9:
        return 0.0

    overlap = corridor.intersection(world.obstacle_union).area
    rho = overlap / area
    return max(0.0, min(1.0, float(rho)))


def fast_cost_estimate(
    vehicle: Vehicle,
    task: Task,
    world: WorldMap,
    cfg: SimulationConfig,
) -> CostDetail:
    src = vehicle.current_pos
    dst = task.position

    d = math.hypot(dst[0] - src[0], dst[1] - src[1])
    tgt_heading = heading_to_point(src, dst)
    delta = abs(wrap_to_pi(tgt_heading - vehicle.current_heading))

    rho = corridor_density(
        world=world,
        src=src,
        dst=dst,
        corridor_width=cfg.corridor_width,
    )

    turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)
    est_length = d + cfg.lambda_psi * turn_radius * delta + cfg.lambda_rho * d * rho
    est_time = est_length / vehicle.speed

    return CostDetail(
        distance=d,
        delta_heading=delta,
        obstacle_density=rho,
        estimated_length=est_length,
        estimated_time=est_time,
    )
