from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, box

from .config import SimulationConfig
from .entities import Task, Vehicle
from .map_utils import WorldMap


def _random_polygon(
    rng: np.random.Generator,
    center: Tuple[float, float],
    n_vertices: int,
    radius_min: float,
    radius_max: float,
    concave: bool,
) -> Polygon:
    cx, cy = center
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=n_vertices))

    if concave:
        radii = []
        for i in range(n_vertices):
            base = rng.uniform(radius_min, radius_max)
            if i % 2 == 1:
                base *= rng.uniform(0.35, 0.65)
            radii.append(base)
        radii = np.asarray(radii)
    else:
        radii = rng.uniform(radius_min, radius_max, size=n_vertices)

    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a, r in zip(angles, radii)]
    poly = Polygon(pts).buffer(0.0)
    return poly


def generate_world_map(cfg: SimulationConfig, rng: np.random.Generator) -> WorldMap:
    depot = box(cfg.depot_min_x, cfg.depot_min_y, cfg.depot_max_x, cfg.depot_max_y)
    world = WorldMap(width=cfg.map_width, height=cfg.map_height, depot_polygon=depot, obstacles=[])

    retries = 0
    while len(world.obstacles) < cfg.num_obstacles and retries < cfg.obstacle_max_retries:
        retries += 1
        n_vertices = int(rng.integers(cfg.obstacle_vertex_min, cfg.obstacle_vertex_max + 1))
        concave = bool(rng.random() < 0.45)

        cx = float(
            rng.uniform(
                cfg.obstacle_margin_from_border,
                cfg.map_width - cfg.obstacle_margin_from_border,
            )
        )
        cy = float(
            rng.uniform(
                cfg.obstacle_margin_from_border,
                cfg.map_height - cfg.obstacle_margin_from_border,
            )
        )
        radius_max = float(rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max))
        radius_min = max(1.2, radius_max * float(rng.uniform(0.45, 0.75)))

        poly = _random_polygon(
            rng=rng,
            center=(cx, cy),
            n_vertices=n_vertices,
            radius_min=radius_min,
            radius_max=radius_max,
            concave=concave,
        )

        if poly.is_empty or not poly.is_valid or poly.area < 8.0:
            continue

        if not poly.within(world.boundary_polygon):
            continue

        if poly.intersects(world.depot_polygon):
            continue

        if any(poly.intersects(existing) for existing in world.obstacles):
            continue

        world.add_obstacle(poly)

    if len(world.obstacles) < cfg.num_obstacles:
        raise RuntimeError(
            f"Obstacle generation failed: got {len(world.obstacles)} / {cfg.num_obstacles}. "
            "Try lowering obstacle density or increasing retries."
        )

    return world


def _sample_free_point(
    world: WorldMap,
    rng: np.random.Generator,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    clearance: float,
    avoid_depot: bool,
    max_tries: int = 5000,
) -> Tuple[float, float]:
    for _ in range(max_tries):
        p = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
        if avoid_depot and Point(p).within(world.depot_polygon):
            continue
        if world.point_is_free(p, clearance=clearance):
            return p
    raise RuntimeError("Failed to sample a free point in the map.")


def generate_vehicles(cfg: SimulationConfig, world: WorldMap, rng: np.random.Generator) -> List[Vehicle]:
    vehicles: List[Vehicle] = []

    for i in range(cfg.num_vehicles):
        start = _sample_free_point(
            world=world,
            rng=rng,
            x_min=cfg.depot_min_x,
            x_max=cfg.depot_max_x,
            y_min=cfg.depot_min_y,
            y_max=cfg.depot_max_y,
            clearance=cfg.vehicle_radius + 0.2,
            avoid_depot=False,
        )

        speed = float(rng.uniform(cfg.vehicle_speed_min, cfg.vehicle_speed_max))
        omega = float(rng.uniform(cfg.vehicle_omega_min, cfg.vehicle_omega_max))
        capacity = int(rng.integers(cfg.vehicle_capacity_min, cfg.vehicle_capacity_max + 1))

        vehicle = Vehicle(
            id=i,
            start_pos=start,
            heading=float(rng.uniform(-math.pi, math.pi)),
            speed=speed,
            max_omega=omega,
            capacity=capacity,
            remaining_capacity=capacity,
        )
        vehicle.reset_runtime_state()
        vehicles.append(vehicle)

    return vehicles


def generate_tasks(cfg: SimulationConfig, world: WorldMap, rng: np.random.Generator) -> List[Task]:
    tasks: List[Task] = []

    for t_id in range(cfg.num_tasks):
        pos = _sample_free_point(
            world=world,
            rng=rng,
            x_min=0.0,
            x_max=cfg.map_width,
            y_min=0.0,
            y_max=cfg.map_height,
            clearance=cfg.vehicle_radius + 0.2,
            avoid_depot=True,
        )

        demand = int(rng.integers(cfg.task_demand_min, cfg.task_demand_max + 1))
        tasks.append(Task(id=t_id, position=pos, demand=demand))

    total_demand = sum(t.demand for t in tasks)
    total_capacity = cfg.num_vehicles * cfg.vehicle_capacity_max
    if total_demand > total_capacity:
        raise RuntimeError(
            f"Generated task demand {total_demand} exceeds max fleet capacity {total_capacity}."
        )

    return tasks
