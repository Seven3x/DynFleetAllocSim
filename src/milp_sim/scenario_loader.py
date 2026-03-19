from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Point, Polygon

from .entities import Task, Vehicle
from .map_utils import WorldMap


@dataclass
class LoadedScenario:
    world: WorldMap
    vehicles: list[Vehicle]
    tasks: list[Task]


def load_scenario_file(path: str | Path) -> LoadedScenario:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"scenario file not found: {source}")
    if not source.is_file():
        raise ValueError(f"scenario path is not a file: {source}")

    payload = _read_payload(source)
    return _parse_payload(payload, source=source)


def _read_payload(source: Path) -> dict[str, Any]:
    suffix = source.suffix.lower()
    text = source.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ValueError("YAML scenario requires PyYAML (`pip install pyyaml`)") from exc
        loaded = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported scenario format: {source.suffix} (use .json/.yaml/.yml)")

    if not isinstance(loaded, dict):
        raise ValueError("scenario root must be a mapping")
    return loaded


def _parse_payload(payload: dict[str, Any], source: Path) -> LoadedScenario:
    world = _parse_world(_require_mapping(payload, "world"))
    vehicles = _parse_vehicles(_require_list(payload, "vehicles"), world=world)
    tasks = _parse_tasks(_require_list(payload, "tasks"), world=world)

    if not vehicles:
        raise ValueError("scenario must contain at least one vehicle")
    if not tasks:
        raise ValueError("scenario must contain at least one task")

    total_demand = sum(task.demand for task in tasks)
    total_capacity = sum(vehicle.capacity for vehicle in vehicles)
    if total_demand > total_capacity:
        raise ValueError(
            f"scenario demand exceeds capacity: demand={total_demand}, capacity={total_capacity} ({source})"
        )

    return LoadedScenario(world=world, vehicles=vehicles, tasks=tasks)


def _parse_world(raw: dict[str, Any]) -> WorldMap:
    width = _read_positive_float(raw, "width")
    height = _read_positive_float(raw, "height")
    depot_points = _read_polygon_points(raw, key="depot_polygon", label="world.depot_polygon")
    depot_polygon = _build_polygon(depot_points, label="world.depot_polygon")

    raw_obstacles = raw.get("obstacles", [])
    if raw_obstacles is None:
        raw_obstacles = []
    if not isinstance(raw_obstacles, list):
        raise ValueError("world.obstacles must be a list")

    obstacles: list[Polygon] = []
    for idx, item in enumerate(raw_obstacles):
        points = _read_polygon_points_from_value(item, label=f"world.obstacles[{idx}]")
        obstacles.append(_build_polygon(points, label=f"world.obstacles[{idx}]"))

    world = WorldMap(width=width, height=height, depot_polygon=depot_polygon, obstacles=obstacles)

    if not world.boundary_polygon.covers(depot_polygon):
        raise ValueError("world.depot_polygon must be inside world boundary")
    for idx, obstacle in enumerate(obstacles):
        if not world.boundary_polygon.covers(obstacle):
            raise ValueError(f"world.obstacles[{idx}] must be inside world boundary")
        if obstacle.intersects(depot_polygon):
            raise ValueError(f"world.obstacles[{idx}] intersects depot_polygon")

    return world


def _parse_vehicles(raw: list[Any], world: WorldMap) -> list[Vehicle]:
    vehicles: list[Vehicle] = []
    seen_ids: set[int] = set()

    for idx, item in enumerate(raw):
        row = _as_mapping(item, label=f"vehicles[{idx}]")
        vid = _read_int(row, "id", label=f"vehicles[{idx}].id")
        if vid in seen_ids:
            raise ValueError(f"duplicate vehicle id: {vid}")
        seen_ids.add(vid)

        start_raw = row.get("start_pos")
        if start_raw is None:
            start_raw = row.get("start")
        if start_raw is None:
            raise ValueError(f"vehicles[{idx}] missing start/start_pos")
        start = _read_point(start_raw, label=f"vehicles[{idx}].start")

        heading = _read_float(row, "heading", label=f"vehicles[{idx}].heading")
        speed = _read_positive_float(row, "speed", label=f"vehicles[{idx}].speed")
        max_omega = _read_positive_float(row, "max_omega", label=f"vehicles[{idx}].max_omega")
        capacity = _read_positive_int(row, "capacity", label=f"vehicles[{idx}].capacity")

        if not world.point_in_bounds(start):
            raise ValueError(f"vehicles[{idx}] start is out of world bounds: {start}")
        if not world.point_is_free(start, clearance=0.0):
            raise ValueError(f"vehicles[{idx}] start is inside obstacle: {start}")

        vehicle = Vehicle(
            id=vid,
            start_pos=start,
            heading=heading,
            speed=speed,
            max_omega=max_omega,
            capacity=capacity,
            remaining_capacity=capacity,
        )
        vehicle.reset_runtime_state()
        vehicles.append(vehicle)

    return vehicles


def _parse_tasks(raw: list[Any], world: WorldMap) -> list[Task]:
    tasks: list[Task] = []
    seen_ids: set[int] = set()

    for idx, item in enumerate(raw):
        row = _as_mapping(item, label=f"tasks[{idx}]")
        tid = _read_int(row, "id", label=f"tasks[{idx}].id")
        if tid in seen_ids:
            raise ValueError(f"duplicate task id: {tid}")
        seen_ids.add(tid)

        pos_raw = row.get("position")
        if pos_raw is None:
            pos_raw = row.get("pos")
        if pos_raw is None:
            raise ValueError(f"tasks[{idx}] missing position/pos")
        position = _read_point(pos_raw, label=f"tasks[{idx}].position")
        demand = _read_positive_int(row, "demand", label=f"tasks[{idx}].demand")

        if not world.point_in_bounds(position):
            raise ValueError(f"tasks[{idx}] position is out of world bounds: {position}")
        if not world.point_is_free(position, clearance=0.0):
            raise ValueError(f"tasks[{idx}] position is inside obstacle: {position}")
        if world.depot_polygon.contains(Point(position)):
            raise ValueError(f"tasks[{idx}] position is inside depot: {position}")

        tasks.append(Task(id=tid, position=position, demand=demand))

    return tasks


def _read_polygon_points(raw: dict[str, Any], key: str, label: str) -> list[tuple[float, float]]:
    if key not in raw:
        raise ValueError(f"missing {label}")
    return _read_polygon_points_from_value(raw[key], label=label)


def _read_polygon_points_from_value(raw: Any, label: str) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list of points")
    points = [_read_point(item, label=f"{label}[{idx}]") for idx, item in enumerate(raw)]
    if len(points) < 3:
        raise ValueError(f"{label} must contain at least 3 points")
    return points


def _build_polygon(points: list[tuple[float, float]], label: str) -> Polygon:
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 1e-9:
        raise ValueError(f"{label} is not a valid polygon")
    return polygon


def _read_point(raw: Any, label: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{label} must be [x, y]")
    x = float(raw[0])
    y = float(raw[1])
    return x, y


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _as_mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _read_float(raw: dict[str, Any], key: str, label: str) -> float:
    if key not in raw:
        raise ValueError(f"missing {label}")
    return float(raw[key])


def _read_positive_float(raw: dict[str, Any], key: str, label: str | None = None) -> float:
    field_label = label or key
    value = _read_float(raw, key, field_label)
    if value <= 0.0:
        raise ValueError(f"{field_label} must be > 0")
    return value


def _read_int(raw: dict[str, Any], key: str, label: str) -> int:
    if key not in raw:
        raise ValueError(f"missing {label}")
    return int(raw[key])


def _read_positive_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _read_int(raw, key, label)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value
