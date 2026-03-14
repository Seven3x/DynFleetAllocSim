from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from shapely.geometry import Point

from .config import SimulationConfig
from .entities import Task
from .map_utils import WorldMap


@dataclass
class DynamicEvent:
    step: int
    event_type: str  # add_task / cancel_task
    task: Optional[Task] = None
    task_id: Optional[int] = None


def _sample_new_task_point(world: WorldMap, cfg: SimulationConfig, rng: np.random.Generator) -> tuple[float, float]:
    for _ in range(5000):
        p = (float(rng.uniform(0.0, cfg.map_width)), float(rng.uniform(0.0, cfg.map_height)))
        if Point(p).within(world.depot_polygon):
            continue
        if world.point_is_free(p, clearance=cfg.vehicle_radius + 0.2):
            return p
    raise RuntimeError("Failed to sample position for dynamic task.")


def generate_new_task(
    world: WorldMap,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    task_id: int,
) -> Task:
    pos = _sample_new_task_point(world=world, cfg=cfg, rng=rng)
    demand = int(rng.integers(cfg.task_demand_min, cfg.task_demand_max + 1))
    return Task(id=task_id, position=pos, demand=demand, status="unassigned")


def build_dynamic_events(
    tasks: List[Task],
    cfg: SimulationConfig,
    world: WorldMap,
    rng: np.random.Generator,
) -> List[DynamicEvent]:
    events: List[DynamicEvent] = []

    next_id = (max(t.id for t in tasks) + 1) if tasks else 0
    for i in range(cfg.dynamic_new_tasks):
        task = generate_new_task(world=world, cfg=cfg, rng=rng, task_id=next_id)
        next_id += 1
        events.append(DynamicEvent(step=i + 1, event_type="add_task", task=task))

    cancel_candidates = [t.id for t in tasks if t.status == "locked"]
    rng.shuffle(cancel_candidates)
    for i, tid in enumerate(cancel_candidates[: cfg.dynamic_cancel_tasks]):
        events.append(DynamicEvent(step=cfg.dynamic_new_tasks + i + 1, event_type="cancel_task", task_id=tid))

    events.sort(key=lambda e: e.step)
    return events
