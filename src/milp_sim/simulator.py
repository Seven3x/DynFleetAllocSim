from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .auction_core import AllocationResult, run_online_allocation, run_static_auction
from .config import SimulationConfig
from .dynamic_events import DynamicEvent, build_dynamic_events
from .entities import Task, Vehicle
from .map_utils import WorldMap
from .obstacle_generator import generate_tasks, generate_vehicles, generate_world_map
from .planner_astar import AStarPlanner


@dataclass
class SimulationArtifacts:
    cfg: SimulationConfig
    world: WorldMap
    vehicles: list[Vehicle]
    tasks: list[Task]
    planner: AStarPlanner


@dataclass
class SimulationResult:
    artifacts: SimulationArtifacts
    allocation: AllocationResult
    events: list[DynamicEvent] | None = None


def build_static_scenario(cfg: SimulationConfig) -> SimulationArtifacts:
    rng = np.random.default_rng(cfg.seed)

    world = generate_world_map(cfg, rng)
    vehicles = generate_vehicles(cfg, world, rng)
    tasks = generate_tasks(cfg, world, rng)

    planner = AStarPlanner(
        world=world,
        resolution=cfg.astar_resolution,
        inflation_radius=cfg.vehicle_radius + cfg.safety_margin,
        connect_diagonal=cfg.astar_connect_diagonal,
    )

    return SimulationArtifacts(
        cfg=cfg,
        world=world,
        vehicles=vehicles,
        tasks=tasks,
        planner=planner,
    )


def run_static_pipeline(cfg: SimulationConfig) -> SimulationResult:
    artifacts = build_static_scenario(cfg)
    allocation = run_static_auction(
        vehicles=artifacts.vehicles,
        tasks=artifacts.tasks,
        world=artifacts.world,
        cfg=cfg,
        planner=artifacts.planner,
    )

    return SimulationResult(artifacts=artifacts, allocation=allocation)


def run_round2_pipeline(cfg: SimulationConfig) -> SimulationResult:
    artifacts = build_static_scenario(cfg)

    # Use a dedicated RNG stream for event generation to keep reproducibility explicit.
    rng_events = np.random.default_rng(cfg.seed + 1000)

    # Build events from initial task set. Cancellations target currently locked tasks after initial phase.
    # We first run online allocator with no events to materialize initial lock states for event synthesis.
    warmup_allocation = run_online_allocation(
        vehicles=artifacts.vehicles,
        tasks=artifacts.tasks,
        world=artifacts.world,
        cfg=cfg,
        planner=artifacts.planner,
        events=[],
    )

    events = build_dynamic_events(
        tasks=warmup_allocation.tasks,
        cfg=cfg,
        world=artifacts.world,
        rng=rng_events,
    )

    # Rebuild scenario so the formal run starts from a clean initial state.
    artifacts = build_static_scenario(cfg)
    allocation = run_online_allocation(
        vehicles=artifacts.vehicles,
        tasks=artifacts.tasks,
        world=artifacts.world,
        cfg=cfg,
        planner=artifacts.planner,
        events=[
            {"step": e.step, "event_type": e.event_type, "task": e.task, "task_id": e.task_id}
            for e in events
        ],
    )

    return SimulationResult(artifacts=artifacts, allocation=allocation, events=events)
