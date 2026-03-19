from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

from .config import DEFAULT_CONFIG
from .entities import Point2D, Task, Vehicle
from .map_utils import WorldMap
from .obstacle_generator import generate_tasks, generate_vehicles, generate_world_map
from .planner_astar import AStarPlanner


ModeName = Literal["incremental_only", "prefix_aware"]


@dataclass
class ModeResult:
    mode: ModeName
    vehicle_sequences: Dict[int, List[int]]
    task_assignments: Dict[int, int]
    vehicle_route_lengths: Dict[int, float]
    vehicle_execution_times: Dict[int, float]
    system_total_length: float
    system_completion_time: float
    latest_finishing_vehicle_id: int | None
    latest_finishing_length: float
    route_points_by_vehicle: Dict[int, List[Point2D]]


@dataclass
class ComparisonOutputs:
    baseline: ModeResult
    prefix_aware: ModeResult
    image_path: Path
    summary_path: Path
    seed: int | None = None


@dataclass
class _Scenario:
    world: WorldMap
    planner: AStarPlanner
    vehicles: List[Vehicle]
    tasks: List[Task]


@dataclass
class _AuctionState:
    frontier: Point2D
    prefix_length: float
    sequence: List[int]


def _build_fixed_scenario() -> _Scenario:
    world = WorldMap(
        width=120.0,
        height=100.0,
        depot_polygon=box(4.0, 4.0, 24.0, 36.0),
        obstacles=[
            box(30.0, 0.0, 36.0, 58.0),
            box(52.0, 42.0, 58.0, 100.0),
            box(74.0, 0.0, 80.0, 58.0),
            box(92.0, 42.0, 98.0, 100.0),
        ],
    )
    planner = AStarPlanner(
        world=world,
        resolution=1.0,
        inflation_radius=0.0,
        connect_diagonal=True,
    )
    vehicles = [
        Vehicle(
            id=0,
            start_pos=(8.0, 8.0),
            heading=0.0,
            speed=5.0,
            max_omega=1.0,
            capacity=20,
            remaining_capacity=20,
        ),
        Vehicle(
            id=1,
            start_pos=(8.0, 22.0),
            heading=0.0,
            speed=5.0,
            max_omega=1.0,
            capacity=20,
            remaining_capacity=20,
        ),
        Vehicle(
            id=2,
            start_pos=(8.0, 34.0),
            heading=0.0,
            speed=5.0,
            max_omega=1.0,
            capacity=20,
            remaining_capacity=20,
        ),
    ]
    tasks = [
        Task(id=0, position=(20.0, 82.0), demand=1),
        Task(id=1, position=(46.0, 16.0), demand=1),
        Task(id=2, position=(46.0, 80.0), demand=1),
        Task(id=3, position=(68.0, 18.0), demand=1),
        Task(id=4, position=(70.0, 78.0), demand=1),
        Task(id=5, position=(102.0, 18.0), demand=1),
        Task(id=6, position=(104.0, 74.0), demand=1),
        Task(id=7, position=(112.0, 52.0), demand=1),
        Task(id=8, position=(60.0, 60.0), demand=1),
    ]
    return _Scenario(world=world, planner=planner, vehicles=vehicles, tasks=tasks)


def _build_seeded_scenario(seed: int) -> _Scenario:
    cfg = replace(
        DEFAULT_CONFIG,
        seed=seed,
        map_width=120.0,
        map_height=100.0,
        depot_min_x=4.0,
        depot_min_y=4.0,
        depot_max_x=24.0,
        depot_max_y=36.0,
        num_obstacles=6,
        obstacle_vertex_min=4,
        obstacle_vertex_max=7,
        obstacle_radius_min=5.0,
        obstacle_radius_max=9.0,
        obstacle_margin_from_border=8.0,
        obstacle_max_retries=2000,
        num_vehicles=4,
        num_tasks=12,
        vehicle_speed_min=5.0,
        vehicle_speed_max=5.0,
        vehicle_omega_min=1.0,
        vehicle_omega_max=1.0,
        vehicle_capacity_min=30,
        vehicle_capacity_max=30,
        task_demand_min=1,
        task_demand_max=1,
    )

    for attempt in range(30):
        rng = np.random.default_rng(seed + attempt * 9973)
        world = generate_world_map(cfg, rng)
        vehicles = generate_vehicles(cfg, world, rng)
        tasks = generate_tasks(cfg, world, rng)

        for vehicle in vehicles:
            vehicle.heading = 0.0
            vehicle.speed = 5.0
            vehicle.max_omega = 1.0
            vehicle.capacity = 30
            vehicle.reset_runtime_state()
        for task in tasks:
            task.demand = 1

        planner = AStarPlanner(
            world=world,
            resolution=1.0,
            inflation_radius=0.0,
            connect_diagonal=True,
        )

        all_points = [vehicle.start_pos for vehicle in vehicles] + [task.position for task in tasks]
        if _points_fully_connected(planner, all_points):
            return _Scenario(world=world, planner=planner, vehicles=vehicles, tasks=tasks)

    raise RuntimeError(f"failed to build a connected seeded scenario for seed={seed}")


def _points_fully_connected(planner: AStarPlanner, points: List[Point2D]) -> bool:
    cache: Dict[Tuple[Point2D, Point2D], Tuple[List[Point2D], float]] = {}
    for i, src in enumerate(points):
        for dst in points[i + 1 :]:
            try:
                _, length = _plan_leg(planner=planner, cache=cache, src=src, dst=dst)
            except RuntimeError:
                return False
            if length == float("inf"):
                return False
    return True


def _plan_leg(
    planner: AStarPlanner,
    cache: Dict[Tuple[Point2D, Point2D], Tuple[List[Point2D], float]],
    src: Point2D,
    dst: Point2D,
) -> Tuple[List[Point2D], float]:
    key = (src, dst)
    cached = cache.get(key)
    if cached is not None:
        return cached

    path, length = planner.plan(src, dst)
    if not path or length == float("inf"):
        raise RuntimeError(f"A* failed from {src} to {dst}")

    cache[key] = (list(path), float(length))
    return cache[key]


def _bid_cost(mode: ModeName, state: _AuctionState, leg_length: float) -> float:
    if mode == "incremental_only":
        return leg_length
    if mode == "prefix_aware":
        return state.prefix_length + leg_length
    raise ValueError(f"unsupported mode: {mode}")


def _run_mode(scenario: _Scenario, mode: ModeName) -> ModeResult:
    tasks_by_id = {task.id: task for task in scenario.tasks}
    unassigned = {task.id for task in scenario.tasks}
    path_cache: Dict[Tuple[Point2D, Point2D], Tuple[List[Point2D], float]] = {}
    states = {
        vehicle.id: _AuctionState(
            frontier=vehicle.start_pos,
            prefix_length=0.0,
            sequence=[],
        )
        for vehicle in scenario.vehicles
    }
    assignments: Dict[int, int] = {}

    while unassigned:
        round_bids: List[Tuple[int, int, float, float]] = []
        for vehicle in scenario.vehicles:
            state = states[vehicle.id]
            best_bid: Tuple[int, int, float, float] | None = None
            for task_id in sorted(unassigned):
                task = tasks_by_id[task_id]
                _, leg_length = _plan_leg(
                    planner=scenario.planner,
                    cache=path_cache,
                    src=state.frontier,
                    dst=task.position,
                )
                bid_cost = _bid_cost(mode=mode, state=state, leg_length=leg_length)
                candidate = (vehicle.id, task_id, bid_cost, leg_length)
                if best_bid is None or (candidate[2], candidate[1]) < (best_bid[2], best_bid[1]):
                    best_bid = candidate
            if best_bid is not None:
                round_bids.append(best_bid)

        if not round_bids:
            raise RuntimeError(f"no bids available for remaining tasks: {sorted(unassigned)}")

        winners_by_task: Dict[int, Tuple[int, int, float, float]] = {}
        for bid in round_bids:
            vehicle_id, task_id, bid_cost, _ = bid
            current = winners_by_task.get(task_id)
            if current is None or (bid_cost, vehicle_id) < (current[2], current[0]):
                winners_by_task[task_id] = bid

        for vehicle_id, task_id, _, leg_length in sorted(
            winners_by_task.values(),
            key=lambda item: (item[1], item[0]),
        ):
            if task_id not in unassigned:
                continue
            task = tasks_by_id[task_id]
            state = states[vehicle_id]
            state.sequence.append(task_id)
            state.frontier = task.position
            state.prefix_length += leg_length
            assignments[task_id] = vehicle_id
            unassigned.remove(task_id)

    vehicle_sequences = {vehicle.id: list(states[vehicle.id].sequence) for vehicle in scenario.vehicles}
    route_points_by_vehicle: Dict[int, List[Point2D]] = {}
    vehicle_route_lengths: Dict[int, float] = {}
    vehicle_execution_times: Dict[int, float] = {}
    total_length = 0.0

    for vehicle in scenario.vehicles:
        current = vehicle.start_pos
        route_points: List[Point2D] = [current]
        route_length = 0.0
        for task_id in vehicle_sequences[vehicle.id]:
            task = tasks_by_id[task_id]
            leg_points, leg_length = _plan_leg(
                planner=scenario.planner,
                cache=path_cache,
                src=current,
                dst=task.position,
            )
            if len(leg_points) > 1:
                route_points.extend(leg_points[1:])
            route_length += leg_length
            current = task.position
        route_points_by_vehicle[vehicle.id] = route_points
        vehicle_route_lengths[vehicle.id] = route_length
        vehicle_execution_times[vehicle.id] = route_length / max(vehicle.speed, 1e-9)
        total_length += route_length

    latest_finishing_vehicle_id: int | None = None
    latest_finishing_length = 0.0
    system_completion_time = 0.0
    if vehicle_execution_times:
        latest_finishing_vehicle_id = max(
            vehicle_execution_times,
            key=lambda vehicle_id: (vehicle_execution_times[vehicle_id], vehicle_id),
        )
        system_completion_time = vehicle_execution_times[latest_finishing_vehicle_id]
        latest_finishing_length = vehicle_route_lengths[latest_finishing_vehicle_id]

    return ModeResult(
        mode=mode,
        vehicle_sequences=vehicle_sequences,
        task_assignments=assignments,
        vehicle_route_lengths=vehicle_route_lengths,
        vehicle_execution_times=vehicle_execution_times,
        system_total_length=total_length,
        system_completion_time=system_completion_time,
        latest_finishing_vehicle_id=latest_finishing_vehicle_id,
        latest_finishing_length=latest_finishing_length,
        route_points_by_vehicle=route_points_by_vehicle,
    )


def _draw_world(ax, world: WorldMap) -> None:
    bx, by = world.boundary_polygon.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.2)

    dx, dy = world.depot_polygon.exterior.xy
    ax.fill(dx, dy, color="#d9f2d9", alpha=0.85)

    for obstacle in world.obstacles:
        ox, oy = obstacle.exterior.xy
        ax.fill(ox, oy, color="#6b7280", alpha=0.88)

    ax.set_xlim(0.0, world.width)
    ax.set_ylim(0.0, world.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def _plot_mode(ax, scenario: _Scenario, result: ModeResult) -> None:
    _draw_world(ax, scenario.world)
    colors = plt.get_cmap("tab10")

    for vehicle in scenario.vehicles:
        color = colors(vehicle.id % 10)
        ax.scatter(vehicle.start_pos[0], vehicle.start_pos[1], s=90, color=color, edgecolor="black", zorder=4)
        ax.text(vehicle.start_pos[0] + 1.0, vehicle.start_pos[1] + 1.0, f"V{vehicle.id}", fontsize=9, color=color)

        points = result.route_points_by_vehicle[vehicle.id]
        if len(points) >= 2:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95)

        seq = "->".join(f"T{task_id}" for task_id in result.vehicle_sequences[vehicle.id]) or "-"
        end_point = points[-1]
        ax.text(
            end_point[0] + 1.0,
            end_point[1] - 1.0,
            seq,
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.72, edgecolor="none"),
        )

    for task in scenario.tasks:
        winner = result.task_assignments[task.id]
        color = colors(winner % 10)
        ax.scatter(task.position[0], task.position[1], s=52, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            task.position[0] + 0.8,
            task.position[1] + 0.8,
            f"T{task.id}/V{winner}",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.68, edgecolor="none"),
        )

    ax.set_title(
        f"{result.mode}\n"
        f"total_length={result.system_total_length:.3f}, "
        f"completion_time={result.system_completion_time:.3f}"
    )


def _write_comparison_plot(
    scenario: _Scenario,
    baseline: ModeResult,
    prefix_aware: ModeResult,
    image_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    _plot_mode(axes[0], scenario, baseline)
    _plot_mode(axes[1], scenario, prefix_aware)
    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path)
    plt.close(fig)


def _mode_result_to_jsonable(result: ModeResult) -> dict:
    return {
        "mode": result.mode,
        "vehicle_sequences": {str(key): value for key, value in result.vehicle_sequences.items()},
        "task_assignments": {str(key): value for key, value in result.task_assignments.items()},
        "vehicle_route_lengths": {
            str(key): round(value, 6) for key, value in result.vehicle_route_lengths.items()
        },
        "vehicle_execution_times": {
            str(key): round(value, 6) for key, value in result.vehicle_execution_times.items()
        },
        "system_total_length": round(result.system_total_length, 6),
        "system_completion_time": round(result.system_completion_time, 6),
        "latest_finishing_vehicle_id": result.latest_finishing_vehicle_id,
        "latest_finishing_length": round(result.latest_finishing_length, 6),
        "route_points_by_vehicle": {
            str(key): [[float(point[0]), float(point[1])] for point in value]
            for key, value in result.route_points_by_vehicle.items()
        },
    }


def _write_summary(outputs: ComparisonOutputs) -> None:
    payload = {
        "baseline": _mode_result_to_jsonable(outputs.baseline),
        "prefix_aware": _mode_result_to_jsonable(outputs.prefix_aware),
        "image_path": str(outputs.image_path),
        "summary_path": str(outputs.summary_path),
        "seed": outputs.seed,
    }
    outputs.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_summary(outputs: ComparisonOutputs) -> None:
    if outputs.seed is None:
        print("[scenario] fixed_complex")
    else:
        print(f"[scenario] seeded_random seed={outputs.seed}")
    for result in (outputs.baseline, outputs.prefix_aware):
        print(f"[{result.mode}]")
        for vehicle_id in sorted(result.vehicle_sequences):
            sequence = result.vehicle_sequences[vehicle_id]
            route_length = result.vehicle_route_lengths[vehicle_id]
            execution_time = result.vehicle_execution_times[vehicle_id]
            print(
                f"  V{vehicle_id}: tasks={sequence} "
                f"route_length={route_length:.3f} execution_time={execution_time:.3f}"
            )
        print(f"  total_length={result.system_total_length:.3f}")
        print(
            f"  completion_time={result.system_completion_time:.3f} "
            f"(latest=V{result.latest_finishing_vehicle_id}, "
            f"latest_length={result.latest_finishing_length:.3f})"
        )
    print(f"image: {outputs.image_path}")
    print(f"summary: {outputs.summary_path}")


def run_assignment_cost_experiment(
    output_dir: str | Path = "outputs",
    seed: int | None = None,
) -> ComparisonOutputs:
    scenario = _build_fixed_scenario() if seed is None else _build_seeded_scenario(seed)
    baseline = _run_mode(scenario, mode="incremental_only")
    prefix_aware = _run_mode(scenario, mode="prefix_aware")

    output_root = Path(output_dir)
    suffix = "" if seed is None else f"_seed_{seed}"
    image_path = output_root / f"assignment_cost_comparison{suffix}.png"
    summary_path = output_root / f"assignment_cost_comparison_summary{suffix}.json"

    outputs = ComparisonOutputs(
        baseline=baseline,
        prefix_aware=prefix_aware,
        image_path=image_path,
        summary_path=summary_path,
        seed=seed,
    )
    _write_comparison_plot(
        scenario=scenario,
        baseline=baseline,
        prefix_aware=prefix_aware,
        image_path=image_path,
    )
    _write_summary(outputs)
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare auction costs with and without committed prefix path cost.")
    parser.add_argument("--seed", type=int, default=None, help="Generate a seeded random complex scenario.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for PNG and JSON outputs.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    outputs = run_assignment_cost_experiment(output_dir=args.output_dir, seed=args.seed)
    _print_summary(outputs)


if __name__ == "__main__":
    main()
