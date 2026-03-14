from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Point, Polygon

from .auction_core import AllocationEngine, AllocationResult
from .config import DEFAULT_CONFIG, SimulationConfig
from .dynamic_events import generate_new_task
from .entities import Task
from .planner_astar import AStarPlanner
from .simulator import SimulationArtifacts, build_static_scenario
from .visualization import draw_final_scene_on_axis, plot_final_scene


@dataclass
class SessionSnapshot:
    step: int
    total_tasks: int
    status_counts: dict[str, int]
    system_total_time: float


class SimulationSession:
    def __init__(self, cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self.output_dir = Path("outputs")
        self.max_undo_steps = 30

        self.artifacts: SimulationArtifacts | None = None
        self.engine: AllocationEngine | None = None
        self._undo_stack: list[tuple[SimulationArtifacts, AllocationEngine, int, dict]] = []

        self.rng = np.random.default_rng(cfg.seed + 2026)
        self.step = 10_000
        self.reset()

    def reset(self) -> None:
        self.artifacts = build_static_scenario(self.cfg)
        self.engine = AllocationEngine(
            vehicles=self.artifacts.vehicles,
            tasks=self.artifacts.tasks,
            world=self.artifacts.world,
            cfg=self.cfg,
            planner=self.artifacts.planner,
        )
        self.engine.reset()
        self.engine.allocate_until_stable("initial")
        self.step = 10_000
        self._undo_stack.clear()

    def _assert_ready(self) -> None:
        if self.engine is None or self.artifacts is None:
            raise RuntimeError("Session is not initialized.")

    def _next_task_id(self) -> int:
        self._assert_ready()
        assert self.engine is not None
        if not self.engine.tasks:
            return 0
        return max(t.id for t in self.engine.tasks) + 1

    def _push_undo_state(self) -> None:
        self._assert_ready()
        assert self.artifacts is not None
        assert self.engine is not None

        snapshot = (
            copy.deepcopy(self.artifacts),
            copy.deepcopy(self.engine),
            int(self.step),
            copy.deepcopy(self.rng.bit_generator.state),
        )
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self.max_undo_steps:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> None:
        self._assert_ready()
        if not self._undo_stack:
            raise ValueError("nothing to undo")

        artifacts, engine, step, rng_state = self._undo_stack.pop()
        self.artifacts = artifacts
        self.engine = engine
        self.step = step
        self.rng.bit_generator.state = rng_state

    def _rebuild_planner(self) -> None:
        self._assert_ready()
        assert self.artifacts is not None
        assert self.engine is not None

        planner = AStarPlanner(
            world=self.artifacts.world,
            resolution=self.cfg.astar_resolution,
            inflation_radius=self.cfg.vehicle_radius + self.cfg.safety_margin,
            connect_diagonal=self.cfg.astar_connect_diagonal,
        )
        self.artifacts.planner = planner
        self.engine.planner = planner

    def add_obstacle_polygon(self, points: list[tuple[float, float]]) -> Polygon:
        self._assert_ready()
        assert self.artifacts is not None
        assert self.engine is not None

        if len(points) < 3:
            raise ValueError("obstacle polygon requires at least 3 points")

        poly = Polygon(points).buffer(0.0)
        if poly.is_empty or not poly.is_valid:
            raise ValueError("invalid polygon shape (self-intersection or degenerate geometry)")
        if poly.area < 1.0:
            raise ValueError("polygon area is too small")
        if not poly.within(self.artifacts.world.boundary_polygon):
            raise ValueError("polygon must be fully inside map boundary")
        if poly.intersects(self.artifacts.world.depot_polygon):
            raise ValueError("polygon intersects depot area")
        if any(poly.intersects(obs) for obs in self.artifacts.world.obstacles):
            raise ValueError("polygon intersects existing obstacle")

        inflated = poly.buffer(self.cfg.vehicle_radius + self.cfg.safety_margin)

        for v in self.engine.vehicles:
            if Point(v.start_pos).within(inflated) or Point(v.current_pos).within(inflated):
                raise ValueError(f"polygon blocks vehicle V{v.id} start/current position")

        for t in self.engine.tasks:
            if t.status == "canceled":
                continue
            if Point(t.position).within(inflated):
                raise ValueError(f"polygon covers task T{t.id}; move obstacle or cancel task first")

        prev_obstacle_count = len(self.artifacts.world.obstacles)
        prev_planner = self.engine.planner
        self._push_undo_state()

        self.artifacts.world.add_obstacle(poly)
        self._rebuild_planner()
        try:
            # Validate that current fixed task sequences remain route-feasible.
            self.engine.finalize()
        except Exception as exc:
            self.artifacts.world.obstacles = self.artifacts.world.obstacles[:prev_obstacle_count]
            self.artifacts.world.invalidate_cache()
            self.engine.planner = prev_planner
            self.artifacts.planner = prev_planner
            if self._undo_stack:
                self._undo_stack.pop()
            raise ValueError(f"obstacle makes current routes infeasible: {exc}") from exc

        return poly

    def add_task(self, x: float, y: float, demand: int, task_id: int | None = None) -> Task:
        self._assert_ready()
        assert self.engine is not None
        assert self.artifacts is not None

        p = (x, y)
        if not self.artifacts.world.point_in_bounds(p, margin=0.0):
            raise ValueError("point is out of map bounds")
        if Point(p).within(self.artifacts.world.depot_polygon):
            raise ValueError("point is inside depot")
        if not self.artifacts.world.point_is_free(p, clearance=self.cfg.vehicle_radius + 0.2):
            raise ValueError("point is in obstacle/safety area")
        if demand <= 0:
            raise ValueError("demand must be positive")

        tid = self._next_task_id() if task_id is None else int(task_id)
        task = Task(id=tid, position=p, demand=int(demand), status="unassigned")

        self._push_undo_state()
        self.step += 1
        self.engine.add_dynamic_task(task=task, step=self.step)
        self.engine.allocate_until_stable(phase=f"session:add@{self.step}")
        return task

    def add_random_task(self, demand: int | None = None) -> Task:
        self._assert_ready()
        assert self.engine is not None
        assert self.artifacts is not None

        task = generate_new_task(
            world=self.artifacts.world,
            cfg=self.cfg,
            rng=self.rng,
            task_id=self._next_task_id(),
        )
        if demand is not None:
            if int(demand) <= 0:
                raise ValueError("demand must be positive")
            task.demand = int(demand)

        self._push_undo_state()
        self.step += 1
        self.engine.add_dynamic_task(task=task, step=self.step)
        self.engine.allocate_until_stable(phase=f"session:add_random@{self.step}")
        return task

    def cancel_task(self, task_id: int) -> None:
        self._assert_ready()
        assert self.engine is not None

        self._push_undo_state()
        self.step += 1
        self.engine.cancel_task(task_id=int(task_id), step=self.step)
        self.engine.allocate_until_stable(phase=f"session:cancel@{self.step}")

    def result(self) -> AllocationResult:
        self._assert_ready()
        assert self.engine is not None
        return self.engine.finalize()

    def status_snapshot(self) -> SessionSnapshot:
        result = self.result()
        return SessionSnapshot(
            step=self.step,
            total_tasks=len(result.tasks),
            status_counts=self.status_counts(),
            system_total_time=result.system_total_time,
        )

    def status_counts(self) -> dict[str, int]:
        self._assert_ready()
        assert self.engine is not None

        out: dict[str, int] = {}
        for t in self.engine.tasks:
            out[t.status] = out.get(t.status, 0) + 1
        return out

    def list_tasks(self, status_filter: str | None = None) -> list[Task]:
        self._assert_ready()
        assert self.engine is not None

        tasks = self.engine.tasks
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return sorted(tasks, key=lambda t: t.id)

    def recent_logs(self, n: int = 5):
        self._assert_ready()
        assert self.engine is not None

        n = max(0, int(n))
        return self.engine.verification_logs[-n:], self.engine.coordination_logs[-n:]

    def format_status_text(self) -> str:
        result = self.result()
        lines = [
            "-" * 72,
            (
                f"step={self.step} total_tasks={len(result.tasks)} "
                f"status={self.status_counts()}"
            ),
        ]
        for v in result.vehicles:
            seq = ",".join(f"T{tid}" for tid in v.task_sequence) if v.task_sequence else "(none)"
            v_time = v.route_length / v.speed
            lines.append(
                f"V{v.id}: remain={v.remaining_capacity}/{v.capacity} "
                f"tasks=[{seq}] time={v_time:.3f}"
            )
        lines.append(f"system_total_time={result.system_total_time:.3f}")
        lines.append("-" * 72)
        return "\n".join(lines)

    def format_logs_text(self, n: int = 8) -> str:
        ver, coord = self.recent_logs(n=n)
        lines = ["Verification logs:"]
        for item in ver:
            lines.append(
                f"  round={item.round_idx} T{item.task_id} V{item.vehicle_id} "
                f"passed={item.passed} e_under={item.e_under:.3f} forced={item.forced_accept}"
            )

        lines.append("Coordination logs:")
        for item in coord:
            lines.append(
                f"  task={item.task_id} event={item.event} rounds={item.rounds} "
                f"converged={item.converged} final=({item.final_status}, V{item.final_winner})"
            )
        return "\n".join(lines)

    def format_tasks_text(self, status_filter: Optional[str] = None, limit: int = 60) -> str:
        tasks = self.list_tasks(status_filter=status_filter)
        lines = []
        for t in tasks[:limit]:
            lines.append(
                f"T{t.id}: pos=({t.position[0]:.2f},{t.position[1]:.2f}) "
                f"d={t.demand} s={t.status} w={t.assigned_vehicle}"
            )
        if len(tasks) > limit:
            lines.append(f"... ({len(tasks) - limit} more tasks)")
        return "\n".join(lines)

    def draw_on_axis(self, ax) -> None:
        self._assert_ready()
        assert self.artifacts is not None

        result = self.result()
        draw_final_scene_on_axis(
            ax=ax,
            world=self.artifacts.world,
            vehicles=result.vehicles,
            tasks=result.tasks,
            title="Realtime Allocation View",
            show_task_meta=False,
            show_vehicle_sequences=False,
            task_font_size=8,
            vehicle_font_size=9,
            label_box=True,
            curve_width=1.6,
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_snapshot(self, filename: str | None = None) -> Path:
        self._assert_ready()
        assert self.artifacts is not None

        result = self.result()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            path = self.output_dir / filename
        else:
            path = self.output_dir / f"snapshot_{self._timestamp()}_step_{self.step}.png"

        plot_final_scene(
            world=self.artifacts.world,
            vehicles=result.vehicles,
            tasks=result.tasks,
            save_path=path,
            dpi=self.cfg.figure_dpi,
            fig_size=self.cfg.figure_size,
        )
        return path

    def export_logs(self, prefix: str | None = None) -> tuple[Path, Path]:
        self._assert_ready()
        assert self.engine is not None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"session_{self._timestamp()}_step_{self.step}"

        coord_path = self.output_dir / f"{prefix}_coordination_log.txt"
        verify_path = self.output_dir / f"{prefix}_verification_log.txt"

        with coord_path.open("w", encoding="utf-8") as f:
            f.write("task_id,event,rounds,converged,final_winner,final_status,trace\n")
            for item in self.engine.coordination_logs:
                trace = ";".join(
                    f"step{t.step}:d{t.distinct_records}:s{t.stable_count}" for t in item.traces
                )
                f.write(
                    f"{item.task_id},{item.event},{item.rounds},{item.converged},"
                    f"{item.final_winner},{item.final_status},{trace}\n"
                )

        with verify_path.open("w", encoding="utf-8") as f:
            f.write("round,task_id,vehicle_id,c_hat,c_tilde,e_under,passed,forced_accept\n")
            for item in self.engine.verification_logs:
                f.write(
                    f"{item.round_idx},{item.task_id},{item.vehicle_id},"
                    f"{item.c_hat:.6f},{item.c_tilde:.6f},{item.e_under:.6f},"
                    f"{item.passed},{item.forced_accept}\n"
                )

        return coord_path, verify_path
