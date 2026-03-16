from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

from .auction_core import AllocationEngine, AllocationResult, EventLog
from .config import DEFAULT_CONFIG, SimulationConfig
from .cost_estimator import goal_heading_candidates, heading_to_point, wrap_to_pi
from .dubins_path import build_dubins_hybrid_path
from .dynamic_events import generate_new_task
from .entities import Task, Vehicle
from .neighbor_coordination import TaskRecord
from .planner_astar import AStarPlanner
from .simulator import SimulationArtifacts, build_static_scenario
from .visualization import draw_final_scene_on_axis, plot_final_scene


@dataclass
class SessionSnapshot:
    step: int
    total_tasks: int
    status_counts: dict[str, int]
    system_total_time: float


@dataclass
class OnlineEvent:
    event_id: int
    time_s: float
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    result_message: str = ""


@dataclass
class RuntimeSnapshot:
    sim_time: float
    online_running: bool
    dt: float
    replan_period_s: float
    next_replan_time: float
    pending_events: list[OnlineEvent]
    last_replan_reason: str
    last_event_message: str


@dataclass
class OnlineFrameState:
    artifacts: SimulationArtifacts
    engine: AllocationEngine
    sim_time: float
    step: int
    next_periodic_replan: float
    pending_events: list[OnlineEvent]
    event_history: list[OnlineEvent]
    last_replan_reason: str
    last_event_message: str
    rng_state: dict[str, Any]


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

        self.online_enabled = False
        self.online_running = False
        self.sim_time = 0.0
        self.online_dt = cfg.online_dt
        self.replan_period_s = cfg.online_replan_period_s
        self.next_periodic_replan = cfg.online_replan_period_s
        self._event_id_counter = 0
        self.pending_events: list[OnlineEvent] = []
        self.event_history: list[OnlineEvent] = []
        self.last_replan_reason = ""
        self.last_event_message = ""
        self._frame_history: list[OnlineFrameState] = []
        self._frame_cursor: int = -1
        self.max_frame_history = 800

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

        self.online_enabled = False
        self.online_running = False
        self.sim_time = 0.0
        self.online_dt = self.cfg.online_dt
        self.replan_period_s = self.cfg.online_replan_period_s
        self.next_periodic_replan = self.replan_period_s
        self.pending_events.clear()
        self.event_history.clear()
        self.last_replan_reason = ""
        self.last_event_message = ""
        self._frame_history.clear()
        self._frame_cursor = -1

    def start_online(self, dt: float | None = None, replan_period_s: float | None = None) -> RuntimeSnapshot:
        self._assert_ready()
        assert self.engine is not None

        if dt is not None:
            if float(dt) <= 0.0:
                raise ValueError("dt must be positive")
            self.online_dt = float(dt)
        if replan_period_s is not None:
            if float(replan_period_s) <= 0.0:
                raise ValueError("replan_period_s must be positive")
            self.replan_period_s = float(replan_period_s)

        if not self.online_enabled:
            self._initialize_online_runtime()

        self.online_enabled = True
        self.online_running = True
        return self.runtime_snapshot()

    def pause_online(self) -> None:
        self.online_running = False

    def resume_online(self) -> None:
        if not self.online_enabled:
            self.start_online()
            return
        self.online_running = True

    def runtime_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            sim_time=self.sim_time,
            online_running=self.online_running,
            dt=self.online_dt,
            replan_period_s=self.replan_period_s,
            next_replan_time=self.next_periodic_replan,
            pending_events=list(self.pending_events),
            last_replan_reason=self.last_replan_reason,
            last_event_message=self.last_event_message,
        )

    def frame_prev(self) -> RuntimeSnapshot:
        self._assert_ready()
        if not self.online_enabled:
            self.start_online(dt=self.online_dt, replan_period_s=self.replan_period_s)
        if self._frame_cursor <= 0:
            return self.runtime_snapshot()
        self._frame_cursor -= 1
        self._restore_frame(self._frame_history[self._frame_cursor])
        return self.runtime_snapshot()

    def frame_next(self) -> RuntimeSnapshot:
        self._assert_ready()
        if not self.online_enabled:
            self.start_online(dt=self.online_dt, replan_period_s=self.replan_period_s)
        if self._frame_cursor < len(self._frame_history) - 1:
            self._frame_cursor += 1
            self._restore_frame(self._frame_history[self._frame_cursor])
            return self.runtime_snapshot()
        return self.tick(n=1)

    def schedule_event(self, at_time: float, event_type: str, payload: Optional[dict[str, Any]] = None) -> OnlineEvent:
        at = float(at_time)
        if at < 0.0:
            raise ValueError("event time must be >= 0")
        payload = {} if payload is None else dict(payload)
        self._event_id_counter += 1
        evt = OnlineEvent(
            event_id=self._event_id_counter,
            time_s=at,
            event_type=str(event_type),
            payload=payload,
        )
        self.pending_events.append(evt)
        self.pending_events.sort(key=lambda e: (e.time_s, e.event_id))
        return evt

    def tick(self, n: int = 1) -> RuntimeSnapshot:
        self._assert_ready()
        if n < 0:
            raise ValueError("n must be >= 0")
        if not self.online_enabled:
            self.start_online(dt=self.online_dt, replan_period_s=self.replan_period_s)

        if n == 0:
            self._process_due_events_and_replan()
            self._capture_frame()
            return self.runtime_snapshot()

        for _ in range(n):
            self._process_due_events_and_replan()
            self._advance_vehicles(self.online_dt)
            self.sim_time += self.online_dt
            self.step += 1
            self._capture_frame()

        return self.runtime_snapshot()

    def _initialize_online_runtime(self) -> None:
        self._assert_ready()
        assert self.engine is not None

        for v in self.engine.vehicles:
            v.current_pos = v.start_pos
            v.current_heading = v.heading
            v.active_task_id = None
            v.path_cursor = 0
            v.distance_to_next_waypoint = 0.0
            v.is_moving = False
            v.last_replan_time = self.sim_time
            v.route_points = [v.current_pos]
            v.history_points = [v.current_pos]
            v.route_length = 0.0

        for t in self.engine.tasks:
            if t.status in {"tentative", "verifying"}:
                t.status = "withdrawn"
                t.assigned_vehicle = None
            if t.status == "completed":
                t.status = "locked"

        self.sim_time = 0.0
        self.next_periodic_replan = self.replan_period_s
        self.last_replan_reason = "startup"
        self._replan_online_routes(reason="startup")
        self._frame_history.clear()
        self._frame_cursor = -1
        self._capture_frame()

    def _snapshot_frame(self) -> OnlineFrameState:
        self._assert_ready()
        assert self.artifacts is not None
        assert self.engine is not None
        return OnlineFrameState(
            artifacts=copy.deepcopy(self.artifacts),
            engine=copy.deepcopy(self.engine),
            sim_time=float(self.sim_time),
            step=int(self.step),
            next_periodic_replan=float(self.next_periodic_replan),
            pending_events=copy.deepcopy(self.pending_events),
            event_history=copy.deepcopy(self.event_history),
            last_replan_reason=str(self.last_replan_reason),
            last_event_message=str(self.last_event_message),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def _restore_frame(self, frame: OnlineFrameState) -> None:
        self.artifacts = copy.deepcopy(frame.artifacts)
        self.engine = copy.deepcopy(frame.engine)
        self.sim_time = float(frame.sim_time)
        self.step = int(frame.step)
        self.next_periodic_replan = float(frame.next_periodic_replan)
        self.pending_events = copy.deepcopy(frame.pending_events)
        self.event_history = copy.deepcopy(frame.event_history)
        self.last_replan_reason = str(frame.last_replan_reason)
        self.last_event_message = str(frame.last_event_message)
        self.rng.bit_generator.state = copy.deepcopy(frame.rng_state)

    def _capture_frame(self) -> None:
        if not self.online_enabled:
            return
        frame = self._snapshot_frame()
        if self._frame_cursor < len(self._frame_history) - 1:
            self._frame_history = self._frame_history[: self._frame_cursor + 1]
        self._frame_history.append(frame)
        if len(self._frame_history) > self.max_frame_history:
            overflow = len(self._frame_history) - self.max_frame_history
            self._frame_history = self._frame_history[overflow:]
            self._frame_cursor = max(-1, self._frame_cursor - overflow)
        self._frame_cursor = len(self._frame_history) - 1

    def _process_due_events_and_replan(self) -> None:
        self._assert_ready()
        assert self.engine is not None

        due_events = []
        while self.pending_events and self.pending_events[0].time_s <= self.sim_time + 1e-9:
            due_events.append(self.pending_events.pop(0))

        event_trigger = False
        route_refresh_trigger = False
        for evt in due_events:
            try:
                msg, should_reauction, should_refresh_routes = self._apply_online_event(evt)
                evt.result_message = msg
                evt.applied = True
                event_trigger = event_trigger or should_reauction
                route_refresh_trigger = route_refresh_trigger or should_refresh_routes
                self.last_event_message = evt.result_message
            except Exception as exc:
                evt.result_message = f"failed: {exc}"
                evt.applied = False
                self.last_event_message = evt.result_message
            self.event_history.append(evt)

        periodic_trigger = self.sim_time + 1e-9 >= self.next_periodic_replan

        if event_trigger or periodic_trigger:
            did_preempt = self._try_soft_preempt()
            tags = []
            if event_trigger:
                tags.append("event")
            if periodic_trigger:
                tags.append("periodic")
            if did_preempt:
                tags.append("preempt")
            reason = "+".join(tags) if tags else "periodic"
            self._replan_online_routes(reason=reason)

            if periodic_trigger:
                self.next_periodic_replan = self.sim_time + self.replan_period_s
        elif route_refresh_trigger:
            self._refresh_active_tasks_and_routes()
            self.last_replan_reason = "event:route_refresh"
            for v in self.engine.vehicles:
                v.last_replan_time = self.sim_time

    def _apply_online_event(self, evt: OnlineEvent) -> tuple[str, bool, bool]:
        self._assert_ready()
        assert self.engine is not None
        assert self.artifacts is not None

        if evt.event_type == "add_task":
            task = evt.payload.get("task")
            if not isinstance(task, Task):
                raise ValueError("add_task event requires payload['task']=Task")
            self.engine.add_dynamic_task(task=task, step=self.step)
            return f"event add_task applied: T{task.id}", True, False

        if evt.event_type == "cancel_task":
            tid = int(evt.payload.get("task_id"))
            pose_cache = {v.id: (v.current_pos, v.current_heading) for v in self.engine.vehicles}
            self.engine.cancel_task(task_id=tid, step=self.step)
            for v in self.engine.vehicles:
                if v.id in pose_cache:
                    pos, heading = pose_cache[v.id]
                    v.current_pos = pos
                    v.current_heading = heading
            return f"event cancel_task applied: T{tid}", True, False

        if evt.event_type == "add_obstacle":
            points = evt.payload.get("points")
            if not isinstance(points, list):
                raise ValueError("add_obstacle event requires payload['points']=list")
            poly = self._apply_obstacle_polygon_now(points)
            return f"event add_obstacle applied: area={poly.area:.2f}", False, True

        if evt.event_type == "remove_obstacle":
            idx = int(evt.payload.get("obstacle_idx"))
            poly = self.artifacts.world.remove_obstacle(idx)
            self._rebuild_planner()
            return f"event remove_obstacle applied: idx={idx} area={poly.area:.2f}", False, True

        raise ValueError(f"unsupported event_type: {evt.event_type}")

    def _try_soft_preempt(self) -> bool:
        keep = self._estimate_remaining_system_time(preempt=False)
        switch = self._estimate_remaining_system_time(preempt=True)
        if not self.soft_preempt_passes(keep, switch, self.cfg.preempt_gain_threshold):
            return False

        gain = (keep - switch) / keep
        changed = self._apply_preempt_release()
        if changed:
            self.last_event_message = (
                f"soft preempt accepted: gain={gain:.3f} threshold={self.cfg.preempt_gain_threshold:.3f}"
            )
        return changed

    @staticmethod
    def soft_preempt_passes(keep: float, switch: float, threshold: float) -> bool:
        if not math.isfinite(keep) or keep <= 1e-9:
            return False
        if not math.isfinite(switch):
            return False
        gain = (keep - switch) / keep
        return gain + 1e-12 >= threshold

    def _estimate_remaining_system_time(self, preempt: bool) -> float:
        self._assert_ready()
        assert self.engine is not None

        total = 0.0
        pending_ids: set[int] = set()

        for task in self.engine.tasks:
            if task.status in {"unassigned", "withdrawn"}:
                pending_ids.add(task.id)

        for v in self.engine.vehicles:
            pos = v.current_pos
            seq = list(v.task_sequence)
            if preempt and v.active_task_id is not None:
                if seq and seq[0] == v.active_task_id:
                    seq = seq[1:]
                pending_ids.add(v.active_task_id)

            for tid in seq:
                task = self.engine.tasks_by_id.get(tid)
                if task is None:
                    continue
                if task.status in {"canceled", "completed"}:
                    continue
                seg = math.hypot(task.position[0] - pos[0], task.position[1] - pos[1])
                total += seg / max(v.speed, 1e-9)
                pos = task.position

        if pending_ids:
            for tid in pending_ids:
                task = self.engine.tasks_by_id.get(tid)
                if task is None:
                    continue
                best = float("inf")
                for v in self.engine.vehicles:
                    seg = math.hypot(task.position[0] - v.current_pos[0], task.position[1] - v.current_pos[1])
                    best = min(best, seg / max(v.speed, 1e-9))
                if math.isfinite(best):
                    total += best

        return total

    def _apply_preempt_release(self) -> bool:
        self._assert_ready()
        assert self.engine is not None

        changed = False
        for v in self.engine.vehicles:
            tid = v.active_task_id
            if tid is None:
                continue
            task = self.engine.tasks_by_id.get(tid)
            if task is None:
                continue

            if tid in v.task_sequence:
                v.task_sequence = [x for x in v.task_sequence if x != tid]
                v.remaining_capacity = min(v.capacity, v.remaining_capacity + task.demand)

            task.status = "withdrawn"
            task.assigned_vehicle = None

            base = self.engine.records.get(tid)
            next_version = (base.version + 1) if base is not None else 1
            self.engine.records[tid] = TaskRecord(
                task_id=tid,
                winner=None,
                bid=float("inf"),
                status="withdrawn",
                version=next_version,
            )

            v.active_task_id = None
            v.path_cursor = 0
            v.distance_to_next_waypoint = 0.0
            v.is_moving = False
            v.route_points = [v.current_pos]
            v.route_length = 0.0

            self.engine.event_logs.append(
                EventLog(
                    step=self.step,
                    event_type="soft_preempt",
                    task_id=tid,
                    message=f"soft preempt released in-progress task T{tid} from V{v.id}",
                )
            )
            changed = True

        return changed

    def _replan_online_routes(self, reason: str) -> None:
        self._assert_ready()
        assert self.engine is not None

        # Use runtime poses as bidding inputs, but keep them intact after allocation.
        pose_cache = {v.id: (v.current_pos, v.current_heading, v.active_task_id) for v in self.engine.vehicles}

        self.engine.allocate_until_stable(phase=f"online:{reason}@{self.step}")

        for v in self.engine.vehicles:
            pos, heading, active_tid = pose_cache[v.id]
            v.current_pos = pos
            v.current_heading = heading
            if active_tid is not None and active_tid not in v.task_sequence:
                v.active_task_id = None
            elif active_tid is not None:
                v.active_task_id = active_tid

        self._refresh_active_tasks_and_routes()
        self.last_replan_reason = reason
        for v in self.engine.vehicles:
            v.last_replan_time = self.sim_time

    def _refresh_active_tasks_and_routes(self) -> None:
        self._assert_ready()
        assert self.engine is not None

        for v in self.engine.vehicles:
            filtered: list[int] = []
            for tid in v.task_sequence:
                task = self.engine.tasks_by_id.get(tid)
                if task is None:
                    continue
                if task.status in {"canceled", "completed"}:
                    continue
                if task.status not in {"locked", "in_progress"}:
                    continue
                filtered.append(tid)
            v.task_sequence = filtered

            if v.active_task_id not in v.task_sequence:
                v.active_task_id = None

            if v.active_task_id is None and v.task_sequence:
                v.active_task_id = v.task_sequence[0]

            if v.active_task_id is not None and v.task_sequence and v.task_sequence[0] != v.active_task_id:
                v.task_sequence = [v.active_task_id] + [x for x in v.task_sequence if x != v.active_task_id]

            # Keep only short online horizon: active task + limited future locked tasks.
            future_horizon = max(0, int(getattr(self.cfg, "online_future_task_horizon", 2)))
            keep_limit = future_horizon + (1 if v.active_task_id is not None else 0)
            if len(v.task_sequence) > keep_limit:
                overflow = v.task_sequence[keep_limit:]
                v.task_sequence = v.task_sequence[:keep_limit]
                for tid in overflow:
                    task = self.engine.tasks_by_id.get(tid)
                    if task is None:
                        continue
                    if task.status in {"canceled", "completed"}:
                        continue
                    if task.assigned_vehicle == v.id:
                        task.assigned_vehicle = None
                    task.status = "withdrawn"
                    base = self.engine.records.get(tid)
                    next_version = (base.version + 1) if base is not None else 1
                    self.engine.records[tid] = TaskRecord(
                        task_id=tid,
                        winner=None,
                        bid=float("inf"),
                        status="withdrawn",
                        version=next_version,
                    )
                    v.remaining_capacity = min(v.capacity, v.remaining_capacity + task.demand)

            for tid in v.task_sequence:
                task = self.engine.tasks_by_id[tid]
                if tid == v.active_task_id:
                    task.status = "in_progress"
                    task.assigned_vehicle = v.id
                elif task.status == "in_progress":
                    task.status = "locked"

            if v.active_task_id is None:
                v.path_cursor = 0
                v.distance_to_next_waypoint = 0.0
                v.is_moving = False
                v.route_points = [v.current_pos]
                v.route_length = 0.0
            else:
                self._build_active_segment(v)

    def _build_active_segment(self, v: Vehicle) -> None:
        self._assert_ready()
        assert self.engine is not None
        assert self.artifacts is not None

        tid = v.active_task_id
        if tid is None:
            v.route_points = [v.current_pos]
            v.route_length = 0.0
            v.path_cursor = 0
            v.distance_to_next_waypoint = 0.0
            v.is_moving = False
            return

        task = self.engine.tasks_by_id.get(tid)
        if task is None:
            v.active_task_id = None
            v.route_points = [v.current_pos]
            v.route_length = 0.0
            v.path_cursor = 0
            v.distance_to_next_waypoint = 0.0
            v.is_moving = False
            return

        turn_radius = v.speed / max(v.max_omega, 1e-6)
        next_task_pos: tuple[float, float] | None = None
        if len(v.task_sequence) >= 2 and v.task_sequence[0] == tid:
            nxt = self.engine.tasks_by_id.get(v.task_sequence[1])
            if nxt is not None:
                next_task_pos = nxt.position
        headings = goal_heading_candidates(
            current_pos=v.current_pos,
            task_pos=task.position,
            next_task_pos=next_task_pos,
            turn_radius=turn_radius,
            blend_turn_radius_factor=self.cfg.goal_heading_blend_turn_radius_factor,
            tolerance_rad=self.cfg.goal_heading_tolerance_rad,
            num_samples=self.cfg.goal_heading_num_samples,
        )
        best_path: list[tuple[float, float]] = []
        best_length = float("inf")
        best_score = float("inf")
        best_goal_heading = heading_to_point(v.current_pos, task.position)
        cand_meta: list[tuple[float, float, bool, float, float]] = []
        turn_penalty = max(0.0, float(getattr(self.cfg, "goal_heading_turn_penalty", 0.0)))
        dpsi_limit = max(0.0, float(getattr(self.cfg, "goal_heading_max_dpsi_rad", 0.0)))
        dpsi_slack = max(0.0, float(getattr(self.cfg, "goal_heading_max_dpsi_slack_rad", 0.0)))
        dpsi_limit_eff = dpsi_limit + dpsi_slack
        fallback_path: list[tuple[float, float]] = []
        fallback_length = float("inf")
        fallback_score = float("inf")
        fallback_goal_heading = best_goal_heading
        for goal_heading in headings:
            cand_path, cand_length, _ = build_dubins_hybrid_path(
                world=self.artifacts.world,
                cfg=self.cfg,
                start_pose=(v.current_pos[0], v.current_pos[1], v.current_heading),
                goal_pose=(task.position[0], task.position[1], goal_heading),
                astar_planner=self.artifacts.planner,
                turn_radius=turn_radius,
            )
            ok = bool(cand_path) and cand_length != float("inf")
            dpsi = abs(wrap_to_pi(goal_heading - v.current_heading))
            score = float("inf")
            if ok:
                score = cand_length + turn_penalty * turn_radius * dpsi
            cand_meta.append((goal_heading, cand_length, ok, score, dpsi))
            if ok and score < fallback_score:
                fallback_path = cand_path
                fallback_length = cand_length
                fallback_score = score
                fallback_goal_heading = goal_heading
            if ok and dpsi <= dpsi_limit_eff + 1e-9 and score < best_score:
                best_path = cand_path
                best_length = cand_length
                best_score = score
                best_goal_heading = goal_heading

        if not best_path and fallback_path:
            best_path = fallback_path
            best_length = fallback_length
            best_score = fallback_score
            best_goal_heading = fallback_goal_heading

        path, length = best_path, best_length

        if bool(getattr(self.cfg, "plan_debug_enabled", False)):
            target_vid = int(getattr(self.cfg, "plan_debug_vehicle_id", -1))
            target_tid = int(getattr(self.cfg, "plan_debug_task_id", -1))
            if (target_vid < 0 or target_vid == v.id) and (target_tid < 0 or target_tid == tid):
                top_k = max(1, int(getattr(self.cfg, "plan_debug_top_k", 5)))
                ranked = sorted(
                    cand_meta,
                    key=lambda x: (0 if x[2] else 1, x[3] if x[2] else float("inf")),
                )[:top_k]
                parts: list[str] = []
                for h, l, ok, score, dpsi in ranked:
                    dpsi_deg = math.degrees(dpsi)
                    ls = f"{l:.3f}" if ok else "inf"
                    ss = f"{score:.3f}" if ok else "inf"
                    parts.append(f"h={math.degrees(h):.1f}deg dpsi={dpsi_deg:.1f} len={ls} score={ss} ok={ok}")
                chosen_dpsi = math.degrees(abs(wrap_to_pi(best_goal_heading - v.current_heading)))
                msg = (
                    f"V{v.id} T{tid} choose_h={math.degrees(best_goal_heading):.1f}deg "
                    f"choose_dpsi={chosen_dpsi:.1f} choose_len={length:.3f} choose_score={best_score:.3f} "
                    f"dpsi_limit={math.degrees(dpsi_limit_eff):.1f}deg "
                    f"cand[{len(cand_meta)}]: " + " | ".join(parts)
                )
                self.engine.event_logs.append(
                    EventLog(
                        step=self.step,
                        event_type="plan_debug",
                        task_id=tid,
                        message=msg,
                    )
                )

        if not path or length == float("inf"):
            path = [v.current_pos, task.position]
            length = math.hypot(task.position[0] - v.current_pos[0], task.position[1] - v.current_pos[1])

        path[0] = v.current_pos
        path[-1] = task.position
        v.route_points = path
        v.route_length = length
        v.path_cursor = 0
        v.distance_to_next_waypoint = self._distance_to_next_waypoint(v)
        v.is_moving = len(path) >= 2

    def _distance_to_next_waypoint(self, v: Vehicle) -> float:
        if len(v.route_points) < 2:
            return 0.0
        idx = min(max(v.path_cursor, 0), len(v.route_points) - 2)
        nxt = v.route_points[idx + 1]
        return math.hypot(nxt[0] - v.current_pos[0], nxt[1] - v.current_pos[1])

    def _active_task_reached(self, v: Vehicle) -> bool:
        self._assert_ready()
        assert self.engine is not None

        if v.active_task_id is None:
            return False
        task = self.engine.tasks_by_id.get(v.active_task_id)
        if task is None:
            return False
        tol = max(1e-6, float(getattr(self.cfg, "online_task_reach_tolerance", 0.25)))
        dist = math.hypot(task.position[0] - v.current_pos[0], task.position[1] - v.current_pos[1])
        return dist <= tol

    def _advance_vehicles(self, dt: float) -> None:
        self._assert_ready()
        assert self.engine is not None

        for v in self.engine.vehicles:
            self._advance_vehicle(v, dt)

    @staticmethod
    def _append_history_point(v: Vehicle, p: tuple[float, float]) -> None:
        if not v.history_points:
            v.history_points = [p]
            return
        last = v.history_points[-1]
        if math.hypot(p[0] - last[0], p[1] - last[1]) > 1e-6:
            v.history_points.append(p)

    def _advance_vehicle(self, v: Vehicle, dt: float) -> None:
        remaining = max(0.0, v.speed * dt)

        while remaining > 1e-9:
            if v.active_task_id is None:
                self._activate_next_task(v)
                if v.active_task_id is None:
                    break

            if not v.is_moving or len(v.route_points) < 2:
                self._build_active_segment(v)
                if not v.is_moving:
                    if self._active_task_reached(v):
                        self._complete_active_task(v)
                        continue
                    break

            if v.path_cursor >= len(v.route_points) - 1:
                self._complete_active_task(v)
                continue

            nxt = v.route_points[v.path_cursor + 1]
            dx = nxt[0] - v.current_pos[0]
            dy = nxt[1] - v.current_pos[1]
            seg = math.hypot(dx, dy)

            if seg <= 1e-9:
                v.path_cursor += 1
                continue

            if remaining < seg:
                ratio = remaining / seg
                new_pos = (v.current_pos[0] + dx * ratio, v.current_pos[1] + dy * ratio)
                v.current_heading = heading_to_point(v.current_pos, nxt)
                v.current_pos = new_pos
                self._append_history_point(v, v.current_pos)
                remaining = 0.0
            else:
                v.current_heading = heading_to_point(v.current_pos, nxt)
                v.current_pos = nxt
                self._append_history_point(v, v.current_pos)
                remaining -= seg
                v.path_cursor += 1
                if v.path_cursor >= len(v.route_points) - 1:
                    self._complete_active_task(v)

            v.distance_to_next_waypoint = self._distance_to_next_waypoint(v)

    def _activate_next_task(self, v: Vehicle) -> None:
        self._assert_ready()
        assert self.engine is not None

        while v.task_sequence:
            tid = v.task_sequence[0]
            task = self.engine.tasks_by_id.get(tid)
            if task is None or task.status in {"canceled", "completed"}:
                v.task_sequence = v.task_sequence[1:]
                continue
            v.active_task_id = tid
            task.status = "in_progress"
            task.assigned_vehicle = v.id
            self._build_active_segment(v)
            return

        v.active_task_id = None
        v.route_points = [v.current_pos]
        v.route_length = 0.0
        v.path_cursor = 0
        v.distance_to_next_waypoint = 0.0
        v.is_moving = False

    def _complete_active_task(self, v: Vehicle) -> None:
        self._assert_ready()
        assert self.engine is not None

        tid = v.active_task_id
        if tid is None:
            return

        task = self.engine.tasks_by_id.get(tid)
        if task is not None:
            v.current_pos = task.position
            task.status = "completed"
            task.assigned_vehicle = v.id

        if tid in v.task_sequence:
            v.task_sequence = [x for x in v.task_sequence if x != tid]

        self.engine.event_logs.append(
            EventLog(
                step=self.step,
                event_type="complete_task",
                task_id=tid,
                message=f"vehicle V{v.id} completed task T{tid} at t={self.sim_time:.2f}s",
            )
        )

        v.active_task_id = None
        v.path_cursor = 0
        v.distance_to_next_waypoint = 0.0
        v.is_moving = False
        v.route_points = [v.current_pos]
        v.route_length = 0.0

        # May continue moving in same tick if there is time left.
        self._activate_next_task(v)

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
        if self.online_enabled:
            raise ValueError("undo is disabled in online mode")

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

    def _validate_task_point(self, p: tuple[float, float]) -> None:
        self._assert_ready()
        assert self.artifacts is not None

        if not self.artifacts.world.point_in_bounds(p, margin=0.0):
            raise ValueError("point is out of map bounds")
        if Point(p).within(self.artifacts.world.depot_polygon):
            raise ValueError("point is inside depot")
        if not self.artifacts.world.point_is_free(p, clearance=self.cfg.vehicle_radius + 0.2):
            raise ValueError("point is in obstacle/safety area")

    def can_undo(self) -> bool:
        return (not self.online_enabled) and len(self._undo_stack) > 0

    def undo(self) -> None:
        self._assert_ready()
        if self.online_enabled:
            raise ValueError("undo is disabled in online mode")
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
        self.engine.corrected_bid_cache.clear()
        self.engine.force_accept_pairs.clear()

    def _validate_obstacle_polygon(self, points: list[tuple[float, float]]) -> Polygon:
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
            if Point(v.current_pos).within(inflated):
                raise ValueError(f"polygon blocks vehicle V{v.id} current position")

        return poly

    def _apply_obstacle_polygon_now(self, points: list[tuple[float, float]]) -> Polygon:
        self._assert_ready()
        assert self.artifacts is not None

        poly = self._validate_obstacle_polygon(points)
        self.artifacts.world.add_obstacle(poly)
        self._rebuild_planner()
        return poly

    def add_obstacle_polygon(self, points: list[tuple[float, float]]) -> Polygon:
        self._assert_ready()

        poly = self._validate_obstacle_polygon(points)
        if self.online_enabled:
            self.schedule_event(
                at_time=self.sim_time,
                event_type="add_obstacle",
                payload={"points": [(float(x), float(y)) for x, y in points]},
            )
            self.tick(0)
            return poly

        assert self.artifacts is not None
        assert self.engine is not None

        prev_obstacle_count = len(self.artifacts.world.obstacles)
        prev_planner = self.engine.planner
        self._push_undo_state()

        self.artifacts.world.add_obstacle(poly)
        self._rebuild_planner()
        try:
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

    def remove_obstacle(self, obstacle_idx: int) -> None:
        self._assert_ready()
        assert self.artifacts is not None

        idx = int(obstacle_idx)
        if idx < 0 or idx >= len(self.artifacts.world.obstacles):
            raise ValueError(f"invalid obstacle index: {idx}")

        if self.online_enabled:
            self.schedule_event(
                at_time=self.sim_time,
                event_type="remove_obstacle",
                payload={"obstacle_idx": idx},
            )
            self.tick(0)
            return

        self._push_undo_state()
        self.artifacts.world.remove_obstacle(idx)
        self._rebuild_planner()
        self.engine.allocate_until_stable(phase=f"session:remove_obstacle@{self.step}")

    def list_obstacles(self) -> list[tuple[int, float]]:
        self._assert_ready()
        assert self.artifacts is not None
        out: list[tuple[int, float]] = []
        for idx, obs in enumerate(self.artifacts.world.obstacles):
            out.append((idx, float(obs.area)))
        return out

    def add_task(self, x: float, y: float, demand: int, task_id: int | None = None) -> Task:
        self._assert_ready()
        assert self.engine is not None

        p = (x, y)
        self._validate_task_point(p)
        if demand <= 0:
            raise ValueError("demand must be positive")

        tid = self._next_task_id() if task_id is None else int(task_id)
        task = Task(id=tid, position=p, demand=int(demand), status="unassigned")

        if self.online_enabled:
            self.schedule_event(at_time=self.sim_time, event_type="add_task", payload={"task": task})
            self.tick(0)
            return task

        self._push_undo_state()
        self.step += 1
        self.engine.add_dynamic_task(task=task, step=self.step)
        self.engine.allocate_until_stable(phase=f"session:add@{self.step}")
        return task

    def move_task(self, task_id: int, x: float, y: float) -> Task:
        self._assert_ready()
        assert self.engine is not None

        tid = int(task_id)
        task = self.engine.tasks_by_id.get(tid)
        if task is None:
            raise ValueError(f"task {tid} not found")
        if task.status == "canceled":
            raise ValueError(f"task T{tid} already canceled")

        p = (float(x), float(y))
        self._validate_task_point(p)

        old_pos = task.position
        if abs(old_pos[0] - p[0]) < 1e-9 and abs(old_pos[1] - p[1]) < 1e-9:
            return task

        if self.online_enabled:
            # Treat as cancel+add with same id to keep event model compact.
            self.schedule_event(at_time=self.sim_time, event_type="cancel_task", payload={"task_id": tid})
            moved = Task(id=tid, position=p, demand=task.demand, status="unassigned")
            self.schedule_event(at_time=self.sim_time, event_type="add_task", payload={"task": moved})
            self.tick(0)
            return moved

        self._push_undo_state()
        self.step += 1

        owner = task.assigned_vehicle
        if owner is not None:
            vehicle = self.engine.vehicles[owner]
            if tid in vehicle.task_sequence:
                vehicle.task_sequence = [seq_tid for seq_tid in vehicle.task_sequence if seq_tid != tid]
            self.engine._recompute_vehicle_state(vehicle)

        task.position = p
        task.status = "withdrawn"
        task.assigned_vehicle = None

        for vid in range(len(self.engine.vehicles)):
            pair = (vid, tid)
            self.engine.corrected_bid_cache.pop(pair, None)
            self.engine.force_accept_pairs.discard(pair)

        base = self.engine.records.get(tid)
        next_version = (base.version + 1) if base is not None else 1
        self.engine.records[tid] = TaskRecord(
            task_id=tid,
            winner=None,
            bid=float("inf"),
            status="withdrawn",
            version=next_version,
        )

        self.engine.event_logs.append(
            EventLog(
                step=self.step,
                event_type="move_task",
                task_id=tid,
                message=f"moved task T{tid} from {old_pos} to {p}",
            )
        )
        self.engine.allocate_until_stable(phase=f"session:move@{self.step}")
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

        if self.online_enabled:
            self.schedule_event(at_time=self.sim_time, event_type="add_task", payload={"task": task})
            self.tick(0)
            return task

        self._push_undo_state()
        self.step += 1
        self.engine.add_dynamic_task(task=task, step=self.step)
        self.engine.allocate_until_stable(phase=f"session:add_random@{self.step}")
        return task

    def cancel_task(self, task_id: int) -> None:
        self._assert_ready()
        assert self.engine is not None

        tid = int(task_id)
        if self.online_enabled:
            self.schedule_event(at_time=self.sim_time, event_type="cancel_task", payload={"task_id": tid})
            self.tick(0)
            return

        self._push_undo_state()
        self.step += 1
        self.engine.cancel_task(task_id=tid, step=self.step)
        self.engine.allocate_until_stable(phase=f"session:cancel@{self.step}")

    def result(self) -> AllocationResult:
        self._assert_ready()
        assert self.engine is not None

        if not self.online_enabled:
            return self.engine.finalize()

        return AllocationResult(
            vehicles=self.engine.vehicles,
            tasks=self.engine.tasks,
            auction_logs=self.engine.auction_logs,
            coordination_logs=self.engine.coordination_logs,
            verification_logs=self.engine.verification_logs,
            event_logs=self.engine.event_logs,
            system_total_time=self._estimate_remaining_system_time(preempt=False),
        )

    def status_snapshot(self) -> SessionSnapshot:
        if self.online_enabled:
            return SessionSnapshot(
                step=self.step,
                total_tasks=len(self.engine.tasks if self.engine else []),
                status_counts=self.status_counts(),
                system_total_time=self._estimate_remaining_system_time(preempt=False),
            )

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
        self._assert_ready()
        assert self.engine is not None

        if not self.online_enabled:
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

        snap = self.runtime_snapshot()
        lines = [
            "-" * 88,
            (
                f"step={self.step} sim_time={snap.sim_time:.2f}s running={snap.online_running} "
                f"dt={snap.dt:.2f} replan_period={snap.replan_period_s:.2f}s "
                f"next_replan={snap.next_replan_time:.2f}s"
            ),
            (
                f"status={self.status_counts()} pending_events={len(snap.pending_events)} "
                f"last_replan={snap.last_replan_reason or '-'}"
            ),
            f"last_event={snap.last_event_message or '-'}",
        ]

        for v in self.engine.vehicles:
            seq = ",".join(f"T{tid}" for tid in v.task_sequence) if v.task_sequence else "(none)"
            lines.append(
                (
                    f"V{v.id}: pos=({v.current_pos[0]:.2f},{v.current_pos[1]:.2f}) "
                    f"active={v.active_task_id} moving={v.is_moving} remain={v.remaining_capacity}/{v.capacity} "
                    f"tasks=[{seq}]"
                )
            )

        lines.append(f"estimated_remaining_total_time={self._estimate_remaining_system_time(preempt=False):.3f}")
        lines.append("-" * 88)
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

        if self.online_enabled and self.event_history:
            lines.append("Online events:")
            for evt in self.event_history[-n:]:
                lines.append(
                    f"  id={evt.event_id} t={evt.time_s:.2f} {evt.event_type} applied={evt.applied} {evt.result_message}"
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
        assert self.engine is not None

        title = "Realtime Allocation View"
        if self.online_enabled:
            title = f"Realtime Allocation View (t={self.sim_time:.1f}s)"

        draw_final_scene_on_axis(
            ax=ax,
            world=self.artifacts.world,
            vehicles=self.engine.vehicles,
            tasks=self.engine.tasks,
            title=title,
            show_task_meta=False,
            show_vehicle_sequences=False,
            task_font_size=8,
            vehicle_font_size=9,
            label_box=True,
            curve_width=1.6,
        )

        if self.online_enabled:
            colors = plt.get_cmap("tab10")
            for v in self.engine.vehicles:
                c = colors(v.id % 10)
                if len(v.history_points) >= 2:
                    hx = [p[0] for p in v.history_points]
                    hy = [p[1] for p in v.history_points]
                    ax.plot(
                        hx,
                        hy,
                        color=c,
                        linewidth=1.4,
                        linestyle="--",
                        alpha=0.65,
                        zorder=9,
                    )
                ax.scatter(
                    [v.current_pos[0]],
                    [v.current_pos[1]],
                    s=95,
                    color=c,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=12,
                )
                ux = math.cos(v.current_heading)
                uy = math.sin(v.current_heading)
                ax.arrow(
                    v.current_pos[0],
                    v.current_pos[1],
                    ux * 1.2,
                    uy * 1.2,
                    color=c,
                    linewidth=1.2,
                    head_width=0.6,
                    head_length=0.8,
                    length_includes_head=True,
                    zorder=13,
                )

                if v.active_task_id is not None:
                    task = self.engine.tasks_by_id.get(v.active_task_id)
                    if task is not None:
                        ax.scatter(
                            [task.position[0]],
                            [task.position[1]],
                            s=120,
                            facecolors="none",
                            edgecolors=c,
                            linewidths=1.8,
                            zorder=11,
                        )

            ax.text(
                0.01,
                0.98,
                f"replan={self.last_replan_reason or '-'} | event={self.last_event_message or '-'}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=20,
            )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_snapshot(self, filename: str | None = None) -> Path:
        self._assert_ready()
        assert self.artifacts is not None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            path = self.output_dir / filename
        else:
            path = self.output_dir / f"snapshot_{self._timestamp()}_step_{self.step}.png"

        if not self.online_enabled:
            result = self.result()
            plot_final_scene(
                world=self.artifacts.world,
                vehicles=result.vehicles,
                tasks=result.tasks,
                save_path=path,
                dpi=self.cfg.figure_dpi,
                fig_size=self.cfg.figure_size,
            )
            return path

        fig, ax = plt.subplots(figsize=self.cfg.figure_size, dpi=self.cfg.figure_dpi)
        self.draw_on_axis(ax)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def export_logs(self, prefix: str | None = None) -> tuple[Path, Path]:
        self._assert_ready()
        assert self.engine is not None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"session_{self._timestamp()}_step_{self.step}"

        coord_path = self.output_dir / f"{prefix}_coordination_log.txt"
        verify_path = self.output_dir / f"{prefix}_verification_log.txt"
        event_path = self.output_dir / f"{prefix}_event_log.txt"
        runtime_path = self.output_dir / f"{prefix}_online_runtime_log.txt"

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

        with event_path.open("w", encoding="utf-8") as f:
            f.write("step,event_type,task_id,message\n")
            for item in self.engine.event_logs:
                msg = str(item.message).replace("\n", "\\n")
                f.write(f"{item.step},{item.event_type},{item.task_id},{msg}\n")

        with runtime_path.open("w", encoding="utf-8") as f:
            f.write(
                "frame_idx,step,sim_time,vehicle_id,current_x,current_y,current_heading,"
                "active_task_id,is_moving,distance_to_next,task_sequence\n"
            )
            for idx, frame in enumerate(self._frame_history):
                for v in frame.engine.vehicles:
                    seq = "|".join(str(tid) for tid in v.task_sequence)
                    active = "" if v.active_task_id is None else str(v.active_task_id)
                    f.write(
                        f"{idx},{frame.step},{frame.sim_time:.3f},{v.id},"
                        f"{v.current_pos[0]:.6f},{v.current_pos[1]:.6f},{v.current_heading:.6f},"
                        f"{active},{v.is_moving},{v.distance_to_next_waypoint:.6f},{seq}\n"
                    )

        return coord_path, verify_path
