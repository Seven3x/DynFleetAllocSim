from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import SimulationConfig
from .cost_estimator import (
    fast_cost_estimate_from_state,
    goal_heading_candidates,
    heading_to_point,
    wrap_to_pi,
)
from .dubins_path import build_dubins_hybrid_path
from .entities import Task, Vehicle
from .map_utils import WorldMap
from .neighbor_coordination import CoordinationLog, TaskRecord, build_neighbors, run_coordination
from .path_postprocess import maybe_buffer_initial_turn_path, resample_path
from .planner_astar import AStarPlanner
from .verification import VerificationResult, verify_bid


AUCTIONABLE_STATES = {"unassigned", "withdrawn"}


@dataclass
class Bid:
    vehicle_id: int
    task_id: int
    value: float


@dataclass
class AuctionRoundLog:
    round_idx: int
    phase: str
    bids: List[Bid]
    tentative_winners: List[Bid]
    vehicle_logs: List["VehicleAuctionLog"]


@dataclass
class VehicleAuctionLog:
    vehicle_id: int
    remaining_capacity: int
    current_pos: Tuple[float, float]
    current_heading: float
    task_sequence: List[int]
    candidate_bids: List[Bid]
    capacity_blocked_task_ids: List[int]
    unreachable_task_ids: List[int]
    chosen_task_id: Optional[int]
    chosen_cost: float


@dataclass
class VerificationLog:
    round_idx: int
    task_id: int
    vehicle_id: int
    c_hat: float
    c_tilde: float
    e_under: float
    passed: bool
    forced_accept: bool
    dubins_used_fallback: bool = False
    dubins_fallback_details: str = ""


@dataclass
class EventLog:
    step: int
    event_type: str
    task_id: int
    message: str


@dataclass
class AllocationResult:
    vehicles: List[Vehicle]
    tasks: List[Task]
    auction_logs: List[AuctionRoundLog]
    coordination_logs: List[CoordinationLog]
    verification_logs: List[VerificationLog]
    event_logs: List[EventLog]
    system_total_time: float


class AllocationEngine:
    def __init__(
        self,
        vehicles: List[Vehicle],
        tasks: List[Task],
        world: WorldMap,
        cfg: SimulationConfig,
        planner: AStarPlanner,
    ) -> None:
        self.vehicles = vehicles
        self.tasks = tasks
        self.world = world
        self.cfg = cfg
        self.planner = planner

        self.tasks_by_id: Dict[int, Task] = {t.id: t for t in self.tasks}
        self.records: Dict[int, TaskRecord] = {}
        self.corrected_bid_cache: Dict[Tuple[int, int], float] = {}
        self.force_accept_pairs: set[Tuple[int, int]] = set()

        self.auction_logs: List[AuctionRoundLog] = []
        self.coordination_logs: List[CoordinationLog] = []
        self.verification_logs: List[VerificationLog] = []
        self.event_logs: List[EventLog] = []

        self._round_idx = 0

    def reset(self) -> None:
        for v in self.vehicles:
            v.reset_runtime_state()

        self.records = {}
        for t in self.tasks:
            t.status = "unassigned"
            t.assigned_vehicle = None
            self.records[t.id] = TaskRecord(
                task_id=t.id,
                winner=None,
                bid=float("inf"),
                status="unassigned",
                version=0,
            )

        self.corrected_bid_cache.clear()
        self.force_accept_pairs.clear()

        self.auction_logs.clear()
        self.coordination_logs.clear()
        self.verification_logs.clear()
        self.event_logs.clear()
        self._round_idx = 0

    def add_dynamic_task(self, task: Task, step: int) -> None:
        if task.id in self.tasks_by_id:
            raise ValueError(f"Task id {task.id} already exists.")

        task.status = "unassigned"
        task.assigned_vehicle = None
        self.tasks.append(task)
        self.tasks_by_id[task.id] = task
        self.records[task.id] = TaskRecord(
            task_id=task.id,
            winner=None,
            bid=float("inf"),
            status="unassigned",
            version=0,
        )

        self.event_logs.append(
            EventLog(
                step=step,
                event_type="add_task",
                task_id=task.id,
                message=f"added task T{task.id} at {task.position} demand={task.demand}",
            )
        )

    def cancel_task(self, task_id: int, step: int) -> None:
        if task_id not in self.tasks_by_id:
            return

        task = self.tasks_by_id[task_id]
        if task.status in {"canceled", "completed"}:
            return

        if task.status not in {"locked", "in_progress"}:
            self._apply_canceled_record(task_id=task_id, step=step, message="canceled before lock")
            task.status = "canceled"
            task.assigned_vehicle = None
            return

        # locked/in-progress but not irreversibly executed in current simulator model
        owner = task.assigned_vehicle
        if owner is not None:
            vehicle = self.vehicles[owner]
            if task_id in vehicle.task_sequence:
                vehicle.task_sequence = [tid for tid in vehicle.task_sequence if tid != task_id]
            self._recompute_vehicle_state(vehicle)

        task.status = "canceled"
        task.assigned_vehicle = None
        self._apply_canceled_record(task_id=task_id, step=step, message="canceled and removed from vehicle sequence")

    def allocate_until_stable(self, phase: str) -> None:
        guard = 0
        while self._has_pending_auction_tasks():
            guard += 1
            if guard > 3000:
                raise RuntimeError("Auction loop exceeded safety limit.")

            bids, vehicle_logs = self._collect_bids()
            if not bids:
                pending = [t.id for t in self.tasks if t.status in AUCTIONABLE_STATES]
                raise RuntimeError(f"No feasible bids for pending tasks: {pending}")

            winners = self._choose_tentative_winners(bids)
            self.auction_logs.append(
                AuctionRoundLog(
                    round_idx=self._round_idx,
                    phase=phase,
                    bids=bids,
                    tentative_winners=winners,
                    vehicle_logs=vehicle_logs,
                )
            )

            self._run_neighbor_sync_for_winners(bids=bids, winners=winners, event=f"{phase}:auction")
            self._verify_tentatives(phase=phase)
            self._round_idx += 1

    def finalize(self) -> AllocationResult:
        self._build_routes_and_time()
        total_time = sum(v.route_length / v.speed for v in self.vehicles)

        return AllocationResult(
            vehicles=self.vehicles,
            tasks=self.tasks,
            auction_logs=self.auction_logs,
            coordination_logs=self.coordination_logs,
            verification_logs=self.verification_logs,
            event_logs=self.event_logs,
            system_total_time=total_time,
        )

    def _has_pending_auction_tasks(self) -> bool:
        for t in self.tasks:
            if t.status == "canceled":
                continue
            if t.status in AUCTIONABLE_STATES:
                return True
        return False

    def _committed_prefix_task_ids(self, vehicle: Vehicle, exclude_task_id: int | None = None) -> List[int]:
        if vehicle.active_task_id is None:
            return []

        out: List[int] = []
        seen: set[int] = set()

        active_tid = vehicle.active_task_id
        if active_tid is not None and active_tid != exclude_task_id:
            task = self.tasks_by_id.get(active_tid)
            if task is not None and task.status in {"locked", "in_progress"}:
                out.append(active_tid)
                seen.add(active_tid)

        for tid in vehicle.task_sequence:
            if tid in seen or tid == exclude_task_id:
                continue
            task = self.tasks_by_id.get(tid)
            if task is None:
                continue
            if task.status not in {"locked", "in_progress"}:
                continue
            out.append(tid)
            seen.add(tid)

        return out

    def _estimate_committed_prefix_time_and_frontier(
        self,
        vehicle: Vehicle,
        exclude_task_id: int | None = None,
    ) -> Tuple[float, Tuple[float, float], float]:
        total_time = 0.0
        frontier_pos = vehicle.current_pos
        frontier_heading = vehicle.current_heading

        for tid in self._committed_prefix_task_ids(vehicle=vehicle, exclude_task_id=exclude_task_id):
            task = self.tasks_by_id.get(tid)
            if task is None:
                continue
            detail = fast_cost_estimate_from_state(
                speed=vehicle.speed,
                max_omega=vehicle.max_omega,
                current_pos=frontier_pos,
                current_heading=frontier_heading,
                task=task,
                world=self.world,
                cfg=self.cfg,
            )
            if detail.estimated_time == float("inf"):
                return float("inf"), frontier_pos, frontier_heading
            total_time += detail.estimated_time
            frontier_heading = heading_to_point(frontier_pos, task.position)
            frontier_pos = task.position

        return total_time, frontier_pos, frontier_heading

    def _estimate_task_cost(self, vehicle: Vehicle, task: Task) -> float:
        prefix_weight = max(0.0, float(getattr(self.cfg, "committed_prefix_time_weight", 1.0)))
        if prefix_weight <= 1e-12:
            prefix_time = 0.0
            frontier_pos = vehicle.current_pos
            frontier_heading = vehicle.current_heading
        else:
            prefix_time, frontier_pos, frontier_heading = self._estimate_committed_prefix_time_and_frontier(
                vehicle=vehicle,
                exclude_task_id=task.id,
            )
            if prefix_time == float("inf"):
                return float("inf")

        cached = self.corrected_bid_cache.get((vehicle.id, task.id))
        if cached is not None and prefix_time <= 1e-9:
            return cached

        suffix = fast_cost_estimate_from_state(
            speed=vehicle.speed,
            max_omega=vehicle.max_omega,
            current_pos=frontier_pos,
            current_heading=frontier_heading,
            task=task,
            world=self.world,
            cfg=self.cfg,
        )
        if suffix.estimated_time == float("inf"):
            return float("inf")
        return prefix_weight * prefix_time + suffix.estimated_time

    def _collect_bids(self) -> Tuple[List[Bid], List[VehicleAuctionLog]]:
        candidate_tasks = [t for t in self.tasks if t.status in AUCTIONABLE_STATES]
        bids: List[Bid] = []
        vehicle_logs: List[VehicleAuctionLog] = []

        for v in self.vehicles:
            best_tid: Optional[int] = None
            best_cost = float("inf")
            candidate_bids: List[Bid] = []
            capacity_blocked_task_ids: List[int] = []
            unreachable_task_ids: List[int] = []
            for t in candidate_tasks:
                if t.demand > v.remaining_capacity:
                    capacity_blocked_task_ids.append(t.id)
                    continue
                c_hat = self._estimate_task_cost(v, t)
                if c_hat == float("inf"):
                    unreachable_task_ids.append(t.id)
                    continue
                candidate_bids.append(Bid(vehicle_id=v.id, task_id=t.id, value=c_hat))
                if c_hat < best_cost:
                    best_cost = c_hat
                    best_tid = t.id

            candidate_bids.sort(key=lambda x: (x.value, x.task_id))
            vehicle_logs.append(
                VehicleAuctionLog(
                    vehicle_id=v.id,
                    remaining_capacity=v.remaining_capacity,
                    current_pos=(float(v.current_pos[0]), float(v.current_pos[1])),
                    current_heading=float(v.current_heading),
                    task_sequence=list(v.task_sequence),
                    candidate_bids=candidate_bids,
                    capacity_blocked_task_ids=capacity_blocked_task_ids,
                    unreachable_task_ids=unreachable_task_ids,
                    chosen_task_id=best_tid,
                    chosen_cost=best_cost,
                )
            )
            if best_tid is not None:
                bids.append(Bid(vehicle_id=v.id, task_id=best_tid, value=best_cost))

        return bids, vehicle_logs

    @staticmethod
    def _choose_tentative_winners(bids: List[Bid]) -> List[Bid]:
        grouped: Dict[int, List[Bid]] = {}
        for b in bids:
            grouped.setdefault(b.task_id, []).append(b)

        winners: List[Bid] = []
        for task_id, task_bids in grouped.items():
            task_bids.sort(key=lambda x: (x.value, x.vehicle_id))
            winners.append(task_bids[0])

        winners.sort(key=lambda x: (x.task_id, x.value, x.vehicle_id))
        return winners

    def _run_neighbor_sync_for_winners(self, bids: List[Bid], winners: List[Bid], event: str) -> None:
        neighbors = build_neighbors(self.vehicles, self.cfg.comm_radius)
        bids_by_task: Dict[int, List[Bid]] = {}
        for b in bids:
            bids_by_task.setdefault(b.task_id, []).append(b)

        winner_ids = {w.task_id for w in winners}
        for task_id in winner_ids:
            task = self.tasks_by_id[task_id]
            if task.status == "canceled":
                continue

            base = self.records[task_id]
            proposals: Dict[int, TaskRecord] = {}
            for b in bids_by_task.get(task_id, []):
                proposals[b.vehicle_id] = TaskRecord(
                    task_id=task_id,
                    winner=b.vehicle_id,
                    bid=b.value,
                    status="tentative",
                    version=base.version + 1,
                )

            final_rec, clog = run_coordination(
                task_id=task_id,
                event=event,
                base_record=base,
                proposals=proposals,
                neighbors=neighbors,
                cfg=self.cfg,
            )
            self.coordination_logs.append(clog)

            if final_rec.status == "tentative" and final_rec.winner is not None:
                owner = self.vehicles[final_rec.winner]
                if task.demand <= owner.remaining_capacity:
                    task.status = "tentative"
                    task.assigned_vehicle = final_rec.winner
                    self.records[task_id] = final_rec
                    continue

            # fallback when tentative winner is infeasible due capacity drift
            self.records[task_id] = TaskRecord(
                task_id=task_id,
                winner=None,
                bid=float("inf"),
                status="withdrawn",
                version=base.version + 1,
            )
            task.status = "withdrawn"
            task.assigned_vehicle = None

    def _verify_tentatives(self, phase: str) -> None:
        tentative = [t for t in self.tasks if t.status == "tentative" and t.assigned_vehicle is not None]

        if not bool(getattr(self.cfg, "enable_bid_verification", True)):
            for task in tentative:
                vid = int(task.assigned_vehicle)
                vehicle = self.vehicles[vid]
                rec = self.records[task.id]
                self._lock_task(task=task, vehicle=vehicle, bid=rec.bid)
            return

        for task in tentative:
            vid = int(task.assigned_vehicle)
            vehicle = self.vehicles[vid]
            rec = self.records[task.id]
            c_hat = rec.bid
            pair = (vid, task.id)
            forced_accept = pair in self.force_accept_pairs

            task.status = "verifying"
            self.records[task.id] = TaskRecord(
                task_id=task.id,
                winner=vid,
                bid=c_hat,
                status="verifying",
                version=rec.version + 1,
            )

            if forced_accept:
                verify_res = VerificationResult(
                    passed=True,
                    path_length=0.0,
                    c_tilde=c_hat,
                    e_under=0.0,
                )
            else:
                verify_res = verify_bid(
                    vehicle=vehicle,
                    task=task,
                    c_hat=c_hat,
                    cfg=self.cfg,
                    planner=self.planner,
                    tasks_by_id=self.tasks_by_id,
                )

            self.verification_logs.append(
                VerificationLog(
                    round_idx=self._round_idx,
                    task_id=task.id,
                    vehicle_id=vid,
                    c_hat=c_hat,
                    c_tilde=verify_res.c_tilde,
                    e_under=verify_res.e_under,
                    passed=verify_res.passed,
                    forced_accept=forced_accept,
                    dubins_used_fallback=verify_res.dubins_used_fallback,
                    dubins_fallback_details=verify_res.dubins_fallback_details,
                )
            )
            if verify_res.debug_message:
                self.event_logs.append(
                    EventLog(
                        step=self._round_idx,
                        event_type="verify_plan_debug",
                        task_id=task.id,
                        message=verify_res.debug_message,
                    )
                )

            if verify_res.passed:
                self._lock_task(task=task, vehicle=vehicle, bid=c_hat)
                if forced_accept:
                    self.force_accept_pairs.discard(pair)
                continue

            self.corrected_bid_cache[pair] = verify_res.c_tilde
            self.force_accept_pairs.add(pair)
            task.status = "withdrawn"
            task.assigned_vehicle = None
            self.records[task.id] = TaskRecord(
                task_id=task.id,
                winner=None,
                bid=float("inf"),
                status="withdrawn",
                version=rec.version + 2,
            )

            neighbors = build_neighbors(self.vehicles, self.cfg.comm_radius)
            final_rec, clog = run_coordination(
                task_id=task.id,
                event=f"{phase}:withdraw",
                base_record=self.records[task.id],
                proposals={},
                neighbors=neighbors,
                cfg=self.cfg,
            )
            self.coordination_logs.append(clog)
            self.records[task.id] = final_rec

    def _lock_task(self, task: Task, vehicle: Vehicle, bid: float) -> None:
        if task.id not in vehicle.task_sequence:
            vehicle.task_sequence.append(task.id)
            vehicle.remaining_capacity -= task.demand

        task.status = "locked"
        task.assigned_vehicle = vehicle.id

        new_heading = heading_to_point(vehicle.current_pos, task.position)
        vehicle.current_pos = task.position
        vehicle.current_heading = new_heading

        rec = self.records[task.id]
        self.records[task.id] = TaskRecord(
            task_id=task.id,
            winner=vehicle.id,
            bid=bid,
            status="locked",
            version=rec.version + 1,
        )

    def _recompute_vehicle_state(self, vehicle: Vehicle) -> None:
        filtered: List[int] = []
        pos = vehicle.start_pos
        heading = vehicle.heading
        remaining = vehicle.capacity

        for tid in vehicle.task_sequence:
            task = self.tasks_by_id.get(tid)
            if task is None or task.status not in {"locked", "in_progress"}:
                continue
            if task.demand > remaining:
                continue
            filtered.append(tid)
            remaining -= task.demand
            heading = heading_to_point(pos, task.position)
            pos = task.position

        vehicle.task_sequence = filtered
        vehicle.remaining_capacity = remaining
        vehicle.current_pos = pos
        vehicle.current_heading = heading

    def _apply_canceled_record(self, task_id: int, step: int, message: str) -> None:
        base = self.records[task_id]
        self.records[task_id] = TaskRecord(
            task_id=task_id,
            winner=None,
            bid=float("inf"),
            status="canceled",
            version=base.version + 1,
        )

        neighbors = build_neighbors(self.vehicles, self.cfg.comm_radius)
        final_rec, clog = run_coordination(
            task_id=task_id,
            event="cancel",
            base_record=base,
            proposals={
                v.id: TaskRecord(
                    task_id=task_id,
                    winner=None,
                    bid=float("inf"),
                    status="canceled",
                    version=base.version + 1,
                )
                for v in self.vehicles
            },
            neighbors=neighbors,
            cfg=self.cfg,
        )
        self.records[task_id] = final_rec
        self.coordination_logs.append(clog)

        self.event_logs.append(
            EventLog(step=step, event_type="cancel_task", task_id=task_id, message=message)
        )

    def _build_routes_and_time(self) -> None:
        for v in self.vehicles:
            self._recompute_vehicle_state(v)
            v.route_points = [v.start_pos]
            v.route_length = 0.0
            cur = v.start_pos
            cur_heading = v.heading

            for idx, tid in enumerate(v.task_sequence):
                task = self.tasks_by_id[tid]
                turn_radius = v.speed / max(v.max_omega, 1e-6)
                next_task_pos: Optional[Tuple[float, float]] = None
                if idx + 1 < len(v.task_sequence):
                    nxt_task = self.tasks_by_id[v.task_sequence[idx + 1]]
                    next_task_pos = nxt_task.position
                headings = goal_heading_candidates(
                    current_pos=cur,
                    task_pos=task.position,
                    next_task_pos=next_task_pos,
                    turn_radius=turn_radius,
                    blend_turn_radius_factor=self.cfg.goal_heading_blend_turn_radius_factor,
                    tolerance_rad=self.cfg.goal_heading_tolerance_rad,
                    num_samples=self.cfg.goal_heading_num_samples,
                )
                best_path: List[Tuple[float, float]] = []
                best_len = float("inf")
                best_score = float("inf")
                best_goal_heading = heading_to_point(cur, task.position)
                turn_penalty = max(0.0, float(getattr(self.cfg, "goal_heading_turn_penalty", 0.0)))
                dpsi_limit = max(0.0, float(getattr(self.cfg, "goal_heading_max_dpsi_rad", 0.0)))
                dpsi_slack = max(0.0, float(getattr(self.cfg, "goal_heading_max_dpsi_slack_rad", 0.0)))
                dpsi_limit_eff = dpsi_limit + dpsi_slack
                fallback_path: List[Tuple[float, float]] = []
                fallback_len = float("inf")
                fallback_score = float("inf")
                fallback_heading = best_goal_heading
                for goal_heading in headings:
                    cand_path, cand_len, _ = build_dubins_hybrid_path(
                        world=self.world,
                        cfg=self.cfg,
                        start_pose=(cur[0], cur[1], cur_heading),
                        goal_pose=(task.position[0], task.position[1], goal_heading),
                        astar_planner=self.planner,
                        turn_radius=turn_radius,
                    )
                    ok = bool(cand_path) and cand_len != float("inf")
                    if not ok:
                        continue
                    dpsi = abs(wrap_to_pi(goal_heading - cur_heading))
                    score = cand_len + turn_penalty * turn_radius * dpsi
                    if score < fallback_score:
                        fallback_path = cand_path
                        fallback_len = cand_len
                        fallback_score = score
                        fallback_heading = goal_heading
                    if dpsi > dpsi_limit_eff + 1e-9:
                        continue
                    if score < best_score:
                        best_path = cand_path
                        best_len = cand_len
                        best_score = score
                        best_goal_heading = goal_heading

                if not best_path and fallback_path:
                    best_path = fallback_path
                    best_len = fallback_len
                    best_score = fallback_score
                    best_goal_heading = fallback_heading

                path, length = best_path, best_len
                if not path or length == float("inf"):
                    raise RuntimeError(
                        f"Hybrid planning failed for vehicle={v.id}, task={tid}, from={cur} to={task.position}."
                    )

                path, length = maybe_buffer_initial_turn_path(
                    world=self.world,
                    cfg=self.cfg,
                    planner=self.planner,
                    start_pos=cur,
                    start_heading=cur_heading,
                    task_pos=task.position,
                    path=path,
                    length=length,
                    goal_heading=best_goal_heading,
                    turn_radius=turn_radius,
                )
                runtime_step = max(
                    0.05,
                    float(
                        getattr(
                            self.cfg,
                            "online_path_sample_step",
                            min(0.25, float(getattr(self.cfg, "dubins_sample_step", 0.5))),
                        )
                    ),
                )
                path = resample_path(path, max_step=runtime_step)
                path[0] = cur
                path[-1] = task.position

                if len(path) > 1:
                    v.route_points.extend(path[1:])
                v.route_length += length
                cur = task.position
                cur_heading = best_goal_heading


def run_static_auction(
    vehicles: List[Vehicle],
    tasks: List[Task],
    world: WorldMap,
    cfg: SimulationConfig,
    planner: AStarPlanner,
) -> AllocationResult:
    engine = AllocationEngine(vehicles=vehicles, tasks=tasks, world=world, cfg=cfg, planner=planner)
    engine.reset()
    engine.allocate_until_stable(phase="initial")
    return engine.finalize()


def run_online_allocation(
    vehicles: List[Vehicle],
    tasks: List[Task],
    world: WorldMap,
    cfg: SimulationConfig,
    planner: AStarPlanner,
    events: Sequence[dict],
) -> AllocationResult:
    engine = AllocationEngine(vehicles=vehicles, tasks=tasks, world=world, cfg=cfg, planner=planner)
    engine.reset()

    engine.allocate_until_stable(phase="initial")

    for evt in events:
        step = int(evt.get("step", 0))
        etype = evt.get("event_type")
        if etype == "add_task":
            task = evt["task"]
            engine.add_dynamic_task(task=task, step=step)
            engine.allocate_until_stable(phase=f"step{step}:add")
        elif etype == "cancel_task":
            task_id = int(evt["task_id"])
            engine.cancel_task(task_id=task_id, step=step)
            engine.allocate_until_stable(phase=f"step{step}:cancel")

    return engine.finalize()
