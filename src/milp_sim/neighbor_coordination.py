from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import SimulationConfig
from .entities import Vehicle


STATUS_PRIORITY = {
    "canceled": 6,
    "locked": 5,
    "verifying": 4,
    "tentative": 3,
    "withdrawn": 2,
    "unassigned": 1,
}


@dataclass
class TaskRecord:
    task_id: int
    winner: Optional[int]
    bid: float
    status: str
    version: int


@dataclass
class SyncRoundTrace:
    step: int
    distinct_records: int
    stable_count: int


@dataclass
class CoordinationLog:
    task_id: int
    event: str
    rounds: int
    converged: bool
    final_winner: Optional[int]
    final_status: str
    traces: List[SyncRoundTrace]


def resolve_record(a: TaskRecord, b: TaskRecord) -> TaskRecord:
    pa = STATUS_PRIORITY.get(a.status, 0)
    pb = STATUS_PRIORITY.get(b.status, 0)

    if pa != pb:
        return a if pa > pb else b
    if a.version != b.version:
        return a if a.version > b.version else b
    if a.bid != b.bid:
        return a if a.bid < b.bid else b

    aw = 10**9 if a.winner is None else a.winner
    bw = 10**9 if b.winner is None else b.winner
    return a if aw <= bw else b


def build_neighbors(vehicles: List[Vehicle], comm_radius: float) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {v.id: [] for v in vehicles}
    r2 = comm_radius * comm_radius

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            vi = vehicles[i]
            vj = vehicles[j]
            dx = vi.current_pos[0] - vj.current_pos[0]
            dy = vi.current_pos[1] - vj.current_pos[1]
            if dx * dx + dy * dy <= r2:
                neighbors[vi.id].append(vj.id)
                neighbors[vj.id].append(vi.id)

    return neighbors


def run_coordination(
    task_id: int,
    event: str,
    base_record: TaskRecord,
    proposals: Dict[int, TaskRecord],
    neighbors: Dict[int, List[int]],
    cfg: SimulationConfig,
) -> tuple[TaskRecord, CoordinationLog]:
    views: Dict[int, TaskRecord] = {}

    for vid in neighbors:
        views[vid] = proposals.get(vid, base_record)

    stable_count = 0
    traces: List[SyncRoundTrace] = []
    converged = False

    for step in range(1, cfg.sync_rmax + 1):
        next_views: Dict[int, TaskRecord] = {}
        for vid, rec in views.items():
            best = rec
            for nb in neighbors[vid]:
                best = resolve_record(best, views[nb])
            next_views[vid] = best

        unique = {(r.winner, r.status, r.version, round(r.bid, 8)) for r in next_views.values()}
        if len(unique) == 1:
            stable_count += 1
        else:
            stable_count = 0

        traces.append(SyncRoundTrace(step=step, distinct_records=len(unique), stable_count=stable_count))
        views = next_views

        if stable_count >= cfg.sync_stable_h:
            converged = True
            break

    final = base_record
    for rec in views.values():
        final = resolve_record(final, rec)

    log = CoordinationLog(
        task_id=task_id,
        event=event,
        rounds=len(traces),
        converged=converged,
        final_winner=final.winner,
        final_status=final.status,
        traces=traces,
    )
    return final, log
