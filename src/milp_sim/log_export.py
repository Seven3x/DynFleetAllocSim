from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .auction_core import AuctionRoundLog, CoordinationLog, EventLog, VerificationLog


def write_coordination_log(path: Path, coordination_logs: Iterable[CoordinationLog]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("task_id,event,rounds,converged,final_winner,final_status,trace\n")
        for item in coordination_logs:
            trace = ";".join(f"step{t.step}:d{t.distinct_records}:s{t.stable_count}" for t in item.traces)
            f.write(
                f"{item.task_id},{item.event},{item.rounds},{item.converged},"
                f"{item.final_winner},{item.final_status},{trace}\n"
            )


def write_verification_log(path: Path, verification_logs: Iterable[VerificationLog]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("round,task_id,vehicle_id,c_hat,c_tilde,e_under,passed,forced_accept\n")
        for item in verification_logs:
            f.write(
                f"{item.round_idx},{item.task_id},{item.vehicle_id},"
                f"{item.c_hat:.6f},{item.c_tilde:.6f},{item.e_under:.6f},"
                f"{item.passed},{item.forced_accept}\n"
            )


def write_event_log(path: Path, event_logs: Iterable[EventLog]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("step,event_type,task_id,message\n")
        for item in event_logs:
            msg = str(item.message).replace("\n", "\\n")
            f.write(f"{item.step},{item.event_type},{item.task_id},{msg}\n")


def write_auction_big_log(path: Path, auction_logs: Iterable[AuctionRoundLog]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "round,phase,vehicle_id,remaining_capacity,current_x,current_y,current_heading,"
            "task_sequence,capacity_blocked_task_ids,unreachable_task_ids,candidate_task_id,"
            "candidate_cost,is_best_bid,is_tentative_winner\n"
        )
        for round_log in auction_logs:
            winner_by_vehicle = {
                winner.vehicle_id: winner.task_id
                for winner in round_log.tentative_winners
            }
            for vehicle_log in round_log.vehicle_logs:
                seq = "|".join(str(tid) for tid in vehicle_log.task_sequence)
                blocked = "|".join(str(tid) for tid in vehicle_log.capacity_blocked_task_ids)
                unreachable = "|".join(str(tid) for tid in vehicle_log.unreachable_task_ids)
                if vehicle_log.candidate_bids:
                    for bid in vehicle_log.candidate_bids:
                        is_best_bid = bid.task_id == vehicle_log.chosen_task_id
                        is_tentative_winner = winner_by_vehicle.get(vehicle_log.vehicle_id) == bid.task_id
                        f.write(
                            f"{round_log.round_idx},{round_log.phase},{vehicle_log.vehicle_id},"
                            f"{vehicle_log.remaining_capacity},{vehicle_log.current_pos[0]:.6f},"
                            f"{vehicle_log.current_pos[1]:.6f},{vehicle_log.current_heading:.6f},"
                            f"{seq},{blocked},{unreachable},{bid.task_id},{bid.value:.6f},"
                            f"{is_best_bid},{is_tentative_winner}\n"
                        )
                else:
                    f.write(
                        f"{round_log.round_idx},{round_log.phase},{vehicle_log.vehicle_id},"
                        f"{vehicle_log.remaining_capacity},{vehicle_log.current_pos[0]:.6f},"
                        f"{vehicle_log.current_pos[1]:.6f},{vehicle_log.current_heading:.6f},"
                        f"{seq},{blocked},{unreachable},,,False,False\n"
                    )
