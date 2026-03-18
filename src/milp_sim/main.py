from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, SimulationConfig
from .log_export import write_auction_big_log, write_coordination_log, write_verification_log
from .simulator import run_round2_pipeline
from .visualization import plot_final_scene, plot_initial_scene


def dump_logs(result, out_dir: Path) -> tuple[Path, Path, Path]:
    coord_path = out_dir / "coordination_log.txt"
    verify_path = out_dir / "verification_log.txt"
    big_path = out_dir / "auction_big_log.txt"

    write_coordination_log(coord_path, result.allocation.coordination_logs)
    write_verification_log(verify_path, result.allocation.verification_logs)
    write_auction_big_log(big_path, result.allocation.auction_logs)

    return coord_path, verify_path, big_path


def print_summary(result) -> None:
    print("=" * 72)
    print("Round-2 Online Allocation Summary")
    print("=" * 72)

    for v in result.allocation.vehicles:
        seq = ", ".join(f"T{tid}" for tid in v.task_sequence) if v.task_sequence else "(none)"
        time_cost = v.route_length / v.speed
        print(
            f"Vehicle V{v.id}: speed={v.speed:.2f}, capacity={v.capacity}, "
            f"remaining={v.remaining_capacity}, tasks=[{seq}], total_time={time_cost:.3f}"
        )

    print("-" * 72)
    print(f"System total execution time: {result.allocation.system_total_time:.3f}")
    print(f"Auction rounds: {len(result.allocation.auction_logs)}")
    print(f"Verification checks: {len(result.allocation.verification_logs)}")
    print(f"Coordination logs: {len(result.allocation.coordination_logs)}")
    print(f"Dynamic events: {len(result.allocation.event_logs)}")

    if result.allocation.event_logs:
        print("-" * 72)
        print("Dynamic event timeline:")
        for e in result.allocation.event_logs:
            print(f"  step={e.step:02d} {e.event_type} T{e.task_id}: {e.message}")

    failed = [v for v in result.allocation.verification_logs if not v.passed]
    if failed:
        print("-" * 72)
        print(f"Verification withdrawals: {len(failed)}")
        for item in failed[:8]:
            print(
                f"  round={item.round_idx} T{item.task_id} by V{item.vehicle_id}: "
                f"c_hat={item.c_hat:.3f}, c_tilde={item.c_tilde:.3f}, e_under={item.e_under:.3f}"
            )

    print("=" * 72)


def run(cfg: SimulationConfig = DEFAULT_CONFIG) -> None:
    result = run_round2_pipeline(cfg)

    out_dir = Path("outputs")
    initial_fig = out_dir / "01_initial_scene.png"
    final_fig = out_dir / "03_round2_final_result.png"

    plot_initial_scene(
        world=result.artifacts.world,
        vehicles=result.artifacts.vehicles,
        tasks=result.artifacts.tasks,
        save_path=initial_fig,
        dpi=cfg.figure_dpi,
        fig_size=cfg.figure_size,
    )

    plot_final_scene(
        world=result.artifacts.world,
        vehicles=result.allocation.vehicles,
        tasks=result.allocation.tasks,
        save_path=final_fig,
        dpi=cfg.figure_dpi,
        fig_size=cfg.figure_size,
        predicted_future_links=max(0, int(cfg.online_future_task_horizon)),
    )

    coord_log, verify_log, big_log = dump_logs(result, out_dir)
    print_summary(result)
    print(f"Saved figures: {initial_fig} , {final_fig}")
    print(f"Saved logs: {coord_log} , {verify_log} , {big_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MILP dynamic task allocation simulator")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="start interactive console for realtime add/cancel operations",
    )
    mode.add_argument(
        "--gui",
        action="store_true",
        help="start Tkinter GUI for realtime add/cancel operations",
    )
    args = parser.parse_args()

    if args.interactive:
        from .interactive_console import run_interactive

        run_interactive(DEFAULT_CONFIG)
    elif args.gui:
        from .gui_app import run_gui

        run_gui(DEFAULT_CONFIG)
    else:
        run(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
