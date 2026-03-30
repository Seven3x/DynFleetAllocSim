from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "outputs" / ".mplconfig").resolve()))

from shapely.geometry import Point

from .config import DEFAULT_CONFIG, SimulationConfig
from .session import OfflineSession


DEFAULT_SCENARIOS = [
    "examples/scenario_offline_5v50_corridor.json",
    "examples/scenario_offline_5v50_fragmented_many.json",
    "examples/scenario_offline_5v50_irregular_many_large.json",
    "examples/scenario_verification_demo.json",
]


@dataclass
class ScenarioBenchmarkResult:
    scenario_file: str
    report_path: str
    summary_path: str
    vehicle_count: int
    task_count: int
    task_min_obstacle_clearance: float
    task_p05_obstacle_clearance: float
    task_clearance_conflict: bool
    system_total_time: float
    success_rate: float
    fallback_rate: float
    mean_length_ratio: float | None
    p95_length_ratio: float | None
    mean_min_clearance: float
    p05_min_clearance: float
    mean_curvature_violation_ratio: float
    p95_curvature_violation_ratio: float
    mean_curvature_jump_p95: float
    mean_max_initial_turn_delta_rad: float
    mean_heading_sign_flip_count: float
    p95_heading_sign_flip_count: float
    mean_oscillation_energy_rad: float
    mean_task_joint_turn_delta_rad: float
    p95_task_joint_turn_delta_rad: float
    max_task_joint_turn_delta_rad: float
    mean_segment_dubins_usage_rate: float
    mean_guard_fallback_rate: float
    mean_dubins_length_ratio: float
    pqi_score: float


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one_scenario(
    *,
    scenario_file: str,
    output_dir: Path,
    cfg: SimulationConfig,
) -> ScenarioBenchmarkResult:
    stem = Path(scenario_file).stem
    session = OfflineSession(cfg=replace(cfg, scenario_file=scenario_file))
    report_path, summary_path = session.export_path_quality_report(prefix=stem)
    assert session.artifacts is not None

    scenario_dir = output_dir / stem
    scenario_dir.mkdir(parents=True, exist_ok=True)
    final_report = scenario_dir / report_path.name
    final_summary = scenario_dir / summary_path.name
    final_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    final_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    payload = _load_summary(final_summary)
    metrics = payload["metrics"]
    task_clearances = sorted(
        Point(task.position).distance(session.artifacts.world.obstacle_union)
        for task in session.result().tasks
    )
    collision_margin = float(getattr(cfg, "dubins_collision_margin", 0.0))
    return ScenarioBenchmarkResult(
        scenario_file=scenario_file,
        report_path=str(final_report),
        summary_path=str(final_summary),
        vehicle_count=int(payload["vehicle_count"]),
        task_count=int(payload["task_count"]),
        task_min_obstacle_clearance=float(task_clearances[0]) if task_clearances else 0.0,
        task_p05_obstacle_clearance=(
            float(task_clearances[max(0, int(0.05 * (len(task_clearances) - 1)))]) if task_clearances else 0.0
        ),
        task_clearance_conflict=bool(task_clearances and (task_clearances[0] + 1e-9 < collision_margin)),
        system_total_time=float(payload["system_total_time"]),
        success_rate=float(metrics["success_rate"]),
        fallback_rate=float(metrics["fallback_rate"]),
        mean_length_ratio=(None if metrics["mean_length_ratio"] is None else float(metrics["mean_length_ratio"])),
        p95_length_ratio=(None if metrics["p95_length_ratio"] is None else float(metrics["p95_length_ratio"])),
        mean_min_clearance=float(metrics["mean_min_clearance"]),
        p05_min_clearance=float(metrics["p05_min_clearance"]),
        mean_curvature_violation_ratio=float(metrics["mean_curvature_violation_ratio"]),
        p95_curvature_violation_ratio=float(metrics["p95_curvature_violation_ratio"]),
        mean_curvature_jump_p95=float(metrics["mean_curvature_jump_p95"]),
        mean_max_initial_turn_delta_rad=float(metrics["mean_max_initial_turn_delta_rad"]),
        mean_heading_sign_flip_count=float(metrics["mean_heading_sign_flip_count"]),
        p95_heading_sign_flip_count=float(metrics["p95_heading_sign_flip_count"]),
        mean_oscillation_energy_rad=float(metrics["mean_oscillation_energy_rad"]),
        mean_task_joint_turn_delta_rad=float(metrics.get("mean_task_joint_turn_delta_rad", 0.0)),
        p95_task_joint_turn_delta_rad=float(metrics.get("p95_task_joint_turn_delta_rad", 0.0)),
        max_task_joint_turn_delta_rad=float(metrics.get("max_task_joint_turn_delta_rad", 0.0)),
        mean_segment_dubins_usage_rate=float(metrics.get("mean_segment_dubins_usage_rate", 0.0)),
        mean_guard_fallback_rate=float(metrics.get("mean_guard_fallback_rate", 0.0)),
        mean_dubins_length_ratio=float(metrics.get("mean_dubins_length_ratio", 0.0)),
        pqi_score=float(metrics["pqi_score"]),
    )


def run_benchmark(
    *,
    scenario_files: list[str],
    output_dir: str | Path = "outputs/path_quality_benchmark",
    cfg: SimulationConfig = DEFAULT_CONFIG,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_results = [
        _run_one_scenario(scenario_file=scenario_file, output_dir=out_dir, cfg=cfg)
        for scenario_file in scenario_files
    ]

    scores = [item.pqi_score for item in scenario_results]
    success_rates = [item.success_rate for item in scenario_results]
    fallback_rates = [item.fallback_rate for item in scenario_results]
    violation_rates = [item.mean_curvature_violation_ratio for item in scenario_results]
    dubins_usage_rates = [item.mean_segment_dubins_usage_rate for item in scenario_results]
    guard_fallback_rates = [item.mean_guard_fallback_rate for item in scenario_results]
    dubins_length_ratios = [item.mean_dubins_length_ratio for item in scenario_results]

    aggregate = {
        "scenario_count": len(scenario_results),
        "mean_pqi_score": (sum(scores) / len(scores)) if scores else 0.0,
        "min_pqi_score": min(scores) if scores else 0.0,
        "mean_success_rate": (sum(success_rates) / len(success_rates)) if success_rates else 0.0,
        "max_fallback_rate": max(fallback_rates) if fallback_rates else 0.0,
        "max_mean_curvature_violation_ratio": max(violation_rates) if violation_rates else 0.0,
        "mean_segment_dubins_usage_rate": (sum(dubins_usage_rates) / len(dubins_usage_rates)) if dubins_usage_rates else 0.0,
        "max_mean_guard_fallback_rate": max(guard_fallback_rates) if guard_fallback_rates else 0.0,
        "mean_dubins_length_ratio": (sum(dubins_length_ratios) / len(dubins_length_ratios)) if dubins_length_ratios else 0.0,
    }

    payload = {
        "scenarios": [asdict(item) for item in scenario_results],
        "aggregate": aggregate,
        "config": {
            "use_dubins_hybrid": bool(cfg.use_dubins_hybrid),
            "force_astar_only": bool(cfg.force_astar_only),
            "dubins_sample_step": float(cfg.dubins_sample_step),
            "dubins_collision_margin": float(cfg.dubins_collision_margin),
            "goal_heading_tolerance_rad": float(cfg.goal_heading_tolerance_rad),
            "goal_heading_num_samples": int(cfg.goal_heading_num_samples),
            "online_max_initial_turn_rad": float(cfg.online_max_initial_turn_rad),
        },
    }
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run complex-scenario Dubins path quality benchmark.")
    parser.add_argument(
        "--scenario-file",
        action="append",
        dest="scenario_files",
        default=None,
        help="scenario file to benchmark; can be passed multiple times",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/path_quality_benchmark",
        help="directory for benchmark outputs",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    scenario_files = args.scenario_files or list(DEFAULT_SCENARIOS)
    summary_path = run_benchmark(
        scenario_files=scenario_files,
        output_dir=args.output_dir,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
