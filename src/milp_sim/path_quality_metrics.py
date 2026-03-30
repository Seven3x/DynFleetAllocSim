from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Sequence

from shapely.geometry import Point

from .cost_estimator import wrap_to_pi
from .entities import Point2D
from .map_utils import WorldMap
from .path_postprocess import polyline_is_clear


@dataclass
class PathQualityMetrics:
    path_length: float
    straight_distance: float
    astar_length: float | None
    length_ratio: float | None
    euclid_ratio: float | None
    collision_free: bool
    min_clearance: float
    mean_clearance: float
    p05_clearance: float
    sample_count: int
    max_initial_turn_delta_rad: float
    heading_jump_p95_rad: float
    heading_sign_flip_count: int
    oscillation_energy_rad: float
    mean_task_joint_turn_delta_rad: float
    max_task_joint_turn_delta_rad: float
    curvature_abs_mean: float
    curvature_abs_p95: float
    curvature_jump_p95: float
    curvature_violation_ratio: float
    dubins_ratio: float | None = None
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PathQualitySummary:
    path_count: int
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
    pqi_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _polyline_length(points: Sequence[Point2D]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    return total


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-9 or (not math.isfinite(denominator)):
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = max(0.0, min(1.0, q)) * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    t = pos - lo
    return float(values[lo] * (1.0 - t) + values[hi] * t)


def _resample_polyline(points: Sequence[Point2D], max_step: float) -> List[Point2D]:
    if len(points) <= 1:
        return list(points)
    step = max(1e-3, float(max_step))
    out: List[Point2D] = [points[0]]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len <= 1e-9:
            continue
        n = max(1, int(math.ceil(seg_len / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    deduped: List[Point2D] = [out[0]]
    for point in out[1:]:
        if math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) > 1e-9:
            deduped.append(point)
    return deduped


def _segment_headings(points: Sequence[Point2D]) -> tuple[list[float], list[float]]:
    headings: list[float] = []
    seg_lengths: list[float] = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        ds = math.hypot(dx, dy)
        if ds <= 1e-9:
            continue
        headings.append(math.atan2(dy, dx))
        seg_lengths.append(ds)
    return headings, seg_lengths


def _clearance_to_boundary(world: WorldMap, point: Point2D) -> float:
    x, y = point
    return min(x, y, world.width - x, world.height - y)


def _sample_clearances(world: WorldMap, points: Sequence[Point2D]) -> list[float]:
    clearances: list[float] = []
    obstacle_union = world.obstacle_union
    obstacle_empty = getattr(obstacle_union, "is_empty", True)
    for point in points:
        boundary_clearance = _clearance_to_boundary(world, point)
        if obstacle_empty:
            obstacle_clearance = float("inf")
        else:
            obstacle_clearance = Point(point).distance(obstacle_union)
        clearances.append(min(boundary_clearance, obstacle_clearance))
    return clearances


def _task_joint_turn_deltas(
    original_points: Sequence[Point2D],
    resampled_points: Sequence[Point2D],
    task_waypoint_indices: Sequence[int] | None,
) -> list[float]:
    if not task_waypoint_indices or len(original_points) < 3 or len(resampled_points) < 3:
        return []
    task_points: list[Point2D] = []
    for idx in task_waypoint_indices:
        if idx <= 0 or idx >= len(original_points) - 1:
            continue
        task_points.append(original_points[idx])
    deltas: list[float] = []
    search_start = 1
    for task_point in task_points:
        found_idx: int | None = None
        for idx in range(search_start, len(resampled_points) - 1):
            if (
                abs(resampled_points[idx][0] - task_point[0]) <= 1e-9
                and abs(resampled_points[idx][1] - task_point[1]) <= 1e-9
            ):
                found_idx = idx
                search_start = idx + 1
                break
        if found_idx is None or found_idx <= 0 or found_idx >= len(resampled_points) - 1:
            continue
        h_in = math.atan2(
            resampled_points[found_idx][1] - resampled_points[found_idx - 1][1],
            resampled_points[found_idx][0] - resampled_points[found_idx - 1][0],
        )
        h_out = math.atan2(
            resampled_points[found_idx + 1][1] - resampled_points[found_idx][1],
            resampled_points[found_idx + 1][0] - resampled_points[found_idx][0],
        )
        deltas.append(abs(wrap_to_pi(h_out - h_in)))
    return deltas


def evaluate_path_quality(
    *,
    world: WorldMap,
    points: Sequence[Point2D],
    turn_radius: float,
    sample_step: float = 0.25,
    start_heading: float | None = None,
    astar_length: float | None = None,
    dubins_ratio: float | None = None,
    fallback_used: bool = False,
    fallback_reason: str = "",
    collision_margin: float = 0.0,
    task_waypoint_indices: Sequence[int] | None = None,
) -> PathQualityMetrics:
    resampled = _resample_polyline(points, max_step=sample_step)
    path_length = _polyline_length(resampled)
    straight_distance = 0.0
    if len(resampled) >= 2:
        straight_distance = math.hypot(
            resampled[-1][0] - resampled[0][0],
            resampled[-1][1] - resampled[0][1],
        )

    headings, seg_lengths = _segment_headings(resampled)
    signed_heading_deltas = [wrap_to_pi(headings[i + 1] - headings[i]) for i in range(len(headings) - 1)]
    heading_jumps = [abs(delta) for delta in signed_heading_deltas]
    wiggle_threshold = math.radians(3.0)
    heading_sign_flip_count = 0
    prev_sign = 0
    for delta in signed_heading_deltas:
        if abs(delta) <= wiggle_threshold:
            continue
        sign = 1 if delta > 0.0 else -1
        if prev_sign != 0 and sign != prev_sign:
            heading_sign_flip_count += 1
        prev_sign = sign
    oscillation_energy = sum(abs(delta) for delta in signed_heading_deltas if abs(delta) > wiggle_threshold)
    task_joint_turn_deltas = sorted(_task_joint_turn_deltas(points, resampled, task_waypoint_indices))

    curvatures: list[float] = []
    for i, delta in enumerate(signed_heading_deltas):
        avg_ds = 0.5 * (seg_lengths[i] + seg_lengths[i + 1])
        if avg_ds <= 1e-9:
            continue
        curvatures.append(delta / avg_ds)
    curvature_abs = sorted(abs(value) for value in curvatures)
    curvature_jumps = sorted(abs(curvatures[i + 1] - curvatures[i]) for i in range(len(curvatures) - 1))

    max_curvature = float("inf")
    if turn_radius > 1e-9:
        max_curvature = 1.0 / turn_radius
    violations = [value for value in curvatures if abs(value) > max_curvature + 1e-9]

    clearances = sorted(_sample_clearances(world, resampled)) if resampled else [0.0]
    collision_free = polyline_is_clear(world, list(resampled), margin=collision_margin)

    max_initial_turn_delta = 0.0
    if start_heading is not None and headings:
        max_initial_turn_delta = abs(wrap_to_pi(headings[0] - start_heading))

    return PathQualityMetrics(
        path_length=path_length,
        straight_distance=straight_distance,
        astar_length=astar_length,
        length_ratio=_safe_ratio(path_length, astar_length) if astar_length is not None else None,
        euclid_ratio=_safe_ratio(path_length, straight_distance),
        collision_free=collision_free,
        min_clearance=clearances[0],
        mean_clearance=sum(clearances) / len(clearances),
        p05_clearance=_percentile(clearances, 0.05),
        sample_count=len(resampled),
        max_initial_turn_delta_rad=max_initial_turn_delta,
        heading_jump_p95_rad=_percentile(sorted(heading_jumps), 0.95),
        heading_sign_flip_count=heading_sign_flip_count,
        oscillation_energy_rad=oscillation_energy,
        mean_task_joint_turn_delta_rad=(
            (sum(task_joint_turn_deltas) / len(task_joint_turn_deltas)) if task_joint_turn_deltas else 0.0
        ),
        max_task_joint_turn_delta_rad=(task_joint_turn_deltas[-1] if task_joint_turn_deltas else 0.0),
        curvature_abs_mean=(sum(curvature_abs) / len(curvature_abs)) if curvature_abs else 0.0,
        curvature_abs_p95=_percentile(curvature_abs, 0.95),
        curvature_jump_p95=_percentile(curvature_jumps, 0.95),
        curvature_violation_ratio=(len(violations) / len(curvatures)) if curvatures else 0.0,
        dubins_ratio=dubins_ratio,
        fallback_used=bool(fallback_used),
        fallback_reason=str(fallback_reason),
    )


def summarize_path_quality(metrics: Iterable[PathQualityMetrics]) -> PathQualitySummary:
    items = list(metrics)
    if not items:
        raise ValueError("metrics must not be empty")

    success_rate = sum(1 for item in items if item.collision_free) / len(items)
    fallback_rate = sum(1 for item in items if item.fallback_used) / len(items)
    length_ratios = sorted(item.length_ratio for item in items if item.length_ratio is not None)
    min_clearances = sorted(item.min_clearance for item in items)
    curvature_violation_ratios = sorted(item.curvature_violation_ratio for item in items)
    curvature_jump_p95_values = [item.curvature_jump_p95 for item in items]
    initial_turn_values = [item.max_initial_turn_delta_rad for item in items]
    sign_flip_counts = sorted(float(item.heading_sign_flip_count) for item in items)
    oscillation_energy_values = [item.oscillation_energy_rad for item in items]
    task_joint_turn_values = sorted(item.mean_task_joint_turn_delta_rad for item in items)
    max_task_joint_turn_values = [item.max_task_joint_turn_delta_rad for item in items]

    feasibility = max(0.0, min(1.0, success_rate)) * max(0.0, 1.0 - fallback_rate)
    mean_violation = sum(curvature_violation_ratios) / len(curvature_violation_ratios)
    mean_turn = sum(initial_turn_values) / len(initial_turn_values)
    mean_jump = sum(curvature_jump_p95_values) / len(curvature_jump_p95_values)
    mean_flip_count = sum(sign_flip_counts) / len(sign_flip_counts)
    mean_oscillation_energy = sum(oscillation_energy_values) / len(oscillation_energy_values)
    mean_task_joint_turn = sum(task_joint_turn_values) / len(task_joint_turn_values)
    mean_max_task_joint_turn = sum(max_task_joint_turn_values) / len(max_task_joint_turn_values)
    smoothness = max(
        0.0,
        1.0
        - min(
            1.0,
            0.35 * mean_violation
            + 0.25 * (mean_turn / math.pi)
            + 0.15 * mean_jump
            + 0.15 * min(1.0, mean_flip_count / 6.0)
            + 0.07 * min(1.0, mean_oscillation_energy / (2.0 * math.pi))
            + 0.03 * min(1.0, mean_max_task_joint_turn / (0.75 * math.pi)),
        ),
    )

    if length_ratios:
        mean_length_ratio = sum(length_ratios) / len(length_ratios)
        p95_length_ratio = _percentile(length_ratios, 0.95)
        efficiency = max(0.0, 1.0 - min(1.0, max(0.0, mean_length_ratio - 1.0) / 0.5))
    else:
        mean_length_ratio = None
        p95_length_ratio = None
        efficiency = 1.0

    online_stability = 1.0
    pqi_score = 100.0 * (
        0.35 * feasibility
        + 0.30 * smoothness
        + 0.20 * efficiency
        + 0.15 * online_stability
    )

    return PathQualitySummary(
        path_count=len(items),
        success_rate=success_rate,
        fallback_rate=fallback_rate,
        mean_length_ratio=mean_length_ratio,
        p95_length_ratio=p95_length_ratio,
        mean_min_clearance=sum(min_clearances) / len(min_clearances),
        p05_min_clearance=_percentile(min_clearances, 0.05),
        mean_curvature_violation_ratio=mean_violation,
        p95_curvature_violation_ratio=_percentile(curvature_violation_ratios, 0.95),
        mean_curvature_jump_p95=mean_jump,
        mean_max_initial_turn_delta_rad=mean_turn,
        mean_heading_sign_flip_count=mean_flip_count,
        p95_heading_sign_flip_count=_percentile(sign_flip_counts, 0.95),
        mean_oscillation_energy_rad=mean_oscillation_energy,
        mean_task_joint_turn_delta_rad=mean_task_joint_turn,
        p95_task_joint_turn_delta_rad=_percentile(task_joint_turn_values, 0.95),
        max_task_joint_turn_delta_rad=max(max_task_joint_turn_values) if max_task_joint_turn_values else 0.0,
        pqi_score=pqi_score,
    )
