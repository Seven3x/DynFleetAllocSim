from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from .map_utils import WorldMap


GridPoint = Tuple[int, int]
Point2D = Tuple[float, float]
Pose2D = Tuple[float, float, float]
HybridState = Tuple[int, int, int]


@dataclass
class HybridPlanDiagnostics:
    reason: str = ""
    reason_tags: Tuple[str, ...] = ()
    debug_trace: str = ""
    expansions: int = 0
    used_connector: bool = False
    connector_mode: str = ""


class AStarPlanner:
    def __init__(
        self,
        world: WorldMap,
        resolution: float,
        inflation_radius: float,
        connect_diagonal: bool = True,
    ) -> None:
        self.world = world
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.connect_diagonal = connect_diagonal

        self.nx = int(round(world.width / resolution)) + 1
        self.ny = int(round(world.height / resolution)) + 1
        self.occ_grid = np.zeros((self.ny, self.nx), dtype=np.uint8)

        inflated_obs = world.obstacle_union.buffer(inflation_radius)

        for iy in range(self.ny):
            y = iy * resolution
            for ix in range(self.nx):
                x = ix * resolution
                pt = Point(x, y)
                if not world.boundary_polygon.contains(pt) and not world.boundary_polygon.touches(pt):
                    self.occ_grid[iy, ix] = 1
                    continue
                if inflated_obs.is_empty:
                    continue
                if pt.within(inflated_obs):
                    self.occ_grid[iy, ix] = 1

        if connect_diagonal:
            self.neighbors = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._hybrid_cache: Dict[Tuple, Tuple[Tuple[Point2D, ...], float]] = {}
        self._hybrid_cache_max: int = 4096
        self._grid_heuristic_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._grid_heuristic_cache_max: int = 64

    def world_to_grid(self, p: Point2D) -> GridPoint:
        x, y = p
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy

    def grid_to_world(self, g: GridPoint) -> Point2D:
        ix, iy = g
        return ix * self.resolution, iy * self.resolution

    def _in_bounds(self, g: GridPoint) -> bool:
        ix, iy = g
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def _is_free(self, g: GridPoint) -> bool:
        ix, iy = g
        return self.occ_grid[iy, ix] == 0

    def _nearest_free(self, src: GridPoint) -> Optional[GridPoint]:
        if self._in_bounds(src) and self._is_free(src):
            return src

        q = deque([src])
        visited = {src}

        while q:
            node = q.popleft()
            for dx, dy in self.neighbors:
                nxt = (node[0] + dx, node[1] + dy)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt):
                    continue
                if self._is_free(nxt):
                    return nxt
                visited.add(nxt)
                q.append(nxt)
        return None

    @staticmethod
    def _heuristic(a: GridPoint, b: GridPoint) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _step_cost(self, a: GridPoint, b: GridPoint) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        while angle > math.pi:
            angle -= 2.0 * math.pi
        return angle

    def _point_is_free_on_grid(self, p: Point2D) -> bool:
        if not self.world.point_in_bounds(p):
            return False
        x, y = p
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return False
        return self.occ_grid[iy, ix] == 0

    @staticmethod
    def _yaw_to_bin(yaw: float, heading_bins: int) -> int:
        wrapped = yaw % (2.0 * math.pi)
        step = (2.0 * math.pi) / max(heading_bins, 1)
        # Use floor-style quantization for stable state hashing.
        # `round` can collapse nearby turning primitives into the same heading
        # bin as the straight primitive (especially with large turn radius),
        # which may make hybrid search appear unreachable in open space.
        return int(wrapped / step) % heading_bins

    def _pose_to_hybrid_state(self, pose: Pose2D, heading_bins: int) -> HybridState:
        x, y, yaw = pose
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        return ix, iy, self._yaw_to_bin(yaw, heading_bins)

    def _simulate_motion_primitive(
        self,
        start_pose: Pose2D,
        curvature: float,
        step_size: float,
        direction: float,
        sample_step: float,
    ) -> Tuple[List[Point2D], Pose2D]:
        x0, y0, yaw0 = start_pose
        n = max(1, int(math.ceil(step_size / max(sample_step, 1e-6))))
        samples: List[Point2D] = []
        end_yaw = yaw0

        for i in range(1, n + 1):
            s = direction * step_size * (i / n)
            if abs(curvature) <= 1e-9:
                x = x0 + s * math.cos(yaw0)
                y = y0 + s * math.sin(yaw0)
                yaw = yaw0
            else:
                dtheta = curvature * s
                radius = 1.0 / curvature
                x = x0 + radius * (math.sin(yaw0 + dtheta) - math.sin(yaw0))
                y = y0 - radius * (math.cos(yaw0 + dtheta) - math.cos(yaw0))
                yaw = yaw0 + dtheta
            samples.append((x, y))
            end_yaw = yaw

        end_pose = (samples[-1][0], samples[-1][1], self._wrap_to_pi(end_yaw))
        return samples, end_pose

    def _primitive_is_free(self, samples: List[Point2D]) -> bool:
        for p in samples:
            if not self._point_is_free_on_grid(p):
                return False
        return True

    def _segment_is_free(self, a: Point2D, b: Point2D, sample_step: float) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        if dist <= 1e-9:
            return self._point_is_free_on_grid(a) and self._point_is_free_on_grid(b)
        n = max(1, int(math.ceil(dist / max(sample_step, 1e-6))))
        for i in range(n + 1):
            t = i / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if not self._point_is_free_on_grid((x, y)):
                return False
        return True

    def _grid_heuristic_get(self, goal_grid: GridPoint) -> np.ndarray:
        cached = self._grid_heuristic_cache.get(goal_grid)
        if cached is not None:
            del self._grid_heuristic_cache[goal_grid]
            self._grid_heuristic_cache[goal_grid] = cached
            return cached

        dist = np.full((self.ny, self.nx), np.inf, dtype=np.float64)
        if not self._in_bounds(goal_grid) or not self._is_free(goal_grid):
            self._grid_heuristic_cache[goal_grid] = dist
            return dist

        gx, gy = goal_grid
        dist[gy, gx] = 0.0
        heap: List[Tuple[float, int, int]] = [(0.0, gx, gy)]
        while heap:
            d, cx, cy = heapq.heappop(heap)
            if d > dist[cy, cx] + 1e-12:
                continue
            for dx, dy in self.neighbors:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= self.nx or ny < 0 or ny >= self.ny:
                    continue
                if self.occ_grid[ny, nx] != 0:
                    continue
                w = math.hypot(dx, dy) * self.resolution
                nd = d + w
                if nd + 1e-12 >= dist[ny, nx]:
                    continue
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny))

        self._grid_heuristic_cache[goal_grid] = dist
        while len(self._grid_heuristic_cache) > self._grid_heuristic_cache_max:
            oldest_key = next(iter(self._grid_heuristic_cache))
            del self._grid_heuristic_cache[oldest_key]
        return dist

    def _hybrid_cache_get(self, key: Tuple) -> Optional[Tuple[List[Point2D], float]]:
        cached = self._hybrid_cache.get(key)
        if cached is None:
            return None
        path_tuple, length = cached
        # Refresh LRU order.
        del self._hybrid_cache[key]
        self._hybrid_cache[key] = cached
        return list(path_tuple), float(length)

    def _hybrid_cache_set(self, key: Tuple, path: List[Point2D], length: float) -> None:
        if key in self._hybrid_cache:
            del self._hybrid_cache[key]
        self._hybrid_cache[key] = (tuple(path), float(length))
        while len(self._hybrid_cache) > self._hybrid_cache_max:
            oldest_key = next(iter(self._hybrid_cache))
            del self._hybrid_cache[oldest_key]

    @staticmethod
    def _polyline_length(points: List[Point2D]) -> float:
        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        return total

    @staticmethod
    def _dedupe_points(points: List[Point2D]) -> List[Point2D]:
        if not points:
            return []
        out = [points[0]]
        for p in points[1:]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
                out.append(p)
        return out

    def _trace_hybrid_points(
        self,
        goal_key: HybridState,
        parent: Dict[HybridState, HybridState],
        poses: Dict[HybridState, Pose2D],
    ) -> List[Point2D]:
        trace_keys: List[HybridState] = [goal_key]
        cursor = goal_key
        while cursor in parent:
            cursor = parent[cursor]
            trace_keys.append(cursor)
        trace_keys.reverse()
        return [(poses[key][0], poses[key][1]) for key in trace_keys]

    def _stitch_requested_endpoints(
        self,
        points: List[Point2D],
        requested_start: Point2D,
        requested_goal: Point2D,
        connect_step: float,
    ) -> List[Point2D]:
        points = self._dedupe_points(points)
        if not points:
            return []

        if math.hypot(points[0][0] - requested_start[0], points[0][1] - requested_start[1]) > 1e-6:
            if not self._segment_is_free(requested_start, points[0], connect_step):
                return []
            points = [requested_start] + points
        else:
            points[0] = requested_start

        if math.hypot(points[-1][0] - requested_goal[0], points[-1][1] - requested_goal[1]) > 1e-6:
            if not self._segment_is_free(points[-1], requested_goal, connect_step):
                return []
            points = points + [requested_goal]
        else:
            points[-1] = requested_goal
        return self._dedupe_points(points)

    def _try_local_connector(
        self,
        *,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        turn_radius: float,
        step_size: float,
        heading_bins: int,
        allow_reverse: bool,
        goal_pos_tolerance: float,
        goal_heading_tolerance: float,
        max_expansions: int,
        max_depth: int,
        connector_radius: float,
    ) -> Tuple[List[Point2D], float, HybridPlanDiagnostics]:
        mode = "reeds_shepp_like" if allow_reverse else "dubins_like"
        dist_start_to_goal = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
        if dist_start_to_goal > connector_radius + 1e-9:
            return [], float("inf"), HybridPlanDiagnostics(
                reason="near_goal_connector_failed",
                reason_tags=("near_goal_connector_failed",),
                connector_mode=mode,
            )

        sample_step = max(0.08, min(step_size * 0.5, self.resolution))
        curvature_abs = 1.0 / max(turn_radius, 1e-6)
        curvatures = tuple(curvature_abs * c for c in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0))
        directions = (1.0, -1.0) if allow_reverse else (1.0,)
        local_bins = max(heading_bins, 24)

        node_pose: List[Pose2D] = [start_pose]
        node_parent: List[int] = [-1]
        node_segment: List[List[Point2D]] = [[(start_pose[0], start_pose[1])]]
        node_depth: List[int] = [0]
        node_g: List[float] = [0.0]
        open_heap: List[Tuple[float, int, int]] = []
        push_count = 0
        best_seen: Dict[Tuple[HybridState, int], float] = {}
        start_key = self._pose_to_hybrid_state(start_pose, local_bins)
        best_seen[(start_key, 0)] = 0.0

        def heuristic(pose: Pose2D) -> float:
            dist = math.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])
            yaw_err = abs(self._wrap_to_pi(goal_pose[2] - pose[2]))
            return dist + 0.20 * turn_radius * yaw_err

        heapq.heappush(open_heap, (heuristic(start_pose), push_count, 0))
        push_count += 1

        saw_collision = False
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, node_idx = heapq.heappop(open_heap)
            pose = node_pose[node_idx]
            depth = node_depth[node_idx]
            expansions += 1

            dist_to_goal = math.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])
            yaw_to_goal = abs(self._wrap_to_pi(goal_pose[2] - pose[2]))
            if dist_to_goal <= goal_pos_tolerance and yaw_to_goal <= goal_heading_tolerance:
                path_points: List[Point2D] = []
                cursor = node_idx
                chain: List[int] = []
                while cursor >= 0:
                    chain.append(cursor)
                    cursor = node_parent[cursor]
                chain.reverse()
                for idx in chain:
                    segment = node_segment[idx]
                    if not path_points:
                        path_points.extend(segment)
                    else:
                        path_points.extend(segment[1:])
                path_points = self._dedupe_points(path_points)
                return path_points, self._polyline_length(path_points), HybridPlanDiagnostics(
                    debug_trace=f"connector:{mode}:ok",
                    expansions=expansions,
                    used_connector=True,
                    connector_mode=mode,
                )

            if depth >= max_depth:
                continue

            step_candidates: List[float] = []
            for value in (
                min(step_size, max(goal_pos_tolerance * 0.8, dist_to_goal)),
                min(step_size, max(goal_pos_tolerance * 0.8, dist_to_goal * 0.7)),
                min(step_size, max(goal_pos_tolerance * 0.8, turn_radius * 0.55)),
                min(step_size, max(goal_pos_tolerance * 0.8, abs(self._wrap_to_pi(goal_pose[2] - pose[2])) * turn_radius)),
            ):
                if value <= 1e-6:
                    continue
                if all(abs(value - existing) > 1e-6 for existing in step_candidates):
                    step_candidates.append(value)

            for direction in directions:
                for primitive_step in step_candidates:
                    turn_penalty = 0.02 * primitive_step
                    for curvature in curvatures:
                        samples, nxt_pose = self._simulate_motion_primitive(
                            start_pose=pose,
                            curvature=curvature,
                            step_size=primitive_step,
                            direction=direction,
                            sample_step=sample_step,
                        )
                        if not self._primitive_is_free(samples):
                            saw_collision = True
                            continue
                        if math.hypot(nxt_pose[0] - start_pose[0], nxt_pose[1] - start_pose[1]) > connector_radius + 1e-9:
                            continue

                        nxt_depth = depth + 1
                        state_key = self._pose_to_hybrid_state(nxt_pose, local_bins)
                        step_cost = primitive_step * (1.25 if direction < 0.0 else 1.0)
                        if abs(curvature) > 1e-9:
                            step_cost += turn_penalty
                        tentative_g = node_g[node_idx] + step_cost
                        best_key = (state_key, nxt_depth)
                        if tentative_g + 1e-9 >= best_seen.get(best_key, float("inf")):
                            continue

                        best_seen[best_key] = tentative_g
                        node_pose.append(nxt_pose)
                        node_parent.append(node_idx)
                        node_segment.append([pose[:2]] + samples)
                        node_depth.append(nxt_depth)
                        node_g.append(tentative_g)
                        score = tentative_g + heuristic(nxt_pose) + 0.1 * nxt_depth
                        heapq.heappush(open_heap, (score, push_count, len(node_pose) - 1))
                        push_count += 1

        reason = "collision_on_connector" if saw_collision else "near_goal_connector_failed"
        debug_trace = f"connector:{mode}:{reason}"
        return [], float("inf"), HybridPlanDiagnostics(
            reason=reason,
            reason_tags=(reason,),
            debug_trace=debug_trace,
            expansions=expansions,
            connector_mode=mode,
        )

    def plan_hybrid_detailed(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        turn_radius: float,
        step_size: Optional[float] = None,
        heading_bins: int = 72,
        max_expansions: int = 45000,
        goal_pos_tolerance: Optional[float] = None,
        goal_heading_tolerance: float = 0.8,
        allow_reverse: bool = False,
        reverse_penalty: float = 1.6,
        heuristic_weight: float = 1.05,
        near_goal_connector_expansions: int = 160,
        near_goal_connector_depth: int = 6,
        near_goal_connector_radius_factor: float = 2.75,
    ) -> Tuple[List[Point2D], float, HybridPlanDiagnostics]:
        heading_bins = max(8, int(heading_bins))
        max_expansions = max(1000, int(max_expansions))
        step = max(0.2, float(step_size if step_size is not None else max(self.resolution, 0.8)))
        step_min = max(0.2, min(step, 0.6 * max(self.resolution, 0.5)))
        base_sample_step = max(0.08, min(step * 0.5, self.resolution))
        pos_tol = (
            max(self.resolution, step * 1.2)
            if goal_pos_tolerance is None
            else max(self.resolution * 0.5, float(goal_pos_tolerance))
        )
        yaw_tol = max((2.0 * math.pi) / heading_bins, float(goal_heading_tolerance))
        reverse_penalty = max(1.0, float(reverse_penalty))
        heuristic_weight = max(0.2, float(heuristic_weight))
        turn_radius = max(1e-6, float(turn_radius))
        min_bins_for_turn = int(math.ceil((2.0 * math.pi) / max(step / turn_radius, 1e-3)))
        state_heading_bins = max(heading_bins, min_bins_for_turn)
        connector_radius = max(
            pos_tol * max(1.5, float(near_goal_connector_radius_factor)),
            step * max(1.0, float(near_goal_connector_radius_factor)),
            0.9 * turn_radius,
        )
        connector_expansions = max(1, int(near_goal_connector_expansions))
        connector_depth = max(1, int(near_goal_connector_depth))

        start = (float(start_pose[0]), float(start_pose[1]))
        goal = (float(goal_pose[0]), float(goal_pose[1]))
        start_grid = self._nearest_free(self.world_to_grid(start))
        goal_grid = self._nearest_free(self.world_to_grid(goal))
        if start_grid is None or goal_grid is None:
            return [], float("inf"), HybridPlanDiagnostics(
                reason="unreachable_main_search",
                reason_tags=("unreachable_main_search",),
            )
        goal_grid_dist = self._grid_heuristic_get(goal_grid)

        if not self._point_is_free_on_grid(start):
            start = self.grid_to_world(start_grid)
        if not self._point_is_free_on_grid(goal):
            goal = self.grid_to_world(goal_grid)

        start_yaw = self._wrap_to_pi(float(start_pose[2]))
        goal_yaw = self._wrap_to_pi(float(goal_pose[2]))

        cache_key = (
            int(round(start[0] * 100.0)),
            int(round(start[1] * 100.0)),
            self._yaw_to_bin(start_yaw, state_heading_bins),
            int(round(goal[0] * 100.0)),
            int(round(goal[1] * 100.0)),
            self._yaw_to_bin(goal_yaw, state_heading_bins),
            int(round(turn_radius / max(self.resolution, 1e-6) * 100.0)),
            int(round(step * 100.0)),
            int(state_heading_bins),
            int(max_expansions),
            int(round(pos_tol * 100.0)),
            int(round(yaw_tol * 1000.0)),
            bool(allow_reverse),
            int(round(reverse_penalty * 100.0)),
            int(round(heuristic_weight * 100.0)),
        )
        cached = self._hybrid_cache_get(cache_key)
        if cached is not None:
            path, length = cached
            return path, length, HybridPlanDiagnostics(debug_trace="hybrid_astar:cache")

        start_state = (start[0], start[1], start_yaw)
        start_key = self._pose_to_hybrid_state(start_state, state_heading_bins)

        g_cost: Dict[HybridState, float] = {start_key: 0.0}
        parent: Dict[HybridState, HybridState] = {}
        poses: Dict[HybridState, Pose2D] = {start_key: start_state}
        closed: set[HybridState] = set()
        open_heap: List[Tuple[float, int, HybridState]] = []
        push_count = 0

        def heuristic(pose: Pose2D) -> float:
            dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
            ix, iy = self.world_to_grid((pose[0], pose[1]))
            grid_dist = float(goal_grid_dist[iy, ix])
            base = grid_dist if math.isfinite(grid_dist) else dist
            yaw_err = abs(self._wrap_to_pi(goal_yaw - pose[2]))
            return base + 0.20 * turn_radius * yaw_err

        start_f = g_cost[start_key] + heuristic_weight * heuristic(start_state)
        heapq.heappush(open_heap, (start_f, push_count, start_key))
        push_count += 1

        best_key = start_key
        best_progress = float("inf")
        goal_key: Optional[HybridState] = None
        connector_failure: Optional[HybridPlanDiagnostics] = None

        curvature_abs = 1.0 / turn_radius
        curvature_scales = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
        curvatures = tuple(curvature_abs * c for c in curvature_scales)
        directions = (1.0, -1.0) if allow_reverse else (1.0,)

        expansions = 0
        while open_heap and expansions < max_expansions:
            _, _, current_key = heapq.heappop(open_heap)
            if current_key in closed:
                continue
            closed.add(current_key)
            expansions += 1

            pose = poses[current_key]
            dist_to_goal = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
            yaw_to_goal = abs(self._wrap_to_pi(goal_yaw - pose[2]))
            progress_score = dist_to_goal + 0.25 * turn_radius * yaw_to_goal
            if progress_score < best_progress:
                best_progress = progress_score
                best_key = current_key

            if dist_to_goal <= pos_tol and yaw_to_goal <= yaw_tol:
                goal_key = current_key
                break

            if dist_to_goal <= connector_radius + 1e-9:
                connector_points, connector_len, connector_diag = self._try_local_connector(
                    start_pose=pose,
                    goal_pose=(goal[0], goal[1], goal_yaw),
                    turn_radius=turn_radius,
                    step_size=max(step_min, min(step, dist_to_goal + pos_tol)),
                    heading_bins=state_heading_bins,
                    allow_reverse=allow_reverse,
                    goal_pos_tolerance=pos_tol,
                    goal_heading_tolerance=yaw_tol,
                    max_expansions=connector_expansions,
                    max_depth=connector_depth,
                    connector_radius=connector_radius,
                )
                if connector_points and math.isfinite(connector_len):
                    prefix = self._trace_hybrid_points(current_key, parent, poses)
                    combined = self._dedupe_points(prefix + connector_points[1:])
                    requested_start = (float(start_pose[0]), float(start_pose[1]))
                    requested_goal = (float(goal_pose[0]), float(goal_pose[1]))
                    connect_step = max(0.08, min(self.resolution, step * 0.5))
                    stitched = self._stitch_requested_endpoints(
                        combined,
                        requested_start=requested_start,
                        requested_goal=requested_goal,
                        connect_step=connect_step,
                    )
                    if stitched:
                        length = self._polyline_length(stitched)
                        self._hybrid_cache_set(cache_key, stitched, length)
                        connector_diag.debug_trace = f"hybrid_astar:{connector_diag.debug_trace}"
                        connector_diag.expansions += expansions
                        return stitched, length, connector_diag
                elif connector_failure is None or connector_failure.reason != "collision_on_connector":
                    connector_failure = connector_diag

            adaptive_step = max(
                step_min,
                min(
                    step,
                    max(step_min, dist_to_goal * 0.35),
                ),
            )
            short_step = max(step_min, adaptive_step * 0.65)
            step_candidates = [adaptive_step]
            if short_step + 1e-6 < adaptive_step:
                step_candidates.append(short_step)

            for direction in directions:
                for primitive_step in step_candidates:
                    sample_step = max(0.08, min(base_sample_step, primitive_step * 0.5, self.resolution))
                    turn_penalty = 0.02 * primitive_step
                    for curvature in curvatures:
                        samples, nxt_pose = self._simulate_motion_primitive(
                            start_pose=pose,
                            curvature=curvature,
                            step_size=primitive_step,
                            direction=direction,
                            sample_step=sample_step,
                        )
                        if not self._primitive_is_free(samples):
                            continue

                        nxt_key = self._pose_to_hybrid_state(nxt_pose, state_heading_bins)
                        if nxt_key in closed:
                            continue

                        step_cost = primitive_step * (reverse_penalty if direction < 0.0 else 1.0)
                        if abs(curvature) > 1e-9:
                            step_cost += turn_penalty
                        tentative_g = g_cost[current_key] + step_cost
                        if tentative_g + 1e-9 >= g_cost.get(nxt_key, float("inf")):
                            continue

                        parent[nxt_key] = current_key
                        g_cost[nxt_key] = tentative_g
                        poses[nxt_key] = nxt_pose
                        score = tentative_g + heuristic_weight * heuristic(nxt_pose)
                        heapq.heappush(open_heap, (score, push_count, nxt_key))
                        push_count += 1

        if goal_key is None:
            near_pose = poses.get(best_key)
            if near_pose is None:
                reason = "exceeded_expansions" if expansions >= max_expansions else "unreachable_main_search"
                return [], float("inf"), HybridPlanDiagnostics(reason=reason, reason_tags=(reason,), expansions=expansions)

            connector_points, connector_len, connector_diag = self._try_local_connector(
                start_pose=near_pose,
                goal_pose=(goal[0], goal[1], goal_yaw),
                turn_radius=turn_radius,
                step_size=max(step_min, min(step, math.hypot(near_pose[0] - goal[0], near_pose[1] - goal[1]) + pos_tol)),
                heading_bins=state_heading_bins,
                allow_reverse=allow_reverse,
                goal_pos_tolerance=pos_tol,
                goal_heading_tolerance=yaw_tol,
                max_expansions=connector_expansions,
                max_depth=connector_depth,
                connector_radius=connector_radius,
            )
            if connector_points and math.isfinite(connector_len):
                prefix = self._trace_hybrid_points(best_key, parent, poses)
                combined = self._dedupe_points(prefix + connector_points[1:])
                requested_start = (float(start_pose[0]), float(start_pose[1]))
                requested_goal = (float(goal_pose[0]), float(goal_pose[1]))
                connect_step = max(0.08, min(self.resolution, step * 0.5))
                stitched = self._stitch_requested_endpoints(
                    combined,
                    requested_start=requested_start,
                    requested_goal=requested_goal,
                    connect_step=connect_step,
                )
                if stitched:
                    length = self._polyline_length(stitched)
                    self._hybrid_cache_set(cache_key, stitched, length)
                    connector_diag.debug_trace = f"hybrid_astar:{connector_diag.debug_trace}"
                    connector_diag.expansions += expansions
                    return stitched, length, connector_diag

            reason_tags: List[str] = []
            if expansions >= max_expansions:
                reason_tags.append("exceeded_expansions")
            else:
                reason_tags.append("unreachable_main_search")
            final_connector_failure = connector_diag if connector_diag.reason else connector_failure
            if final_connector_failure is not None and final_connector_failure.reason:
                reason_tags.append(final_connector_failure.reason)
            return [], float("inf"), HybridPlanDiagnostics(
                reason=reason_tags[0],
                reason_tags=tuple(reason_tags),
                debug_trace=(
                    f"hybrid_astar:{final_connector_failure.debug_trace}"
                    if final_connector_failure is not None and final_connector_failure.debug_trace
                    else ""
                ),
                expansions=expansions + connector_diag.expansions,
                connector_mode=(
                    final_connector_failure.connector_mode
                    if final_connector_failure is not None
                    else connector_diag.connector_mode
                ),
            )

        points = self._trace_hybrid_points(goal_key, parent, poses)
        if not points:
            return [], float("inf"), HybridPlanDiagnostics(
                reason="unreachable_main_search",
                reason_tags=("unreachable_main_search",),
                expansions=expansions,
            )

        requested_start = (float(start_pose[0]), float(start_pose[1]))
        requested_goal = (float(goal_pose[0]), float(goal_pose[1]))
        connect_step = max(0.08, min(self.resolution, step * 0.5))
        points = self._stitch_requested_endpoints(
            points,
            requested_start=requested_start,
            requested_goal=requested_goal,
            connect_step=connect_step,
        )
        if not points:
            return [], float("inf"), HybridPlanDiagnostics(
                reason="near_goal_connector_failed",
                reason_tags=("near_goal_connector_failed",),
                expansions=expansions,
            )

        length = self._polyline_length(points)

        self._hybrid_cache_set(cache_key, points, length)
        return points, length, HybridPlanDiagnostics(debug_trace="hybrid_astar:ok", expansions=expansions)

    def plan_hybrid(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        turn_radius: float,
        step_size: Optional[float] = None,
        heading_bins: int = 72,
        max_expansions: int = 45000,
        goal_pos_tolerance: Optional[float] = None,
        goal_heading_tolerance: float = 0.8,
        allow_reverse: bool = False,
        reverse_penalty: float = 1.6,
        heuristic_weight: float = 1.05,
    ) -> Tuple[List[Point2D], float]:
        points, length, _ = self.plan_hybrid_detailed(
            start_pose=start_pose,
            goal_pose=goal_pose,
            turn_radius=turn_radius,
            step_size=step_size,
            heading_bins=heading_bins,
            max_expansions=max_expansions,
            goal_pos_tolerance=goal_pos_tolerance,
            goal_heading_tolerance=goal_heading_tolerance,
            allow_reverse=allow_reverse,
            reverse_penalty=reverse_penalty,
            heuristic_weight=heuristic_weight,
        )
        return points, length

    def plan(self, start: Point2D, goal: Point2D) -> Tuple[List[Point2D], float]:
        gs = self.world_to_grid(start)
        gg = self.world_to_grid(goal)

        gs = self._nearest_free(gs)
        gg = self._nearest_free(gg)

        if gs is None or gg is None:
            return [], float("inf")

        open_heap: List[Tuple[float, GridPoint]] = []
        heapq.heappush(open_heap, (self._heuristic(gs, gg), gs))

        g_score: Dict[GridPoint, float] = {gs: 0.0}
        parent: Dict[GridPoint, GridPoint] = {}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == gg:
                path_g = [current]
                while current in parent:
                    current = parent[current]
                    path_g.append(current)
                path_g.reverse()

                path_w = [self.grid_to_world(g) for g in path_g]
                if path_w:
                    path_w[0] = start
                    path_w[-1] = goal

                length = 0.0
                for i in range(len(path_w) - 1):
                    length += math.hypot(
                        path_w[i + 1][0] - path_w[i][0],
                        path_w[i + 1][1] - path_w[i][1],
                    )
                return path_w, length

            for dx, dy in self.neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._in_bounds(nxt):
                    continue
                if not self._is_free(nxt):
                    continue

                tentative = g_score[current] + self._step_cost(current, nxt)
                if tentative < g_score.get(nxt, float("inf")):
                    parent[nxt] = current
                    g_score[nxt] = tentative
                    f_score = tentative + self._heuristic(nxt, gg) * self.resolution
                    heapq.heappush(open_heap, (f_score, nxt))

        return [], float("inf")
