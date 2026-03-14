from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from .map_utils import WorldMap


GridPoint = Tuple[int, int]
Point2D = Tuple[float, float]


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
