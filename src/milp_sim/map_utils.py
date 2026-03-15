from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


Point2D = Tuple[float, float]


@dataclass
class WorldMap:
    width: float
    height: float
    depot_polygon: Polygon
    obstacles: List[Polygon]

    def __post_init__(self) -> None:
        self.boundary_polygon = box(0.0, 0.0, self.width, self.height)
        self._obstacle_union: Optional[Polygon] = None

    @property
    def obstacle_union(self):
        if self._obstacle_union is None:
            if not self.obstacles:
                self._obstacle_union = Polygon()
            else:
                self._obstacle_union = unary_union(self.obstacles)
        return self._obstacle_union

    def invalidate_cache(self) -> None:
        self._obstacle_union = None

    def point_in_bounds(self, p: Point2D, margin: float = 0.0) -> bool:
        x, y = p
        return margin <= x <= self.width - margin and margin <= y <= self.height - margin

    def point_is_free(self, p: Point2D, clearance: float = 0.0) -> bool:
        if not self.point_in_bounds(p, margin=clearance):
            return False

        pt = Point(p)
        if clearance > 0.0:
            zone = pt.buffer(clearance)
            return not zone.intersects(self.obstacle_union)
        return not pt.within(self.obstacle_union)

    def add_obstacle(self, polygon: Polygon) -> None:
        self.obstacles.append(polygon)
        self.invalidate_cache()

    def remove_obstacle(self, idx: int) -> Polygon:
        if idx < 0 or idx >= len(self.obstacles):
            raise IndexError(f"obstacle index out of range: {idx}")
        poly = self.obstacles.pop(idx)
        self.invalidate_cache()
        return poly

    def free_space_area(self) -> float:
        return self.boundary_polygon.difference(self.obstacle_union).area
