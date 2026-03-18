from dataclasses import dataclass, field
from typing import List, Optional, Tuple


Point2D = Tuple[float, float]


@dataclass
class Task:
    id: int
    position: Point2D
    demand: int
    status: str = "unassigned"
    assigned_vehicle: Optional[int] = None


@dataclass
class Vehicle:
    id: int
    start_pos: Point2D
    heading: float
    speed: float
    max_omega: float
    capacity: int
    remaining_capacity: int
    task_sequence: List[int] = field(default_factory=list)

    # Runtime states used by static assignment + planning
    current_pos: Point2D = (0.0, 0.0)
    current_heading: float = 0.0
    route_points: List[Point2D] = field(default_factory=list)
    history_points: List[Point2D] = field(default_factory=list)
    route_length: float = 0.0

    # Online runtime fields
    active_task_id: Optional[int] = None
    active_goal_heading: Optional[float] = None
    path_cursor: int = 0
    distance_to_next_waypoint: float = 0.0
    is_moving: bool = False
    last_replan_time: float = -1.0

    def reset_runtime_state(self) -> None:
        self.remaining_capacity = self.capacity
        self.task_sequence = []
        self.current_pos = self.start_pos
        self.current_heading = self.heading
        self.route_points = [self.start_pos]
        self.history_points = [self.start_pos]
        self.route_length = 0.0
        self.active_task_id = None
        self.active_goal_heading = None
        self.path_cursor = 0
        self.distance_to_next_waypoint = 0.0
        self.is_moving = False
        self.last_replan_time = -1.0
