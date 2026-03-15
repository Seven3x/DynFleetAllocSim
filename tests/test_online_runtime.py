import unittest

from milp_sim.config import SimulationConfig
from milp_sim.dynamic_events import generate_new_task
from milp_sim.session import SimulationSession


class TestOnlineRuntime(unittest.TestCase):
    def setUp(self) -> None:
        cfg = SimulationConfig(seed=7, online_dt=0.5, online_replan_period_s=2.0)
        self.session = SimulationSession(cfg=cfg)

    def _find_square_points(self, side: float = 2.5) -> list[tuple[float, float]]:
        assert self.session.artifacts is not None
        world = self.session.artifacts.world
        cfg = self.session.cfg
        half = side / 2.0
        for xi in range(8, int(cfg.map_width) - 8, 3):
            for yi in range(8, int(cfg.map_height) - 8, 3):
                cx = float(xi)
                cy = float(yi)
                pts = [
                    (cx - half, cy - half),
                    (cx + half, cy - half),
                    (cx + half, cy + half),
                    (cx - half, cy + half),
                ]
                if not all(world.point_is_free(p, clearance=cfg.vehicle_radius + cfg.safety_margin + 0.1) for p in pts):
                    continue
                try:
                    self.session._validate_obstacle_polygon(pts)
                    return pts
                except Exception:
                    continue
        raise AssertionError("failed to find free square obstacle region for test")

    def test_soft_preempt_threshold_boundary(self) -> None:
        self.assertFalse(SimulationSession.soft_preempt_passes(10.0, 9.01, 0.10))
        self.assertTrue(SimulationSession.soft_preempt_passes(10.0, 9.0, 0.10))

    def test_event_queue_sorted_by_time(self) -> None:
        self.session.start_online()
        self.session.schedule_event(at_time=5.0, event_type="cancel_task", payload={"task_id": 0})
        self.session.schedule_event(at_time=1.0, event_type="cancel_task", payload={"task_id": 1})
        self.session.schedule_event(at_time=3.0, event_type="cancel_task", payload={"task_id": 2})
        snap = self.session.runtime_snapshot()
        times = [e.time_s for e in snap.pending_events]
        self.assertEqual(times, sorted(times))

    def test_tick_advances_time_and_moves_vehicle(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None
        before = {v.id: v.current_pos for v in self.session.engine.vehicles}
        self.session.tick(n=4)
        after = {v.id: v.current_pos for v in self.session.engine.vehicles}

        self.assertGreater(self.session.sim_time, 0.0)
        moved = any(before[k] != after[k] for k in before)
        self.assertTrue(moved)

    def test_in_progress_and_completed_states_exist(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        self.session.tick(n=1)
        status1 = {t.status for t in self.session.engine.tasks}
        self.assertIn("in_progress", status1)

        self.session.tick(n=220)
        status2 = {t.status for t in self.session.engine.tasks}
        self.assertIn("completed", status2)

    def test_add_and_remove_obstacle_online_events(self) -> None:
        self.session.start_online()
        assert self.session.artifacts is not None
        before = len(self.session.artifacts.world.obstacles)

        points = self._find_square_points(side=2.5)
        self.session.schedule_event(at_time=self.session.sim_time, event_type="add_obstacle", payload={"points": points})
        self.session.tick(n=0)
        mid = len(self.session.artifacts.world.obstacles)
        self.assertEqual(mid, before + 1)

        self.session.schedule_event(
            at_time=self.session.sim_time,
            event_type="remove_obstacle",
            payload={"obstacle_idx": mid - 1},
        )
        self.session.tick(n=0)
        after = len(self.session.artifacts.world.obstacles)
        self.assertEqual(after, before)

    def test_inflight_add_task_event_applies(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None
        assert self.session.artifacts is not None

        before = len(self.session.engine.tasks)
        task = generate_new_task(
            world=self.session.artifacts.world,
            cfg=self.session.cfg,
            rng=self.session.rng,
            task_id=self.session._next_task_id(),
        )
        self.session.schedule_event(at_time=self.session.sim_time + 0.5, event_type="add_task", payload={"task": task})
        self.session.tick(n=2)
        self.assertGreater(len(self.session.engine.tasks), before)


if __name__ == "__main__":
    unittest.main()
