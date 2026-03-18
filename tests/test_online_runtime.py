import math
import unittest
from unittest.mock import patch

from shapely.geometry import Point

from milp_sim.config import SimulationConfig
from milp_sim.dynamic_events import generate_new_task
from milp_sim.entities import Task
from milp_sim.session import SimulationSession


class TestOnlineRuntime(unittest.TestCase):
    def setUp(self) -> None:
        cfg = SimulationConfig(
            seed=7,
            online_dt=0.5,
            online_replan_period_s=2.0,
            online_new_task_replan_batch_size=3,
        )
        self.session = SimulationSession(cfg=cfg)

    def _new_task(self):
        assert self.session.artifacts is not None
        return generate_new_task(
            world=self.session.artifacts.world,
            cfg=self.session.cfg,
            rng=self.session.rng,
            task_id=self.session._next_task_id(),
        )

    def _add_online_task(self):
        task = self._new_task()
        self.session.add_task(
            x=task.position[0],
            y=task.position[1],
            demand=task.demand,
            task_id=task.id,
        )
        return task

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

    def _iter_free_points(self, x_start: int = 20, y_start: int = 20, step: int = 4):
        assert self.session.artifacts is not None
        world = self.session.artifacts.world
        cfg = self.session.cfg
        clearance = cfg.vehicle_radius + cfg.safety_margin + 0.2
        for xi in range(x_start, int(cfg.map_width) - 15, step):
            for yi in range(y_start, int(cfg.map_height) - 20, step):
                p = (float(xi), float(yi))
                if not world.point_is_free(p, clearance=clearance):
                    continue
                if world.depot_polygon.contains(Point(p)):
                    continue
                yield p

    def test_soft_preempt_threshold_boundary(self) -> None:
        self.assertFalse(SimulationSession.soft_preempt_passes(10.0, 9.01, 0.10))
        self.assertTrue(SimulationSession.soft_preempt_passes(10.0, 9.0, 0.10))

    def test_batch_replan_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(online_new_task_replan_batch_size=0)

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

        before = len(self.session.engine.tasks)
        task = self._new_task()
        self.session.schedule_event(at_time=self.session.sim_time + 0.5, event_type="add_task", payload={"task": task})
        self.session.tick(n=2)
        self.assertGreater(len(self.session.engine.tasks), before)

    def test_new_tasks_wait_for_batch_threshold_before_replan(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        before = len(self.session.engine.tasks)
        self._add_online_task()
        snap1 = self.session.runtime_snapshot()
        self.assertEqual(len(self.session.engine.tasks), before + 1)
        self.assertEqual(snap1.pending_new_task_replan_count, 1)
        self.assertEqual(snap1.last_replan_reason, "startup")

        self._add_online_task()
        snap2 = self.session.runtime_snapshot()
        self.assertEqual(len(self.session.engine.tasks), before + 2)
        self.assertEqual(snap2.pending_new_task_replan_count, 2)
        self.assertEqual(snap2.last_replan_reason, "startup")

        self._add_online_task()
        snap3 = self.session.runtime_snapshot()
        self.assertEqual(len(self.session.engine.tasks), before + 3)
        self.assertEqual(snap3.pending_new_task_replan_count, 0)
        self.assertIn("new_task_batch", snap3.last_replan_reason)

    def test_periodic_replan_flushes_partial_new_task_batch(self) -> None:
        self.session.start_online()

        self._add_online_task()
        snap_before = self.session.runtime_snapshot()
        self.assertEqual(snap_before.pending_new_task_replan_count, 1)
        self.assertEqual(snap_before.last_replan_reason, "startup")

        self.session.tick(n=5)
        snap_after = self.session.runtime_snapshot()
        self.assertEqual(snap_after.pending_new_task_replan_count, 0)
        self.assertIn("periodic", snap_after.last_replan_reason)

    def test_move_task_does_not_increment_new_task_batch_counter(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        self._add_online_task()
        self.assertEqual(self.session.runtime_snapshot().pending_new_task_replan_count, 1)

        moved = self._new_task()
        original = self.session.engine.tasks[0]
        self.session.move_task(task_id=original.id, x=moved.position[0], y=moved.position[1])
        snap = self.session.runtime_snapshot()
        self.assertEqual(snap.pending_new_task_replan_count, 0)
        self.assertEqual(self.session.last_replan_reason, "event")

    def test_non_batch_replan_event_clears_pending_new_task_counter(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        self._add_online_task()
        self.assertEqual(self.session.runtime_snapshot().pending_new_task_replan_count, 1)

        self.session.cancel_task(task_id=self.session.engine.tasks[0].id)
        snap = self.session.runtime_snapshot()
        self.assertEqual(snap.pending_new_task_replan_count, 0)
        self.assertEqual(snap.last_replan_reason, "event")

    def test_frame_history_restores_pending_new_task_counter(self) -> None:
        self.session.start_online()

        self._add_online_task()
        self._add_online_task()
        self.assertEqual(self.session.runtime_snapshot().pending_new_task_replan_count, 2)

        prev_snap = self.session.frame_prev()
        self.assertEqual(prev_snap.pending_new_task_replan_count, 1)

        next_snap = self.session.frame_next()
        self.assertEqual(next_snap.pending_new_task_replan_count, 2)

    def test_initial_turn_buffer_softens_sharp_replan_turn(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        vehicle = self.session.engine.vehicles[0]
        turn_radius = vehicle.speed / max(vehicle.max_omega, 1e-6)

        def fake_tail_plan(*, start_pose, goal_pose, **kwargs):
            pts = [(start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1])]
            seg_len = math.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
            return pts, seg_len, None

        found = False
        with patch("milp_sim.session.build_dubins_hybrid_path", side_effect=fake_tail_plan):
            for start in self._iter_free_points():
                goal = (start[0], start[1] + 16.0)
                try:
                    self.session._validate_task_point(goal)
                except Exception:
                    continue

                vehicle.current_pos = start
                vehicle.current_heading = 0.0
                task = Task(id=-1, position=goal, demand=1, status="unassigned")
                raw_path = [start, goal]
                raw_len = math.hypot(goal[0] - start[0], goal[1] - start[1])
                raw_delta = self.session._path_initial_turn_delta(raw_path, vehicle.current_heading)
                if raw_delta <= math.radians(75.0):
                    continue

                buffered_path, buffered_len = self.session._maybe_buffer_initial_turn_path(
                    v=vehicle,
                    task=task,
                    path=raw_path,
                    length=raw_len,
                    goal_heading=math.pi / 2.0,
                    turn_radius=turn_radius,
                )
                if len(buffered_path) <= 2:
                    continue

                buffered_delta = self.session._path_initial_turn_delta(buffered_path, vehicle.current_heading)
                self.assertLess(buffered_delta, raw_delta)
                self.assertLessEqual(buffered_delta, math.radians(30.0))
                self.assertGreater(buffered_len, raw_len)
                found = True
                break

        self.assertTrue(found, "failed to find a free-space case for initial-turn buffering")

    def test_complete_active_task_preserves_planned_goal_heading_into_next_activation(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        vehicle = self.session.engine.vehicles[0]
        first_tid = vehicle.task_sequence[0]
        second_tid = vehicle.task_sequence[1]
        first_task = self.session.engine.tasks_by_id[first_tid]
        second_task = self.session.engine.tasks_by_id[second_tid]

        vehicle.active_task_id = first_tid
        vehicle.active_goal_heading = 1.234
        vehicle.current_heading = 0.0
        vehicle.task_sequence = [first_tid, second_tid]
        first_task.status = "in_progress"
        first_task.assigned_vehicle = vehicle.id
        second_task.status = "locked"
        second_task.assigned_vehicle = vehicle.id

        seen_headings: list[float] = []

        def fake_build_active_segment(session_obj, vehicle_obj):
            seen_headings.append(vehicle_obj.current_heading)
            vehicle_obj.route_points = [vehicle_obj.current_pos]
            vehicle_obj.route_length = 0.0
            vehicle_obj.path_cursor = 0
            vehicle_obj.distance_to_next_waypoint = 0.0
            vehicle_obj.is_moving = False

        with patch.object(SimulationSession, "_build_active_segment", autospec=True, side_effect=fake_build_active_segment):
            self.session._complete_active_task(vehicle)

        self.assertEqual(first_task.status, "completed")
        self.assertEqual(vehicle.active_task_id, second_tid)
        self.assertAlmostEqual(vehicle.current_heading, 1.234, places=6)
        self.assertGreaterEqual(len(seen_headings), 1)
        self.assertAlmostEqual(seen_headings[0], 1.234, places=6)

    def test_refresh_active_tasks_keeps_clear_in_progress_route(self) -> None:
        self.session.start_online()
        assert self.session.engine is not None

        vehicle = self.session.engine.vehicles[0]
        self.assertIsNotNone(vehicle.active_task_id)
        saved_route = list(vehicle.route_points)
        saved_heading = vehicle.active_goal_heading

        for other in self.session.engine.vehicles[1:]:
            other.active_task_id = None
            other.active_goal_heading = None
            other.task_sequence = []
            other.route_points = [other.current_pos]
            other.route_length = 0.0
            other.path_cursor = 0
            other.distance_to_next_waypoint = 0.0
            other.is_moving = False

        target_calls: list[int] = []
        original_build = SimulationSession._build_active_segment

        def wrapped_build(session_obj, vehicle_obj):
            if vehicle_obj.id == vehicle.id:
                target_calls.append(vehicle_obj.id)
            return original_build(session_obj, vehicle_obj)

        with patch.object(SimulationSession, "_build_active_segment", autospec=True, side_effect=wrapped_build):
            self.session._refresh_active_tasks_and_routes()

        self.assertEqual(target_calls, [])
        self.assertEqual(vehicle.route_points, saved_route)
        self.assertEqual(vehicle.active_goal_heading, saved_heading)

    def test_reset_replay_restores_online_task_and_obstacle_actions(self) -> None:
        assert self.session.artifacts is not None
        base_tasks = len(self.session.list_tasks())
        base_obstacles = len(self.session.artifacts.world.obstacles)

        self.session.start_online()
        added = self._add_online_task()
        self.session.tick(n=2)
        obstacle_points = self._find_square_points(side=2.5)
        self.session.add_obstacle_polygon(obstacle_points)

        actions_before = self.session.replayable_user_actions()
        self.assertEqual([a.action_type for a in actions_before], ["add_task", "add_obstacle"])
        self.assertEqual([round(a.sim_time or 0.0, 6) for a in actions_before], [0.0, 1.0])

        self.session.reset(replay_last_actions=True)
        assert self.session.artifacts is not None

        self.assertTrue(self.session.online_enabled)
        self.assertFalse(self.session.online_running)
        self.assertAlmostEqual(self.session.sim_time, 1.0, places=6)
        self.assertEqual(len(self.session.list_tasks()), base_tasks + 1)
        self.assertEqual(len(self.session.artifacts.world.obstacles), base_obstacles + 1)
        replayed = self.session.engine.tasks_by_id.get(added.id)
        self.assertIsNotNone(replayed)
        assert replayed is not None
        self.assertAlmostEqual(replayed.position[0], added.position[0], places=6)
        self.assertAlmostEqual(replayed.position[1], added.position[1], places=6)

        actions_after = self.session.replayable_user_actions()
        self.assertEqual(
            [(a.action_type, round(a.sim_time or 0.0, 6)) for a in actions_after],
            [(a.action_type, round(a.sim_time or 0.0, 6)) for a in actions_before],
        )

    def test_reset_archives_actions_for_later_manual_replay(self) -> None:
        self.session.start_online()
        added = self._add_online_task()

        self.session.reset()
        self.assertFalse(self.session.online_enabled)
        archived = self.session.replayable_user_actions()
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].action_type, "add_task")

        self.session.replay_last_user_actions()
        self.assertTrue(self.session.online_enabled)
        self.assertFalse(self.session.online_running)
        replayed = self.session.engine.tasks_by_id.get(added.id)
        self.assertIsNotNone(replayed)


if __name__ == "__main__":
    unittest.main()
