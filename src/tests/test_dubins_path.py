import math
import unittest
from dataclasses import replace
from unittest.mock import patch

try:
    from shapely.geometry import LineString, box

    from milp_sim.config import DEFAULT_CONFIG
    from milp_sim.cost_estimator import heading_to_point
    from milp_sim.dubins_path import DUBINS_WORDS, build_dubins_hybrid_path, build_rsplan_connector, solve_dubins_word
    from milp_sim.entities import Task, Vehicle
    from milp_sim.map_utils import WorldMap
    from milp_sim.planner_astar import AStarPlanner, HybridPlanDiagnostics
    from milp_sim.session import SimulationSession
    from milp_sim.verification import verify_bid

    HAS_SHAPELY = True
except ModuleNotFoundError:
    HAS_SHAPELY = False


@unittest.skipUnless(HAS_SHAPELY, "shapely is required for Dubins path tests")
class TestDubinsPath(unittest.TestCase):
    def setUp(self) -> None:
        self.world = WorldMap(
            width=100.0,
            height=100.0,
            depot_polygon=box(0.0, 0.0, 10.0, 10.0),
            obstacles=[],
        )
        self.planner = AStarPlanner(
            world=self.world,
            resolution=1.0,
            inflation_radius=DEFAULT_CONFIG.vehicle_radius + DEFAULT_CONFIG.safety_margin,
            connect_diagonal=True,
        )

    def test_dubins_words_find_nonnegative_solution(self) -> None:
        alphas = [i * math.pi / 8.0 for i in range(16)]
        betas = [i * math.pi / 8.0 for i in range(16)]
        ds = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

        for word in DUBINS_WORDS:
            found = None
            for alpha in alphas:
                for beta in betas:
                    for d in ds:
                        out = solve_dubins_word(word, alpha=alpha, beta=beta, d=d)
                        if out is not None:
                            found = out
                            break
                    if found is not None:
                        break
                if found is not None:
                    break

            self.assertIsNotNone(found, msg=f"{word} did not produce a feasible sample")
            assert found is not None
            self.assertGreaterEqual(found[0], 0.0)
            self.assertGreaterEqual(found[1], 0.0)
            self.assertGreaterEqual(found[2], 0.0)

    def test_hybrid_path_preserves_endpoints(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            dubins_sample_step=0.8,
            dubins_collision_margin=1.8,
        )
        start = (12.0, 14.0, 0.2)
        goal = (85.0, 76.0, 1.0)

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=12.0,
        )

        self.assertGreater(len(points), 1)
        self.assertTrue(math.isfinite(length))
        self.assertAlmostEqual(points[0][0], start[0], places=6)
        self.assertAlmostEqual(points[0][1], start[1], places=6)
        self.assertAlmostEqual(points[-1][0], goal[0], places=6)
        self.assertAlmostEqual(points[-1][1], goal[1], places=6)
        self.assertGreaterEqual(meta.sample_count, len(points))

    def test_hybrid_astar_primary_mode_returns_feasible_path(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=False,
            hybrid_astar_fallback_to_legacy=False,
            hybrid_astar_step_size=1.0,
            hybrid_astar_max_expansions=30000,
        )
        start = (12.0, 14.0, 0.2)
        goal = (85.0, 76.0, 1.0)

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=12.0,
        )

        self.assertGreater(len(points), 2)
        self.assertTrue(math.isfinite(length))
        self.assertFalse(meta.used_fallback)
        self.assertTrue(meta.debug_trace.startswith("hybrid_astar:"))

    def test_hybrid_astar_primary_search_can_use_reverse_without_retry(self) -> None:
        world = WorldMap(
            width=30.0,
            height=30.0,
            depot_polygon=box(0.0, 0.0, 4.0, 4.0),
            obstacles=[
                box(13.55, 5.81, 18.44, 11.67),
                box(20.61, 21.40, 25.73, 28.66),
            ],
        )
        planner = AStarPlanner(
            world=world,
            resolution=1.0,
            inflation_radius=DEFAULT_CONFIG.vehicle_radius + DEFAULT_CONFIG.safety_margin,
            connect_diagonal=True,
        )
        start = (8.0, 8.0, 0.0)
        goal = (9.730830828623107, 21.71061219428649, 0.30598675032964895)

        forward_cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=False,
            hybrid_astar_fallback_to_legacy=False,
            hybrid_astar_allow_reverse=False,
            hybrid_astar_retry_on_fail=False,
            hybrid_astar_step_size=1.0,
            hybrid_astar_max_expansions=12000,
        )
        reverse_cfg = replace(
            forward_cfg,
            hybrid_astar_allow_reverse=True,
            hybrid_astar_reverse_penalty=1.22,
        )

        forward_points, forward_length, forward_meta = build_dubins_hybrid_path(
            world=world,
            cfg=forward_cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=planner,
            turn_radius=5.0,
        )
        reverse_points, reverse_length, reverse_meta = build_dubins_hybrid_path(
            world=world,
            cfg=reverse_cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=planner,
            turn_radius=5.0,
        )

        self.assertEqual(forward_points, [])
        self.assertEqual(forward_length, float("inf"))
        self.assertTrue(forward_meta.used_fallback)
        self.assertIn("unreachable_main_search", forward_meta.fallback_details)

        self.assertGreater(len(reverse_points), 2)
        self.assertTrue(math.isfinite(reverse_length))
        self.assertFalse(reverse_meta.used_fallback)
        self.assertIn("reeds_shepp_like", reverse_meta.debug_trace)

    def test_near_goal_connector_can_rescue_search_before_fallback(self) -> None:
        goal = (29.277877625513483, 47.65375351775711, -0.8170957868197246)
        weak_planner = AStarPlanner(
            world=self.world,
            resolution=1.0,
            inflation_radius=DEFAULT_CONFIG.vehicle_radius + DEFAULT_CONFIG.safety_margin,
            connect_diagonal=True,
        )
        strong_planner = AStarPlanner(
            world=self.world,
            resolution=1.0,
            inflation_radius=DEFAULT_CONFIG.vehicle_radius + DEFAULT_CONFIG.safety_margin,
            connect_diagonal=True,
        )

        weak_points, weak_length, weak_diag = weak_planner.plan_hybrid_detailed(
            start_pose=(12.0, 14.0, 0.2),
            goal_pose=goal,
            turn_radius=10.0,
            step_size=1.2,
            heading_bins=48,
            max_expansions=4000,
            goal_pos_tolerance=1.2,
            goal_heading_tolerance=0.35,
            allow_reverse=True,
            reverse_penalty=1.22,
            heuristic_weight=1.15,
            near_goal_connector_expansions=1,
            near_goal_connector_depth=1,
            near_goal_connector_radius_factor=0.3,
        )
        strong_points, strong_length, strong_diag = strong_planner.plan_hybrid_detailed(
            start_pose=(12.0, 14.0, 0.2),
            goal_pose=goal,
            turn_radius=10.0,
            step_size=1.2,
            heading_bins=48,
            max_expansions=4000,
            goal_pos_tolerance=1.2,
            goal_heading_tolerance=0.35,
            allow_reverse=True,
            reverse_penalty=1.22,
            heuristic_weight=1.15,
            near_goal_connector_expansions=160,
            near_goal_connector_depth=6,
            near_goal_connector_radius_factor=2.75,
        )

        self.assertEqual(weak_points, [])
        self.assertEqual(weak_length, float("inf"))
        self.assertIn("near_goal_connector_failed", weak_diag.reason_tags)

        self.assertGreater(len(strong_points), 2)
        self.assertTrue(math.isfinite(strong_length))
        self.assertTrue(strong_diag.used_connector)
        self.assertIn("connector:reeds_shepp_like:ok", strong_diag.debug_trace)

    def test_plain_astar_fallback_still_available_as_last_resort(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=False,
            hybrid_astar_fallback_to_legacy=True,
            hybrid_astar_direct_astar_fallback=True,
        )
        start = (12.0, 14.0, 0.2)
        goal = (85.0, 76.0, 1.0)
        fail_diag = HybridPlanDiagnostics(
            reason="unreachable_main_search",
            reason_tags=("unreachable_main_search", "near_goal_connector_failed"),
            debug_trace="hybrid_astar:primary_unreachable_main_search|near_goal_connector_failed",
        )

        with patch.object(self.planner, "plan_hybrid_detailed", return_value=([], float("inf"), fail_diag)):
            points, length, meta = build_dubins_hybrid_path(
                world=self.world,
                cfg=cfg,
                start_pose=start,
                goal_pose=goal,
                astar_planner=self.planner,
                turn_radius=12.0,
            )

        self.assertGreater(len(points), 1)
        self.assertTrue(math.isfinite(length))
        self.assertTrue(meta.used_fallback)
        self.assertEqual(meta.fallback_reason, "degraded_to_plain_astar")
        self.assertIn("unreachable_main_search", meta.fallback_details)
        self.assertIn("degraded_to_plain_astar", meta.fallback_details)
        self.assertIn("near_goal_connector_failed", meta.fallback_details)

    def test_hybrid_fallback_to_astar_when_collision_too_strict(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            enable_connector_first_planner=False,
            astar_smooth_before_dubins=False,
            dubins_sample_step=0.8,
            dubins_collision_margin=40.0,
            dubins_fallback_to_astar=True,
        )
        start = (12.0, 14.0, 0.0)
        goal = (85.0, 76.0, 0.0)

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
        )

        self.assertGreater(len(points), 1)
        self.assertTrue(math.isfinite(length))
        self.assertTrue(meta.used_fallback)
        self.assertGreater(meta.fallback_segments, 0)
        # In this strict-clearance setup, the fallback reason depends on the
        # selected legacy branch (direct Dubins vs. multi-corner fillet).
        self.assertIn(meta.fallback_reason, {"direct_collision", "corner_collision", "final_collision"})
        self.assertIn("collision", meta.fallback_details)

    def test_build_rsplan_connector_handles_missing_dependency(self) -> None:
        cfg = replace(DEFAULT_CONFIG, use_rsplan_connector=True)
        start = (12.0, 14.0, 0.2)
        goal = (18.0, 20.0, 0.7)

        with patch("milp_sim.dubins_path.rsplan", None):
            points, length, reason = build_rsplan_connector(
                start_pose=start,
                goal_pose=goal,
                turn_radius=8.0,
                world=self.world,
                cfg=cfg,
                margin=cfg.dubins_collision_margin,
                require_terminal_heading=True,
            )

        self.assertEqual(points, [])
        self.assertEqual(length, float("inf"))
        self.assertEqual(reason, "rsplan_import_unavailable")

    def test_connector_first_can_use_rsplan_primary(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=True,
            connector_use_straight_first=False,
            connector_use_reeds_shepp=False,
            connector_use_dubins=False,
            connector_use_hybrid_local_rescue=False,
            connector_use_plain_astar_fallback=False,
            hybrid_astar_allow_reverse=True,
            hybrid_astar_fallback_to_legacy=False,
            use_rsplan_connector=True,
        )
        start = (12.0, 14.0, 0.0)
        goal = (28.0, 18.0, 0.2)
        fake_path = [(12.0, 14.0), (20.0, 16.0), (28.0, 18.0)]

        with patch(
            "milp_sim.dubins_path.build_rsplan_connector",
            return_value=(fake_path, 16.5, "connector_rsplan_success"),
        ):
            points, length, meta = build_dubins_hybrid_path(
                world=self.world,
                cfg=cfg,
                start_pose=start,
                goal_pose=goal,
                astar_planner=self.planner,
                turn_radius=8.0,
            )

        self.assertEqual(points, fake_path)
        self.assertAlmostEqual(length, 16.5, places=6)
        self.assertEqual(meta.connector_type, "rsplan")
        self.assertIn("connector_rsplan_success", meta.debug_trace)

    def test_connector_first_prefers_straight_in_open_space(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=True,
            hybrid_astar_fallback_to_legacy=False,
        )
        start = (12.0, 14.0, heading_to_point((12.0, 14.0), (85.0, 76.0)))
        goal = (85.0, 76.0, heading_to_point((12.0, 14.0), (85.0, 76.0)))

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
        )

        self.assertEqual(points, [(start[0], start[1]), (goal[0], goal[1])])
        self.assertTrue(math.isfinite(length))
        self.assertFalse(meta.used_fallback)
        self.assertEqual(meta.connector_type, "straight")
        self.assertIn("connector_straight_success", meta.debug_trace)

    def test_connector_first_rejects_large_rs_detour_and_uses_hybrid_local(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=True,
            enable_connector_first_planner=True,
            use_rsplan_connector=False,
            connector_use_straight_first=False,
            connector_use_dubins=False,
            connector_use_reeds_shepp=True,
            hybrid_astar_allow_reverse=True,
            connector_max_detour_ratio_vs_reference=1.05,
            connector_max_detour_abs_vs_reference=1.0,
            hybrid_astar_fallback_to_legacy=False,
        )
        start = (12.0, 14.0, 0.0)
        goal = (28.0, 18.0, 0.2)
        long_rs = [(12.0, 14.0), (12.0, 40.0), (28.0, 18.0)]
        hybrid_path = [(12.0, 14.0), (20.0, 16.0), (28.0, 18.0)]

        with patch.object(
            self.planner,
            "plan_local_connector",
            return_value=(
                long_rs,
                120.0,
                HybridPlanDiagnostics(debug_trace="connector:reeds_shepp_like:ok", used_connector=True, connector_mode="reeds_shepp_like"),
            ),
        ), patch.object(
            self.planner,
            "plan_hybrid_detailed",
            return_value=(hybrid_path, 16.5, HybridPlanDiagnostics(debug_trace="hybrid_astar:ok")),
        ):
            points, length, meta = build_dubins_hybrid_path(
                world=self.world,
                cfg=cfg,
                start_pose=start,
                goal_pose=goal,
                astar_planner=self.planner,
                turn_radius=8.0,
            )

        self.assertEqual(points, hybrid_path)
        self.assertAlmostEqual(length, 2.0 * math.hypot(8.0, 2.0), places=6)
        self.assertEqual(meta.connector_type, "hybrid_local")
        self.assertIn("connector_rs_detour_reject", meta.debug_trace)
        self.assertIn("connector_hybrid_local_success", meta.debug_trace)

    def test_hybrid_returns_inf_when_fallback_disabled_and_collision_strict(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            enable_connector_first_planner=False,
            astar_smooth_before_dubins=False,
            dubins_sample_step=0.8,
            dubins_collision_margin=40.0,
            dubins_fallback_to_astar=False,
        )
        start = (12.0, 14.0, 0.0)
        goal = (85.0, 76.0, 0.0)

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
        )

        self.assertEqual(points, [])
        self.assertEqual(length, float("inf"))
        self.assertTrue(meta.used_fallback)

    def test_astar_shortcut_smoothing_reduces_samples_when_dubins_disabled(self) -> None:
        start = (12.0, 14.0, 0.0)
        goal = (85.0, 76.0, 0.0)
        raw_path, raw_len = self.planner.plan((start[0], start[1]), (goal[0], goal[1]))
        self.assertGreater(len(raw_path), 2)
        self.assertTrue(math.isfinite(raw_len))

        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=False,
            use_hybrid_astar=False,
            astar_smooth_before_dubins=True,
        )
        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
            astar_path=raw_path,
            astar_length=raw_len,
        )

        self.assertLess(len(points), len(raw_path))
        self.assertTrue(math.isfinite(length))
        self.assertLessEqual(length, raw_len + 1e-6)
        self.assertEqual(meta.sample_count, len(points))

    def test_two_point_smoothed_path_still_uses_dubins_curve(self) -> None:
        start = (12.0, 14.0, 0.0)
        goal = (85.0, 76.0, 1.2)
        two_point = [(start[0], start[1]), (goal[0], goal[1])]
        line_len = math.hypot(goal[0] - start[0], goal[1] - start[1])

        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            astar_smooth_before_dubins=True,
        )
        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
            astar_path=two_point,
            astar_length=line_len,
        )

        self.assertGreater(len(points), 2)
        self.assertTrue(math.isfinite(length))
        self.assertGreater(length, line_len)
        self.assertGreater(meta.dubins_segments, 0)

    def test_overlap_rejection_reports_adjacent_overlap_reason(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            enable_connector_first_planner=False,
            astar_smooth_before_dubins=False,
            dubins_fallback_to_astar=True,
            dubins_collision_margin=1.8,
        )
        start = (10.0, 10.0, 0.0)
        goal = (21.0, 30.0, math.pi / 2.0)
        path = [(10.0, 10.0), (20.0, 10.0), (21.0, 11.0), (21.0, 30.0)]
        path_len = 0.0
        for i in range(len(path) - 1):
            path_len += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])

        points, length, meta = build_dubins_hybrid_path(
            world=self.world,
            cfg=cfg,
            start_pose=start,
            goal_pose=goal,
            astar_planner=self.planner,
            turn_radius=10.0,
            astar_path=path,
            astar_length=path_len,
        )

        self.assertGreater(len(points), 1)
        self.assertTrue(math.isfinite(length))
        self.assertTrue(meta.used_fallback)
        self.assertEqual(meta.fallback_segments, 1)
        self.assertEqual(meta.fallback_reason, "adjacent_overlap")
        self.assertIn("adjacent_overlap:1", meta.fallback_details)

    def test_verify_bid_can_use_heading_candidates_to_avoid_direct_collision(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            goal_heading_tolerance_rad=0.9,
            goal_heading_num_samples=5,
            goal_heading_max_dpsi_rad=1.05,
            goal_heading_max_dpsi_slack_rad=0.12,
        )
        vehicle = Vehicle(
            id=0,
            start_pos=(12.0, 14.0),
            heading=0.0,
            speed=5.0,
            max_omega=0.5,
            capacity=10,
            remaining_capacity=10,
            current_pos=(12.0, 14.0),
            current_heading=0.0,
        )
        task = Task(id=1, position=(85.0, 76.0), demand=1)
        direct_heading = math.atan2(task.position[1] - vehicle.current_pos[1], task.position[0] - vehicle.current_pos[0])

        def fake_build_dubins_hybrid_path(
            world,
            cfg,
            start_pose,
            goal_pose,
            astar_planner,
            turn_radius,
            astar_path=None,
            astar_length=None,
        ):
            goal_heading = goal_pose[2]
            if abs(goal_heading - direct_heading) <= 1e-6:
                return (
                    [(start_pose[0], start_pose[1]), (goal_pose[0], goal_pose[1])],
                    100.0,
                    type(
                        "Meta",
                        (),
                        {
                            "used_fallback": True,
                            "fallback_details": "direct_collision:1",
                            "fallback_reason": "direct_collision",
                        },
                    )(),
                )
            return (
                [(start_pose[0], start_pose[1]), (48.0, 46.0), (goal_pose[0], goal_pose[1])],
                20.0,
                type(
                    "Meta",
                    (),
                    {
                        "used_fallback": False,
                        "fallback_details": "",
                        "fallback_reason": "",
                    },
                )(),
            )

        with patch("milp_sim.verification.build_dubins_hybrid_path", side_effect=fake_build_dubins_hybrid_path):
            result = verify_bid(
                vehicle=vehicle,
                task=task,
                c_hat=25.0,
                cfg=cfg,
                planner=self.planner,
                tasks_by_id={task.id: task},
            )

        self.assertTrue(result.passed)
        self.assertFalse(result.dubins_used_fallback)
        self.assertEqual(result.dubins_fallback_details, "")
        self.assertLess(result.c_tilde, 25.0)

    def test_verify_bid_zero_prefix_starts_from_active_task_endpoint(self) -> None:
        cfg = replace(DEFAULT_CONFIG, committed_prefix_time_weight=0.0)
        active_task = Task(id=10, position=(30.0, 40.0), demand=1, status="in_progress")
        candidate_task = Task(id=11, position=(80.0, 82.0), demand=1, status="withdrawn")
        vehicle = Vehicle(
            id=0,
            start_pos=(12.0, 14.0),
            heading=0.0,
            speed=5.0,
            max_omega=0.5,
            capacity=10,
            remaining_capacity=10,
            current_pos=(12.0, 14.0),
            current_heading=0.2,
            active_task_id=active_task.id,
            active_goal_heading=0.9,
        )
        segment_calls = []

        def fake_segment_corrected_time(**kwargs):
            segment_calls.append((kwargs["start_pos"], kwargs["start_heading"], kwargs["task"].id))
            return 5.0, 2.0, 1.1, None, ""

        with patch("milp_sim.verification._segment_corrected_time", side_effect=fake_segment_corrected_time):
            result = verify_bid(
                vehicle=vehicle,
                task=candidate_task,
                c_hat=2.0,
                cfg=cfg,
                planner=self.planner,
                tasks_by_id={
                    active_task.id: active_task,
                    candidate_task.id: candidate_task,
                },
            )

        self.assertTrue(result.passed)
        self.assertEqual(len(segment_calls), 1)
        self.assertEqual(segment_calls[0][0], active_task.position)
        self.assertAlmostEqual(segment_calls[0][1], 0.9)
        self.assertEqual(segment_calls[0][2], candidate_task.id)

    def test_seeded_session_routes_do_not_cross_obstacles(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            seed=0,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            dubins_fallback_to_astar=True,
            dubins_force_mode=False,
        )
        session = SimulationSession(cfg=cfg)
        result = session.result()
        assert session.artifacts is not None
        world = session.artifacts.world

        for vehicle in result.vehicles:
            pts = vehicle.route_points
            for i in range(len(pts) - 1):
                seg = LineString([pts[i], pts[i + 1]])
                crossed = seg.crosses(world.obstacle_union) or seg.within(world.obstacle_union)
                self.assertFalse(
                    crossed,
                    msg=(
                        f"V{vehicle.id} route segment {i}->{i + 1} crosses obstacles: "
                        f"{pts[i]} -> {pts[i + 1]}"
                    ),
                )

    def test_seeded_session_routes_have_no_jump_backtracking_turns(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            seed=7,
            use_dubins_hybrid=True,
            use_hybrid_astar=False,
            dubins_fallback_to_astar=True,
            dubins_force_mode=False,
        )
        session = SimulationSession(cfg=cfg)
        result = session.result()

        for vehicle in result.vehicles:
            pts = vehicle.route_points
            for i in range(1, len(pts) - 1):
                ax = pts[i][0] - pts[i - 1][0]
                ay = pts[i][1] - pts[i - 1][1]
                bx = pts[i + 1][0] - pts[i][0]
                by = pts[i + 1][1] - pts[i][1]
                la = math.hypot(ax, ay)
                lb = math.hypot(bx, by)
                if la <= 0.2 or lb <= 0.2:
                    continue

                dot = (ax * bx + ay * by) / (la * lb)
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                # Real jump-lines are short local spikes with near reverse angle.
                is_spike = angle_deg > 170.0 and min(la, lb) <= 2.0
                self.assertFalse(
                    is_spike,
                    msg=(
                        f"V{vehicle.id} has backtracking turn around point {i}: "
                        f"angle={angle_deg:.2f} la={la:.3f} lb={lb:.3f} "
                        f"p_prev={pts[i - 1]} p={pts[i]} p_next={pts[i + 1]}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
