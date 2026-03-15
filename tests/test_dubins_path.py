import math
import unittest
from dataclasses import replace

try:
    from shapely.geometry import box

    from milp_sim.config import DEFAULT_CONFIG
    from milp_sim.dubins_path import DUBINS_WORDS, build_dubins_hybrid_path, solve_dubins_word
    from milp_sim.map_utils import WorldMap
    from milp_sim.planner_astar import AStarPlanner

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
        cfg = replace(DEFAULT_CONFIG, use_dubins_hybrid=True, dubins_sample_step=0.8, dubins_collision_margin=1.8)
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

    def test_hybrid_fallback_to_astar_when_collision_too_strict(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
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

    def test_hybrid_returns_inf_when_fallback_disabled_and_collision_strict(self) -> None:
        cfg = replace(
            DEFAULT_CONFIG,
            use_dubins_hybrid=True,
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


if __name__ == "__main__":
    unittest.main()
