import math
import unittest

from shapely.geometry import Polygon

from milp_sim.map_utils import WorldMap
from milp_sim.path_quality_metrics import evaluate_path_quality, summarize_path_quality


class PathQualityMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = WorldMap(
            width=20.0,
            height=20.0,
            depot_polygon=Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            obstacles=[],
        )

    def test_straight_path_has_zero_turn_and_no_curvature_violation(self) -> None:
        metrics = evaluate_path_quality(
            world=self.world,
            points=[(2.0, 2.0), (10.0, 2.0)],
            turn_radius=2.0,
            sample_step=0.5,
            start_heading=0.0,
            astar_length=8.0,
        )

        self.assertTrue(metrics.collision_free)
        self.assertAlmostEqual(metrics.path_length, 8.0, places=6)
        self.assertAlmostEqual(metrics.length_ratio or 0.0, 1.0, places=6)
        self.assertAlmostEqual(metrics.max_initial_turn_delta_rad, 0.0, places=6)
        self.assertAlmostEqual(metrics.curvature_violation_ratio, 0.0, places=6)
        self.assertAlmostEqual(metrics.heading_jump_p95_rad, 0.0, places=6)

    def test_right_angle_path_has_heading_jump_and_lower_summary_score(self) -> None:
        straight = evaluate_path_quality(
            world=self.world,
            points=[(2.0, 2.0), (10.0, 2.0)],
            turn_radius=2.0,
            sample_step=0.5,
            start_heading=0.0,
            astar_length=8.0,
        )
        corner = evaluate_path_quality(
            world=self.world,
            points=[(2.0, 2.0), (6.0, 2.0), (6.0, 6.0)],
            turn_radius=10.0,
            sample_step=0.5,
            start_heading=0.0,
            astar_length=7.0,
            fallback_used=True,
            fallback_reason="unit_test",
        )

        self.assertGreater(corner.heading_jump_p95_rad, 0.0)
        self.assertGreater(corner.curvature_violation_ratio, 0.0)
        summary = summarize_path_quality([straight, corner])
        self.assertEqual(summary.path_count, 2)
        self.assertGreater(summary.fallback_rate, 0.0)
        self.assertLess(summary.pqi_score, 100.0)
        self.assertGreaterEqual(summary.mean_max_initial_turn_delta_rad, 0.0)

    def test_path_crossing_obstacle_is_detected(self) -> None:
        world = WorldMap(
            width=20.0,
            height=20.0,
            depot_polygon=Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            obstacles=[Polygon([(5, 4), (7, 4), (7, 6), (5, 6)])],
        )
        metrics = evaluate_path_quality(
            world=world,
            points=[(2.0, 5.0), (10.0, 5.0)],
            turn_radius=2.0,
            sample_step=0.25,
            start_heading=0.0,
        )

        self.assertFalse(metrics.collision_free)
        self.assertAlmostEqual(metrics.min_clearance, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
