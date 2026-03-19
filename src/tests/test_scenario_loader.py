import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from milp_sim.config import DEFAULT_CONFIG
from milp_sim.scenario_loader import load_scenario_file
from milp_sim.simulator import build_static_scenario


def _sample_scenario_payload() -> dict:
    return {
        "world": {
            "width": 60.0,
            "height": 50.0,
            "depot_polygon": [[4.0, 4.0], [12.0, 4.0], [12.0, 12.0], [4.0, 12.0]],
            "obstacles": [
                [[25.0, 8.0], [30.0, 8.0], [30.0, 18.0], [25.0, 18.0]],
                [[40.0, 28.0], [45.0, 28.0], [45.0, 40.0], [40.0, 40.0]],
            ],
        },
        "vehicles": [
            {"id": 0, "start": [6.0, 6.0], "heading": 0.0, "speed": 4.5, "max_omega": 1.2, "capacity": 8},
            {"id": 1, "start": [10.0, 10.0], "heading": 0.5, "speed": 5.0, "max_omega": 1.1, "capacity": 9},
        ],
        "tasks": [
            {"id": 10, "position": [16.0, 30.0], "demand": 2},
            {"id": 11, "pos": [52.0, 16.0], "demand": 3},
        ],
    }


class TestScenarioLoader(unittest.TestCase):
    def test_build_static_scenario_prefers_scenario_file_over_random_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "scenario.json"
            scenario_path.write_text(json.dumps(_sample_scenario_payload()), encoding="utf-8")

            cfg = replace(
                DEFAULT_CONFIG,
                scenario_file=str(scenario_path),
                seed=999,
                num_vehicles=99,
                num_tasks=77,
                map_width=999.0,
                map_height=888.0,
            )
            artifacts = build_static_scenario(cfg)

        self.assertEqual(artifacts.world.width, 60.0)
        self.assertEqual(artifacts.world.height, 50.0)
        self.assertEqual(len(artifacts.world.obstacles), 2)
        self.assertEqual(len(artifacts.vehicles), 2)
        self.assertEqual(len(artifacts.tasks), 2)

        self.assertEqual([v.id for v in artifacts.vehicles], [0, 1])
        self.assertEqual(artifacts.vehicles[0].start_pos, (6.0, 6.0))
        self.assertEqual(artifacts.vehicles[1].capacity, 9)
        self.assertEqual([t.id for t in artifacts.tasks], [10, 11])
        self.assertEqual(artifacts.tasks[0].position, (16.0, 30.0))
        self.assertEqual(artifacts.tasks[1].position, (52.0, 16.0))

    def test_loader_rejects_missing_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_scenario.json"
            bad_path.write_text(json.dumps({"world": {"width": 10.0, "height": 10.0}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scenario_file(bad_path)


if __name__ == "__main__":
    unittest.main()
