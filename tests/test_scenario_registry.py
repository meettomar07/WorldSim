"""Unit and Integration Tests for Scenario Registry, Procedural Generation & Split Isolation."""

from __future__ import annotations

import unittest

from worldsim.models import Position
from worldsim.scenario_generator import bfs_reachable_cells
from worldsim.scenario_registry import (
    Split,
    assert_not_held_out,
    build_scenario_instance,
    get_scenario_descriptor,
    list_scenarios,
)


class TestScenarioRegistry(unittest.TestCase):
    """Test suite verifying split isolation, deterministic generation, and seed variation."""

    def test_split_isolation_guard(self) -> None:
        """Verify assert_not_held_out raises RuntimeError if a TEST scenario is called in training."""
        with self.assertRaises(RuntimeError) as ctx:
            assert_not_held_out("TEST_H01", context="RL Policy Tuning")
        self.assertIn("DATA CONTAMINATION DETECTED", str(ctx.exception))

        # Train and Val scenarios should pass without error
        assert_not_held_out("TRAIN_ENV1", context="RL Policy Tuning")
        assert_not_held_out("VAL_U06", context="Remediation")

    def test_list_scenarios_by_split(self) -> None:
        """Verify scenarios are correctly categorized into TRAIN, VAL, and TEST."""
        train_scens = list_scenarios(Split.TRAIN)
        val_scens = list_scenarios(Split.VAL)
        test_scens = list_scenarios(Split.TEST)

        self.assertGreaterEqual(len(train_scens), 3)
        self.assertGreaterEqual(len(val_scens), 5)
        self.assertEqual(len(test_scens), 6)

        # Confirm IDs are distinct
        all_ids = [s.scenario_id for s in list_scenarios()]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_deterministic_generation_same_seed(self) -> None:
        """Verify procedural scenarios produce bit-for-bit identical fingerprints for the same seed."""
        for scen_id in ["TEST_H01", "TEST_H02", "TEST_H03", "TEST_H04", "TEST_H05", "TEST_H06"]:
            cfg1, walls1, fp1 = build_scenario_instance(scen_id, env_seed=123)
            cfg2, walls2, fp2 = build_scenario_instance(scen_id, env_seed=123)

            self.assertEqual(fp1, fp2, f"Fingerprints diverged for seed 123 on {scen_id}")
            self.assertEqual(walls1, walls2)
            self.assertEqual(len(cfg1.agents), len(cfg2.agents))
            for a1, a2 in zip(cfg1.agents, cfg2.agents):
                self.assertEqual((a1.initial_x, a1.initial_y), (a2.initial_x, a2.initial_y))
                self.assertEqual(a1.target, a2.target)

    def test_environmental_variation_across_seeds(self) -> None:
        """Verify that different seeds produce genuinely distinct environment fingerprints."""
        for scen_id in ["TEST_H01", "TEST_H02", "TEST_H03", "TEST_H04"]:
            fingerprints = set()
            for seed in range(100, 115):
                _, _, fp = build_scenario_instance(scen_id, env_seed=seed)
                fingerprints.add(fp)

            # Across 15 seeds, there should be multiple distinct layouts
            self.assertGreater(
                len(fingerprints),
                1,
                f"Scenario {scen_id} produced no environmental variation across seeds",
            )

    def test_maze_connectivity_invariant(self) -> None:
        """Verify procedural maze generator guarantees reachability between corners."""
        for seed in (42, 99, 101, 777):
            cfg, walls, _ = build_scenario_instance("TEST_H03", env_seed=seed)
            corners = [Position(0, 0), Position(7, 0), Position(0, 7), Position(7, 7)]
            reachable = bfs_reachable_cells(cfg.width, cfg.height, walls, corners[0])
            for c in corners:
                self.assertIn(c, reachable, f"Corner {c} disconnected in maze seed {seed}")


if __name__ == "__main__":
    unittest.main()
