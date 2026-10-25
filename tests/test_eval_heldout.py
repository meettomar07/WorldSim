"""Tests for Held-Out Evaluation Pipeline, Split Isolation, and Reproducibility."""

import unittest
from pathlib import Path

from worldsim.scenario_registry import Split, get_scenario_descriptor
from experiments.eval_heldout import compute_all_paired_comparisons, run_heldout_benchmark


class TestHeldOutEvaluation(unittest.TestCase):
    """Verify held-out evaluation runner, data split isolation, and reproducibility."""

    def test_split_isolation_rejects_train_or_val(self) -> None:
        """Verify that training or validation scenarios cannot be passed to heldout benchmark."""
        # Attempting to run with TRAIN or VAL scenario must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            run_heldout_benchmark(
                scenarios=["TRAIN_ENV1"],
                policies=["cbs"],
                seeds=[300],
            )
        self.assertIn("SPLIT ISOLATION VIOLATION", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            run_heldout_benchmark(
                scenarios=["VAL_U01"],
                policies=["cbs"],
                seeds=[300],
            )
        self.assertIn("SPLIT ISOLATION VIOLATION", str(ctx.exception))

    def test_reproducibility_same_seeds(self) -> None:
        """Verify identical results across consecutive runs with the same seeds."""
        scenarios = ["TEST_H01"]
        policies = ["cbs", "hybrid_comm"]
        seeds = [301, 302]

        sum1, eps1 = run_heldout_benchmark(scenarios, policies, seeds)
        sum2, eps2 = run_heldout_benchmark(scenarios, policies, seeds)

        self.assertEqual(len(eps1), len(eps2))
        for ep1, ep2 in zip(eps1, eps2):
            self.assertEqual(ep1["scenario_id"], ep2["scenario_id"])
            self.assertEqual(ep1["policy"], ep2["policy"])
            self.assertEqual(ep1["env_seed"], ep2["env_seed"])
            self.assertEqual(ep1["environment_fingerprint"], ep2["environment_fingerprint"])
            self.assertEqual(ep1["completed"], ep2["completed"])
            self.assertEqual(ep1["steps"], ep2["steps"])
            self.assertEqual(ep1["actual_collisions"], ep2["actual_collisions"])

        # Check cluster fields in summaries
        for s in sum1:
            self.assertGreaterEqual(s.num_episodes, 2)
            self.assertGreaterEqual(s.num_distinct_environments, 1)
            self.assertGreaterEqual(s.num_repeated_environments, 0)
            self.assertTrue(hasattr(s, "cluster_completion_rate"))
            self.assertTrue(hasattr(s, "cluster_ci_95"))

    def test_paired_comparisons_structure(self) -> None:
        """Verify compute_all_paired_comparisons generates expected pairs."""
        scenarios = ["TEST_H01"]
        policies = ["cbs", "hybrid_comm", "centralized"]
        seeds = [301]

        _, eps = run_heldout_benchmark(scenarios, policies, seeds)
        paired = compute_all_paired_comparisons(eps)
        self.assertGreaterEqual(len(paired), 2)
        for p in paired:
            self.assertEqual(p.num_matched_pairs, 1)
            self.assertEqual(p.wins_a + p.wins_b + p.ties, 1)


if __name__ == "__main__":
    unittest.main()
