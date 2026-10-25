"""Unit Tests for Statistical Rigor and Wilson Confidence Intervals."""

from __future__ import annotations

import unittest

from worldsim.statistics import (
    aggregate_raw_episodes,
    cluster_bootstrap_interval,
    compute_paired_comparison,
    summarize_continuous,
    wilson_score_interval,
)


class TestStatisticalRigor(unittest.TestCase):
    """Test suite verifying mathematical correctness of confidence intervals and summaries."""

    def test_wilson_interval_bounds(self) -> None:
        """Verify Wilson score intervals are bounded in [0, 1] and match standard values."""
        # 10 successes out of 20 trials (p = 0.50)
        low, high = wilson_score_interval(10, 20, 0.95)
        self.assertAlmostEqual(low, 0.2993, places=2)
        self.assertAlmostEqual(high, 0.7007, places=2)
        self.assertTrue(0.0 <= low <= high <= 1.0)

    def test_wilson_interval_zero_successes(self) -> None:
        """Verify Wilson interval handles 0/20 without collapsing or crashing."""
        low, high = wilson_score_interval(0, 20, 0.95)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.20)  # Rule of three bound

    def test_wilson_interval_all_successes(self) -> None:
        """Verify Wilson interval handles 20/20 without reporting [1.0, 1.0]."""
        low, high = wilson_score_interval(20, 20, 0.95)
        self.assertGreater(low, 0.80)
        self.assertEqual(high, 1.0)

    def test_summarize_continuous_empty(self) -> None:
        """Verify continuous summary handles empty arrays gracefully."""
        summary = summarize_continuous([])
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.std)

    def test_summarize_continuous_single(self) -> None:
        """Verify continuous summary handles n=1 without division by zero in std."""
        summary = summarize_continuous([14])
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.mean, 14.0)
        self.assertEqual(summary.std, 0.0)

    def test_aggregate_raw_episodes(self) -> None:
        """Verify aggregate_raw_episodes computes correct rates and distinct fingerprints."""
        raw_records = [
            {
                "completed": True,
                "num_agents": 2,
                "successful_agents": 2,
                "steps": 12,
                "environment_fingerprint": "fp_alpha",
                "exhausted": False,
                "timed_out": False,
                "deadlocked": False,
                "actual_collisions": 0,
                "blocked_attempts": 1,
                "runtime_ms": 5.2,
            },
            {
                "completed": False,
                "num_agents": 2,
                "successful_agents": 1,
                "steps": 50,
                "environment_fingerprint": "fp_beta",
                "exhausted": True,
                "timed_out": False,
                "deadlocked": False,
                "actual_collisions": 2,
                "blocked_attempts": 3,
                "runtime_ms": 12.0,
            },
        ]

        summary = aggregate_raw_episodes("TEST_01", "Test Scenario", "heuristic", raw_records)
        self.assertEqual(summary.num_episodes, 2)
        self.assertEqual(summary.num_distinct_environments, 2)
        self.assertEqual(summary.num_repeated_environments, 0)
        self.assertEqual(summary.completed_episodes, 1)
        self.assertEqual(summary.completion_rate, 0.5)
        self.assertEqual(summary.total_agent_slots, 4)
        self.assertEqual(summary.successful_agent_slots, 3)
        self.assertEqual(summary.per_agent_success_rate, 0.75)
        self.assertEqual(summary.exhaustion_count, 1)
        self.assertEqual(summary.success_steps.count, 1)
        self.assertEqual(summary.success_steps.mean, 12.0)
        self.assertEqual(summary.all_episode_steps.mean, 31.0)

    def test_cluster_bootstrap_insufficient_clusters(self) -> None:
        """Verify cluster bootstrap refuses to output degenerate intervals for K < 5."""
        episodes = [
            {"environment_fingerprint": "c1", "completed": True},
            {"environment_fingerprint": "c1", "completed": True},
            {"environment_fingerprint": "c2", "completed": False},
            {"environment_fingerprint": "c3", "completed": True},
        ]
        low, high, status = cluster_bootstrap_interval(episodes, min_clusters=5)
        self.assertIsNone(low)
        self.assertIsNone(high)
        self.assertIn("INSUFFICIENT_CLUSTERS", status)

    def test_cluster_bootstrap_valid_clusters(self) -> None:
        """Verify cluster bootstrap computes non-degenerate bounds for K >= 5."""
        episodes = [
            {"environment_fingerprint": f"c{i}", "completed": (i % 2 == 0)}
            for i in range(6)
            for _ in range(2)
        ]
        low, high, status = cluster_bootstrap_interval(episodes, min_clusters=5, seed=42)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        assert low is not None and high is not None
        self.assertTrue(0.0 <= low <= high <= 1.0)
        self.assertIn("Cluster Bootstrap", status)

    def test_cluster_bootstrap_boundary_cases(self) -> None:
        """Verify cluster bootstrap handles all-success and all-failure gracefully."""
        all_succ = [{"environment_fingerprint": f"c{i}", "completed": True} for i in range(6)]
        low_s, high_s, _ = cluster_bootstrap_interval(all_succ, min_clusters=5)
        self.assertEqual(low_s, 1.0)
        self.assertEqual(high_s, 1.0)

        all_fail = [{"environment_fingerprint": f"c{i}", "completed": False} for i in range(6)]
        low_f, high_f, _ = cluster_bootstrap_interval(all_fail, min_clusters=5)
        self.assertEqual(low_f, 0.0)
        self.assertEqual(high_f, 0.0)

    def test_compute_paired_comparison(self) -> None:
        """Verify compute_paired_comparison matches environments and calculates wins/ties."""
        eps_a = [
            {"scenario_id": "H01", "env_seed": 300, "completed": True, "steps": 10},
            {"scenario_id": "H01", "env_seed": 301, "completed": False, "steps": 60},
            {"scenario_id": "H01", "env_seed": 302, "completed": True, "steps": 12},
        ]
        eps_b = [
            {"scenario_id": "H01", "env_seed": 300, "completed": False, "steps": 60},
            {"scenario_id": "H01", "env_seed": 301, "completed": False, "steps": 60},
            {"scenario_id": "H01", "env_seed": 302, "completed": True, "steps": 15},
        ]

        paired = compute_paired_comparison(eps_a, eps_b, "policy_a", "policy_b")
        self.assertEqual(paired.num_matched_pairs, 3)
        self.assertEqual(paired.wins_a, 1)  # Seed 300
        self.assertEqual(paired.wins_b, 0)
        self.assertEqual(paired.ties, 2)    # Seed 301 (both fail), 302 (both win)
        self.assertEqual(paired.mean_step_diff, -3.0)


if __name__ == "__main__":
    unittest.main()
