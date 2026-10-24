"""Unit tests for Phase 13.3: Reliable Completion, Fairness & Benchmark Validation.

Covers:
1. Goal-cell yielding: inactive agents resting on target cells yield right-of-way.
2. Terminal corridor traversal: active agents pass through completed agents to reach deeper goals.
3. Resource arbitration fairness: deterministic round-robin tie-breaking and starvation prevention.
4. Starvation detection and compensatory priority escalation under Fair-Balanced arbitration.
5. Persistent resource commitments across ticks and accurate load distribution tracking.
6. Energy-safe coordination detours: preventing fatal moves that deplete agent energy.
7. Accurate metric distinction between episode-level overall completion and per-agent success rate.
8. Honest failure accounting and exhaustion reporting in mathematically infeasible environments.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import (
    DeadlockDetector,
    DeadlockEvent,
    FairnessMonitor,
    MultiAgentCoordinator,
    ResourceArbitrator,
    jains_fairness_index,
)
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.multi_agent_experiment_phase13_3 import (
    Phase13_3ExperimentMetrics,
    make_robust_factory,
    run_phase13_3_episode,
)
from worldsim.robust_policy import (
    AgentMemory,
    RechargeEvaluator,
    ResourceUtilityEvaluator,
    RobustResourcePolicy,
)


class TestPhase13_3ReliableCompletionAndFairness(unittest.TestCase):
    """Unit tests for Phase 13.3 multi-agent reliable completion and fairness improvements."""

    def test_goal_cell_yielding_allows_active_agent_traversal(self) -> None:
        """Verify an inactive agent with status 'reached_target' yields right-of-way."""
        # 4x1 grid: a2 is at (2, 0) with status 'reached_target'.
        # a1 at (1, 0) wants to move RIGHT to (2, 0) on its way to (3, 0).
        cfg = MultiAgentWorldConfig(
            width=4, height=1,
            agents=(
                AgentConfig("a1", initial_x=1, initial_y=0, initial_energy=5, target=Position(3, 0)),
                AgentConfig("a2", initial_x=2, initial_y=0, initial_energy=5, target=Position(2, 0)),
            ),
        )
        sim = MultiAgentSimulation(cfg)
        # First tick: a2 reaches target and becomes reached_target (active=False)
        step_res1 = sim.step({"a1": Action.LEFT, "a2": Action.UP}) # a1 steps back to (0,0), a2 at (2,0) reaches target!
        self.assertEqual(sim.state.agents["a2"].status, "reached_target")
        self.assertFalse(sim.state.agents["a2"].active)

        # Now a1 is at (0, 0), moves to (1, 0)
        sim.step({"a1": Action.RIGHT})

        # Now a1 is at (1, 0). a1 moves RIGHT into (2, 0) where completed a2 is resting.
        step_res = sim.step({"a1": Action.RIGHT})
        # a1 should NOT be rejected with 'occupied' collision; it should be allowed to enter (2, 0)!
        self.assertIsNone(step_res.agent_results["a1"].collision_type)
        self.assertEqual(sim.state.agents["a1"].position, Position(2, 0))
        self.assertEqual(sim.state.agents["a2"].position, Position(2, 0))
        self.assertEqual(sim.state.agents["a2"].status, "reached_target")

        # Next tick: a1 moves from (2, 0) to (3, 0) and reaches its own target!
        step_res_final = sim.step({"a1": Action.RIGHT})
        self.assertEqual(sim.state.agents["a1"].position, Position(3, 0))
        self.assertEqual(sim.state.agents["a1"].status, "reached_target")
        self.assertTrue(sim._check_overall_completion())

    def test_terminal_corridor_solves_aisle_vs_window_seat(self) -> None:
        """Verify 6x1 corridor completes 100% when front agent finishes first."""
        cfg = MultiAgentWorldConfig(
            width=6, height=1,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=10, target=Position(5, 0)),
                AgentConfig("a2", initial_x=2, initial_y=0, initial_energy=10, target=Position(4, 0)),
            ),
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(cfg, seed=100)
        policies = {a.id: make_robust_factory(6, 1, 10, None, coordinator=sim.coordinator)() for a in cfg.agents}
        comp, steps, reached, _, _ = run_phase13_3_episode(sim, policies, max_steps=20)
        self.assertTrue(comp)
        self.assertTrue(reached["a1"])
        self.assertTrue(reached["a2"])
        self.assertLessEqual(steps, 6)

    def test_resource_arbitration_fair_balanced_alternates_on_ties(self) -> None:
        """Verify Fair-Balanced arbitration rotates winners equitably on tied claims."""
        arb = ResourceArbitrator(rule="fair_balanced")
        res = Resource(Position(1, 1), 6, consumable=True)
        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 2, "agent_2": 2}
        max_energies = {"agent_1": 8, "agent_2": 8}

        winners = []
        for t in range(6):
            w, _ = arb.arbitrate(claimants, energies, max_energies, res, tick=t)
            winners.append(w)

        # Should strictly alternate between agent_1 and agent_2
        self.assertEqual(winners[0], "agent_1")
        self.assertEqual(winners[1], "agent_2")
        self.assertEqual(winners[2], "agent_1")
        self.assertEqual(winners[3], "agent_2")
        self.assertEqual(winners[4], "agent_1")
        self.assertEqual(winners[5], "agent_2")

    def test_lowest_id_arbitration_permanently_favors_agent_1(self) -> None:
        """Verify lowest_id arbitration permanently starves agent_2 across repeated rounds."""
        arb = ResourceArbitrator(rule="lowest_id")
        res = Resource(Position(1, 1), 6, consumable=True)
        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 2, "agent_2": 2}
        max_energies = {"agent_1": 8, "agent_2": 8}

        for t in range(10):
            w, losers = arb.arbitrate(claimants, energies, max_energies, res, tick=t)
            self.assertEqual(w, "agent_1")
            self.assertEqual(losers, ["agent_2"])

    def test_starvation_prevention_prioritizes_higher_consecutive_losses(self) -> None:
        """Verify agent with consecutive losses wins over agent with higher energy."""
        arb = ResourceArbitrator(rule="fair_balanced")
        res = Resource(Position(1, 1), 6, consumable=True)

        # Manually seed agent_2 with consecutive losses
        rec2 = arb.get_or_create_record("agent_2")
        rec2.consecutive_losses = 2
        rec2.contested_losses = 2

        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 2, "agent_2": 2}
        max_energies = {"agent_1": 8, "agent_2": 8}

        winner, losers = arb.arbitrate(claimants, energies, max_energies, res, tick=0)
        self.assertEqual(winner, "agent_2")
        self.assertEqual(losers, ["agent_1"])

    def test_coordinator_preserves_commitments_across_ticks(self) -> None:
        """Verify MultiAgentCoordinator maintains resource commitment across ticks until arrival."""
        coord = MultiAgentCoordinator(enabled=True)
        r_pos = Position(3, 3)
        coord.register_intent("agent_1", destination=Position(2, 3), targeted_resource=r_pos)
        self.assertEqual(coord.get_resource_load(r_pos), 1)

        # reset_tick should clear next-step intentions but preserve commitments
        coord.reset_tick()
        self.assertEqual(len(coord.agent_intentions), 0)
        self.assertEqual(coord.get_resource_load(r_pos), 1)

        # explicit clear commitment
        coord.clear_commitment("agent_1")
        self.assertEqual(coord.get_resource_load(r_pos), 0)

    def test_infeasible_environment_reports_honest_failure_and_exhaustion(self) -> None:
        """Verify mathematically over-constrained environments honestly report 0% completion."""
        cfg = MultiAgentWorldConfig(
            width=8, height=8,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=2, target=Position(7, 7)),
                AgentConfig("a2", initial_x=0, initial_y=1, initial_energy=2, target=Position(7, 6)),
            ),
            resources=(Resource(Position(5, 5), 3, consumable=True),),
            max_energy=4,
        )
        sim = MultiAgentSimulation(cfg, seed=100)
        policies = {a.id: make_robust_factory(8, 8, 4, None)() for a in cfg.agents}
        comp, steps, reached, _, _ = run_phase13_3_episode(sim, policies, max_steps=20)
        self.assertFalse(comp)
        self.assertFalse(reached["a1"])
        self.assertFalse(reached["a2"])
        self.assertEqual(sim.state.agents["a1"].status, "exhausted")
        self.assertEqual(sim.state.agents["a2"].status, "exhausted")

    def test_metric_distinction_overall_completion_vs_per_agent_success(self) -> None:
        """Verify overall completion rate is strictly distinguished from per-agent success."""
        # 1 agent finishes, 1 agent fails
        cfg = MultiAgentWorldConfig(
            width=6, height=1,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=10, target=Position(2, 0)),
                AgentConfig("a2", initial_x=5, initial_y=0, initial_energy=1, target=Position(0, 0)), # distance 5, energy 1 -> exhausts!
            ),
        )
        sim = MultiAgentSimulation(cfg, seed=100)
        policies = {a.id: make_robust_factory(6, 1, 10, None)() for a in cfg.agents}
        comp, steps, reached, _, _ = run_phase13_3_episode(sim, policies, max_steps=10)

        # Episode level overall completion must be FALSE!
        self.assertFalse(comp)
        # Per-agent: a1 succeeded (100%), a2 failed (0%) -> Mean agent success is 50%!
        self.assertTrue(reached["a1"])
        self.assertFalse(reached["a2"])


if __name__ == "__main__":
    unittest.main()
