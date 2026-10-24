"""Tests for Phase 13.1: Multi-Agent Coordination, Resource Allocation, and Congestion Control.

Covers:
1. ResourceArbitrator rules: lowest_id, priority_rotation, need_urgency, starvation_prevention, fair_balanced.
2. DeadlockDetector: repeated_swap, occupied_blocking, vertex_deadlock, and recovery tracking.
3. FairnessMonitor: Jain's fairness index, per-agent execution metrics, starvation event detection.
4. MultiAgentCoordinator: intention exchange, resource load-balancing, alternative move suggestions.
5. MultiAgentSimulation integration: priority rotation vs bounce, resource contention arbitration, cooperative discovery.
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
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)


class TestResourceArbitrator(unittest.TestCase):
    """Unit tests for ResourceArbitrator allocation rules."""

    def setUp(self) -> None:
        self.res = Resource(position=Position(2, 2), recharge_amount=6, consumable=True)

    def test_lowest_id_rule(self) -> None:
        arb = ResourceArbitrator(rule="lowest_id")
        claimants = ["agent_3", "agent_1", "agent_2"]
        energies = {"agent_1": 5, "agent_2": 2, "agent_3": 1}
        max_energies = {aid: 10 for aid in claimants}
        winner, losers = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        self.assertEqual(winner, "agent_1")
        self.assertEqual(losers, ["agent_2", "agent_3"])

    def test_priority_rotation_rule(self) -> None:
        arb = ResourceArbitrator(rule="priority_rotation")
        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 3, "agent_2": 3}
        max_energies = {"agent_1": 10, "agent_2": 10}

        w0, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        w1, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=1)
        self.assertEqual(w0, "agent_1")
        self.assertEqual(w1, "agent_2")

    def test_need_urgency_rule_prioritizes_lowest_energy(self) -> None:
        arb = ResourceArbitrator(rule="need_urgency")
        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 5, "agent_2": 1}  # agent_2 has 9 deficit vs agent_1 with 5 deficit
        max_energies = {"agent_1": 10, "agent_2": 10}

        winner, losers = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        self.assertEqual(winner, "agent_2")
        self.assertEqual(losers, ["agent_1"])

    def test_starvation_prevention_prioritizes_consecutive_losses(self) -> None:
        arb = ResourceArbitrator(rule="starvation_prevention")
        claimants = ["agent_1", "agent_2"]
        energies = {"agent_1": 5, "agent_2": 5}
        max_energies = {"agent_1": 10, "agent_2": 10}

        # Give agent_2 two recorded consecutive losses
        arb.get_or_create_record("agent_2").consecutive_losses = 2
        arb.get_or_create_record("agent_1").consecutive_losses = 0

        winner, losers = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        self.assertEqual(winner, "agent_2")

    def test_fair_balanced_balances_urgency_and_starvation(self) -> None:
        arb = ResourceArbitrator(rule="fair_balanced")
        claimants = ["agent_1", "agent_2"]
        # agent_1 has lower energy, but agent_2 has suffered multiple starvation losses
        energies = {"agent_1": 2, "agent_2": 4}
        max_energies = {"agent_1": 10, "agent_2": 10}
        arb.get_or_create_record("agent_2").consecutive_losses = 3

        winner, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        self.assertEqual(winner, "agent_2")


class TestDeadlockDetector(unittest.TestCase):
    """Unit tests for DeadlockDetector and recovery tracking."""

    def test_detects_repeated_swap_deadlock(self) -> None:
        detector = DeadlockDetector(persistence_threshold=2)
        pos = {"agent_1": Position(1, 1), "agent_2": Position(2, 1)}
        intents = {
            "agent_1": (Action.RIGHT, Position(2, 1)),
            "agent_2": (Action.LEFT, Position(1, 1)),
        }
        cols = {"agent_1": "swap", "agent_2": "swap"}
        moved = {"agent_1": False, "agent_2": False}

        # Tick 1: first swap encounter
        events1 = detector.update_tick(1, pos, intents, cols, moved)
        self.assertEqual(len(events1), 0)

        # Tick 2: repeated swap encounter triggers deadlock detection
        events2 = detector.update_tick(2, pos, intents, cols, moved)
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0].deadlock_type, "repeated_swap")
        self.assertEqual(detector.recovery_attempts, 1)

    def test_detects_occupied_blocking_deadlock(self) -> None:
        detector = DeadlockDetector(persistence_threshold=2)
        pos = {"agent_1": Position(1, 1), "agent_2": Position(2, 1)}
        intents = {
            "agent_1": (Action.RIGHT, Position(2, 1)),
            "agent_2": (Action.UP, Position(2, 1)),  # stationary
        }
        cols = {"agent_1": "occupied", "agent_2": None}
        moved = {"agent_1": False, "agent_2": False}

        detector.update_tick(1, pos, intents, cols, moved)
        events = detector.update_tick(2, pos, intents, cols, moved)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].deadlock_type, "occupied_blocking")

    def test_tracks_successful_deadlock_recovery(self) -> None:
        detector = DeadlockDetector(persistence_threshold=2)
        pos = {"agent_1": Position(1, 1), "agent_2": Position(2, 1)}
        intents = {
            "agent_1": (Action.RIGHT, Position(2, 1)),
            "agent_2": (Action.LEFT, Position(1, 1)),
        }
        cols = {"agent_1": "swap", "agent_2": "swap"}
        moved = {"agent_1": False, "agent_2": False}

        detector.update_tick(1, pos, intents, cols, moved)
        detector.update_tick(2, pos, intents, cols, moved)
        self.assertEqual(detector.unresolved_deadlocks, 1)

        # Tick 3: Agents escape / move
        pos_after = {"agent_1": Position(1, 2), "agent_2": Position(2, 1)}
        intents_after = {
            "agent_1": (Action.DOWN, Position(1, 2)),
            "agent_2": (Action.UP, Position(2, 0)),
        }
        cols_after = {"agent_1": None, "agent_2": None}
        moved_after = {"agent_1": True, "agent_2": True}

        detector.update_tick(3, pos_after, intents_after, cols_after, moved_after)
        self.assertEqual(detector.successful_recoveries, 1)
        self.assertEqual(detector.unresolved_deadlocks, 0)


class TestFairnessMonitorAndJainsIndex(unittest.TestCase):
    """Unit tests for Jain's Fairness Index and FairnessMonitor."""

    def test_jains_fairness_equal_values(self) -> None:
        self.assertAlmostEqual(jains_fairness_index([10.0, 10.0, 10.0]), 1.0)
        self.assertAlmostEqual(jains_fairness_index([1.0, 1.0]), 1.0)

    def test_jains_fairness_maximally_unfair(self) -> None:
        # One 1.0, others 0.0 -> 1/N
        self.assertAlmostEqual(jains_fairness_index([1.0, 0.0]), 0.5)
        self.assertAlmostEqual(jains_fairness_index([1.0, 0.0, 0.0, 0.0]), 0.25)

    def test_jains_fairness_all_zeros(self) -> None:
        self.assertEqual(jains_fairness_index([0.0, 0.0]), 1.0)

    def test_fairness_monitor_tracks_starvation_after_3_blocked_ticks(self) -> None:
        fm = FairnessMonitor(["agent_1", "agent_2"])
        pos_b = {"agent_1": Position(0, 0), "agent_2": Position(1, 0)}
        pos_a = {"agent_1": Position(0, 0), "agent_2": Position(1, 0)}
        e_b = {"agent_1": 5, "agent_2": 5}
        e_a = {"agent_1": 5, "agent_2": 5}
        acts = {"agent_1": Action.RIGHT, "agent_2": Action.LEFT}
        succ = {"agent_1": False, "agent_2": False}
        cols = {"agent_1": "vertex", "agent_2": "vertex"}

        for t in range(1, 4):
            fm.record_tick(t, pos_b, pos_a, e_b, e_a, acts, succ, cols, {}, {})

        self.assertGreaterEqual(fm.metrics["agent_1"].starvation_events, 1)
        self.assertEqual(fm.metrics["agent_1"].ticks_blocked, 3)


class TestMultiAgentCoordinator(unittest.TestCase):
    """Unit tests for MultiAgentCoordinator de-confliction and load balancing."""

    def test_suggests_alternative_move_when_cell_contested(self) -> None:
        coord = MultiAgentCoordinator(enabled=True)
        # agent_1 registers intent for (2, 2)
        coord.register_intent("agent_1", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 2),  # contested
            Action.DOWN: Position(1, 3),   # free alternative
        }

        # agent_2 wants (2, 2)
        alt = coord.suggest_alternative_move("agent_2", Position(1, 2), candidate_actions, dest_map, tick=0)
        self.assertEqual(alt, Action.DOWN)

    def test_resource_load_tracking(self) -> None:
        coord = MultiAgentCoordinator(enabled=True)
        r_pos = Position(3, 3)
        self.assertEqual(coord.get_resource_load(r_pos), 0)

        coord.register_intent("agent_1", Position(2, 3), targeted_resource=r_pos)
        self.assertEqual(coord.get_resource_load(r_pos), 1)

        coord.reset()
        self.assertEqual(coord.get_resource_load(r_pos), 0)


class TestMultiAgentSimulationPhase13_1(unittest.TestCase):
    """Integration tests for Phase 13.1 simulation features."""

    def test_priority_rotation_resolves_symmetric_vertex_collision(self) -> None:
        cfg = MultiAgentWorldConfig(
            width=5, height=5,
            agents=(
                AgentConfig(id="agent_1", initial_x=1, initial_y=2, initial_energy=5, target=Position(2, 2)),
                AgentConfig(id="agent_2", initial_x=3, initial_y=2, initial_energy=5, target=Position(2, 2)),
            ),
            resources=(),
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(cfg, seed=0)
        # Both agents attempt to step into (2, 2)
        res = sim.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})

        # Exactly one agent moves, one bounces (no symmetric deadlock)
        moved_agents = [aid for aid, ar in res.agent_results.items() if ar.moved]
        self.assertEqual(len(moved_agents), 1)
        self.assertEqual(res.vertex_collisions, 1)

    def test_resource_arbitration_at_resource_cell(self) -> None:
        cfg = MultiAgentWorldConfig(
            width=5, height=5,
            agents=(
                AgentConfig(id="agent_1", initial_x=1, initial_y=2, initial_energy=5, target=Position(4, 4)),
                AgentConfig(id="agent_2", initial_x=3, initial_y=2, initial_energy=2, target=Position(4, 4)),
            ),
            resources=(Resource(Position(2, 2), 6, consumable=True),),
            collision_rule="priority_rotation",
            resource_allocation_rule="need_urgency",
        )
        sim = MultiAgentSimulation(cfg, seed=0)
        # Both agents attempt to enter resource cell (2, 2)
        res = sim.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})

        # Under need_urgency, agent_2 (energy 2) has higher urgency than agent_1 (energy 5)
        self.assertTrue(res.agent_results["agent_2"].moved)
        self.assertFalse(res.agent_results["agent_1"].moved)
        self.assertEqual(sim.cumulative_resource_contentions, 1)

    def test_cooperative_discovery_pool(self) -> None:
        cfg = MultiAgentWorldConfig(
            width=6, height=6,
            agents=(
                AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=5),
                AgentConfig(id="agent_2", initial_x=5, initial_y=5, initial_energy=5),
            ),
            resources=(Resource(Position(1, 0), 4, False),),
            cooperative_sharing=True,
        )
        sim = MultiAgentSimulation(cfg, seed=0)
        # Tick 1: agent_1 sees (1, 0)
        sim.step({"agent_1": Action.UP, "agent_2": Action.UP})

        obs2 = sim.observe_agent("agent_2")
        # agent_2 cannot see (1, 0) directly (dist 9 > 2), but receives it in shared_discoveries
        self.assertEqual(len(obs2.observed_resources), 0)
        self.assertIn(Resource(Position(1, 0), 4, False), obs2.shared_discoveries)


if __name__ == "__main__":
    unittest.main()
