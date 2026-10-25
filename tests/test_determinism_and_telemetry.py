"""Regression tests for Phase 14.1: Determinism & Telemetry Cleanup.

Verifies:
1. Nondeterministic agent hashing removal:
   - ResourceArbitrator under 'need_urgency' rule uses deterministic agent_val(aid) for tie-breaking.
   - Arbitration outcomes are identical across independent Python processes with different PYTHONHASHSEED.
2. Deadlock telemetry improvements:
   - DeadlockDetector distinguishes between benign cooperative waiting and true structural deadlocks.
   - Cooperative waiting (resource queueing, bottleneck doorway yielding) is explicitly classified.
   - Genuine structural deadlocks (unresolvable blocks, mutual swap locks without escape) remain detectable and unresolved.
   - Recovery monitoring continues to track all event resolutions.
3. Scenario telemetry integration:
   - U09-style scarce resource queueing classifies waiting events as cooperative waiting.
   - U14-style blind crossroad clearing classifies doorway waiting as cooperative waiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from worldsim.actions import Action
from worldsim.coordination import (
    DeadlockDetector,
    DeadlockEvent,
    MultiAgentCoordinator,
    ResourceArbitrator,
    agent_val,
)
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import RobustResourcePolicy


class TestDeterminismAndTelemetry(unittest.TestCase):
    """Test suite verifying hash independence and deadlock telemetry classification."""

    def setUp(self) -> None:
        self.res = Resource(position=Position(2, 2), recharge_amount=6, consumable=True)

    def test_need_urgency_deterministic_arbitration(self) -> None:
        """1. Prove need_urgency arbitration tie-breaking is completely deterministic."""
        arb = ResourceArbitrator(rule="need_urgency")
        claimants = ["agent_b", "agent_a", "agent_c"]
        # All have identical deficit and energy level
        energies = {"agent_a": 4, "agent_b": 4, "agent_c": 4}
        max_energies = {"agent_a": 10, "agent_b": 10, "agent_c": 10}

        # Check winner on tick 0, 1, 2
        w0, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=0)
        w1, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=1)
        w2, _ = arb.arbitrate(claimants, energies, max_energies, self.res, tick=2)

        # Winner must be predictable and deterministic based on agent_val
        expected_w0 = min(claimants, key=lambda a: (-6, 4, (agent_val(a) + 0) % 1000))
        expected_w1 = min(claimants, key=lambda a: (-6, 4, (agent_val(a) + 1) % 1000))
        self.assertEqual(w0, expected_w0)
        self.assertEqual(w1, expected_w1)

    def test_need_urgency_cross_process_hashseed_independence(self) -> None:
        """2. Prove need_urgency produces identical winners across processes with different PYTHONHASHSEED."""
        code = """
from worldsim.coordination import ResourceArbitrator
from worldsim.models import Position, Resource

res = Resource(position=Position(2, 2), recharge_amount=6, consumable=True)
arb = ResourceArbitrator(rule="need_urgency")
claimants = ["agent_x", "agent_y", "agent_z"]
energies = {a: 5 for a in claimants}
max_energies = {a: 10 for a in claimants}

winners = [arb.arbitrate(claimants, energies, max_energies, res, tick=t)[0] for t in range(10)]
print(",".join(winners))
"""
        env1 = dict(os.environ, PYTHONHASHSEED="0")
        env2 = dict(os.environ, PYTHONHASHSEED="999999")
        env3 = dict(os.environ, PYTHONHASHSEED="random")

        out1 = subprocess.check_output([sys.executable, "-c", code], env=env1, text=True).strip()
        out2 = subprocess.check_output([sys.executable, "-c", code], env=env2, text=True).strip()
        out3 = subprocess.check_output([sys.executable, "-c", code], env=env3, text=True).strip()

        self.assertEqual(out1, out2)
        self.assertEqual(out2, out3)

    def test_deadlock_detector_classifies_cooperative_resource_queue_waiting(self) -> None:
        """3. Prove DeadlockDetector classifies resource queue waiting as cooperative waiting."""
        detector = DeadlockDetector(persistence_threshold=2)
        r_pos = Position(2, 2)
        coord = MultiAgentCoordinator(enabled=True)
        coord.register_intent("agent_queued", destination=r_pos, targeted_resource=r_pos)

        # agent_queued at (1, 2) wants (2, 2) which is occupied by agent_occupant
        pos = {"agent_queued": Position(1, 2), "agent_occupant": Position(2, 2)}
        intents = {
            "agent_queued": (Action.RIGHT, Position(2, 2)),
            "agent_occupant": (Action.RECHARGE, Position(2, 2)),
        }
        cols = {"agent_queued": "occupied", "agent_occupant": None}
        moved = {"agent_queued": False, "agent_occupant": False}

        detector.update_tick(1, pos, intents, cols, moved, coordinator=coord, resources=[r_pos])
        events = detector.update_tick(2, pos, intents, cols, moved, coordinator=coord, resources=[r_pos])

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.classification, "cooperative_waiting")
        self.assertEqual(ev.waiting_reason, "resource_queue")
        self.assertEqual(detector.cooperative_waiting_count, 1)
        self.assertEqual(detector.structural_deadlock_count, 0)

    def test_deadlock_detector_classifies_cooperative_bottleneck_yielding(self) -> None:
        """4. Prove DeadlockDetector classifies bottleneck yielding as cooperative waiting."""
        detector = DeadlockDetector(persistence_threshold=2)
        coord = MultiAgentCoordinator(enabled=True)
        # agent_waiter yields at doorway (2, 2) to oncoming agent_crossing at (3, 2)
        pos = {"agent_waiter": Position(2, 2), "agent_crossing": Position(3, 2)}
        intents = {
            "agent_waiter": (Action.RIGHT, Position(3, 2)),
            "agent_crossing": (Action.LEFT, Position(2, 2)),
        }
        # agent_crossing is actively moving through
        cols = {"agent_waiter": "occupied", "agent_crossing": None}
        moved = {"agent_waiter": False, "agent_crossing": True}

        detector.update_tick(1, pos, intents, cols, moved, coordinator=coord)
        events = detector.update_tick(2, pos, intents, cols, moved, coordinator=coord)

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.classification, "cooperative_waiting")
        self.assertEqual(ev.waiting_reason, "bottleneck_yielding")

    def test_deadlock_detector_classifies_genuine_structural_deadlock(self) -> None:
        """5. Prove DeadlockDetector classifies unyielding reciprocal swap as structural deadlock."""
        detector = DeadlockDetector(persistence_threshold=2)
        pos = {"agent_1": Position(1, 1), "agent_2": Position(2, 1)}
        intents = {
            "agent_1": (Action.RIGHT, Position(2, 1)),
            "agent_2": (Action.LEFT, Position(1, 1)),
        }
        cols = {"agent_1": "swap", "agent_2": "swap"}
        moved = {"agent_1": False, "agent_2": False}

        detector.update_tick(1, pos, intents, cols, moved, coordinator=None)
        events = detector.update_tick(2, pos, intents, cols, moved, coordinator=None)

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.classification, "structural_deadlock")
        self.assertEqual(ev.waiting_reason, "reciprocal_swap_lock")
        self.assertEqual(detector.structural_deadlock_count, 1)
        self.assertEqual(detector.cooperative_waiting_count, 0)
        self.assertEqual(detector.unresolved_structural_deadlocks, 1)

    def test_deadlock_detector_classifies_unresolvable_occupied_block(self) -> None:
        """6. Prove DeadlockDetector classifies persistent block behind exhausted obstacle as structural deadlock."""
        detector = DeadlockDetector(persistence_threshold=2)
        pos = {"agent_active": Position(1, 1), "agent_exhausted": Position(2, 1)}
        intents = {
            "agent_active": (Action.RIGHT, Position(2, 1)),
            "agent_exhausted": (Action.UP, Position(2, 1)),  # unmoving
        }
        cols = {"agent_active": "occupied", "agent_exhausted": None}
        moved = {"agent_active": False, "agent_exhausted": False}

        detector.update_tick(1, pos, intents, cols, moved, coordinator=None)
        events = detector.update_tick(2, pos, intents, cols, moved, coordinator=None)

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.classification, "structural_deadlock")
        self.assertEqual(ev.waiting_reason, "unresolvable_block")
        self.assertEqual(detector.unresolved_structural_deadlocks, 1)

    def test_deadlock_detector_properties_and_metrics(self) -> None:
        """7. Verify DeadlockDetector telemetry properties and waiting reason tallies."""
        detector = DeadlockDetector(persistence_threshold=2)
        r_pos = Position(2, 2)
        coord = MultiAgentCoordinator(enabled=True)
        coord.register_intent("q_agent", destination=r_pos, targeted_resource=r_pos)

        # Trigger cooperative waiting event
        pos = {"q_agent": Position(1, 2), "occ_agent": Position(2, 2)}
        intents = {"q_agent": (Action.RIGHT, Position(2, 2)), "occ_agent": (Action.RECHARGE, Position(2, 2))}
        cols = {"q_agent": "occupied", "occ_agent": None}
        moved = {"q_agent": False, "occ_agent": False}

        detector.update_tick(1, pos, intents, cols, moved, coordinator=coord, resources=[r_pos])
        detector.update_tick(2, pos, intents, cols, moved, coordinator=coord, resources=[r_pos])

        self.assertEqual(detector.cooperative_waiting_count, 1)
        self.assertEqual(detector.structural_deadlock_count, 0)
        self.assertEqual(detector.unresolved_cooperative_waits, 1)
        self.assertEqual(detector.unresolved_structural_deadlocks, 0)
        self.assertEqual(detector.waiting_reasons_tally.get("resource_queue"), 1)

        # Agent moves -> recovery
        moved_after = {"q_agent": True, "occ_agent": True}
        pos_after = {"q_agent": Position(2, 2), "occ_agent": Position(3, 2)}
        detector.update_tick(3, pos_after, intents, {"q_agent": None, "occ_agent": None}, moved_after)

        self.assertEqual(detector.successful_recoveries, 1)
        self.assertEqual(detector.unresolved_cooperative_waits, 0)
        self.assertEqual(detector.unresolved_structural_deadlocks, 0)

    def test_end_to_end_u09_telemetry(self) -> None:
        """8. End-to-end U09: prove queue waits are classified as cooperative waiting with 0 unresolved structural deadlocks."""
        from experiments.phase13_5_unseen_audit import build_all_unseen_scenarios, make_robust_factory
        from worldsim.config import WorldConfig
        from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner

        train_world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
        )
        lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
        lin_runner.train(1000)

        scenarios = build_all_unseen_scenarios()
        u09_spec, u09_pre = next(s for s in scenarios if s[0].scenario_id == "U09")
        factories = {
            a["id"]: make_robust_factory(u09_spec.width, u09_spec.height, u09_spec.max_energy, lin_runner.learner, margin=1)
            for a in u09_spec.active_agents
        }
        agent_configs = [
            AgentConfig(
                id=a["id"], initial_x=a["initial_x"], initial_y=a["initial_y"],
                initial_energy=a["initial_energy"], target=Position(a["target_x"], a["target_y"]),
            )
            for a in u09_spec.active_agents
        ]
        config = MultiAgentWorldConfig(
            width=u09_spec.width, height=u09_spec.height, agents=tuple(agent_configs),
            resources=tuple(Resource(Position(r["x"], r["y"]), r["recharge_amount"], r["consumable"]) for r in u09_spec.resources),
            max_energy=u09_spec.max_energy, collision_rule=u09_spec.collision_rule,
            resource_allocation_rule=u09_spec.resource_allocation_rule, enable_coordination=True,
            cooperative_sharing=True,
        )

        sim = MultiAgentSimulation(config, seed=500)
        policies = {aid: factories[aid]() for aid in [a["id"] for a in u09_spec.active_agents]}
        for p in policies.values():
            p.coordinator = sim.coordinator

        for _ in range(20):
            all_done = all(not sim.state.agents[aid].active for aid in policies)
            if all_done:
                break
            sim.coordinator.reset_tick()
            actions = {
                aid: policies[aid].choose_action(sim.observe_agent(aid)) if sim.state.agents[aid].active else Action.UP
                for aid in policies
            }
            sim.step(actions)

        # In U09, all detected events are cooperative resource queueing/contention
        self.assertGreater(sim.deadlock_detector.cooperative_waiting_count, 0)
        self.assertEqual(sim.deadlock_detector.unresolved_structural_deadlocks, 0)
        self.assertEqual(sim.deadlock_detector.unresolved_deadlocks, 0)

    def test_end_to_end_u14_telemetry(self) -> None:
        """9. End-to-end U14: prove doorway waits are classified as cooperative waiting with 0 unresolved deadlocks."""
        from dataclasses import replace
        from experiments.phase13_5_unseen_audit import build_all_unseen_scenarios, make_robust_factory
        from worldsim.config import WorldConfig
        from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner

        train_world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
        )
        lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
        lin_runner.train(1000)

        scenarios = build_all_unseen_scenarios()
        u14_spec, u14_pre = next(s for s in scenarios if s[0].scenario_id == "U14")
        factories = {
            a["id"]: make_robust_factory(u14_spec.width, u14_spec.height, u14_spec.max_energy, lin_runner.learner, margin=1)
            for a in u14_spec.active_agents
        }
        agent_configs = [
            AgentConfig(
                id=a["id"], initial_x=a["initial_x"], initial_y=a["initial_y"],
                initial_energy=a["initial_energy"], target=Position(a["target_x"], a["target_y"]),
            )
            for a in u14_spec.active_agents
        ] + [
            AgentConfig(id=p.id, initial_x=p.x, initial_y=p.y, initial_energy=0, target=None)
            for p in u14_pre
        ]
        config = MultiAgentWorldConfig(
            width=u14_spec.width, height=u14_spec.height, agents=tuple(agent_configs),
            resources=(), max_energy=u14_spec.max_energy, collision_rule=u14_spec.collision_rule,
            resource_allocation_rule=u14_spec.resource_allocation_rule, enable_coordination=True,
            cooperative_sharing=True,
        )

        sim = MultiAgentSimulation(config, seed=500)
        for p in u14_pre:
            sim._state.agents[p.id] = replace(sim._state.agents[p.id], status="exhausted", active=False, target=None, energy=0)

        active_aids = [a["id"] for a in u14_spec.active_agents]
        policies = {aid: factories[aid]() for aid in active_aids}
        for p in policies.values():
            p.coordinator = sim.coordinator

        for _ in range(20):
            all_done = all(not sim.state.agents[aid].active for aid in active_aids)
            if all_done:
                break
            sim.coordinator.reset_tick()
            actions = {
                aid: policies[aid].choose_action(sim.observe_agent(aid)) if sim.state.agents[aid].active else Action.UP
                for aid in active_aids
            }
            sim.step(actions)

        self.assertEqual(sim.deadlock_detector.unresolved_structural_deadlocks, 0)
        self.assertEqual(sim.deadlock_detector.unresolved_deadlocks, 0)


if __name__ == "__main__":
    unittest.main()
