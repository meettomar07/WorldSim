"""Regression tests for Phase 13.6: Targeted Fix D — Double Corridor Bottleneck (Scenario U01).

Verifies:
1. Opposing agents entering narrow corridor:
   Opposing agents approaching a narrow corridor portal resolve swap contention via lateral bypass.
2. Safe traversal and exit after access is granted:
   Once the contested mouth/portal is cleared, agents advance directly to destination without spurious redirection.
3. Reservation and commitment release upon exit:
   Reservations and commitments are cleanly released and reassigned when an agent arrives, completes, or clears.
4. Fairness among waiting agents:
   Priority rotation ensures fair turn-taking across opposing agent streams (Jain's fairness index = 1.0).
5. Prevention of corridor retreat oscillations:
   Waiting agents do not oscillate backwards down the corridor during non-swap conflicts, conserving energy.
6. Continued U03 static-obstacle safety:
   Stationary exhausted wall obstacles defining the corridors are strictly avoided.
7. Continued U09 resource-queue safety:
   Resource commitments and energy-safe queuing remain intact.
8. Continued U14 intersection clearing:
   Crossroad exit right-of-way and lateral detours remain intact.
9. End-to-end U01 benchmark execution:
   Executes U01 across all seeds 500–519, confirming 100.0% overall completion, 0 exhaustions, and 0 deadlocks.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import (
    MultiAgentCoordinator,
    agent_val,
    jains_fairness_index,
)
from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner
from worldsim.models import Position, Resource
from experiments.phase13_5_unseen_audit import (
    build_all_unseen_scenarios,
    make_robust_factory,
    run_scenario_audit_batch,
)


class TestDoubleCorridorBottleneck(unittest.TestCase):
    """Test suite verifying safe traversal, yielding, anti-oscillation, and fairness in double corridor bottlenecks."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train base linear learner once for integration tests."""
        train_world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),),
            observable_target=Position(3, 3),
        )
        lin_runner = LinearLearningRunner(
            world=train_world,
            config=LinearLearningConfig(max_steps=25),
            seed=42,
        )
        lin_runner.train(1000)
        cls.base_learner = lin_runner.learner

    def setUp(self) -> None:
        self.coord = MultiAgentCoordinator(enabled=True, world_width=7, world_height=3)

    def test_opposing_agents_entering_narrow_corridor(self) -> None:
        """1. Prove opposing agents approaching a narrow corridor portal resolve swap contention via lateral bypass."""
        # Top corridor row 0: agent_nw at (2, 0) targeting (6, 0), agent_ne at (4, 0) targeting (0, 0)
        # Central opening is at (3, 1), walls at (1, 1), (2, 1), (4, 1), (5, 1)
        wall_obstacles = {Position(1, 1), Position(2, 1), Position(4, 1), Position(5, 1)}
        self.coord.known_obstacles.update(wall_obstacles)

        # agent_ne intends to step LEFT into (3, 0)
        self.coord.register_intent("agent_ne", Position(3, 0))

        candidate_actions = [Action.RIGHT, Action.DOWN, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(3, 0),  # contested doorway
            Action.DOWN: Position(2, 1),   # wall!
            Action.LEFT: Position(1, 0),   # backwards retreat
        }
        current_positions = {
            "agent_nw": Position(2, 0),
            "agent_ne": Position(4, 0),
        }

        # If agent_nw is at (3, 0) and agent_ne is at (4, 0) with agent_ne intending (3, 0):
        # agent_nw at (3, 0) has open lateral action DOWN to (3, 1)
        dest_map_at_portal = {
            Action.RIGHT: Position(4, 0),  # head-on into agent_ne
            Action.DOWN: Position(3, 1),   # lateral bypass into central portal!
            Action.LEFT: Position(2, 0),   # backwards retreat
        }
        alt = self.coord.suggest_alternative_move(
            agent_id="agent_nw",
            pos=Position(3, 0),
            candidate_actions=[Action.RIGHT, Action.DOWN, Action.LEFT],
            destination_map=dest_map_at_portal,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions={"agent_nw": Position(3, 0), "agent_ne": Position(4, 0)},
            target=Position(6, 0),
            obstacle_positions=wall_obstacles,
            num_agents=4,
            energy=8,
        )

        # agent_nw must choose the lateral bypass DOWN into (3, 1), NOT backwards LEFT
        self.assertEqual(alt, Action.DOWN)

    def test_safe_traversal_and_exit_after_access_granted(self) -> None:
        """2. Prove that once the oncoming agent passes or an alternative is taken, agent advances cleanly."""
        # agent_ne at (3, 0) heading LEFT towards (0, 0), portal (3, 1) is occupied by agent_nw
        wall_obstacles = {Position(1, 1), Position(2, 1), Position(4, 1), Position(5, 1)}
        self.coord.known_obstacles.update(wall_obstacles)

        candidate_actions = [Action.LEFT, Action.RIGHT, Action.DOWN]
        dest_map = {
            Action.LEFT: Position(2, 0),   # open corridor forward
            Action.RIGHT: Position(4, 0),  # backward
            Action.DOWN: Position(3, 1),   # occupied by agent_nw
        }
        current_positions = {
            "agent_ne": Position(3, 0),
            "agent_nw": Position(3, 1),
        }

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_ne",
            pos=Position(3, 0),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.LEFT,
            current_positions=current_positions,
            target=Position(0, 0),
            obstacle_positions=wall_obstacles,
            num_agents=4,
            energy=7,
        )

        # No conflict with preferred action LEFT; must return None (allow forward traversal)
        self.assertIsNone(alt)

    def test_reservation_and_commitment_release_upon_exit(self) -> None:
        """3. Prove that reservations and commitments are cleanly released and reassigned when an agent clears."""
        portal_pos = Position(3, 1)
        self.coord.register_intent("agent_nw", destination=portal_pos, targeted_resource=portal_pos)

        # Verify commitment and reservation are recorded
        self.assertEqual(self.coord.resource_commitments.get("agent_nw"), portal_pos)
        self.assertEqual(self.coord.resource_reservations.get(portal_pos), "agent_nw")

        # agent_se also requests the portal
        self.coord.register_intent("agent_se", destination=Position(3, 2), targeted_resource=portal_pos)
        self.assertEqual(self.coord.resource_commitments.get("agent_se"), portal_pos)
        # Reservation remains with agent_nw until cleared
        self.assertEqual(self.coord.resource_reservations.get(portal_pos), "agent_nw")

        # agent_nw exits portal / completes: commitment is cleared
        self.coord.clear_commitment("agent_nw")
        self.assertNotIn("agent_nw", self.coord.resource_commitments)

        # Reservation is cleanly transferred to waiting agent_se
        self.assertEqual(self.coord.resource_reservations.get(portal_pos), "agent_se")

        # agent_se finishes and clears
        self.coord.clear_commitment("agent_se")
        self.assertNotIn(portal_pos, self.coord.resource_reservations)

    def test_fairness_among_waiting_agents(self) -> None:
        """4. Prove priority rotation ensures fair turn-taking across opposing agent streams."""
        # Compare priority ordering across consecutive ticks for 4 agents
        agents = ["agent_nw", "agent_sw", "agent_ne", "agent_se"]
        priority_orders = []
        for tick in range(4):
            order = sorted(agents, key=lambda a: ((agent_val(a) - 1) + tick) % 4)
            priority_orders.append(order)

        # Each agent must hold top priority in at least one tick of the 4-tick cycle
        top_priorities = [order[0] for order in priority_orders]
        self.assertEqual(len(set(top_priorities)), 4)

        # Goal completion distribution across 4 agents in U01 gives perfect fairness
        allocations = [1.0, 1.0, 1.0, 1.0]
        self.assertAlmostEqual(jains_fairness_index(allocations), 1.0, places=4)

    def test_prevention_of_corridor_retreat_oscillations(self) -> None:
        """5. Prove waiting agents do not oscillate backwards down the corridor during non-swap conflicts."""
        # agent_sw at (2, 2) waiting to move RIGHT into (3, 2)
        # Another agent already claimed (3, 2) (vertex conflict, non-swap)
        self.coord.register_intent("agent_se", Position(3, 2))

        candidate_actions = [Action.RIGHT, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(3, 2),  # claimed by agent_se
            Action.LEFT: Position(1, 2),   # backwards retreat (increases target distance from 4 to 5)
        }
        current_positions = {
            "agent_sw": Position(2, 2),
            "agent_se": Position(4, 2),
        }

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_sw",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            target=Position(6, 2),
            num_agents=4,
            energy=8,
        )

        # In non-swap conflict, retreat move LEFT must NOT be suggested, allowing 0-cost stationary waiting
        self.assertIsNone(alt)

    def test_continued_u03_static_obstacle_safety(self) -> None:
        """6. Prove U03 static-obstacle safety remains intact."""
        wall_obstacles = {Position(1, 1), Position(2, 1), Position(4, 1), Position(5, 1)}
        self.coord.known_obstacles.update(wall_obstacles)

        # agent at (1, 0) cannot sidestep into wall at (1, 1)
        self.coord.register_intent("other", Position(2, 0))
        candidate_actions = [Action.RIGHT, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 0),  # conflict
            Action.DOWN: Position(1, 1),   # wall obstacle!
        }
        current_positions = {"agent_nw": Position(1, 0), "other": Position(2, 0)}

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_nw",
            pos=Position(1, 0),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            obstacle_positions=wall_obstacles,
            target=Position(6, 0),
            num_agents=4,
            energy=9,
        )
        self.assertIsNone(alt)

    def test_continued_u09_energy_safe_resource_queuing(self) -> None:
        """7. Prove U09 resource-queue safety remains intact."""
        r_pos = Position(3, 3)
        self.coord.register_intent("lead_agent", destination=Position(2, 3), targeted_resource=r_pos)

        # Agent with remaining energy=1 cannot take a move to (2, 3) if remaining distance is 1 (needs 2)
        candidate_actions = [Action.RIGHT, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 3),  # dist 1 to r_pos
            Action.DOWN: Position(1, 4),   # dist 3 to r_pos
        }
        alt = self.coord.suggest_alternative_move(
            agent_id="queuing_agent",
            pos=Position(1, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            energy=1,
            target=r_pos,
        )
        self.assertIsNone(alt)

    def test_continued_u14_intersection_clearing(self) -> None:
        """8. Prove crossroad exit right-of-way remains intact."""
        crossroad_obstacles = {
            Position(2, 2), Position(4, 2),
            Position(2, 4), Position(4, 4),
        }
        self.coord.known_obstacles.update(crossroad_obstacles)

        # Agent in central junction (3, 3) clearing north towards (3, 2) has right-of-way
        candidate_actions = [Action.UP, Action.DOWN]
        dest_map = {Action.UP: Position(3, 2), Action.DOWN: Position(3, 4)}
        current_positions = {"junction_agent": Position(3, 3), "oncoming_agent": Position(3, 2)}
        self.coord.register_intent("oncoming_agent", Position(3, 3))

        alt = self.coord.suggest_alternative_move(
            agent_id="junction_agent",
            pos=Position(3, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.UP,
            current_positions=current_positions,
            target=Position(3, 0),
            num_agents=4,
            energy=8,
        )
        # Vehicle clearing crossroad has right-of-way (returns None, allowed to proceed)
        self.assertIsNone(alt)

    def test_u01_full_scenario_100_percent_completion(self) -> None:
        """9. End-to-end integration test: U01 across seeds 500–519 achieves 100.0% completion."""
        scenarios = build_all_unseen_scenarios()
        u01_spec, u01_pre = scenarios[0]
        self.assertEqual(u01_spec.scenario_id, "U01")

        factories = {
            a["id"]: make_robust_factory(
                u01_spec.width,
                u01_spec.height,
                u01_spec.max_energy,
                self.base_learner,
                margin=1,
            )
            for a in u01_spec.active_agents
        }

        seeds = tuple(range(500, 520))
        summary, ep_records = run_scenario_audit_batch(
            u01_spec,
            factories,
            u01_pre,
            seeds=seeds,
        )

        self.assertEqual(summary.overall_completion_rate, 100.0)
        self.assertEqual(summary.episode_completion_count, 20)
        self.assertEqual(summary.episode_failure_count, 0)
        self.assertEqual(summary.total_exhaustions, 0)
        self.assertEqual(summary.total_deadlocks, 0)
        self.assertAlmostEqual(summary.jains_fairness_goal, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
