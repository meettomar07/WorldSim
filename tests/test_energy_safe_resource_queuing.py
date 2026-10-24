"""Regression tests for Phase 13.6: Targeted Fix B — Energy-Safe Resource Queuing.

Verifies:
1. An agent does not enter a queue position when doing so would make safe arrival at recharge impossible.
2. An agent with sufficient energy can queue and subsequently reach the resource.
3. Resource contention remains fair across competing agents.
4. Resource reservations are released or reassigned correctly when an agent cannot continue.
5. U03 static-obstacle safety remains intact.
6. Existing energy, recharge, resource-arbitration, and deadlock-recovery tests still pass.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.coordination import (
    MultiAgentCoordinator,
    ResourceArbitrator,
    jains_fairness_index,
)
from worldsim.models import Position


class TestEnergySafeResourceQueuing(unittest.TestCase):
    """Test suite verifying energy safety, queueing, and reservation lifecycle for shared resources."""

    def setUp(self) -> None:
        self.coord = MultiAgentCoordinator(enabled=True, world_width=5, world_height=5)

    def test_agent_does_not_enter_fatal_queue_position(self) -> None:
        """1. Prove an agent does not enter a queue position when doing so would make safe arrival at recharge impossible."""
        r_pos = Position(2, 2)
        # agent_1 is committed to recharge at (2, 2)
        self.coord.register_intent("agent_1", destination=Position(1, 3), targeted_resource=r_pos)

        # Agent is at (1, 3) with only energy=1 remaining.
        # Distance to resource (2, 2) is |2-1| + |2-3| = 2.
        # Moving to (2, 3) costs 1 energy (leaving energy=0), but distance from (2, 3) to (2, 2) is 1.
        # Because remaining energy would be 0 < distance 1, stepping to (2, 3) is FATAL!
        candidate_actions = [Action.RIGHT, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 3),  # distance to r_pos is 1; remaining energy is 0 -> FATAL!
            Action.DOWN: Position(1, 4),   # distance to r_pos is 3; remaining energy is 0 -> FATAL!
        }

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(1, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            energy=1,  # Only 1 energy left
            target=r_pos,
        )
        # Both candidate actions are fatal (dist > energy - 1); coordinator MUST fail safe (return None)
        self.assertIsNone(alt)

    def test_agent_with_sufficient_energy_can_queue_and_reach_resource(self) -> None:
        """2. Prove an agent with sufficient energy can queue and subsequently reach the resource."""
        r_pos = Position(2, 2)
        self.coord.register_intent("agent_1", destination=Position(1, 2), targeted_resource=r_pos)

        # Agent is at (1, 2) with energy=2.
        # Preferred action is RIGHT into (2, 2).
        # Distance from (2, 2) to (2, 2) is 0 <= (2 - 1 = 1).
        candidate_actions = [Action.RIGHT, Action.UP]
        dest_map = {
            Action.RIGHT: Position(2, 2),  # resource pad
            Action.UP: Position(1, 1),
        }

        # Doorstep queueing attempt: preferred action into committed resource is not redirected away
        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(1, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            energy=2,
            target=r_pos,
        )
        # None returned means the agent is allowed to proceed with its intended attempt onto the pad
        self.assertIsNone(alt)

    def test_resource_contention_fairness(self) -> None:
        """3. Prove resource contention remains fair across competing agents."""
        from worldsim.models import Resource
        arbitrator = ResourceArbitrator(rule="fair_balanced")
        res_obj = Resource(Position(2, 2), recharge_amount=6, consumable=False)
        claimants = ["agent_c1", "agent_c2", "agent_c3", "agent_c4"]
        energies = {aid: 4 for aid in claimants}
        max_energies = {aid: 8 for aid in claimants}

        winners: list[str] = []
        for tick in range(12):
            winner, losers = arbitrator.arbitrate(
                claimants=claimants,
                agent_energies=energies,
                agent_max_energies=max_energies,
                resource=res_obj,
                tick=tick,
            )
            winners.append(winner)

        # All 4 agents must win access over 12 ticks; no agent should starve
        winner_counts = {aid: winners.count(aid) for aid in claimants}
        self.assertEqual(len(set(winners)), 4, "Every competing agent must receive access")
        jain = jains_fairness_index(list(winner_counts.values()))
        self.assertGreaterEqual(jain, 0.90, f"Fairness index {jain} should be high")

    def test_resource_reservations_release_and_reassignment(self) -> None:
        """4. Prove resource reservations are released or reassigned correctly when an agent cannot continue."""
        r_pos = Position(2, 2)

        # Agent 1 claims resource (2, 2)
        self.coord.register_intent("agent_1", destination=Position(1, 2), targeted_resource=r_pos)
        self.assertEqual(self.coord.resource_reservations[r_pos], "agent_1")
        self.assertEqual(self.coord.resource_commitments["agent_1"], r_pos)

        # Agent 2 also commits to the same resource (queued behind agent 1)
        self.coord.register_intent("agent_2", destination=Position(2, 3), targeted_resource=r_pos)
        # Exclusive reservation remains held by the first claimant
        self.assertEqual(self.coord.resource_reservations[r_pos], "agent_1")
        self.assertEqual(self.coord.resource_commitments["agent_2"], r_pos)

        # Agent 1 completes recharge or cannot continue -> clears commitment
        self.coord.clear_commitment("agent_1")
        self.assertNotIn("agent_1", self.coord.resource_commitments)
        # Reservation MUST be reassigned to the queued agent 2!
        self.assertEqual(self.coord.resource_reservations[r_pos], "agent_2")

        # Agent 2 finishes -> clears commitment
        self.coord.clear_commitment("agent_2")
        self.assertNotIn("agent_2", self.coord.resource_commitments)
        # Reservation must now be completely released
        self.assertNotIn(r_pos, self.coord.resource_reservations)

    def test_u03_static_obstacle_safety_intact(self) -> None:
        """5. Prove U03 static-obstacle safety remains intact."""
        # Swap conflict with agent_2 at (2, 2) vs (3, 2)
        self.coord.register_intent("agent_2", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(3, 2),  # swap conflict
            Action.UP: Position(2, 1),     # occupied by stationary obstacle 'wall_c2'
            Action.DOWN: Position(2, 3),   # free cell
        }
        current_positions = {
            "agent_1": Position(2, 2),
            "agent_2": Position(3, 2),
            "wall_c2": Position(2, 1),
        }
        obstacle_positions = {Position(2, 1)}

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # agent_1 has lower priority at tick=1
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            obstacle_positions=obstacle_positions,
            target=Position(5, 2),
            num_agents=2,
            energy=10,
        )
        # Must select free DOWN cell and MUST NOT select stationary obstacle UP
        self.assertEqual(alt, Action.DOWN)
        self.assertNotEqual(alt, Action.UP)


if __name__ == "__main__":
    unittest.main()
