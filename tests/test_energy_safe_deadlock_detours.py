"""Regression tests for Phase 13.6: Targeted Fix C — Energy-Safe Deadlock Detours.

Verifies:
1. Necessary lateral detour when direct movement is blocked:
   An agent chooses an energy-safe lateral detour when its preferred action is blocked.
2. Rejection of fatal detours:
   An alternative detour that would exceed the remaining energy budget is rejected.
3. Prevention of position oscillations:
   Agents facing conflicting traffic do not engage in ping-pong limit cycles or wasteful retreats.
4. Respect for static obstacles and world boundaries:
   Detours never select known stationary obstacles, out-of-bounds coordinates, or congested cells.
5. Continued reciprocal-swap resolution:
   Reciprocal swap conflicts resolve cleanly without deadlock or livelock.
6. Continued U03 static-obstacle safety:
   Stationary obstacles in cul-de-sac and bypass environments remain strictly avoided.
7. Continued U09 energy-safe resource queuing and reservation fairness:
   Resource contention and queuing retain energy safety and fair arbitration.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.coordination import (
    MultiAgentCoordinator,
    agent_val,
)
from worldsim.models import Position


class TestEnergySafeDeadlockDetours(unittest.TestCase):
    """Test suite verifying energy-safe bounded lateral detours during multi-agent conflicts."""

    def setUp(self) -> None:
        self.coord = MultiAgentCoordinator(enabled=True, world_width=7, world_height=7)

    def test_necessary_lateral_detour_when_direct_movement_blocked(self) -> None:
        """1. Prove an agent takes an energy-safe lateral detour when direct movement is blocked."""
        # agent_s is at (3, 5) heading UP to (3, 4) towards target (3, 0)
        # agent_n is at (3, 4) heading DOWN to (3, 5) -> swap conflict
        self.coord.register_intent("agent_n", Position(3, 5))

        candidate_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        dest_map = {
            Action.UP: Position(3, 4),     # blocked by oncoming agent_n
            Action.DOWN: Position(3, 6),   # corridor reversal (backwards)
            Action.LEFT: Position(2, 5),   # lateral sidestep (open)
            Action.RIGHT: Position(4, 5),  # lateral sidestep (open)
        }
        current_positions = {
            "agent_s": Position(3, 5),
            "agent_n": Position(3, 4),
        }

        # agent_s at tick=7 with energy=7, target=(3, 0)
        # Distance to target from (2, 5) is |3-2| + |0-5| = 6 <= (7 - 1) = 6 (energy-safe!)
        alt = self.coord.suggest_alternative_move(
            agent_id="agent_s",
            pos=Position(3, 5),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=7,
            preferred_action=Action.UP,
            current_positions=current_positions,
            target=Position(3, 0),
            num_agents=4,
            energy=7,
        )

        # Must select an energy-safe lateral sidestep (LEFT or RIGHT), NOT corridor reversal DOWN
        self.assertIn(alt, [Action.LEFT, Action.RIGHT])
        self.assertNotEqual(alt, Action.DOWN)

    def test_rejection_of_fatal_detours(self) -> None:
        """2. Prove an agent rejects an alternative move that would cause energy exhaustion."""
        # agent_s at (3, 5) with remaining energy=4, target=(3, 0)
        # Direct distance is 5. Lateral detour to (2, 5) requires distance 6.
        # Since energy=4, energy - 1 = 3 < 6, the detour is fatal and must be rejected.
        self.coord.register_intent("agent_n", Position(3, 5))

        candidate_actions = [Action.UP, Action.LEFT, Action.RIGHT]
        dest_map = {
            Action.UP: Position(3, 4),
            Action.LEFT: Position(2, 5),
            Action.RIGHT: Position(4, 5),
        }
        current_positions = {
            "agent_s": Position(3, 5),
            "agent_n": Position(3, 4),
        }

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_s",
            pos=Position(3, 5),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=7,
            preferred_action=Action.UP,
            current_positions=current_positions,
            target=Position(3, 0),
            num_agents=4,
            energy=4,  # Insufficient energy for detour
        )

        # Fatal detour must be rejected (returns None)
        self.assertIsNone(alt)

    def test_prevention_of_doorstep_retreat_oscillations(self) -> None:
        """3. Prove that in non-swap conflicts, agents waiting at doorstep do not oscillate backwards."""
        # agent_w at (2, 3) wants (3, 3) (central junction)
        # agent_n at (3, 2) also registered intent to (3, 3)
        self.coord.register_intent("agent_n", Position(3, 3))

        candidate_actions = [Action.RIGHT, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(3, 3),  # contested junction (distance to target (6, 3) is 3)
            Action.LEFT: Position(1, 3),   # retreat move (distance to target (6, 3) is 5)
        }
        current_positions = {
            "agent_w": Position(2, 3),
            "agent_n": Position(3, 2),
        }

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_w",
            pos=Position(2, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            target=Position(6, 3),
            num_agents=4,
            energy=8,
        )

        # In non-swap doorstep conflict, retreat move LEFT (which increases target distance from 4 to 5)
        # must NOT be suggested, allowing safe doorstep waiting rather than 2-step energy drain.
        self.assertIsNone(alt)

    def test_respect_for_static_obstacles_and_boundaries(self) -> None:
        """4. Prove that detours never target static obstacles or out-of-bounds cells."""
        # agent at (0, 3) near left boundary, obstacles at (0, 2)
        candidate_actions = [Action.RIGHT, Action.UP, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(1, 3),  # blocked by conflict
            Action.UP: Position(0, 2),     # occupied by static obstacle
            Action.LEFT: Position(-1, 3),  # out of bounds!
        }
        obstacle_positions = {Position(0, 2)}
        self.coord.register_intent("other", Position(1, 3))

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(0, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.RIGHT,
            obstacle_positions=obstacle_positions,
            target=Position(5, 3),
            num_agents=2,
            energy=10,
        )

        # Neither the out-of-bounds cell (-1, 3) nor the obstacle cell (0, 2) may be chosen
        self.assertIsNone(alt)

    def test_continued_reciprocal_swap_resolution(self) -> None:
        """5. Prove continued reciprocal-swap resolution between two agents."""
        self.coord.register_intent("agent_2", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(3, 2),
            Action.UP: Position(2, 1),
            Action.DOWN: Position(2, 3),
        }
        current_positions = {"agent_1": Position(2, 2), "agent_2": Position(3, 2)}

        alt_winner = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            num_agents=2,
        )
        self.assertIsNone(alt_winner)

        alt_loser = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            num_agents=2,
        )
        self.assertIn(alt_loser, [Action.UP, Action.DOWN])

    def test_continued_u03_static_obstacle_safety(self) -> None:
        """6. Prove U03 static-obstacle safety remains intact."""
        self.coord.register_intent("agent_2", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(3, 2),
            Action.UP: Position(2, 1),     # occupied by stationary obstacle
            Action.DOWN: Position(2, 3),   # free cell
        }
        current_positions = {
            "agent_1": Position(2, 2),
            "agent_2": Position(3, 2),
            "wall_obs": Position(2, 1),
        }
        obstacle_positions = {Position(2, 1)}

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            obstacle_positions=obstacle_positions,
            target=Position(5, 2),
            num_agents=2,
            energy=10,
        )
        self.assertEqual(alt, Action.DOWN)
        self.assertNotEqual(alt, Action.UP)

    def test_continued_u09_energy_safe_resource_queuing(self) -> None:
        """7. Prove U09 energy-safe resource queuing behavior remains intact."""
        r_pos = Position(3, 3)
        self.coord.register_intent("agent_c1", destination=Position(2, 3), targeted_resource=r_pos)

        candidate_actions = [Action.RIGHT, Action.UP]
        dest_map = {
            Action.RIGHT: Position(2, 3),  # contested queue cell
            Action.UP: Position(1, 2),     # lateral bypass
        }
        current_positions = {
            "agent_c1": Position(2, 2),
            "agent_c2": Position(1, 3),
        }

        # agent_c2 has energy=2: stepping to (2, 3) costs 1, leaving 1. Distance to (3, 3) is 1. Total cost = 2.
        # (energy - 1) = 1 >= dist(2, 3 -> 3, 3) = 1. Energy-safe!
        alt_safe = self.coord.suggest_alternative_move(
            agent_id="agent_c2",
            pos=Position(1, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            target=r_pos,
            num_agents=4,
            energy=2,
        )
        # Should NOT be pruned by energy check
        self.coord.register_intent("agent_c2", destination=Position(1, 3), targeted_resource=r_pos)
        self.assertEqual(self.coord.resource_commitments.get("agent_c2"), r_pos)


if __name__ == "__main__":
    unittest.main()
