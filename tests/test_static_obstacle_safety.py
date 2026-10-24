"""Regression tests for Phase 13.6: Static Obstacle Safety & Sidestep Validation.

Verifies:
1. A sidestep cannot target a known occupied stationary obstacle cell.
2. An out-of-bounds sidestep is rejected.
3. A valid free sidestep remains available when one exists.
4. If every alternative is blocked, the coordinator fails safely rather than deliberately selecting a known-invalid cell.
5. Existing reciprocal-swap resolution and completed-agent yielding still work.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.coordination import MultiAgentCoordinator
from worldsim.models import Position


class TestStaticObstacleSafety(unittest.TestCase):
    """Test suite verifying coordinator sidestep safety against stationary obstacles and bounds."""

    def setUp(self) -> None:
        self.coord = MultiAgentCoordinator(enabled=True, world_width=6, world_height=6)

    def test_sidestep_cannot_target_stationary_obstacle_cell(self) -> None:
        """1. Prove a sidestep cannot target a known occupied stationary obstacle cell."""
        # agent_1 is at (2, 2) aiming for (3, 2)
        # agent_2 is at (3, 2) aiming for (2, 2) -> swap conflict!
        self.coord.register_intent("agent_2", Position(2, 2))

        # agent_1 has candidates: preferred RIGHT (3, 2), alternative UP (2, 1), alternative DOWN (2, 3)
        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(3, 2),  # swap conflict
            Action.UP: Position(2, 1),     # occupied by stationary exhausted obstacle 'wall_obs'
            Action.DOWN: Position(2, 3),   # completely free
        }
        current_positions = {
            "agent_1": Position(2, 2),
            "agent_2": Position(3, 2),
            "wall_obs": Position(2, 1),    # exhausted obstacle agent with no registered intent
        }
        obstacle_positions = {Position(2, 1)}

        # Agent 1 has lower priority at tick=0 so it must sidestep
        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # tick=1 gives agent_2 higher priority, agent_1 must sidestep
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            target=Position(5, 2),
            num_agents=2,
            obstacle_positions=obstacle_positions,
        )

        # Must select DOWN (2, 3) and NEVER select UP (2, 1)
        self.assertEqual(alt, Action.DOWN)
        self.assertNotEqual(alt, Action.UP)

    def test_out_of_bounds_sidestep_is_rejected(self) -> None:
        """2. Prove an out-of-bounds sidestep is rejected."""
        # agent_2 at top-left corner (0, 0)
        coord = MultiAgentCoordinator(enabled=True, world_width=4, world_height=4)
        # agent_2 has higher priority at tick=1 and claims (1, 0)
        coord.register_intent("agent_2", Position(1, 0))

        # Preferred is RIGHT (1, 0) which is contested with agent_1
        # UP is (0, -1) -> out of bounds (y < 0)
        # LEFT is (-1, 0) -> out of bounds (x < 0)
        # DOWN is (0, 1) -> in bounds and free
        candidate_actions = [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(1, 0),   # contested
            Action.UP: Position(0, -1),     # out of bounds
            Action.LEFT: Position(-1, 0),   # out of bounds
            Action.DOWN: Position(0, 1),    # in bounds
        }

        alt = coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(0, 0),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # agent_1 has lower priority at tick=1
            preferred_action=Action.RIGHT,
            target=Position(3, 3),
            num_agents=2,
        )
        self.assertEqual(alt, Action.DOWN)

        # Test upper boundary out-of-bounds at (3, 3) in 4x4 grid
        coord.reset()
        coord.register_intent("agent_2", Position(2, 3))
        dest_map_bottom_right = {
            Action.LEFT: Position(2, 3),   # contested
            Action.RIGHT: Position(4, 3),  # out of bounds (x >= width=4)
            Action.DOWN: Position(3, 4),   # out of bounds (y >= height=4)
            Action.UP: Position(3, 2),     # valid in bounds
        }
        alt_br = coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(3, 3),
            candidate_actions=[Action.LEFT, Action.RIGHT, Action.DOWN, Action.UP],
            destination_map=dest_map_bottom_right,
            tick=1,  # agent_1 has lower priority at tick=1
            preferred_action=Action.LEFT,
            target=Position(0, 0),
            num_agents=2,
        )
        self.assertEqual(alt_br, Action.UP)

    def test_valid_free_sidestep_remains_available(self) -> None:
        """3. Prove a valid free sidestep remains available when one exists."""
        self.coord.register_intent("agent_2", Position(2, 2))
        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 2),  # contested
            Action.UP: Position(1, 1),     # free
            Action.DOWN: Position(1, 3),   # free
        }
        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(1, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # agent_1 has lower priority at tick=1
            preferred_action=Action.RIGHT,
            target=Position(1, 0),  # UP is closer to target (1, 0)
            num_agents=2,
        )
        # Closer to (1, 0) is UP
        self.assertEqual(alt, Action.UP)

    def test_fails_safely_when_all_alternatives_blocked(self) -> None:
        """4. Prove that if every alternative is blocked, the coordinator fails safely (returns None)."""
        self.coord.register_intent("agent_other", Position(2, 2))
        self.coord.register_intent("agent_up", Position(1, 1))

        # At (1, 2):
        # Preferred RIGHT (2, 2) is contested
        # UP (1, 1) is targeted by agent_up
        # DOWN (1, 3) is blocked by stationary obstacle 'wall'
        # LEFT (0, 2) is marked congested
        self.coord.mark_congested(Position(0, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(2, 2),
            Action.UP: Position(1, 1),
            Action.DOWN: Position(1, 3),
            Action.LEFT: Position(0, 2),
        }
        current_positions = {
            "agent_1": Position(1, 2),
            "wall": Position(1, 3),
        }
        obstacle_positions = {Position(1, 3)}

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(1, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            obstacle_positions=obstacle_positions,
            num_agents=2,
        )
        # Every alternative is invalid; coordinator MUST fail safely and return None
        self.assertIsNone(alt)

    def test_reciprocal_swap_resolution_still_works(self) -> None:
        """5a. Prove reciprocal-swap resolution continues to prioritize and resolve correctly."""
        # agent_1 at (2, 2) wants (3, 2)
        # agent_2 at (3, 2) wants (2, 2)
        self.coord.register_intent("agent_2", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(3, 2),
            Action.UP: Position(2, 1),
            Action.DOWN: Position(2, 3),
        }
        current_positions = {"agent_1": Position(2, 2), "agent_2": Position(3, 2)}

        # Higher priority agent proceeds straight (returns None so preferred action is kept)
        alt_winner = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,  # tick=0 gives agent_1 higher priority
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            num_agents=2,
        )
        self.assertIsNone(alt_winner)

        # Lower priority agent is redirected to an orthogonal sidestep
        alt_loser = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # tick=1 gives agent_1 lower priority
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            num_agents=2,
        )
        self.assertIn(alt_loser, [Action.UP, Action.DOWN])

    def test_completed_agent_yielding_still_works(self) -> None:
        """5b. Prove completed agents on goal cells yield right-of-way during sidesteps."""
        # Preferred RIGHT is contested with active agent_2
        self.coord.register_intent("agent_2", Position(2, 2))

        candidate_actions = [Action.RIGHT, Action.UP, Action.DOWN]
        dest_map = {
            Action.RIGHT: Position(2, 2),
            Action.UP: Position(1, 1),     # occupied by resting completed agent
            Action.DOWN: Position(1, 3),   # occupied by dead exhausted obstacle
        }
        current_positions = {
            "agent_1": Position(1, 2),
            "agent_completed": Position(1, 1),
            "wall_exhausted": Position(1, 3),
        }
        completed_positions = {Position(1, 1)}
        obstacle_positions = {Position(1, 3)}

        alt = self.coord.suggest_alternative_move(
            agent_id="agent_1",
            pos=Position(1, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=1,  # agent_1 has lower priority at tick=1
            preferred_action=Action.RIGHT,
            current_positions=current_positions,
            completed_positions=completed_positions,
            obstacle_positions=obstacle_positions,
            target=Position(1, 0),
            num_agents=2,
        )
        # UP is traversable because occupant is completed and yields!
        # DOWN is blocked because occupant is exhausted obstacle!
        self.assertEqual(alt, Action.UP)

    def test_recovery_action_rejects_obstacles_and_oob(self) -> None:
        """Verify get_recovery_action also respects obstacles and out-of-bounds boundaries."""
        coord = MultiAgentCoordinator(enabled=True, world_width=4, world_height=4)
        # Agent at (0, 0), blocked cell is (1, 0)
        # Orthogonal candidates for horizontal block at y=0 are DOWN (0, 1) and UP (0, -1)
        dest_map = {
            Action.DOWN: Position(0, 1),   # free
            Action.UP: Position(0, -1),    # out of bounds
        }
        act = coord.get_recovery_action(
            agent_id="agent_1",
            pos=Position(0, 0),
            legal_actions=[Action.DOWN, Action.UP],
            destination_map=dest_map,
            blocked_cell=Position(1, 0),
            partner_id="agent_2",
            tick=1,  # loser
            num_agents=2,
            obstacle_positions={Position(0, 1)},  # DOWN is obstacle!
        )
        # Both UP (OOB) and DOWN (obstacle) are invalid; must return None (fail safe)
        self.assertIsNone(act)


if __name__ == "__main__":
    unittest.main()
