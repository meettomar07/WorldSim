"""Unit tests for Phase 13.2: Resolving the Eight-Agent Bottleneck.

Covers:
1. Reusable resource multi-recharge capacity evaluation in ResourceUtilityEvaluator.
2. MultiAgentCoordinator doorstep queuing: preventing fatal redirects away from resources/targets.
3. Coordinator energy-aware filtering: preventing alternative moves that cause energy exhaustion.
4. Persistent resource commitments across ticks and resource load balancing.
5. Inactive agent obstacle detection: distinguishing stationary inactive agents from active traffic.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import MultiAgentCoordinator
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import (
    AgentMemory,
    RechargeEvaluator,
    ResourceUtilityEvaluator,
    RobustResourcePolicy,
)


class TestPhase13_2CoordinationMechanisms(unittest.TestCase):
    """Unit tests for Phase 13.2 multi-agent coordination and robust policy improvements."""

    def setUp(self) -> None:
        self.coordinator = MultiAgentCoordinator(enabled=True)

    def test_reusable_resource_multi_recharge_utility(self) -> None:
        """Verify reusable resources account for multiple recharges up to max_energy."""
        evaluator = ResourceUtilityEvaluator(
            max_energy=10,
            movement_energy_cost=1,
            coordinator=self.coordinator,
        )
        reusable_res = Resource(position=Position(3, 3), recharge_amount=6, consumable=False)
        pos = Position(0, 0)
        energy = 6
        target = Position(7, 7)

        # Dist to (3, 3) is 6, cost is 6. Energy arrives = 0.
        # Target dist is 8. Target cost is 8.
        # Single recharge gives 6 energy (insufficient for 8 steps).
        # Multi-recharge gives 10 energy (sufficient for 8 steps!).
        util = evaluator.evaluate_resource(reusable_res, pos, energy, target)
        self.assertTrue(util.reachable_now)
        self.assertTrue(util.is_feasible_route)
        self.assertEqual(util.status, "sufficient")
        self.assertEqual(util.energy_after_recharge, 10)

    def test_doorstep_queuing_does_not_redirect_from_resource(self) -> None:
        """Verify agents targeting a resource are never redirected away from the resource pad."""
        coord = MultiAgentCoordinator(enabled=True)
        r_pos = Position(3, 3)
        coord.register_intent("agent_1", destination=r_pos, targeted_resource=r_pos)

        # agent_2 also wants to step into (3, 3)
        candidate_actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        dest_map = {
            Action.RIGHT: Position(3, 3),  # resource pad
            Action.DOWN: Position(2, 4),
            Action.LEFT: Position(1, 3),
            Action.UP: Position(2, 2),
        }
        alt = coord.suggest_alternative_move(
            agent_id="agent_2",
            pos=Position(2, 3),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions={"agent_1": Position(3, 3)},
            target=r_pos,
        )
        # Should return None (allowing agent_2 to attempt entering or bounce cost-free at the doorstep)
        self.assertIsNone(alt)

    def test_energy_aware_alternative_move_filters_fatal_moves(self) -> None:
        """Verify coordinator never suggests an alternative move that makes the target unreachable."""
        coord = MultiAgentCoordinator(enabled=True)
        coord.register_intent("agent_1", destination=Position(3, 2))

        # agent_2 wants (3, 2), has energy=2, target=(3, 3)
        candidate_actions = [Action.RIGHT, Action.UP, Action.LEFT]
        dest_map = {
            Action.RIGHT: Position(3, 2),  # conflicted with agent_1
            Action.UP: Position(2, 1),     # distance to (3, 3) is 3 -> fatal with energy 2!
            Action.LEFT: Position(1, 2),   # distance to (3, 3) is 3 -> fatal with energy 2!
        }
        alt = coord.suggest_alternative_move(
            agent_id="agent_2",
            pos=Position(2, 2),
            candidate_actions=candidate_actions,
            destination_map=dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            current_positions={"agent_1": Position(3, 2)},
            target=Position(3, 3),
            energy=2,
        )
        # Both UP and LEFT are fatal (dist 3 > energy - 1 = 1); coordinator must NOT suggest them!
        self.assertIsNone(alt)

    def test_persistent_commitments_survive_reset_tick(self) -> None:
        """Verify reset_tick clears per-tick intentions but preserves resource commitments."""
        coord = MultiAgentCoordinator(enabled=True)
        r_pos = Position(4, 4)
        coord.register_intent("agent_1", destination=Position(3, 4), targeted_resource=r_pos)

        self.assertEqual(coord.get_resource_load(r_pos), 1)
        coord.reset_tick()

        # Intentions cleared for next tick
        self.assertEqual(len(coord.agent_intentions), 0)
        # Resource commitment preserved!
        self.assertEqual(coord.get_resource_load(r_pos), 1)

        # Explicit reset clears everything
        coord.reset()
        self.assertEqual(coord.get_resource_load(r_pos), 0)

    def test_stationary_inactive_obstacle_avoidance(self) -> None:
        """Verify RobustResourcePolicy detects and routes around stationary inactive agents."""
        policy = RobustResourcePolicy(world_width=5, world_height=5, max_energy=10)
        # Agent at (1, 1), target at (3, 1). Inactive obstacle at (2, 1).
        obs = MultiAgentObservation(
            agent_id="agent_1",
            entity_position=Position(1, 1),
            cells=(Position(1, 1), Position(2, 1), Position(1, 2)),
            energy=8,
            target=Position(3, 1),
            observed_agents=(
                AgentState(
                    id="agent_2",
                    position=Position(2, 1),
                    energy=0,
                    target=Position(2, 1),
                    active=False,
                    status="reached_target",
                ),
            ),
        )
        action = policy.choose_action(obs)
        # Moving RIGHT goes directly into the inactive obstacle at (2, 1); policy should avoid it
        self.assertNotEqual(action, Action.RIGHT)


if __name__ == "__main__":
    unittest.main()
