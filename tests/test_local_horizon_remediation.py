"""Tests for WorldSim Phase 14.2 Remediation — U06 and U14.

Validates:
1. U06: Unknown queue load representation as None outside communication horizon (R=2.0).
2. U06: Conservative unknown load policy with travel energy margin protection.
3. U06: Preference for reachable local resource over suicidal distant travel.
4. U14: Crossroad junction recognition from doorway under partial obstacle visibility (2 pillars known).
5. U14: Non-crossroad cells (e.g. boundary or room cells) not falsely classified as crossroads.
6. U14: Lateral detour and yielding behavior in 4-way conflict under R=2.0 local horizon.
7. U14: Full end-to-end 100% completion on U14 scenario in decentralized mode.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.coordination import MultiAgentCoordinator
from worldsim.models import Observation, Position, Resource
from worldsim.multi_agent import AgentConfig, MultiAgentSimulation, MultiAgentWorldConfig
from worldsim.robust_policy import AgentMemory, ResourceUtilityEvaluator, RobustResourcePolicy
from phase13_5_unseen_audit import build_all_unseen_scenarios, make_robust_factory
from phase14_2_local_horizon_eval import run_unseen_scenario_eval


class TestLocalHorizonRemediation(unittest.TestCase):
    """Unit and integration tests for Phase 14.2 remediation of U06 and U14."""

    def test_u06_unknown_resource_load_beyond_horizon(self) -> None:
        """Verify that get_resource_load returns None when resource is beyond R=2.0."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
        )
        r_distant = Position(6, 6)
        r_local = Position(2, 2)
        coord.register_intent("agent_peer", Position(6, 6), targeted_resource=r_distant, current_position=Position(6, 5))
        coord.register_intent("agent_local", Position(2, 2), targeted_resource=r_local, current_position=Position(2, 1))

        # Query from (1, 2): distance to (2, 2) is 1.0 <= 2.0; distance to (6, 6) is sqrt(25+16) > 2.0
        pos_agent = Position(1, 2)
        load_local = coord.get_resource_load(r_local, querying_agent_pos=pos_agent)
        load_distant = coord.get_resource_load(r_distant, querying_agent_pos=pos_agent)

        self.assertEqual(load_local, 1)
        self.assertIsNone(load_distant)

    def test_u06_conservative_utility_penalizes_distant_resource_when_energy_tight(self) -> None:
        """Verify ResourceUtilityEvaluator penalizes unknown distant load when travel margin is tight."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            world_width=8,
            world_height=8,
        )
        r_local = Position(2, 2)
        r_distant = Position(6, 6)
        coord.register_intent("agent_occupant", Position(2, 2), targeted_resource=r_local, current_position=Position(2, 2))

        evaluator = ResourceUtilityEvaluator(
            max_energy=10,
            coordinator=coord,
        )
        mem = AgentMemory(world_width=8, world_height=8)
        mem.known_resources[r_local] = Resource(r_local, 10, consumable=False)
        mem.known_resources[r_distant] = Resource(r_distant, 10, consumable=False)

        # Agent at (1, 2) has energy=7. Target is (7, 7).
        # Distance to r_local is 1. Distance to r_distant is 5+4 = 9 (cannot reach!)
        # Distance to (5, 5) is 4+3 = 7 steps. With E=7, travel_margin = 7 - 7 = 0 < 2!
        obs = Observation(
            entity_position=Position(1, 2),
            cells=(Position(1, 2), Position(2, 2)),
            energy=7,
            target=Position(7, 7),
            observed_resources=(Resource(r_local, 10, consumable=False),),
        )
        decision = evaluator.compare_routes(obs, mem)
        # Conservative policy refuses distant resource when travel margin is insufficient
        self.assertNotEqual(decision.recommended_target, r_distant)

    def test_u14_junction_recognition_under_partial_visibility(self) -> None:
        """Verify crossroad recognition with 2 known entrance pillars and all cardinals open."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            world_width=7,
            world_height=7,
        )
        # Agent at East doorway (4, 3) looking West into (3, 3)
        # Pillars visible to agent: (4, 2) and (4, 4)
        pos = Position(4, 3)
        pref_dest = Position(3, 3)
        known_obs = [Position(4, 2), Position(4, 4)]
        dest_map = {
            Action.LEFT: pref_dest,
            Action.RIGHT: Position(5, 3),
            Action.UP: Position(4, 2),
            Action.DOWN: Position(4, 4),
            Action.RECHARGE: pos,
        }
        # Swap partner agent_w is inside (3, 3) wanting to exit to (4, 3)
        coord.register_intent("agent_w", pos, Position(6, 3), current_position=pref_dest)

        alt = coord.suggest_alternative_move(
            agent_id="agent_e",
            pos=pos,
            candidate_actions=[Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.RECHARGE],
            destination_map=dest_map,
            tick=3,
            preferred_action=Action.LEFT,
            current_positions={"agent_w": pref_dest},
            target=Position(0, 3),
            energy=8,
            obstacle_positions=known_obs,
        )
        # Agent at entrance must yield to vehicle clearing the intersection by backing up
        self.assertEqual(alt, Action.RIGHT)

    def test_u14_non_crossroad_boundary_cell_not_falsely_classified(self) -> None:
        """Verify boundary cells or doorway exits like (3, 1) and (5, 3) are not falsely recognized as crossroads."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            world_width=7,
            world_height=7,
        )
        # Agent at (3, 2) with partner at (3, 1).
        # (3, 1) is adjacent to boundary y=0 and must NOT be classified as an internal 4-way crossroad.
        pos = Position(3, 2)
        pref_dest = Position(3, 1)
        coord.register_intent("agent_n", pos, Position(3, 6), current_position=pref_dest)
        dest_map = {
            Action.UP: pref_dest,
            Action.DOWN: Position(3, 3),
            Action.LEFT: Position(2, 2),
            Action.RIGHT: Position(4, 2),
            Action.RECHARGE: pos,
        }
        alt = coord.suggest_alternative_move(
            agent_id="agent_s",
            pos=pos,
            candidate_actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.RECHARGE],
            destination_map=dest_map,
            tick=6,
            preferred_action=Action.UP,
            current_positions={"agent_n": pref_dest},
            target=Position(3, 0),
            energy=6,
            obstacle_positions=[Position(2, 2), Position(4, 2)],
        )
        # Partner has lateral moves at (3, 1), so agent_s inside the narrow cell does NOT yield
        self.assertIsNone(alt)

    def test_u06_end_to_end_100_percent_completion(self) -> None:
        """Verify U06 achieves 100.0% completion across seeds 500-519 in decentralized mode."""
        specs = {spec.scenario_id: (spec, prep) for spec, prep in build_all_unseen_scenarios()}
        spec, prep = specs["U06"]
        seeds = list(range(500, 520))
        res = run_unseen_scenario_eval(spec, prep, None, "decentralized", seeds, max_steps=50)
        self.assertEqual(res.overall_completion_rate, 100.0)
        self.assertEqual(res.total_exhaustions, 0)

    def test_u14_end_to_end_100_percent_completion(self) -> None:
        """Verify U14 achieves 100.0% completion across seeds 500-519 in decentralized mode."""
        specs = {spec.scenario_id: (spec, prep) for spec, prep in build_all_unseen_scenarios()}
        spec, prep = specs["U14"]
        seeds = list(range(500, 520))
        res = run_unseen_scenario_eval(spec, prep, None, "decentralized", seeds, max_steps=50)
        self.assertEqual(res.overall_completion_rate, 100.0)
        self.assertEqual(res.total_exhaustions, 0)


if __name__ == "__main__":
    unittest.main()
