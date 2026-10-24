"""Tests for WorldSim Phase 14.2 Local-Horizon Coordination.

Validates:
1. Default coordination mode remains centralized for backward compatibility.
2. Decentralized coordination mode restricts intention and reservation exchange to Euclidean distance <= 2.0.
3. Communication boundary cases at exactly distance <= 2.0 vs > 2.0.
4. Distant intentions and hidden obstacle knowledge are strictly NOT leaked to local decisions.
5. Local obstacle memory maintains visual revalidation and TTL expiration.
6. Reciprocal swap conflicts and doorway yielding resolve safely within the local horizon.
7. Resource load queries in decentralized mode do not leak distant commitments.
8. Authoritative environment physics prevents illegal collisions even with partial coordination.
9. Determinism across identical seeds in decentralized mode.
"""

from __future__ import annotations

import math
import unittest

from worldsim.actions import Action
from worldsim.coordination import (
    DeadlockDetector,
    MultiAgentCoordinator,
    ResourceArbitrator,
)
from worldsim.linear_learning import FeatureExtractor, LinearLearningConfig, LinearLearningRunner
from worldsim.models import Observation, Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import AgentMemory, RobustResourcePolicy


class TestLocalHorizonCoordination(unittest.TestCase):
    """Unit tests for Phase 14.2 decentralized local-horizon coordination."""

    def test_default_coordination_mode_is_centralized(self) -> None:
        """Verify default configuration and coordinator remain centralized."""
        coord = MultiAgentCoordinator(enabled=True)
        self.assertEqual(coord.coordination_mode, "centralized")
        self.assertEqual(coord.communication_radius, 2.0)

        cfg = MultiAgentWorldConfig(
            width=5,
            height=5,
            agents=(
                AgentConfig(id="agent_1", initial_x=0, initial_y=0, target=Position(4, 4)),
            ),
        )
        self.assertEqual(cfg.coordination_mode, "centralized")
        self.assertEqual(cfg.communication_radius, 2.0)

        sim = MultiAgentSimulation(cfg, seed=0)
        self.assertEqual(sim.coordinator.coordination_mode, "centralized")

    def test_invalid_coordination_mode_raises_error(self) -> None:
        """Verify invalid coordination mode or radius raises ValueError."""
        with self.assertRaises(ValueError):
            MultiAgentWorldConfig(
                width=5,
                height=5,
                agents=(AgentConfig(id="a1", initial_x=0, initial_y=0),),
                coordination_mode="invalid_mode",
            )

        with self.assertRaises(ValueError):
            MultiAgentWorldConfig(
                width=5,
                height=5,
                agents=(AgentConfig(id="a1", initial_x=0, initial_y=0),),
                communication_radius=-1.0,
            )

    def test_distant_intentions_not_leaked_in_decentralized_mode(self) -> None:
        """Verify intentions of agents farther than R=2.0 are invisible in decentralized mode."""
        coord_decent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            world_width=10,
            world_height=10,
        )
        coord_cent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="centralized",
            world_width=10,
            world_height=10,
        )

        pos_a = Position(1, 1)
        pos_b = Position(8, 8)  # Distance = sqrt(49+49) approx 9.9 > 2.0
        dest_b = Position(8, 9)

        # Register Agent 1's intention (Agent 1 has higher priority than Agent 2)
        coord_decent.register_intent("agent_1", dest_b, current_position=pos_b)
        coord_cent.register_intent("agent_1", dest_b, current_position=pos_b)

        # Query visible peers for Agent 2 from (1, 1)
        peers_decent = coord_decent.get_visible_peers("agent_2", pos_a)
        peers_cent = coord_cent.get_visible_peers("agent_2", pos_a)

        self.assertNotIn("agent_1", peers_decent)
        self.assertIn("agent_1", peers_cent)

        # In decentralized mode, Agent 2 can propose moving to (8, 9) without conflict check picking up Agent 1
        candidate_actions = [Action.DOWN, Action.RIGHT]
        dest_map = {Action.DOWN: Position(1, 2), Action.RIGHT: dest_b}
        alt_decent = coord_decent.suggest_alternative_move(
            "agent_2",
            pos_a,
            candidate_actions,
            dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
        )
        # Because Agent 1 is distant and invisible, Agent 2 is not redirected away from dest_b by coordinator
        self.assertIsNone(alt_decent)

        # In centralized mode, Agent 2 immediately detects the vertex conflict with higher-priority Agent 1 and is redirected
        alt_cent = coord_cent.suggest_alternative_move(
            "agent_2",
            pos_a,
            candidate_actions,
            dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
        )
        self.assertEqual(alt_cent, Action.DOWN)

    def test_communication_radius_boundary_cases(self) -> None:
        """Verify strict Euclidean distance boundary: <= 2.0 is visible, > 2.0 is invisible."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
        )
        origin = Position(5, 5)

        # Table of test points and expected visibility
        # (dx, dy, distance, expected_visible)
        test_cases = [
            (0, 1, 1.0, True),                # Cardinal 1 step
            (0, 2, 2.0, True),                # Cardinal 2 steps (exact boundary)
            (1, 1, math.sqrt(2), True),        # Diagonal 1 step (~1.414 <= 2.0)
            (-2, 0, 2.0, True),               # Cardinal 2 steps left (exact boundary)
            (0, 3, 3.0, False),               # Cardinal 3 steps (outside)
            (2, 1, math.sqrt(5), False),       # Knight move (~2.236 > 2.0)
            (1, 2, math.sqrt(5), False),       # Knight move (~2.236 > 2.0)
            (2, 2, math.sqrt(8), False),       # Diagonal 2 steps (~2.828 > 2.0)
        ]

        for dx, dy, dist, expected_visible in test_cases:
            peer_pos = Position(origin.x + dx, origin.y + dy)
            peer_id = f"peer_{dx}_{dy}"
            coord.reset()
            coord.register_intent(peer_id, Position(peer_pos.x, peer_pos.y + 1), current_position=peer_pos)
            visible = coord.get_visible_peers("observer", origin)

            if expected_visible:
                self.assertIn(peer_id, visible, f"Failed: {peer_pos} at dist {dist:.3f} should be visible")
            else:
                self.assertNotIn(peer_id, visible, f"Failed: {peer_pos} at dist {dist:.3f} should NOT be visible")

    def test_distant_obstacle_knowledge_not_leaked_in_decentralized_mode(self) -> None:
        """Verify coordinator does not pool distant obstacle knowledge in decentralized mode."""
        coord_decent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            world_width=10,
            world_height=10,
        )
        coord_cent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="centralized",
            world_width=10,
            world_height=10,
        )

        distant_obstacle = Position(8, 8)
        # Agent 1 at (8, 7) encounters and reports distant obstacle
        coord_cent.suggest_alternative_move(
            "agent_1",
            Position(8, 7),
            [Action.DOWN, Action.LEFT],
            {Action.DOWN: distant_obstacle, Action.LEFT: Position(7, 7)},
            tick=0,
            obstacle_positions=[distant_obstacle],
        )
        coord_decent.suggest_alternative_move(
            "agent_1",
            Position(8, 7),
            [Action.DOWN, Action.LEFT],
            {Action.DOWN: distant_obstacle, Action.LEFT: Position(7, 7)},
            tick=0,
            obstacle_positions=[distant_obstacle],
        )

        # Now Agent 2 at (1, 1) queries an alternative move near (8, 8) with no local obstacle memory
        cand_actions = [Action.RIGHT, Action.DOWN]
        dest_map = {Action.RIGHT: distant_obstacle, Action.DOWN: Position(1, 2)}

        # In centralized mode, Agent 2 has the distant obstacle leaked via known_obstacles
        self.assertIn(distant_obstacle, coord_cent.known_obstacles)

        # In decentralized mode, known_obstacles is not shared into Agent 2's decision
        # Agent 2's local obstacle memory is empty
        alt_decent = coord_decent.suggest_alternative_move(
            "agent_2",
            Position(1, 1),
            cand_actions,
            dest_map,
            tick=0,
            preferred_action=Action.RIGHT,
            obstacle_positions=[],
        )
        # In decentralized mode, Agent 2 does NOT know about (8, 8) from Agent 1
        # Therefore, coordinator does NOT redirect Agent 2 away based on Agent 1's secret obstacle knowledge
        self.assertIsNone(alt_decent)

    def test_local_obstacle_memory_ttl_and_visual_revalidation(self) -> None:
        """Verify AgentMemory maintains local TTL expiration and visual revalidation."""
        mem = AgentMemory(world_width=10, world_height=10, obstacle_memory_ttl=5)

        # Step 0: Observe an exhausted agent at (2, 2) from (1, 2) (distance = 1.0 <= 2.0)
        obs_step0 = MultiAgentObservation(
            agent_id="agent_1",
            entity_position=Position(1, 2),
            energy=5,
            cells=tuple(Position(x, y) for x in range(3) for y in range(4)),
            observed_resources=(),
            shared_discoveries=(),
            observed_agents=(
                AgentState(id="agent_2", position=Position(2, 2), energy=0, active=False, status="exhausted"),
            ),
        )
        mem.update(obs_step0)
        self.assertIn(Position(2, 2), mem.remembered_obstacles)

        # Steps 1 to 5: Move far away to (8, 8) where (2, 2) is outside sensory range
        for _ in range(5):
            obs_step = MultiAgentObservation(
                agent_id="agent_1",
                entity_position=Position(8, 8),
                energy=5,
                cells=tuple(Position(x, y) for x in range(7, 10) for y in range(7, 10)),
                observed_resources=(),
                shared_discoveries=(),
                observed_agents=(),
            )
            mem.update(obs_step)
            # Memory should still be remembered within TTL (TTL = 5)
            self.assertIn(Position(2, 2), mem.remembered_obstacles)

        # Step 6: Step count is now 6 (elapsed 6 > 5) -> TTL expires
        obs_step6 = MultiAgentObservation(
            agent_id="agent_1",
            entity_position=Position(8, 8),
            energy=5,
            cells=tuple(Position(x, y) for x in range(7, 10) for y in range(7, 10)),
            observed_resources=(),
            shared_discoveries=(),
            observed_agents=(),
        )
        mem.update(obs_step6)
        self.assertNotIn(Position(2, 2), mem.remembered_obstacles)

        # Also verify visual revalidation:
        # Re-add an obstacle at (2, 2)
        mem.update(obs_step0)
        self.assertIn(Position(2, 2), mem.remembered_obstacles)
        # Agent observes cell (2, 2) again, but with NO obstacle
        obs_empty = MultiAgentObservation(
            agent_id="agent_1",
            entity_position=Position(1, 2),
            energy=5,
            cells=tuple(Position(x, y) for x in range(3) for y in range(4)),
            observed_resources=(),
            shared_discoveries=(),
            observed_agents=(),  # no agents present now
        )
        mem.update(obs_empty)
        self.assertNotIn(Position(2, 2), mem.remembered_obstacles)

    def test_reciprocal_swap_conflict_resolved_within_local_horizon(self) -> None:
        """Verify agents within R=2.0 resolve reciprocal head-on swap in decentralized mode."""
        cfg = MultiAgentWorldConfig(
            width=5,
            height=3,
            agents=(
                AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=5, target=Position(2, 1)),
                AgentConfig(id="agent_2", initial_x=2, initial_y=1, initial_energy=5, target=Position(1, 1)),
            ),
            enable_coordination=True,
            coordination_mode="decentralized",
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(cfg, seed=0)
        self.assertEqual(sim.coordinator.coordination_mode, "decentralized")

        coord = sim.coordinator
        # Distance between (1, 1) and (2, 1) is 1.0 <= 2.0 (well within local horizon)
        coord.register_intent("agent_1", Position(2, 1), current_position=Position(1, 1))

        # Agent 2 checks alternative move facing Agent 1
        dest_map = {
            Action.LEFT: Position(1, 1),   # swap target
            Action.UP: Position(2, 0),     # safe lateral sidestep
            Action.DOWN: Position(2, 2),   # safe lateral sidestep
        }
        alt = coord.suggest_alternative_move(
            "agent_2",
            Position(2, 1),
            [Action.LEFT, Action.UP, Action.DOWN],
            dest_map,
            tick=0,
            preferred_action=Action.LEFT,
            current_positions={"agent_1": Position(1, 1)},
            target=Position(1, 1),
        )
        # Agent 2 successfully receives a lateral sidestep suggestion (UP or DOWN)
        self.assertIn(alt, (Action.UP, Action.DOWN))

    def test_doorway_bottleneck_yielding_within_local_horizon(self) -> None:
        """Verify agents approaching a shared bottleneck doorway yield safely within R=2.0."""
        coord = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            world_width=5,
            world_height=3,
        )
        # Doorway is at (2, 1). Agent 1 at (1, 1) steps RIGHT. Agent 2 at (3, 1) steps LEFT.
        # Distance between agents is 2.0 <= 2.0 (on the boundary of the local horizon).
        coord.register_intent("agent_1", Position(2, 1), current_position=Position(1, 1))

        dest_map_2 = {
            Action.LEFT: Position(2, 1),   # contested doorway
            Action.UP: Position(3, 0),     # safe lateral waiting cell
        }
        alt_2 = coord.suggest_alternative_move(
            "agent_2",
            Position(3, 1),
            [Action.LEFT, Action.UP],
            dest_map_2,
            tick=0,
            preferred_action=Action.LEFT,
            current_positions={"agent_1": Position(1, 1)},
        )
        # Agent 2 detects vertex contention and yields safely to Action.UP
        self.assertEqual(alt_2, Action.UP)

    def test_distant_resource_commitments_not_leaked_in_decentralized_load(self) -> None:
        """Verify get_resource_load does not leak distant commitments in decentralized mode."""
        coord_decent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
        )
        coord_cent = MultiAgentCoordinator(
            enabled=True,
            coordination_mode="centralized",
        )

        r_far = Position(9, 9)
        # Agent at (9, 8) targets resource at (9, 9)
        coord_decent.register_intent("agent_distant", Position(9, 9), targeted_resource=r_far, current_position=Position(9, 8))
        coord_cent.register_intent("agent_distant", Position(9, 9), targeted_resource=r_far, current_position=Position(9, 8))

        # Observer at (0, 0) queries resource load of r_far
        pos_obs = Position(0, 0)
        load_decent = coord_decent.get_resource_load(r_far, querying_agent_pos=pos_obs)
        load_cent = coord_cent.get_resource_load(r_far, querying_agent_pos=pos_obs)

        # Decentralized mode cannot see the distant commitment
        self.assertEqual(load_decent, 0)
        # Centralized mode sees the distant commitment
        self.assertEqual(load_cent, 1)

    def test_authoritative_simulation_physics_prevents_illegal_collision(self) -> None:
        """Verify simulation physics authoritatively prevents illegal double occupancy."""
        cfg = MultiAgentWorldConfig(
            width=5,
            height=5,
            agents=(
                AgentConfig(id="agent_1", initial_x=1, initial_y=2, initial_energy=5, target=Position(2, 2)),
                AgentConfig(id="agent_2", initial_x=3, initial_y=2, initial_energy=5, target=Position(2, 2)),
            ),
            enable_coordination=True,
            coordination_mode="decentralized",
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(cfg, seed=0)

        # Both step into (2, 2)
        res = sim.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})

        # Physical conflict resolution authoritatively bounces one agent
        moved = [aid for aid, ar in res.agent_results.items() if ar.moved]
        self.assertEqual(len(moved), 1)
        self.assertEqual(res.vertex_collisions, 1)
        # End positions are strictly distinct
        pos1 = sim.state.agents["agent_1"].position
        pos2 = sim.state.agents["agent_2"].position
        self.assertNotEqual(pos1, pos2)

    def test_decentralized_simulation_determinism(self) -> None:
        """Verify two simulation runs with identical seed produce bit-for-bit identical results."""
        def run_sim(seed: int) -> list[tuple[Position, Position]]:
            cfg = MultiAgentWorldConfig(
                width=6,
                height=4,
                agents=(
                    AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=10, target=Position(4, 1)),
                    AgentConfig(id="agent_2", initial_x=4, initial_y=1, initial_energy=10, target=Position(1, 1)),
                ),
                enable_coordination=True,
                coordination_mode="decentralized",
                collision_rule="priority_rotation",
            )
            sim = MultiAgentSimulation(cfg, seed=seed)
            positions = []
            for _ in range(10):
                sim.coordinator.reset_tick()
                # Dummy alternating step
                sim.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})
                positions.append((sim.state.agents["agent_1"].position, sim.state.agents["agent_2"].position))
            return positions

        trace1 = run_sim(seed=42)
        trace2 = run_sim(seed=42)
        self.assertEqual(trace1, trace2)


if __name__ == "__main__":
    unittest.main()
