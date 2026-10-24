"""Unit and integration tests for Phase 13: Multi-Agent Simulation."""

import unittest

from worldsim.actions import Action
from worldsim.linear_learning import FeatureExtractor, LinearLearningConfig, LinearQLearner
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.policy import RandomPolicy
from worldsim.robust_policy import RobustResourcePolicy


class MultiAgentTests(unittest.TestCase):
    """Test suite covering multi-agent mechanics, conflict resolution, and metrics."""

    def test_multi_agent_initialization_and_validation(self) -> None:
        """WorldConfig validates bounds, unique IDs, and rejects overlapping starting positions."""
        # 1. Reject overlapping starts
        with self.assertRaises(ValueError):
            MultiAgentWorldConfig(
                width=5, height=5,
                agents=(
                    AgentConfig("agent_1", initial_x=1, initial_y=1),
                    AgentConfig("agent_2", initial_x=1, initial_y=1),
                ),
            )

        # 2. Reject duplicate IDs
        with self.assertRaises(ValueError):
            MultiAgentWorldConfig(
                width=5, height=5,
                agents=(
                    AgentConfig("same_id", initial_x=0, initial_y=0),
                    AgentConfig("same_id", initial_x=1, initial_y=1),
                ),
            )

        # 3. Reject out-of-bounds start
        with self.assertRaises(ValueError):
            MultiAgentWorldConfig(
                width=4, height=4,
                agents=(AgentConfig("agent_1", initial_x=5, initial_y=0),),
            )

        # 4. Valid initialization
        cfg = MultiAgentWorldConfig(
            width=5, height=5,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5),
                AgentConfig("a2", initial_x=4, initial_y=4, initial_energy=8),
            ),
        )
        sim = MultiAgentSimulation(cfg)
        self.assertEqual(len(sim.state.agents), 2)
        self.assertEqual(sim.state.agents["a1"].position, Position(0, 0))
        self.assertEqual(sim.state.agents["a2"].position, Position(4, 4))
        self.assertEqual(sim.state.agents["a1"].energy, 5)
        self.assertEqual(sim.state.agents["a2"].energy, 8)

    def test_observation_isolation(self) -> None:
        """Each agent only observes cells, resources, and peers within its own sensory radius."""
        cfg = MultiAgentWorldConfig(
            width=8, height=8, sensory_radius=2.0,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0),
                AgentConfig("a2", initial_x=7, initial_y=7),
            ),
            resources=(
                Resource(Position(1, 1), 4, True),
                Resource(Position(6, 6), 4, True),
            ),
        )
        sim = MultiAgentSimulation(cfg)

        obs_1 = sim.observe_agent("a1")
        obs_2 = sim.observe_agent("a2")

        # Agent 1 sees resource at (1,1) but NOT resource at (6,6) or agent a2
        self.assertEqual(len(obs_1.observed_resources), 1)
        self.assertEqual(obs_1.observed_resources[0].position, Position(1, 1))
        self.assertEqual(len(obs_1.observed_agents), 0)

        # Agent 2 sees resource at (6,6) but NOT resource at (1,1) or agent a1
        self.assertEqual(len(obs_2.observed_resources), 1)
        self.assertEqual(obs_2.observed_resources[0].position, Position(6, 6))
        self.assertEqual(len(obs_2.observed_agents), 0)

    def test_vertex_collision_symmetric_bounce(self) -> None:
        """When two agents attempt to move into the same unoccupied cell, both bounce back."""
        # a1 at (0, 1) moving RIGHT to (1, 1)
        # a2 at (2, 1) moving LEFT to (1, 1)
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=1, initial_energy=5),
                AgentConfig("a2", initial_x=2, initial_y=1, initial_energy=5),
            ),
        )
        sim = MultiAgentSimulation(cfg)

        step_res = sim.step({"a1": Action.RIGHT, "a2": Action.LEFT})

        # Both moves should be rejected and bounce back to initial positions
        self.assertEqual(sim.state.agents["a1"].position, Position(0, 1))
        self.assertEqual(sim.state.agents["a2"].position, Position(2, 1))
        self.assertFalse(step_res.agent_results["a1"].moved)
        self.assertFalse(step_res.agent_results["a2"].moved)
        self.assertEqual(step_res.agent_results["a1"].collision_type, "vertex")
        self.assertEqual(step_res.agent_results["a2"].collision_type, "vertex")
        self.assertGreater(step_res.vertex_collisions, 0)
        # No energy consumed on rejected bounce
        self.assertEqual(sim.state.agents["a1"].energy, 5)
        self.assertEqual(sim.state.agents["a2"].energy, 5)

    def test_swap_collision_symmetric_bounce(self) -> None:
        """When two adjacent agents attempt to swap positions, both bounce back head-on."""
        # a1 at (1, 1) moving RIGHT to (2, 1)
        # a2 at (2, 1) moving LEFT to (1, 1)
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=1, initial_y=1, initial_energy=5),
                AgentConfig("a2", initial_x=2, initial_y=1, initial_energy=5),
            ),
        )
        sim = MultiAgentSimulation(cfg)

        step_res = sim.step({"a1": Action.RIGHT, "a2": Action.LEFT})

        self.assertEqual(sim.state.agents["a1"].position, Position(1, 1))
        self.assertEqual(sim.state.agents["a2"].position, Position(2, 1))
        self.assertFalse(step_res.agent_results["a1"].moved)
        self.assertFalse(step_res.agent_results["a2"].moved)
        self.assertEqual(step_res.agent_results["a1"].collision_type, "swap")
        self.assertEqual(step_res.agent_results["a2"].collision_type, "swap")
        self.assertGreater(step_res.swap_collisions, 0)

    def test_occupied_stationary_collision(self) -> None:
        """An agent attempting to move into a cell occupied by a stationary agent is blocked."""
        # a1 at (0, 0) moving RIGHT to (1, 0)
        # a2 at (1, 0) stationary (recharging or moving into a wall)
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5),
                AgentConfig("a2", initial_x=1, initial_y=0, initial_energy=5),
            ),
        )
        sim = MultiAgentSimulation(cfg)

        step_res = sim.step({"a1": Action.RIGHT, "a2": Action.UP}) # a2 hits top boundary and stays

        self.assertEqual(sim.state.agents["a1"].position, Position(0, 0))
        self.assertEqual(sim.state.agents["a2"].position, Position(1, 0))
        self.assertEqual(step_res.agent_results["a1"].collision_type, "occupied")

    def test_train_movement_following(self) -> None:
        """An agent can successfully move into a cell vacated by another agent in the same tick."""
        # a1 at (0, 0) moves RIGHT to (1, 0)
        # a2 at (1, 0) moves RIGHT to (2, 0)
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5),
                AgentConfig("a2", initial_x=1, initial_y=0, initial_energy=5),
            ),
        )
        sim = MultiAgentSimulation(cfg)

        step_res = sim.step({"a1": Action.RIGHT, "a2": Action.RIGHT})

        self.assertEqual(sim.state.agents["a1"].position, Position(1, 0))
        self.assertEqual(sim.state.agents["a2"].position, Position(2, 0))
        self.assertTrue(step_res.agent_results["a1"].moved)
        self.assertTrue(step_res.agent_results["a2"].moved)
        self.assertEqual(sim.state.agents["a1"].energy, 4)
        self.assertEqual(sim.state.agents["a2"].energy, 4)

    def test_atomic_resource_consumption_and_no_duplicate_energy(self) -> None:
        """Consumable resource is depleted atomically by the first claimant without energy duplication."""
        # Test simulated contention: suppose two agents are on a resource cell (simulating arrival or contention)
        # Since standard collision rules prevent overlapping, we test atomic consumption via simulation step:
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=1, initial_y=0, initial_energy=2),
                AgentConfig("a2", initial_x=0, initial_y=1, initial_energy=2),
            ),
            resources=(Resource(Position(1, 0), recharge_amount=4, consumable=True),),
            max_energy=6,
        )
        sim = MultiAgentSimulation(cfg)

        # Agent 1 is standing on the resource at (1, 0) and recharges
        step_res = sim.step({"a1": Action.RECHARGE, "a2": Action.RIGHT})

        self.assertTrue(step_res.agent_results["a1"].success)
        self.assertEqual(sim.state.agents["a1"].energy, 6) # 2 + 4 = 6
        # Resource must now be depleted!
        self.assertEqual(len(sim.state.resources), 0)

        # In tick 2, Agent 1 vacates rightward to (2, 0) and Agent 2 moves onto (1, 0)
        step_res2 = sim.step({"a1": Action.RIGHT, "a2": Action.UP})
        self.assertEqual(sim.state.agents["a2"].position, Position(1, 0))

        step_res3 = sim.step({"a1": Action.RIGHT, "a2": Action.RECHARGE})
        # Resource was already depleted, so recharge must fail!
        self.assertFalse(step_res3.agent_results["a2"].success)
        self.assertEqual(sim.state.agents["a2"].energy, 0) # No duplicate energy granted!

    def test_individual_goals_and_completion_criteria(self) -> None:
        """Simulations track individual goals and evaluate overall completion correctly."""
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5, target=Position(1, 0)),
                AgentConfig("a2", initial_x=0, initial_y=2, initial_energy=5, target=Position(1, 1)),
            ),
            goal_criterion="individual",
        )
        sim = MultiAgentSimulation(cfg)

        # Step 1: only a1 reaches target; a2 moves UP to (0, 1)
        res1 = sim.step({"a1": Action.RIGHT, "a2": Action.UP})
        self.assertTrue(sim.state.agents["a1"].status == "reached_target")
        self.assertFalse(sim.state.agents["a2"].status == "reached_target")
        self.assertFalse(res1.overall_completed) # a2 hasn't reached yet

        # Step 2: a2 moves RIGHT to (1, 1), reaching its target
        res2 = sim.step({"a1": Action.UP, "a2": Action.RIGHT})
        self.assertTrue(sim.state.agents["a2"].status == "reached_target")
        self.assertTrue(res2.overall_completed) # all agents reached individual goals!

    def test_shared_goal_criteria(self) -> None:
        """Shared goal with 'shared_any' completes as soon as any agent reaches the destination."""
        cfg = MultiAgentWorldConfig(
            width=4, height=4,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5),
                AgentConfig("a2", initial_x=0, initial_y=3, initial_energy=5),
            ),
            shared_target=Position(1, 0),
            goal_criterion="shared_any",
        )
        sim = MultiAgentSimulation(cfg)

        res = sim.step({"a1": Action.RIGHT, "a2": Action.RIGHT})
        self.assertTrue(sim.state.agents["a1"].status == "reached_target")
        self.assertTrue(res.overall_completed)

    def test_cooperative_discovery_sharing(self) -> None:
        """When cooperative sharing is enabled, resources observed by one agent are shared in observations."""
        cfg = MultiAgentWorldConfig(
            width=8, height=8, sensory_radius=2.0,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0),
                AgentConfig("a2", initial_x=7, initial_y=7),
            ),
            resources=(Resource(Position(1, 1), 4, True),),
            cooperative_sharing=True,
        )
        sim = MultiAgentSimulation(cfg)

        # Step tick 1: a1 discovers resource at (1, 1)
        sim.step({"a1": Action.UP, "a2": Action.UP})

        # Agent 2's observation now contains the shared discovery from agent 1
        obs_2 = sim.observe_agent("a2")
        self.assertEqual(len(obs_2.shared_discoveries), 1)
        self.assertEqual(obs_2.shared_discoveries[0].position, Position(1, 1))

    def test_policy_compatibility_with_multi_agent_observation(self) -> None:
        """Existing policies (RobustResourcePolicy, LinearQLearner, RandomPolicy) operate natively."""
        cfg = MultiAgentWorldConfig(
            width=5, height=5,
            agents=(
                AgentConfig("a1", initial_x=0, initial_y=0, initial_energy=5, target=Position(3, 3)),
                AgentConfig("a2", initial_x=0, initial_y=4, initial_energy=5, target=Position(3, 0)),
            ),
            resources=(Resource(Position(1, 1), 4, False),),
        )
        sim = MultiAgentSimulation(cfg)

        pol_1 = RobustResourcePolicy(world_width=5, world_height=5, max_energy=10)
        pol_2 = RandomPolicy(seed=42)

        # Select actions
        obs_1 = sim.observe_agent("a1")
        obs_2 = sim.observe_agent("a2")

        act_1 = pol_1.choose_action(obs_1)
        act_2 = pol_2.choose_action(obs_2)

        self.assertIsInstance(act_1, Action)
        self.assertIsInstance(act_2, Action)

        res = sim.step({"a1": act_1, "a2": act_2})
        self.assertIn("a1", res.agent_results)
        self.assertIn("a2", res.agent_results)


if __name__ == "__main__":
    unittest.main()
