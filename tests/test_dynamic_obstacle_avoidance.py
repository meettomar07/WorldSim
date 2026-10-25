"""Tests for Phase 14.3: Dynamic Obstacle Avoidance.

Verifies:
1. Deterministic obstacle trajectories across repeated runs and seeds.
2. An obstacle approaching an agent's planned destination (local conflict prediction & evasion).
3. Two moving obstacles crossing paths (authoritative physical resolution, zero double occupancy).
4. Stale observations and obstacles outside communication range (R=2.0 boundary integrity).
5. An obstacle blocked by a wall, agent, or resource (authoritative bounce and halt).
6. Safe waiting and eventual progress when a route becomes available.
7. Regression coverage for U06 and U14.
"""

from __future__ import annotations

import unittest

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import MultiAgentCoordinator
from worldsim.dynamic_obstacles import (
    DynamicObstacleConfig,
    DynamicObstacleState,
    predict_next_obstacle_state,
    predict_obstacle_trajectory,
)
from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import AgentMemory, RobustResourcePolicy
from experiments.phase13_5_unseen_audit import build_all_unseen_scenarios
from experiments.phase14_2_local_horizon_eval import run_unseen_scenario_eval


class TestDynamicObstacleAvoidance(unittest.TestCase):
    """Test suite verifying dynamic obstacle simulation and local avoidance."""

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

    def test_deterministic_obstacle_trajectories(self) -> None:
        """1. Prove obstacle trajectories are bit-for-bit deterministic across repeated runs."""
        obs_cfg1 = DynamicObstacleConfig(
            id="obs_bounce",
            initial_x=2,
            initial_y=1,
            pattern="linear_bounce",
            velocity=(1, 0),
        )
        obs_cfg2 = DynamicObstacleConfig(
            id="obs_patrol",
            initial_x=0,
            initial_y=0,
            pattern="patrol",
            waypoints=(Position(0, 0), Position(0, 3)),
        )

        agent_cfg = AgentConfig(id="agent_1", initial_x=4, initial_y=4, initial_energy=10)
        world_cfg = MultiAgentWorldConfig(
            width=6,
            height=6,
            agents=(agent_cfg,),
            dynamic_obstacles=(obs_cfg1, obs_cfg2),
        )

        # Run 1
        sim1 = MultiAgentSimulation(world_cfg, seed=123)
        history1 = []
        for _ in range(15):
            sim1.step({"agent_1": Action.UP})
            history1.append({oid: (s.position, s.velocity) for oid, s in sim1.state.dynamic_obstacles.items()})

        # Run 2 with same seed
        sim2 = MultiAgentSimulation(world_cfg, seed=123)
        history2 = []
        for _ in range(15):
            sim2.step({"agent_1": Action.UP})
            history2.append({oid: (s.position, s.velocity) for oid, s in sim2.state.dynamic_obstacles.items()})

        self.assertEqual(len(history1), len(history2))
        for t, (step1, step2) in enumerate(zip(history1, history2)):
            self.assertEqual(step1, step2, f"Discrepancy at tick {t}")

    def test_obstacle_approaching_agent_destination(self) -> None:
        """2. Prove that an agent detects an oncoming obstacle and avoids vertex conflict."""
        # Agent at (1, 1) wants to move RIGHT to (2, 1).
        # Obstacle at (3, 1) moving LEFT to (2, 1) with velocity (-1, 0).
        agent_cfg = AgentConfig(
            id="agent_1",
            initial_x=1,
            initial_y=1,
            initial_energy=10,
            target=Position(5, 1),
        )
        obs_cfg = DynamicObstacleConfig(
            id="obs_oncoming",
            initial_x=3,
            initial_y=1,
            pattern="linear_bounce",
            velocity=(-1, 0),
        )
        world_cfg = MultiAgentWorldConfig(
            width=6,
            height=4,
            agents=(agent_cfg,),
            dynamic_obstacles=(obs_cfg,),
            enable_coordination=True,
            coordination_mode="decentralized",
        )

        sim = MultiAgentSimulation(world_cfg, seed=42)
        policy = RobustResourcePolicy(
            base_learner=self.base_learner,
            world_width=6,
            world_height=4,
            max_energy=10,
            coordinator=sim.coordinator,
        )

        obs = sim.observe_agent("agent_1")
        action = policy.select_action(obs)

        # Action should NOT be Action.RIGHT into (2, 1)
        self.assertNotEqual(action, Action.RIGHT, "Agent must not step into oncoming obstacle path at (2, 1)")

        # Execute step and verify physics
        step_res = sim.step({"agent_1": action})
        agent_pos = step_res.state.agents["agent_1"].position
        obs_pos = step_res.state.dynamic_obstacles["obs_oncoming"].position

        # No double occupancy
        self.assertNotEqual(agent_pos, obs_pos, "Agent and obstacle must never occupy the same cell")
        self.assertEqual(obs_pos, Position(2, 1))

    def test_two_moving_obstacles_crossing_paths(self) -> None:
        """3. Prove that when two obstacles cross paths, authoritative physics prevents double occupancy."""
        # Obstacle 1 at (2, 1) moving DOWN to (2, 2).
        # Obstacle 2 at (1, 2) moving RIGHT to (2, 2).
        obs1 = DynamicObstacleConfig(id="obs_v", initial_x=2, initial_y=1, pattern="linear_bounce", velocity=(0, 1))
        obs2 = DynamicObstacleConfig(id="obs_h", initial_x=1, initial_y=2, pattern="linear_bounce", velocity=(1, 0))

        agent_cfg = AgentConfig(id="agent_dummy", initial_x=0, initial_y=0, initial_energy=10)
        world_cfg = MultiAgentWorldConfig(
            width=5,
            height=5,
            agents=(agent_cfg,),
            dynamic_obstacles=(obs1, obs2),
        )

        sim = MultiAgentSimulation(world_cfg, seed=10)
        step_res = sim.step({"agent_dummy": Action.DOWN})

        p1 = step_res.state.dynamic_obstacles["obs_v"].position
        p2 = step_res.state.dynamic_obstacles["obs_h"].position

        # Authoritative collision resolution guarantees no double occupancy
        self.assertNotEqual(p1, p2, "Two obstacles must never occupy the same cell")
        self.assertFalse(p1 == Position(2, 2) and p2 == Position(2, 2), "Vertex contention must be resolved")

    def test_stale_observations_and_outside_communication_range(self) -> None:
        """4. Prove that obstacles outside R=2.0 are invisible and memory handles staleness via TTL."""
        # Agent at (1, 1). Obstacle at (2, 1) (dist = 1.0 <= 2.0).
        # Moves RIGHT with velocity (1, 0) -> (3, 1) [dist=2.0] -> (4, 1) [dist=3.0 > 2.0].
        agent_cfg = AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=10)
        obs_cfg = DynamicObstacleConfig(id="obs_moving", initial_x=2, initial_y=1, pattern="linear_bounce", velocity=(1, 0))

        world_cfg = MultiAgentWorldConfig(
            width=8,
            height=5,
            agents=(agent_cfg,),
            dynamic_obstacles=(obs_cfg,),
            sensory_radius=2.0,
            coordination_mode="decentralized",
        )

        sim = MultiAgentSimulation(world_cfg, seed=42)
        policy = RobustResourcePolicy(
            base_learner=self.base_learner,
            world_width=8,
            world_height=5,
            max_energy=10,
            coordinator=sim.coordinator,
        )

        # Tick 0: within range
        obs0 = sim.observe_agent("agent_1")
        self.assertEqual(len(obs0.observed_dynamic_obstacles), 1)
        self.assertEqual(obs0.observed_dynamic_obstacles[0].id, "obs_moving")
        policy.memory.update(obs0)
        self.assertIn("obs_moving", policy.memory.remembered_dynamic_obstacles)

        # Step 1: obstacle moves to (3, 1), dist = 2.0 <= 2.0
        sim.step({"agent_1": Action.DOWN})  # agent to (1, 2), dist to (3, 1) = sqrt(4+1) = sqrt(5) ~= 2.236 > 2.0!
        obs1 = sim.observe_agent("agent_1")
        # Beyond R=2.0: obstacle is NOT in observation
        self.assertEqual(len(obs1.observed_dynamic_obstacles), 0, "Distant obstacle must not leak into observation")

        # Memory retains previous belief until visual revalidation or TTL
        policy.memory.update(obs1)
        # Advance steps to exceed TTL (dynamic_obstacle_ttl = 6)
        for _ in range(7):
            obs_dummy = sim.observe_agent("agent_1")
            policy.memory.update(obs_dummy)

        # Stale memory has expired
        self.assertNotIn("obs_moving", policy.memory.remembered_dynamic_obstacles, "Stale obstacle belief must expire after TTL")

    def test_obstacle_blocked_by_wall_agent_and_resource(self) -> None:
        """5. Prove obstacles bounce or halt when blocked by walls, resources, or stationary agents."""
        # Obstacle at (0, 0) moving LEFT (into boundary)
        obs_wall = DynamicObstacleConfig(id="obs_w", initial_x=0, initial_y=0, pattern="linear_bounce", velocity=(-1, 0))
        # Obstacle at (1, 2) moving RIGHT into resource at (2, 2)
        obs_res = DynamicObstacleConfig(id="obs_r", initial_x=1, initial_y=2, pattern="linear_bounce", velocity=(1, 0))
        # Obstacle at (4, 4) moving UP into stationary agent at (4, 3)
        obs_ag = DynamicObstacleConfig(id="obs_a", initial_x=4, initial_y=4, pattern="linear_bounce", velocity=(0, -1))

        agent_stationary = AgentConfig(id="agent_stat", initial_x=4, initial_y=3, initial_energy=10)
        res = Resource(Position(2, 2), 5, True)

        world_cfg = MultiAgentWorldConfig(
            width=6,
            height=6,
            agents=(agent_stationary,),
            resources=(res,),
            dynamic_obstacles=(obs_wall, obs_res, obs_ag),
        )

        sim = MultiAgentSimulation(world_cfg, seed=5)
        # Agent remains stationary with RECHARGE
        step_res = sim.step({"agent_stat": Action.RECHARGE})

        # Wall bounce: stayed at (0, 0) and inverted velocity to (+1, 0)
        obs_w_state = step_res.state.dynamic_obstacles["obs_w"]
        self.assertEqual(obs_w_state.position, Position(0, 0))
        self.assertEqual(obs_w_state.velocity, (1, 0))

        # Resource bounce: cannot enter (2, 2), stayed at (1, 2) and inverted velocity to (-1, 0)
        obs_r_state = step_res.state.dynamic_obstacles["obs_r"]
        self.assertEqual(obs_r_state.position, Position(1, 2))
        self.assertEqual(obs_r_state.velocity, (-1, 0))

        # Agent obstruction: cannot enter (4, 3), stayed at (4, 4) and inverted velocity to (0, 1)
        obs_a_state = step_res.state.dynamic_obstacles["obs_a"]
        self.assertEqual(obs_a_state.position, Position(4, 4))
        self.assertEqual(obs_a_state.velocity, (0, 1))

    def test_safe_waiting_and_eventual_progress(self) -> None:
        """6. Prove an agent safely waits for an obstacle to cross a chokepoint and makes progress."""
        # Corridor: doorway at (2, 2). Agent at (1, 2) targeting (4, 2).
        # Obstacle oscillates vertically across doorway: (2, 4) <-> (2, 0) through (2, 2).
        agent_cfg = AgentConfig(id="agent_runner", initial_x=1, initial_y=2, initial_energy=10, target=Position(4, 2))
        obs_cfg = DynamicObstacleConfig(
            id="obs_patrol",
            initial_x=2,
            initial_y=2,
            pattern="patrol",
            waypoints=(Position(2, 4), Position(2, 0)),
        )

        world_cfg = MultiAgentWorldConfig(
            width=6,
            height=5,
            agents=(agent_cfg,),
            dynamic_obstacles=(obs_cfg,),
            enable_coordination=True,
            coordination_mode="decentralized",
        )

        sim = MultiAgentSimulation(world_cfg, seed=1)
        policy = RobustResourcePolicy(
            base_learner=self.base_learner,
            world_width=6,
            world_height=5,
            max_energy=10,
            coordinator=sim.coordinator,
        )

        completed = False
        for step in range(20):
            obs = sim.observe_agent("agent_runner")
            action = policy.select_action(obs)
            step_res = sim.step({"agent_runner": action})
            agent_state = step_res.state.agents["agent_runner"]
            policy.record_step_result(agent_state.position, agent_state.energy, action, step_res.agent_results["agent_runner"].success)

            # Assert no collision ever occurred
            self.assertNotEqual(agent_state.position, step_res.state.dynamic_obstacles["obs_patrol"].position)

            if agent_state.status == "reached_target":
                completed = True
                break

        self.assertTrue(completed, "Agent must wait and eventually traverse to target")

    def test_u06_quadrant_mesh_regression_coverage(self) -> None:
        """7. Verify U06 achieves >= 95% (target 100%) completion under decentralized mode."""
        specs = {spec.scenario_id: (spec, prep) for spec, prep in build_all_unseen_scenarios()}
        spec, prep = specs["U06"]
        seeds = list(range(500, 520))
        res = run_unseen_scenario_eval(spec, prep, None, "decentralized", seeds, max_steps=50)
        self.assertEqual(res.overall_completion_rate, 100.0, "U06 must achieve 100% completion in decentralized mode")
        self.assertEqual(res.total_exhaustions, 0, "U06 must have 0 exhaustions")

    def test_u14_blind_junction_regression_coverage(self) -> None:
        """8. Verify U14 achieves >= 95% (target 100%) completion under decentralized mode."""
        specs = {spec.scenario_id: (spec, prep) for spec, prep in build_all_unseen_scenarios()}
        spec, prep = specs["U14"]
        seeds = list(range(500, 520))
        res = run_unseen_scenario_eval(spec, prep, None, "decentralized", seeds, max_steps=50)
        self.assertEqual(res.overall_completion_rate, 100.0, "U14 must achieve 100% completion in decentralized mode")
        self.assertEqual(res.total_exhaustions, 0, "U14 must have 0 exhaustions")


if __name__ == "__main__":
    unittest.main()
