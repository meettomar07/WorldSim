"""Unit tests for Phase 13.5: Fixes for Reciprocal Target Lock and Sensory-Horizon Oscillation.

Validates:
1. Deterministic resolution of Failure A (reciprocal target lock / head-on swap conflict)
   without persistent deadlock or timeout.
2. Invariance across arbitrary agent IDs and coordinates (no hardcoding).
3. Deterministic resolution of Failure B (sensory-horizon oscillation) via local
   obstacle memory and smooth perimeter detour.
4. Visual revalidation: purging cleared obstacles when re-observed within sensory range.
5. TTL expiration: purging unobserved stale obstacle memory after bounded lifespan.
6. Energy-bounded BFS fallback: routing through yielding completed agents when
   avoidance detour exceeds remaining energy.
7. Verification that feasible benchmark scenarios (Scenario 4B & 5B) achieve 100% completion.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import MultiAgentCoordinator
from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.multi_agent_experiment_phase13_4 import (
    make_robust_factory,
    run_phase13_4_episode,
)
from worldsim.robust_policy import AgentMemory, RobustResourcePolicy


class TestPhase13_5CoordinationAndMemoryFixes(unittest.TestCase):
    """Unit tests for Phase 13.5 coordination and memory improvements."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train standard base linear learner once for all tests."""
        train_world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
        )
        lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
        lin_runner.train(1000)
        cls.base_learner = lin_runner.learner

    def test_failure_a_reciprocal_target_lock_resolved(self) -> None:
        """Failure A: Head-on reciprocal target lock resolves cleanly within 4 steps."""
        cfg = MultiAgentWorldConfig(
            width=8, height=8,
            agents=(
                AgentConfig("agent_3", initial_x=4, initial_y=7, initial_energy=10, target=Position(5, 7)),
                AgentConfig("agent_4", initial_x=5, initial_y=7, initial_energy=10, target=Position(4, 7)),
            ),
            collision_rule="priority_rotation",
            enable_coordination=True,
        )
        sim = MultiAgentSimulation(cfg, seed=100)
        p3 = RobustResourcePolicy(8, 8, 10, base_learner=self.base_learner, coordinator=sim.coordinator)
        p4 = RobustResourcePolicy(8, 8, 10, base_learner=self.base_learner, coordinator=sim.coordinator)

        steps = 0
        max_steps = 10
        swap_collisions = 0
        while steps < max_steps and any(a.active for a in sim.state.agents.values()):
            if sim.coordinator:
                sim.coordinator.reset_tick()
            obs3 = sim.observe_agent("agent_3")
            obs4 = sim.observe_agent("agent_4")
            act3 = p3.choose_action(obs3)
            act4 = p4.choose_action(obs4)
            res = sim.step({"agent_3": act3, "agent_4": act4})
            steps += 1
            if res.agent_results["agent_3"].collision_type == "swap" or res.agent_results["agent_4"].collision_type == "swap":
                swap_collisions += 1

        self.assertEqual(swap_collisions, 0, "No swap collisions should occur during reciprocal target resolution")
        self.assertEqual(sim.state.agents["agent_3"].status, "reached_target")
        self.assertEqual(sim.state.agents["agent_4"].status, "reached_target")
        self.assertLessEqual(steps, 4, f"Should resolve in <= 4 steps, took {steps}")

    def test_failure_a_arbitrary_agent_ids_and_positions(self) -> None:
        """Verify swap resolution works with arbitrary non-numbered IDs and different coordinates."""
        cfg = MultiAgentWorldConfig(
            width=6, height=6,
            agents=(
                AgentConfig("alpha", initial_x=2, initial_y=3, initial_energy=10, target=Position(3, 3)),
                AgentConfig("beta", initial_x=3, initial_y=3, initial_energy=10, target=Position(2, 3)),
            ),
            collision_rule="priority_rotation",
            enable_coordination=True,
        )
        sim = MultiAgentSimulation(cfg, seed=42)
        pa = RobustResourcePolicy(6, 6, 10, base_learner=self.base_learner, coordinator=sim.coordinator)
        pb = RobustResourcePolicy(6, 6, 10, base_learner=self.base_learner, coordinator=sim.coordinator)

        steps = 0
        max_steps = 10
        while steps < max_steps and any(a.active for a in sim.state.agents.values()):
            if sim.coordinator:
                sim.coordinator.reset_tick()
            obsa = sim.observe_agent("alpha")
            obsb = sim.observe_agent("beta")
            acta = pa.choose_action(obsa)
            actb = pb.choose_action(obsb)
            sim.step({"alpha": acta, "beta": actb})
            steps += 1

        self.assertEqual(sim.state.agents["alpha"].status, "reached_target")
        self.assertEqual(sim.state.agents["beta"].status, "reached_target")
        self.assertLessEqual(steps, 4)

    def test_failure_b_sensory_horizon_oscillation_resolved(self) -> None:
        """Failure B: Sensory-horizon obstacle memory prevents ping-pong oscillation."""
        cfg = MultiAgentWorldConfig(
            width=8, height=8,
            agents=(
                AgentConfig("agent_1", initial_x=4, initial_y=7, initial_energy=10, target=Position(7, 7)),
                AgentConfig("agent_completed", initial_x=6, initial_y=7, initial_energy=10, target=Position(6, 7)),
            ),
            sensory_radius=2.0,
            collision_rule="priority_rotation",
            enable_coordination=True,
        )
        sim = MultiAgentSimulation(cfg, seed=100)
        # Put agent_completed directly in reached_target resting state
        sim._state.agents["agent_completed"] = replace(
            sim._state.agents["agent_completed"],
            status="reached_target",
            active=False,
        )

        p1 = RobustResourcePolicy(8, 8, 10, base_learner=self.base_learner, sensory_radius=2.0, coordinator=sim.coordinator)

        steps = 0
        max_steps = 10
        positions = []
        while steps < max_steps and any(a.active for a in sim.state.agents.values()):
            if sim.coordinator:
                sim.coordinator.reset_tick()
            obs1 = sim.observe_agent("agent_1")
            act1 = p1.choose_action(obs1)
            sim.step({"agent_1": act1, "agent_completed": Action.UP})
            steps += 1
            pos = sim.state.agents["agent_1"].position
            positions.append((pos.x, pos.y))

        # Check trajectory: Agent 1 should make a smooth detour via y=6 without oscillating back to (4, 7)
        self.assertNotIn((4, 7), positions[1:], "Agent should not oscillate back to (4, 7) after detouring to (4, 6)")
        self.assertEqual(sim.state.agents["agent_1"].status, "reached_target")
        self.assertEqual(sim.state.agents["agent_1"].position, Position(7, 7))
        self.assertEqual(steps, 5, f"Smooth perimeter detour takes exactly 5 steps, took {steps}")

    def test_obstacle_memory_visual_revalidation(self) -> None:
        """Obstacle memory correctly clears when a cell is re-observed and found unoccupied."""
        mem = AgentMemory(world_width=8, world_height=8)
        mem.step_count = 1

        # Simulate observing an inactive agent at (3, 3)
        obs_agent = AgentState(id="other", position=Position(3, 3), energy=5, status="reached_target", active=False)
        obs_cells = tuple(Position(x, y) for x in range(2, 5) for y in range(2, 5))
        obs = MultiAgentObservation(
            agent_id="test_agent",
            entity_position=Position(2, 3),
            cells=obs_cells,
            energy=10,
            observed_resources=(),
            target=Position(7, 7),
            observed_agents=(obs_agent,),
        )
        mem.update(obs)
        self.assertIn(Position(3, 3), mem.remembered_obstacles)
        self.assertEqual(mem.get_remembered_inactive_obstacles()[Position(3, 3)], "reached_target")

        # Simulate agent moving to Position(4, 3) and looking at (3, 3) where the obstacle is now GONE
        obs_empty = MultiAgentObservation(
            agent_id="test_agent",
            entity_position=Position(4, 3),
            cells=obs_cells,
            energy=9,
            observed_resources=(),
            target=Position(7, 7),
            observed_agents=(),  # No agent observed now!
        )
        mem.update(obs_empty)
        self.assertNotIn(Position(3, 3), mem.remembered_obstacles, "Obstacle at (3, 3) must be purged via visual revalidation")

    def test_obstacle_memory_ttl_expiration(self) -> None:
        """Obstacle memory expires after TTL when the obstacle is never re-observed."""
        mem = AgentMemory(world_width=8, world_height=8, obstacle_memory_ttl=10)
        mem.step_count = 1

        # Record obstacle at (6, 7)
        obs_agent = AgentState(id="other", position=Position(6, 7), energy=5, status="exhausted", active=False)
        obs_cells = (Position(5, 7), Position(6, 7))
        obs = MultiAgentObservation(
            agent_id="test_agent",
            entity_position=Position(5, 7),
            cells=obs_cells,
            energy=10,
            observed_resources=(),
            target=Position(7, 7),
            observed_agents=(obs_agent,),
        )
        mem.update(obs)
        self.assertIn(Position(6, 7), mem.remembered_obstacles)

        # Step forward 12 steps observing far-away cells (0, 0)
        far_cells = (Position(0, 0), Position(0, 1))
        for step in range(2, 14):
            mem.step_count = step
            obs_far = MultiAgentObservation(
                agent_id="test_agent",
                entity_position=Position(0, 0),
                cells=far_cells,
                energy=10,
                observed_resources=(),
                target=Position(7, 7),
                observed_agents=(),
            )
            mem.update(obs_far)

        # Since step 13 - step 1 = 12 > ttl=10, it must be expired and pruned
        self.assertNotIn(Position(6, 7), mem.remembered_obstacles, "Obstacle must expire after TTL")

    def test_energy_bounded_bfs_fallback(self) -> None:
        """When detour path length exceeds remaining energy, agent falls back to yielding completed path."""
        policy = RobustResourcePolicy(world_width=8, world_height=8, max_energy=10, base_learner=self.base_learner)
        # Position at (0, 0), target at (2, 0). Completed agent at (1, 0).
        # Direct path length through (1, 0) is 2 steps: (1, 0) -> (2, 0).
        # Perimeter detour avoiding (1, 0) might take 4 steps: (0, 1) -> (1, 1) -> (2, 1) -> (2, 0).
        # If agent only has energy=2: len(detour)=4 > energy=2.
        # Agent MUST fall back to routing through (1, 0) because completed agent yields!
        completed_positions = {Position(1, 0)}
        exhausted_positions = set()

        path_strict = policy._find_path_bfs(Position(0, 0), Position(2, 0), completed_positions)
        self.assertGreater(len(path_strict), 2, "Strict avoidance must take a detour")

        # Now simulate the fallback logic used in choose_action
        energy = 2
        path = path_strict
        if not path or len(path) > energy:
            path = policy._find_path_bfs(Position(0, 0), Position(2, 0), exhausted_positions)

        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], Position(1, 0))

    def test_scenario_4b_100_percent_completion(self) -> None:
        """Scenario 4B (Relocated resources with energy=7) achieves 100% completion under Phase 13.5."""
        agents = [
            AgentConfig(id=f"agent_{i+1}", initial_x=i % 4, initial_y=i // 4, initial_energy=7, target=Position(7 - (i % 4), 7 - (i // 4)))
            for i in range(8)
        ]
        cfg = MultiAgentWorldConfig(
            width=8, height=8,
            agents=tuple(agents),
            resources=(Resource(Position(2, 5), 6, False), Resource(Position(5, 2), 6, False)),
            max_energy=10,
            collision_rule="priority_rotation",
            resource_allocation_rule="fair_balanced",
            enable_coordination=True,
            cooperative_sharing=True,
            goal_criterion="individual",
        )
        for seed in (100, 101, 102, 103, 104):
            sim = MultiAgentSimulation(cfg, seed=seed)
            policies = {a.id: make_robust_factory(8, 8, 10, self.base_learner, margin=1)() for a in agents}
            for p in policies.values():
                if hasattr(p, "coordinator") and p.coordinator is None:
                    p.coordinator = sim.coordinator
                if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                    p.utility_evaluator.coordinator = sim.coordinator

            comp, steps, reached, consumed, recharges = run_phase13_4_episode(sim, policies, max_steps=50)
            self.assertTrue(comp, f"Scenario 4B failed on seed {seed}")
            self.assertEqual(sum(reached.values()), 8, f"All 8 agents must reach target on seed {seed}")

    def test_scenario_5b_100_percent_completion(self) -> None:
        """Scenario 5B (Asymmetric starts with energy=8) achieves 100% completion under Phase 13.5."""
        asym_starts = [
            (0, 0, 7, 7), (0, 7, 7, 0), (7, 0, 0, 7), (7, 7, 0, 0),
            (1, 0, 6, 7), (0, 6, 7, 1), (6, 7, 1, 0), (7, 1, 0, 6),
        ]
        agents = [
            AgentConfig(id=f"agent_{i+1}", initial_x=sx, initial_y=sy, initial_energy=8, target=Position(gx, gy))
            for i, (sx, sy, gx, gy) in enumerate(asym_starts)
        ]
        cfg = MultiAgentWorldConfig(
            width=8, height=8,
            agents=tuple(agents),
            resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
            max_energy=10,
            collision_rule="priority_rotation",
            resource_allocation_rule="fair_balanced",
            enable_coordination=True,
            cooperative_sharing=True,
            goal_criterion="individual",
        )
        for seed in (100, 101, 102, 103, 104):
            sim = MultiAgentSimulation(cfg, seed=seed)
            policies = {a.id: make_robust_factory(8, 8, 10, self.base_learner, margin=1)() for a in agents}
            for p in policies.values():
                if hasattr(p, "coordinator") and p.coordinator is None:
                    p.coordinator = sim.coordinator
                if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                    p.utility_evaluator.coordinator = sim.coordinator

            comp, steps, reached, consumed, recharges = run_phase13_4_episode(sim, policies, max_steps=50)
            self.assertTrue(comp, f"Scenario 5B failed on seed {seed}")
            self.assertEqual(sum(reached.values()), 8, f"All 8 agents must reach target on seed {seed}")


if __name__ == "__main__":
    unittest.main()
