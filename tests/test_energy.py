import unittest

from worldsim import (
    Action,
    EnergyAwarePolicy,
    EnergyIgnorantPolicy,
    ExperienceMemory,
    LearningConfig,
    LearningRunner,
    PlanningProblem,
    Position,
    Resource,
    Simulation,
    WorldConfig,
    execute_plan,
    plan,
    run_experiment,
    run_policy_experiment,
)


class EnergyTests(unittest.TestCase):
    def test_energy_initialization_and_validation(self) -> None:
        config = WorldConfig(initial_energy=8, max_energy=12, movement_energy_cost=2)
        sim = Simulation(config)
        self.assertEqual(sim.state.entity.energy, 8)
        self.assertEqual(sim.observe().energy, 8)

        # Invalid energy configurations
        with self.assertRaises(ValueError):
            WorldConfig(initial_energy=15, max_energy=10)
        with self.assertRaises(ValueError):
            WorldConfig(initial_energy=-1, max_energy=10)
        with self.assertRaises(ValueError):
            WorldConfig(max_energy=0)
        with self.assertRaises(ValueError):
            WorldConfig(movement_energy_cost=-1)

    def test_energy_consumption_on_movement(self) -> None:
        config = WorldConfig(width=5, height=5, initial_x=0, initial_y=0, initial_energy=5, movement_energy_cost=2)
        sim = Simulation(config)
        result1 = sim.step(Action.RIGHT)
        self.assertTrue(result1.moved)
        self.assertEqual(result1.state.entity.energy, 3)
        self.assertEqual(result1.observation.energy, 3)

        result2 = sim.step(Action.DOWN)
        self.assertTrue(result2.moved)
        self.assertEqual(result2.state.entity.energy, 1)

    def test_insufficient_energy_prevents_movement(self) -> None:
        config = WorldConfig(width=4, height=4, initial_x=0, initial_y=0, initial_energy=1, movement_energy_cost=2)
        sim = Simulation(config)
        result = sim.step(Action.RIGHT)
        self.assertFalse(result.moved)
        self.assertFalse(result.success)
        self.assertEqual(sim.state.tick, 0)
        self.assertEqual(sim.state.entity.position, Position(0, 0))
        self.assertEqual(sim.state.entity.energy, 1)
        self.assertEqual(len(sim.experiences), 1)
        self.assertFalse(sim.experiences[0].moved)
        self.assertFalse(sim.experiences[0].success)

    def test_boundary_and_paused_rejections_consume_no_energy(self) -> None:
        config = WorldConfig(width=3, height=3, initial_x=0, initial_y=0, initial_energy=5)
        sim = Simulation(config)
        # Attempt move into top boundary
        result = sim.step(Action.UP)
        self.assertFalse(result.moved)
        self.assertEqual(sim.state.entity.energy, 5)
        self.assertEqual(sim.state.tick, 0)

        # Paused rejection
        sim.pause()
        result_paused = sim.step(Action.RIGHT)
        self.assertFalse(result_paused.moved)
        self.assertEqual(sim.state.entity.energy, 5)
        self.assertEqual(sim.state.tick, 0)

    def test_resource_validation(self) -> None:
        # Out of bounds
        with self.assertRaises(ValueError):
            WorldConfig(width=3, height=3, resources=(Resource(Position(3, 1)),))
        with self.assertRaises(ValueError):
            WorldConfig(width=3, height=3, resources=(Resource(Position(-1, 0)),))
        # Negative / zero recharge amount
        with self.assertRaises(ValueError):
            Resource(Position(1, 1), recharge_amount=0)
        with self.assertRaises(ValueError):
            Resource(Position(1, 1), recharge_amount=-2)
        # Duplicate resource coordinates
        with self.assertRaises(ValueError):
            WorldConfig(
                width=3,
                height=3,
                resources=(Resource(Position(1, 1)), Resource(Position(1, 1))),
            )

    def test_resource_recharge_consumable_and_reusable(self) -> None:
        # Consumable resource
        res = Resource(Position(1, 0), recharge_amount=4, consumable=True)
        config = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            max_energy=5,
            resources=(res,),
        )
        sim = Simulation(config)
        # Move to (1, 0)
        sim.step(Action.RIGHT)
        self.assertEqual(sim.state.entity.energy, 1)

        # Recharge at (1, 0)
        recharge_res = sim.step(Action.RECHARGE)
        self.assertFalse(recharge_res.moved)
        self.assertTrue(recharge_res.success)
        self.assertEqual(sim.state.tick, 2)
        self.assertEqual(sim.state.entity.energy, 5)  # 1 + 4 = 5
        self.assertEqual(sim.state.resources, ())  # Consumed

        # Second recharge attempt on now-empty cell fails
        failed_res = sim.step(Action.RECHARGE)
        self.assertFalse(failed_res.success)
        self.assertEqual(sim.state.tick, 2)  # Tick does not advance on rejection

        # Reusable resource
        reusable_res = Resource(Position(0, 0), recharge_amount=3, consumable=False)
        sim_reusable = Simulation(
            WorldConfig(width=3, height=3, initial_x=0, initial_y=0, initial_energy=2, max_energy=6, resources=(reusable_res,))
        )
        sim_reusable.step(Action.RECHARGE)
        self.assertEqual(sim_reusable.state.entity.energy, 5)
        self.assertEqual(len(sim_reusable.state.resources), 1)  # Stays active

    def test_invalid_resource_interaction_rejected(self) -> None:
        config = WorldConfig(width=3, height=3, initial_x=0, initial_y=0, resources=())
        sim = Simulation(config)
        result = sim.step(Action.RECHARGE)
        self.assertFalse(result.moved)
        self.assertFalse(result.success)
        self.assertEqual(sim.state.tick, 0)
        self.assertEqual(len(sim.experiences), 1)
        self.assertFalse(sim.experiences[0].success)

    def test_observation_correctness_and_immutability(self) -> None:
        r_near = Resource(Position(1, 1), recharge_amount=3)
        r_far = Resource(Position(4, 4), recharge_amount=3)
        config = WorldConfig(
            width=5,
            height=5,
            initial_x=0,
            initial_y=0,
            sensory_radius=2.0,
            resources=(r_near, r_far),
        )
        sim = Simulation(config)
        obs = sim.observe()
        self.assertEqual(obs.energy, 10)
        self.assertEqual(obs.observed_resources, (r_near,))
        self.assertNotIn(r_far, obs.observed_resources)

        # Immutability check
        with self.assertRaises(AttributeError):
            obs.energy = 20  # type: ignore[misc]

    def test_multi_step_experiment_with_stop_on_rejected(self) -> None:
        config = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            movement_energy_cost=1,
        )
        # 3 moves: 2 will succeed, 3rd will fail due to 0 energy
        actions = (Action.RIGHT, Action.RIGHT, Action.RIGHT)
        res_continue = run_experiment(config, actions, stop_on_rejected=False)
        self.assertTrue(res_continue.completed)
        self.assertEqual(len(res_continue.results), 3)
        self.assertFalse(res_continue.results[2].success)

        res_stop = run_experiment(config, actions, stop_on_rejected=True)
        self.assertFalse(res_stop.completed)
        self.assertEqual(res_stop.stopped_at_index, 2)
        self.assertEqual(len(res_stop.results), 3)
        self.assertEqual(res_stop.final_energy, 0)

    def test_energy_aware_policy_vs_energy_ignorant(self) -> None:
        target = Position(3, 3)
        resource = Resource(Position(2, 0), recharge_amount=6, consumable=True)
        config = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=8,
            movement_energy_cost=1,
            sensory_radius=3.0,
            resources=(resource,),
        )

        # Ignorant policy fails due to energy exhaustion
        ignorant = EnergyIgnorantPolicy(target)
        res_ignorant = run_policy_experiment(config, ignorant, max_steps=20, stop_on_rejected=True)
        self.assertFalse(res_ignorant.completed)
        self.assertNotEqual(res_ignorant.final_position, target)

        # Energy aware policy detours to resource, recharges, and reaches target
        aware = EnergyAwarePolicy(target=target, movement_cost=1)
        res_aware = run_policy_experiment(config, aware, max_steps=20, stop_on_rejected=True)
        self.assertTrue(res_aware.completed)
        self.assertEqual(res_aware.final_position, target)
        self.assertIn(Action.RECHARGE, res_aware.actions)

    def test_compatibility_with_existing_components(self) -> None:
        # Planning: BFS works, execute_plan halts gracefully if energy runs out
        prob = PlanningProblem(4, 4, Position(0, 0), Position(3, 3))
        res_plan = plan(prob)
        self.assertTrue(res_plan.found)
        # Execute with low energy (3 energy for 6 steps)
        steps = execute_plan(
            WorldConfig(width=4, height=4, initial_x=0, initial_y=0, initial_energy=3),
            res_plan,
        )
        self.assertEqual(len(steps), 4)  # 3 successful moves + 1 rejected move
        self.assertFalse(steps[-1].moved)

        # Q-Learning: runner resets restore energy
        runner = LearningRunner(
            WorldConfig(width=3, height=3, initial_x=0, initial_y=0, initial_energy=5),
            LearningConfig(target=Position(1, 0), epsilon=0.0, max_steps=5),
        )
        episodes = runner.train(2)
        self.assertTrue(episodes[0].reached_target)

        # Experience Memory
        memory = ExperienceMemory(10)
        sim = Simulation(WorldConfig(initial_energy=4))
        obs_before = sim.observe()
        step = sim.step(Action.RIGHT)
        memory.record(obs_before, Action.RIGHT, 1.0, step.observation)
        retrieved = memory.retrieve(obs_before)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].observation.energy, 4)
        self.assertEqual(retrieved[0].resulting_observation.energy, 3)


if __name__ == "__main__":
    unittest.main()
