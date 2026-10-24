import unittest

from worldsim import (
    Action,
    EnergyPlanResult,
    EnergyPlannerPolicy,
    EnergyPlanningProblem,
    LearningConfig,
    PlanExecutionResult,
    Position,
    QLearner,
    Resource,
    WorldConfig,
    execute_energy_plan,
    plan_energy,
    run_policy_experiment,
)


class EnergyPlanningTests(unittest.TestCase):
    def test_energy_planning_direct_path(self) -> None:
        # Distance = 3, initial_energy = 5 -> Reachable directly without recharge
        problem = EnergyPlanningProblem(
            width=5,
            height=5,
            start=Position(0, 0),
            target=Position(2, 1),
            initial_energy=5,
            max_energy=10,
            movement_energy_cost=1,
            resources=(),
        )
        result = plan_energy(problem)
        self.assertTrue(result.found)
        self.assertEqual(result.plan_length, 3)
        self.assertEqual(len(result.actions), 3)
        self.assertNotIn(Action.RECHARGE, result.actions)
        self.assertEqual(result.expected_final_position, Position(2, 1))
        self.assertEqual(result.expected_final_energy, 2)
        self.assertIsNone(result.failure_reason)

    def test_energy_planning_with_single_recharge(self) -> None:
        # Distance = 6, initial_energy = 3 -> Cannot reach directly.
        # Resource at (1, 0) provides 5 energy.
        res = Resource(Position(1, 0), recharge_amount=5, consumable=True)
        problem = EnergyPlanningProblem(
            width=5,
            height=5,
            start=Position(0, 0),
            target=Position(3, 3),
            initial_energy=3,
            max_energy=8,
            movement_energy_cost=1,
            resources=(res,),
        )
        result = plan_energy(problem)
        self.assertTrue(result.found)
        # 1 move to (1, 0) + 1 recharge + 5 moves to (3, 3) = 7 actions
        self.assertEqual(result.plan_length, 7)
        self.assertIn(Action.RECHARGE, result.actions)
        self.assertEqual(result.actions.count(Action.RECHARGE), 1)
        self.assertEqual(result.expected_final_position, Position(3, 3))
        # Initial 3 -> move to (1, 0) has 2 -> recharge min(8, 2+5)=7 -> 5 moves -> 2 energy
        self.assertEqual(result.expected_final_energy, 2)

    def test_energy_planning_with_multiple_recharges(self) -> None:
        # Distance = 10, initial_energy = 4, max_energy = 6.
        # Two consumable resources along the path.
        res1 = Resource(Position(2, 0), recharge_amount=4, consumable=True)
        res2 = Resource(Position(4, 2), recharge_amount=4, consumable=True)
        problem = EnergyPlanningProblem(
            width=6,
            height=6,
            start=Position(0, 0),
            target=Position(5, 5),
            initial_energy=4,
            max_energy=6,
            movement_energy_cost=1,
            resources=(res1, res2),
        )
        result = plan_energy(problem)
        self.assertTrue(result.found)
        # (0,0)->(2,0) [2] + recharge [1] + ->(4,2) [4] + recharge [1] + ->(5,5) [4] = 12
        self.assertEqual(result.plan_length, 12)
        self.assertEqual(result.actions.count(Action.RECHARGE), 2)
        self.assertEqual(result.expected_final_position, Position(5, 5))
        self.assertEqual(result.expected_final_energy, 2)

    def test_energy_planning_unreachable_target(self) -> None:
        # Distance = 8, initial_energy = 3, no resources
        problem = EnergyPlanningProblem(
            width=5,
            height=5,
            start=Position(0, 0),
            target=Position(4, 4),
            initial_energy=3,
            max_energy=5,
            movement_energy_cost=1,
            resources=(),
        )
        result = plan_energy(problem)
        self.assertFalse(result.found)
        self.assertEqual(result.actions, ())
        self.assertIsNone(result.expected_final_position)
        self.assertIsNone(result.expected_final_energy)
        self.assertIn("unreachable", result.failure_reason.lower())

    def test_consumable_vs_reusable_resource_planning(self) -> None:
        # Reusable resource at (1, 0) can be recharged multiple times if max_energy allows
        reusable_res = Resource(Position(1, 0), recharge_amount=2, consumable=False)
        problem = EnergyPlanningProblem(
            width=5,
            height=5,
            start=Position(0, 0),
            target=Position(3, 1),  # Distance 4
            initial_energy=2,
            max_energy=5,
            movement_energy_cost=1,
            resources=(reusable_res,),
        )
        result = plan_energy(problem)
        self.assertTrue(result.found)
        self.assertEqual(result.expected_final_position, Position(3, 1))

    def test_invalid_planning_inputs(self) -> None:
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(0, 5, Position(0, 0), Position(1, 1))
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(5, 5, Position(-1, 0), Position(1, 1))
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(5, 5, Position(0, 0), Position(5, 1))
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(5, 5, Position(0, 0), Position(1, 1), initial_energy=10, max_energy=5)
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(5, 5, Position(0, 0), Position(1, 1), initial_energy=-1)
        with self.assertRaises(ValueError):
            EnergyPlanningProblem(5, 5, Position(0, 0), Position(1, 1), movement_energy_cost=-1)

    def test_execute_energy_plan_success(self) -> None:
        res = Resource(Position(2, 0), recharge_amount=5, consumable=True)
        problem = EnergyPlanningProblem(
            width=5,
            height=5,
            start=Position(0, 0),
            target=Position(3, 1),
            initial_energy=3,
            max_energy=6,
            movement_energy_cost=1,
            resources=(res,),
        )
        plan_res = plan_energy(problem)
        self.assertTrue(plan_res.found)

        config = WorldConfig(
            width=5,
            height=5,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=6,
            movement_energy_cost=1,
            resources=(res,),
        )
        exec_res = execute_energy_plan(config, problem, plan_res)
        self.assertTrue(exec_res.completed)
        self.assertTrue(exec_res.reached_target)
        self.assertEqual(exec_res.final_position, Position(3, 1))
        self.assertEqual(exec_res.final_energy, plan_res.expected_final_energy)
        self.assertEqual(exec_res.actual_steps, plan_res.plan_length)
        self.assertIsNone(exec_res.divergence_reason)

    def test_execute_energy_plan_consistency_validation(self) -> None:
        problem = EnergyPlanningProblem(
            width=4,
            height=4,
            start=Position(0, 0),
            target=Position(2, 2),
            initial_energy=5,
        )
        plan_res = plan_energy(problem)

        # Mismatched start position
        with self.assertRaises(ValueError):
            execute_energy_plan(
                WorldConfig(width=4, height=4, initial_x=1, initial_y=0, initial_energy=5),
                problem,
                plan_res,
            )

        # Mismatched energy
        with self.assertRaises(ValueError):
            execute_energy_plan(
                WorldConfig(width=4, height=4, initial_x=0, initial_y=0, initial_energy=2),
                problem,
                plan_res,
            )

    def test_execute_energy_plan_unfound_handling(self) -> None:
        problem = EnergyPlanningProblem(
            width=4,
            height=4,
            start=Position(0, 0),
            target=Position(3, 3),
            initial_energy=2,
            resources=(),
        )
        plan_res = plan_energy(problem)
        self.assertFalse(plan_res.found)

        config = WorldConfig(width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, resources=())
        exec_res = execute_energy_plan(config, problem, plan_res)
        self.assertFalse(exec_res.completed)
        self.assertFalse(exec_res.reached_target)
        self.assertIn("cannot execute unfound plan", exec_res.divergence_reason)

    def test_planner_policy_reaches_target_with_recharge(self) -> None:
        target = Position(4, 2)
        res = Resource(Position(2, 0), recharge_amount=6, consumable=True)
        config = WorldConfig(
            width=5,
            height=5,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=8,
            movement_energy_cost=1,
            sensory_radius=3.0,
            resources=(res,),
        )
        policy = EnergyPlannerPolicy(
            target=target,
            world_width=5,
            world_height=5,
            max_energy=8,
            movement_energy_cost=1,
        )
        exp_res = run_policy_experiment(config, policy, max_steps=20, stop_on_rejected=True)
        self.assertTrue(exp_res.completed)
        self.assertEqual(exp_res.final_position, target)
        self.assertIn(Action.RECHARGE, exp_res.actions)

    def test_q_learner_energy_state_key_opt_in(self) -> None:
        cfg_default = LearningConfig(target=Position(2, 2), include_energy=False)
        learner_default = QLearner(cfg_default)

        cfg_energy = LearningConfig(target=Position(2, 2), include_energy=True)
        learner_energy = QLearner(cfg_energy)

        from worldsim import Simulation
        sim = Simulation(WorldConfig(initial_x=1, initial_y=1, initial_energy=7))
        obs = sim.observe()

        self.assertEqual(learner_default.state_key(obs), (1, 1))
        self.assertEqual(learner_energy.state_key(obs), (1, 1, 7))


if __name__ == "__main__":
    unittest.main()
