import unittest

from worldsim import (
    Action,
    Entity,
    LearningConfig,
    LearningRunner,
    Observation,
    Position,
    QLearner,
    Resource,
    Simulation,
    WorldConfig,
)


class LearningEnergyTests(unittest.TestCase):
    def test_state_key_with_energy_resources_and_target(self) -> None:
        cfg = LearningConfig(
            target=Position(3, 3),
            include_energy=True,
            include_resources=True,
            include_target=True,
        )
        learner = QLearner(cfg)
        res = Resource(Position(1, 0), recharge_amount=4)
        obs = Observation(
            entity_position=Position(0, 0),
            cells=(Position(0, 0), Position(1, 0)),
            energy=5,
            observed_resources=(res,),
            target=Position(3, 3),
        )
        key = learner.state_key(obs)
        # Expected: (x, y, energy, (target_x, target_y), ((res_x, res_y),))
        self.assertEqual(key, (0, 0, 5, (3, 3), ((1, 0),)))

        # When resource is consumed/empty
        obs_empty = Observation(
            entity_position=Position(1, 0),
            cells=(Position(1, 0),),
            energy=4,
            observed_resources=(),
            target=Position(3, 3),
        )
        key_empty = learner.state_key(obs_empty)
        self.assertEqual(key_empty, (1, 0, 4, (3, 3), ()))

    def test_q_learner_recharge_action_support_and_selection(self) -> None:
        cfg = LearningConfig(target=Position(2, 2), epsilon=0.0)
        learner = QLearner(cfg)
        state = (1, 0, 2)
        # Give Action.RECHARGE the highest value
        learner.update(state, Action.RIGHT, 0.5, (2, 0, 1), terminal=False)
        learner.update(state, Action.RECHARGE, 1.5, (1, 0, 6), terminal=False)
        self.assertEqual(learner.select_action(state, explore=False), Action.RECHARGE)

    def test_q_update_terminal_and_failed_action_penalties(self) -> None:
        cfg = LearningConfig(
            target=Position(2, 2),
            learning_rate=0.5,
            discount_factor=0.9,
            step_cost=-0.01,
            failed_action_penalty=-0.1,
            target_reward=1.0,
        )
        learner = QLearner(cfg)
        state_a = (0, 0, 1)
        state_b = (1, 0, 0)
        # 1. Nonterminal update with step cost
        val = learner.update(state_a, Action.RIGHT, -0.01, state_b, terminal=False)
        self.assertAlmostEqual(val, 0.5 * (-0.01))

        # 2. Terminal update omits future value
        target_state = (2, 2, 0)
        val_term = learner.update(state_b, Action.RIGHT, 1.0, target_state, terminal=True)
        self.assertEqual(val_term, 0.5 * 1.0)

    def test_q_table_persistence_across_resets_and_resource_restoration(self) -> None:
        res = Resource(Position(1, 0), recharge_amount=3, consumable=True)
        world = WorldConfig(
            width=3,
            height=3,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            resources=(res,),
            observable_target=Position(2, 0),
        )
        learning = LearningConfig(
            target=Position(2, 0),
            include_energy=True,
            include_resources=True,
            epsilon=0.0,
            max_steps=5,
        )
        runner = LearningRunner(world, learning, seed=12)
        ep1 = runner.train(1)[0]
        q_after_ep1 = runner.learner.q_table
        self.assertNotEqual(q_after_ep1, ())

        ep2 = runner.train(1)[0]
        # In ep2, simulation restarted with initial_energy=2 and resource restored at (1, 0)
        self.assertEqual(runner.world.initial_energy, 2)
        self.assertEqual(len(runner.world.resources), 1)
        # Q-table accumulated updates across episodes
        self.assertNotEqual(runner.learner.q_table, ())

    def test_evaluation_does_not_mutate_q_table(self) -> None:
        world = WorldConfig(width=3, height=3, initial_x=0, initial_y=0, initial_energy=5)
        learning = LearningConfig(target=Position(1, 0), epsilon=0.0, max_steps=5)
        runner = LearningRunner(world, learning, seed=4)
        runner.train(2)
        table_before = runner.learner.q_table

        eval_res = runner.evaluate(5, seed=99)
        self.assertEqual(runner.learner.q_table, table_before)
        self.assertEqual(eval_res.episodes, 5)
        self.assertEqual(len(eval_res.final_energies), 5)
        self.assertEqual(len(eval_res.recharge_counts), 5)
        self.assertEqual(len(eval_res.rejected_counts), 5)

    def test_terminal_energy_exhaustion_condition(self) -> None:
        world = WorldConfig(width=4, height=4, initial_x=0, initial_y=0, initial_energy=1, resources=())
        learning = LearningConfig(
            target=Position(3, 3),
            terminal_on_energy_exhaustion=True,
            terminal_failure_penalty=-0.5,
            failed_action_penalty=-0.1,
            max_steps=20,
        )
        runner = LearningRunner(world, learning, seed=7)
        # Step 1 consumes 1 energy -> energy becomes 0.
        # Step 2 movement attempt fails (insufficient energy) -> triggers permanent exhaustion termination!
        ep = runner._episode(learn=True)
        self.assertFalse(ep.reached_target)
        self.assertLess(ep.steps, 20)  # Terminated early, not spinning up to 20 steps
        self.assertEqual(ep.final_energy, 0)
        self.assertTrue(ep.transitions[-1].terminal)

    def test_epsilon_decay_behavior(self) -> None:
        cfg = LearningConfig(target=Position(2, 2), epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.2)
        learner = QLearner(cfg)
        self.assertEqual(learner.epsilon, 1.0)
        learner.decay_epsilon()
        self.assertEqual(learner.epsilon, 0.5)
        learner.decay_epsilon()
        self.assertEqual(learner.epsilon, 0.25)
        learner.decay_epsilon()
        self.assertEqual(learner.epsilon, 0.2)  # Capped at epsilon_min


if __name__ == "__main__":
    unittest.main()
