import unittest

from worldsim import Action, LearningConfig, LearningRunner, Position, QLearner, WorldConfig


class LearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LearningConfig(target=Position(2, 2), learning_rate=0.5, discount_factor=0.9, epsilon=0.0, max_steps=10)

    def test_q_update_uses_future_value_for_nonterminal_transition(self) -> None:
        learner = QLearner(self.config, seed=1)
        learner.update((1, 1), Action.RIGHT, 0.0, (2, 1), terminal=True)
        learner.update((2, 1), Action.UP, 4.0, (2, 1), terminal=True)
        updated = learner.update((1, 1), Action.RIGHT, 1.0, (2, 1), terminal=False)
        self.assertEqual(updated, 0.5 * (1.0 + 0.9 * 2.0))

    def test_terminal_update_omits_future_bootstrap(self) -> None:
        learner = QLearner(self.config)
        learner.update((2, 2), Action.UP, 100.0, (2, 2), terminal=True)
        self.assertEqual(learner.update((1, 1), Action.RIGHT, 2.0, (2, 2), terminal=True), 1.0)

    def test_epsilon_greedy_exploitation_and_exploration(self) -> None:
        learner = QLearner(self.config, seed=3)
        learner.update((1, 1), Action.RIGHT, 3.0, (1, 1), terminal=True)
        self.assertEqual(learner.select_action((1, 1)), Action.RIGHT)
        explorer = QLearner(LearningConfig(target=Position(2, 2), epsilon=1.0), seed=3)
        self.assertGreater(len({explorer.select_action((1, 1)) for _ in range(12)}), 1)

    def test_training_is_reproducible_and_retains_q_table_between_calls(self) -> None:
        world = WorldConfig(width=3, height=3, initial_x=0, initial_y=0)
        config = LearningConfig(target=Position(2, 2), epsilon=0.3, max_steps=12)
        first, second = LearningRunner(world, config, seed=7), LearningRunner(world, config, seed=7)
        self.assertEqual(first.train(20), second.train(20))
        before = first.learner.q_table
        first.train(1)
        self.assertNotEqual(first.learner.q_table, ())
        self.assertNotEqual(before, ())

    def test_reward_termination_and_evaluation_without_learning_mutation(self) -> None:
        world = WorldConfig(width=3, height=3, initial_x=1, initial_y=1)
        runner = LearningRunner(world, LearningConfig(target=Position(1, 0), epsilon=0.0, max_steps=4), seed=2)
        episode = runner.train(1)[0]
        self.assertTrue(episode.reached_target)
        self.assertEqual(episode.transitions[-1].reward, 1.0)
        table = runner.learner.q_table
        runner.evaluate(3, seed=9)
        self.assertEqual(runner.learner.q_table, table)


if __name__ == "__main__":
    unittest.main()
