"""Tests for Phase 12: Feature-based linear function-approximation reinforcement learning."""

import unittest
from random import Random

from worldsim import (
    Action,
    Entity,
    FeatureExtractor,
    LinearLearningConfig,
    LinearLearningRunner,
    LinearQLearner,
    Observation,
    Position,
    Resource,
    Simulation,
    WorldConfig,
    WorldState,
)


class LinearLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fe = FeatureExtractor(world_width=5, world_height=5, max_energy=10, movement_energy_cost=1)
        self.config = LinearLearningConfig(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.2,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            step_cost=-0.01,
            target_reward=1.0,
            failed_action_penalty=-0.05,
            max_steps=20,
        )

    def test_feature_extraction_and_normalization(self) -> None:
        pos = Position(1, 1)
        target = Position(3, 1)
        res = Resource(Position(1, 2), recharge_amount=4, consumable=True)
        obs = Observation(
            entity_position=pos,
            cells=(pos, target, res.position),
            energy=5,
            observed_resources=(res,),
            target=target,
        )

        # Candidate Action.RIGHT moves toward target
        feats_right = self.fe.extract(obs, Action.RIGHT)
        self.assertEqual(feats_right["action_right"], 1.0)
        self.assertEqual(feats_right["valid_move"], 1.0)
        self.assertEqual(feats_right["move_closer_target"], 1.0)
        self.assertNotIn("move_away_target", feats_right)
        self.assertAlmostEqual(feats_right["energy_norm"], 0.5)

        # Candidate Action.UP moves away from target and resource
        feats_up = self.fe.extract(obs, Action.UP)
        self.assertEqual(feats_up["valid_move"], 1.0)
        self.assertEqual(feats_up["move_away_target"], 1.0)

        # Candidate Action.RECHARGE when not on resource
        feats_rech = self.fe.extract(obs, Action.RECHARGE)
        self.assertEqual(feats_rech["recharge_no_resource"], 1.0)
        self.assertEqual(feats_rech["invalid_action"], 1.0)

        # Boundary checks: moving LEFT from x=0
        obs_edge = Observation(
            entity_position=Position(0, 0),
            cells=(Position(0, 0),),
            energy=5,
            observed_resources=(),
            target=target,
        )
        feats_left = self.fe.extract(obs_edge, Action.LEFT)
        self.assertEqual(feats_left["move_into_wall"], 1.0)
        self.assertEqual(feats_left["invalid_action"], 1.0)

    def test_recharge_feature_on_resource_cell(self) -> None:
        pos = Position(2, 2)
        res = Resource(Position(2, 2), recharge_amount=5, consumable=True)
        obs = Observation(
            entity_position=pos,
            cells=(pos,),
            energy=3,
            observed_resources=(res,),
            target=Position(4, 4),
        )
        feats = self.fe.extract(obs, Action.RECHARGE)
        self.assertEqual(feats["valid_recharge"], 1.0)
        self.assertEqual(feats["resource_consumable"], 1.0)
        # Gain is min(5, 10 - 3) / 10 = 5 / 10 = 0.5
        self.assertAlmostEqual(feats["recharge_gain"], 0.5)
        self.assertEqual(feats["recharge_when_needed"], 1.0)

    def test_action_selection_and_masking(self) -> None:
        learner = LinearQLearner(self.config, self.fe, seed=42)
        learner.weights = {"action_up": 2.0, "action_down": 1.0}

        # Observation at top boundary (y=0): UP is illegal
        obs_top = Observation(
            entity_position=Position(2, 0),
            cells=(),
            energy=5,
            observed_resources=(),
        )
        legal = learner.get_legal_actions(obs_top)
        self.assertNotIn(Action.UP, legal)
        self.assertIn(Action.DOWN, legal)

        # Greedy choice should mask Action.UP and pick Action.DOWN
        best = learner.select_action(obs_top, explore=False)
        self.assertEqual(best, Action.DOWN)

    def test_q_value_update_and_weights_modification(self) -> None:
        learner = LinearQLearner(self.config, self.fe, seed=10)
        obs = Observation(
            entity_position=Position(1, 1),
            cells=(),
            energy=5,
            observed_resources=(),
            target=Position(2, 1),
        )
        next_obs = Observation(
            entity_position=Position(2, 1),
            cells=(),
            energy=4,
            observed_resources=(),
            target=Position(2, 1),
        )

        initial_weights = learner.copy_weights()
        self.assertEqual(len(initial_weights), 0)

        # Successful step reaching target
        updated_q = learner.update(obs, Action.RIGHT, reward=1.0, next_observation=next_obs, terminal=True)
        self.assertGreater(updated_q, 0.0)
        self.assertGreater(learner.weights["reaches_target"], 0.0)

    def test_training_and_evaluation_separation(self) -> None:
        world = WorldConfig(
            width=5,
            height=5,
            initial_x=0,
            initial_y=0,
            initial_energy=5,
            max_energy=10,
            observable_target=Position(2, 2),
        )
        runner = LinearLearningRunner(world, self.config, seed=99)
        runner.train(5)
        weights_after_train = runner.learner.copy_weights()
        self.assertTrue(bool(weights_after_train))

        # Evaluate should not mutate weights
        runner.evaluate(10, seed=123, policy="learned")
        weights_after_eval = runner.learner.copy_weights()
        self.assertEqual(weights_after_train, weights_after_eval)

    def test_epsilon_decay_and_determinism(self) -> None:
        cfg = LinearLearningConfig(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1)
        learner1 = LinearQLearner(cfg, self.fe, seed=7)
        learner2 = LinearQLearner(cfg, self.fe, seed=7)

        self.assertEqual(learner1.epsilon, 0.5)
        learner1.decay_epsilon()
        self.assertAlmostEqual(learner1.epsilon, 0.45)

        obs = Observation(Position(1, 1), cells=(), energy=5, target=Position(3, 3))
        # Deterministic exploration with same seed
        actions1 = [learner1.select_action(obs, explore=True) for _ in range(5)]
        learner2.decay_epsilon()
        actions2 = [learner2.select_action(obs, explore=True) for _ in range(5)]
        self.assertEqual(actions1, actions2)

    def test_generalization_across_unseen_target_and_start(self) -> None:
        # Train on target (2, 2) from start (0, 0)
        train_world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=6,
            max_energy=6,
            movement_energy_cost=1,
            observable_target=Position(2, 2),
        )
        cfg = LinearLearningConfig(learning_rate=0.08, epsilon=0.3, epsilon_decay=0.995, max_steps=20)
        runner = LinearLearningRunner(train_world, cfg, seed=42)
        runner.train(150)

        # Test on UNSEEN target (1, 3) from UNSEEN start (3, 0) without retraining
        test_world = WorldConfig(
            width=4,
            height=4,
            initial_x=3,
            initial_y=0,
            initial_energy=6,
            max_energy=6,
            movement_energy_cost=1,
            observable_target=Position(1, 3),
        )
        eval_res = runner.evaluate(20, world=test_world, seed=100)
        # Should achieve 100% success on unseen target and start
        self.assertEqual(eval_res.success_rate, 1.0)
        for steps in eval_res.steps_to_target:
            self.assertIsNotNone(steps)
            # Optimal Manhattan distance from (3, 0) to (1, 3) is |1-3| + |3-0| = 5
            self.assertEqual(steps, 5)

    def test_generalization_with_energy_and_resource_recharge(self) -> None:
        # Train on a setup with resource recharge
        res = Resource(Position(2, 0), recharge_amount=4, consumable=True)
        train_world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=6,
            movement_energy_cost=1,
            resources=(res,),
            observable_target=Position(3, 2),
        )
        cfg = LinearLearningConfig(
            learning_rate=0.08,
            discount_factor=0.95,
            epsilon=0.4,
            epsilon_decay=0.995,
            max_steps=25,
        )
        runner = LinearLearningRunner(train_world, cfg, seed=42)
        runner.train(300)

        # Held-out target (1, 3) requiring recharge at (2, 0)
        held_out_world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=3,
            max_energy=6,
            movement_energy_cost=1,
            resources=(res,),
            observable_target=Position(1, 3),
        )
        eval_res = runner.evaluate(20, world=held_out_world, seed=99)
        self.assertEqual(eval_res.success_rate, 1.0)
        # All successes should have performed exactly 1 recharge
        self.assertTrue(all(c == 1 for c in eval_res.recharge_counts))
        # 0 rejected actions
        self.assertTrue(all(r == 0 for r in eval_res.rejected_counts))

    def test_unsolvable_energy_exhaustion_terminates(self) -> None:
        # Distance is 6, energy is only 2, no resources -> strictly unsolvable
        unsolv_world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            max_energy=6,
            movement_energy_cost=1,
            resources=(),
            observable_target=Position(3, 3),
        )
        cfg = LinearLearningConfig(max_steps=20, terminal_on_energy_exhaustion=True)
        runner = LinearLearningRunner(unsolv_world, cfg, seed=42)
        eval_res = runner.evaluate(10, seed=5)
        self.assertEqual(eval_res.success_rate, 0.0)
        # Early termination should stop around step 2-3 when energy is exhausted, not run to max_steps=20
        self.assertTrue(all(e == 0 for e in eval_res.final_energies))


    def test_consumable_vs_reusable_resource_features(self) -> None:
        pos = Position(1, 1)
        r_consumable = Resource(Position(1, 1), recharge_amount=3, consumable=True)
        r_reusable = Resource(Position(1, 1), recharge_amount=3, consumable=False)

        obs_c = Observation(entity_position=pos, cells=(pos,), energy=5, observed_resources=(r_consumable,))
        feats_c = self.fe.extract(obs_c, Action.RECHARGE)
        self.assertEqual(feats_c["resource_consumable"], 1.0)

        obs_r = Observation(entity_position=pos, cells=(pos,), energy=5, observed_resources=(r_reusable,))
        feats_r = self.fe.extract(obs_r, Action.RECHARGE)
        self.assertNotIn("resource_consumable", feats_r)

    def test_observation_limitations_when_target_not_observable(self) -> None:
        pos = Position(0, 0)
        obs_no_target = Observation(
            entity_position=pos,
            cells=(pos,),
            energy=5,
            observed_resources=(),
            target=None,
        )
        feats = self.fe.extract(obs_no_target, Action.RIGHT)
        self.assertNotIn("move_closer_target", feats)
        self.assertNotIn("move_away_target", feats)
        self.assertNotIn("reaches_target", feats)
        self.assertNotIn("target_dist_norm", feats)
        self.assertNotIn("energy_deficit", feats)
        self.assertNotIn("energy_sufficient", feats)

    def test_multi_task_generalization_across_unseen_resource_layouts(self) -> None:
        env1 = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            movement_energy_cost=1, sensory_radius=3.0,
            resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
            observable_target=Position(3, 2),
        )
        env2 = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            movement_energy_cost=1, sensory_radius=3.0,
            resources=(Resource(Position(0, 2), recharge_amount=4, consumable=True),),
            observable_target=Position(3, 2),
        )
        env3 = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            movement_energy_cost=1, sensory_radius=3.0,
            resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
            observable_target=Position(1, 3),
        )
        cfg = LinearLearningConfig(
            learning_rate=0.08, discount_factor=0.95, epsilon=0.4, epsilon_decay=0.995, max_steps=25,
        )
        runner = LinearLearningRunner(env1, cfg, seed=123)
        runner.train_tasks([env1, env2, env3], episodes_per_task=200)

        # Held-out resource layout: Resource at (1, 1), target (3, 2)
        env_held_out = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            movement_energy_cost=1, sensory_radius=3.0,
            resources=(Resource(Position(1, 1), recharge_amount=4, consumable=True),),
            observable_target=Position(3, 2),
        )
        eval_res = runner.evaluate(20, world=env_held_out, seed=42)
        self.assertEqual(eval_res.success_rate, 1.0)
        self.assertEqual(eval_res.steps_to_target[0], 6)
        self.assertEqual(eval_res.recharge_counts[0], 1)


if __name__ == "__main__":
    unittest.main()
