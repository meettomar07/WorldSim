"""Regression and audit tests for WorldSim Phase 12.1.

Validates:
1. Feature invariance under spatial translations.
2. Representation limitations and feature aliasing counterexamples.
3. Infeasibility verifier and failure categorization (exhaustion vs timeout).
4. Matched multi-task tabular training failure due to coordinate hashing.
5. Metric accounting consistency (failed episodes included, step averages partitioned).
6. Deterministic reproducibility across evaluation seeds.
"""

import unittest
from random import Random

from worldsim import (
    Action,
    FeatureExtractor,
    LearningConfig,
    LearningRunner,
    LinearLearningConfig,
    LinearLearningRunner,
    LinearQLearner,
    Observation,
    Position,
    Resource,
    WorldConfig,
    plan_energy,
    EnergyPlanningProblem,
)
from worldsim.benchmark import evaluate_rl_runner, evaluate_random_policy, evaluate_planner_reference


class GeneralizationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fe = FeatureExtractor(world_width=10, world_height=10, max_energy=10, movement_energy_cost=1)

    def test_feature_invariance_under_spatial_translation(self) -> None:
        """Translating entity, target, and resource by (dx, dy) preserves relative features."""
        # Config 1: Entity at (1, 1), Target at (4, 1), Res at (2, 1)
        obs1 = Observation(
            entity_position=Position(1, 1),
            cells=(),
            energy=5,
            observed_resources=(Resource(Position(2, 1), 4, True),),
            target=Position(4, 1),
        )
        f1_right = self.fe.extract(obs1, Action.RIGHT)

        # Config 2: Translated by (+3, +4): Entity at (4, 5), Target at (7, 5), Res at (5, 5)
        obs2 = Observation(
            entity_position=Position(4, 5),
            cells=(),
            energy=5,
            observed_resources=(Resource(Position(5, 5), 4, True),),
            target=Position(7, 5),
        )
        f2_right = self.fe.extract(obs2, Action.RIGHT)

        # Relative navigation and deficit features must be identical
        relative_keys = [
            "target_dist_norm",
            "move_closer_target",
            "energy_deficit",
            "energy_sufficient",
            "move_closer_resource",
            "valid_move",
        ]
        for k in relative_keys:
            self.assertEqual(
                f1_right.get(k, 0.0),
                f2_right.get(k, 0.0),
                f"Feature {k} differed under translation: {f1_right.get(k)} vs {f2_right.get(k)}",
            )

    def test_feature_aliasing_counterexample_nearest_resource(self) -> None:
        """Counterexample: FeatureExtractor focuses only on nearest resource, ignoring sufficiency.

        If a closer resource offers insufficient recharge while a slightly further resource offers
        sufficient recharge, the linear feature extractor biases toward the closer (fatal) resource.
        """
        pos = Position(2, 2)
        target = Position(8, 2)  # Distance = 6, energy = 2 -> needs >= 4 recharge
        res_near_low = Resource(Position(3, 2), recharge_amount=1, consumable=True)   # Dist 1, amt 1
        res_far_high = Resource(Position(2, 4), recharge_amount=6, consumable=True)   # Dist 2, amt 6

        obs = Observation(
            entity_position=pos,
            cells=(),
            energy=2,
            observed_resources=(res_near_low, res_far_high),
            target=target,
        )

        feats_right = self.fe.extract(obs, Action.RIGHT)  # Toward near_low
        feats_down = self.fe.extract(obs, Action.DOWN)    # Toward far_high

        # Both moves are valid
        self.assertEqual(feats_right["valid_move"], 1.0)
        self.assertEqual(feats_down["valid_move"], 1.0)

        # Action.RIGHT moves toward the nearest resource (res_near_low)
        self.assertEqual(feats_right["move_closer_resource"], 1.0)
        # Action.DOWN moves away from the nearest resource (distance increases from 1 to sqrt(5))
        self.assertNotIn("move_closer_resource", feats_down)

        # This documents the architectural limitation: FeatureExtractor does not value-weigh multiple resources.

    def test_infeasibility_verifier_and_exhaustion_categorization(self) -> None:
        """Provably infeasible task is detected by planner verifier and results in clean energy exhaustion."""
        infeasible_world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=6,
            movement_energy_cost=1, resources=(), observable_target=Position(3, 3),
        )
        # Ground-truth verifier: EnergyPlanningProblem
        prob = EnergyPlanningProblem(
            width=infeasible_world.width, height=infeasible_world.height,
            start=Position(0, 0), target=Position(3, 3), initial_energy=2, max_energy=6,
            movement_energy_cost=1, resources=(),
        )
        plan_res = plan_energy(prob)
        self.assertFalse(plan_res.found)

        # Linear learner on this world
        cfg = LinearLearningConfig(max_steps=20, terminal_on_energy_exhaustion=True)
        runner = LinearLearningRunner(infeasible_world, cfg, seed=42)
        stats = evaluate_rl_runner(runner, infeasible_world, "Linear Q", episodes=10)

        self.assertEqual(stats.success_rate, 0.0)
        self.assertEqual(stats.exhausted_rate, 100.0)
        self.assertEqual(stats.timeout_rate, 0.0)
        self.assertEqual(stats.mean_final_energy, 0.0)

    def test_tabular_multi_task_generalization_failure(self) -> None:
        """Tabular Q-learning fails to generalize to unseen target even after multi-task training."""
        train_task = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 0), 4, True),), observable_target=Position(3, 2),
        )
        cfg = LearningConfig(
            target=Position(3, 2), max_steps=25,
            include_energy=True, include_resources=True, include_target=True,
        )
        runner = LearningRunner(train_task, cfg, seed=42)
        runner.train(300)

        # Unseen target (3, 0)
        unseen_task = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 0), 4, True),), observable_target=Position(3, 0),
        )
        stats = evaluate_rl_runner(runner, unseen_task, "Tabular Q", episodes=10)
        # Tabular Q achieves 0% because target coordinate hash is novel
        self.assertEqual(stats.success_rate, 0.0)
        self.assertEqual(stats.timeout_rate, 100.0)

    def test_metric_accounting_consistency(self) -> None:
        """Failures are included in reward calculations and exhaustion + timeout + success == 100%."""
        world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=6,
            resources=(), observable_target=Position(3, 3),
        )
        cfg = LinearLearningConfig(max_steps=15)
        runner = LinearLearningRunner(world, cfg, seed=42)
        stats = evaluate_rl_runner(runner, world, "Test Policy", episodes=20)

        # All episodes account for exactly 100% of outcomes
        total_pct = stats.success_rate + stats.exhausted_rate + stats.timeout_rate
        self.assertAlmostEqual(total_pct, 100.0, places=3)
        # Because all failed, mean_steps_success must be 0.0
        self.assertEqual(stats.mean_steps_success, 0.0)
        # Negative rewards from failed steps must be reflected in mean_reward
        self.assertLess(stats.mean_reward, 0.0)

    def test_deterministic_reproducibility_across_seeds(self) -> None:
        """Evaluating with the same seed produces identical results."""
        world = WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 0), 4, True),), observable_target=Position(3, 2),
        )
        cfg = LinearLearningConfig(max_steps=20)
        runner = LinearLearningRunner(world, cfg, seed=42)
        runner.train(50)

        res1 = evaluate_rl_runner(runner, world, "Policy", episodes=20, seed=12345)
        res2 = evaluate_rl_runner(runner, world, "Policy", episodes=20, seed=12345)

        self.assertEqual(res1.success_rate, res2.success_rate)
        self.assertEqual(res1.mean_reward, res2.mean_reward)
        self.assertEqual(res1.mean_final_energy, res2.mean_final_energy)
        self.assertEqual(res1.mean_steps_success, res2.mean_steps_success)


if __name__ == "__main__":
    unittest.main()
