from dataclasses import FrozenInstanceError
import unittest

from worldsim import Action, Position, WorldConfig, run_experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorldConfig(width=4, height=4, initial_x=1, initial_y=1, sensory_radius=1.0)

    def test_known_sequence_reaches_expected_position_and_records_actions(self) -> None:
        result = run_experiment(self.config, (Action.RIGHT, Action.DOWN, Action.LEFT))

        self.assertTrue(result.completed)
        self.assertEqual(result.initial_position, Position(1, 1))
        self.assertEqual(result.final_position, Position(1, 2))
        self.assertEqual(result.final_tick, 3)
        self.assertEqual(tuple(record.action for record in result.experiences), result.actions)
        self.assertEqual(tuple(step.moved for step in result.results), (True, True, True))

    def test_identical_experiments_have_equivalent_results(self) -> None:
        actions = (Action.RIGHT, Action.DOWN, Action.LEFT)
        self.assertEqual(run_experiment(self.config, actions), run_experiment(self.config, actions))

    def test_boundary_rejection_uses_existing_step_semantics(self) -> None:
        config = WorldConfig(width=3, height=3, initial_x=0, initial_y=1, sensory_radius=1.0)
        result = run_experiment(config, (Action.LEFT, Action.DOWN))

        self.assertTrue(result.completed)
        self.assertEqual(tuple(step.moved for step in result.results), (False, True))
        self.assertEqual(tuple(record.moved for record in result.experiences), (False, True))
        self.assertEqual(result.experiences[0].tick_after, 0)
        self.assertEqual(result.final_position, Position(0, 2))
        self.assertEqual(result.final_tick, 1)

    def test_invalid_action_stops_execution_and_reports_error(self) -> None:
        result = run_experiment(self.config, (Action.RIGHT, "north", Action.DOWN))  # type: ignore[arg-type]

        self.assertFalse(result.completed)
        self.assertIn("Action enum", result.error or "")
        self.assertEqual(result.stopped_at_index, 1)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(tuple(record.action for record in result.experiences), (Action.RIGHT,))
        self.assertEqual(result.final_position, Position(2, 1))
        self.assertEqual(result.final_tick, 1)

    def test_result_and_snapshots_are_immutable(self) -> None:
        result = run_experiment(self.config, (Action.RIGHT, Action.DOWN))

        with self.assertRaises(FrozenInstanceError):
            result.final_tick = 99  # type: ignore[misc]
        with self.assertRaises(TypeError):
            result.results[0] = result.results[1]  # type: ignore[index]
        with self.assertRaises(AttributeError):
            result.experiences.append(result.experiences[0])  # type: ignore[attr-defined]
        with self.assertRaises(FrozenInstanceError):
            result.experiences[0].position_after = Position(0, 0)  # type: ignore[misc]
        self.assertEqual(result.final_position, Position(2, 2))
        self.assertEqual(result.experiences[-1].position_after, Position(2, 2))


if __name__ == "__main__":
    unittest.main()
