import unittest

from worldsim import Action, PlanningProblem, Position, WorldConfig, execute_plan, plan
from worldsim.planning import PlanResult


class PlanningTests(unittest.TestCase):
    def test_start_equals_target(self) -> None:
        problem = PlanningProblem(3, 3, Position(1, 1), Position(1, 1))
        result = plan(problem)
        self.assertTrue(result.found)
        self.assertEqual(result.actions, ())
        steps = execute_plan(WorldConfig(width=3, height=3, initial_x=1, initial_y=1), result)
        self.assertEqual(steps, ())

    def test_shortest_path_optimality(self) -> None:
        cases = [
            (Position(0, 0), Position(1, 0), 1),
            (Position(0, 0), Position(2, 1), 3),
            (Position(0, 0), Position(3, 3), 6),
            (Position(3, 2), Position(0, 1), 4),
        ]
        for start, target, expected_distance in cases:
            with self.subTest(start=start, target=target):
                problem = PlanningProblem(4, 4, start, target)
                result = plan(problem)
                self.assertTrue(result.found)
                self.assertEqual(len(result.actions), expected_distance)

                steps = execute_plan(
                    WorldConfig(width=4, height=4, initial_x=start.x, initial_y=start.y),
                    result,
                )
                self.assertEqual(len(steps), expected_distance)
                self.assertTrue(all(step.moved for step in steps))
                self.assertEqual(steps[-1].state.entity.position, target)

    def test_invalid_dimensions_and_out_of_bounds_inputs(self) -> None:
        # Invalid dimensions
        with self.assertRaises(ValueError):
            PlanningProblem(0, 3, Position(0, 0), Position(1, 1))
        with self.assertRaises(ValueError):
            PlanningProblem(3, -1, Position(0, 0), Position(1, 1))

        # Start out of bounds
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(-1, 0), Position(1, 1))
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(3, 0), Position(1, 1))
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(0, 3), Position(1, 1))

        # Target out of bounds
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(0, 0), Position(-1, 1))
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(0, 0), Position(3, 1))
        with self.assertRaises(ValueError):
            PlanningProblem(3, 3, Position(0, 0), Position(1, 3))

    def test_unreachable_and_empty_plan_execution(self) -> None:
        unfound = PlanResult(found=False, actions=())
        steps = execute_plan(WorldConfig(width=3, height=3, initial_x=0, initial_y=0), unfound)
        self.assertEqual(steps, ())

    def test_plan_execution_consistency_with_simulation_start(self) -> None:
        problem = PlanningProblem(3, 3, Position(0, 0), Position(2, 0))
        result = plan(problem)
        self.assertEqual(result.actions, (Action.RIGHT, Action.RIGHT))

        # Consistent start: reaches destination
        consistent_steps = execute_plan(
            WorldConfig(width=3, height=3, initial_x=0, initial_y=0),
            result,
        )
        self.assertEqual(len(consistent_steps), 2)
        self.assertTrue(all(s.moved for s in consistent_steps))
        self.assertEqual(consistent_steps[-1].state.entity.position, Position(2, 0))

        # Inconsistent start: initial_x=2 is already at right boundary
        inconsistent_steps = execute_plan(
            WorldConfig(width=3, height=3, initial_x=2, initial_y=0),
            result,
        )
        self.assertEqual(len(inconsistent_steps), 1)
        self.assertFalse(inconsistent_steps[0].moved)
        self.assertEqual(inconsistent_steps[0].state.entity.position, Position(2, 0))

    def test_plan_execution_uses_simulation_api(self) -> None:
        problem = PlanningProblem(3, 3, Position(0, 0), Position(1, 1))
        result = plan(problem)
        steps = execute_plan(WorldConfig(width=3, height=3, initial_x=0, initial_y=0), result)
        self.assertEqual(len(steps), 2)
        for step in steps:
            self.assertIsNotNone(step.state)
            self.assertIsNotNone(step.observation)
            self.assertTrue(step.moved)


if __name__ == "__main__":
    unittest.main()

