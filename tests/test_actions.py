import unittest

from worldsim import Action, Position, Simulation, WorldConfig


class ActionTests(unittest.TestCase):
    def make_simulation(self, x: int = 2, y: int = 2, width: int = 5, height: int = 5) -> Simulation:
        return Simulation(WorldConfig(width=width, height=height, initial_x=x, initial_y=y, sensory_radius=1.0))

    def test_each_direction_moves_one_cell_and_advances_once(self) -> None:
        expected_positions = {
            Action.UP: Position(2, 1),
            Action.DOWN: Position(2, 3),
            Action.LEFT: Position(1, 2),
            Action.RIGHT: Position(3, 2),
        }
        for action, expected_position in expected_positions.items():
            with self.subTest(action=action):
                result = self.make_simulation().step(action)
                self.assertTrue(result.moved)
                self.assertEqual(result.state.entity.position, expected_position)
                self.assertEqual(result.state.tick, 1)

    def test_each_boundary_rejects_the_outward_action_without_changing_state(self) -> None:
        cases = (
            (0, 2, Action.LEFT),
            (4, 2, Action.RIGHT),
            (2, 0, Action.UP),
            (2, 4, Action.DOWN),
        )
        for x, y, action in cases:
            with self.subTest(position=(x, y), action=action):
                simulation = self.make_simulation(x, y)
                before = simulation.state
                result = simulation.step(action)
                self.assertFalse(result.moved)
                self.assertEqual(result.state, before)
                self.assertEqual(simulation.state, before)

    def test_unknown_action_is_rejected_without_mutating_state(self) -> None:
        simulation = self.make_simulation()
        before = simulation.state
        with self.assertRaisesRegex(ValueError, "Action enum"):
            simulation.step("north")  # type: ignore[arg-type]
        self.assertEqual(simulation.state, before)

    def test_observation_uses_the_new_position_after_a_successful_action(self) -> None:
        result = self.make_simulation().step(Action.RIGHT)
        self.assertEqual(result.observation.entity_position, Position(3, 2))
        self.assertIn(Position(4, 2), result.observation.cells)

    def test_paused_simulation_rejects_actions_without_advancing_tick(self) -> None:
        simulation = self.make_simulation()
        simulation.pause()
        result = simulation.step(Action.UP)
        self.assertFalse(result.moved)
        self.assertEqual(result.state.entity.position, Position(2, 2))
        self.assertEqual(result.state.tick, 0)


if __name__ == "__main__":
    unittest.main()
