from dataclasses import FrozenInstanceError
import unittest

from worldsim import Action, Position, Simulation, WorldConfig


class ExperienceTests(unittest.TestCase):
    def make_simulation(self, x: int = 1, y: int = 1) -> Simulation:
        return Simulation(WorldConfig(width=3, height=3, initial_x=x, initial_y=y, sensory_radius=1.0))

    def test_successful_action_records_complete_before_and_after_state(self) -> None:
        simulation = self.make_simulation()
        observation_before = simulation.observe()
        result = simulation.step(Action.RIGHT)
        experience = simulation.experiences[0]

        self.assertEqual(experience.tick_before, 0)
        self.assertEqual(experience.tick_after, 1)
        self.assertEqual(experience.action, Action.RIGHT)
        self.assertTrue(experience.moved)
        self.assertEqual(experience.position_before, Position(1, 1))
        self.assertEqual(experience.position_after, Position(2, 1))
        self.assertEqual(experience.observation_before, observation_before)
        self.assertEqual(experience.observation_after, result.observation)
        self.assertEqual(experience.observation_after.entity_position, Position(2, 1))

    def test_boundary_rejection_is_recorded_without_state_change(self) -> None:
        simulation = self.make_simulation(x=0, y=1)
        result = simulation.step(Action.LEFT)
        experience = simulation.experiences[0]

        self.assertFalse(result.moved)
        self.assertFalse(experience.moved)
        self.assertEqual(experience.position_before, Position(0, 1))
        self.assertEqual(experience.position_after, Position(0, 1))
        self.assertEqual((experience.tick_before, experience.tick_after), (0, 0))

    def test_paused_rejection_is_recorded_without_state_change(self) -> None:
        simulation = self.make_simulation()
        simulation.pause()
        result = simulation.step(Action.UP)
        experience = simulation.experiences[0]

        self.assertFalse(result.moved)
        self.assertFalse(experience.moved)
        self.assertEqual(experience.position_before, experience.position_after)
        self.assertEqual((experience.tick_before, experience.tick_after), (0, 0))

    def test_unknown_action_does_not_create_an_experience(self) -> None:
        simulation = self.make_simulation()
        with self.assertRaisesRegex(ValueError, "Action enum"):
            simulation.step("north")  # type: ignore[arg-type]
        self.assertEqual(simulation.experiences, ())

    def test_experiences_are_in_chronological_order(self) -> None:
        simulation = self.make_simulation()
        simulation.step(Action.RIGHT)
        simulation.step(Action.DOWN)
        history = simulation.experiences

        self.assertEqual(tuple(record.action for record in history), (Action.RIGHT, Action.DOWN))
        self.assertEqual(tuple(record.tick_before for record in history), (0, 1))
        self.assertEqual(tuple(record.tick_after for record in history), (1, 2))

    def test_public_history_and_nested_records_are_immutable(self) -> None:
        simulation = self.make_simulation()
        simulation.step(Action.RIGHT)
        simulation.step(Action.DOWN)
        history = simulation.experiences
        record = history[0]
        later_record = history[1]

        with self.assertRaises(AttributeError):
            history.append(record)  # type: ignore[attr-defined]
        with self.assertRaises(FrozenInstanceError):
            record.moved = False  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            record.observation_after.entity_position = Position(0, 0)  # type: ignore[misc]
        with self.assertRaises(TypeError):
            record.observation_after.cells[0] = Position(0, 0)  # type: ignore[index]
        self.assertEqual(simulation.state.entity.position, Position(2, 2))
        self.assertEqual(simulation.experiences[0], record)
        self.assertEqual(simulation.experiences[1], later_record)


if __name__ == "__main__":
    unittest.main()
