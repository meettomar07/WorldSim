from dataclasses import FrozenInstanceError
import math
import unittest

from worldsim import Position, Simulation, WorldConfig


class PerceptionTests(unittest.TestCase):
    def make_simulation(self, x: int, y: int, radius: float, width: int = 5, height: int = 5) -> Simulation:
        return Simulation(
            WorldConfig(
                width=width,
                height=height,
                initial_x=x,
                initial_y=y,
                sensory_radius=radius,
            )
        )

    def test_observation_has_expected_cells_in_an_unobstructed_grid(self) -> None:
        observation = self.make_simulation(2, 2, 1.5).observe()
        expected = tuple(Position(x, y) for y in range(1, 4) for x in range(1, 4))
        self.assertEqual(observation.entity_position, Position(2, 2))
        self.assertEqual(observation.cells, expected)

    def test_euclidean_radius_includes_boundary_and_excludes_more_distant_cells(self) -> None:
        observation = self.make_simulation(3, 3, math.sqrt(2), width=7, height=7).observe()
        self.assertIn(Position(4, 4), observation.cells)  # Distance sqrt(2).
        self.assertNotIn(Position(5, 3), observation.cells)  # Distance 2.

    def test_observation_clips_at_each_world_boundary(self) -> None:
        expected_by_corner = {
            (0, 0): (Position(0, 0), Position(1, 0), Position(0, 1), Position(1, 1)),
            (3, 0): (Position(2, 0), Position(3, 0), Position(2, 1), Position(3, 1)),
            (0, 3): (Position(0, 2), Position(1, 2), Position(0, 3), Position(1, 3)),
            (3, 3): (Position(2, 2), Position(3, 2), Position(2, 3), Position(3, 3)),
        }
        for (x, y), expected in expected_by_corner.items():
            with self.subTest(position=(x, y)):
                observation = self.make_simulation(x, y, 1.5, width=4, height=4).observe()
                self.assertEqual(observation.cells, expected)

    def test_observation_never_contains_invalid_coordinates(self) -> None:
        simulation = self.make_simulation(0, 0, 20, width=3, height=2)
        self.assertTrue(all(simulation.state.contains(cell) for cell in simulation.observe().cells))

    def test_observations_are_deterministic_for_an_unchanged_state(self) -> None:
        simulation = self.make_simulation(2, 2, 2.0)
        self.assertEqual(simulation.observe(), simulation.observe())

    def test_observation_cannot_mutate_simulation_state(self) -> None:
        simulation = self.make_simulation(2, 2, 1.0)
        observation = simulation.observe()
        with self.assertRaises(FrozenInstanceError):
            observation.entity_position = Position(0, 0)  # type: ignore[misc]
        with self.assertRaises(TypeError):
            observation.cells[0] = Position(0, 0)  # type: ignore[index]
        self.assertEqual(simulation.state.entity.position, Position(2, 2))


if __name__ == "__main__":
    unittest.main()
