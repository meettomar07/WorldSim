import unittest

from worldsim import Position, Simulation, WorldConfig


class WorldConfigTests(unittest.TestCase):
    def test_defaults_create_a_valid_world(self) -> None:
        config = WorldConfig()
        self.assertEqual((config.width, config.height), (20, 12))

    def test_rejects_out_of_bounds_initial_position(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(width=2, height=2, initial_x=2, initial_y=0)


class SimulationTests(unittest.TestCase):
    def test_initialization_creates_one_entity_at_configured_coordinates(self) -> None:
        simulation = Simulation(WorldConfig(width=7, height=5, entity_id="alpha", initial_x=6, initial_y=4))
        state = simulation.state
        self.assertEqual(state.tick, 0)
        self.assertEqual(state.entity.id, "alpha")
        self.assertEqual(state.entity.position, Position(6, 4))
        self.assertTrue(state.contains(state.entity.position))

    def test_tick_progression_advances_the_clock_without_moving_entity(self) -> None:
        simulation = Simulation(WorldConfig())
        initial_position = simulation.state.entity.position
        simulation.tick()
        state = simulation.tick()
        self.assertEqual(state.tick, 2)
        self.assertEqual(state.entity.position, initial_position)

    def test_pause_prevents_progress_and_resume_restarts_it(self) -> None:
        simulation = Simulation(WorldConfig())
        simulation.pause()
        self.assertTrue(simulation.paused)
        self.assertEqual(simulation.tick().tick, 0)
        simulation.resume()
        self.assertEqual(simulation.tick().tick, 1)

    def test_identical_configs_produce_identical_state_sequences(self) -> None:
        config = WorldConfig(width=4, height=3, entity_id="repro", initial_x=1, initial_y=2)
        first, second = Simulation(config), Simulation(config)
        first_states = [first.tick() for _ in range(5)]
        second_states = [second.tick() for _ in range(5)]
        self.assertEqual(first_states, second_states)


if __name__ == "__main__":
    unittest.main()
