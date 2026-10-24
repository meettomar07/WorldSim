import unittest
from worldsim import Action, ExperienceMemory, LearningConfig, LearningRunner, Position, Simulation, WorldConfig


class MemoryTests(unittest.TestCase):
    def observation(self, x: int, y: int):
        return Simulation(WorldConfig(width=3, height=3, initial_x=x, initial_y=y)).observe()

    def test_record_retrieve_capacity_and_clear(self):
        memory = ExperienceMemory(2)
        a, b, c = self.observation(0, 0), self.observation(1, 0), self.observation(2, 0)
        memory.record(a, Action.RIGHT, 1, b); memory.record(a, Action.DOWN, 2, c); memory.record(b, Action.RIGHT, 3, c)
        self.assertEqual(len(memory.entries), 2)
        self.assertEqual(tuple(e.action for e in memory.retrieve(a)), (Action.DOWN,))
        self.assertEqual(memory.retrieve(c), ())
        memory.clear(); self.assertEqual(memory.entries, ())

    def test_memory_persists_training_resets_and_evaluation_does_not_mutate_it(self):
        world = WorldConfig(width=3, height=3, initial_x=1, initial_y=1)
        memory = ExperienceMemory(20)
        runner = LearningRunner(world, LearningConfig(target=Position(1, 0), epsilon=0, max_steps=3), seed=1, memory=memory)
        runner.train(2); count = len(memory.entries); table = runner.learner.q_table
        self.assertGreater(count, 0)
        runner.evaluate(2, seed=2)
        self.assertEqual(len(memory.entries), count); self.assertEqual(runner.learner.q_table, table)

    def test_memory_is_observation_only_and_separate_from_q_values(self):
        memory = ExperienceMemory(); before = memory.entries
        runner = LearningRunner(WorldConfig(width=3,height=3,initial_x=1,initial_y=1), LearningConfig(target=Position(1,0),epsilon=0,max_steps=2), memory=memory)
        runner.train(1)
        self.assertNotEqual(memory.entries, before)
        self.assertNotIn(Position(1,0), [e.observation.entity_position for e in memory.entries])


if __name__ == '__main__': unittest.main()
