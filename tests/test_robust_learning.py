"""Unit and integration tests for Phase 12.2: Robust Resource-Aware Learning."""

import unittest
from random import Random

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.engine import Simulation
from worldsim.linear_learning import FeatureExtractor, LinearLearningConfig, LinearQLearner
from worldsim.models import Observation, Position, Resource
from worldsim.robust_policy import (
    AgentMemory,
    CycleDetector,
    ExplorationController,
    RechargeEvaluator,
    ResourceUtilityEvaluator,
    RobustResourcePolicy,
)


class RobustLearningTests(unittest.TestCase):
    """Tests for Phase 12.2 components and integrated robust policy."""

    def setUp(self) -> None:
        self.max_energy = 8
        self.cost = 1
        self.recharge_eval = RechargeEvaluator(max_energy=self.max_energy, movement_energy_cost=self.cost)
        self.utility_eval = ResourceUtilityEvaluator(max_energy=self.max_energy, movement_energy_cost=self.cost)

    # --------------------------------------------------------------------------
    # Stage 1: Intelligent Recharge Control Tests
    # --------------------------------------------------------------------------

    def test_recharge_at_full_energy_is_rejected(self) -> None:
        """When energy is already at max_energy, recharge assessment returns is_beneficial=False."""
        mem = AgentMemory(world_width=5, world_height=5)
        res = Resource(Position(1, 1), recharge_amount=4, consumable=False)
        obs = Observation(
            entity_position=Position(1, 1),
            cells=(Position(1, 1),),
            energy=self.max_energy,
            observed_resources=(res,),
            target=Position(4, 4),
        )
        mem.update(obs)
        assessment = self.recharge_eval.evaluate(obs, mem)
        self.assertFalse(assessment.is_beneficial)
        self.assertEqual(assessment.potential_gain, 0)
        self.assertEqual(assessment.reason, "energy_already_at_maximum")

    def test_recharge_when_energy_already_sufficient_for_target(self) -> None:
        """If current energy is sufficient to reach target directly, recharge is not beneficial."""
        mem = AgentMemory(world_width=5, world_height=5)
        res = Resource(Position(1, 1), recharge_amount=4, consumable=False)
        # Position (1,1), Target (2,2) -> distance = 2. Energy = 5 >= 2.
        obs = Observation(
            entity_position=Position(1, 1),
            cells=(Position(1, 1),),
            energy=5,
            observed_resources=(res,),
            target=Position(2, 2),
        )
        mem.update(obs)
        assessment = self.recharge_eval.evaluate(obs, mem)
        self.assertFalse(assessment.is_beneficial)
        self.assertEqual(assessment.reason, "energy_already_sufficient_for_target")

    def test_recharge_when_energy_insufficient_for_target(self) -> None:
        """If current energy is less than required for target, recharge is beneficial."""
        mem = AgentMemory(world_width=5, world_height=5)
        res = Resource(Position(1, 1), recharge_amount=4, consumable=False)
        # Position (1,1), Target (4,4) -> distance = 6. Energy = 2 < 6.
        obs = Observation(
            entity_position=Position(1, 1),
            cells=(Position(1, 1),),
            energy=2,
            observed_resources=(res,),
            target=Position(4, 4),
        )
        mem.update(obs)
        assessment = self.recharge_eval.evaluate(obs, mem)
        self.assertTrue(assessment.is_beneficial)
        self.assertEqual(assessment.potential_gain, 4)
        self.assertEqual(assessment.reason, "beneficial_recharge")

    def test_reusable_repeated_recharge_prevention(self) -> None:
        """On a reusable resource, after recharging to sufficient energy, subsequent recharge is prevented."""
        mem = AgentMemory(world_width=5, world_height=5)
        res = Resource(Position(1, 1), recharge_amount=4, consumable=False)
        # First step: arrive with energy 1, target at (3,3) (distance 4)
        obs1 = Observation(Position(1, 1), (Position(1, 1),), 1, (res,), Position(3, 3))
        mem.update(obs1)
        self.assertTrue(self.recharge_eval.evaluate(obs1, mem).is_beneficial)

        # Agent executes RECHARGE: energy becomes 5
        mem.record_action_result(Position(1, 1), 1, Action.RECHARGE, success=True)
        obs2 = Observation(Position(1, 1), (Position(1, 1),), 5, (res,), Position(3, 3))
        mem.update(obs2)

        # Now energy is 5 >= 4 (target dist). Assessment must be False!
        assessment = self.recharge_eval.evaluate(obs2, mem)
        self.assertFalse(assessment.is_beneficial)

    def test_consumable_resource_marked_depleted(self) -> None:
        """When a consumable resource is recharged, it is recorded as depleted in agent memory."""
        mem = AgentMemory(world_width=5, world_height=5)
        res = Resource(Position(1, 1), recharge_amount=3, consumable=True)
        obs = Observation(Position(1, 1), (Position(1, 1),), 2, (res,), Position(3, 3))
        mem.update(obs)
        self.assertEqual(len(mem.available_resources()), 1)

        mem.record_action_result(Position(1, 1), 2, Action.RECHARGE, success=True)
        self.assertTrue(mem.is_depleted(Position(1, 1)))
        self.assertEqual(len(mem.available_resources()), 0)

    # --------------------------------------------------------------------------
    # Stage 2: Utility-Based Resource Selection Tests
    # --------------------------------------------------------------------------

    def test_nearest_insufficient_vs_farther_sufficient_counterexample(self) -> None:
        """Utility evaluator correctly selects the farther sufficient resource over the nearest insufficient one."""
        pos = Position(0, 0)
        target = Position(3, 3)  # distance = 6, cost = 6
        energy = 2  # deficit = 4
        # Nearest resource: dist = 1, recharge = 1 (leaves agent stranded)
        res_near = Resource(Position(1, 0), recharge_amount=1, consumable=True)
        # Farther resource: dist = 2, recharge = 6 (provides sufficient energy)
        res_far = Resource(Position(2, 0), recharge_amount=6, consumable=True)

        mem = AgentMemory(world_width=4, world_height=4)
        obs = Observation(pos, (pos,), energy, (res_near, res_far), target)
        mem.update(obs)

        comparison = self.utility_eval.compare_routes(obs, mem)
        self.assertFalse(comparison.direct_feasible)
        self.assertEqual(comparison.decision, "detour_to_resource")
        self.assertIsNotNone(comparison.best_resource_utility)
        # Recommended target MUST be the farther resource at (2,0), NOT (1,0)
        self.assertEqual(comparison.recommended_target, Position(2, 0))
        self.assertEqual(comparison.best_resource_utility.resource.position, Position(2, 0))
        self.assertTrue(comparison.best_resource_utility.is_feasible_route)

    def test_detour_more_expensive_than_benefit_rejected(self) -> None:
        """A resource whose travel cost exceeds its recharge gain is assessed as having negative net margin."""
        pos = Position(0, 0)
        target = Position(4, 0)
        energy = 4
        # Resource at (0, 3): dist = 3, cost = 3. Recharge amount = 2.
        res = Resource(Position(0, 3), recharge_amount=2, consumable=True)
        mem = AgentMemory(world_width=5, world_height=5)
        obs = Observation(pos, (pos,), energy, (res,), target)
        mem.update(obs)

        util = self.utility_eval.evaluate_resource(res, pos, energy, target)
        self.assertFalse(util.is_feasible_route)
        self.assertLess(util.utility_score, 0)

    def test_unreachable_resource_identified(self) -> None:
        """When agent has insufficient energy to reach a resource, it is marked unreachable."""
        pos = Position(0, 0)
        target = Position(4, 4)
        energy = 1
        res = Resource(Position(3, 0), recharge_amount=5, consumable=True)  # distance = 3 > energy 1
        util = self.utility_eval.evaluate_resource(res, pos, energy, target)
        self.assertFalse(util.reachable_now)
        self.assertEqual(util.status, "unreachable")

    # --------------------------------------------------------------------------
    # Stage 3: Safe Exploration Tests
    # --------------------------------------------------------------------------

    def test_safe_exploration_scores_unobserved_cells_and_penalizes_visits(self) -> None:
        """Exploration controller rewards moves expanding coverage and penalizes previously visited cells."""
        controller = ExplorationController(world_width=5, world_height=5, sensory_radius=2.0)
        mem = AgentMemory(world_width=5, world_height=5)
        pos = Position(1, 1)
        obs = Observation(pos, (pos, Position(1, 0)), 4, (), Position(4, 4))
        mem.update(obs)
        # Artificially record 3 visits to Position(0, 1) [Action.LEFT]
        mem.visit_counts[Position(0, 1)] = 3

        scores = controller.evaluate_exploratory_actions(obs, mem, (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        # Action.LEFT should be scored lower due to repeat visits
        self.assertLess(scores[Action.LEFT], scores[Action.RIGHT])
        self.assertLess(scores[Action.LEFT], scores[Action.DOWN])

    def test_discovery_of_unseen_resource_updates_memory(self) -> None:
        """When a resource enters sensory radius after initial step, exploration_discoveries increments."""
        mem = AgentMemory(world_width=5, world_height=5)
        # Step 0: no resources visible
        obs0 = Observation(Position(0, 0), (Position(0, 0),), 4, (), Position(4, 4))
        mem.update(obs0)
        self.assertEqual(mem.exploration_discoveries, 0)

        # Step 1: moves to (1, 0), discovering resource at (2, 1)
        res = Resource(Position(2, 1), 5, True)
        obs1 = Observation(Position(1, 0), (Position(1, 0),), 3, (res,), Position(4, 4))
        mem.update(obs1)
        self.assertEqual(mem.exploration_discoveries, 1)
        self.assertIn(Position(2, 1), mem.known_resources)

    # --------------------------------------------------------------------------
    # Stage 4: Cycle Detection and Recovery Tests
    # --------------------------------------------------------------------------

    def test_detects_consecutive_recharges(self) -> None:
        """CycleDetector triggers when two consecutive recharges occur."""
        detector = CycleDetector()
        mem = AgentMemory(world_width=4, world_height=4)
        mem.record_action_result(Position(1, 1), 3, Action.RECHARGE, True)
        mem.record_action_result(Position(1, 1), 5, Action.RECHARGE, True)

        res = detector.detect(mem)
        self.assertTrue(res.detected)
        self.assertEqual(res.cycle_type, "recharge_loop")
        self.assertEqual(res.penalized_action, Action.RECHARGE)

    def test_detects_ping_pong_oscillation(self) -> None:
        """CycleDetector triggers when agent ping-pongs between two positions."""
        detector = CycleDetector()
        mem = AgentMemory(world_width=4, world_height=4)
        pA, pB = Position(1, 1), Position(2, 1)
        mem.record_action_result(pA, 4, Action.RIGHT, True)
        mem.record_action_result(pB, 3, Action.LEFT, True)
        mem.record_action_result(pA, 2, Action.RIGHT, True)
        mem.record_action_result(pB, 1, Action.LEFT, True)

        res = detector.detect(mem)
        self.assertTrue(res.detected)
        self.assertEqual(res.cycle_type, "ping_pong")
        self.assertEqual(res.penalized_action, Action.LEFT)

    def test_detects_repeated_rejected_actions(self) -> None:
        """CycleDetector triggers when an action is repeatedly rejected."""
        detector = CycleDetector()
        mem = AgentMemory(world_width=4, world_height=4)
        mem.record_action_result(Position(0, 0), 3, Action.UP, False)
        mem.record_action_result(Position(0, 0), 3, Action.UP, False)

        res = detector.detect(mem)
        self.assertTrue(res.detected)
        self.assertEqual(res.cycle_type, "repeated_rejection")
        self.assertEqual(res.penalized_action, Action.UP)

    # --------------------------------------------------------------------------
    # Integrated Robust Policy Tests (Scenarios 5, 6, 7)
    # --------------------------------------------------------------------------

    def test_robust_policy_solves_reusable_loop_scenario(self) -> None:
        """RobustResourcePolicy eliminates infinite recharge loop on reusable resource (Phase 12.1 Test-5)."""
        world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            max_energy=5,
            resources=(Resource(Position(1, 1), 3, consumable=False),),
            observable_target=Position(3, 3),
        )
        sim = Simulation(world)
        policy = RobustResourcePolicy(
            world_width=world.width,
            world_height=world.height,
            max_energy=world.max_energy,
            movement_energy_cost=world.movement_energy_cost,
        )

        steps = 0
        recharges = 0
        reached = False
        while not reached and steps < 20:
            obs = sim.observe()
            action = policy.choose_action(obs)
            res = sim.step(action)
            steps += 1
            if action == Action.RECHARGE and res.success:
                recharges += 1
            reached = res.observation.entity_position == world.observable_target

        self.assertTrue(reached, f"Failed to reach target within {steps} steps.")
        # Must recharge at most twice to reach capacity, then proceed to target
        self.assertLessEqual(recharges, 2)
        self.assertEqual(recharges, 2)
        self.assertEqual(steps, 8)  # 2 steps to res + 2 recharges + 4 steps to target = 8 steps

    def test_robust_policy_solves_resource_outside_sensory_radius(self) -> None:
        """RobustResourcePolicy explores safely and discovers resource beyond initial sensory range (Phase 12.1 Test-6)."""
        world = WorldConfig(
            width=5,
            height=5,
            initial_x=0,
            initial_y=0,
            initial_energy=4,
            max_energy=8,
            resources=(Resource(Position(2, 1), 6, consumable=True),),
            observable_target=Position(4, 3),
        )
        sim = Simulation(world)
        policy = RobustResourcePolicy(
            world_width=world.width,
            world_height=world.height,
            max_energy=world.max_energy,
            movement_energy_cost=world.movement_energy_cost,
        )

        steps = 0
        reached = False
        while not reached and steps < 25:
            obs = sim.observe()
            action = policy.choose_action(obs)
            res = sim.step(action)
            steps += 1
            reached = res.observation.entity_position == world.observable_target

        self.assertTrue(reached, f"Failed to reach target within {steps} steps.")
        self.assertGreater(policy.memory.exploration_discoveries, 0)

    def test_robust_policy_handles_nearest_insufficient_counterexample(self) -> None:
        """RobustResourcePolicy navigates to the farther sufficient resource instead of nearest insufficient."""
        world = WorldConfig(
            width=4,
            height=4,
            initial_x=0,
            initial_y=0,
            initial_energy=2,
            max_energy=8,
            resources=(
                Resource(Position(1, 0), recharge_amount=1, consumable=True),
                Resource(Position(2, 0), recharge_amount=6, consumable=True),
            ),
            observable_target=Position(3, 3),
        )
        sim = Simulation(world)
        policy = RobustResourcePolicy(
            world_width=world.width,
            world_height=world.height,
            max_energy=world.max_energy,
            movement_energy_cost=world.movement_energy_cost,
        )

        steps = 0
        reached = False
        recharges_at_pos: dict[Position, int] = {}
        while not reached and steps < 25:
            obs = sim.observe()
            pos = obs.entity_position
            action = policy.choose_action(obs)
            res = sim.step(action)
            steps += 1
            if action == Action.RECHARGE and res.success:
                recharges_at_pos[pos] = recharges_at_pos.get(pos, 0) + 1
            reached = res.observation.entity_position == world.observable_target

        self.assertTrue(reached, f"Failed to reach target within {steps} steps.")
        # Must have recharged at Position(2, 0), NOT at Position(1, 0)
        self.assertEqual(recharges_at_pos.get(Position(2, 0), 0), 1)
        self.assertEqual(recharges_at_pos.get(Position(1, 0), 0), 0)


if __name__ == "__main__":
    unittest.main()
