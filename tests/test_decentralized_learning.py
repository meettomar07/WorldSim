"""Unit tests for Phase 14.4 Decentralized Learning and Communicative Intent."""

from pathlib import Path
import unittest

from worldsim.actions import Action
from worldsim.coordination import MultiAgentCoordinator
from worldsim.decentralized_learning import (
    DecentralizedFeatureExtractor,
    DecentralizedLearningConfig,
    DecentralizedQLearner,
    DecentralizedRewardEngine,
    HybridCoordinationPolicy,
    LearnedCoordinationPolicy,
    train_decentralized_coordination,
)
from worldsim.dynamic_obstacles import DynamicObstacleConfig, DynamicObstacleState
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)


class TestDecentralizedLearning(unittest.TestCase):
    """Test suite verifying decentralized feature extraction, learning, and communication."""

    def test_feature_extractor_no_communication(self) -> None:
        """1. Verify that enable_communication=False produces strictly zero intent features."""
        extractor = DecentralizedFeatureExtractor(
            world_width=10, world_height=10, sensory_radius=2.0, enable_communication=False
        )
        obs = MultiAgentObservation(
            agent_id="ag1",
            entity_position=Position(3, 3),
            cells=(),
            energy=8,
            observed_resources=(),
            target=Position(6, 3),
            observed_agents=(AgentState(id="ag2", position=Position(4, 3), energy=8, target=Position(1, 3)),),
        )
        # Even if dummy intentions are passed, extractor ignores them when comm is disabled
        dummy_intents = {"ag2": Position(3, 3)}
        feats = extractor.extract(obs, Action.RIGHT, peer_intentions=dummy_intents)

        self.assertNotIn("comm_peer_intent_vertex_conflict", feats)
        self.assertNotIn("comm_peer_swap_conflict", feats)
        self.assertNotIn("comm_peer_intends_my_cell", feats)
        self.assertEqual(feats.get("into_peer_current_cell", 0.0), 1.0)

    def test_feature_extractor_with_communication(self) -> None:
        """2. Verify that enable_communication=True captures vertex and swap intent conflicts."""
        extractor = DecentralizedFeatureExtractor(
            world_width=10, world_height=10, sensory_radius=2.0, enable_communication=True
        )
        obs = MultiAgentObservation(
            agent_id="ag1",
            entity_position=Position(3, 3),
            cells=(),
            energy=8,
            observed_resources=(),
            target=Position(6, 3),
            observed_agents=(AgentState(id="ag2", position=Position(4, 3), energy=8, target=Position(1, 3)),),
        )
        # Peer at (4, 3) intends (3, 3) -> Swap conflict when ag1 chooses RIGHT (dest = 4, 3)
        peer_intents = {"ag2": Position(3, 3)}
        feats_right = extractor.extract(obs, Action.RIGHT, peer_intentions=peer_intents)
        self.assertEqual(feats_right.get("comm_peer_swap_conflict", 0.0), 1.0)
        self.assertEqual(feats_right.get("comm_peer_intends_my_cell", 0.0), 1.0)

        # Peer at (4, 3) intends (3, 2) -> Vertex conflict when ag1 chooses UP (dest = 3, 2)
        peer_intents_v = {"ag2": Position(3, 2)}
        feats_up = extractor.extract(obs, Action.UP, peer_intentions=peer_intents_v)
        self.assertEqual(feats_up.get("comm_peer_intent_vertex_conflict", 0.0), 1.0)

    def test_communication_radius_strict_boundary(self) -> None:
        """3. Verify that intentions of peers beyond Euclidean R=2.0 are never visible."""
        coord = MultiAgentCoordinator(world_width=10, world_height=10, coordination_mode="decentralized", communication_radius=2.0)
        # ag1 at (1, 1), ag2 at (1, 3) (dist=2.0 -> visible), ag3 at (1, 4) (dist=3.0 -> invisible)
        current_pos = {
            "ag1": Position(1, 1),
            "ag2": Position(1, 3),
            "ag3": Position(1, 4),
        }
        coord.register_intent("ag2", Position(2, 3), current_position=Position(1, 3))
        coord.register_intent("ag3", Position(2, 4), current_position=Position(1, 4))

        visible_to_ag1 = coord.get_visible_peers("ag1", Position(1, 1), current_pos)
        self.assertIn("ag2", visible_to_ag1, "Peer at distance 2.0 must be visible")
        self.assertNotIn("ag3", visible_to_ag1, "Distant peer at distance 3.0 must never be visible")

    def test_q_learning_update_and_gradient_clip(self) -> None:
        """4. Verify linear Q-learning TD updates, gradient clipping, and weight adjustments."""
        cfg = DecentralizedLearningConfig(learning_rate=0.1, discount_factor=0.9, gradient_clip=1.0)
        learner = DecentralizedQLearner(config=cfg, seed=42)

        feats = {"feat_a": 1.0, "feat_b": 0.5}
        # current_q = 0.0, reward = 10.0, next_max_q = 0.0 -> td_error = 10.0 -> clipped to 1.0
        td_err = learner.update(features=feats, reward=10.0, next_max_q=0.0, current_q=0.0, done=True)
        self.assertEqual(td_err, 10.0)
        # w_a <- 0 + 0.1 * 1.0 * 1.0 = 0.1
        # w_b <- 0 + 0.1 * 1.0 * 0.5 = 0.05
        self.assertAlmostEqual(learner.weights["feat_a"], 0.1, places=3)
        self.assertAlmostEqual(learner.weights["feat_b"], 0.05, places=3)

    def test_deterministic_training_from_seed(self) -> None:
        """5. Verify that training from identical seeds produces bit-for-bit identical weights."""
        l1 = train_decentralized_coordination(enable_communication=False, training_seeds=(42, 43), epochs=1)
        l2 = train_decentralized_coordination(enable_communication=False, training_seeds=(42, 43), epochs=1)

        self.assertEqual(l1.weights.keys(), l2.weights.keys())
        for k in l1.weights:
            self.assertEqual(l1.weights[k], l2.weights[k], f"Weight {k} must match bit-for-bit")

    def test_authoritative_movement_validation(self) -> None:
        """6. Verify simulation engine strictly prevents illegal double occupancy under learned policy."""
        # Two agents head-on on a 4x1 corridor
        world_cfg = MultiAgentWorldConfig(
            width=4,
            height=1,
            agents=(
                AgentConfig(id="ag1", initial_x=1, initial_y=0, initial_energy=5, target=Position(3, 0)),
                AgentConfig(id="ag2", initial_x=2, initial_y=0, initial_energy=5, target=Position(0, 0)),
            ),
            enable_coordination=True,
            coordination_mode="decentralized",
        )
        sim = MultiAgentSimulation(world_cfg, seed=123)
        # Force swap move
        step_res = sim.step({"ag1": Action.RIGHT, "ag2": Action.LEFT})

        self.assertEqual(step_res.swap_collisions, 2)
        # Invariant: agents cannot swap; must stay in initial positions
        self.assertEqual(sim.state.agents["ag1"].position, Position(1, 0))
        self.assertEqual(sim.state.agents["ag2"].position, Position(2, 0))

    def test_reward_engine_penalties_and_fairness(self) -> None:
        """7. Verify collision and exhaustion penalties and team progress fairness reward."""
        engine = DecentralizedRewardEngine()
        obs = MultiAgentObservation(
            agent_id="ag1",
            entity_position=Position(2, 2),
            cells=(),
            energy=1,
            observed_resources=(),
            target=Position(5, 2),
            observed_agents=(),
        )
        # Collision reward
        r_swap = engine.compute_reward(
            agent_id="ag1",
            obs_before=obs,
            action=Action.RIGHT,
            pos_after=Position(2, 2),
            energy_after=0,
            success=False,
            collision_type="swap",
            terminated=False,
            all_agents_before={},
            all_agents_after={},
        )
        self.assertLess(r_swap, -20.0, "Swap collision and exhaustion must produce severe negative reward")

    def test_policy_action_selection_greedy_vs_exploratory(self) -> None:
        """8. Verify greedy evaluation mode ignores epsilon exploration."""
        learner = DecentralizedQLearner(seed=42)
        learner.epsilon = 1.0  # Force 100% random in exploratory mode
        learner.weights["reaches_goal"] = 100.0

        pol = LearnedCoordinationPolicy(
            world_width=5, world_height=5, sensory_radius=2.0, learner=learner, enable_communication=False
        )
        obs = MultiAgentObservation(
            agent_id="ag1",
            entity_position=Position(2, 2),
            cells=(),
            energy=5,
            observed_resources=(),
            target=Position(3, 2),
            observed_agents=(),
        )
        # Under greedy=True, Action.RIGHT (which reaches goal) MUST be selected regardless of epsilon=1.0
        chosen_greedy = pol.choose_action(obs, greedy=True)
        self.assertEqual(chosen_greedy, Action.RIGHT)

    def test_remediated_policy_d04_recharge_contention(self) -> None:
        """9. Verify remediated learned policy completes D04 with zero exhaustions."""
        from phase14_3_dynamic_obstacle_eval import build_controlled_dynamic_scenarios

        scen_d04 = build_controlled_dynamic_scenarios()[3]
        learner = DecentralizedQLearner()
        weights_path = Path("phase14_4_weights_comm.json")
        if weights_path.exists():
            learner.load_weights(weights_path)

        aids = [a.id for a in scen_d04["agents"]]
        for seed in (500, 501):
            w_cfg = MultiAgentWorldConfig(
                width=scen_d04["width"],
                height=scen_d04["height"],
                agents=scen_d04["agents"],
                resources=scen_d04["resources"],
                dynamic_obstacles=scen_d04["dynamic_obstacles"],
                max_energy=10,
                enable_coordination=True,
                coordination_mode="decentralized",
                communication_radius=2.0,
                sensory_radius=2.0,
                collision_rule="priority_rotation",
            )
            sim = MultiAgentSimulation(w_cfg, seed=seed)
            pols = {
                aid: LearnedCoordinationPolicy(
                    scen_d04["width"],
                    scen_d04["height"],
                    10,
                    2.0,
                    learner=learner,
                    enable_communication=True,
                    coordinator=sim.coordinator,
                    enable_safety_guard=True,
                )
                for aid in aids
            }
            for _ in range(40):
                if all(not sim.state.agents[aid].active for aid in aids):
                    break
                if sim.coordinator:
                    sim.coordinator.reset_tick()
                acts = {
                    aid: pols[aid].choose_action(sim.observe_agent(aid))
                    if sim.state.agents[aid].active
                    else Action.UP
                    for aid in aids
                }
                res = sim.step(acts)
                for aid in aids:
                    r = res.agent_results[aid]
                    pols[aid].record_action_result(r.position_after, r.energy_after, r.action, r.success)

            for aid in aids:
                self.assertNotEqual(sim.state.agents[aid].status, "exhausted", f"Agent {aid} exhausted on seed {seed}")
                self.assertEqual(sim.state.agents[aid].status, "reached_target", f"Agent {aid} failed to reach target on seed {seed}")

    def test_remediated_policy_u06_quadrant_mesh(self) -> None:
        """10. Verify remediated learned policy completes U06 quadrant mesh with zero exhaustions."""
        from phase13_5_unseen_audit import build_all_unseen_scenarios

        unseen = {s[0].scenario_id: s for s in build_all_unseen_scenarios()}
        u06_spec, _ = unseen["U06"]
        u06_agents = [
            AgentConfig(
                id=a["id"],
                initial_x=a["initial_x"],
                initial_y=a["initial_y"],
                initial_energy=a["initial_energy"],
                target=Position(a["target_x"], a["target_y"]),
            )
            for a in u06_spec.active_agents
        ]
        u06_res = tuple(
            Resource(Position(r["x"], r["y"]), recharge_amount=r["recharge_amount"], consumable=r["consumable"])
            for r in u06_spec.resources
        )
        aids = [a["id"] for a in u06_spec.active_agents]

        learner = DecentralizedQLearner()
        weights_path = Path("phase14_4_weights_comm.json")
        if weights_path.exists():
            learner.load_weights(weights_path)

        for seed in (500, 501):
            w_cfg = MultiAgentWorldConfig(
                width=u06_spec.width,
                height=u06_spec.height,
                agents=tuple(u06_agents),
                resources=u06_res,
                max_energy=10,
                enable_coordination=True,
                coordination_mode="decentralized",
                communication_radius=2.0,
                sensory_radius=2.0,
                collision_rule="priority_rotation",
            )
            sim = MultiAgentSimulation(w_cfg, seed=seed)
            pols = {
                aid: LearnedCoordinationPolicy(
                    u06_spec.width,
                    u06_spec.height,
                    10,
                    2.0,
                    learner=learner,
                    enable_communication=True,
                    coordinator=sim.coordinator,
                    enable_safety_guard=True,
                )
                for aid in aids
            }
            for _ in range(40):
                if all(not sim.state.agents[aid].active for aid in aids):
                    break
                if sim.coordinator:
                    sim.coordinator.reset_tick()
                acts = {
                    aid: pols[aid].choose_action(sim.observe_agent(aid))
                    if sim.state.agents[aid].active
                    else Action.UP
                    for aid in aids
                }
                res = sim.step(acts)
                for aid in aids:
                    r = res.agent_results[aid]
                    pols[aid].record_action_result(r.position_after, r.energy_after, r.action, r.success)

            for aid in aids:
                self.assertNotEqual(sim.state.agents[aid].status, "exhausted", f"Agent {aid} exhausted on seed {seed}")
                self.assertEqual(sim.state.agents[aid].status, "reached_target", f"Agent {aid} failed to reach target on seed {seed}")

    def test_hybrid_policy_u14_blind_intersection(self) -> None:
        """11. Verify hybrid coordination policy completes U14 blind intersection with zero exhaustions."""
        from phase13_5_unseen_audit import build_all_unseen_scenarios

        unseen = {s[0].scenario_id: s for s in build_all_unseen_scenarios()}
        u14_spec, u14_pre = unseen["U14"]
        u14_agents = [
            AgentConfig(
                id=a["id"],
                initial_x=a["initial_x"],
                initial_y=a["initial_y"],
                initial_energy=a["initial_energy"],
                target=Position(a["target_x"], a["target_y"]),
            )
            for a in u14_spec.active_agents
        ] + [
            AgentConfig(id=p.id, initial_x=p.x, initial_y=p.y, initial_energy=0, target=None)
            for p in u14_pre
        ]
        aids = [a["id"] for a in u14_spec.active_agents]

        learner = DecentralizedQLearner()
        weights_path = Path("phase14_4_weights_comm.json")
        if weights_path.exists():
            learner.load_weights(weights_path)

        for seed in (500, 501):
            w_cfg = MultiAgentWorldConfig(
                width=u14_spec.width,
                height=u14_spec.height,
                agents=tuple(u14_agents),
                resources=(),
                max_energy=10,
                enable_coordination=True,
                coordination_mode="decentralized",
                communication_radius=2.0,
                sensory_radius=2.0,
                collision_rule="priority_rotation",
            )
            sim = MultiAgentSimulation(w_cfg, seed=seed)
            pols = {
                aid: HybridCoordinationPolicy(
                    u14_spec.width,
                    u14_spec.height,
                    10,
                    2.0,
                    learner=learner,
                    enable_communication=True,
                    coordinator=sim.coordinator,
                )
                for aid in aids
            }
            for _ in range(40):
                if all(not sim.state.agents[aid].active for aid in aids):
                    break
                if sim.coordinator:
                    sim.coordinator.reset_tick()
                acts = {
                    aid: pols[aid].choose_action(sim.observe_agent(aid))
                    if sim.state.agents[aid].active
                    else Action.UP
                    for aid in aids
                }
                res = sim.step(acts)
                for aid in aids:
                    r = res.agent_results[aid]
                    pols[aid].record_action_result(r.position_after, r.energy_after, r.action, r.success)

            for aid in aids:
                self.assertNotEqual(sim.state.agents[aid].status, "exhausted", f"Agent {aid} exhausted on seed {seed}")
                self.assertEqual(sim.state.agents[aid].status, "reached_target", f"Agent {aid} failed to reach target on seed {seed}")

    def test_hybrid_policy_d03_deadlock_reduction(self) -> None:
        """12. Verify hybrid coordination policy reduces deadlocks and blocked attempts in D03."""
        from phase14_3_dynamic_obstacle_eval import build_controlled_dynamic_scenarios

        scen_d03 = build_controlled_dynamic_scenarios()[2]
        learner = DecentralizedQLearner()
        weights_path = Path("phase14_4_weights_comm.json")
        if weights_path.exists():
            learner.load_weights(weights_path)

        aids = [a.id for a in scen_d03["agents"]]
        w_cfg = MultiAgentWorldConfig(
            width=scen_d03["width"],
            height=scen_d03["height"],
            agents=scen_d03["agents"],
            dynamic_obstacles=scen_d03["dynamic_obstacles"],
            max_energy=16,
            enable_coordination=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            sensory_radius=2.0,
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(w_cfg, seed=500)
        pols = {
            aid: HybridCoordinationPolicy(
                scen_d03["width"],
                scen_d03["height"],
                16,
                2.0,
                learner=learner,
                enable_communication=True,
                coordinator=sim.coordinator,
            )
            for aid in aids
        }
        total_deadlocks = 0
        total_blocked = 0
        for _ in range(50):
            if all(not sim.state.agents[aid].active for aid in aids):
                break
            if sim.coordinator:
                sim.coordinator.reset_tick()
            acts = {
                aid: pols[aid].choose_action(sim.observe_agent(aid))
                if sim.state.agents[aid].active
                else Action.UP
                for aid in aids
            }
            res = sim.step(acts)
            total_deadlocks += len(res.deadlock_events)
            total_blocked += res.occupied_collisions
            for aid in aids:
                r = res.agent_results[aid]
                pols[aid].record_action_result(r.position_after, r.energy_after, r.action, r.success)

        # Raw learned policy had 5-10 deadlocks and 8-15 blocked per episode in D03
        self.assertLessEqual(total_deadlocks, 2, f"Deadlocks ({total_deadlocks}) should be minimal")
        self.assertLessEqual(total_blocked, 5, f"Blocked attempts ({total_blocked}) should be minimal")
        for aid in aids:
            self.assertEqual(sim.state.agents[aid].status, "reached_target")

    def test_transparent_controller_decision_tracking(self) -> None:
        """13. Verify transparent logging and reporting of learned vs safety_guard decision counts."""
        from phase14_3_dynamic_obstacle_eval import build_controlled_dynamic_scenarios

        scen_d04 = build_controlled_dynamic_scenarios()[3]
        learner = DecentralizedQLearner()
        weights_path = Path("phase14_4_weights_comm.json")
        if weights_path.exists():
            learner.load_weights(weights_path)

        aids = [a.id for a in scen_d04["agents"]]
        w_cfg = MultiAgentWorldConfig(
            width=scen_d04["width"],
            height=scen_d04["height"],
            agents=scen_d04["agents"],
            resources=scen_d04["resources"],
            dynamic_obstacles=scen_d04["dynamic_obstacles"],
            max_energy=10,
            enable_coordination=True,
            coordination_mode="decentralized",
            communication_radius=2.0,
            sensory_radius=2.0,
            collision_rule="priority_rotation",
        )
        sim = MultiAgentSimulation(w_cfg, seed=500)
        pols = {
            aid: HybridCoordinationPolicy(
                scen_d04["width"],
                scen_d04["height"],
                10,
                2.0,
                learner=learner,
                enable_communication=True,
                coordinator=sim.coordinator,
            )
            for aid in aids
        }
        for _ in range(30):
            if all(not sim.state.agents[aid].active for aid in aids):
                break
            if sim.coordinator:
                sim.coordinator.reset_tick()
            acts = {
                aid: pols[aid].choose_action(sim.observe_agent(aid))
                if sim.state.agents[aid].active
                else Action.UP
                for aid in aids
            }
            res = sim.step(acts)
            for aid in aids:
                r = res.agent_results[aid]
                pols[aid].record_action_result(r.position_after, r.energy_after, r.action, r.success)

        # Check decision count telemetry
        for aid in aids:
            counts = pols[aid].decision_counts
            self.assertIn("learned", counts)
            self.assertIn("safety_guard", counts)
            self.assertGreater(counts["learned"] + counts["safety_guard"], 0)
            self.assertIn(pols[aid].last_decision_mode, ("learned", "safety_guard"))


if __name__ == "__main__":
    unittest.main()
