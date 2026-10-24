"""WorldSim Phase 13.2: Resolve the Eight-Agent Bottleneck Benchmark.

Implements all 6 controlled experimental scenarios:
- Scenario 1: Original 8-Agent Bottleneck (Phase 13 baseline vs Phase 13.1 vs Phase 13.2).
- Scenario 2: 4-Agent Intersection Regression (confirm 100% completion maintained).
- Scenario 3: 2-Agent Resource Competition (lowest_id vs fair_balanced arbitration).
- Scenario 4: New 8-Agent Layout with Different Resource Positions (asymmetric resource pads).
- Scenario 5: 8-Agent Layout with Asymmetric Starting Positions (corners & quadrants).
- Scenario 6: Intentionally Infeasible Scenario (honest failure accounting & over-constrained bounds).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from .actions import Action
from .config import WorldConfig
from .coordination import (
    DeadlockDetector,
    DeadlockEvent,
    FairnessMonitor,
    MultiAgentCoordinator,
    ResourceArbitrator,
    jains_fairness_index,
)
from .linear_learning import LinearLearningConfig, LinearLearningRunner
from .models import Position, Resource
from .multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentSimulation,
    MultiAgentStepResult,
    MultiAgentWorldConfig,
)
from .policy import RandomPolicy
from .robust_policy import RobustResourcePolicy


@dataclass(frozen=True, slots=True)
class Phase13_2ExperimentMetrics:
    """Detailed quantitative metrics for Phase 13.2 multi-agent experiments."""

    experiment_name: str
    description: str
    episodes: int
    agent_ids: tuple[str, ...]
    per_agent_success_rate: dict[str, float]
    overall_completion_rate: float
    mean_steps_to_goal: float
    mean_episode_duration: float
    mean_energy_consumed: float
    total_energy_consumed: float
    resource_consumption_count: int
    per_agent_recharges: dict[str, float]
    vertex_collisions: int
    swap_collisions: int
    occupied_collisions: int
    total_collisions: int
    resource_contentions: int
    deadlock_detections: int
    recovery_attempts: int
    successful_recoveries: int
    unresolved_deadlocks: int
    mean_ticks_blocked: float
    starvation_events: int
    exhaustion_rates: dict[str, float]
    timeout_rates: dict[str, float]
    jains_fairness_goal: float
    jains_fairness_energy: float
    resource_load_distribution: dict[str, int]
    unused_resources_count: int


def run_phase13_2_episode(
    sim: MultiAgentSimulation,
    policies: dict[str, Any],
    max_steps: int = 50,
) -> tuple[bool, int, dict[str, bool], dict[str, int], dict[str, int]]:
    """Execute a single episode and return high-level outcomes."""
    for aid, pol in policies.items():
        if hasattr(pol, "reset_memory"):
            pol.reset_memory()

    step_count = 0
    recharges = {aid: 0 for aid in sim.state.agents}
    reached = {aid: False for aid in sim.state.agents}

    while step_count < max_steps:
        all_done = all(not a.active for a in sim.state.agents.values())
        if all_done:
            break

        if sim.coordinator is not None:
            sim.coordinator.reset_tick()

        actions: dict[str, Action] = {}
        for aid, agent in sim.state.agents.items():
            if not agent.active:
                actions[aid] = Action.UP
                continue
            obs = sim.observe_agent(aid)
            pol = policies.get(aid)
            if pol is not None and hasattr(pol, "choose_action"):
                actions[aid] = pol.choose_action(obs)
            elif pol is not None and hasattr(pol, "select_action"):
                actions[aid] = pol.select_action(obs, explore=False)
            else:
                actions[aid] = Action.UP

        res = sim.step(actions)
        step_count += 1

        for aid, ar in res.agent_results.items():
            if ar.action == Action.RECHARGE and ar.success:
                recharges[aid] += 1
            if ar.reached_target:
                reached[aid] = True

            pol = policies.get(aid)
            if pol is not None and hasattr(pol, "memory") and hasattr(pol.memory, "record_action_result"):
                pol.memory.record_action_result(
                    ar.position_before,
                    ar.energy_before,
                    ar.action,
                    ar.success,
                )

    completed = all(reached[aid] for aid in sim.state.agents)
    consumed_energy = {
        aid: sim.fairness_monitor.metrics[aid].total_energy_consumed for aid in sim.state.agents
    }
    return completed, step_count, reached, consumed_energy, recharges


def run_experiment_batch_phase13_2(
    config: MultiAgentWorldConfig,
    policy_factories: dict[str, Any],
    experiment_name: str,
    description: str,
    episodes: int = 20,
    max_steps: int = 50,
) -> Phase13_2ExperimentMetrics:
    """Execute repeated trials for a configured scenario and aggregate metrics."""
    agent_ids = tuple(config.agents[i].id for i in range(len(config.agents)))

    durations: list[int] = []
    success_steps: list[int] = []
    overall_success_count = 0
    per_agent_success_counts = {aid: 0 for aid in agent_ids}
    total_energy_per_agent = {aid: 0 for aid in agent_ids}
    total_recharges_per_agent = {aid: 0 for aid in agent_ids}
    exhaustion_counts = {aid: 0 for aid in agent_ids}
    timeout_counts = {aid: 0 for aid in agent_ids}

    total_v = 0
    total_s = 0
    total_o = 0
    total_c = 0
    total_depleted = 0
    total_deadlocks = 0
    total_recoveries = 0
    total_successful_recoveries = 0
    total_unresolved = 0
    total_ticks_blocked = 0
    total_starvation = 0

    goal_fairness_scores: list[float] = []
    energy_fairness_scores: list[float] = []
    agg_resource_load: dict[str, int] = {}
    agg_unused_resources = 0

    for ep in range(episodes):
        sim = MultiAgentSimulation(config, seed=ep + 100)
        policies = {aid: factory() for aid, factory in policy_factories.items()}
        if config.enable_coordination:
            for p in policies.values():
                if hasattr(p, "coordinator") and p.coordinator is None:
                    p.coordinator = sim.coordinator
                    if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                        p.utility_evaluator.coordinator = sim.coordinator

        comp, steps, reached, consumed, recharges = run_phase13_2_episode(sim, policies, max_steps=max_steps)

        durations.append(steps)
        if comp:
            overall_success_count += 1
            success_steps.append(steps)

        for aid in agent_ids:
            if reached[aid]:
                per_agent_success_counts[aid] += 1
            elif sim.state.agents[aid].status == "exhausted":
                exhaustion_counts[aid] += 1
            else:
                timeout_counts[aid] += 1

            total_energy_per_agent[aid] += consumed.get(aid, 0)
            total_recharges_per_agent[aid] += recharges.get(aid, 0)

        total_v += sim.cumulative_vertex_collisions
        total_s += sim.cumulative_swap_collisions
        total_o += sim.cumulative_occupied_collisions
        total_c += sim.cumulative_resource_contentions
        total_depleted += sim.cumulative_resources_depleted

        total_deadlocks += len(sim.deadlock_detector.detected_events)
        total_recoveries += sim.deadlock_detector.recovery_attempts
        total_successful_recoveries += sim.deadlock_detector.successful_recoveries
        total_unresolved += sim.deadlock_detector.unresolved_deadlocks

        total_ticks_blocked += sum(m.ticks_blocked for m in sim.fairness_monitor.metrics.values())
        total_starvation += sum(m.starvation_events for m in sim.fairness_monitor.metrics.values())

        goal_fairness_scores.append(sim.fairness_monitor.compute_jains_fairness("goal_completion"))
        energy_fairness_scores.append(sim.fairness_monitor.compute_jains_fairness("energy_gained"))

        # Aggregate resource load distribution
        for pos, count in sim.fairness_monitor.resource_load_distribution.items():
            pos_key = f"({pos.x},{pos.y})"
            agg_resource_load[pos_key] = agg_resource_load.get(pos_key, 0) + count
        agg_unused_resources += sim.fairness_monitor.get_unused_resources_count(config.resources)

    per_agent_succ_rate = {aid: (per_agent_success_counts[aid] / episodes) * 100.0 for aid in agent_ids}
    overall_succ_rate = (overall_success_count / episodes) * 100.0
    mean_dur = float(statistics.mean(durations)) if durations else 0.0
    mean_succ_steps = float(statistics.mean(success_steps)) if success_steps else 0.0
    tot_energy = sum(total_energy_per_agent.values())
    mean_energy = tot_energy / episodes if episodes else 0.0
    per_agent_rech = {aid: total_recharges_per_agent[aid] / episodes for aid in agent_ids}
    exhaustion_rates = {aid: (exhaustion_counts[aid] / episodes) * 100.0 for aid in agent_ids}
    timeout_rates = {aid: (timeout_counts[aid] / episodes) * 100.0 for aid in agent_ids}

    mean_goal_fairness = float(statistics.mean(goal_fairness_scores)) if goal_fairness_scores else 1.0
    mean_energy_fairness = float(statistics.mean(energy_fairness_scores)) if energy_fairness_scores else 1.0
    mean_unused = round(agg_unused_resources / episodes, 2) if episodes else 0

    return Phase13_2ExperimentMetrics(
        experiment_name=experiment_name,
        description=description,
        episodes=episodes,
        agent_ids=agent_ids,
        per_agent_success_rate=per_agent_succ_rate,
        overall_completion_rate=overall_succ_rate,
        mean_steps_to_goal=round(mean_succ_steps, 2),
        mean_episode_duration=round(mean_dur, 2),
        mean_energy_consumed=round(mean_energy, 2),
        total_energy_consumed=tot_energy,
        resource_consumption_count=total_depleted,
        per_agent_recharges={k: round(v, 2) for k, v in per_agent_rech.items()},
        vertex_collisions=total_v,
        swap_collisions=total_s,
        occupied_collisions=total_o,
        total_collisions=total_v + total_s + total_o,
        resource_contentions=total_c,
        deadlock_detections=total_deadlocks,
        recovery_attempts=total_recoveries,
        successful_recoveries=total_successful_recoveries,
        unresolved_deadlocks=total_unresolved,
        mean_ticks_blocked=round(total_ticks_blocked / episodes, 2),
        starvation_events=total_starvation,
        exhaustion_rates=exhaustion_rates,
        timeout_rates=timeout_rates,
        jains_fairness_goal=round(mean_goal_fairness, 4),
        jains_fairness_energy=round(mean_energy_fairness, 4),
        resource_load_distribution=agg_resource_load,
        unused_resources_count=int(mean_unused),
    )


def make_robust_factory(
    width: int,
    height: int,
    max_e: int,
    learner: Any | None = None,
    margin: int = 0,
    coordinator: Any | None = None,
) -> Any:
    return lambda: RobustResourcePolicy(
        world_width=width,
        world_height=height,
        max_energy=max_e,
        base_learner=learner,
        congestion_safety_margin=margin,
        coordinator=coordinator,
    )


def run_all_phase13_2_experiments(output_path: str = "benchmark_results_phase13_2.json") -> dict[str, Any]:
    """Execute all Phase 13.2 controlled experiments and export machine-readable JSON."""
    print("=" * 80)
    print("WORLDSIM PHASE 13.2 BENCHMARK: RESOLVING THE EIGHT-AGENT BOTTLENECK")
    print("=" * 80)

    # Train base Linear Q learner on standard WorldConfig
    train_world = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
    )
    lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
    lin_runner.train(1000)
    base_linear_learner = lin_runner.learner

    experiments_data = []

    # --------------------------------------------------------------------------
    # Scenario 1: Original 8-Agent Bottleneck
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 1] Original 8-Agent Bottleneck (Phase 13 vs 13.1 vs 13.2)...")
    agents_1 = []
    for i in range(8):
        ax = i % 4
        ay = i // 4
        tx = 7 - ax
        ty = 7 - ay
        agents_1.append(AgentConfig(id=f"agent_{i+1}", initial_x=ax, initial_y=ay, initial_energy=6, target=Position(tx, ty)))

    # 1.1 Phase 13 Baseline: Symmetric Bounce
    cfg_s1_base = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_1),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="bounce",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_s1_base = {a.id: make_robust_factory(8, 8, 10, base_linear_learner) for a in agents_1}
    res_s1_base = run_experiment_batch_phase13_2(
        cfg_s1_base, pol_s1_base,
        "Scenario 1A: Eight-Agent Bottleneck (Phase 13 Baseline)",
        "8 agents funneling into central resources (pure bounce, lowest_id)",
        episodes=20,
    )
    print(f"  S1A Base: Succ={res_s1_base.overall_completion_rate:.1f}%, Mean Agent Succ={statistics.mean(res_s1_base.per_agent_success_rate.values()):.1f}%, Cols={res_s1_base.total_collisions}")
    experiments_data.append(asdict(res_s1_base))

    # 1.2 Phase 13.1 Coordinator
    cfg_s1_p13_1 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_1),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s1_p13_1 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=0) for a in agents_1}
    res_s1_p13_1 = run_experiment_batch_phase13_2(
        cfg_s1_p13_1, pol_s1_p13_1,
        "Scenario 1B: Eight-Agent Bottleneck (Phase 13.1 Configuration)",
        "8 agents with priority rotation and basic coordination (margin=0)",
        episodes=20,
    )
    print(f"  S1B P13.1: Succ={res_s1_p13_1.overall_completion_rate:.1f}%, Mean Agent Succ={statistics.mean(res_s1_p13_1.per_agent_success_rate.values()):.1f}%, Cols={res_s1_p13_1.total_collisions}")
    experiments_data.append(asdict(res_s1_p13_1))

    # 1.3 Phase 13.2 Resolved: Doorstep Queuing, Multi-Recharge Utility, Energy-Aware Alternatives
    cfg_s1_p13_2 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_1),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s1_p13_2 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_1}
    res_s1_p13_2 = run_experiment_batch_phase13_2(
        cfg_s1_p13_2, pol_s1_p13_2,
        "Scenario 1C: Eight-Agent Bottleneck (Phase 13.2 Resolved)",
        "8 agents with multi-recharge capacity, doorstep queuing, and load balancing",
        episodes=20,
    )
    print(f"  S1C P13.2: Succ={res_s1_p13_2.overall_completion_rate:.1f}%, Mean Agent Succ={statistics.mean(res_s1_p13_2.per_agent_success_rate.values()):.1f}%, Cols={res_s1_p13_2.total_collisions}")
    experiments_data.append(asdict(res_s1_p13_2))

    # --------------------------------------------------------------------------
    # Scenario 2: Four-Agent Intersection Regression Verification
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 2] Four-Agent Intersection Regression...")
    cfg_s2 = MultiAgentWorldConfig(
        width=6, height=6,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=2, initial_energy=5, target=Position(5, 2)),
            AgentConfig(id="agent_2", initial_x=5, initial_y=3, initial_energy=5, target=Position(0, 3)),
            AgentConfig(id="agent_3", initial_x=2, initial_y=0, initial_energy=5, target=Position(2, 5)),
            AgentConfig(id="agent_4", initial_x=3, initial_y=5, initial_energy=5, target=Position(3, 0)),
        ),
        resources=(Resource(Position(2, 2), 6, False), Resource(Position(3, 3), 6, False)),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        goal_criterion="individual",
    )
    pol_s2 = {a.id: make_robust_factory(6, 6, 8, base_linear_learner, margin=1) for a in cfg_s2.agents}
    res_s2 = run_experiment_batch_phase13_2(
        cfg_s2, pol_s2,
        "Scenario 2: Four-Agent Cross Congestion Regression",
        "4-way intersection regression confirming 100% completion maintained",
        episodes=20,
    )
    print(f"  S2 Intersection: Succ={res_s2.overall_completion_rate:.1f}%, Cols={res_s2.total_collisions}, Duration={res_s2.mean_episode_duration:.1f}")
    experiments_data.append(asdict(res_s2))

    # --------------------------------------------------------------------------
    # Scenario 3: Two-Agent Resource Competition (Lowest-ID vs Fair Balanced)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 3] Two-Agent Resource Competition...")
    cfg_s3_lowest = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=1, initial_energy=2, target=Position(3, 3)),
            AgentConfig(id="agent_2", initial_x=1, initial_y=0, initial_energy=2, target=Position(3, 3)),
        ),
        resources=(Resource(Position(1, 1), 6, consumable=True),),
        max_energy=8,
        collision_rule="priority",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_s3 = {
        "agent_1": make_robust_factory(4, 4, 8, base_linear_learner),
        "agent_2": make_robust_factory(4, 4, 8, base_linear_learner),
    }
    res_s3_low = run_experiment_batch_phase13_2(
        cfg_s3_lowest, pol_s3,
        "Scenario 3A: Resource Competition (Lowest-ID)",
        "Two agents competing for consumable resource under lowest_id",
        episodes=30,
    )

    cfg_s3_fair = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=1, initial_energy=2, target=Position(3, 3)),
            AgentConfig(id="agent_2", initial_x=1, initial_y=0, initial_energy=2, target=Position(3, 3)),
        ),
        resources=(Resource(Position(1, 1), 6, consumable=True),),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    res_s3_fair = run_experiment_batch_phase13_2(
        cfg_s3_fair, pol_s3,
        "Scenario 3B: Resource Competition (Fair Balanced)",
        "Two agents competing for consumable resource under fair_balanced",
        episodes=30,
    )
    print(f"  S3A Lowest-ID: Agent1={res_s3_low.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_low.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_low.jains_fairness_goal:.3f}")
    print(f"  S3B Fair:      Agent1={res_s3_fair.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_fair.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_fair.jains_fairness_goal:.3f}")
    experiments_data.append(asdict(res_s3_low))
    experiments_data.append(asdict(res_s3_fair))

    # --------------------------------------------------------------------------
    # Scenario 4: New 8-Agent Layout with Different Resource Positions
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 4] New 8-Agent Layout with Different Resource Positions...")
    agents_s4 = []
    for i in range(8):
        ax = i % 4
        ay = i // 4
        tx = 7 - ax
        ty = 7 - ay
        agents_s4.append(AgentConfig(id=f"agent_{i+1}", initial_x=ax, initial_y=ay, initial_energy=6, target=Position(tx, ty)))

    cfg_s4 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s4),
        resources=(Resource(Position(2, 5), 6, False), Resource(Position(5, 2), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s4 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s4}
    res_s4 = run_experiment_batch_phase13_2(
        cfg_s4, pol_s4,
        "Scenario 4: New 8-Agent Layout with Asymmetric Resource Positions",
        "8 agents with resources relocated to (2, 5) and (5, 2)",
        episodes=20,
    )
    print(f"  S4 Relocated Res: Succ={res_s4.overall_completion_rate:.1f}%, Mean Agent Succ={statistics.mean(res_s4.per_agent_success_rate.values()):.1f}%, Cols={res_s4.total_collisions}")
    experiments_data.append(asdict(res_s4))

    # --------------------------------------------------------------------------
    # Scenario 5: 8-Agent Layout with Asymmetric Starting Positions
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 5] 8-Agent Layout with Asymmetric Starting Positions...")
    # Agents start in 4 different quadrants/corners instead of a compact 4x2 block
    asym_starts = [
        (0, 0, 7, 7),
        (0, 7, 7, 0),
        (7, 0, 0, 7),
        (7, 7, 0, 0),
        (1, 0, 6, 7),
        (0, 6, 7, 1),
        (6, 7, 1, 0),
        (7, 1, 0, 6),
    ]
    agents_s5 = [
        AgentConfig(id=f"agent_{i+1}", initial_x=sx, initial_y=sy, initial_energy=6, target=Position(gx, gy))
        for i, (sx, sy, gx, gy) in enumerate(asym_starts)
    ]
    cfg_s5 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s5),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s5 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s5}
    res_s5 = run_experiment_batch_phase13_2(
        cfg_s5, pol_s5,
        "Scenario 5: 8-Agent Asymmetric Start Positions",
        "8 agents starting in opposing corners funneling across central intersection",
        episodes=20,
    )
    print(f"  S5 Asymmetric: Succ={res_s5.overall_completion_rate:.1f}%, Mean Agent Succ={statistics.mean(res_s5.per_agent_success_rate.values()):.1f}%, Cols={res_s5.total_collisions}")
    experiments_data.append(asdict(res_s5))

    # --------------------------------------------------------------------------
    # Scenario 6: Intentionally Infeasible Scenario (Honest Failure Accounting)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 6] Intentionally Infeasible Scenario...")
    # Agent requires distance 10, initial energy 2, max energy 4, only 1 consumable resource of 3 units at distance 5
    # Mathematically impossible: energy=2, resource at dist=5 -> cannot even reach resource!
    cfg_s6 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=2, target=Position(7, 7)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=1, initial_energy=2, target=Position(7, 6)),
        ),
        resources=(Resource(Position(5, 5), 3, consumable=True),),
        max_energy=4,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        goal_criterion="individual",
    )
    pol_s6 = {
        "agent_1": make_robust_factory(8, 8, 4, base_linear_learner),
        "agent_2": make_robust_factory(8, 8, 4, base_linear_learner),
    }
    res_s6 = run_experiment_batch_phase13_2(
        cfg_s6, pol_s6,
        "Scenario 6: Intentionally Infeasible Environment",
        "Overconstrained parameters where target and resource are mathematically unreachable",
        episodes=20,
    )
    print(f"  S6 Infeasible: Overall Succ={res_s6.overall_completion_rate:.1f}%, Exhaustion={res_s6.exhaustion_rates['agent_1']:.1f}% (honest accounting)")
    experiments_data.append(asdict(res_s6))

    # Save to JSON
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"version": "13.2", "scenarios": experiments_data}, f, indent=2)

    print(f"\nSaved machine-readable Phase 13.2 benchmark results to {out_file.resolve()}")
    return {"version": "13.2", "scenarios": experiments_data}


if __name__ == "__main__":
    run_all_phase13_2_experiments()
