"""WorldSim Phase 13.3: Reliable Completion, Fairness & Benchmark Validation.

Implements all 9 controlled experimental scenarios with rigorous metric separation:
- Scenario 1: Original 8-Agent Bottleneck (Phase 13 vs Phase 13.1 vs Phase 13.2 vs Phase 13.3).
- Scenario 2: 4-Agent Cross Intersection Regression (confirm 100% completion maintained).
- Scenario 3: 2-Agent Resource Competition (Lowest-ID vs Priority-Rotation vs Fair-Balanced).
- Scenario 4: 8-Agent Relocated Resources (Phase 13.2 baseline vs Feasible parameterization).
- Scenario 5: 8-Agent Asymmetric Starts (Phase 13.2 baseline vs Feasible parameterization).
- Scenario 6: Completed Agents Blocking Terminal Cells / Narrow Corridors Stress Test.
- Scenario 7: Deadlock and Oscillation Head-On Swap Stress Test.
- Scenario 8: Genuinely Infeasible Scenario (Honest Failure Accounting & Exhaustion Tracking).
- Scenario 9: Single-Agent Baseline Regression.

Metrics explicitly separate:
1. Overall completion rate: percentage of episodes where 100% of agents complete.
2. Per-agent success rate: percentage of agent-task instances completed, per agent ID.
3. Mean per-agent success rate: arithmetic average of per-agent success rates.
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
class Phase13_3ExperimentMetrics:
    """Detailed quantitative metrics for Phase 13.3 multi-agent experiments."""

    experiment_name: str
    description: str
    episodes: int
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    overall_completion_rate: float
    episode_completion_count: int
    episode_failure_count: int
    mean_per_agent_success_rate: float
    per_agent_success_rate: dict[str, float]
    mean_steps_to_goal: float
    median_steps_to_goal: float
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
    exhaustion_counts: dict[str, int]
    exhaustion_rates: dict[str, float]
    timeout_counts: dict[str, int]
    timeout_rates: dict[str, float]
    jains_fairness_goal: float
    jains_fairness_energy: float
    resource_load_distribution: dict[str, int]
    unused_resources_count: int


def run_phase13_3_episode(
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


def run_experiment_batch_phase13_3(
    config: MultiAgentWorldConfig,
    policy_factories: dict[str, Any],
    experiment_name: str,
    description: str,
    episodes: int = 20,
    max_steps: int = 50,
    base_seed: int = 100,
    shared_arbitrator: ResourceArbitrator | None = None,
) -> Phase13_3ExperimentMetrics:
    """Execute repeated trials for a configured scenario and aggregate metrics."""
    agent_ids = tuple(config.agents[i].id for i in range(len(config.agents)))
    seed_list = tuple(base_seed + i for i in range(episodes))

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
        cur_seed = seed_list[ep]
        sim = MultiAgentSimulation(config, seed=cur_seed, resource_arbitrator=shared_arbitrator)
        policies = {aid: factory() for aid, factory in policy_factories.items()}
        if config.enable_coordination:
            for p in policies.values():
                if hasattr(p, "coordinator") and p.coordinator is None:
                    p.coordinator = sim.coordinator
                if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                    p.utility_evaluator.coordinator = sim.coordinator

        comp, steps, reached, consumed, recharges = run_phase13_3_episode(sim, policies, max_steps=max_steps)

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
    mean_agent_succ = float(statistics.mean(per_agent_succ_rate.values())) if per_agent_succ_rate else 0.0
    overall_succ_rate = (overall_success_count / episodes) * 100.0
    failure_count = episodes - overall_success_count
    mean_dur = float(statistics.mean(durations)) if durations else 0.0
    mean_succ_steps = float(statistics.mean(success_steps)) if success_steps else 0.0
    median_succ_steps = float(statistics.median(success_steps)) if success_steps else 0.0
    tot_energy = sum(total_energy_per_agent.values())
    mean_energy = tot_energy / episodes if episodes else 0.0
    per_agent_rech = {aid: total_recharges_per_agent[aid] / episodes for aid in agent_ids}
    exhaustion_rates = {aid: (exhaustion_counts[aid] / episodes) * 100.0 for aid in agent_ids}
    timeout_rates = {aid: (timeout_counts[aid] / episodes) * 100.0 for aid in agent_ids}

    mean_goal_fairness = float(statistics.mean(goal_fairness_scores)) if goal_fairness_scores else 1.0
    mean_energy_fairness = float(statistics.mean(energy_fairness_scores)) if energy_fairness_scores else 1.0
    mean_unused = round(agg_unused_resources / episodes, 2) if episodes else 0

    return Phase13_3ExperimentMetrics(
        experiment_name=experiment_name,
        description=description,
        episodes=episodes,
        seeds=seed_list,
        agent_ids=agent_ids,
        overall_completion_rate=overall_succ_rate,
        episode_completion_count=overall_success_count,
        episode_failure_count=failure_count,
        mean_per_agent_success_rate=round(mean_agent_succ, 2),
        per_agent_success_rate=per_agent_succ_rate,
        mean_steps_to_goal=round(mean_succ_steps, 2),
        median_steps_to_goal=round(median_succ_steps, 2),
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
        exhaustion_counts=exhaustion_counts,
        exhaustion_rates=exhaustion_rates,
        timeout_counts=timeout_counts,
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


def run_all_phase13_3_experiments(output_path: str = "benchmark_results_phase13_3.json") -> dict[str, Any]:
    """Execute all Phase 13.3 controlled experiments and export machine-readable JSON."""
    print("=" * 80)
    print("WORLDSIM PHASE 13.3 BENCHMARK: RELIABLE COMPLETION, FAIRNESS & VALIDATION")
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
    # Scenario 1: Original 8-Agent Bottleneck (Phase 13 vs 13.1 vs 13.2 vs 13.3)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 1] Original 8-Agent Bottleneck Progression...")
    agents_1 = []
    for i in range(8):
        ax = i % 4
        ay = i // 4
        tx = 7 - ax
        ty = 7 - ay
        agents_1.append(AgentConfig(id=f"agent_{i+1}", initial_x=ax, initial_y=ay, initial_energy=6, target=Position(tx, ty)))

    # 1A: Phase 13 Baseline (Symmetric Bounce, Lowest-ID, no coordination)
    cfg_s1_p13 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_1),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="bounce",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_s1_p13 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner) for a in agents_1}
    res_s1_p13 = run_experiment_batch_phase13_3(
        cfg_s1_p13, pol_s1_p13,
        "Scenario 1A: Eight-Agent Bottleneck (Phase 13 Baseline)",
        "8 agents funneling into central resources (pure bounce, lowest_id, no coordination)",
        episodes=20, base_seed=100,
    )
    print(f"  S1A Phase 13:   Overall={res_s1_p13.overall_completion_rate:.1f}%, Mean Agent={res_s1_p13.mean_per_agent_success_rate:.1f}%, Cols={res_s1_p13.total_collisions}")
    experiments_data.append(asdict(res_s1_p13))

    # 1B: Phase 13.1 Baseline (Priority rotation, basic coordination, margin=0)
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
    res_s1_p13_1 = run_experiment_batch_phase13_3(
        cfg_s1_p13_1, pol_s1_p13_1,
        "Scenario 1B: Eight-Agent Bottleneck (Phase 13.1 Configuration)",
        "8 agents with priority rotation and basic coordination (margin=0)",
        episodes=20, base_seed=100,
    )
    print(f"  S1B Phase 13.1: Overall={res_s1_p13_1.overall_completion_rate:.1f}%, Mean Agent={res_s1_p13_1.mean_per_agent_success_rate:.1f}%, Cols={res_s1_p13_1.total_collisions}")
    experiments_data.append(asdict(res_s1_p13_1))

    # 1C: Phase 13.2 Baseline (Doorstep queuing, multi-recharge utility, margin=1)
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
    res_s1_p13_2 = run_experiment_batch_phase13_3(
        cfg_s1_p13_2, pol_s1_p13_2,
        "Scenario 1C: Eight-Agent Bottleneck (Phase 13.2 Configuration)",
        "8 agents with doorstep queuing and resource commitments (margin=1)",
        episodes=20, base_seed=100,
    )
    print(f"  S1C Phase 13.2: Overall={res_s1_p13_2.overall_completion_rate:.1f}%, Mean Agent={res_s1_p13_2.mean_per_agent_success_rate:.1f}%, Cols={res_s1_p13_2.total_collisions}")
    experiments_data.append(asdict(res_s1_p13_2))

    # 1D: Phase 13.3 (Goal-cell yielding, corridor traversal, fair load balancing)
    cfg_s1_p13_3 = MultiAgentWorldConfig(
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
    pol_s1_p13_3 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_1}
    res_s1_p13_3 = run_experiment_batch_phase13_3(
        cfg_s1_p13_3, pol_s1_p13_3,
        "Scenario 1D: Eight-Agent Bottleneck (Phase 13.3 Resolved)",
        "8 agents with goal-cell yielding, corridor traversal, and robust load balancing",
        episodes=20, base_seed=100,
    )
    print(f"  S1D Phase 13.3: Overall={res_s1_p13_3.overall_completion_rate:.1f}%, Mean Agent={res_s1_p13_3.mean_per_agent_success_rate:.1f}%, Cols={res_s1_p13_3.total_collisions}, Steps={res_s1_p13_3.mean_steps_to_goal:.1f}")
    experiments_data.append(asdict(res_s1_p13_3))

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
    res_s2 = run_experiment_batch_phase13_3(
        cfg_s2, pol_s2,
        "Scenario 2: Four-Agent Cross Congestion Regression",
        "4-way intersection regression confirming 100% completion maintained",
        episodes=20, base_seed=100,
    )
    print(f"  S2 Intersection: Overall={res_s2.overall_completion_rate:.1f}%, Cols={res_s2.total_collisions}, Duration={res_s2.mean_episode_duration:.1f}")
    experiments_data.append(asdict(res_s2))

    # --------------------------------------------------------------------------
    # Scenario 3: Two-Agent Resource Competition (Lowest-ID vs Priority-Rotation vs Fair-Balanced)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 3] Two-Agent Resource Competition Arbitration Strategies...")
    agents_s3 = (
        AgentConfig(id="agent_1", initial_x=0, initial_y=1, initial_energy=2, target=Position(3, 3)),
        AgentConfig(id="agent_2", initial_x=1, initial_y=0, initial_energy=2, target=Position(3, 3)),
    )
    # 3A: Lowest-ID
    cfg_s3_low = MultiAgentWorldConfig(
        width=4, height=4,
        agents=agents_s3,
        resources=(Resource(Position(1, 1), 6, consumable=True),),
        max_energy=8,
        collision_rule="priority",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_s3 = {a.id: make_robust_factory(4, 4, 8, base_linear_learner) for a in agents_s3}
    res_s3_low = run_experiment_batch_phase13_3(
        cfg_s3_low, pol_s3,
        "Scenario 3A: Two-Agent Resource Competition (Lowest-ID)",
        "Consumable resource arbitration under legacy lowest_id (demonstrates 100% starvation for Agent 2)",
        episodes=30, base_seed=100,
    )

    # 3B: Priority Rotation
    cfg_s3_rot = MultiAgentWorldConfig(
        width=4, height=4,
        agents=agents_s3,
        resources=(Resource(Position(1, 1), 6, consumable=True),),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="priority_rotation",
        goal_criterion="individual",
    )
    res_s3_rot = run_experiment_batch_phase13_3(
        cfg_s3_rot, pol_s3,
        "Scenario 3B: Two-Agent Resource Competition (Priority-Rotation)",
        "Consumable resource arbitration under deterministic priority rotation (50/50 equitable access)",
        episodes=30, base_seed=100,
    )

    # 3C: Fair-Balanced (Multi-round arbitration & starvation compensation)
    cfg_s3_fair = MultiAgentWorldConfig(
        width=4, height=4,
        agents=agents_s3,
        resources=(Resource(Position(1, 1), 6, consumable=True),),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    res_s3_fair = run_experiment_batch_phase13_3(
        cfg_s3_fair, pol_s3,
        "Scenario 3C: Two-Agent Resource Competition (Fair-Balanced)",
        "Consumable resource arbitration under Fair-Balanced (balanced urgency and starvation compensation)",
        episodes=30, base_seed=100,
    )

    # 3D: Reusable Resource Contention (Both agents recharge and achieve 100% overall completion)
    cfg_s3_reusable = MultiAgentWorldConfig(
        width=4, height=4,
        agents=agents_s3,
        resources=(Resource(Position(1, 1), 6, consumable=False),),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        goal_criterion="individual",
    )
    res_s3_reusable = run_experiment_batch_phase13_3(
        cfg_s3_reusable, pol_s3,
        "Scenario 3D: Two-Agent Reusable Resource (Within-Episode Contention)",
        "Reusable resource shared within episodes; Fair-Balanced ensures both agents recharge",
        episodes=30, base_seed=100,
    )

    print(f"  S3A Lowest-ID:    Agent1={res_s3_low.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_low.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_low.jains_fairness_goal:.3f}")
    print(f"  S3B Rotation:     Agent1={res_s3_rot.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_rot.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_rot.jains_fairness_goal:.3f}")
    print(f"  S3C Fair:         Agent1={res_s3_fair.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_fair.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_fair.jains_fairness_goal:.3f}")
    print(f"  S3D Reusable:     Overall={res_s3_reusable.overall_completion_rate:.1f}%, Agent1={res_s3_reusable.per_agent_success_rate['agent_1']:.1f}%, Agent2={res_s3_reusable.per_agent_success_rate['agent_2']:.1f}%, Jain={res_s3_reusable.jains_fairness_goal:.3f}")
    experiments_data.append(asdict(res_s3_low))
    experiments_data.append(asdict(res_s3_rot))
    experiments_data.append(asdict(res_s3_fair))
    experiments_data.append(asdict(res_s3_reusable))

    # --------------------------------------------------------------------------
    # Scenario 4: New 8-Agent Layout with Relocated Resources
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 4] 8-Agent Layout with Relocated Resources...")
    agents_s4_p13_2 = [
        AgentConfig(id=f"agent_{i+1}", initial_x=i % 4, initial_y=i // 4, initial_energy=6, target=Position(7 - (i % 4), 7 - (i // 4)))
        for i in range(8)
    ]
    # 4A: Phase 13.2 baseline configuration (energy=6; agent_1 at distance 7 has 0% feasibility)
    cfg_s4_base = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s4_p13_2),
        resources=(Resource(Position(2, 5), 6, False), Resource(Position(5, 2), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s4_base = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s4_p13_2}
    res_s4_base = run_experiment_batch_phase13_3(
        cfg_s4_base, pol_s4_base,
        "Scenario 4A: 8-Agent Relocated Resources (Phase 13.2 Configuration)",
        "Resources at (2, 5) and (5, 2) with initial_energy=6 (Agent 1 distance=7 is geometrically infeasible)",
        episodes=20, base_seed=100,
    )

    # 4B: Feasible parameterization (energy=7 matching distance to relocated resources)
    agents_s4_feas = [
        AgentConfig(id=f"agent_{i+1}", initial_x=i % 4, initial_y=i // 4, initial_energy=7, target=Position(7 - (i % 4), 7 - (i // 4)))
        for i in range(8)
    ]
    cfg_s4_feas = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s4_feas),
        resources=(Resource(Position(2, 5), 6, False), Resource(Position(5, 2), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s4_feas = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s4_feas}
    res_s4_feas = run_experiment_batch_phase13_3(
        cfg_s4_feas, pol_s4_feas,
        "Scenario 4B: 8-Agent Relocated Resources (Feasible Parameterization)",
        "Resources at (2, 5) and (5, 2) with initial_energy=7 ensuring all agents have feasible paths",
        episodes=20, base_seed=100,
    )
    print(f"  S4A Baseline: Overall={res_s4_base.overall_completion_rate:.1f}%, Mean Agent={res_s4_base.mean_per_agent_success_rate:.1f}%, Agent1 Succ={res_s4_base.per_agent_success_rate['agent_1']:.1f}%")
    print(f"  S4B Feasible: Overall={res_s4_feas.overall_completion_rate:.1f}%, Mean Agent={res_s4_feas.mean_per_agent_success_rate:.1f}%, Agent1 Succ={res_s4_feas.per_agent_success_rate['agent_1']:.1f}%")
    experiments_data.append(asdict(res_s4_base))
    experiments_data.append(asdict(res_s4_feas))

    # --------------------------------------------------------------------------
    # Scenario 5: 8-Agent Layout with Asymmetric Starting Positions
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 5] 8-Agent Layout with Asymmetric Starting Positions...")
    asym_starts = [
        (0, 0, 7, 7), (0, 7, 7, 0), (7, 0, 0, 7), (7, 7, 0, 0),
        (1, 0, 6, 7), (0, 6, 7, 1), (6, 7, 1, 0), (7, 1, 0, 6),
    ]
    # 5A: Phase 13.2 baseline configuration (energy=6; agents 2 and 3 distance=7 are geometrically infeasible)
    agents_s5_base = [
        AgentConfig(id=f"agent_{i+1}", initial_x=sx, initial_y=sy, initial_energy=6, target=Position(gx, gy))
        for i, (sx, sy, gx, gy) in enumerate(asym_starts)
    ]
    cfg_s5_base = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s5_base),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s5_base = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s5_base}
    res_s5_base = run_experiment_batch_phase13_3(
        cfg_s5_base, pol_s5_base,
        "Scenario 5A: 8-Agent Asymmetric Starts (Phase 13.2 Configuration)",
        "Opposing corners funneling into center; agents 2 & 3 have distance=7 > energy=6",
        episodes=20, base_seed=100,
    )

    # 5B: Feasible parameterization (energy=8; all agents can reach central resources)
    agents_s5_feas = [
        AgentConfig(id=f"agent_{i+1}", initial_x=sx, initial_y=sy, initial_energy=8, target=Position(gx, gy))
        for i, (sx, sy, gx, gy) in enumerate(asym_starts)
    ]
    cfg_s5_feas = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_s5_feas),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_s5_feas = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_s5_feas}
    res_s5_feas = run_experiment_batch_phase13_3(
        cfg_s5_feas, pol_s5_feas,
        "Scenario 5B: 8-Agent Asymmetric Starts (Feasible Parameterization)",
        "Opposing corners with initial_energy=8 ensuring all agents reach central resources",
        episodes=20, base_seed=100,
    )
    print(f"  S5A Baseline: Overall={res_s5_base.overall_completion_rate:.1f}%, Mean Agent={res_s5_base.mean_per_agent_success_rate:.1f}%, Exhaustions={sum(res_s5_base.exhaustion_counts.values())}")
    print(f"  S5B Feasible: Overall={res_s5_feas.overall_completion_rate:.1f}%, Mean Agent={res_s5_feas.mean_per_agent_success_rate:.1f}%, Exhaustions={sum(res_s5_feas.exhaustion_counts.values())}")
    experiments_data.append(asdict(res_s5_base))
    experiments_data.append(asdict(res_s5_feas))

    # --------------------------------------------------------------------------
    # Scenario 6: Completed Agents Blocking Terminal Cells / Narrow Corridors
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 6] Goal-Cell Yielding in Narrow Corridor / Terminal Cells...")
    cfg_s6 = MultiAgentWorldConfig(
        width=6, height=1,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=10, target=Position(5, 0)),
            AgentConfig(id="agent_2", initial_x=2, initial_y=0, initial_energy=10, target=Position(4, 0)),
        ),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        goal_criterion="individual",
    )
    pol_s6 = {a.id: make_robust_factory(6, 1, 10, base_linear_learner) for a in cfg_s6.agents}
    res_s6 = run_experiment_batch_phase13_3(
        cfg_s6, pol_s6,
        "Scenario 6: Completed Agent Goal-Cell Yielding (Narrow Corridor)",
        "1-wide corridor where Agent 2 finishes at (4, 0) directly in the path to Agent 1's goal at (5, 0)",
        episodes=20, base_seed=100,
    )
    print(f"  S6 Corridor:  Overall={res_s6.overall_completion_rate:.1f}%, Steps={res_s6.mean_steps_to_goal:.1f}, Occupied Cols={res_s6.occupied_collisions}")
    experiments_data.append(asdict(res_s6))

    # --------------------------------------------------------------------------
    # Scenario 7: Deadlock and Oscillation Stress Test (Head-On Swap)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 7] Deadlock & Head-On Swap Conflict Resolution...")
    cfg_s7 = MultiAgentWorldConfig(
        width=5, height=3,
        agents=(
            AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=10, target=Position(3, 1)),
            AgentConfig(id="agent_2", initial_x=3, initial_y=1, initial_energy=10, target=Position(1, 1)),
        ),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        enable_coordination=True,
        goal_criterion="individual",
    )
    pol_s7 = {a.id: make_robust_factory(5, 3, 10, base_linear_learner) for a in cfg_s7.agents}
    res_s7 = run_experiment_batch_phase13_3(
        cfg_s7, pol_s7,
        "Scenario 7: Deadlock and Oscillation Stress Test",
        "Two agents starting on opposite sides of a narrow corridor heading into head-on swap",
        episodes=20, base_seed=100,
    )
    print(f"  S7 Deadlock:  Overall={res_s7.overall_completion_rate:.1f}%, Deadlocks={res_s7.deadlock_detections}, Swaps={res_s7.swap_collisions}")
    experiments_data.append(asdict(res_s7))

    # --------------------------------------------------------------------------
    # Scenario 8: Genuinely Infeasible Environment (Honest Failure Accounting)
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 8] Genuinely Infeasible Environment (Honest Accounting)...")
    cfg_s8 = MultiAgentWorldConfig(
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
    pol_s8 = {a.id: make_robust_factory(8, 8, 4, base_linear_learner) for a in cfg_s8.agents}
    res_s8 = run_experiment_batch_phase13_3(
        cfg_s8, pol_s8,
        "Scenario 8: Intentionally Infeasible Environment",
        "Overconstrained parameters where target and resource are mathematically unreachable",
        episodes=20, base_seed=100,
    )
    print(f"  S8 Infeasible: Overall={res_s8.overall_completion_rate:.1f}%, Mean Agent={res_s8.mean_per_agent_success_rate:.1f}%, Exhaustion={res_s8.exhaustion_rates['agent_1']:.1f}%")
    experiments_data.append(asdict(res_s8))

    # --------------------------------------------------------------------------
    # Scenario 9: Single-Agent Baseline Regression
    # --------------------------------------------------------------------------
    print("\n[Running Scenario 9] Single-Agent Baseline Regression...")
    cfg_s9 = MultiAgentWorldConfig(
        width=6, height=6,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=5, target=Position(5, 5)),
        ),
        resources=(Resource(Position(2, 2), 6, consumable=False),),
        max_energy=8,
        collision_rule="bounce",
        goal_criterion="individual",
    )
    pol_s9 = {a.id: make_robust_factory(6, 6, 8, base_linear_learner) for a in cfg_s9.agents}
    res_s9 = run_experiment_batch_phase13_3(
        cfg_s9, pol_s9,
        "Scenario 9: Single-Agent Baseline Regression",
        "Single-agent baseline verifying 100% backward compatibility and regression prevention",
        episodes=20, base_seed=100,
    )
    print(f"  S9 Single-Agent: Overall={res_s9.overall_completion_rate:.1f}%, Steps={res_s9.mean_steps_to_goal:.1f}")
    experiments_data.append(asdict(res_s9))

    # Save to JSON
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"version": "13.3", "scenarios": experiments_data}, f, indent=2)

    print(f"\nSaved machine-readable Phase 13.3 benchmark results to {out_file.resolve()}")
    return {"version": "13.3", "scenarios": experiments_data}


if __name__ == "__main__":
    run_all_phase13_3_experiments()
