"""WorldSim Phase 13.1: Multi-Agent Coordination, Resource Allocation, and Congestion Control Benchmark.

Implements all 10 controlled experiments (Exp A through Exp J) and 4 ablation studies:
- Exp A: Single-agent regression verification
- Exp B: Two-agent movement conflict (swap & blocking comparison)
- Exp C: Four-agent cross-intersection congestion
- Exp D: Eight-agent scaling (investigating root cause of Phase 13 congestion)
- Exp E: Fair resource allocation (lowest-ID vs fair balanced arbitration)
- Exp F: Shared goals (independent vs coordinated)
- Exp G: Cooperative discovery sharing (isolated vs shared)
- Exp H: Starvation monitoring & lower-priority access
- Exp I: Unsolvable / over-constrained environment (honest failure accounting)
- Exp J: Generalization across 3 distinct layouts (J1, J2, J3)
- Ablation Suite on Exp D (Baseline, Congestion Only, Allocation Only, Full 13.1)
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
class Phase13_1ExperimentMetrics:
    """Detailed quantitative metrics for Phase 13.1 multi-agent experiments."""

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
    cooperative_completion_time: float | None


def run_phase13_1_episode(
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
            sim.coordinator.reset()

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

        if res.overall_completed:
            break

    consumed = {aid: sim.fairness_monitor.metrics[aid].total_energy_consumed for aid in sim.state.agents}
    overall_completed = sim._check_overall_completion()
    return overall_completed, step_count, reached, consumed, recharges


def run_experiment_batch_phase13_1(
    config: MultiAgentWorldConfig,
    policy_factories: dict[str, Any],
    experiment_name: str,
    description: str,
    episodes: int = 50,
    max_steps: int = 50,
) -> Phase13_1ExperimentMetrics:
    """Execute a batch of episodes and collect comprehensive Phase 13.1 metrics."""
    agent_ids = tuple(a.id for a in config.agents)
    per_agent_success_counts = {aid: 0 for aid in agent_ids}
    overall_success_count = 0
    durations: list[int] = []
    success_steps: list[int] = []
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

    for ep in range(episodes):
        sim = MultiAgentSimulation(config, seed=ep + 100)
        policies = {aid: factory() for aid, factory in policy_factories.items()}
        if config.enable_coordination:
            for p in policies.values():
                if hasattr(p, "coordinator") and p.coordinator is None:
                    p.coordinator = sim.coordinator
                    if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                        p.utility_evaluator.coordinator = sim.coordinator

        comp, steps, reached, consumed, recharges = run_phase13_1_episode(sim, policies, max_steps=max_steps)

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

    return Phase13_1ExperimentMetrics(
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
        jains_fairness_goal=round(mean_goal_fairness, 3),
        jains_fairness_energy=round(mean_energy_fairness, 3),
        cooperative_completion_time=round(mean_succ_steps, 2) if overall_succ_rate > 0 else None,
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


def run_all_phase13_1_experiments(output_path: str = "benchmark_results_phase13_1.json") -> dict[str, Any]:
    """Execute all Phase 13.1 controlled experiments and ablations, saving results to JSON."""
    print("=" * 80)
    print("WORLDSIM PHASE 13.1 BENCHMARK: COORDINATION, RESOURCE ALLOCATION & CONGESTION")
    print("=" * 80)

    # Train base Linear Q learner on Train-1
    train_world = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
    )
    lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
    lin_runner.train(1000)
    base_linear_learner = lin_runner.learner

    experiments_data = []

    # --------------------------------------------------------------------------
    # Experiment A: Single-Agent Regression Verification
    # --------------------------------------------------------------------------
    print("\n[Running Experiment A] Single-Agent Regression (Baseline Verification)...")
    cfg_a = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=3, target=Position(3, 3)),),
        resources=(Resource(Position(1, 0), 4, True),),
        max_energy=6,
    )
    pol_a = {"agent_1": make_robust_factory(4, 4, 6, base_linear_learner)}
    res_a = run_experiment_batch_phase13_1(cfg_a, pol_a, "Experiment A: Single-Agent Regression", "1-agent baseline matching Phase 12.2 and Phase 13", episodes=50)
    print(f"  Exp A Success: {res_a.overall_completion_rate:.1f}% | Steps: {res_a.mean_steps_to_goal:.1f} | Cols: {res_a.total_collisions}")
    experiments_data.append(asdict(res_a))

    # --------------------------------------------------------------------------
    # Experiment B: Two-Agent Movement Conflict (Swap & Collision Resolution)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment B] Two-Agent Movement Conflict (Swap & Blocking)...")
    cfg_b_baseline = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=5, target=Position(2, 1)),
            AgentConfig(id="agent_2", initial_x=2, initial_y=1, initial_energy=5, target=Position(1, 1)),
        ),
        resources=(),
        max_energy=6,
        collision_rule="bounce",
        goal_criterion="individual",
    )
    pol_b = {
        "agent_1": make_robust_factory(4, 4, 6, base_linear_learner),
        "agent_2": make_robust_factory(4, 4, 6, base_linear_learner),
    }
    res_b_base = run_experiment_batch_phase13_1(cfg_b_baseline, pol_b, "Experiment B1: Movement Conflict (Phase 13 Baseline)", "Head-on swap with symmetric bounce", episodes=50)

    cfg_b_improved = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=5, target=Position(2, 1)),
            AgentConfig(id="agent_2", initial_x=2, initial_y=1, initial_energy=5, target=Position(1, 1)),
        ),
        resources=(),
        max_energy=6,
        collision_rule="priority_rotation",
        goal_criterion="individual",
    )
    res_b_imp = run_experiment_batch_phase13_1(cfg_b_improved, pol_b, "Experiment B2: Movement Conflict (Phase 13.1 Improved)", "Head-on swap with priority rotation and blocked-cell avoidance", episodes=50)
    print(f"  Exp B Base: Succ={res_b_base.overall_completion_rate:.1f}%, Cols={res_b_base.total_collisions} | Imp: Succ={res_b_imp.overall_completion_rate:.1f}%, Cols={res_b_imp.total_collisions}")
    experiments_data.append(asdict(res_b_base))
    experiments_data.append(asdict(res_b_imp))

    # --------------------------------------------------------------------------
    # Experiment C: Four-Agent Congestion & Chokepoints
    # --------------------------------------------------------------------------
    print("\n[Running Experiment C] Four-Agent Cross Congestion...")
    cfg_c_base = MultiAgentWorldConfig(
        width=6, height=6,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=2, initial_energy=5, target=Position(5, 2)),
            AgentConfig(id="agent_2", initial_x=5, initial_y=3, initial_energy=5, target=Position(0, 3)),
            AgentConfig(id="agent_3", initial_x=2, initial_y=0, initial_energy=5, target=Position(2, 5)),
            AgentConfig(id="agent_4", initial_x=3, initial_y=5, initial_energy=5, target=Position(3, 0)),
        ),
        resources=(Resource(Position(2, 2), 6, False), Resource(Position(3, 3), 6, False)),
        max_energy=8,
        collision_rule="bounce",
        goal_criterion="individual",
    )
    pol_c_base = {a.id: make_robust_factory(6, 6, 8, base_linear_learner) for a in cfg_c_base.agents}
    res_c_base = run_experiment_batch_phase13_1(cfg_c_base, pol_c_base, "Experiment C1: Four-Agent Congestion (Phase 13 Baseline)", "4-way intersection under symmetric bounce", episodes=30)

    cfg_c_imp = MultiAgentWorldConfig(
        width=6, height=6,
        agents=cfg_c_base.agents,
        resources=cfg_c_base.resources,
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        goal_criterion="individual",
    )
    pol_c_imp = {a.id: make_robust_factory(6, 6, 8, base_linear_learner, margin=1) for a in cfg_c_imp.agents}
    res_c_imp = run_experiment_batch_phase13_1(cfg_c_imp, pol_c_imp, "Experiment C2: Four-Agent Congestion (Phase 13.1 Improved)", "4-way intersection under priority rotation & fair allocation", episodes=30)
    print(f"  Exp C Base: Succ={res_c_base.overall_completion_rate:.1f}%, Cols={res_c_base.total_collisions} | Imp: Succ={res_c_imp.overall_completion_rate:.1f}%, Cols={res_c_imp.total_collisions}")
    experiments_data.append(asdict(res_c_base))
    experiments_data.append(asdict(res_c_imp))

    # --------------------------------------------------------------------------
    # Experiment D: Eight-Agent Scaling (Root Cause Investigation)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment D] Eight-Agent Bottleneck Scaling...")
    agents_d = []
    for i in range(8):
        ax = i % 4
        ay = i // 4
        tx = 7 - ax
        ty = 7 - ay
        agents_d.append(AgentConfig(id=f"agent_{i+1}", initial_x=ax, initial_y=ay, initial_energy=6, target=Position(tx, ty)))

    cfg_d_base = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_d),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="bounce",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_d_base = {a.id: make_robust_factory(8, 8, 10, base_linear_learner) for a in agents_d}
    res_d_base = run_experiment_batch_phase13_1(cfg_d_base, pol_d_base, "Experiment D1: Eight-Agent Scaling (Phase 13 Baseline)", "8 agents in 4x2 block funneling into central resources (pure bounce)", episodes=20)

    cfg_d_imp = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_d),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_d_imp = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in agents_d}
    res_d_imp = run_experiment_batch_phase13_1(cfg_d_imp, pol_d_imp, "Experiment D2: Eight-Agent Scaling (Phase 13.1 Improved)", "8 agents with priority rotation, fair resource sharing, and congestion avoidance", episodes=20)
    print(f"  Exp D Base: Succ={res_d_base.overall_completion_rate:.1f}%, Cols={res_d_base.total_collisions} | Imp: Succ={res_d_imp.overall_completion_rate:.1f}%, Cols={res_d_imp.total_collisions}")
    experiments_data.append(asdict(res_d_base))
    experiments_data.append(asdict(res_d_imp))

    # --------------------------------------------------------------------------
    # Experiment E: Fair Resource Allocation
    # --------------------------------------------------------------------------
    print("\n[Running Experiment E] Fair Resource Allocation...")
    cfg_e_lowest = MultiAgentWorldConfig(
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
    pol_e = {
        "agent_1": make_robust_factory(4, 4, 8, base_linear_learner),
        "agent_2": make_robust_factory(4, 4, 8, base_linear_learner),
    }
    res_e_lowest = run_experiment_batch_phase13_1(cfg_e_lowest, pol_e, "Experiment E1: Resource Competition (Lowest-ID Baseline)", "Equidistant competition with lowest-ID priority", episodes=30)

    cfg_e_fair = MultiAgentWorldConfig(
        width=4, height=4,
        agents=cfg_e_lowest.agents,
        resources=cfg_e_lowest.resources,
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    res_e_fair = run_experiment_batch_phase13_1(cfg_e_fair, pol_e, "Experiment E2: Resource Competition (Fair Balanced Arbitration)", "Equidistant competition with fair balanced arbitration", episodes=30)
    print(f"  Exp E Lowest-ID: Per-agent={res_e_lowest.per_agent_success_rate}, Jain={res_e_lowest.jains_fairness_energy:.2f} | Fair: Per-agent={res_e_fair.per_agent_success_rate}, Jain={res_e_fair.jains_fairness_energy:.2f}")
    experiments_data.append(asdict(res_e_lowest))
    experiments_data.append(asdict(res_e_fair))

    # --------------------------------------------------------------------------
    # Experiment F: Shared Goals (Independent vs Coordinated)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment F] Shared Goals...")
    cfg_f = MultiAgentWorldConfig(
        width=6, height=6,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=5, target=Position(5, 5)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=5, initial_energy=5, target=Position(5, 5)),
        ),
        resources=(Resource(Position(2, 2), 6, False),),
        max_energy=8,
        goal_criterion="shared_any",
    )
    pol_f = {a.id: make_robust_factory(6, 6, 8, base_linear_learner) for a in cfg_f.agents}
    res_f = run_experiment_batch_phase13_1(cfg_f, pol_f, "Experiment F: Shared Goals (Any-Agent Criterion)", "Any-agent arrival completes shared goal", episodes=30)
    print(f"  Exp F Completion: {res_f.overall_completion_rate:.1f}% | Steps: {res_f.mean_steps_to_goal:.1f}")
    experiments_data.append(asdict(res_f))

    # --------------------------------------------------------------------------
    # Experiment G: Cooperative Discovery Sharing
    # --------------------------------------------------------------------------
    print("\n[Running Experiment G] Cooperative Discovery Sharing...")
    cfg_g_iso = MultiAgentWorldConfig(
        width=8, height=8,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=2, initial_energy=4, target=Position(7, 2)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=6, initial_energy=4, target=Position(7, 6)),
        ),
        resources=(Resource(Position(1, 3), 10, False),),
        max_energy=10,
        cooperative_sharing=False,
        goal_criterion="individual",
    )
    pol_g = {
        "agent_1": make_robust_factory(8, 8, 10, base_linear_learner),
        "agent_2": make_robust_factory(8, 8, 10, base_linear_learner),
    }
    res_g_iso = run_experiment_batch_phase13_1(cfg_g_iso, pol_g, "Experiment G1: Isolated Exploration", "Sensor-isolated exploration without sharing", episodes=30)

    cfg_g_shared = MultiAgentWorldConfig(
        width=8, height=8,
        agents=cfg_g_iso.agents,
        resources=cfg_g_iso.resources,
        max_energy=10,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    res_g_shared = run_experiment_batch_phase13_1(cfg_g_shared, pol_g, "Experiment G2: Cooperative Sharing", "Cooperative discovery sharing pool", episodes=30)
    print(f"  Exp G Isolated: {res_g_iso.overall_completion_rate:.1f}% vs Shared: {res_g_shared.overall_completion_rate:.1f}%")
    experiments_data.append(asdict(res_g_iso))
    experiments_data.append(asdict(res_g_shared))

    # --------------------------------------------------------------------------
    # Experiment H: Starvation Monitoring & Opportunity Guarantee
    # --------------------------------------------------------------------------
    print("\n[Running Experiment H] Starvation Monitoring...")
    cfg_h = MultiAgentWorldConfig(
        width=5, height=5,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=4, target=Position(4, 4)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=1, initial_energy=3, target=Position(4, 4)),
            AgentConfig(id="agent_3", initial_x=0, initial_y=2, initial_energy=2, target=Position(4, 4)),
        ),
        resources=(Resource(Position(2, 2), 6, False),),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    pol_h = {a.id: make_robust_factory(5, 5, 8, base_linear_learner, margin=1) for a in cfg_h.agents}
    res_h = run_experiment_batch_phase13_1(cfg_h, pol_h, "Experiment H: Starvation Monitoring", "3 agents competing for single bottleneck resource under fair arbitration", episodes=30)
    print(f"  Exp H Starvation Events: {res_h.starvation_events} | Recoveries: {res_h.successful_recoveries} | Jain Goal: {res_h.jains_fairness_goal:.2f}")
    experiments_data.append(asdict(res_h))

    # --------------------------------------------------------------------------
    # Experiment I: Unsolvable / Over-Constrained Scenarios
    # --------------------------------------------------------------------------
    print("\n[Running Experiment I] Over-Constrained / Impossible Scenarios...")
    cfg_i = MultiAgentWorldConfig(
        width=8, height=8,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=2, target=Position(7, 7)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=7, initial_energy=2, target=Position(7, 0)),
        ),
        resources=(),  # No resources available: provably impossible
        max_energy=6,
        goal_criterion="individual",
    )
    pol_i = {a.id: make_robust_factory(8, 8, 6, base_linear_learner) for a in cfg_i.agents}
    res_i = run_experiment_batch_phase13_1(cfg_i, pol_i, "Experiment I: Provably Infeasible Multi-Agent", "Insufficient energy with zero resources: honest exhaustion verification", episodes=30)
    print(f"  Exp I Success Rate: {res_i.overall_completion_rate:.1f}% | Exhaustion: {res_i.exhaustion_rates}")
    experiments_data.append(asdict(res_i))

    # --------------------------------------------------------------------------
    # Experiment J: Generalization Across Distinct Layouts
    # --------------------------------------------------------------------------
    print("\n[Running Experiment J] Generalization Across 3 Distinct Layouts...")

    # J1: Asymmetric 6x6 Crossway (3 Agents)
    cfg_j1 = MultiAgentWorldConfig(
        width=6, height=6,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=1, initial_energy=5, target=Position(5, 4)),
            AgentConfig(id="agent_2", initial_x=5, initial_y=1, initial_energy=5, target=Position(0, 4)),
            AgentConfig(id="agent_3", initial_x=2, initial_y=0, initial_energy=5, target=Position(2, 5)),
        ),
        resources=(Resource(Position(1, 3), 6, False), Resource(Position(4, 3), 6, False)),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    pol_j1 = {a.id: make_robust_factory(6, 6, 8, base_linear_learner, margin=1) for a in cfg_j1.agents}
    res_j1 = run_experiment_batch_phase13_1(cfg_j1, pol_j1, "Experiment J1: Asymmetric 6x6 Crossway", "3 agents navigating crossing paths with 2 offset resources", episodes=30)
    print(f"  Exp J1 (Crossway): Success={res_j1.overall_completion_rate:.1f}%, Cols={res_j1.total_collisions}")
    experiments_data.append(asdict(res_j1))

    # J2: 7x7 Perimeter Corridor (4 Agents)
    cfg_j2 = MultiAgentWorldConfig(
        width=7, height=7,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=8, target=Position(6, 6)),
            AgentConfig(id="agent_2", initial_x=6, initial_y=0, initial_energy=8, target=Position(0, 6)),
            AgentConfig(id="agent_3", initial_x=0, initial_y=6, initial_energy=8, target=Position(6, 0)),
            AgentConfig(id="agent_4", initial_x=6, initial_y=6, initial_energy=8, target=Position(0, 0)),
        ),
        resources=(Resource(Position(3, 3), 8, False),),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_j2 = {a.id: make_robust_factory(7, 7, 10, base_linear_learner, margin=1) for a in cfg_j2.agents}
    res_j2 = run_experiment_batch_phase13_1(cfg_j2, pol_j2, "Experiment J2: 7x7 Perimeter Diagonal", "4 corner agents crossing opposite corners via central resource", episodes=30)
    print(f"  Exp J2 (Perimeter Diagonal): Success={res_j2.overall_completion_rate:.1f}%, Cols={res_j2.total_collisions}")
    experiments_data.append(asdict(res_j2))

    # J3: 8x8 Dispersed Multi-Resource Layout (6 Agents)
    cfg_j3 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=7, target=Position(7, 2)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=3, initial_energy=7, target=Position(7, 5)),
            AgentConfig(id="agent_3", initial_x=0, initial_y=7, initial_energy=7, target=Position(7, 7)),
            AgentConfig(id="agent_4", initial_x=7, initial_y=0, initial_energy=7, target=Position(0, 2)),
            AgentConfig(id="agent_5", initial_x=7, initial_y=3, initial_energy=7, target=Position(0, 5)),
            AgentConfig(id="agent_6", initial_x=7, initial_y=7, initial_energy=7, target=Position(0, 7)),
        ),
        resources=(
            Resource(Position(2, 2), 6, False),
            Resource(Position(4, 4), 6, False),
            Resource(Position(5, 5), 6, False),
        ),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    pol_j3 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner, margin=1) for a in cfg_j3.agents}
    res_j3 = run_experiment_batch_phase13_1(cfg_j3, pol_j3, "Experiment J3: 8x8 Dispersed Multi-Resource", "6 opposing agents with 3 dispersed resources", episodes=30)
    print(f"  Exp J3 (Dispersed): Success={res_j3.overall_completion_rate:.1f}%, Cols={res_j3.total_collisions}")
    experiments_data.append(asdict(res_j3))

    # --------------------------------------------------------------------------
    # Ablation Studies (on 8-Agent Scaling Configuration)
    # --------------------------------------------------------------------------
    print("\n[Running Ablation Studies on 8-Agent Scenario]...")

    # Ablation 1: Phase 13 Baseline (Pure Bounce, Lowest-ID, No Coordination)
    res_abl1 = res_d_base

    # Ablation 2: Congestion Handling Only (Priority Rotation, Lowest-ID Allocation)
    cfg_abl2 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_d),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="lowest_id",
        goal_criterion="individual",
    )
    pol_abl2 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner) for a in agents_d}
    res_abl2 = run_experiment_batch_phase13_1(cfg_abl2, pol_abl2, "Ablation 2: Congestion Handling Only", "Priority rotation with lowest-ID resource allocation", episodes=20)
    print(f"  Ablation 2 (Congestion Only): Success={res_abl2.overall_completion_rate:.1f}%, Cols={res_abl2.total_collisions}")
    experiments_data.append(asdict(res_abl2))

    # Ablation 3: Resource Allocation Only (Symmetric Bounce, Fair Balanced Allocation)
    cfg_abl3 = MultiAgentWorldConfig(
        width=8, height=8,
        agents=tuple(agents_d),
        resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
        max_energy=10,
        collision_rule="bounce",
        resource_allocation_rule="fair_balanced",
        goal_criterion="individual",
    )
    pol_abl3 = {a.id: make_robust_factory(8, 8, 10, base_linear_learner) for a in agents_d}
    res_abl3 = run_experiment_batch_phase13_1(cfg_abl3, pol_abl3, "Ablation 3: Fair Resource Allocation Only", "Symmetric bounce with fair balanced resource allocation", episodes=20)
    print(f"  Ablation 3 (Resource Alloc Only): Success={res_abl3.overall_completion_rate:.1f}%, Cols={res_abl3.total_collisions}")
    experiments_data.append(asdict(res_abl3))

    # Ablation 4: Full Phase 13.1
    res_abl4 = res_d_imp

    # Save to JSON
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"version": "13.1", "experiments": experiments_data}, f, indent=2)

    print(f"\nSaved machine-readable Phase 13.1 benchmark results to {out_file.resolve()}")
    return {"version": "13.1", "experiments": experiments_data}


if __name__ == "__main__":
    run_all_phase13_1_experiments()
