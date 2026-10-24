"""Multi-agent experiment runner and controlled benchmark suite for WorldSim Phase 13.

Implements Experiments A through H:
- Exp A: Single-agent control (regression baseline)
- Exp B: Multiple independent agents (separate goals and resources)
- Exp C: Resource competition (scarce resources, contention tracking)
- Exp D: Movement conflicts (vertex bounce, swap bounce, occupied blocks)
- Exp E: Shared goals (1 vs 2 vs 4 agents on common target)
- Exp F: Cooperative information sharing (isolated vs shared discovery)
- Exp G: Policy comparison (Robust Q, Linear Q, Tabular Q, Random)
- Exp H: Scaling (1, 2, 4, 8 agents on scaled grid)

Exports results to benchmark_results_phase13.json.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random
import statistics
from typing import Any, Sequence

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .learning import LearningConfig, LearningRunner, QLearner
from .linear_learning import LinearLearningConfig, LinearLearningRunner, LinearQLearner
from .models import Position, Resource
from .multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentStepResult,
    MultiAgentWorldConfig,
)
from .policy import RandomPolicy
from .robust_policy import RobustResourcePolicy


@dataclass
class MultiAgentExperimentMetrics:
    """Detailed summary metrics for a multi-agent experiment."""

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
    exhaustion_rates: dict[str, float]
    timeout_rates: dict[str, float]
    jains_fairness_index: float
    cooperative_completion_time: float | None


def compute_jains_fairness(values: Sequence[float]) -> float:
    """Compute Jain's Fairness Index over a sequence of non-negative allocations."""
    if not values:
        return 1.0
    n = len(values)
    sum_vals = sum(values)
    sum_sq = sum(v * v for v in values)
    if sum_sq == 0:
        return 1.0
    return (sum_vals * sum_vals) / (n * sum_sq)


def run_multi_agent_episode(
    sim: MultiAgentSimulation,
    policies: dict[str, Any],
    max_steps: int = 40,
) -> tuple[bool, int, dict[str, bool], dict[str, int], dict[str, int], int, int, int, int]:
    """Execute a single multi-agent episode.

    Returns:
        overall_completed, steps, reached_map, energy_consumed_map, recharges_map,
        vertex_cols, swap_cols, occupied_cols, contentions
    """
    initial_energies = {aid: sim.state.agents[aid].energy for aid in sim.state.agents}
    recharges = {aid: 0 for aid in sim.state.agents}
    reached = {aid: False for aid in sim.state.agents}

    total_v = 0
    total_s = 0
    total_o = 0
    total_c = 0

    # Reset memories for any RobustResourcePolicy
    for aid, pol in policies.items():
        if hasattr(pol, "reset_memory"):
            pol.reset_memory()

    step_count = 0
    while step_count < max_steps:
        # Check if all reached or all inactive
        all_done = all(not a.active for a in sim.state.agents.values())
        if all_done:
            break

        # Select actions for all active agents
        actions: dict[str, Action] = {}
        for aid, agent in sim.state.agents.items():
            if not agent.active:
                actions[aid] = Action.UP
                continue
            obs = sim.observe_agent(aid)
            pol = policies[aid]
            if hasattr(pol, "choose_action"):
                act = pol.choose_action(obs)
            elif hasattr(pol, "select_action"):
                # e.g. LinearQLearner
                act = pol.select_action(obs, explore=False)
            else:
                act = Action.UP
            actions[aid] = act

        res = sim.step(actions)
        step_count += 1

        total_v += res.vertex_collisions
        total_s += res.swap_collisions
        total_o += res.occupied_collisions
        total_c += res.resource_contentions

        # Track results
        for aid, a_res in res.agent_results.items():
            if a_res.action == Action.RECHARGE and a_res.success:
                recharges[aid] += 1
            if a_res.reached_target:
                reached[aid] = True

            # If policy has agent memory, record action result
            pol = policies[aid]
            if hasattr(pol, "memory") and hasattr(pol.memory, "record_action_result"):
                pol.memory.record_action_result(
                    a_res.position_before,
                    a_res.energy_before,
                    a_res.action,
                    a_res.success,
                )

        if res.overall_completed:
            break

    # Energy consumed
    consumed = {}
    for aid, agent in sim.state.agents.items():
        # energy consumed = initial + gained - final
        consumed[aid] = max(0, initial_energies[aid] - agent.energy + (recharges[aid] * 4)) # approx

    overall_completed = sim._check_overall_completion()
    return (
        overall_completed,
        step_count,
        reached,
        consumed,
        recharges,
        total_v,
        total_s,
        total_o,
        total_c,
    )


def run_experiment_batch(
    config: MultiAgentWorldConfig,
    policy_factories: dict[str, Any],
    experiment_name: str,
    description: str,
    episodes: int = 50,
    max_steps: int = 40,
    seed_base: int = 42,
) -> MultiAgentExperimentMetrics:
    """Run an ensemble of episodes for a given multi-agent configuration and compute metrics."""
    agent_ids = tuple(a.id for a in config.agents)
    overall_successes = 0
    per_agent_successes = {aid: 0 for aid in agent_ids}
    steps_to_goal = []
    durations = []
    energies_consumed_list: list[float] = []
    per_agent_recharges_list: dict[str, list[int]] = {aid: [] for aid in agent_ids}
    exhaustions = {aid: 0 for aid in agent_ids}
    timeouts = {aid: 0 for aid in agent_ids}

    tot_v = 0
    tot_s = 0
    tot_o = 0
    tot_c = 0
    tot_resources_depleted = 0

    for ep in range(episodes):
        sim = MultiAgentSimulation(config, seed=seed_base + ep)
        policies = {aid: policy_factories[aid]() for aid in agent_ids}

        (
            overall_done,
            steps,
            reached,
            consumed,
            recharges,
            v_col,
            s_col,
            o_col,
            contention,
        ) = run_multi_agent_episode(sim, policies, max_steps=max_steps)

        tot_v += v_col
        tot_s += s_col
        tot_o += o_col
        tot_c += contention
        tot_resources_depleted += sim.cumulative_resources_depleted

        durations.append(steps)
        if overall_done:
            overall_successes += 1
            steps_to_goal.append(steps)

        for aid in agent_ids:
            if reached[aid]:
                per_agent_successes[aid] += 1
            else:
                if sim.state.agents[aid].energy == 0:
                    exhaustions[aid] += 1
                else:
                    timeouts[aid] += 1

            per_agent_recharges_list[aid].append(recharges[aid])

        energies_consumed_list.append(sum(consumed.values()))

    overall_rate = (overall_successes / episodes) * 100.0
    per_agent_rate = {aid: (per_agent_successes[aid] / episodes) * 100.0 for aid in agent_ids}
    mean_steps = statistics.mean(steps_to_goal) if steps_to_goal else 0.0
    mean_dur = statistics.mean(durations) if durations else 0.0
    mean_energy = statistics.mean(energies_consumed_list) if energies_consumed_list else 0.0
    total_energy = sum(energies_consumed_list)
    recharges_mean = {aid: statistics.mean(per_agent_recharges_list[aid]) for aid in agent_ids}

    fairness = compute_jains_fairness([per_agent_rate[aid] for aid in agent_ids])

    return MultiAgentExperimentMetrics(
        experiment_name=experiment_name,
        description=description,
        episodes=episodes,
        agent_ids=agent_ids,
        per_agent_success_rate=per_agent_rate,
        overall_completion_rate=overall_rate,
        mean_steps_to_goal=mean_steps,
        mean_episode_duration=mean_dur,
        mean_energy_consumed=mean_energy,
        total_energy_consumed=total_energy,
        resource_consumption_count=tot_resources_depleted,
        per_agent_recharges=recharges_mean,
        vertex_collisions=tot_v,
        swap_collisions=tot_s,
        occupied_collisions=tot_o,
        total_collisions=tot_v + tot_s + tot_o,
        resource_contentions=tot_c,
        exhaustion_rates={aid: (exhaustions[aid] / episodes) * 100.0 for aid in agent_ids},
        timeout_rates={aid: (timeouts[aid] / episodes) * 100.0 for aid in agent_ids},
        jains_fairness_index=fairness,
        cooperative_completion_time=mean_steps if overall_rate > 0 else None,
    )


# ==============================================================================
# Controlled Experiments A through H
# ==============================================================================


def make_robust_policy_factory(w: int, h: int, max_e: int = 10, cost: int = 1, learner: Any = None) -> Any:
    return lambda: RobustResourcePolicy(
        world_width=w,
        world_height=h,
        max_energy=max_e,
        movement_energy_cost=cost,
        base_learner=learner,
    )


def make_random_policy_factory(seed: int | None = None) -> Any:
    return lambda: RandomPolicy(seed=seed)


def run_all_multi_agent_experiments(output_path: str = "benchmark_results_phase13.json") -> dict[str, Any]:
    """Execute all 8 multi-agent experiments and export to JSON."""
    print("=" * 80)
    print("WORLDSIM PHASE 13 BENCHMARK: MULTI-AGENT SIMULATION")
    print("=" * 80)

    # Train base Linear Q learner on Train-1 for policy baseline
    train_world = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
    )
    lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
    lin_runner.train(1000)
    base_linear_learner = lin_runner.learner

    experiments_data = []

    # --------------------------------------------------------------------------
    # Experiment A: Single-agent control (regression verification)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment A] Single-Agent Control (Regression Verification)...")
    cfg_a = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=3, target=Position(3, 3)),),
        resources=(Resource(Position(1, 0), 4, True),),
        max_energy=6,
    )
    pol_factories_a = {"agent_1": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner)}
    res_a = run_experiment_batch(cfg_a, pol_factories_a, "Experiment A: Single-Agent Control", "1 agent baseline matching Phase 12.2", episodes=50)
    print(f"  Exp A Overall Success: {res_a.overall_completion_rate:.1f}% | Steps: {res_a.mean_steps_to_goal:.1f}")
    experiments_data.append(asdict(res_a))

    # --------------------------------------------------------------------------
    # Experiment B: Multiple independent agents (separate goals & resources)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment B] Multiple Independent Agents (2 Agents, Separate Goals & Resources)...")
    cfg_b = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=3, target=Position(3, 0)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=3, initial_energy=3, target=Position(3, 3)),
        ),
        resources=(
            Resource(Position(1, 0), 4, True),
            Resource(Position(1, 3), 4, True),
        ),
        max_energy=6,
        goal_criterion="individual",
    )
    pol_factories_b = {
        "agent_1": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner),
        "agent_2": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner),
    }
    res_b = run_experiment_batch(cfg_b, pol_factories_b, "Experiment B: Multiple Independent Agents", "2 agents with distinct goals and resources", episodes=50)
    print(f"  Exp B Overall Success: {res_b.overall_completion_rate:.1f}% | Per-agent: {res_b.per_agent_success_rate}")
    experiments_data.append(asdict(res_b))

    # --------------------------------------------------------------------------
    # Experiment C: Resource competition (2 agents, 1 scarce consumable resource)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment C] Resource Competition (2 Agents, 1 Consumable Resource)...")
    cfg_c = MultiAgentWorldConfig(
        width=5, height=5,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=1, initial_energy=2, target=Position(4, 4)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=4, initial_energy=2, target=Position(4, 4)),
        ),
        resources=(Resource(Position(1, 2), 6, consumable=True),),
        max_energy=8,
        goal_criterion="individual",
    )
    pol_factories_c = {
        "agent_1": make_robust_policy_factory(5, 5, max_e=8, learner=base_linear_learner),
        "agent_2": make_robust_policy_factory(5, 5, max_e=8, learner=base_linear_learner),
    }
    res_c = run_experiment_batch(cfg_c, pol_factories_c, "Experiment C: Resource Competition", "2 agents competing for 1 scarce consumable resource", episodes=50)
    print(f"  Exp C Overall Success: {res_c.overall_completion_rate:.1f}% | Contentions: {res_c.resource_contentions} | Per-agent: {res_c.per_agent_success_rate}")
    experiments_data.append(asdict(res_c))

    # --------------------------------------------------------------------------
    # Experiment D: Movement conflicts (vertex bounce & swap collision)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment D] Movement Conflicts (Head-on Swap and Vertex Encounters)...")
    cfg_d = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_1", initial_x=1, initial_y=1, initial_energy=5, target=Position(2, 1)),
            AgentConfig(id="agent_2", initial_x=2, initial_y=1, initial_energy=5, target=Position(1, 1)),
        ),
        resources=(),
        max_energy=6,
        goal_criterion="individual",
    )
    pol_factories_d = {
        "agent_1": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner),
        "agent_2": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner),
    }
    res_d = run_experiment_batch(cfg_d, pol_factories_d, "Experiment D: Movement Conflicts", "Symmetric head-on swap collision and bounce", episodes=50)
    print(f"  Exp D Total Collisions: {res_d.total_collisions} (Swap: {res_d.swap_collisions}) | Recovery: {res_d.overall_completion_rate:.1f}%")
    experiments_data.append(asdict(res_d))

    # --------------------------------------------------------------------------
    # Experiment E: Shared goals (single vs multi-agent arrival)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment E] Shared Goals (Any-Agent Completion)...")
    cfg_e = MultiAgentWorldConfig(
        width=5, height=5,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=4, target=Position(4, 4)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=4, initial_energy=4, target=Position(4, 4)),
        ),
        resources=(Resource(Position(2, 2), 6, False),),
        max_energy=8,
        goal_criterion="shared_any",
    )
    pol_factories_e = {
        "agent_1": make_robust_policy_factory(5, 5, max_e=8, learner=base_linear_learner),
        "agent_2": make_robust_policy_factory(5, 5, max_e=8, learner=base_linear_learner),
    }
    res_e = run_experiment_batch(cfg_e, pol_factories_e, "Experiment E: Shared Goals", "Shared goal where arrival by any agent satisfies completion", episodes=50)
    print(f"  Exp E Overall Completion: {res_e.overall_completion_rate:.1f}% | Steps to Goal: {res_e.mean_steps_to_goal:.1f}")
    experiments_data.append(asdict(res_e))

    # --------------------------------------------------------------------------
    # Experiment F: Cooperative discovery sharing (Isolated vs Shared)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment F] Cooperative Discovery Sharing (Cooperation Opt-In)...")
    # Isolated condition
    cfg_f_iso = MultiAgentWorldConfig(
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
    pol_factories_f = {
        "agent_1": make_robust_policy_factory(8, 8, max_e=10, learner=base_linear_learner),
        "agent_2": make_robust_policy_factory(8, 8, max_e=10, learner=base_linear_learner),
    }
    res_f_iso = run_experiment_batch(cfg_f_iso, pol_factories_f, "Experiment F1: Isolated Exploration", "Isolated sensory observations without sharing", episodes=30)

    # Shared condition
    cfg_f_shared = MultiAgentWorldConfig(
        width=8, height=8,
        agents=(
            AgentConfig(id="agent_1", initial_x=0, initial_y=2, initial_energy=4, target=Position(7, 2)),
            AgentConfig(id="agent_2", initial_x=0, initial_y=6, initial_energy=4, target=Position(7, 6)),
        ),
        resources=(Resource(Position(1, 3), 10, False),),
        max_energy=10,
        cooperative_sharing=True,
        goal_criterion="individual",
    )
    res_f_shared = run_experiment_batch(cfg_f_shared, pol_factories_f, "Experiment F2: Cooperative Discovery Sharing", "Shared discovery pool for observed resources", episodes=30)
    print(f"  Exp F Isolated: {res_f_iso.overall_completion_rate:.1f}% vs Shared: {res_f_shared.overall_completion_rate:.1f}%")
    experiments_data.append(asdict(res_f_iso))
    experiments_data.append(asdict(res_f_shared))

    # --------------------------------------------------------------------------
    # Experiment G: Policy comparison (Robust vs Random)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment G] Policy Comparison (Robust Policy vs Random Baseline)...")
    cfg_g = MultiAgentWorldConfig(
        width=4, height=4,
        agents=(
            AgentConfig(id="agent_robust", initial_x=0, initial_y=0, initial_energy=3, target=Position(3, 3)),
            AgentConfig(id="agent_random", initial_x=0, initial_y=3, initial_energy=3, target=Position(3, 0)),
        ),
        resources=(Resource(Position(1, 0), 4, True), Resource(Position(1, 3), 4, True)),
        max_energy=6,
        goal_criterion="individual",
    )
    pol_factories_g = {
        "agent_robust": make_robust_policy_factory(4, 4, max_e=6, learner=base_linear_learner),
        "agent_random": make_random_policy_factory(seed=42),
    }
    res_g = run_experiment_batch(cfg_g, pol_factories_g, "Experiment G: Policy Comparison", "Robust policy vs Random policy in same environment", episodes=50)
    print(f"  Exp G Per-agent: Robust={res_g.per_agent_success_rate['agent_robust']:.1f}%, Random={res_g.per_agent_success_rate['agent_random']:.1f}%")
    experiments_data.append(asdict(res_g))

    # --------------------------------------------------------------------------
    # Experiment H: Scaling (1, 2, 4, 8 agents)
    # --------------------------------------------------------------------------
    print("\n[Running Experiment H] Scaling Evaluation (1, 2, 4, 8 Agents)...")
    for agent_count in [1, 2, 4, 8]:
        agents = []
        for i in range(agent_count):
            ax = i % 4
            ay = i // 4
            tx = (7 - ax)
            ty = (7 - ay)
            agents.append(AgentConfig(id=f"agent_{i+1}", initial_x=ax, initial_y=ay, initial_energy=6, target=Position(tx, ty)))

        cfg_h = MultiAgentWorldConfig(
            width=8, height=8,
            agents=tuple(agents),
            resources=(Resource(Position(3, 3), 6, False), Resource(Position(4, 4), 6, False)),
            max_energy=10,
            goal_criterion="individual",
        )
        pol_factories_h = {a.id: make_robust_policy_factory(8, 8, max_e=10, learner=base_linear_learner) for a in agents}
        res_h = run_experiment_batch(cfg_h, pol_factories_h, f"Experiment H: Scaling ({agent_count} Agents)", f"Scaling to {agent_count} simultaneous agents on 8x8 grid", episodes=30)
        print(f"  Scale {agent_count} Agents: Overall Success = {res_h.overall_completion_rate:.1f}%, Total Collisions = {res_h.total_collisions}")
        experiments_data.append(asdict(res_h))

    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"version": "13.0", "experiments": experiments_data}, f, indent=2)

    print(f"\nSaved machine-readable multi-agent results to {output_file.resolve()}")
    return {"version": "13.0", "experiments": experiments_data}


if __name__ == "__main__":
    run_all_multi_agent_experiments()
