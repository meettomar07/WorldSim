"""Comprehensive benchmark and validation runner for WorldSim Phase 12.2.

Evaluates 15 distinct scenarios comparing:
1. Phase 12.1 Linear Q-learner (without robustness layer)
2. Phase 12.2 Robust Resource-Aware Policy (with robustness layer)
3. Tabular Q-learning baseline
4. Random (Masked) baseline
5. Classical Planner Reference (EnergyPlannerPolicy)

Evaluates:
- Success rate
- Mean steps for successful episodes
- Mean reward
- Mean final energy
- Recharge count
- Rejected actions
- Energy exhaustion rate
- Timeout rate
- Loop detections / Recovery attempts
- Exploration discoveries
- Infeasibility characterization

Saves machine-readable results to benchmark_results_phase12_2.json.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random
import statistics
import sys
from typing import Any, Sequence

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .learning import LearningConfig, LearningRunner
from .linear_learning import (
    FeatureExtractor,
    LinearLearningConfig,
    LinearLearningRunner,
    LinearQLearner,
)
from .models import Observation, Position, Resource
from .planning import EnergyPlanningProblem, plan_energy
from .policy import EnergyPlannerPolicy, RandomPolicy
from .robust_policy import RobustResourcePolicy


@dataclass
class ScenarioTask:
    scenario_id: int
    name: str
    category: str
    world: WorldConfig
    is_feasible: bool
    optimal_steps: int | None = None


@dataclass
class RobustMetricSummary:
    policy: str
    episodes: int
    success_rate: float
    mean_steps_success: float
    median_steps_success: float
    mean_reward: float
    mean_final_energy: float
    mean_recharges: float
    mean_rejected: float
    exhausted_rate: float
    timeout_rate: float
    mean_recovery_attempts: float
    mean_exploration_discoveries: float


def get_phase12_2_scenarios() -> tuple[ScenarioTask, ...]:
    """Return the 15 verified scenarios required for Phase 12.2 evaluation."""
    raw_scenarios = [
        # 1. Familiar Phase 12.1 configuration
        (1, "Scenario-1: Familiar Phase 12.1", "Familiar Baseline", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
        )),
        # 2. Held-out target location
        (2, "Scenario-2: Held-out Target (3,0)", "Held-out Target", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 0),
        )),
        # 3. Held-out starting position
        (3, "Scenario-3: Inverted Start (3,3)->(0,0)", "Held-out Start", WorldConfig(
            width=4, height=4, initial_x=3, initial_y=3, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 2), 4, True),), observable_target=Position(0, 0),
        )),
        # 4. Held-out resource location
        (4, "Scenario-4: Held-out Res (1,2)", "Held-out Resource", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 2), 4, True),), observable_target=Position(3, 2),
        )),
        # 5. Reusable resource loop
        (5, "Scenario-5: Reusable Res Loop", "Reusable Resource", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=5,
            resources=(Resource(Position(1, 1), 3, False),), observable_target=Position(3, 3),
        )),
        # 6. Resource outside initial sensory radius
        (6, "Scenario-6: Outside Sensory Radius (5x5)", "Scale & Horizon Shift", WorldConfig(
            width=5, height=5, initial_x=0, initial_y=0, initial_energy=4, max_energy=8,
            resources=(Resource(Position(2, 1), 6, True),), observable_target=Position(4, 3),
        )),
        # 7. Nearest resource insufficient, farther resource sufficient
        (7, "Scenario-7: Nearest Insuff vs Farther Suff", "Utility Counterexample", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=8,
            resources=(Resource(Position(1, 0), 1, True), Resource(Position(2, 0), 6, True)),
            observable_target=Position(3, 3),
        )),
        # 8. Multiple resources with different recharge amounts
        (8, "Scenario-8: Multiple Diverse Resources", "Multiple Resources", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=5,
            resources=(Resource(Position(1, 0), 2, True), Resource(Position(2, 2), 3, True)),
            observable_target=Position(3, 3),
        )),
        # 9. Resource depletion after first use
        (9, "Scenario-9: Consumable Depletion", "Resource Depletion", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=6,
            resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
        )),
        # 10. Infeasible task with insufficient total energy
        (10, "Scenario-10: Infeasible Deficit > Res", "Provably Infeasible", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=4,
            resources=(Resource(Position(1, 0), 1, True),), observable_target=Position(3, 3),
        )),
        # 11. Task with no resources
        (11, "Scenario-11: Infeasible No Resources", "Provably Infeasible", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=6,
            resources=(), observable_target=Position(3, 3),
        )),
        # 12. Repeated-action cycle recovery
        (12, "Scenario-12: Repeated Action Cycle", "Cycle Recovery", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(0, 2), 5, True),), observable_target=Position(3, 3),
        )),
        # 13. Unproductive exploration
        (13, "Scenario-13: Unproductive Exploration", "Infeasible Exploration", WorldConfig(
            width=5, height=5, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(), observable_target=Position(4, 4),
        )),
        # 14. Solvable task requiring resource detour
        (14, "Scenario-14: High Deficit Detour", "Multi-step Detour", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=3, initial_energy=3, max_energy=7,
            resources=(Resource(Position(0, 1), 5, True),), observable_target=Position(3, 0),
        )),
        # 15. Novel holdout configuration
        (15, "Scenario-15: Novel Holdout (5x5)", "Novel Evaluation", WorldConfig(
            width=5, height=5, initial_x=4, initial_y=0, initial_energy=3, max_energy=7,
            resources=(Resource(Position(2, 0), 5, True),), observable_target=Position(0, 4),
        )),
    ]

    verified = []
    for sid, name, cat, w in raw_scenarios:
        prob = EnergyPlanningProblem(
            width=w.width, height=w.height, start=Position(w.initial_x, w.initial_y),
            target=w.observable_target, initial_energy=w.initial_energy, max_energy=w.max_energy,
            movement_energy_cost=w.movement_energy_cost, resources=w.resources,
        )
        plan_res = plan_energy(prob)
        verified.append(ScenarioTask(
            scenario_id=sid,
            name=name,
            category=cat,
            world=w,
            is_feasible=plan_res.found,
            optimal_steps=plan_res.plan_length if plan_res.found else None,
        ))
    return tuple(verified)


def evaluate_policy_generic(
    world: WorldConfig,
    policy_fn: Any,
    episodes: int = 50,
    policy_name: str = "Policy",
    max_steps: int = 30,
    seed_base: int = 42,
    is_robust_policy: bool = False,
) -> RobustMetricSummary:
    """Evaluate any policy deterministically over a batch of episodes."""
    successes = 0
    exhausted = 0
    timeouts = 0
    succ_steps = []
    rewards = []
    energies = []
    recharges = []
    rejections = []
    recoveries = []
    discoveries = []

    target = world.observable_target

    for ep in range(episodes):
        sim = Simulation(world)
        if is_robust_policy:
            policy_fn.reset_memory()

        r_total = 0.0
        step_count = 0
        rech_count = 0
        rej_count = 0
        reached = False

        while not reached and step_count < max_steps:
            obs = sim.observe()
            if target is not None and obs.entity_position == target:
                reached = True
                break

            action = policy_fn.choose_action(obs)
            if action is None:
                break

            res = sim.step(action)
            step_count += 1

            if is_robust_policy:
                policy_fn.memory.record_action_result(
                    obs.entity_position, obs.energy, action, res.success
                )

            if action == Action.RECHARGE and res.success:
                rech_count += 1
            if not res.success:
                rej_count += 1

            reached = target is not None and res.observation.entity_position == target
            if reached:
                reward = 1.0
                r_total += reward
                break
            elif not res.success:
                reward = -0.05
                if (
                    res.observation.energy == 0
                    and not any(r.position == res.observation.entity_position for r in res.observation.observed_resources)
                ):
                    reward += -0.1
                    r_total += reward
                    break
            else:
                reward = -0.01
            r_total += reward

            if (
                res.observation.energy == 0
                and not any(r.position == res.observation.entity_position for r in res.observation.observed_resources)
            ):
                break

        final_e = sim.state.entity.energy
        rewards.append(r_total)
        energies.append(final_e)
        recharges.append(rech_count)
        rejections.append(rej_count)

        if is_robust_policy:
            recoveries.append(policy_fn.memory.recovery_attempts)
            discoveries.append(policy_fn.memory.exploration_discoveries)
        else:
            recoveries.append(0)
            discoveries.append(0)

        if reached:
            successes += 1
            succ_steps.append(step_count)
        else:
            if final_e == 0:
                exhausted += 1
            else:
                timeouts += 1

    succ_rate = (successes / episodes) * 100.0
    mean_steps = statistics.mean(succ_steps) if succ_steps else 0.0
    med_steps = statistics.median(succ_steps) if succ_steps else 0.0

    return RobustMetricSummary(
        policy=policy_name,
        episodes=episodes,
        success_rate=succ_rate,
        mean_steps_success=mean_steps,
        median_steps_success=med_steps,
        mean_reward=statistics.mean(rewards),
        mean_final_energy=statistics.mean(energies),
        mean_recharges=statistics.mean(recharges),
        mean_rejected=statistics.mean(rejections),
        exhausted_rate=(exhausted / episodes) * 100.0,
        timeout_rate=(timeouts / episodes) * 100.0,
        mean_recovery_attempts=statistics.mean(recoveries),
        mean_exploration_discoveries=statistics.mean(discoveries),
    )


class LinearQLearnerPolicyWrapper:
    """Wraps LinearQLearner into a policy selecting actions greedily without exploration."""

    def __init__(self, learner: LinearQLearner) -> None:
        self.learner = learner

    def choose_action(self, observation: Observation) -> Action:
        return self.learner.select_action(observation, explore=False)


class TabularQPolicyWrapper:
    """Wraps Tabular QLearner into a policy selecting actions greedily."""

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    def choose_action(self, observation: Observation) -> Action:
        state = self.learner.state_key(observation)
        return self.learner.select_action(state, explore=False)


def run_phase12_2_benchmark(output_path: str = "benchmark_results_phase12_2.json") -> dict[str, Any]:
    """Execute the complete Phase 12.2 benchmark comparing all 5 policies across 15 scenarios."""
    print("=" * 80)
    print("WORLDSIM PHASE 12.2 BENCHMARK: ROBUST RESOURCE-AWARE LEARNING")
    print("=" * 80)

    train_world = WorldConfig(
        width=4,
        height=4,
        initial_x=0,
        initial_y=0,
        initial_energy=3,
        max_energy=6,
        resources=(Resource(Position(1, 0), 4, True),),
        observable_target=Position(3, 3),
    )

    # 1. Train base models (matched 1500 episodes)
    print("\n[Step 1] Training baseline models on Train-1 (1500 episodes)...")
    lin_config = LinearLearningConfig(max_steps=25)
    lin_runner = LinearLearningRunner(world=train_world, config=lin_config, seed=42)
    lin_runner.train(1500)
    print("  Linear Q-learner trained successfully.")

    tab_config = LearningConfig(
        target=train_world.observable_target,
        max_steps=25,
        include_energy=True,
        include_resources=True,
        include_target=True,
        terminal_on_energy_exhaustion=True,
    )
    tab_runner = LearningRunner(
        world=train_world,
        config=tab_config,
        seed=42,
    )
    tab_runner.train(1500)
    print("  Tabular Q-learner trained successfully.")

    # 2. Prepare policies
    linear_baseline_policy = LinearQLearnerPolicyWrapper(lin_runner.learner)
    tabular_baseline_policy = TabularQPolicyWrapper(tab_runner.learner)

    scenarios = get_phase12_2_scenarios()
    eval_episodes = 50
    results: list[dict[str, Any]] = []

    print(f"\n[Step 2] Evaluating 15 Scenarios across 5 policies ({eval_episodes} episodes each)...")
    print(f"{'ID':<3} | {'Scenario Name':<35} | {'Feas?':<5} | {'Lin Q':<7} | {'Robust Q':<8} | {'Tab Q':<7} | {'Planner':<7} | {'Random':<7}")
    print("-" * 90)

    robust_feasible_successes = []
    linear_feasible_successes = []
    tabular_feasible_successes = []
    planner_feasible_successes = []

    for sc in scenarios:
        w = sc.world
        target = w.observable_target or Position(0, 0)

        # A. Phase 12.1 Linear Baseline
        lin_res = evaluate_policy_generic(
            w, linear_baseline_policy, episodes=eval_episodes, policy_name="Phase 12.1 Linear Q"
        )

        # B. Phase 12.2 Robust Policy (wrapping the same trained linear learner)
        robust_policy = RobustResourcePolicy(
            world_width=w.width,
            world_height=w.height,
            max_energy=w.max_energy,
            movement_energy_cost=w.movement_energy_cost,
            sensory_radius=w.sensory_radius,
            base_learner=lin_runner.learner,
        )
        robust_res = evaluate_policy_generic(
            w, robust_policy, episodes=eval_episodes, policy_name="Phase 12.2 Robust Linear Q", is_robust_policy=True
        )

        # C. Tabular Baseline
        tab_res = evaluate_policy_generic(
            w, tabular_baseline_policy, episodes=eval_episodes, policy_name="Tabular Q Baseline"
        )

        # D. Planner Reference (Observation-Limited)
        planner_policy = EnergyPlannerPolicy(
            target=target,
            world_width=w.width,
            world_height=w.height,
            max_energy=w.max_energy,
            movement_energy_cost=w.movement_energy_cost,
        )
        planner_res = evaluate_policy_generic(
            w, planner_policy, episodes=1, policy_name="Planner Reference (EnergyPlannerPolicy)"
        )

        # E. Random Baseline (Masked)
        rand_policy = RandomPolicy(seed=42)
        rand_res = evaluate_policy_generic(
            w, rand_policy, episodes=eval_episodes, policy_name="Random (Masked)"
        )

        if sc.is_feasible:
            robust_feasible_successes.append(robust_res.success_rate)
            linear_feasible_successes.append(lin_res.success_rate)
            tabular_feasible_successes.append(tab_res.success_rate)
            planner_feasible_successes.append(planner_res.success_rate)

        feas_str = "Yes" if sc.is_feasible else "No"
        print(
            f"{sc.scenario_id:<3} | {sc.name:<35} | {feas_str:<5} | "
            f"{lin_res.success_rate:>6.1f}% | {robust_res.success_rate:>7.1f}% | "
            f"{tab_res.success_rate:>6.1f}% | {planner_res.success_rate:>6.1f}% | "
            f"{rand_res.success_rate:>6.1f}%"
        )

        results.append({
            "scenario_id": sc.scenario_id,
            "name": sc.name,
            "category": sc.category,
            "is_feasible": sc.is_feasible,
            "optimal_steps": sc.optimal_steps,
            "linear_baseline": asdict(lin_res),
            "robust_linear": asdict(robust_res),
            "tabular_baseline": asdict(tab_res),
            "planner_reference": asdict(planner_res),
            "random_masked": asdict(rand_res),
        })

    mean_robust_feas = statistics.mean(robust_feasible_successes)
    mean_linear_feas = statistics.mean(linear_feasible_successes)
    mean_tabular_feas = statistics.mean(tabular_feasible_successes)
    mean_planner_feas = statistics.mean(planner_feasible_successes)

    print("-" * 90)
    print(
        f"Mean Feasible Success Rate: "
        f"Linear 12.1 = {mean_linear_feas:.1f}%, "
        f"Robust 12.2 = {mean_robust_feas:.1f}%, "
        f"Tabular = {mean_tabular_feas:.1f}%, "
        f"Planner Ref = {mean_planner_feas:.1f}%"
    )

    summary_data = {
        "metadata": {
            "version": "12.2",
            "eval_episodes_per_scenario": eval_episodes,
            "seed_base": 42,
            "scenarios_evaluated": len(scenarios),
        },
        "aggregate_feasible_metrics": {
            "mean_robust_success": mean_robust_feas,
            "mean_linear_success": mean_linear_feas,
            "mean_tabular_success": mean_tabular_feas,
            "mean_planner_success": mean_planner_feas,
        },
        "scenarios": results,
    }

    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved machine-readable results to {out_file.resolve()}")
    return summary_data


if __name__ == "__main__":
    run_phase12_2_benchmark()
