"""Comprehensive benchmark and validation runner for WorldSim Phase 12.1.

Reproducibly evaluates:
1. Condition A: Matched Single-Task Training (Tabular vs Linear across budgets 300, 900, 1500)
2. Condition B: Matched Multi-Task Training (Tabular vs Linear across budgets 300, 900, 1500)
3. Test Suite: 10 Final Held-Out Configurations (8 Feasible, 2 Provably Infeasible)
4. Baselines: Unmasked Random, Masked Random, and Privileged Classical Planner (EnergyPlannerPolicy)

Produces machine-readable JSON output and structured Markdown tables.
"""

import json
from random import Random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
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
from .models import Position, Resource
from .planning import EnergyPlanningProblem, plan_energy
from .policy import EnergyPlannerPolicy, RandomPolicy


@dataclass
class BenchmarkTask:
    name: str
    category: str
    world: WorldConfig
    is_feasible: bool
    optimal_steps: int | None = None


@dataclass
class MetricSummary:
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


# ------------------------------------------------------------------------------
# Suite Definitions: Training, Validation, and Final Held-out Test
# ------------------------------------------------------------------------------

def get_training_tasks() -> tuple[WorldConfig, ...]:
    train_1 = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        movement_energy_cost=1, sensory_radius=3.0,
        resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
        observable_target=Position(3, 2),
    )
    train_2 = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        movement_energy_cost=1, sensory_radius=3.0,
        resources=(Resource(Position(0, 2), recharge_amount=4, consumable=True),),
        observable_target=Position(3, 2),
    )
    train_3 = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        movement_energy_cost=1, sensory_radius=3.0,
        resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
        observable_target=Position(1, 3),
    )
    return (train_1, train_2, train_3)


def get_final_test_suite() -> tuple[BenchmarkTask, ...]:
    raw_tasks = [
        ("Test-1: Unseen Target (3,0)", "Held-out Target", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 0), 4, True),), observable_target=Position(3, 0),
        )),
        ("Test-2: Inverted Start (3,3)->(0,0)", "Held-out Start", WorldConfig(
            width=4, height=4, initial_x=3, initial_y=3, initial_energy=3, max_energy=6,
            resources=(Resource(Position(2, 2), 4, True),), observable_target=Position(0, 0),
        )),
        ("Test-3: Unseen Res (1,2)", "Held-out Resource Position", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
            resources=(Resource(Position(1, 2), 4, True),), observable_target=Position(3, 2),
        )),
        ("Test-4: Low Energy E0=2, Rech=6", "Recharge & Capacity Shift", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=8,
            resources=(Resource(Position(2, 0), 6, True),), observable_target=Position(3, 2),
        )),
        ("Test-5: Reusable Res (1,1)", "Reusable Resource", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=5,
            resources=(Resource(Position(1, 1), 3, False),), observable_target=Position(3, 3),
        )),
        ("Test-6: 5x5 Grid Shift", "Grid Scale Shift", WorldConfig(
            width=5, height=5, initial_x=0, initial_y=0, initial_energy=4, max_energy=8,
            resources=(Resource(Position(2, 1), 6, True),), observable_target=Position(4, 3),
        )),
        ("Test-7: High Deficit Detour", "Multi-step Detour", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=3, initial_energy=3, max_energy=7,
            resources=(Resource(Position(0, 1), 5, True),), observable_target=Position(3, 0),
        )),
        ("Test-8: Multiple Resources", "Multiple Resources", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=5,
            resources=(Resource(Position(1, 0), 2, True), Resource(Position(2, 2), 3, True)), observable_target=Position(3, 3),
        )),
        ("Test-9: Infeasible Low Res", "Infeasible: Deficit Exceeds Res", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=4,
            resources=(Resource(Position(1, 0), 1, True),), observable_target=Position(3, 3),
        )),
        ("Test-10: Infeasible No Res", "Infeasible: No Resources", WorldConfig(
            width=4, height=4, initial_x=0, initial_y=0, initial_energy=2, max_energy=6,
            resources=(), observable_target=Position(3, 3),
        )),
    ]

    verified = []
    for name, cat, w in raw_tasks:
        prob = EnergyPlanningProblem(
            width=w.width, height=w.height, start=Position(w.initial_x, w.initial_y),
            target=w.observable_target, initial_energy=w.initial_energy, max_energy=w.max_energy,
            movement_energy_cost=w.movement_energy_cost, resources=w.resources,
        )
        plan_res = plan_energy(prob)
        verified.append(BenchmarkTask(
            name=name,
            category=cat,
            world=w,
            is_feasible=plan_res.found,
            optimal_steps=plan_res.plan_length if plan_res.found else None,
        ))
    return tuple(verified)


# ------------------------------------------------------------------------------
# Evaluation Protocols
# ------------------------------------------------------------------------------

def evaluate_rl_runner(runner, world: WorldConfig, policy_name: str, episodes: int = 50, seed: int = 42) -> MetricSummary:
    eval_res = runner.evaluate(episodes, world=world, seed=seed, policy="learned")
    succ_steps = [s for s in eval_res.steps_to_target if s is not None]
    mean_steps = statistics.mean(succ_steps) if succ_steps else 0.0
    med_steps = statistics.median(succ_steps) if succ_steps else 0.0

    exhausted = sum(1 for s, e in zip(eval_res.steps_to_target, eval_res.final_energies) if s is None and e == 0)
    timeouts = sum(1 for s, e in zip(eval_res.steps_to_target, eval_res.final_energies) if s is None and e > 0)

    return MetricSummary(
        policy=policy_name,
        episodes=episodes,
        success_rate=(eval_res.successes / episodes) * 100.0,
        mean_steps_success=mean_steps,
        median_steps_success=med_steps,
        mean_reward=statistics.mean(eval_res.cumulative_rewards),
        mean_final_energy=statistics.mean(eval_res.final_energies),
        mean_recharges=statistics.mean(eval_res.recharge_counts),
        mean_rejected=statistics.mean(eval_res.rejected_counts),
        exhausted_rate=(exhausted / episodes) * 100.0,
        timeout_rate=(timeouts / episodes) * 100.0,
    )


def evaluate_random_policy(world: WorldConfig, policy_name: str, episodes: int = 50, seed: int = 42, mask_illegal: bool = False, max_steps: int = 30) -> MetricSummary:
    successes = 0
    succ_steps: list[int] = []
    rewards: list[float] = []
    energies: list[int] = []
    recharges: list[int] = []
    rejections: list[int] = []
    exhausted = 0
    timeouts = 0
    target = world.observable_target

    for ep in range(episodes):
        sim = Simulation(world)
        rng = Random(seed + ep)
        step_count = 0
        r_total = 0.0
        rech_count = 0
        rej_count = 0
        reached = False
        terminated = False

        while not reached and not terminated and step_count < max_steps:
            obs = sim.observe()
            if mask_illegal:
                legal = []
                if obs.energy >= world.movement_energy_cost:
                    if obs.entity_position.y > 0: legal.append(Action.UP)
                    if obs.entity_position.y < world.height - 1: legal.append(Action.DOWN)
                    if obs.entity_position.x > 0: legal.append(Action.LEFT)
                    if obs.entity_position.x < world.width - 1: legal.append(Action.RIGHT)
                if any(r.position == obs.entity_position for r in obs.observed_resources):
                    legal.append(Action.RECHARGE)
                action = rng.choice(legal) if legal else Action.UP
            else:
                action = rng.choice(tuple(Action))

            res = sim.step(action)
            step_count += 1
            if action == Action.RECHARGE and res.success:
                rech_count += 1
            if not res.success:
                rej_count += 1

            reached = target is not None and res.observation.entity_position == target
            if reached:
                reward = 1.0
                terminated = True
            elif not res.success:
                reward = -0.05
                if res.observation.energy == 0 and not any(r.position == res.observation.entity_position for r in res.observation.observed_resources):
                    terminated = True
                    reward += -0.1
            else:
                reward = -0.01

            r_total += reward

        final_e = sim.state.entity.energy
        rewards.append(r_total)
        energies.append(final_e)
        recharges.append(rech_count)
        rejections.append(rej_count)

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

    return MetricSummary(
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
    )


def evaluate_planner_reference(world: WorldConfig, policy_name: str = "EnergyPlannerPolicy (Ref)", max_steps: int = 30) -> MetricSummary:
    target = world.observable_target or Position(0, 0)
    pol = EnergyPlannerPolicy(
        target=target,
        world_width=world.width,
        world_height=world.height,
        max_energy=world.max_energy,
        movement_energy_cost=world.movement_energy_cost,
    )
    sim = Simulation(world)
    steps = 0
    recharges = 0
    rejections = 0
    r_total = 0.0
    reached = False

    while not reached and steps < max_steps:
        obs = sim.observe()
        action = pol.choose_action(obs)
        if action is None:
            break
        res = sim.step(action)
        steps += 1
        if action == Action.RECHARGE and res.success:
            recharges += 1
        if not res.success:
            rejections += 1

        reached = res.observation.entity_position == target
        if reached:
            r_total += 1.0
        elif not res.success:
            r_total += -0.05
        else:
            r_total += -0.01

    final_e = sim.state.entity.energy
    exhausted = 100.0 if (not reached and final_e == 0) else 0.0
    timeout = 100.0 if (not reached and final_e > 0) else 0.0

    return MetricSummary(
        policy=policy_name,
        episodes=1,
        success_rate=100.0 if reached else 0.0,
        mean_steps_success=float(steps) if reached else 0.0,
        median_steps_success=float(steps) if reached else 0.0,
        mean_reward=r_total,
        mean_final_energy=float(final_e),
        mean_recharges=float(recharges),
        mean_rejected=float(rejections),
        exhausted_rate=exhausted,
        timeout_rate=timeout,
    )


# ------------------------------------------------------------------------------
# Full Benchmark Runner
# ------------------------------------------------------------------------------

def run_benchmark_audit(output_path: Path | None = None) -> dict[str, Any]:
    training_tasks = get_training_tasks()
    test_tasks = get_final_test_suite()
    familiar_task = training_tasks[0]

    results: dict[str, Any] = {
        "metadata": {
            "version": "12.1",
            "eval_episodes_per_task": 50,
            "seed_base": 42,
            "training_budgets": [300, 900, 1500],
        },
        "condition_a_matched_single_task": {},
        "condition_b_matched_multi_task": {},
        "final_held_out_evaluation": {},
    }

    print("=" * 80)
    print("WORLDSIM PHASE 12.1 BENCHMARK AUDIT")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Condition A: Matched Single-Task Training across Budgets
    # --------------------------------------------------------------------------
    print("\n[Condition A] Matched Single-Task Training on Train-1...")
    for budget in [300, 900, 1500]:
        t_cfg = LearningConfig(
            target=familiar_task.observable_target, step_cost=-0.01, target_reward=1.0,
            failed_action_penalty=-0.05, recharge_cost=-0.01, learning_rate=0.3,
            discount_factor=0.9, epsilon=0.4, epsilon_decay=0.995, epsilon_min=0.02,
            max_steps=25, include_energy=True, include_resources=True, include_target=True,
            terminal_on_energy_exhaustion=True,
        )
        t_runner = LearningRunner(familiar_task, t_cfg, seed=42)
        t_runner.train(budget)

        l_cfg = LinearLearningConfig(
            learning_rate=0.08, discount_factor=0.95, epsilon=0.4, epsilon_decay=0.995,
            epsilon_min=0.02, max_steps=25,
        )
        l_runner = LinearLearningRunner(familiar_task, l_cfg, seed=42)
        l_runner.train(budget)

        # Eval on familiar
        st_fam = evaluate_rl_runner(t_runner, familiar_task, "Tabular Q", episodes=50)
        sl_fam = evaluate_rl_runner(l_runner, familiar_task, "Linear Q", episodes=50)

        # Eval on 8 feasible held-out tasks
        t_held_succ = [evaluate_rl_runner(t_runner, t.world, "Tabular Q", episodes=20).success_rate for t in test_tasks[:8]]
        l_held_succ = [evaluate_rl_runner(l_runner, t.world, "Linear Q", episodes=20).success_rate for t in test_tasks[:8]]

        results["condition_a_matched_single_task"][str(budget)] = {
            "tabular_familiar_success": st_fam.success_rate,
            "linear_familiar_success": sl_fam.success_rate,
            "tabular_held_out_avg_success": statistics.mean(t_held_succ),
            "linear_held_out_avg_success": statistics.mean(l_held_succ),
        }
        print(f"  Budget {budget:4d} ep: Familiar (Tab={st_fam.success_rate:.1f}%, Lin={sl_fam.success_rate:.1f}%) | Held-Out Avg (Tab={statistics.mean(t_held_succ):.1f}%, Lin={statistics.mean(l_held_succ):.1f}%)")

    # --------------------------------------------------------------------------
    # 2. Condition B: Matched Multi-Task Training across Budgets
    # --------------------------------------------------------------------------
    print("\n[Condition B] Matched Multi-Task Training on [Train-1, Train-2, Train-3]...")
    for budget in [300, 900, 1500]:
        ep_per_task = budget // len(training_tasks)

        t_cfg = LearningConfig(
            target=familiar_task.observable_target, step_cost=-0.01, target_reward=1.0,
            failed_action_penalty=-0.05, recharge_cost=-0.01, learning_rate=0.3,
            discount_factor=0.9, epsilon=0.4, epsilon_decay=0.995, epsilon_min=0.02,
            max_steps=25, include_energy=True, include_resources=True, include_target=True,
            terminal_on_energy_exhaustion=True,
        )
        t_runner = LearningRunner(familiar_task, t_cfg, seed=123)
        t_runner.train_tasks(training_tasks, episodes_per_task=ep_per_task)

        l_cfg = LinearLearningConfig(
            learning_rate=0.08, discount_factor=0.95, epsilon=0.4, epsilon_decay=0.995,
            epsilon_min=0.02, max_steps=25,
        )
        l_runner = LinearLearningRunner(familiar_task, l_cfg, seed=123)
        l_runner.train_tasks(training_tasks, episodes_per_task=ep_per_task)

        st_fam = evaluate_rl_runner(t_runner, familiar_task, "Tabular Q", episodes=50)
        sl_fam = evaluate_rl_runner(l_runner, familiar_task, "Linear Q", episodes=50)

        t_held_succ = [evaluate_rl_runner(t_runner, t.world, "Tabular Q", episodes=20).success_rate for t in test_tasks[:8]]
        l_held_succ = [evaluate_rl_runner(l_runner, t.world, "Linear Q", episodes=20).success_rate for t in test_tasks[:8]]

        results["condition_b_matched_multi_task"][str(budget)] = {
            "tabular_familiar_success": st_fam.success_rate,
            "linear_familiar_success": sl_fam.success_rate,
            "tabular_held_out_avg_success": statistics.mean(t_held_succ),
            "linear_held_out_avg_success": statistics.mean(l_held_succ),
        }
        print(f"  Budget {budget:4d} ep: Familiar (Tab={st_fam.success_rate:.1f}%, Lin={sl_fam.success_rate:.1f}%) | Held-Out Avg (Tab={statistics.mean(t_held_succ):.1f}%, Lin={statistics.mean(l_held_succ):.1f}%)")

    # --------------------------------------------------------------------------
    # 3. Final Held-out Test Evaluation (Multi-Task 900 ep models)
    # --------------------------------------------------------------------------
    print("\n[Final Evaluation] Evaluating 10 Test Configurations across all policies...")
    t_runner_final = LearningRunner(familiar_task, t_cfg, seed=123)
    t_runner_final.train_tasks(training_tasks, episodes_per_task=300)

    l_runner_final = LinearLearningRunner(familiar_task, l_cfg, seed=123)
    l_runner_final.train_tasks(training_tasks, episodes_per_task=300)

    task_results = []
    for task in test_tasks:
        s_lin = evaluate_rl_runner(l_runner_final, task.world, "Linear Q", episodes=50, seed=42)
        s_tab = evaluate_rl_runner(t_runner_final, task.world, "Tabular Q", episodes=50, seed=42)
        s_pln = evaluate_planner_reference(task.world, "Planner Ref")
        s_msk = evaluate_random_policy(task.world, "Random (Masked)", episodes=50, seed=42, mask_illegal=True)
        s_unm = evaluate_random_policy(task.world, "Random (Unmasked)", episodes=50, seed=42, mask_illegal=False)

        task_entry = {
            "task_name": task.name,
            "category": task.category,
            "is_feasible": task.is_feasible,
            "optimal_steps": task.optimal_steps,
            "linear_q": asdict(s_lin),
            "tabular_q": asdict(s_tab),
            "planner_ref": asdict(s_pln),
            "random_masked": asdict(s_msk),
            "random_unmasked": asdict(s_unm),
        }
        task_results.append(task_entry)
        print(f"  {task.name:35s}: Lin={s_lin.success_rate:5.1f}%, Tab={s_tab.success_rate:5.1f}%, Pln={s_pln.success_rate:5.1f}%, Rnd={s_msk.success_rate:5.1f}%")

    results["final_held_out_evaluation"] = task_results

    # Aggregate summaries over 8 feasible tasks
    feasible_lin_succ = [t["linear_q"]["success_rate"] for t in task_results if t["is_feasible"]]
    feasible_tab_succ = [t["tabular_q"]["success_rate"] for t in task_results if t["is_feasible"]]
    feasible_pln_succ = [t["planner_ref"]["success_rate"] for t in task_results if t["is_feasible"]]
    feasible_rnd_succ = [t["random_masked"]["success_rate"] for t in task_results if t["is_feasible"]]

    results["aggregate_held_out_feasible"] = {
        "linear_q_mean_success": statistics.mean(feasible_lin_succ),
        "tabular_q_mean_success": statistics.mean(feasible_tab_succ),
        "planner_ref_mean_success": statistics.mean(feasible_pln_succ),
        "random_masked_mean_success": statistics.mean(feasible_rnd_succ),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved machine-readable results to {output_path}")

    return results


if __name__ == "__main__":
    out_file = Path(r"D:\SimWorld\benchmark_results.json")
    run_benchmark_audit(out_file)
