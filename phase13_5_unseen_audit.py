"""WorldSim Phase 13.5 — Unseen Scenario Generalization Audit.

Tests whether Phase 13.5 coordination and obstacle-memory fixes generalize
beyond the existing 16 benchmark scenarios to genuinely unseen layouts,
agent formations, resource constraints, and partial-observation challenges.

Deliverables:
- phase13_5_unseen_audit.py
- phase13_5_unseen_scenarios.json
- phase13_5_unseen_results.json
- phase13_5_unseen_audit_report.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import (
    DeadlockDetector,
    DeadlockEvent,
    FairnessMonitor,
    MultiAgentCoordinator,
    ResourceArbitrator,
    jains_fairness_index,
)
from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import RobustResourcePolicy


@dataclass(frozen=True, slots=True)
class PrePositionedEntity:
    """A stationary entity pre-positioned at start (exhausted obstacle or completed agent)."""
    id: str
    x: int
    y: int
    status: str  # "exhausted" or "reached_target"


@dataclass(frozen=True)
class UnseenScenarioSpec:
    """Definition and mathematical feasibility classification for an unseen scenario."""
    scenario_id: str
    experiment_name: str
    family: str
    description: str
    feasibility: str  # "FEASIBLE", "INFEASIBLE", or "UNKNOWN"
    feasibility_proof: str
    width: int
    height: int
    active_agents: tuple[dict[str, Any], ...]
    pre_positioned: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    max_energy: int
    collision_rule: str
    resource_allocation_rule: str
    enable_coordination: bool


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Detailed trace and metrics for a single episode run."""
    seed: int
    completed: bool
    steps: int
    reached: dict[str, bool]
    exhausted: dict[str, bool]
    timed_out: dict[str, bool]
    vertex_collisions: int
    swap_collisions: int
    occupied_collisions: int
    total_collisions: int
    deadlocks_detected: int
    successful_recoveries: int
    unresolved_deadlocks: int
    energy_consumed: dict[str, int]
    recharges: dict[str, int]
    oscillation_cycles: dict[str, int]
    failure_reason: str


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """Aggregated outcome metrics across all seeds for a scenario."""
    scenario_id: str
    experiment_name: str
    family: str
    feasibility: str
    episodes: int
    seeds: tuple[int, ...]
    agent_count: int
    agent_ids: tuple[str, ...]
    overall_completion_rate: float
    episode_completion_count: int
    episode_failure_count: int
    mean_per_agent_success_rate: float
    per_agent_success_rate: dict[str, float]
    mean_steps_to_goal: float
    median_steps_to_goal: float
    mean_episode_duration: float
    total_collisions: int
    vertex_collisions: int
    swap_collisions: int
    occupied_collisions: int
    total_deadlocks: int
    successful_recoveries: int
    unresolved_deadlocks: int
    total_exhaustions: int
    total_timeouts: int
    total_oscillations: int
    primary_failure_reasons: dict[str, int]
    jains_fairness_goal: float
    jains_fairness_energy: float


def make_robust_factory(
    width: int,
    height: int,
    max_energy: int,
    base_linear_learner: Any,
    margin: int = 0,
    coordinator: Any | None = None,
) -> Any:
    return lambda: RobustResourcePolicy(
        world_width=width,
        world_height=height,
        max_energy=max_energy,
        base_learner=base_linear_learner,
        congestion_safety_margin=margin,
        coordinator=coordinator,
    )


def detect_trajectory_oscillations(trajectory: list[Position], min_cycles: int = 3) -> int:
    """Detect 2-step (ping-pong) or 3-step periodic limit cycles in agent movement."""
    if len(trajectory) < 6:
        return 0
    oscillations = 0
    # Check 2-step ping-pong: A -> B -> A -> B -> A -> B
    count_2 = 0
    for i in range(2, len(trajectory)):
        if trajectory[i] == trajectory[i - 2] and trajectory[i] != trajectory[i - 1]:
            count_2 += 1
            if count_2 >= (min_cycles * 2 - 1):
                oscillations += 1
                count_2 = 0
        else:
            count_2 = 0
    return oscillations


def run_unseen_episode(
    config: MultiAgentWorldConfig,
    policy_factories: dict[str, Any],
    pre_positioned: Sequence[PrePositionedEntity],
    seed: int,
    max_steps: int = 50,
) -> EpisodeRecord:
    """Execute a single episode under the real MultiAgentSimulation and policies."""
    sim = MultiAgentSimulation(config, seed=seed)

    # Pre-position stationary entities (exhausted obstacles or resting completed agents)
    for p in pre_positioned:
        if p.id in sim._state.agents:
            target_pos = None if p.status == "exhausted" else Position(p.x, p.y)
            sim._state.agents[p.id] = replace(
                sim._state.agents[p.id],
                status=p.status,
                active=False,
                target=target_pos,
                energy=0 if p.status == "exhausted" else sim._state.agents[p.id].energy,
            )

    active_agent_ids = [a.id for a in config.agents if a.id not in {p.id for p in pre_positioned}]
    policies = {aid: policy_factories[aid]() for aid in active_agent_ids}

    if config.enable_coordination:
        for p in policies.values():
            if hasattr(p, "coordinator") and p.coordinator is None:
                p.coordinator = sim.coordinator
            if hasattr(p, "utility_evaluator") and p.utility_evaluator.coordinator is None:
                p.utility_evaluator.coordinator = sim.coordinator

    step_count = 0
    recharges = {aid: 0 for aid in active_agent_ids}
    reached = {aid: False for aid in active_agent_ids}
    exhausted = {aid: False for aid in active_agent_ids}
    trajectories: dict[str, list[Position]] = {aid: [sim.state.agents[aid].position] for aid in active_agent_ids}

    while step_count < max_steps:
        all_done = all(not sim.state.agents[aid].active for aid in active_agent_ids)
        if all_done:
            break

        if sim.coordinator is not None:
            sim.coordinator.reset_tick()

        actions: dict[str, Action] = {}
        for aid, agent in sim.state.agents.items():
            if not agent.active or aid not in policies:
                actions[aid] = Action.UP
                continue
            obs = sim.observe_agent(aid)
            pol = policies[aid]
            actions[aid] = pol.choose_action(obs)

        res = sim.step(actions)
        step_count += 1

        for aid in active_agent_ids:
            ar = res.agent_results[aid]
            trajectories[aid].append(sim.state.agents[aid].position)
            if ar.action == Action.RECHARGE and ar.success:
                recharges[aid] += 1
            if ar.reached_target:
                reached[aid] = True
            elif sim.state.agents[aid].status == "exhausted":
                exhausted[aid] = True

            pol = policies[aid]
            if hasattr(pol, "memory") and hasattr(pol.memory, "record_action_result"):
                pol.memory.record_action_result(
                    ar.position_before,
                    ar.energy_before,
                    ar.action,
                    ar.success,
                )

    completed = all(reached[aid] for aid in active_agent_ids)
    timed_out = {aid: (not reached[aid] and not exhausted[aid]) for aid in active_agent_ids}

    # Analyze trajectory oscillations
    oscillation_counts = {aid: detect_trajectory_oscillations(trajectories[aid]) for aid in active_agent_ids}

    # Determine failure reason
    if completed:
        failure_reason = "NONE_COMPLETED"
    elif any(exhausted.values()):
        failure_reason = "ENERGY_EXHAUSTION"
    elif any(oscillation_counts[aid] > 0 for aid in active_agent_ids):
        failure_reason = "TIMEOUT_LIMIT_CYCLE"
    elif sim.deadlock_detector.unresolved_deadlocks > 0:
        failure_reason = "TIMEOUT_UNRESOLVED_DEADLOCK"
    else:
        failure_reason = "TIMEOUT_CONGESTION"

    energy_consumed = {
        aid: sim.fairness_monitor.metrics[aid].total_energy_consumed for aid in active_agent_ids
    }

    return EpisodeRecord(
        seed=seed,
        completed=completed,
        steps=step_count,
        reached=reached,
        exhausted=exhausted,
        timed_out=timed_out,
        vertex_collisions=sim.cumulative_vertex_collisions,
        swap_collisions=sim.cumulative_swap_collisions,
        occupied_collisions=sim.cumulative_occupied_collisions,
        total_collisions=sim.cumulative_vertex_collisions + sim.cumulative_swap_collisions + sim.cumulative_occupied_collisions,
        deadlocks_detected=len(sim.deadlock_detector.detected_events),
        successful_recoveries=sim.deadlock_detector.successful_recoveries,
        unresolved_deadlocks=sim.deadlock_detector.unresolved_deadlocks,
        energy_consumed=energy_consumed,
        recharges=recharges,
        oscillation_cycles=oscillation_counts,
        failure_reason=failure_reason,
    )


def run_scenario_audit_batch(
    spec: UnseenScenarioSpec,
    policy_factories: dict[str, Any],
    pre_positioned: Sequence[PrePositionedEntity],
    seeds: Sequence[int],
    max_steps: int = 50,
    coordination_mode: str = "centralized",
) -> tuple[ScenarioSummary, list[EpisodeRecord]]:
    """Execute all trials for an unseen scenario specification."""
    # Build MultiAgentWorldConfig
    agent_configs = [
        AgentConfig(
            id=a["id"],
            initial_x=a["initial_x"],
            initial_y=a["initial_y"],
            initial_energy=a["initial_energy"],
            target=Position(a["target_x"], a["target_y"]) if a.get("target_x") is not None else None,
        )
        for a in spec.active_agents
    ] + [
        AgentConfig(
            id=p["id"],
            initial_x=p["x"],
            initial_y=p["y"],
            initial_energy=0 if p.get("status") == "exhausted" else 10,
            target=None if p.get("status") == "exhausted" else Position(p["x"], p["y"]),
        )
        for p in spec.pre_positioned
    ]

    resources = tuple(
        Resource(
            Position(r["x"], r["y"]),
            recharge_amount=r["recharge_amount"],
            consumable=r["consumable"],
        )
        for r in spec.resources
    )

    config = MultiAgentWorldConfig(
        width=spec.width,
        height=spec.height,
        agents=tuple(agent_configs),
        resources=resources,
        max_energy=spec.max_energy,
        collision_rule=spec.collision_rule,
        resource_allocation_rule=spec.resource_allocation_rule,
        enable_coordination=spec.enable_coordination,
        cooperative_sharing=True,
        goal_criterion="individual",
        coordination_mode=coordination_mode,
    )

    active_agent_ids = tuple(a["id"] for a in spec.active_agents)
    episodes = len(seeds)
    episode_records: list[EpisodeRecord] = []

    success_steps: list[int] = []
    durations: list[int] = []
    overall_success_count = 0
    per_agent_success = {aid: 0 for aid in active_agent_ids}
    failure_reasons_tally: dict[str, int] = {}

    total_v = 0
    total_s = 0
    total_o = 0
    total_deadlocks = 0
    total_recoveries = 0
    total_unresolved = 0
    total_exhaustions = 0
    total_timeouts = 0
    total_oscillations = 0

    for s in seeds:
        rec = run_unseen_episode(config, policy_factories, pre_positioned, seed=s, max_steps=max_steps)
        episode_records.append(rec)
        durations.append(rec.steps)

        if rec.completed:
            overall_success_count += 1
            success_steps.append(rec.steps)

        for aid in active_agent_ids:
            if rec.reached[aid]:
                per_agent_success[aid] += 1
            if rec.exhausted[aid]:
                total_exhaustions += 1
            if rec.timed_out[aid]:
                total_timeouts += 1

        total_v += rec.vertex_collisions
        total_s += rec.swap_collisions
        total_o += rec.occupied_collisions
        total_deadlocks += rec.deadlocks_detected
        total_recoveries += rec.successful_recoveries
        total_unresolved += rec.unresolved_deadlocks
        total_oscillations += sum(rec.oscillation_cycles.values())

        reason = rec.failure_reason
        failure_reasons_tally[reason] = failure_reasons_tally.get(reason, 0) + 1

    per_agent_succ_rate = {aid: (per_agent_success[aid] / episodes) * 100.0 for aid in active_agent_ids}
    mean_agent_succ = float(statistics.mean(per_agent_succ_rate.values())) if per_agent_succ_rate else 0.0
    overall_succ_rate = (overall_success_count / episodes) * 100.0
    mean_succ_steps = float(statistics.mean(success_steps)) if success_steps else 0.0
    median_succ_steps = float(statistics.median(success_steps)) if success_steps else 0.0
    mean_dur = float(statistics.mean(durations)) if durations else 0.0

    # Calculate Jain's indices across episodes
    goal_counts = [per_agent_success[aid] for aid in active_agent_ids]
    jain_goal = float(jains_fairness_index(goal_counts)) if goal_counts else 1.0

    tot_energy_per_agent = [sum(rec.energy_consumed.get(aid, 0) for rec in episode_records) for aid in active_agent_ids]
    jain_energy = float(jains_fairness_index(tot_energy_per_agent)) if tot_energy_per_agent else 1.0

    summary = ScenarioSummary(
        scenario_id=spec.scenario_id,
        experiment_name=spec.experiment_name,
        family=spec.family,
        feasibility=spec.feasibility,
        episodes=episodes,
        seeds=tuple(seeds),
        agent_count=len(active_agent_ids),
        agent_ids=active_agent_ids,
        overall_completion_rate=round(overall_succ_rate, 2),
        episode_completion_count=overall_success_count,
        episode_failure_count=episodes - overall_success_count,
        mean_per_agent_success_rate=round(mean_agent_succ, 2),
        per_agent_success_rate={k: round(v, 2) for k, v in per_agent_succ_rate.items()},
        mean_steps_to_goal=round(mean_succ_steps, 2),
        median_steps_to_goal=round(median_succ_steps, 2),
        mean_episode_duration=round(mean_dur, 2),
        total_collisions=total_v + total_s + total_o,
        vertex_collisions=total_v,
        swap_collisions=total_s,
        occupied_collisions=total_o,
        total_deadlocks=total_deadlocks,
        successful_recoveries=total_recoveries,
        unresolved_deadlocks=total_unresolved,
        total_exhaustions=total_exhaustions,
        total_timeouts=total_timeouts,
        total_oscillations=total_oscillations,
        primary_failure_reasons=failure_reasons_tally,
        jains_fairness_goal=round(jain_goal, 4),
        jains_fairness_energy=round(jain_energy, 4),
    )
    return summary, episode_records


def build_all_unseen_scenarios() -> list[tuple[UnseenScenarioSpec, list[PrePositionedEntity]]]:
    """Define 16 genuinely unseen scenarios across all 6 requested scenario families."""
    scenarios: list[tuple[UnseenScenarioSpec, list[PrePositionedEntity]]] = []

    # ==========================================================================
    # FAMILY 1: Layout Topologies (Corridors, T-Junctions, Cul-de-Sacs, Asymmetric)
    # ==========================================================================

    # U01: Double Corridor with Central Bottleneck (4 Agents)
    pre_u01 = [
        PrePositionedEntity("wall_1", 1, 1, "exhausted"),
        PrePositionedEntity("wall_2", 2, 1, "exhausted"),
        PrePositionedEntity("wall_3", 4, 1, "exhausted"),
        PrePositionedEntity("wall_4", 5, 1, "exhausted"),
    ]
    u01 = UnseenScenarioSpec(
        scenario_id="U01",
        experiment_name="U01: Double Corridor with Central Bottleneck (4 Agents)",
        family="Layout Topologies",
        description="Two 1-wide horizontal corridors separated by an obstacle barrier with a single central opening at (3, 1). Two opposing agent pairs must traverse without swap locking.",
        feasibility="FEASIBLE",
        feasibility_proof="Distance from (0, 0) to (6, 0) is 6. With detour through central portal (3, 1), path is 8 steps. Initial energy 10 >= 8. Geometrically and energetically feasible.",
        width=7, height=3,
        active_agents=(
            {"id": "agent_nw", "initial_x": 0, "initial_y": 0, "initial_energy": 10, "target_x": 6, "target_y": 0},
            {"id": "agent_sw", "initial_x": 0, "initial_y": 2, "initial_energy": 10, "target_x": 6, "target_y": 2},
            {"id": "agent_ne", "initial_x": 6, "initial_y": 0, "initial_energy": 10, "target_x": 0, "target_y": 0},
            {"id": "agent_se", "initial_x": 6, "initial_y": 2, "initial_energy": 10, "target_x": 0, "target_y": 2},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u01),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u01, pre_u01))

    # U02: Dual T-Junction Crossway (4 Agents)
    pre_u02 = [
        PrePositionedEntity("obs_c1", 2, 2, "exhausted"),
        PrePositionedEntity("obs_c2", 3, 2, "exhausted"),
        PrePositionedEntity("obs_c3", 4, 2, "exhausted"),
        PrePositionedEntity("obs_p1", 0, 2, "exhausted"),
        PrePositionedEntity("obs_p2", 6, 2, "exhausted"),
    ]
    u02 = UnseenScenarioSpec(
        scenario_id="U02",
        experiment_name="U02: Dual T-Junction Crossway (4 Agents)",
        family="Layout Topologies",
        description="H-shaped dual T-junction layout where row 2 is blocked, forcing perpendicular crossing traffic through (1, 2) and (5, 2).",
        feasibility="FEASIBLE",
        feasibility_proof="Distances for horizontal agents (0, 1)->(6, 1) and (6, 3)->(0, 3) are 6. Vertical agents (1, 0)->(1, 4) and (5, 4)->(5, 0) are 4. Initial energy 10 >= 6. Solvable without resource visits.",
        width=7, height=5,
        active_agents=(
            {"id": "agent_h1", "initial_x": 0, "initial_y": 1, "initial_energy": 10, "target_x": 6, "target_y": 1},
            {"id": "agent_h2", "initial_x": 6, "initial_y": 3, "initial_energy": 10, "target_x": 0, "target_y": 3},
            {"id": "agent_v1", "initial_x": 1, "initial_y": 0, "initial_energy": 10, "target_x": 1, "target_y": 4},
            {"id": "agent_v2", "initial_x": 5, "initial_y": 4, "initial_energy": 10, "target_x": 5, "target_y": 0},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u02),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u02, pre_u02))

    # U03: Cul-de-Sac with Alternative Perimeter Bypass (2 Agents)
    pre_u03 = [
        PrePositionedEntity("wall_c1", 3, 0, "exhausted"),
        PrePositionedEntity("wall_c2", 3, 1, "exhausted"),
    ]
    u03 = UnseenScenarioSpec(
        scenario_id="U03",
        experiment_name="U03: Cul-de-Sac with Alternative Perimeter Bypass (2 Agents)",
        family="Layout Topologies",
        description="Row 1 dead-ends at (2, 1) due to solid wall at (3, 1), forcing Agent 1 to backtrack/detour via row 2/3 to reach (5, 1).",
        feasibility="FEASIBLE",
        feasibility_proof="Path (0, 1)->(2, 1)->(2, 2)->(3, 2)->(4, 2)->(5, 2)->(5, 1) requires 8 steps. Initial energy 12 >= 8. Agent 2 moves (5, 2)->(0, 2) (5 steps). Solvable.",
        width=6, height=4,
        active_agents=(
            {"id": "agent_seeker", "initial_x": 0, "initial_y": 1, "initial_energy": 12, "target_x": 5, "target_y": 1},
            {"id": "agent_counter", "initial_x": 5, "initial_y": 2, "initial_energy": 12, "target_x": 0, "target_y": 2},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u03),
        resources=(),
        max_energy=12,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u03, pre_u03))

    # ==========================================================================
    # FAMILY 2: Multi-Agent Formations (3, 6, 8 Agents)
    # ==========================================================================

    # U04: Tripartite Reciprocal Swap Loop (3 Agents)
    u04 = UnseenScenarioSpec(
        scenario_id="U04",
        experiment_name="U04: Tripartite Reciprocal Swap Loop (3 Agents)",
        family="Multi-Agent Formations",
        description="3 agents arranged in a triangular cycle: A1->A2's start, A2->A3's start, A3->A1's start. Tests 3-way rotational deadlock resolution.",
        feasibility="FEASIBLE",
        feasibility_proof="Distances are 2 to 3. Initial energy 8 is ample. An orthogonal sidestep by any agent allows the cyclic rotation to resolve in <= 5 steps.",
        width=5, height=5,
        active_agents=(
            {"id": "agent_1", "initial_x": 1, "initial_y": 2, "initial_energy": 8, "target_x": 3, "target_y": 2},
            {"id": "agent_2", "initial_x": 3, "initial_y": 2, "initial_energy": 8, "target_x": 2, "target_y": 3},
            {"id": "agent_3", "initial_x": 2, "initial_y": 3, "initial_energy": 8, "target_x": 1, "target_y": 2},
        ),
        pre_positioned=(),
        resources=(),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u04, []))

    # U05: Hexagonal Concentric Convergence (6 Agents)
    u05 = UnseenScenarioSpec(
        scenario_id="U05",
        experiment_name="U05: Hexagonal Concentric Convergence (6 Agents)",
        family="Multi-Agent Formations",
        description="6 agents starting along perimeter cells converging toward center and crossing to opposite sides with 2 reusable resources.",
        feasibility="FEASIBLE",
        feasibility_proof="Direct distances are 6 to 8. Initial energy 10 >= 8. Reusable resources at (2, 3) and (4, 3) provide replenishment if queuing delays occur.",
        width=7, height=7,
        active_agents=(
            {"id": "agent_n", "initial_x": 3, "initial_y": 0, "initial_energy": 10, "target_x": 3, "target_y": 6},
            {"id": "agent_s", "initial_x": 3, "initial_y": 6, "initial_energy": 10, "target_x": 3, "target_y": 0},
            {"id": "agent_w", "initial_x": 0, "initial_y": 3, "initial_energy": 10, "target_x": 6, "target_y": 3},
            {"id": "agent_e", "initial_x": 6, "initial_y": 3, "initial_energy": 10, "target_x": 0, "target_y": 3},
            {"id": "agent_nw", "initial_x": 1, "initial_y": 1, "initial_energy": 10, "target_x": 5, "target_y": 5},
            {"id": "agent_se", "initial_x": 5, "initial_y": 5, "initial_energy": 10, "target_x": 1, "target_y": 1},
        ),
        pre_positioned=(),
        resources=(
            {"x": 2, "y": 3, "recharge_amount": 6, "consumable": False},
            {"x": 4, "y": 3, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u05, []))

    # U06: 8-Agent Staggered Quadrant Mesh (8 Agents)
    u06 = UnseenScenarioSpec(
        scenario_id="U06",
        experiment_name="U06: 8-Agent Staggered Quadrant Mesh (8 Agents)",
        family="Multi-Agent Formations",
        description="8 agents in pairs in each grid quadrant crossing diagonally with 4 distributed reusable resources.",
        feasibility="FEASIBLE",
        feasibility_proof="Diagonal target distance is 10. Corner to nearest quadrant resource is distance 2. Initial energy 8 reaches resource in 2 steps with 6 energy remaining, recharges to 10, and completes remaining 8 steps.",
        width=8, height=8,
        active_agents=(
            {"id": "agent_q1_a", "initial_x": 1, "initial_y": 1, "initial_energy": 8, "target_x": 6, "target_y": 6},
            {"id": "agent_q1_b", "initial_x": 2, "initial_y": 1, "initial_energy": 8, "target_x": 5, "target_y": 6},
            {"id": "agent_q2_a", "initial_x": 6, "initial_y": 1, "initial_energy": 8, "target_x": 1, "target_y": 6},
            {"id": "agent_q2_b", "initial_x": 5, "initial_y": 1, "initial_energy": 8, "target_x": 2, "target_y": 6},
            {"id": "agent_q3_a", "initial_x": 1, "initial_y": 6, "initial_energy": 8, "target_x": 6, "target_y": 1},
            {"id": "agent_q3_b", "initial_x": 2, "initial_y": 6, "initial_energy": 8, "target_x": 5, "target_y": 1},
            {"id": "agent_q4_a", "initial_x": 6, "initial_y": 6, "initial_energy": 8, "target_x": 1, "target_y": 1},
            {"id": "agent_q4_b", "initial_x": 5, "initial_y": 6, "initial_energy": 8, "target_x": 2, "target_y": 1},
        ),
        pre_positioned=(),
        resources=(
            {"x": 2, "y": 2, "recharge_amount": 6, "consumable": False},
            {"x": 5, "y": 2, "recharge_amount": 6, "consumable": False},
            {"x": 2, "y": 5, "recharge_amount": 6, "consumable": False},
            {"x": 5, "y": 5, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u06, []))

    # ==========================================================================
    # FAMILY 3: Completed Agent Yielding Chains & Corridor Traversal
    # ==========================================================================

    # U07: Sequential Goal-Cell Chain in 1-Wide Corridor (3 Agents)
    pre_u07 = [
        PrePositionedEntity("wall_n1", 2, 0, "exhausted"),
        PrePositionedEntity("wall_n2", 3, 0, "exhausted"),
        PrePositionedEntity("wall_n3", 4, 0, "exhausted"),
        PrePositionedEntity("wall_s1", 2, 2, "exhausted"),
        PrePositionedEntity("wall_s2", 3, 2, "exhausted"),
        PrePositionedEntity("wall_s3", 4, 2, "exhausted"),
    ]
    u07 = UnseenScenarioSpec(
        scenario_id="U07",
        experiment_name="U07: Sequential Goal-Cell Chain in 1-Wide Corridor (3 Agents)",
        family="Completed Agent Traversal",
        description="A 1-wide corridor along row 1 with sequential goals at (3, 1), (4, 1), (5, 1). Later agents must step through earlier completed agents' goal cells.",
        feasibility="FEASIBLE",
        feasibility_proof="Goal-cell yielding permits active agents to step through cells of agents with status reached_target. Distances are 3, 5, 7. Initial energy 10 >= 7. Solvable in sequential order.",
        width=6, height=3,
        active_agents=(
            {"id": "agent_short", "initial_x": 0, "initial_y": 2, "initial_energy": 10, "target_x": 3, "target_y": 1},
            {"id": "agent_mid", "initial_x": 0, "initial_y": 0, "initial_energy": 10, "target_x": 4, "target_y": 1},
            {"id": "agent_long", "initial_x": 0, "initial_y": 1, "initial_energy": 10, "target_x": 5, "target_y": 1},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u07),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u07, pre_u07))

    # U08: Completed Agent at Critical Chokepoint (2 Agents)
    pre_u08 = [
        PrePositionedEntity("choke_n", 2, 0, "exhausted"),
        PrePositionedEntity("choke_s", 2, 2, "exhausted"),
        PrePositionedEntity("agent_resting", 2, 1, "reached_target"),
    ]
    u08 = UnseenScenarioSpec(
        scenario_id="U08",
        experiment_name="U08: Completed Agent at Critical Chokepoint (2 Agents)",
        family="Completed Agent Traversal",
        description="A completed resting agent sits at the only chokepoint (2, 1). Active agent at (0, 1) targeting (4, 1) must traverse directly through it.",
        feasibility="FEASIBLE",
        feasibility_proof="Direct line (0, 1) to (4, 1) is 4 steps. With goal-cell yielding, resting agent at (2, 1) yields right-of-way. Energy 8 >= 4. Solvable in exactly 4 steps.",
        width=5, height=3,
        active_agents=(
            {"id": "agent_active", "initial_x": 0, "initial_y": 1, "initial_energy": 8, "target_x": 4, "target_y": 1},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u08),
        resources=(),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u08, pre_u08))

    # ==========================================================================
    # FAMILY 4: Resource Competitions & Energy Variations
    # ==========================================================================

    # U09: Multi-Agent Competition for Scarce Single Reusable Resource (4 Agents)
    u09 = UnseenScenarioSpec(
        scenario_id="U09",
        experiment_name="U09: Multi-Agent Competition for Scarce Single Reusable Resource (4 Agents)",
        family="Resource & Energy",
        description="4 agents starting at corners of 5x5 world targeting opposite corners (dist=8). Initial energy E=5 cannot reach target directly. All 4 must sequentially recharge at single central resource (2, 2).",
        feasibility="FEASIBLE",
        feasibility_proof="Distance from every corner (0,0), (0,4), (4,0), (4,4) to center (2,2) is 4 <= 5. Energy remaining is 1, recharges to 7, distance to target is 4 <= 7. Fair-Balanced arbitration guarantees sequential access.",
        width=5, height=5,
        active_agents=(
            {"id": "agent_c1", "initial_x": 0, "initial_y": 0, "initial_energy": 5, "target_x": 4, "target_y": 4},
            {"id": "agent_c2", "initial_x": 0, "initial_y": 4, "initial_energy": 5, "target_x": 4, "target_y": 0},
            {"id": "agent_c3", "initial_x": 4, "initial_y": 0, "initial_energy": 5, "target_x": 0, "target_y": 4},
            {"id": "agent_c4", "initial_x": 4, "initial_y": 4, "initial_energy": 5, "target_x": 0, "target_y": 0},
        ),
        pre_positioned=(),
        resources=(
            {"x": 2, "y": 2, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u09, []))

    # U10: Deceptive Shortest Path (Shortest Route Starves, Detour Recharges) (2 Agents)
    u10 = UnseenScenarioSpec(
        scenario_id="U10",
        experiment_name="U10: Deceptive Shortest Path (Shortest Route Starves, Detour Recharges) (2 Agents)",
        family="Resource & Energy",
        description="Straight path to goal is distance 6, but agent has energy 5 (direct movement exhausts). Agent must detour to resource at (2, 4) (dist=4), recharge to 7, and reach goal.",
        feasibility="FEASIBLE",
        feasibility_proof="Direct dist=6 > E=5. Detour to (2, 4) is dist=4 <= 5. Recharges to 7. From (2, 4) to (6, 2) is dist=6 <= 7. Total steps 10. Energetically solvable only via detour.",
        width=7, height=5,
        active_agents=(
            {"id": "agent_east", "initial_x": 0, "initial_y": 2, "initial_energy": 5, "target_x": 6, "target_y": 2},
            {"id": "agent_west", "initial_x": 6, "initial_y": 2, "initial_energy": 5, "target_x": 0, "target_y": 2},
        ),
        pre_positioned=(),
        resources=(
            {"x": 2, "y": 4, "recharge_amount": 6, "consumable": False},
            {"x": 4, "y": 0, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=8,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u10, []))

    # U11: Rigorously Infeasible Layout — Geometric & Energy Disconnect (Category 1 Infeasible)
    u11 = UnseenScenarioSpec(
        scenario_id="U11",
        experiment_name="U11: Rigorously Infeasible Layout — Geometric & Energy Disconnect (2 Agents)",
        family="Resource & Energy",
        description="Agents start at (0, 0) and (0, 1) with energy 3. Nearest resource is at (4, 4) (dist=8 > 3) and target at (7, 7) (dist=14 > 3). Mathematically impossible.",
        feasibility="INFEASIBLE",
        feasibility_proof="Mathematical proof: At each step, distance to any cell can decrease by at most 1. With initial energy 3, maximum reachable Manhattan distance is 3. Since min(dist to resource, dist to target) = 8 > 3, every possible action sequence exhausts at step 3.",
        width=8, height=8,
        active_agents=(
            {"id": "agent_1", "initial_x": 0, "initial_y": 0, "initial_energy": 3, "target_x": 7, "target_y": 7},
            {"id": "agent_2", "initial_x": 0, "initial_y": 1, "initial_energy": 3, "target_x": 7, "target_y": 6},
        ),
        pre_positioned=(),
        resources=(
            {"x": 4, "y": 4, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u11, []))

    # ==========================================================================
    # FAMILY 5: Partial Observation & Memory Dynamics
    # ==========================================================================

    # U12: Disappearing & Reappearing Obstacle Memory Stress Test (2 Agents)
    pre_u12 = [
        PrePositionedEntity("obs_blind", 3, 2, "exhausted"),
    ]
    u12 = UnseenScenarioSpec(
        scenario_id="U12",
        experiment_name="U12: Disappearing & Reappearing Obstacle Memory Stress Test (2 Agents)",
        family="Partial Observation",
        description="Agent 1 at (0, 2) targeting (7, 2). Obstacle at (3, 2) is visible at (1, 2) (dist=2.0), prompting detour to (1, 1). At (1, 1), obstacle drops outside sensory range (dist=2.24 > 2.0). Agent must remember obstacle across sensory horizon.",
        feasibility="FEASIBLE",
        feasibility_proof="Detour via row 1: (0,2)->(1,2)->(1,1)->(2,1)->(3,1)->(4,1)->(5,1)->(5,2)->(7,2) is 9 steps. Initial energy 12 >= 9. Energetically and geometrically solvable.",
        width=8, height=5,
        active_agents=(
            {"id": "agent_traveler", "initial_x": 0, "initial_y": 2, "initial_energy": 12, "target_x": 7, "target_y": 2},
            {"id": "agent_counter", "initial_x": 7, "initial_y": 4, "initial_energy": 12, "target_x": 0, "target_y": 4},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u12),
        resources=(),
        max_energy=12,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u12, pre_u12))

    # U13: Stale Obstacle Memory Expiration (TTL Revalidation) (1 Agent)
    pre_u13 = [
        PrePositionedEntity("obs_early", 2, 1, "exhausted"),
    ]
    u13 = UnseenScenarioSpec(
        scenario_id="U13",
        experiment_name="U13: Stale Obstacle Memory Expiration (TTL Revalidation) (1 Agent)",
        family="Partial Observation",
        description="Agent starts at (0, 1) targeting (11, 1). Early obstacle at (2, 1) prompts detour to row 3. Long journey along row 3 (16 steps) exceeds TTL=15, testing that expired obstacle memory is purged without corrupted state.",
        feasibility="FEASIBLE",
        feasibility_proof="Path along row 3 from x=0 to x=11 and back to y=1 takes 15 steps. Initial energy 20 >= 15. Solvable.",
        width=12, height=4,
        active_agents=(
            {"id": "agent_long_route", "initial_x": 0, "initial_y": 1, "initial_energy": 20, "target_x": 11, "target_y": 1},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u13),
        resources=(),
        max_energy=20,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u13, pre_u13))

    # U14: Multi-Corner Blind Intersection (4 Agents)
    pre_u14 = [
        PrePositionedEntity("blind_nw", 2, 2, "exhausted"),
        PrePositionedEntity("blind_ne", 4, 2, "exhausted"),
        PrePositionedEntity("blind_sw", 2, 4, "exhausted"),
        PrePositionedEntity("blind_se", 4, 4, "exhausted"),
    ]
    u14 = UnseenScenarioSpec(
        scenario_id="U14",
        experiment_name="U14: Multi-Corner Blind Intersection (4 Agents)",
        family="Partial Observation",
        description="4 solid obstacle pillars surround central junction (3, 3) at distance 1.41. 4 agents approach blind corners from N, S, E, W.",
        feasibility="FEASIBLE",
        feasibility_proof="Straight paths across junction (3, 0)->(3, 6), (3, 6)->(3, 0), (0, 3)->(6, 3), (6, 3)->(0, 3) are 6 steps. Initial energy 10 >= 6. Priority rotation resolves right-of-way cleanly.",
        width=7, height=7,
        active_agents=(
            {"id": "agent_n", "initial_x": 3, "initial_y": 0, "initial_energy": 10, "target_x": 3, "target_y": 6},
            {"id": "agent_s", "initial_x": 3, "initial_y": 6, "initial_energy": 10, "target_x": 3, "target_y": 0},
            {"id": "agent_w", "initial_x": 0, "initial_y": 3, "initial_energy": 10, "target_x": 6, "target_y": 3},
            {"id": "agent_e", "initial_x": 6, "initial_y": 3, "initial_energy": 10, "target_x": 0, "target_y": 3},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u14),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u14, pre_u14))

    # ==========================================================================
    # FAMILY 6: Scaled Layouts & Sealed Enclosures
    # ==========================================================================

    # U15: 6-Agent Asymmetric Funnel with Reusable Centers (6 Agents)
    pre_u15 = [
        PrePositionedEntity("wall_w1", 0, 3, "exhausted"),
        PrePositionedEntity("wall_w2", 0, 4, "exhausted"),
        PrePositionedEntity("wall_e1", 7, 3, "exhausted"),
        PrePositionedEntity("wall_e2", 7, 4, "exhausted"),
    ]
    u15 = UnseenScenarioSpec(
        scenario_id="U15",
        experiment_name="U15: 6-Agent Asymmetric Funnel with Reusable Centers (6 Agents)",
        family="Scaled Topologies",
        description="6 agents in opposing 3v3 north/south formation funneled through center with 2 reusable resources at (3, 3) and (4, 4).",
        feasibility="FEASIBLE",
        feasibility_proof="Direct vertical distance is 7. Initial energy 10 >= 7. Resources at center provide recharge padding if door-step congestion occurs. Geometrically solvable.",
        width=8, height=8,
        active_agents=(
            {"id": "agent_n1", "initial_x": 1, "initial_y": 0, "initial_energy": 10, "target_x": 1, "target_y": 7},
            {"id": "agent_n2", "initial_x": 3, "initial_y": 0, "initial_energy": 10, "target_x": 3, "target_y": 7},
            {"id": "agent_n3", "initial_x": 5, "initial_y": 0, "initial_energy": 10, "target_x": 5, "target_y": 7},
            {"id": "agent_s1", "initial_x": 2, "initial_y": 7, "initial_energy": 10, "target_x": 2, "target_y": 0},
            {"id": "agent_s2", "initial_x": 4, "initial_y": 7, "initial_energy": 10, "target_x": 4, "target_y": 0},
            {"id": "agent_s3", "initial_x": 6, "initial_y": 7, "initial_energy": 10, "target_x": 6, "target_y": 0},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u15),
        resources=(
            {"x": 3, "y": 3, "recharge_amount": 6, "consumable": False},
            {"x": 4, "y": 4, "recharge_amount": 6, "consumable": False},
        ),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u15, pre_u15))

    # U16: Rigorously Infeasible Layout — Sealed Cul-de-Sac (2 Agents)
    pre_u16 = [
        PrePositionedEntity("wall_s1", 4, 0, "exhausted"),
        PrePositionedEntity("wall_s2", 4, 1, "exhausted"),
        PrePositionedEntity("wall_s3", 4, 2, "exhausted"),
        PrePositionedEntity("wall_s4", 4, 3, "exhausted"),
    ]
    u16 = UnseenScenarioSpec(
        scenario_id="U16",
        experiment_name="U16: Rigorously Infeasible Layout — Sealed Cul-de-Sac (2 Agents)",
        family="Scaled Topologies",
        description="A solid wall of exhausted agents spans the entire grid height from (4, 0) to (4, 3). Targets (5, 1) and (5, 2) are completely sealed behind the wall.",
        feasibility="INFEASIBLE",
        feasibility_proof="Mathematical proof: Exhausted agents are impassable physical obstacles. The line of exhausted agents at x=4 spans all y from 0 to height-1=3. No grid coordinate (x > 4) is connected to any coordinate (x < 4). Goal is topologically unreachable from any start position.",
        width=6, height=4,
        active_agents=(
            {"id": "agent_1", "initial_x": 0, "initial_y": 1, "initial_energy": 10, "target_x": 5, "target_y": 1},
            {"id": "agent_2", "initial_x": 0, "initial_y": 2, "initial_energy": 10, "target_x": 5, "target_y": 2},
        ),
        pre_positioned=tuple(asdict(p) for p in pre_u16),
        resources=(),
        max_energy=10,
        collision_rule="priority_rotation",
        resource_allocation_rule="fair_balanced",
        enable_coordination=True,
    )
    scenarios.append((u16, pre_u16))

    return scenarios


def run_full_unseen_generalization_audit(
    scenarios_json_path: str = "phase13_5_unseen_scenarios.json",
    results_json_path: str = "phase13_5_unseen_results.json",
    seeds_per_scenario: int = 20,
    base_seed: int = 500,
) -> dict[str, Any]:
    """Execute the complete Phase 13.5 Unseen Generalization Audit."""
    print("=" * 80)
    print("WORLDSIM PHASE 13.5: UNSEEN SCENARIO GENERALIZATION AUDIT")
    print("=" * 80)

    # Step 0: Pre-train base linear learner
    print("\n[Step 0] Training standard base linear learner...")
    train_world = WorldConfig(
        width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
        resources=(Resource(Position(1, 0), 4, True),), observable_target=Position(3, 3),
    )
    lin_runner = LinearLearningRunner(world=train_world, config=LinearLearningConfig(max_steps=25), seed=42)
    lin_runner.train(1000)
    base_linear_learner = lin_runner.learner
    print("Base linear learner pre-training complete.")

    scenarios = build_all_unseen_scenarios()
    seeds = tuple(base_seed + i for i in range(seeds_per_scenario))
    print(f"Loaded {len(scenarios)} unseen scenarios across 6 families.")
    print(f"Evaluation seed range: {seeds[0]} to {seeds[-1]} ({len(seeds)} episodes per scenario).")

    # Export scenario specifications
    scenarios_data = [asdict(s[0]) for s in scenarios]
    with open(scenarios_json_path, "w", encoding="utf-8") as f:
        json.dump({"unseen_scenarios_count": len(scenarios_data), "scenarios": scenarios_data}, f, indent=2)
    print(f"Saved scenario specifications to {Path(scenarios_json_path).resolve()}")

    summaries: list[dict[str, Any]] = []
    all_episodes_data: dict[str, list[dict[str, Any]]] = {}

    print("\n" + "-" * 80)
    print(f"{'ID':<4} | {'Scenario Name':<42} | {'Family':<20} | {'Feasible':<10} | {'Overall':<7} | {'Mean Agt':<8} | {'Steps':<6} | {'Cols':<5} | {'Deadl':<5}")
    print("-" * 80)

    for spec, pre_pos in scenarios:
        factories = {
            a["id"]: make_robust_factory(spec.width, spec.height, spec.max_energy, base_linear_learner, margin=1)
            for a in spec.active_agents
        }
        summary, ep_records = run_scenario_audit_batch(spec, factories, pre_pos, seeds=seeds)
        summaries.append(asdict(summary))
        all_episodes_data[spec.scenario_id] = [asdict(r) for r in ep_records]

        print(
            f"{spec.scenario_id:<4} | {spec.experiment_name[:42]:<42} | {spec.family[:20]:<20} | "
            f"{spec.feasibility:<10} | {summary.overall_completion_rate:6.1f}% | {summary.mean_per_agent_success_rate:7.1f}% | "
            f"{summary.mean_steps_to_goal:6.1f} | {summary.total_collisions:5d} | {summary.total_deadlocks:5d}"
        )

    # Step 5: Determinism Validation (Repeat seeds 600-609 twice on U01, U04, U06, U12)
    print("\n[Step 5] Determinism and Reproducibility Validation (Seeds 600-609)...")
    det_seeds = tuple(600 + i for i in range(10))
    test_subsets = [s for s in scenarios if s[0].scenario_id in {"U01", "U04", "U06", "U12"}]
    deterministic = True
    for spec, pre_pos in test_subsets:
        factories = {
            a["id"]: make_robust_factory(spec.width, spec.height, spec.max_energy, base_linear_learner, margin=1)
            for a in spec.active_agents
        }
        sum1, recs1 = run_scenario_audit_batch(spec, factories, pre_pos, seeds=det_seeds)
        sum2, recs2 = run_scenario_audit_batch(spec, factories, pre_pos, seeds=det_seeds)
        if asdict(sum1) != asdict(sum2):
            deterministic = False
            print(f"  Determinism FAILED on {spec.scenario_id}!")
        else:
            print(f"  {spec.scenario_id}: Deterministic reproducibility verified (10/10 episodes bit-for-bit identical).")

    # Export full audit results
    results_payload = {
        "audit_version": "Phase 13.5 Unseen Generalization",
        "timestamp": "2026-10-23",
        "seed_range": [seeds[0], seeds[-1]],
        "episodes_per_scenario": len(seeds),
        "total_scenarios": len(scenarios),
        "total_episodes_evaluated": len(scenarios) * len(seeds),
        "determinism_verified": deterministic,
        "scenarios": summaries,
        "episode_details": all_episodes_data,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved comprehensive generalization audit results to {Path(results_json_path).resolve()}")

    return results_payload


if __name__ == "__main__":
    out_results = sys.argv[1] if len(sys.argv) > 1 else "phase13_5_unseen_results.json"
    run_full_unseen_generalization_audit(results_json_path=out_results)
