"""WorldSim Statistical Rigor & Metric Analysis Suite.

Provides mathematically principled confidence intervals and hypothesis testing:
- Wilson score intervals for binomial completion and success rates (handling p=0 and p=1).
- Conditional statistics for completion steps (conditional on successful episodes).
- Paired comparison metrics for matched-seed environment evaluations.
- JSON-serializable statistical summary models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Sequence


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.
    
    Correctly bounded in [0, 1] and valid for small sample sizes and boundary
    cases (successes=0 or successes=trials).
    
    Standard normal critical values:
    - 90%: z = 1.64485
    - 95%: z = 1.95996
    - 99%: z = 2.57583
    """
    if trials <= 0:
        return 0.0, 0.0
    
    if confidence == 0.90:
        z = 1.64485
    elif confidence == 0.99:
        z = 2.57583
    else:  # default 0.95
        z = 1.95996

    p_hat = successes / trials
    z2 = z * z
    n = float(trials)

    denominator = 1.0 + z2 / n
    center_adj = p_hat + z2 / (2.0 * n)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)

    lower = max(0.0, (center_adj - margin) / denominator)
    upper = min(1.0, (center_adj + margin) / denominator)

    return round(lower, 4), round(upper, 4)


@dataclass(frozen=True)
class ContinuousMetricSummary:
    """Statistical summary for continuous metrics (e.g. steps, collisions, energy)."""
    count: int
    mean: float | None
    std: float | None
    median: float | None
    min: float | None
    max: float | None


def summarize_continuous(values: Sequence[float | int]) -> ContinuousMetricSummary:
    """Compute summary statistics for continuous measurements."""
    if not values:
        return ContinuousMetricSummary(count=0, mean=None, std=None, median=None, min=None, max=None)
    
    float_vals = [float(v) for v in values]
    n = len(float_vals)
    mean_val = statistics.mean(float_vals)
    std_val = statistics.stdev(float_vals) if n > 1 else 0.0
    med_val = statistics.median(float_vals)
    min_val = min(float_vals)
    max_val = max(float_vals)

    return ContinuousMetricSummary(
        count=n,
        mean=round(mean_val, 3),
        std=round(std_val, 3),
        median=round(med_val, 3),
        min=round(min_val, 3),
        max=round(max_val, 3),
    )


@dataclass
class ScenarioStatisticalSummary:
    """Comprehensive, defensible evaluation record for a policy-scenario pair."""
    scenario_id: str
    scenario_name: str
    policy: str
    num_episodes: int
    num_distinct_environments: int
    environment_fingerprints: list[str]

    # Whole-episode completion
    completed_episodes: int
    completion_rate: float
    completion_ci_95: tuple[float, float]  # Wilson 95% CI (lower, upper)

    # Per-agent success
    total_agent_slots: int
    successful_agent_slots: int
    per_agent_success_rate: float
    per_agent_ci_95: tuple[float, float]

    # Failure mode counts and rates
    exhaustion_count: int
    exhaustion_rate: float
    timeout_count: int
    timeout_rate: float
    deadlock_count: int
    deadlock_rate: float

    # Path length & efficiency (conditional on success)
    success_steps: ContinuousMetricSummary
    all_episode_steps: ContinuousMetricSummary

    # Collisions & safety
    actual_collisions: ContinuousMetricSummary  # vertex + swap
    blocked_attempts: ContinuousMetricSummary   # occupied

    # Runtime & solver performance
    runtime_ms: ContinuousMetricSummary
    solver_expansions: ContinuousMetricSummary | None = None


def aggregate_raw_episodes(
    scenario_id: str,
    scenario_name: str,
    policy_name: str,
    raw_episodes: list[dict[str, Any]],
) -> ScenarioStatisticalSummary:
    """Aggregate raw episode results into a rigorous statistical summary."""
    n = len(raw_episodes)
    if n == 0:
        raise ValueError("Cannot aggregate empty episode list")

    fingerprints = sorted(list({ep["environment_fingerprint"] for ep in raw_episodes}))
    distinct_envs = len(fingerprints)

    completed_eps = sum(1 for ep in raw_episodes if ep["completed"])
    comp_rate = completed_eps / n
    comp_ci = wilson_score_interval(completed_eps, n, 0.95)

    total_agents = sum(ep["num_agents"] for ep in raw_episodes)
    succ_agents = sum(ep["successful_agents"] for ep in raw_episodes)
    agent_rate = succ_agents / total_agents if total_agents > 0 else 0.0
    agent_ci = wilson_score_interval(succ_agents, total_agents, 0.95)

    exhausted_eps = sum(1 for ep in raw_episodes if ep.get("exhausted", False))
    timeout_eps = sum(1 for ep in raw_episodes if ep.get("timed_out", False))
    deadlock_eps = sum(1 for ep in raw_episodes if ep.get("deadlocked", False))

    success_step_vals = [ep["steps"] for ep in raw_episodes if ep["completed"]]
    all_step_vals = [ep["steps"] for ep in raw_episodes]
    act_col_vals = [ep.get("actual_collisions", 0) for ep in raw_episodes]
    blk_vals = [ep.get("blocked_attempts", 0) for ep in raw_episodes]
    runtime_vals = [ep.get("runtime_ms", 0.0) for ep in raw_episodes]

    expansions = [ep["solver_expansions"] for ep in raw_episodes if "solver_expansions" in ep]
    expansion_summary = summarize_continuous(expansions) if expansions else None

    return ScenarioStatisticalSummary(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        policy=policy_name,
        num_episodes=n,
        num_distinct_environments=distinct_envs,
        environment_fingerprints=fingerprints,
        completed_episodes=completed_eps,
        completion_rate=round(comp_rate, 4),
        completion_ci_95=comp_ci,
        total_agent_slots=total_agents,
        successful_agent_slots=succ_agents,
        per_agent_success_rate=round(agent_rate, 4),
        per_agent_ci_95=agent_ci,
        exhaustion_count=exhausted_eps,
        exhaustion_rate=round(exhausted_eps / n, 4),
        timeout_count=timeout_eps,
        timeout_rate=round(timeout_eps / n, 4),
        deadlock_count=deadlock_eps,
        deadlock_rate=round(deadlock_eps / n, 4),
        success_steps=summarize_continuous(success_step_vals),
        all_episode_steps=summarize_continuous(all_step_vals),
        actual_collisions=summarize_continuous(act_col_vals),
        blocked_attempts=summarize_continuous(blk_vals),
        runtime_ms=summarize_continuous(runtime_vals),
        solver_expansions=expansion_summary,
    )
