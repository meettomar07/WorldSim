"""WorldSim Statistical Rigor & Metric Analysis Suite.

Provides mathematically principled confidence intervals and hypothesis testing:
- Wilson score intervals for binomial completion and success rates (handling p=0 and p=1).
- Cluster bootstrap intervals resampling environment fingerprints to account for repeated environments.
- Conditional statistics for completion steps (conditional on successful episodes).
- Paired comparison metrics for matched-seed environment evaluations.
- JSON-serializable statistical summary models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Any, Sequence


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Sampling unit: Individual episode outcomes.
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


def cluster_bootstrap_interval(
    episodes: Sequence[dict[str, Any]],
    metric_key: str = "completed",
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    min_clusters: int = 5,
) -> tuple[float | None, float | None, str]:
    """Calculate a cluster-aware bootstrap confidence interval resampling environments.

    Groups episodes by 'environment_fingerprint'.
    - If distinct clusters K < min_clusters, returns (None, None, status)
      because cluster bootstrap with < 5 clusters is severely degenerate.
    - Otherwise, performs cluster bootstrap drawing K clusters with replacement
      `num_resamples` times with fixed `seed`.
    - Returns (ci_lower, ci_upper, status_description).
    """
    if not episodes:
        return None, None, "EMPTY_DATA"

    # Group by cluster
    clusters: dict[str, list[float]] = {}
    for ep in episodes:
        fp = ep.get("environment_fingerprint", "unknown")
        val = 1.0 if ep.get(metric_key, False) else 0.0
        clusters.setdefault(fp, []).append(val)

    cluster_keys = sorted(list(clusters.keys()))
    k = len(cluster_keys)

    if k < min_clusters:
        return None, None, f"INSUFFICIENT_CLUSTERS (K={k} < {min_clusters})"

    # Check for all-success or all-fail boundary cases
    all_vals = [v for cl in clusters.values() for v in cl]
    if all(v == 1.0 for v in all_vals):
        return 1.0, 1.0, f"Cluster Bootstrap (B={num_resamples}, seed={seed}, K={k}, all 1.0)"
    if all(v == 0.0 for v in all_vals):
        return 0.0, 0.0, f"Cluster Bootstrap (B={num_resamples}, seed={seed}, K={k}, all 0.0)"

    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(num_resamples):
        # Sample k cluster keys with replacement
        sampled_keys = [cluster_keys[rng.randint(0, k - 1)] for _ in range(k)]
        sampled_vals = [v for key in sampled_keys for v in clusters[key]]
        b_mean = sum(sampled_vals) / len(sampled_vals) if sampled_vals else 0.0
        bootstrap_means.append(b_mean)

    bootstrap_means.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor(num_resamples * (alpha / 2.0))))
    upper_idx = min(num_resamples - 1, int(math.ceil(num_resamples * (1.0 - alpha / 2.0))) - 1)

    ci_lower = max(0.0, min(1.0, bootstrap_means[lower_idx]))
    ci_upper = max(0.0, min(1.0, bootstrap_means[upper_idx]))

    return (
        round(ci_lower, 4),
        round(ci_upper, 4),
        f"Cluster Bootstrap (B={num_resamples}, seed={seed}, K={k})",
    )


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
class PairedComparisonSummary:
    """Exact paired environment comparison between two policies evaluated on identical environments."""

    policy_a: str
    policy_b: str
    num_matched_pairs: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    win_rate_b: float
    mean_step_diff: float | None  # step_a - step_b (conditional on both succeeding)


def compute_paired_comparison(
    episodes_a: Sequence[dict[str, Any]],
    episodes_b: Sequence[dict[str, Any]],
    policy_name_a: str,
    policy_name_b: str,
) -> PairedComparisonSummary:
    """Match episodes on (scenario_id, env_seed) and compute paired win/loss/tie outcomes."""
    lookup_b: dict[tuple[str, int], dict[str, Any]] = {
        (ep["scenario_id"], ep["env_seed"]): ep for ep in episodes_b
    }

    matched_keys = sorted(
        [
            (ep["scenario_id"], ep["env_seed"])
            for ep in episodes_a
            if (ep["scenario_id"], ep["env_seed"]) in lookup_b
        ]
    )

    lookup_a = {(ep["scenario_id"], ep["env_seed"]): ep for ep in episodes_a}

    wins_a = 0
    wins_b = 0
    ties = 0
    step_diffs: list[float] = []

    for key in matched_keys:
        ea = lookup_a[key]
        eb = lookup_b[key]

        comp_a = bool(ea.get("completed", False))
        comp_b = bool(eb.get("completed", False))

        if comp_a and not comp_b:
            wins_a += 1
        elif comp_b and not comp_a:
            wins_b += 1
        else:
            ties += 1

        if comp_a and comp_b:
            diff = float(ea.get("steps", 0)) - float(eb.get("steps", 0))
            step_diffs.append(diff)

    n_matched = len(matched_keys)
    wr_a = round(wins_a / n_matched, 4) if n_matched > 0 else 0.0
    wr_b = round(wins_b / n_matched, 4) if n_matched > 0 else 0.0
    m_diff = round(statistics.mean(step_diffs), 2) if step_diffs else None

    return PairedComparisonSummary(
        policy_a=policy_name_a,
        policy_b=policy_name_b,
        num_matched_pairs=n_matched,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_a=wr_a,
        win_rate_b=wr_b,
        mean_step_diff=m_diff,
    )


@dataclass
class ScenarioStatisticalSummary:
    """Comprehensive, defensible evaluation record for a policy-scenario pair."""

    scenario_id: str
    scenario_name: str
    policy: str
    num_episodes: int
    num_distinct_environments: int
    num_repeated_environments: int
    environment_fingerprints: list[str]

    # Whole-episode completion (Episode sampling unit)
    completed_episodes: int
    completion_rate: float
    completion_ci_95: tuple[float, float]  # Wilson 95% CI (lower, upper)
    episode_completion_rate: float
    episode_ci_95: tuple[float, float]

    # Cluster-aware completion (Environment sampling unit)
    cluster_completion_rate: float
    cluster_ci_95: tuple[float | None, float | None]  # Percentile bootstrap or None if K < 5
    cluster_ci_method: str

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

    # Collisions
    collision_episodes: int
    collision_rate: float

    # Path length & efficiency (conditional on success)
    success_steps: ContinuousMetricSummary
    all_episode_steps: ContinuousMetricSummary

    # Collisions & safety continuous metrics
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
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> ScenarioStatisticalSummary:
    """Aggregate raw episode results into a rigorous statistical summary."""
    n = len(raw_episodes)
    if n == 0:
        raise ValueError("Cannot aggregate empty episode list")

    fingerprints = sorted(list({ep["environment_fingerprint"] for ep in raw_episodes}))
    distinct_envs = len(fingerprints)
    repeated_envs = max(0, n - distinct_envs)

    # 1. Episode-level completion
    completed_eps = sum(1 for ep in raw_episodes if ep["completed"])
    comp_rate = completed_eps / n
    comp_ci = wilson_score_interval(completed_eps, n, 0.95)

    # 2. Cluster-aware environment-level completion
    env_clusters: dict[str, list[float]] = {}
    for ep in raw_episodes:
        fp = ep["environment_fingerprint"]
        env_clusters.setdefault(fp, []).append(1.0 if ep["completed"] else 0.0)

    cluster_means = [sum(v) / len(v) for v in env_clusters.values()]
    cluster_comp_rate = statistics.mean(cluster_means) if cluster_means else 0.0

    cl_lower, cl_upper, cl_method = cluster_bootstrap_interval(
        raw_episodes,
        metric_key="completed",
        num_resamples=bootstrap_resamples,
        confidence=0.95,
        seed=bootstrap_seed,
        min_clusters=5,
    )

    # 3. Per-agent success
    total_agents = sum(ep["num_agents"] for ep in raw_episodes)
    succ_agents = sum(ep["successful_agents"] for ep in raw_episodes)
    agent_rate = succ_agents / total_agents if total_agents > 0 else 0.0
    agent_ci = wilson_score_interval(succ_agents, total_agents, 0.95)

    # 4. Failure mode counts
    exhausted_eps = sum(1 for ep in raw_episodes if ep.get("exhausted", False))
    timeout_eps = sum(1 for ep in raw_episodes if ep.get("timed_out", False))
    deadlock_eps = sum(1 for ep in raw_episodes if ep.get("deadlocked", False))

    col_eps = sum(1 for ep in raw_episodes if ep.get("actual_collisions", 0) > 0)
    col_rate = round(col_eps / n, 4)

    # 5. Continuous metric distributions
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
        num_repeated_environments=repeated_envs,
        environment_fingerprints=fingerprints,
        completed_episodes=completed_eps,
        completion_rate=round(comp_rate, 4),
        completion_ci_95=comp_ci,
        episode_completion_rate=round(comp_rate, 4),
        episode_ci_95=comp_ci,
        cluster_completion_rate=round(cluster_comp_rate, 4),
        cluster_ci_95=(cl_lower, cl_upper),
        cluster_ci_method=cl_method,
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
        collision_episodes=col_eps,
        collision_rate=col_rate,
        success_steps=summarize_continuous(success_step_vals),
        all_episode_steps=summarize_continuous(all_step_vals),
        actual_collisions=summarize_continuous(act_col_vals),
        blocked_attempts=summarize_continuous(blk_vals),
        runtime_ms=summarize_continuous(runtime_vals),
        solver_expansions=expansion_summary,
    )
