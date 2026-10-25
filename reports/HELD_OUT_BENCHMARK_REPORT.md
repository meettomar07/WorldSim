# WorldSim Held-Out Generalization & MAPF Baseline Benchmark Report

> **Generated:** 2026-10-25 17:16:48 UTC  
> **Protocol:** Genuinely Held-Out Evaluation (`Split.TEST`), Zero Prior Exposure  
> **Statistical Methods:** Wilson Score 95% Binomial CIs (Episode Unit), Cluster-Bootstrap 95% CIs (Environment Unit, B=1000, seed=42)  
> **Seeds Evaluated:** 10 seeds (300–309) | **Total Episodes:** 360

## 1. Executive Summary

This report documents the rigorous evaluation of WorldSim multi-agent coordination policies on 
structurally novel, held-out test scenarios (`TEST_H01`–`TEST_H06`). In accordance with research-grade 
methodology:
1. **Data Isolation:** All test scenarios were generated procedural-first and never used during training, tuning, or architecture selection.
2. **MAPF Baseline:** A standard Conflict-Based Search (CBS) solver (Sharon et al. 2012) was implemented from first principles with Space-Time A* and constraint tree branching.
3. **Statistical Defense:** Both episode-level Wilson score intervals and cluster-aware environment bootstrap intervals are reported to prevent overstating independent environmental evidence.
4. **Reproducibility:** Random seeds alter physical topologies verified by SHA-256 fingerprints, and zero numbers are hardcoded.

## 2. Experimental Scenario Split & Fingerprints

| Scenario ID | Name | Physical Variation Dimensions | Agents | Distinct Envs ($K$) | Repeated Envs ($N - K$) | Theoretical Cardinality |
|---|---|---|:---:|:---:|:---:|:---:|
| `TEST_H01` | Held-Out H01: Asymmetric Funnel Bottleneck | Procedural funnel constriction with variable ... | 4 | **8** / 10 | 2 | 16 (2 funnel widths x 2 cols x 4 agent row pairs) |
| `TEST_H02` | Held-Out H02: Double-Chamber Doorway Swap | Two rooms connected by a single doorway with ... | 2 | **9** / 10 | 1 | 120 (3 doorway rows x 5 west energies x 4 east energies x 2 rooms) |
| `TEST_H03` | Held-Out H03: Procedural Connected Maze | 8x8 obstacle field with guaranteed 4-corner r... | 4 | **10** / 10 | 0 | Combinatorial (8x8 obstacle field with BFS reachability) |
| `TEST_H04` | Held-Out H04: Multi-Lane Dynamic Hazard Highway | 4 agents crossing two randomized counter-flow... | 4 | **9** / 10 | 1 | Large (2 patrol velocities x 2 initial phases x 4 agent row pairs) |
| `TEST_H05` | Held-Out H05: Three-Way T-Junction Convergence | 3 agents meeting at a 1-cell T-junction from ... | 3 | **7** / 10 | 3 | 8 (2 north offsets x 2 west offsets x 2 east offsets) |
| `TEST_H06` | Held-Out H06: Critical Recharge Alcove Chokepoint | Single recharge station placed inside a dead-... | 2 | **7** / 10 | 3 | 16 (4 alcove columns x 2 west energies x 2 east energies) |

> [!NOTE]
> When multiple seeds sample the same physical configuration in a discrete finite parameter space, 
> treating all episodes as independent environment samples would overstate environmental diversity. 
> Section 3 reports both episode-level metrics (sampling unit: episodes) and cluster-aware bootstrap metrics (sampling unit: distinct environments).

## 3. Comparative Performance Across Held-Out Scenarios

### Scenario `TEST_H01`: Held-Out H01: Asymmetric Funnel Bottleneck

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 8 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 13.4 ± 1.3 | 0 | 0 | 0.0 | 144.0ms |
| `centralized` | 10 | 8 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 11.0 ± 0.0 | 0 | 0 | 0.0 | 10.0ms |
| `hybrid_comm` | 10 | 8 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 16.2 ± 2.9 | 0 | 0 | 0.0 | 10.1ms |
| `heuristic` | 10 | 8 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 11.0 ± 0.0 | 0 | 0 | 0.0 | 10.0ms |
| `learned_comm` | 10 | 8 | **30.0%** [10.8%, 60.3%] | **37.5%** [7.1%, 75.0%] | 82.5% [68.0%, 91.2%] | 12.0 ± 0.0 | 7 | 0 | 0.0 | 8.8ms |
| `learned_no_comm` | 10 | 8 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 8.8%] | N/A (0 succ) | 0 | 10 | 224.0 | 35.5ms |

### Scenario `TEST_H02`: Held-Out H02: Double-Chamber Doorway Swap

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 9 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.0 ± 0.0 | 0 | 0 | 0.0 | 1.5ms |
| `centralized` | 10 | 9 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 28.4ms |
| `hybrid_comm` | 10 | 9 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.4 | 3.1ms |
| `heuristic` | 10 | 9 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 28.4ms |
| `learned_comm` | 10 | 9 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.4 | 2.7ms |
| `learned_no_comm` | 10 | 9 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 20.2ms |

### Scenario `TEST_H03`: Held-Out H03: Procedural Connected Maze

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 10 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 14.0 ± 0.0 | 0 | 0 | 0.0 | 2.1ms |
| `centralized` | 10 | 10 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 20.4ms |
| `hybrid_comm` | 10 | 10 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 8.8ms |
| `heuristic` | 10 | 10 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 20.4ms |
| `learned_comm` | 10 | 10 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 7.7ms |
| `learned_no_comm` | 10 | 10 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 50.0% [35.2%, 64.8%] | N/A (0 succ) | 0 | 10 | 100.0 | 24.8ms |

### Scenario `TEST_H04`: Held-Out H04: Multi-Lane Dynamic Hazard Highway

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 9 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [91.2%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.0 | 3.0ms |
| `centralized` | 10 | 9 | **80.0%** [49.0%, 94.3%] | **77.8%** [50.0%, 100.0%] | 87.5% [73.9%, 94.5%] | 14.0 ± 2.3 | 0 | 2 | 0.2 | 16.8ms |
| `hybrid_comm` | 10 | 9 | **90.0%** [59.6%, 98.2%] | **88.9%** [66.7%, 100.0%] | 92.5% [80.1%, 97.4%] | 11.1 ± 0.8 | 1 | 0 | 0.0 | 9.5ms |
| `heuristic` | 10 | 9 | **80.0%** [49.0%, 94.3%] | **77.8%** [50.0%, 100.0%] | 87.5% [73.9%, 94.5%] | 14.0 ± 2.3 | 0 | 2 | 0.2 | 17.0ms |
| `learned_comm` | 10 | 9 | **90.0%** [59.6%, 98.2%] | **88.9%** [66.7%, 100.0%] | 92.5% [80.1%, 97.4%] | 11.1 ± 0.8 | 1 | 0 | 0.0 | 8.3ms |
| `learned_no_comm` | 10 | 9 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 15.0% [7.1%, 29.1%] | N/A (0 succ) | 0 | 10 | 194.6 | 34.7ms |

### Scenario `TEST_H05`: Held-Out H05: Three-Way T-Junction Convergence

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 9.1 ± 0.3 | 0 | 0 | 0.0 | 50.9ms |
| `centralized` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 5.8 ± 0.4 | 0 | 0 | 0.0 | 2.9ms |
| `hybrid_comm` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 5.8 ± 0.4 | 0 | 0 | 0.0 | 2.3ms |
| `heuristic` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 5.8 ± 0.4 | 0 | 0 | 0.0 | 2.9ms |
| `learned_comm` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 5.8 ± 0.4 | 0 | 0 | 0.0 | 2.0ms |
| `learned_no_comm` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [88.6%, 100.0%] | 5.8 ± 0.4 | 0 | 0 | 0.0 | 1.9ms |

### Scenario `TEST_H06`: Held-Out H06: Critical Recharge Alcove Chokepoint

| Policy | Episodes ($N$) | Envs ($K$) | Episode Completion (95% Wilson) | Cluster Completion (95% Bootstrap) | Per-Agent Success (95% Wilson) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | 7 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 10 | 0 | 0.0 | 1.0ms |
| `centralized` | 10 | 7 | **60.0%** [31.3%, 83.2%] | **71.4%** [27.3%, 100.0%] | 60.0% [38.7%, 78.1%] | 12.2 ± 0.4 | 0 | 4 | 46.4 | 10.9ms |
| `hybrid_comm` | 10 | 7 | **100.0%** [72.2%, 100.0%] | **100.0%** [100.0%, 100.0%] | 100.0% [83.9%, 100.0%] | 13.1 ± 1.3 | 0 | 0 | 0.8 | 4.5ms |
| `heuristic` | 10 | 7 | **60.0%** [31.3%, 83.2%] | **71.4%** [27.3%, 100.0%] | 60.0% [38.7%, 78.1%] | 12.2 ± 0.4 | 0 | 4 | 46.4 | 11.0ms |
| `learned_comm` | 10 | 7 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 10 | 0 | 1.2 | 2.1ms |
| `learned_no_comm` | 10 | 7 | **0.0%** [0.0%, 27.8%] | **0.0%** [0.0%, 0.0%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 2 | 8 | 92.6 | 14.8ms |

## 4. Overall Policy Benchmark Summary

| Policy | Total Episodes | Overall Completion (95% Wilson) | Overall Cluster Completion | Overall Agent Success | Total Exhaustions | Total Timeouts | Mean Step Count (Success) | Mean Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 60 | **83.3%** [72.0%, 90.7%] | **86.0%** | 89.5% | 10 | 0 | 10.9 | 33.8ms |
| `centralized` | 60 | **73.3%** [61.0%, 82.9%] | **74.0%** | 82.6% | 0 | 16 | 11.4 | 14.9ms |
| `hybrid_comm` | 60 | **98.3%** [91.1%, 99.7%] | **98.0%** | 98.4% | 1 | 0 | 11.7 | 6.4ms |
| `heuristic` | 60 | **73.3%** [61.0%, 82.9%] | **74.0%** | 82.6% | 0 | 16 | 11.4 | 14.9ms |
| `learned_comm` | 60 | **70.0%** [57.5%, 80.1%] | **74.0%** | 84.2% | 18 | 0 | 10.4 | 5.3ms |
| `learned_no_comm` | 60 | **16.7%** [9.3%, 28.0%] | **14.0%** | 29.5% | 2 | 48 | 5.8 | 22.0ms |

## 5. Paired Head-to-Head Comparisons (Preserving Matched Environments)

To account for environment-level difficulty variation, policies are paired on identical environment instances:

| Comparison (Policy A vs. Policy B) | Matched Pairs | Wins A | Wins B | Ties | Win Rate A | Win Rate B | Mean Step Diff (A - B) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `hybrid_comm` vs. `cbs` | 60 | **10** | 1 | 49 | **16.7%** | 1.7% | +0.5 steps |
| `hybrid_comm` vs. `centralized` | 60 | **16** | 1 | 43 | **26.7%** | 1.7% | +0.7 steps |
| `hybrid_comm` vs. `heuristic` | 60 | **16** | 1 | 43 | **26.7%** | 1.7% | +0.7 steps |
| `learned_comm` vs. `learned_no_comm` | 60 | **32** | 0 | 28 | **53.3%** | 0.0% | +0.0 steps |
| `cbs` vs. `centralized` | 60 | **12** | 6 | 42 | **20.0%** | 10.0% | +0.2 steps |

## 6. Conflict-Based Search (CBS) Baseline Analysis

### 6.1 Formulation & Implementation
Conflict-Based Search was integrated into `worldsim/cbs.py` with zero external dependencies:
- **High-Level Constraint Tree:** Best-first search over vertex and edge-swap conflicts, prioritizing nodes by Sum of Costs (SIC).
- **Low-Level Space-Time A*:** Single-agent dynamic shortest-path search in $(x, y, t)$ state-space respecting accumulated vertex constraints, edge-swap constraints, and grid wall boundaries.
- **Deterministic Tie-Breaking:** Canonical deterministic order ensures bit-for-bit reproducible search trees.
- **Dynamic Hazard Scheduling:** Extends Space-Time A* to reserve pre-computed patrol trajectories for moving obstacles.

### 6.2 CBS vs. Omniscient Centralized Coordinator
It is critical to distinguish CBS from WorldSim's `centralized` coordinator:
1. **Offline Trajectory Planning (CBS):** Computes complete paths from initial positions to targets *before* execution begins. Guarantees global optimality on static graphs under unconstrained energy. However, it requires complete global foresight.
2. **Online Reactive Arbitration (`centralized`):** Observes local moves every tick and resolves immediate step conflicts (e.g., via priority rotation or resource queuing). Capable of managing multi-agent recharge contention dynamically as batteries deplete.
3. **Energy & Recharge Limitations in Classical MAPF:** In `TEST_H06` (Critical Recharge Choke), agents require an intermediate recharge detour to reach their destination. Standard CBS seeks the shortest physical path to the target without modeling state-dependent battery drain and recharge queuing, leading to exhaustion. This highlights the precise capability boundary where WorldSim's energy-aware policies outperform classical MAPF formulations.

## 7. Split Isolation Guarantees & Reproducibility

### Code-Level Split Enforcement vs. Historical Claims
- **Code-Level Guarantee:** The scenario registry enforces `assert_not_held_out` and `assert_split(..., allowed_splits=Split.TEST)`. Any attempt by training harnesses or hyperparameter tuners to query `Split.TEST` triggers a hard `RuntimeError`.
- **Historical Disclosure:** Scenarios `VAL_U01`–`VAL_U16` and `VAL_D01`–`VAL_D04` were exposed during Phases 13–14 development and are explicitly relegated to `Split.VAL`. Only `TEST_H01`–`TEST_H06` are recognized as `Split.TEST`.

All reported metrics are computed directly from the raw episode records stored in 
`benchmarks/held_out_generalization_results.json`. The entire experiment can be reproduced with:
```bash
python -m experiments.eval_heldout --seeds 10
```
