# WorldSim Held-Out Generalization & MAPF Baseline Benchmark Report

> **Generated:** 2026-10-25 16:50:49 UTC  
> **Protocol:** Genuinely Held-Out Evaluation (`Split.TEST`), Zero Prior Exposure  
> **Statistical Method:** Wilson Score 95% Binomial CIs, Conditional Step Distributions, Canonical Fingerprinting  
> **Seeds Evaluated:** 10 seeds (300–309) | **Total Episodes:** 360

## 1. Executive Summary

This report documents the rigorous evaluation of WorldSim multi-agent coordination policies on 
structurally novel, held-out test scenarios (`TEST_H01`–`TEST_H06`). In accordance with research-grade 
methodology:
1. **Data Isolation:** All test scenarios were generated procedural-first and never used during training, tuning, or architecture selection.
2. **MAPF Baseline:** A standard Conflict-Based Search (CBS) solver (Sharon et al. 2012) was implemented from first principles with Space-Time A* and constraint tree branching.
3. **Statistical Defense:** All binary rates report Wilson score 95% confidence intervals; step counts are reported conditional on success; and random seeds alter physical topologies verified by SHA-256 fingerprints.

## 2. Experimental Scenario Split & Fingerprints

| Scenario ID | Name | Topology & Dynamic Features | Agents | Energy | Distinct Fingerprints |
|---|---|---|:---:|:---:|:---:|
| `TEST_H01` | Held-Out H01: Asymmetric Funnel Bottleneck | Procedural funnel constriction with variable corri... | 4 | 8–18 | **8** / 10 seeds |
| `TEST_H02` | Held-Out H02: Double-Chamber Doorway Swap | Two rooms connected by a single doorway with asymm... | 2 | 8–18 | **9** / 10 seeds |
| `TEST_H03` | Held-Out H03: Procedural Connected Maze | 8x8 obstacle field with guaranteed 4-corner reacha... | 4 | 8–18 | **10** / 10 seeds |
| `TEST_H04` | Held-Out H04: Multi-Lane Dynamic Hazard Highway | 4 agents crossing two randomized counter-flowing h... | 4 | 8–18 | **9** / 10 seeds |
| `TEST_H05` | Held-Out H05: Three-Way T-Junction Convergence | 3 agents meeting at a 1-cell T-junction from ortho... | 3 | 8–18 | **2** / 10 seeds |
| `TEST_H06` | Held-Out H06: Critical Recharge Alcove Chokepoint | Single recharge station placed inside a dead-end a... | 2 | 8–18 | **2** / 10 seeds |

## 3. Comparative Performance Across Held-Out Scenarios

For each scenario and policy, we report:
- **Episode Completion Rate** with Wilson score 95% confidence interval.
- **Per-Agent Success Rate** (fraction of individual agent target arrivals).
- **Failure Modes:** Energy Exhaustion, Deadlock, and Step Timeout.
- **Efficiency:** Mean and standard deviation of steps *conditional on successful episodes*.
- **Runtime:** Mean solver/coordination runtime in milliseconds.

### Scenario `TEST_H01`: Held-Out H01: Asymmetric Funnel Bottleneck

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 13.4 ± 1.3 | 0 | 0 | 0.0 | 156.5ms |
| `centralized` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 11.0 ± 0.0 | 0 | 0 | 0.0 | 12.1ms |
| `hybrid_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 16.2 ± 2.9 | 0 | 0 | 0.0 | 11.0ms |
| `heuristic` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 11.0 ± 0.0 | 0 | 0 | 0.0 | 10.7ms |
| `learned_comm` | 10 | **30.0%** [10.8%, 60.3%] | 82.5% [68.0%, 91.2%] | 12.0 ± 0.0 | 7 | 0 | 0.0 | 9.4ms |
| `learned_no_comm` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 8.8%] | N/A (0 succ) | 0 | 10 | 224.0 | 38.5ms |

### Scenario `TEST_H02`: Held-Out H02: Double-Chamber Doorway Swap

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.0 ± 0.0 | 0 | 0 | 0.0 | 1.6ms |
| `centralized` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 31.4ms |
| `hybrid_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.4 | 3.5ms |
| `heuristic` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 31.9ms |
| `learned_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.4 | 3.1ms |
| `learned_no_comm` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 0 | 10 | 116.0 | 22.2ms |

### Scenario `TEST_H03`: Held-Out H03: Procedural Connected Maze

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 14.0 ± 0.0 | 0 | 0 | 0.0 | 2.3ms |
| `centralized` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 22.4ms |
| `hybrid_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 9.8ms |
| `heuristic` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 22.7ms |
| `learned_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 15.0 ± 0.0 | 0 | 0 | 0.0 | 8.4ms |
| `learned_no_comm` | 10 | **0.0%** [0.0%, 27.8%] | 50.0% [35.2%, 64.8%] | N/A (0 succ) | 0 | 10 | 100.0 | 27.1ms |

### Scenario `TEST_H04`: Held-Out H04: Multi-Lane Dynamic Hazard Highway

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [91.2%, 100.0%] | 9.2 ± 0.4 | 0 | 0 | 0.0 | 3.3ms |
| `centralized` | 10 | **80.0%** [49.0%, 94.3%] | 87.5% [73.9%, 94.5%] | 14.0 ± 2.3 | 0 | 2 | 0.2 | 18.7ms |
| `hybrid_comm` | 10 | **90.0%** [59.6%, 98.2%] | 92.5% [80.1%, 97.4%] | 11.1 ± 0.8 | 1 | 0 | 0.0 | 11.5ms |
| `heuristic` | 10 | **80.0%** [49.0%, 94.3%] | 87.5% [73.9%, 94.5%] | 14.0 ± 2.3 | 0 | 2 | 0.2 | 18.7ms |
| `learned_comm` | 10 | **90.0%** [59.6%, 98.2%] | 92.5% [80.1%, 97.4%] | 11.1 ± 0.8 | 1 | 0 | 0.0 | 9.2ms |
| `learned_no_comm` | 10 | **0.0%** [0.0%, 27.8%] | 15.0% [7.1%, 29.1%] | N/A (0 succ) | 0 | 10 | 194.6 | 38.8ms |

### Scenario `TEST_H05`: Held-Out H05: Three-Way T-Junction Convergence

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 9.5 ± 0.5 | 0 | 0 | 0.0 | 58.0ms |
| `centralized` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 6.0 ± 0.0 | 0 | 0 | 0.0 | 3.4ms |
| `hybrid_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 6.0 ± 0.0 | 0 | 0 | 0.0 | 2.6ms |
| `heuristic` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 6.0 ± 0.0 | 0 | 0 | 0.0 | 3.4ms |
| `learned_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 6.0 ± 0.0 | 0 | 0 | 0.0 | 2.2ms |
| `learned_no_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [88.6%, 100.0%] | 6.0 ± 0.0 | 0 | 0 | 0.0 | 2.1ms |

### Scenario `TEST_H06`: Held-Out H06: Critical Recharge Alcove Chokepoint

| Policy | Episodes | Completion Rate (95% CI) | Per-Agent Rate (95% CI) | Steps (Mean ± Std) | Exhausted | Timed Out | Collisions | Runtime (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 10 | 0 | 0.0 | 0.7ms |
| `centralized` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 13.0 ± 0.0 | 0 | 0 | 0.0 | 4.1ms |
| `hybrid_comm` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 13.0 ± 0.0 | 0 | 0 | 0.0 | 4.7ms |
| `heuristic` | 10 | **100.0%** [72.2%, 100.0%] | 100.0% [83.9%, 100.0%] | 13.0 ± 0.0 | 0 | 0 | 0.0 | 4.1ms |
| `learned_comm` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 10 | 0 | 0.0 | 2.4ms |
| `learned_no_comm` | 10 | **0.0%** [0.0%, 27.8%] | 0.0% [0.0%, 16.1%] | N/A (0 succ) | 5 | 5 | 57.0 | 10.3ms |

## 4. Overall Policy Benchmark Summary

| Policy | Total Episodes | Overall Completion (95% CI) | Overall Agent Success | Total Exhaustions | Total Timeouts | Mean Step Count (Success) | Mean Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `cbs` | 60 | **83.3%** [72.0%, 90.7%] | 89.5% | 10 | 0 | 11.0 | 37.1ms |
| `centralized` | 60 | **80.0%** [68.2%, 88.2%] | 86.8% | 0 | 12 | 11.7 | 15.4ms |
| `hybrid_comm` | 60 | **98.3%** [91.1%, 99.7%] | 98.4% | 1 | 0 | 11.8 | 7.2ms |
| `heuristic` | 60 | **80.0%** [68.2%, 88.2%] | 86.8% | 0 | 12 | 11.7 | 15.3ms |
| `learned_comm` | 60 | **70.0%** [57.5%, 80.1%] | 84.2% | 18 | 0 | 10.4 | 5.8ms |
| `learned_no_comm` | 60 | **16.7%** [9.3%, 28.0%] | 29.5% | 5 | 45 | 6.0 | 23.2ms |

## 5. Conflict-Based Search (CBS) Baseline Analysis

### 5.1 Formulation & Implementation
Conflict-Based Search was integrated into `worldsim/cbs.py` with:
- **High-Level Constraint Tree:** Best-first search over vertex and edge-swap conflicts, prioritizing nodes by Sum of Costs (SIC).
- **Low-Level Space-Time A*:** Single-agent dynamic shortest-path search in $(x, y, t)$ state-space respecting accumulated vertex constraints, edge-swap constraints, and grid wall boundaries.
- **Deterministic Tie-Breaking:** Canonical deterministic order ensures bit-for-bit reproducible search trees.
- **Dynamic Hazard Scheduling:** Extends Space-Time A* to reserve pre-computed patrol trajectories for moving obstacles.

### 5.2 CBS vs. Omniscient Centralized Coordinator
It is critical to distinguish CBS from WorldSim's `centralized` coordinator:
1. **Offline Trajectory Planning (CBS):** Computes complete paths from initial positions to targets *before* execution begins. Guarantees global optimality on static graphs under unconstrained energy. However, it requires complete global foresight.
2. **Online Reactive Arbitration (`centralized`):** Observes local moves every tick and resolves immediate step conflicts (e.g., via priority rotation or resource queuing). Capable of managing multi-agent recharge contention dynamically as batteries deplete.
3. **Energy & Recharge Limitations in Classical MAPF:** In `TEST_H06` (Critical Recharge Choke), agents require an intermediate recharge detour to reach their destination. Standard CBS seeks the shortest physical path to the target without modeling state-dependent battery drain and recharge queuing, leading to exhaustion. This highlights the precise capability boundary where WorldSim's energy-aware policies outperform classical MAPF formulations.

## 6. Verification and Reproducibility

All reported metrics are computed directly from the raw episode records stored in 
`benchmarks/held_out_generalization_results.json`. The entire experiment can be reproduced with:
```bash
python -m experiments.eval_heldout --seeds 10
```
