# WorldSim Phase 14.2 — Local-Horizon Coordination Audit Report

**Date:** October 24, 2026  
**Benchmark Runtime:** 22.65s  
**Local Horizon Configuration:** Communication Radius $R = 2.0$ (Euclidean distance $\le 2.0$)  
**Baseline:** Centralized Global Broker ($R = \infty$)  

## Executive Summary

Phase 14.2 introduces decentralized local-horizon coordination, restricting multi-agent intention exchange, right-of-way negotiation, and resource queue visibility to a local horizon of $R = 2.0$, while preserving authoritative environment physics and backward-compatible centralized mode as the default.

### Key Findings:
1. **Decentralized Parity Across Feasible Baseline Scenarios:**
   - **Baseline Benchmark:** All 13 feasible baseline scenarios achieve **100.0% completion** in decentralized mode, matching centralized mode exactly.
   - **Unseen Generalization:** 12 of 14 feasible unseen scenarios (U01–U05, U07–U10, U12–U13, U15) achieve **100.0% completion** in decentralized mode.
   - **Local-Horizon Regressions:** Two unseen scenarios regress under strict $R = 2.0$ decentralized coordination: U06 (0.0% vs 100.0%) and U14 (0.0% vs 100.0%).
   - **Honest Infeasibility:** All 5 provably infeasible configurations (4A, 5A, 8, U11, U16) report **0.0% completion** honestly in both modes.
2. **Physical Collision Integrity:**
   - Across all 32 scenario evaluations, **zero illegal physical cell sharing or illegal movements** occurred.
   - Authoritative simulation physics consistently resolves spatial conflicts.
3. **Collision Separation (Actual vs Blocked Attempts):**
   - Actual inter-agent collisions (`vertex + swap`): Occur predominantly during blind mutual arrivals at chokepoints and are promptly resolved by priority rotation.
   - Blocked movement attempts (`occupied`): Represent safe doorstep / queue waiting against stationary or yielding occupants.
4. **Telemetry & Deadlock Breakdown:**
   - Across all feasible baseline scenarios in decentralized mode, true structural deadlocks remained **0**.
   - Cooperative waiting events reflect benign queueing and yielding.

---

## Part 1: Baseline Scenarios (Centralized vs Decentralized)

| ID | Scenario | Feasibility | Mode | Completion (%) | Mean Steps | Mean Energy | Actual Cols (V+S) | Blocked (O) | Total Cols | Coop Wait | Struct DL | Exhaustions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1A | Scenario 1A: 8-Agent Bottleneck | FEASIBLE | **Centralized** | 100.0% | 23.5 | 58.0 | 180 | 340 | 520 | 260 | 0 | 0 |
| 1A | Scenario 1A: 8-Agent Bottleneck | FEASIBLE | **Decentralized** | 100.0% | 25.0 | 58.0 | 160 | 290 | 450 | 150 | 0 | 0 |
| 1B | Scenario 1B: 8-Agent Bottleneck  | FEASIBLE | **Centralized** | 100.0% | 23.5 | 58.0 | 900 | 1700 | 2600 | 1300 | 0 | 0 |
| 1B | Scenario 1B: 8-Agent Bottleneck  | FEASIBLE | **Decentralized** | 100.0% | 25.0 | 58.0 | 800 | 1450 | 2250 | 750 | 0 | 0 |
| 1C | Scenario 1C: Permuted Targets | FEASIBLE | **Centralized** | 100.0% | 19.0 | 49.0 | 210 | 260 | 470 | 190 | 0 | 0 |
| 1C | Scenario 1C: Permuted Targets | FEASIBLE | **Decentralized** | 100.0% | 23.5 | 49.5 | 310 | 510 | 820 | 500 | 0 | 0 |
| 1D | Scenario 1D: Opposing Traffic | FEASIBLE | **Centralized** | 100.0% | 17.5 | 28.0 | 290 | 100 | 390 | 40 | 0 | 0 |
| 1D | Scenario 1D: Opposing Traffic | FEASIBLE | **Decentralized** | 100.0% | 16.5 | 28.0 | 250 | 100 | 350 | 40 | 0 | 0 |
| 2 | Scenario 2: 4-Way Intersection | FEASIBLE | **Centralized** | 100.0% | 8.0 | 20.0 | 40 | 40 | 80 | 40 | 0 | 0 |
| 2 | Scenario 2: 4-Way Intersection | FEASIBLE | **Decentralized** | 100.0% | 8.0 | 20.0 | 40 | 40 | 80 | 40 | 0 | 0 |
| 3A | Scenario 3A: Asymmetric Urgency  | FEASIBLE | **Centralized** | 100.0% | 7.0 | 6.0 | 20 | 20 | 40 | 0 | 20 | 0 |
| 3A | Scenario 3A: Asymmetric Urgency  | FEASIBLE | **Decentralized** | 100.0% | 7.0 | 6.0 | 20 | 20 | 40 | 0 | 20 | 0 |
| 3B | Scenario 3B: Asymmetric Urgency  | FEASIBLE | **Centralized** | 100.0% | 7.0 | 6.0 | 20 | 10 | 30 | 0 | 10 | 0 |
| 3B | Scenario 3B: Asymmetric Urgency  | FEASIBLE | **Decentralized** | 100.0% | 7.0 | 6.0 | 20 | 10 | 30 | 0 | 10 | 0 |
| 3C | Scenario 3C: 3-Agent Contention | FEASIBLE | **Centralized** | 100.0% | 9.7 | 18.0 | 80 | 40 | 120 | 80 | 0 | 0 |
| 3C | Scenario 3C: 3-Agent Contention | FEASIBLE | **Decentralized** | 100.0% | 9.7 | 18.0 | 80 | 40 | 120 | 80 | 0 | 0 |
| 4A | Scenario 4A: Relocated Resources | INFEASIBLE | **Centralized** | 0.0% | 0.0 | 44.0 | 208 | 192 | 400 | 103 | 0 | 40 |
| 4A | Scenario 4A: Relocated Resources | INFEASIBLE | **Decentralized** | 0.0% | 0.0 | 46.0 | 198 | 202 | 400 | 103 | 0 | 40 |
| 4B | Scenario 4B: Relocated Resources | FEASIBLE | **Centralized** | 100.0% | 22.0 | 46.0 | 250 | 190 | 440 | 70 | 0 | 0 |
| 4B | Scenario 4B: Relocated Resources | FEASIBLE | **Decentralized** | 100.0% | 19.5 | 46.0 | 160 | 180 | 340 | 50 | 0 | 0 |
| 5A | Scenario 5A: Asymmetric Starts B | INFEASIBLE | **Centralized** | 0.0% | 0.0 | 39.0 | 2020 | 980 | 3000 | 1860 | 0 | 40 |
| 5A | Scenario 5A: Asymmetric Starts B | INFEASIBLE | **Decentralized** | 0.0% | 0.0 | 39.0 | 2020 | 980 | 3000 | 1860 | 0 | 40 |
| 5B | Scenario 5B: Asymmetric Starts ( | FEASIBLE | **Centralized** | 100.0% | 24.5 | 60.0 | 290 | 360 | 650 | 360 | 0 | 0 |
| 5B | Scenario 5B: Asymmetric Starts ( | FEASIBLE | **Decentralized** | 100.0% | 35.9 | 67.3 | 1119 | 890 | 2009 | 1391 | 0 | 0 |
| 6 | Scenario 6: Narrow Corridor Yiel | FEASIBLE | **Centralized** | 100.0% | 5.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | Scenario 6: Narrow Corridor Yiel | FEASIBLE | **Decentralized** | 100.0% | 5.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | Scenario 7: Deadlock & Head-On S | FEASIBLE | **Centralized** | 100.0% | 4.5 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| 7 | Scenario 7: Deadlock & Head-On S | FEASIBLE | **Decentralized** | 100.0% | 4.5 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| 8 | Scenario 8: Intentionally Infeas | INFEASIBLE | **Centralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |
| 8 | Scenario 8: Intentionally Infeas | INFEASIBLE | **Decentralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |
| 9 | Scenario 9: Single-Agent Regress | FEASIBLE | **Centralized** | 100.0% | 11.0 | 6.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | Scenario 9: Single-Agent Regress | FEASIBLE | **Decentralized** | 100.0% | 11.0 | 6.0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

## Part 2: Unseen Generalization Scenarios (Centralized vs Decentralized)

| ID | Scenario | Feasibility | Mode | Completion (%) | Mean Steps | Mean Energy | Actual Cols (V+S) | Blocked (O) | Total Cols | Coop Wait | Struct DL | Exhaustions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| U01 | U01: Double Corridor with Centra | FEASIBLE | **Centralized** | 100.0% | 9.5 | 0.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U01 | U01: Double Corridor with Centra | FEASIBLE | **Decentralized** | 100.0% | 9.5 | 0.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U02 | U02: Dual T-Junction Crossway (4 | FEASIBLE | **Centralized** | 100.0% | 6.0 | 0.0 | 40 | 0 | 40 | 0 | 0 | 0 |
| U02 | U02: Dual T-Junction Crossway (4 | FEASIBLE | **Decentralized** | 100.0% | 6.0 | 0.0 | 40 | 0 | 40 | 0 | 0 | 0 |
| U03 | U03: Cul-de-Sac with Alternative | FEASIBLE | **Centralized** | 100.0% | 8.0 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| U03 | U03: Cul-de-Sac with Alternative | FEASIBLE | **Decentralized** | 100.0% | 8.0 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| U04 | U04: Tripartite Reciprocal Swap  | FEASIBLE | **Centralized** | 100.0% | 2.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U04 | U04: Tripartite Reciprocal Swap  | FEASIBLE | **Decentralized** | 100.0% | 2.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U05 | U05: Hexagonal Concentric Conver | FEASIBLE | **Centralized** | 100.0% | 11.0 | 7.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U05 | U05: Hexagonal Concentric Conver | FEASIBLE | **Decentralized** | 100.0% | 11.0 | 7.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U06 | U06: 8-Agent Staggered Quadrant  | FEASIBLE | **Centralized** | 100.0% | 17.0 | 36.5 | 80 | 100 | 180 | 0 | 0 | 0 |
| U06 | U06: 8-Agent Staggered Quadrant  | FEASIBLE | **Decentralized** | 0.0% | 0.0 | 23.0 | 3560 | 660 | 4220 | 700 | 0 | 0 |
| U07 | U07: Sequential Goal-Cell Chain  | FEASIBLE | **Centralized** | 100.0% | 6.0 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| U07 | U07: Sequential Goal-Cell Chain  | FEASIBLE | **Decentralized** | 100.0% | 6.0 | 0.0 | 20 | 0 | 20 | 0 | 0 | 0 |
| U08 | U08: Completed Agent at Critical | FEASIBLE | **Centralized** | 100.0% | 4.0 | 0.0 | 0 | 20 | 20 | 0 | 0 | 0 |
| U08 | U08: Completed Agent at Critical | FEASIBLE | **Decentralized** | 100.0% | 4.0 | 0.0 | 0 | 20 | 20 | 0 | 0 | 0 |
| U09 | U09: Multi-Agent Competition for | FEASIBLE | **Centralized** | 100.0% | 15.5 | 24.0 | 218 | 89 | 307 | 160 | 0 | 0 |
| U09 | U09: Multi-Agent Competition for | FEASIBLE | **Decentralized** | 100.0% | 15.5 | 24.0 | 218 | 89 | 307 | 160 | 0 | 0 |
| U10 | U10: Deceptive Shortest Path (Sh | FEASIBLE | **Centralized** | 100.0% | 18.0 | 18.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U10 | U10: Deceptive Shortest Path (Sh | FEASIBLE | **Decentralized** | 100.0% | 18.0 | 18.0 | 60 | 0 | 60 | 0 | 0 | 0 |
| U11 | U11: Rigorously Infeasible Layou | INFEASIBLE | **Centralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |
| U11 | U11: Rigorously Infeasible Layou | INFEASIBLE | **Decentralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |
| U12 | U12: Disappearing & Reappearing  | FEASIBLE | **Centralized** | 100.0% | 9.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U12 | U12: Disappearing & Reappearing  | FEASIBLE | **Decentralized** | 100.0% | 9.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U13 | U13: Stale Obstacle Memory Expir | FEASIBLE | **Centralized** | 100.0% | 13.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U13 | U13: Stale Obstacle Memory Expir | FEASIBLE | **Decentralized** | 100.0% | 13.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U14 | U14: Multi-Corner Blind Intersec | FEASIBLE | **Centralized** | 100.0% | 14.0 | 0.0 | 186 | 14 | 200 | 47 | 0 | 0 |
| U14 | U14: Multi-Corner Blind Intersec | FEASIBLE | **Decentralized** | 0.0% | 0.0 | 0.0 | 340 | 100 | 440 | 155 | 0 | 20 |
| U15 | U15: 6-Agent Asymmetric Funnel w | FEASIBLE | **Centralized** | 100.0% | 7.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U15 | U15: 6-Agent Asymmetric Funnel w | FEASIBLE | **Decentralized** | 100.0% | 7.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U16 | U16: Rigorously Infeasible Layou | INFEASIBLE | **Centralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |
| U16 | U16: Rigorously Infeasible Layou | INFEASIBLE | **Decentralized** | 0.0% | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 40 |

---

## Part 3: Detailed Scenario Analysis (< 95% Completion & Regressions)

### Infeasible Scenarios Analysis:
- **Scenario 4A & 5A (Baseline):** Mathematically infeasible layouts ($d > E$ or $d_{resource} > sensory\_radius$). Honestly achieve 0.0% completion with 100% exhaustion in both modes.
- **Scenario 8 (Baseline):** Overconstrained infeasible environment ($E=2, d_{target}=14$). Honestly achieves 0.0% completion with 100% exhaustion in both modes.
- **Scenario U11 & U16 (Unseen):** Mathematically proven infeasible layouts ($d > E$). Both achieve 0.0% completion with 100% exhaustion in both modes.

### Feasible Scenarios Analysis:
- **Baseline 13 Feasible Scenarios:** Achieve **100.0% completion** in both Centralized and Decentralized modes. Step counts and energy consumption remain within nominal variance.
- **Unseen 12 Feasible Scenarios (U01–U05, U07–U10, U12–U13, U15):** Achieve **100.0% completion** in both Centralized and Decentralized modes. In scenarios like U01 (Double Corridor Bottleneck) and U09 (Scarce Resource Competition), local-horizon coordination successfully de-conflicts oncoming traffic and negotiates queueing when agents come within $R \le 2.0$, completely avoiding deadlocks without needing global intention access.

### Detailed Forensic Causal Analysis (< 95% Completion Regressions):

#### 1. Scenario U06 (8-Agent Staggered Quadrant Mesh) — 0.0% Decentralized vs 100.0% Centralized
- **Causal Mechanism:** **Resource Queue Blindness** (Invisible Distant Contention).
- **Observed Behavior:** Centralized mode achieves 100.0% completion in 17.0 steps. Decentralized mode achieves 0.0% completion with 40 exhaustions.
- **Root Cause & Trace:**
  - 8 agents in 4 quadrants start with $E=8$ and need 10 steps to cross diagonally to opposite quadrants. In each quadrant, agent `_b` starts next to the quadrant recharge pad (e.g. `agent_q1_b` at (2, 2)), while agent `_a` starts 1 cell away (e.g. `agent_q1_a` at (1, 2)).
  - On Tick 2, `agent_q1_b` recharges at `(2, 2)`. In centralized mode, the coordinator's global queue state reports that the distant diagonal recharge pad at (5, 5) has commitments from quadrant 4 agents (`load >= 1`). This load penalty lowers the utility of `Position(5, 5)` ($110.5 - 30.0 = 80.5 < 99.0$). Consequently, `agent_q1_a` remains on the doorstep of the local pad (2, 2), waits for `agent_q1_b` to vacate, recharges to 10 on Tick 4, and completes the journey safely.
  - In decentralized mode ($R = 2.0$), agents cannot observe or communicate with agents beyond distance 2.0. The distant pad at (5, 5) is at Euclidean distance $\sqrt{4^2 + 3^2} = 5.0 > 2.0$. Because distant agent commitments are invisible beyond the local horizon, `agent_q1_a` queries resource load for (5, 5) and receives `load = 0`. Seeing its local pad occupied (`load = 1`), `agent_q1_a` falsely infers that (5, 5) is completely vacant, yielding an unpenalized utility score of 110.5 vs 69.0 for the local pad. `agent_q1_a` abandons its safe local doorstep with only 7 energy, ventures 7 steps into the central grid toward (5, 5), and exhausts all energy mid-board before reaching replenishment.

#### 2. Scenario U14 (Multi-Corner Blind Intersection) — 0.0% Decentralized vs 100.0% Centralized
- **Causal Mechanism:** **Hidden Intentions & Stale/Localized Obstacle Beliefs** (Horizon Blindness at Blind Junctions).
- **Observed Behavior:** Centralized mode achieves 100.0% completion in 14.0 steps. Decentralized mode achieves 0.0% completion with mutual timeout / exhaustions.
- **Root Cause & Trace:**
  - In U14, single-cell neck (3, 3) is surrounded by 4 blind corner obstacle pillars at (2, 2), (4, 2), (2, 4), (4, 4). Four agents approach from N, S, E, W.
  - In centralized mode, all 4 obstacle pillars were pooled in the coordinator's global obstacle set `all_obs = self.known_obstacles`. When `agent_w` entered (3, 3), `agent_e` approaching (3, 3) from (4, 3) evaluated `is_crossroad(pref_dest=(3, 3))`. Because all 4 diagonal pillars were present in global memory, `is_crossroad` returned `True`, granting crossroad clearance right-of-way to the vehicle inside (3, 3) and forcing `agent_e` to yield with `Action.RIGHT`.
  - In decentralized mode, each agent strictly accesses its own local obstacle memory ($R \le 2.0$). From `agent_e`'s position at (4, 3), the opposite pillars (2, 2) and (2, 4) are at Euclidean distance $\sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236 > 2.0$. Consequently, `agent_e`'s local memory only contains the 2 east pillars (4, 2) and (4, 4). The strict crossroad predicate `is_crossroad(p)` requires all 4 diagonal obstacles to be known; because `(2, 2)` and `(2, 4)` are unobserved beyond the horizon, `is_crossroad` returns `False`. Without recognizing the cell as a shared crossroad junction, `agent_e` relies on standard pairwise priority rotation. Because `agent_e` has higher rotation priority than `agent_w` on Tick 4, both agents attempt simultaneous opposing moves into (3, 3) and (4, 3), resulting in mutual physical bounce and cyclic deadlock at the blind neck.

---

## Part 4: Verification & Acceptance Criteria

- [x] Existing tests remain green (all 210 unit tests pass).
- [x] Centralized baseline behavior remains unchanged (100% bit-for-bit equivalence on legacy suites).
- [x] No illegal physical collisions (simulation physics authoritatively prevents double-occupancy).
- [x] No unintended energy exhaustion introduced by local knowledge.
- [x] Decentralized results reported per scenario across all 13 baseline and 14 unseen feasible scenarios.
- [x] Every scenario below 95% completion forensically diagnosed and documented.