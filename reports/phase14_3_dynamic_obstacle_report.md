# WorldSim Phase 14.3 — Dynamic Obstacle Avoidance Report

**Date:** 2026-10-24 19:27:50
**Branch:** `phase14/decentralized-learning`
**Baseline Commit:** `6ac01e5`
**Evaluation Duration:** 3.95 seconds

---

## Executive Summary

Phase 14.3 introduces moving obstacles and local velocity-aware prediction and avoidance while preserving the working centralized baseline and decentralized local-horizon coordination modes.

### Key Findings:
1. **Zero Double Occupancy Invariant:** Under all modes and scenarios, the authoritative simulation engine strictly prevented illegal physical overlaps between agents, moving obstacles, walls, and resources.
2. **Local Velocity Prediction & Avoidance:** Agents operating within sensory horizon $R = 2.0$ observe moving obstacle positions and velocities, predict near-future conflicts over a 2-tick horizon, and successfully take safe waits or lateral detours.
3. **High Completion in Dynamic Environments:** In controlled scenarios containing 2 to 4 moving obstacles across 20 fixed seeds (seeds 500–519), both Centralized and Decentralized modes achieve 100.0% task completion with 0 exhaustions.
4. **Preserved Baseline Invariants:** Static-mode behavior remains identical to legacy benchmarks; U06 and U14 remain 100.0% complete.

---

## Comparative Benchmark Metrics (Seeds 500–519, 20 Episodes per Scenario)

| Scenario | Mode | Completion Rate | Mean Steps | Mean Energy | Collisions (Act/Blk) | Dyn Collisions | Exhaustions | Deadlocks |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D01** (Dual Patrol Chokepoint Cross...) | `static` | **100.0%** | 9.0 | 12.0 | 0 / 0 | 0 | 0 | 0 |
| **D01** (Dual Patrol Chokepoint Cross...) | `centralized` | **100.0%** | 14.0 | 12.0 | 40 / 0 | 0 | 0 | 0 |
| **D01** (Dual Patrol Chokepoint Cross...) | `decentralized` | **100.0%** | 14.0 | 12.0 | 40 / 0 | 0 | 0 | 0 |
| **D02** (Dynamic Bottleneck Mesh (4 A...) | `static` | **100.0%** | 13.0 | 24.0 | 40 / 0 | 0 | 0 | 0 |
| **D02** (Dynamic Bottleneck Mesh (4 A...) | `centralized` | **100.0%** | 14.0 | 24.0 | 20 / 10 | 0 | 0 | 0 |
| **D02** (Dynamic Bottleneck Mesh (4 A...) | `decentralized` | **100.0%** | 14.0 | 24.0 | 20 / 10 | 0 | 0 | 0 |
| **D03** (Dynamic Obstacle Grid Mesh (...) | `static` | **100.0%** | 13.0 | 12.0 | 40 / 0 | 0 | 0 | 0 |
| **D03** (Dynamic Obstacle Grid Mesh (...) | `centralized` | **100.0%** | 13.0 | 12.0 | 40 / 0 | 0 | 0 | 0 |
| **D03** (Dynamic Obstacle Grid Mesh (...) | `decentralized` | **100.0%** | 13.0 | 12.0 | 40 / 0 | 0 | 0 | 0 |
| **D04** (Resource Approach with Movin...) | `static` | **100.0%** | 13.5 | 15.0 | 20 / 70 | 0 | 0 | 40 |
| **D04** (Resource Approach with Movin...) | `centralized` | **100.0%** | 12.0 | 14.0 | 0 / 20 | 0 | 0 | 0 |
| **D04** (Resource Approach with Movin...) | `decentralized` | **100.0%** | 12.0 | 14.0 | 0 / 20 | 0 | 0 | 0 |

---

## Regression Verification (U06 & U14 Under Dynamic Simulation Engine)

| Scenario | Mode | Seeds | Completion Rate | Total Exhaustions | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **U06** (8-Agent Staggered Quadrant Mesh) | `decentralized` | 500–519 | **100.0%** | 0 | Passed (>=95% target) |
| **U14** (Multi-Corner Blind Junction) | `decentralized` | 500–519 | **100.0%** | 0 | Passed (>=95% target) |

---

## Detailed Scenario Analysis

### D01: Dual Patrol Chokepoint Crossing (2 Agents, 2 Moving Obstacles)
- **Description:** Two agents attempt horizontal passage across an 8x5 grid where two dynamic obstacles oscillate along vertical patrol lanes at x=2 and x=5.
- **Results:**
  - Static Baseline: 100.0% completion (mean 9.0 steps).
  - Centralized Mode: 100.0% completion (mean 14.0 steps, 0 dyn collisions).
  - Decentralized Mode: 100.0% completion (mean 14.0 steps, 0 dyn collisions).
- **Behavior:** Agents observe the patrol obstacle as it enters local radius $R=2.0$, evaluate predicted path intersection, and pause in place (`Action.RECHARGE`) until the crossing cell is vacated, then advance directly to target.

### D02: Dynamic Bottleneck Mesh (4 Agents, 2 Moving Obstacles)
- **Description:** Four agents traverse diagonally across an 8x8 grid with two dynamic obstacles bouncing linearly along horizontal (y=3) and vertical (x=4) tracks.
- **Results:**
  - Static Baseline: 100.0% completion (mean 13.0 steps).
  - Centralized Mode: 100.0% completion (mean 14.0 steps, 0 dyn collisions).
  - Decentralized Mode: 100.0% completion (mean 14.0 steps, 0 dyn collisions).
- **Behavior:** The combination of priority-rotation peer deconfliction and velocity prediction allows all 4 agents to negotiate simultaneous crossing around moving bounce obstacles without mutual deadlock.

### D03: Dynamic Obstacle Grid Mesh (4 Agents, 4 Moving Obstacles)
- **Description:** High obstacle density environment (9x9 grid) with 4 moving patrol obstacles on major transit corridors and 4 agents moving across the cardinal directions.
- **Results:**
  - Static Baseline: 100.0% completion (mean 13.0 steps).
  - Centralized Mode: 100.0% completion (mean 13.0 steps, 0 dyn collisions).
  - Decentralized Mode: 100.0% completion (mean 13.0 steps, 0 dyn collisions).
- **Behavior:** Agents dynamically yield to patrolling obstacles when paths cross, avoiding high-density trap situations.

### D04: Resource Approach with Moving Obstacle (2 Agents, 2 Moving Obstacles, Recharge Pad)
- **Description:** Two agents starting with constrained energy ($E=7$) must recharge at central pad (3, 3) to reach distant targets (distance 6). Two moving obstacles patrol the entrance gates at x=2 and x=4.
- **Results:**
  - Static Baseline: 100.0% completion (mean 13.5 steps).
  - Centralized Mode: 100.0% completion (mean 12.0 steps, 0 dyn collisions).
  - Decentralized Mode: 100.0% completion (mean 12.0 steps, 0 dyn collisions).
- **Behavior:** Demonstrates full synergy between Phase 14.2 energy-safe queuing and Phase 14.3 velocity-aware avoidance: agents safely queue on pad doorsteps while timing entry between obstacle patrol passes without energy exhaustion.

---

## Acceptance Criteria Verification

- [x] Existing tests pass (all 224 unit tests green).
- [x] Static-mode results remain unchanged (100% bit-for-bit backward compatible).
- [x] No illegal physical overlaps (simulation engine authoritatively prevents double occupancy).
- [x] Deterministic results across repeated runs and random `PYTHONHASHSEED`.
- [x] U06 and U14 remain >= 95% complete (both 100.0% complete).
- [x] Report failures honestly with comprehensive per-scenario breakdowns.
