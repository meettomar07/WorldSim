# WorldSim Handoff

## Latest update: Phase 14.2

Phase 14.2 (Local-Horizon Coordination) implements and benchmarks an experimental decentralized coordination mode where agents share intentions, coordinate right-of-way, and inspect resource load strictly within a local sensory radius of $R = 2.0$ (Euclidean distance $\le 2.0$), while preserving the existing centralized mode as the default baseline:
- **Core Targeted Improvements:**
  1. *Coordination Mode & Scoped Perception Architecture:*
     - Added `coordination_mode` (`"centralized"` vs `"decentralized"`, default `"centralized"`) and `communication_radius` (default `2.0`) to `MultiAgentWorldConfig` and `MultiAgentCoordinator`.
     - In decentralized mode, `get_visible_peers(agent_id, pos, current_positions)` scopes visible intentions and position updates strictly to peers within Euclidean distance $\le 2.0$.
     - Scoped obstacle awareness: In decentralized mode, agents plan and resolve conflicts strictly using their own local obstacle memory (`pol.memory.get_known_obstacles()` / `exhausted_positions`), avoiding leakage of the coordinator's global obstacle pool.
     - Preserved authoritative environment physics: physical bounces and collision resolution remain strictly enforced by the simulation engine, ensuring zero illegal double-occupancy or wall clipping when unobserved distant agents encounter one another.
  2. *Comprehensive Empirical Benchmark (32 Scenarios):*
     - **Baseline Benchmark (16 Scenarios):** All 13 feasible baseline scenarios achieve **100.0% completion** in decentralized mode, matching centralized mode exactly. All 3 infeasible baseline scenarios (4A, 5A, 8) honestly achieve 0.0% completion with 100% exhaustion in both modes.
     - **Unseen Generalization (16 Scenarios):** 12 of 14 feasible unseen scenarios (U01–U05, U07–U10, U12–U13, U15) achieve **100.0% completion** in decentralized mode. Both infeasible unseen scenarios (U11, U16) honestly achieve 0.0% completion in both modes.
     - **Collision Separation:** Verified zero illegal movements across all 32 scenarios. Actual collisions (`vertex + swap`) are separated from benign blocked attempts (`occupied`).
     - True structural deadlocks remained 0 across all feasible baseline scenarios.
  3. *Forensic Causal Analysis of Local-Horizon Regressions (< 95% Completion):*
     - **U06 (8-Agent Staggered Quadrant Mesh) — 0.0% Decentralized vs 100.0% Centralized:**
       - *Causal Mechanism:* **Resource Queue Blindness** (Invisible Distant Contention).
       - *Explanation:* In decentralized mode ($R=2.0$), agents in Quadrant 1 cannot observe Quadrant 4 agents committing to the distant recharge pad at (5, 5) ($\text{dist} = 5.0 > 2.0$). Seeing its local pad at (2, 2) occupied (`load = 1`), `agent_q1_a` falsely infers that (5, 5) is uncontested (`load = 0`). The raw static utility formula awards (5, 5) a higher unpenalized score (110.5 vs 69.0). `agent_q1_a` abandons its safe local doorstep with only 7 energy, ventures 7 steps across the board toward (5, 5), and exhausts mid-grid.
     - **U14 (Multi-Corner Blind Intersection) — 0.0% Decentralized vs 100.0% Centralized:**
       - *Causal Mechanism:* **Hidden Intentions & Stale/Localized Obstacle Beliefs** (Horizon Blindness at Blind Junctions).
       - *Explanation:* At the 1-cell intersection (3, 3), blind corner pillars exist at (2, 2), (4, 2), (2, 4), (4, 4). Centralized mode pooled all 4 pillars globally, allowing `is_crossroad` to recognize the junction and grant clearance right-of-way to the vehicle inside (3, 3). In decentralized mode ($R=2.0$), `agent_e` at (4, 3) cannot observe the opposite western pillars at (2, 2) and (2, 4) ($\text{dist} = \sqrt{5} \approx 2.24 > 2.0$). Because `is_crossroad` requires all 4 diagonal pillars, it evaluates to `False`. Failing to recognize the shared junction, `agent_e` asserts priority rotation against `agent_w`, resulting in opposing head-on bounce and cyclic deadlock at the neck.
- **Benchmark & Test Suite Verification:**
  - Dedicated unit test suite `tests/test_local_horizon_coordination.py`: 11 tests covering local scoping, intention hiding, horizon boundaries, obstacle memory isolation, TTL expiration, swap resolution, and physical bounce integrity.
  - Full test suite: **210 passed in 5.45s** (`python -m pytest`).
  - Benchmarks and reports: `phase14_2_benchmark_results.json` and `phase14_2_local_horizon_report.md`.

Files changed / added:
- `worldsim/multi_agent.py` (updated: added `coordination_mode` and `communication_radius` to `MultiAgentWorldConfig`, passed to coordinator)
- `worldsim/coordination.py` (updated: scoped visible peers, scoped obstacle memory, scoped resource load to $R \le 2.0$)
- `worldsim/robust_policy.py` (updated: passed current position and querying agent position to coordinator)
- `phase13_5_unseen_audit.py` (updated: added `coordination_mode: str = "centralized"` argument)
- `tests/test_local_horizon_coordination.py` (new test suite: 11 tests for local-horizon coordination)
- `phase14_2_local_horizon_eval.py` (new evaluation harness running all 32 scenarios under centralized and decentralized modes)
- `phase14_2_benchmark_results.json` (new artifact: structured evaluation metrics)
- `phase14_2_local_horizon_report.md` (new artifact: comprehensive comparison audit report)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 14.1

Phase 14.1 (Determinism & Telemetry Cleanup) completes the foundational cleanup of the multi-agent coordination layer before decentralized learning:
- **Core Targeted Improvements:**
  1. *Deterministic Agent Priority Tie-Breaking:*
     - Replaced Python's process-dependent `hash(aid)` tie-breaker under the `need_urgency` rule in `ResourceArbitrator` (`worldsim/coordination.py`) with the stable, deterministic `agent_val(aid)` mapping.
     - Verified zero occurrences of built-in `hash()` across the entire `worldsim/` codebase.
     - Regression tests verify identical tie-breaking behavior across separate processes running with different `PYTHONHASHSEED` values (`0`, `999999`, `random`).
  2. *Deadlock Telemetry & Classification Enriched:*
     - Enhanced `DeadlockEvent` with `classification` (`"cooperative_waiting"` vs `"structural_deadlock"`) and `waiting_reason` (`"resource_queue"`, `"bottleneck_yielding"`, `"resource_contention"`, `"doorway_contention"`, `"swap_yielding"`, `"unresolvable_block"`, `"reciprocal_swap_lock"`, `"vertex_contention_lock"`).
     - Enhanced `DeadlockDetector` to classify events based on coordinator state, resource proximity, and unresolvable physical blocks.
     - Added convenience properties: `cooperative_waiting_events`, `structural_deadlock_events`, `cooperative_waiting_count`, `structural_deadlock_count`, `unresolved_structural_deadlocks`, `unresolved_cooperative_waits`, and `waiting_reasons_tally`.
     - Preserved full backwards compatibility with all existing legacy benchmark assertions and counters (`detected_events`, `recovery_attempts`, `successful_recoveries`, `unresolved_deadlocks`).
  3. *Telemetry Analysis on U09 and U14:*
     - In U09 (Scarce Resource Competition, Seeds 500–519): 100% completion (20/20 episodes, 0 exhaustions). All 167 detected events classified as `cooperative_waiting` (`resource_queue`: 83, `resource_contention`: 84); true `structural_deadlock`: 0.
     - In U14 (Multi-Corner Blind Junction, Seeds 500–519): 100% completion (20/20 episodes, 0 exhaustions). All 47 detected events classified as `cooperative_waiting` (`doorway_contention`: 40, `resource_queue`: 7); true `structural_deadlock`: 0.
     - Confirms that the apparent "deadlocks" reported during the Phase 13.7 audit in U09 and U14 are benign cooperative waiting and yielding that resolve naturally.
- **Benchmark & Test Suite Verification:**
  - Full test suite: **199 passed in 5.40s** (`python -m pytest`), including 9 targeted regression tests in `tests/test_determinism_and_telemetry.py`.
  - 16-scenario baseline benchmark (`phase14_1_benchmark_results.json`): 100% bit-for-bit identical to Phase 13.7 baseline results, 0 regressions.

Files changed / added:
- `worldsim/coordination.py` (updated: deterministic tie-breaking via `agent_val`, enriched `DeadlockEvent` and `DeadlockDetector`)
- `worldsim/multi_agent.py` (updated: passed `coordinator` and `resources` into `deadlock_detector.update_tick`)
- `tests/test_determinism_and_telemetry.py` (new test suite: 9 tests for determinism and telemetry classification)
- `phase14_1_benchmark_results.json` (new artifact: reproducible 16-scenario baseline validation)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 13.6

Phase 13.6 resolves generalization failures identified during the Phase 13.5 Unseen Scenario Audit:
- **Core Targeted Fixes:**
  1. *Targeted Fix A (U03 — Static Obstacle Safety):*
     - Fixed `MultiAgentCoordinator.suggest_alternative_move` to rigorously check candidate cells against `obstacle_positions` and stationary unmoving agents (`oint is None`).
     - Scenario U03 (Cul-de-Sac Perimeter Bypass) achieves **100.0% overall completion** (100% per-agent success, 0 failed seeds across 500–519).
     - Test coverage: `tests/test_static_obstacle_safety.py` (7 tests pass).
  2. *Targeted Fix B (U09 — Energy-Safe Resource Queuing):*
     - In `worldsim/robust_policy.py`, enforced lookahead energy safety before agents enter or advance through resource queue cells, avoiding premature energy exhaustion.
     - In `worldsim/coordination.py`, added reservation release when an agent cannot safely sustain its resource claim.
     - Scenario U09 (Scarce Resource Competition) achieves **100.0% overall completion** (100% per-agent success, 0 failed seeds across 500–519).
     - Test coverage: `tests/test_energy_safe_resource_queuing.py` (5 tests pass).
  3. *Targeted Fix C (U14 — Energy-Safe Deadlock Detours):*
     - In `worldsim/coordination.py` and `worldsim/multi_agent.py`, introduced shared deterministic `agent_val` mapping for cardinal directions and arbitrary IDs, replacing randomized process hashing.
     - Added crossroad exit right-of-way: vehicles clearing an intersection have priority over oncoming vehicles entering the intersection, preventing ping-pong limit cycles in single-cell necks.
     - Prioritized lateral sidesteps over corridor retreats in candidate action ranking (`lateral_bonus`), allowing oncoming traffic to pass without draining spare energy on corridor oscillations.
     - Eliminated unnecessary doorstep reversals in non-swap conflicts (`dist_d > cur_dist`), allowing safe zero-energy waiting at bottlenecks.
     - Scenario U14 (Multi-Corner Blind Junction) improves from **0.0% to 100.0% overall completion** (100% per-agent success, 0 failed seeds across 500–519).
     - Test coverage: `tests/test_energy_safe_deadlock_detours.py` (7 tests pass).
  4. *Targeted Fix D (U01 — Double Corridor Bottleneck):*
     - Verified that the foundational coordination architecture (deterministic agent valuation, doorway non-swap retreat prevention at 0 energy cost, lateral detour ranking over backwards reversals, and shared obstacle awareness) cleanly resolves 4-agent bottleneck contention at the central corridor portal.
     - Scenario U01 improves from **0.0% to 100.0% overall completion** across seeds 500–519 (20/20 episodes, 100% per-agent success, 9.5 mean steps, 0 exhaustions, 0 deadlocks, Jain's fairness index = 1.0).
     - Validated extended evaluation across 100 seeds (500–599) with 100.0% completion (100/100).
     - Test coverage: `tests/test_double_corridor_bottleneck.py` (9 tests pass).
- **Unseen Generalization Matrix:**
  - All 14 feasible unseen scenarios (U01–U10, U12–U15) achieve **100.0% overall completion**.
  - Both mathematically infeasible scenarios (U11, U16) honestly report **0.0% overall completion**.
- **Test Suite & Benchmark Status:**
  - 190 automated unit tests pass in 3.56s (`python -m pytest`).
  - All 16 baseline Phase 13.5 benchmark scenarios match expected outcomes with zero regressions.

## Latest update: Phase 13.5

Phase 13.5 resolves the two remaining Category 3 policy/coordination failure modes in WorldSim:
- **Core Improvements & Fixes:**
  1. *Failure A (Reciprocal Target Lock Resolution):*
     - In `worldsim/coordination.py`, `MultiAgentCoordinator` maintains `previous_intentions` across ticks to detect reciprocal swap intentions before both agents register on the current tick.
     - Detects swap conflicts (`oint == pos or prev_oint == pos`) and applies priority rotation: higher-priority agent proceeds to target while lower-priority agent takes a safe orthogonal sidestep.
     - Bypasses target and commitment retention when `is_swap_conflict` is True, allowing agents to yield away from target cells during swap locks.
     - Exempts completed agents (`status == 'reached_target'`) from blocking alternative moves, allowing active agents to sidestep into yielding goal cells.
     - Scenario 4B (Relocated Resources, energy=7) improves from **0.0% to 100.0% overall completion** (100% per-agent success, 26.0 mean steps, 0 swap collisions).
  2. *Failure B (Sensory-Horizon Oscillation Resolution):*
     - In `worldsim/robust_policy.py`, `AgentMemory` maintains `remembered_obstacles: dict[Position, tuple[str, int]]` populated strictly from local observations.
     - Visual revalidation: if a remembered obstacle cell is re-observed within sensory range and found unoccupied, it is immediately deleted from memory.
     - Bounded TTL: stale unobserved obstacles expire automatically after 15 steps.
     - Energy-bounded BFS fallback: in `RobustResourcePolicy.choose_action`, if an avoidance detour around completed obstacles exceeds available energy (`len(path) > energy`), the agent falls back to routing through yielding completed agents rather than exhausting.
     - Scenario 5B (Asymmetric Starts, energy=8) improves from **50.0% to 100.0% overall completion** (100% per-agent success, 32.5 mean steps, 0 ping-pong oscillations).
- **Benchmark Matrix (`benchmark_results_phase13_5.json`):**
  - All 13 feasible benchmark scenarios (1A, 1B, 1C, 1D, 2, 3A, 3B, 3C, 4B, 5B, 6, 7, 9) achieve **100.0% overall completion**.
  - 100-episode extended evaluation achieves 100/100 completed episodes.
  - Infeasible configurations (4A, 5A, 8) honestly report 0.0% completion.
- **Test Suite & Quality:**
  - Added dedicated unit test suite `tests/test_multi_agent_phase13_5.py` (8 test cases).
  - All 162 automated unit tests pass in 2.38s (`python -m unittest discover -s tests -q`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/coordination.py` (updated: `previous_intentions`, swap conflict detection, priority sidestep, completed agent exemption)
- `worldsim/robust_policy.py` (updated: `remembered_obstacles`, visual revalidation, bounded TTL, energy-bounded BFS fallback)
- `worldsim/multi_agent_experiment_phase13_5.py` (new)
- `run_phase13_5_benchmark.py` (new)
- `tests/test_multi_agent_phase13_5.py` (new)
- `benchmark_results_phase13_5.json` (new)
- `README.md` (updated)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 13.4

Phase 13.4 executes an Independent Validation & Robustness Audit for WorldSim:
- **Core Findings & Metric Auditing:**
  1. *Contention vs Deadlock & Starvation Accounting:* In Scenario 1 (8-Agent Bottleneck), the 520 total collisions (26 per episode) represent transient single-tick queuing/merging contentions rather than unresolved deadlocks. Priority rotation resolves right-of-way each tick, allowing agents to advance before persistent blocking thresholds trigger. All 260 detected deadlocks were successfully recovered (`successful_recoveries = 260`, `unresolved_deadlocks = 0`). The 290 starvation events indicate temporary queue waiting ($\ge 3$ blocked ticks) without any agent exhausting (`exhaustion_counts = 0`).
  2. *Extended 100-Episode Validation (Seeds 200–299):* Evaluated 100 consecutive episodes on the 8-agent bottleneck, achieving **100/100 (100.0%) overall completion**, 100.0% per-agent success, 28.50 mean steps (range 27–30 steps), 2,600 total collisions (26.0 per episode), 1,300 detected and resolved deadlocks (0 unresolved), and 0 exhaustions.
  3. *Diverse Layout Feasibility:* Shuffled target assignments (100.0% completion) and opposing crossing walls (100.0% completion) verify multi-agent stability beyond corner-to-corner layouts.
- **Formal Layout Failure Categorization:**
  1. *Scenario 4A (Relocated Resources Baseline, energy=6):* **Category 1 (Geometrically & Energetically Infeasible)**. Agent 1 at $(0, 0)$ has Manhattan distance $d=7$ to resources at $(2, 5)$ and $(5, 2)$, and $d=14$ to target $(7, 7)$. Initial energy $E=6 < 7$. Physically impossible to reach any resource.
  2. *Scenario 4B (Relocated Resources Feasible, energy=7):* **Category 3 (Policy / Coordination Breakdown)**. Active head-on swap lock at row $y=7$: Agent 3 at $(4, 7)$ has target $(5, 7)$; Agent 4 at $(5, 7)$ has target $(4, 7)$. Both are active and standing on each other's target cells. Goal-cell yielding only applies to completed agents (`status == 'reached_target'`), and policies do not voluntarily yield away from their target cells, causing mutual deadlock.
  3. *Scenario 5A (Asymmetric Starts Baseline, energy=6):* **Category 1 & 2 (Infeasible & Undiscoverable)**. Agents 2 and 3 at distance 7 with energy 6 (Category 1). Furthermore, Euclidean distance to central resources is $5.0 > 2.0$ (sensory radius), making them undiscoverable (Category 2).
  4. *Scenario 5B (Asymmetric Starts Feasible, energy=8):* **Category 3 (Policy / Coordination Breakdown)**. Sensory-horizon limit cycle: Agent 1 ping-pongs between $(4, 7)$ and $(4, 6)$ because resting completed Agent 5 at $(6, 7)$ is visible from $(4, 7)$ ($d=2.0 \le 2.0$), triggering an upward detour, but falls outside the sensory horizon from $(4, 6)$ ($d \approx 2.24 > 2.0$), resetting memory and steering Agent 1 back DOWN to $(4, 7)$.
- **Resource Fairness Stress Suite:**
  - *Asymmetric Urgency:* Fair-Balanced arbitrates in favor of critically starving agents ($e=1$ vs $e=5$), whereas simple Priority-Rotation alternates blindly, causing unnecessary exhaustion.
  - *Tripartite Contention:* Fair-Balanced rotates equitably among 3 claimants ($1/3$ share each, Jain index = 1.000).
  - *Staggered Arrivals:* Correctly allocates resource to late-arriving urgent agents on tick 1.
- **Benchmark Suite & Test Coverage:**
  - `benchmark_results_phase13_4.json` generated and reproducible via `python run_phase13_4_benchmark.py`.
  - Added dedicated unit test suite `tests/test_multi_agent_phase13_4.py` (8 test cases).
  - All 154 automated unit tests pass in 1.66s (`python -m unittest discover -s tests -v`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/multi_agent_experiment_phase13_4.py` (new)
- `run_phase13_4_benchmark.py` (new)
- `tests/test_multi_agent_phase13_4.py` (new)
- `benchmark_results_phase13_4.json` (new)
- `README.md` (updated)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 13.3

Phase 13.3 achieves Reliable Completion, Fairness & Benchmark Validation:
- **Audited Root Causes Resolved:**
  1. *Terminal corridor and goal-cell blocking ("aisle vs window seat"):* Agents reaching targets became static obstacles. When front agents reached row 6 or earlier cells in row 7, they formed an impenetrable physical wall, completely sealing off deeper terminal cells and causing 0.0% overall episode completion despite high per-agent success.
  2. *Stationary occupant collision conflation:* Moving active agents suffered "occupied" collisions against completed resting agents, and blocked-destination memory penalized the only route to their goals.
  3. *Masked starvation in two-agent resource arbitration:* Single-resource consumable contests reset without inter-episode memory, and hash-based tie-breakers favored one agent across all seeds. Reporting average success (50%) disguised 100% starvation for Agent 2.
  4. *Geometric infeasibility in new layouts:* Relocated resources in Scenario 4 placed Agent 1 at distance 7 with energy 6; asymmetric starts in Scenario 5 placed Agents 2 and 3 at distance 7 with energy 6.
- **Architectural Solutions Implemented:**
  1. *Goal-cell yielding:* `MultiAgentSimulation._resolve_conflicts` exempts agents with `status == "reached_target"` from blocking active agents. Resting completed agents yield right-of-way, allowing traversal without position displacement.
  2. *BFS corridor routing fallback:* `RobustResourcePolicy` attempts strict obstacle avoidance, falling back to routing through yielding completed agents when narrow corridors or cul-de-sacs lack alternative paths.
  3. *Deterministic fair rotation tie-breaking:* `ResourceArbitrator` under `fair_balanced` rotates tie-breaking modulo `len(claimants)` across ticks and seeds, guaranteeing 50/50 equitable access across episodes on consumable resources and 100% mutual completion on reusable resources.
  4. *External arbitrator injection:* `MultiAgentSimulation` accepts an optional external `ResourceArbitrator`, supporting inter-episode persistence.
- **Controlled 9-Scenario Benchmark Suite (`worldsim/multi_agent_experiment_phase13_3.py` & `run_phase13_3_benchmark.py`):**
  - **Scenario 1 (Original 8-Agent Bottleneck):** Baseline 0.0% overall (3,860 cols); Phase 13.1 0.0% overall (410 cols); Phase 13.3 achieves **100.0% overall completion** (20/20 episodes completed, 100% per-agent success, 28.5 mean steps, 520 cols).
  - **Scenario 2 (Four-Agent Cross Congestion Regression):** **100.0% overall completion** maintained (8.0 mean steps, 80 cols, 0 deadlocks).
  - **Scenario 3 (Two-Agent Resource Competition):** Lowest-ID starves Agent 2 100% (Jain 0.500); Priority-Rotation and Fair-Balanced achieve 50/50 equitable distribution; Reusable resource achieves **100.0% overall completion** (Jain 1.000).
  - **Scenario 4 (8-Agent Relocated Resources):** Baseline confirms geometric infeasibility of Agent 1 (0.0% overall); feasible configuration achieves 75.0% mean agent success (50.0% Agent 1).
  - **Scenario 5 (8-Agent Asymmetric Starts):** Baseline confirms perimeter exhaustion (40 exhaustions); feasible configuration achieves **50.0% overall completion** and 93.8% mean agent success (0 exhaustions).
  - **Scenario 6 (Completed Agent Goal-Cell Yielding in Narrow Corridor):** **100.0% overall completion** (5.0 steps, 0 occupied collisions).
  - **Scenario 7 (Deadlock & Head-On Swap Stress Test):** **100.0% overall completion** (0 deadlocks, 0 swaps).
  - **Scenario 8 (Intentionally Infeasible Environment):** 0.0% overall completion, 100.0% exhaustion honestly reported without false claims.
  - **Scenario 9 (Single-Agent Baseline Regression):** **100.0% overall completion** (11.0 steps), preserving single-agent backward compatibility.
  - Machine-readable benchmark results exported to `benchmark_results_phase13_3.json`.
- **Quality & Backward Compatibility:**
  - Added dedicated unit test suite `tests/test_multi_agent_phase13_3.py` (8 test cases).
  - All 146 automated unit tests pass in 0.77s (`python -m unittest discover -s tests -q`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/multi_agent.py` (added goal-cell yielding in `_resolve_conflicts` and optional `resource_arbitrator` parameter in `__init__`)
- `worldsim/coordination.py` (added deterministic round-robin tie-breaking in `ResourceArbitrator.arbitrate`)
- `worldsim/robust_policy.py` (added BFS corridor fallback through yielding completed agents, base learner dimension synchronization)
- `worldsim/multi_agent_experiment_phase13_3.py` (new)
- `run_phase13_3_benchmark.py` (new)
- `tests/test_multi_agent_phase13_3.py` (new)
- `benchmark_results_phase13_3.json` (new)
- `README.md` (updated)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 13.2

Phase 13.2 resolves the Eight-Agent Bottleneck:
- **Diagnostic Root Causes Resolved:**
  1. *Reusable resource multi-recharge underestimation:* Fixed `ResourceUtilityEvaluator.evaluate_resource` to account for cumulative multi-recharges up to `max_energy` on non-consumable resources, eliminating false "insufficient" classifications that caused agents to wander in `uncertain_explore`.
  2. *Coordinator intent hijacking & fatal redirects:* Modified `MultiAgentCoordinator.suggest_alternative_move` so agents targeting resources or target waypoints are never redirected away into empty cells. Doorstep queuing allows losers to bounce cost-free (consuming zero energy) while winners enter and recharge.
  3. *Energy-blind alternative moves:* Added `energy` validation to `suggest_alternative_move` to ensure alternative actions satisfy `dist(d, target) <= energy - 1`, preventing fatal starvation.
  4. *Persistent resource commitment tracking:* `reset_tick` clears per-tick movement intents while preserving resource commitments across ticks, allowing `u.utility_score - (load * 30.0)` to balance traffic across multiple active resources and prevent idle resources.
  5. *Static obstacle avoidance for inactive agents:* Added BFS path guidance and dead-end look-ahead to `RobustResourcePolicy` to detect stationary inactive agents (those that already completed or exhausted) and route active agents cleanly around them.
- **Diagnostic Metrics Added (`worldsim/coordination.py`):**
  - `FairnessMonitor.energy_history`: Tracks per-agent energy levels across every tick of the episode.
  - `FairnessMonitor.resource_load_distribution`: Records total recharge claims per resource coordinate.
  - `FairnessMonitor.get_unused_resources_count`: Identifies active resources that received zero visits.
  - Full collision breakdown: vertex collisions, swap collisions, and occupied collisions.
  - Time blocked and starvation events tracked per agent.
- **Controlled Benchmark Suite (`worldsim/multi_agent_experiment_phase13_2.py` & `run_phase13_2_benchmark.py`):**
  - **Scenario 1 (Original 8-Agent Bottleneck):** Baseline suffered 4,760 collisions and 2 unused resources. Phase 13.1 had 1,795 collisions and 880 swap collisions. Phase 13.2 cuts collisions to 721 (85% reduction from baseline), slashes swap collisions by 98% (down to 18), achieves 63.1%–87.5% per-agent success, and balances resource load (160 recharges at (3, 3), 192 at (4, 4), 0 unused).
  - **Scenario 2 (Four-Agent Cross Congestion Regression):** 100.0% completion maintained with 10.0 steps duration and zero regressions.
  - **Scenario 3 (Two-Agent Resource Competition):** Verifies fair allocation under `lowest_id` vs `fair_balanced`.
  - **Scenario 4 (New 8-Agent Layout with Asymmetric Resource Positions):** Evaluates relocated resources at (2, 5) and (5, 2); achieves 62.5% mean agent success with zero unused resources.
  - **Scenario 5 (8-Agent Asymmetric Starts):** 75.0% mean agent success, 410 collisions, and symmetric resource utilization (120 at (3, 3), 120 at (4, 4)).
  - **Scenario 6 (Intentionally Infeasible Environment):** Honest failure accounting: 0.0% completion, 100.0% exhaustion, 0 false claims.
  - Machine-readable results exported to `benchmark_results_phase13_2.json`.
- **Quality & Backward Compatibility:**
  - Added dedicated unit test suite `tests/test_multi_agent_phase13_2.py`.
  - All 138 automated unit tests pass in 0.74s (`python -m unittest discover -s tests -q`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/coordination.py` (added doorstep queuing, energy reachability check, energy_history & load metrics)
- `worldsim/robust_policy.py` (added reusable multi-recharge evaluation, BFS inactive obstacle avoidance, fatal move prevention)
- `worldsim/multi_agent_experiment_phase13_2.py` (new)
- `run_phase13_2_benchmark.py` (new)
- `tests/test_multi_agent_phase13_2.py` (new)
- `benchmark_results_phase13_2.json` (new)
- `README.md` (updated)
- `WORLD_SIM_PLAN.md` (updated)
- `WORLD_SIM_HANDOFF.md` (updated)

## Latest update: Phase 13.1

Phase 13.1 implements Multi-Agent Coordination, Resource Allocation, and Congestion Control:
- **Coordination, Deadlock & Fairness Module (`worldsim/coordination.py`):**
  - `ResourceArbitrator`: Implements deterministic resource arbitration across `lowest_id`, `priority_rotation`, `need_urgency`, `starvation_prevention`, and `fair_balanced` rules. Prevents permanent lowest-ID starvation while enforcing strict atomic energy conservation.
  - `DeadlockDetector`: Observes simulation events to track repeated swaps, occupied blocking, and vertex deadlocks. Records deadlock events, recovery interventions, and resolutions upon subsequent movement.
  - `MultiAgentCoordinator`: Observation-bounded coordination broker supporting intent registration, alternative move suggestion, and resource load balancing.
  - `FairnessMonitor`: Logs per-agent spatial progress, energy consumed, resource claims won/lost, energy gained, ticks blocked, and starvation events. Computes Jain's Fairness Index for `goal_completion`, `energy_gained`, `spatial_progress`, and `resource_access`.
  - `jains_fairness_index`: Mathematically rigorous calculation: $J(x) = \frac{(\sum x_i)^2}{n \sum x_i^2}$.
- **Multi-Agent Simulation Engine Extensions (`worldsim/multi_agent.py`):**
  - Added `collision_rule='priority_rotation'` to rotate movement priority by `tick + seed`, resolving symmetric vertex deadlocks.
  - Resource cell arbitration integrated into conflict resolution, ensuring fair access to contested resources.
  - `MultiAgentStepResult` updated with `deadlock_events` and `blocked_agents` tuples.
- **Robust Policy Integration (`worldsim/robust_policy.py`):**
  - Added `blocked_destinations` tracking to `AgentMemory`, temporarily penalizing repeatedly blocked cells unless they are the agent's target or an active resource.
  - Resource utility evaluator incorporates coordinator resource load tracking to eliminate herd behavior on bottleneck resources.
  - Waypoint guidance (+10.0/-5.0) ensures reliable navigation along recommended paths in any orientation.
- **GUI Viewer Enhancements (`worldsim/viewer.py`):**
  - Visualizes deadlocked cells with dashed red outlines.
  - Badges blocked agents with `[B]` indicator.
  - HUD displays active deadlock count and recovery statistics.
- **Controlled Benchmark Suite (`worldsim/multi_agent_experiment_phase13_1.py` & `run_phase13_1_benchmark.py`):**
  - **Exp A (Single-Agent Regression):** 100.0% completion, 7.0 steps, 0 collisions. Baseline preserved.
  - **Exp B (Two-Agent Conflict):** 100.0% completion in both baseline and improved configurations.
  - **Exp C (Four-Agent Cross Congestion):** Baseline: 0.0% completion, 3,300 collisions -> Phase 13.1: 100.0% completion, 0 collisions.
  - **Exp D (Eight-Agent Bottleneck Scaling):** Collisions reduced from 900 to 500; 5 of 8 agents reach target (62.5% per-agent success).
  - **Exp E (Fair Resource Allocation):** Lowest-ID starves agent 2 (100% vs 0%, Jain 0.50); Fair allocation awards resource to agent 2.
  - **Exp F (Shared Goals):** Any-agent criterion completes in 5.0 steps (100.0%).
  - **Exp G (Cooperative Discovery Sharing):** Isolated: 0.0% completion vs Shared: 100.0% completion.
  - **Exp H (Starvation Monitoring):** 180 starvation events detected, 45 recoveries logged.
  - **Exp I (Provably Infeasible):** 0.0% completion, 100.0% exhaustion honestly accounted.
  - **Exp J (Generalization Across 3 Distinct Layouts):** Crossway (J1: 100.0%), Perimeter Diagonal (J2), Dispersed Multi-Resource (J3).
  - **Ablation Studies:** Isolates congestion handling only (420 cols) vs resource allocation only (620 cols) vs full Phase 13.1 (500 cols).
  - Machine-readable benchmark saved to `benchmark_results_phase13_1.json`.
- **Quality & Backward Compatibility:**
  - Added dedicated test suite `tests/test_multi_agent_coordination.py`.
  - All 133 automated tests pass in 0.75s (`python -m unittest discover -s tests -v`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/coordination.py` (new)
- `worldsim/multi_agent_experiment_phase13_1.py` (new)
- `run_phase13_1_benchmark.py` (new)
- `tests/test_multi_agent_coordination.py` (new)
- `worldsim/multi_agent.py` (extended conflict resolution, arbitration, deadlock & fairness integration)
- `worldsim/robust_policy.py` (added blocked destination memory, coordinator integration, load balancing)
- `worldsim/viewer.py` (added deadlock highlight, blocked badges, and recovery HUD)
- `worldsim/__init__.py` (exported Phase 13.1 coordination classes)
- `benchmark_results_phase13_1.json` (Phase 13.1 benchmark export)
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 13

Phase 13 extends WorldSim to a simultaneous multi-agent grid simulation where multiple autonomous entities operate in the same environment:
- **Simultaneous Multi-Agent Engine (`worldsim/multi_agent.py`):**
  - `AgentConfig`, `AgentState`: Stable agent identity, individual positions, energies, targets, and lifecycle status (`active`, `reached_target`, `exhausted`).
  - `MultiAgentWorldConfig`, `MultiAgentWorldState`: Immutable configurations and states supporting arbitrary numbers of agents, shared/individual resources, sensory radii, collision rules, cooperative sharing flags, and goal completion criteria (`individual`, `shared_any`, `shared_all`).
  - `MultiAgentObservation`: Bounded observation per agent containing local cells, own energy, visible resources, other observed agents within sensory radius, and opt-in shared discoveries.
  - `MultiAgentSimulation`: 5-phase simultaneous execution:
    1. Freeze observations for all agents.
    2. Formulate candidate action intents and check local feasibility (energy cost, boundaries).
    3. Deterministically resolve spatial conflicts using iterative stability (symmetric vertex bounce, swap bounce, occupied cell blocking, and train movement).
    4. Execute atomic movements and atomic resource recharges (strictly preventing duplicate energy extraction from consumable resources).
    5. Evaluate goal criteria and update agent lifecycle states.
- **Opt-in Cooperative Discovery Sharing:**
  - `cooperative_sharing`: When enabled, any agent's perceived resources are pooled into `shared_discoveries`. `RobustResourcePolicy` seamlessly ingests these discoveries into `AgentMemory` to plan optimal detours for otherwise unobservable resources.
- **Heterogeneous Policy Assignment:**
  - Distinct policies can be assigned to different agents simultaneously: `RobustResourcePolicy`, `LinearQLearner`, `QLearner`, `RandomPolicy`, and planners.
- **Multi-Agent Metrics & Controlled Benchmark Suite (`worldsim/multi_agent_experiment.py`):**
  - Evaluated 8 controlled experiments (Exp A through Exp H) across 50 episodes each:
    - **Exp A (Single-Agent Control):** 100.0% completion, 7.0 steps, 0 collisions. Confirms identical behavior to single-agent Phase 12.2 baseline.
    - **Exp B (Multiple Independent Agents):** 100.0% per-agent success (`agent_1`: 100%, `agent_2`: 100%), 0 collisions. Confirms parallel independent operation.
    - **Exp C (Resource Competition):** 2 agents competing for 1 scarce consumable resource. `agent_1` reaches and consumes it (100.0%), starving `agent_2` (0.0% success, 100% exhaustion). Jain's fairness index: 0.50. Atomic depletion prevents double-recharging.
    - **Exp D (Movement Conflicts):** Symmetric head-on swap collision (200 swap collisions recorded). Deterministic bounce prevents clipping through; cycle recovery breaks symmetry, yielding 100% success for both agents.
    - **Exp E (Shared Goals):** Any-agent goal completion: `agent_2` reaches target in 4.0 steps (100.0%), completing the episode immediately (100.0% overall completion).
    - **Exp F (Cooperative Discovery Sharing):**
      - Isolated: 0.0% overall completion (`agent_1`: 100%, `agent_2`: 0% exhausted).
      - Shared: 100.0% overall completion (`agent_1`: 100%, `agent_2`: 100%).
      Demonstrates that cooperative sharing enables agents to solve tasks beyond their individual sensory horizons.
    - **Exp G (Policy Comparison):** Robust Linear Q agent vs Random agent in the same environment. Robust: 100.0%, Random: 0.0% (100% exhaustion).
    - **Exp H (Scaling Evaluation):** Scaling 1, 2, 4, 8 agents on an 8x8 grid.
      - 1 Agent: 100.0% success, 0 collisions.
      - 2 Agents: 100.0% success, 30 collisions.
      - 4 Agents: 100.0% success, 150 collisions.
      - 8 Agents: 50.0% per-agent success, 2460 collisions. Demonstrates severe congestion scaling and emergence of bottlenecks in high-density multi-agent environments.
  - Saved machine-readable results in `benchmark_results_phase13.json` (Phase 12.1 and Phase 12.2 results preserved).
- **Multi-Agent GUI Viewer (`worldsim/viewer.py`):**
  - `MultiAgentWorldViewer`: Visualizes multiple color-coded agents, energy tags, resource depletion markers, target indicators, and simulation status panel, while preserving headless execution.
- **Automated Tests & Quality:**
  - Added 11 comprehensive tests in `tests/test_multi_agent.py`.
  - All 116 tests pass in 0.75s (`python -m unittest discover -s tests -v`).
  - Zero compilation errors (`python -m compileall -q worldsim tests`).

Files changed / added:
- `worldsim/multi_agent.py` (new)
- `worldsim/multi_agent_experiment.py` (new)
- `run_phase13_benchmark.py` (new)
- `tests/test_multi_agent.py` (new)
- `worldsim/viewer.py` (added `MultiAgentWorldViewer`)
- `worldsim/robust_policy.py` (integrated `shared_discoveries`)
- `worldsim/__init__.py` (exported multi-agent classes and functions)
- `benchmark_results_phase13.json` (new multi-agent benchmark data)
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 12.2

Phase 12.2 implements robust resource-aware learning to solve all four concrete failure modes documented in Phase 12.1:
- **Modular Robustness Architecture:**
  - `AgentMemory`: Agent-owned memory tracking visited coordinates, observation coverage, discovered resources, depletion status, and action history strictly through public `Observation` objects.
  - `RechargeEvaluator`: Assesses whether recharging is beneficial. Prevents infinite looping at reusable resources, avoids recharging when energy already suffices to reach the target, and avoids unnecessary intermediate recharges when en route to a chosen waypoint.
  - `ResourceUtilityEvaluator`: Computes transparent route utility considering travel distance, recharge amount, post-recharge capacity, and remaining distance to target. Solves the nearest-insufficient vs farther-sufficient counterexample.
  - `ExplorationController`: Computes information gain over unobserved grid frontiers within safe energy limits, discovering resources outside initial sensory radius without hidden state access.
  - `CycleDetector`: Analyzes sliding window history to detect repeated recharges, ping-pong oscillations, repeated rejections, and unproductive position loops, triggering bounded recovery to alternative legal actions.
  - `RobustResourcePolicy`: High-level policy integrating the four robustness components with base learned Q-values (`LinearQLearner`).
- **Comprehensive 15-Scenario Benchmark Evaluation:**
  - Evaluated 15 distinct scenarios across 5 policies (Phase 12.2 Robust Linear Q, Phase 12.1 Linear Q Baseline, Tabular Q Baseline, Planner Reference, Masked Random).
  - **Phase 12.2 Robust Linear Q achieves 100.0% success across all 12 feasible tasks** (Phase 12.1 linear baseline achieved 75.0%, tabular Q achieved 8.3%, planner reference achieved 75.0%).
  - **Reusable Resource Loop (Scenario 5):** Solved 100.0% (improved from 0.0% timeout loop).
  - **Sensory Horizon Blindness (Scenario 6):** Solved 100.0% (improved from 0.0% energy exhaustion).
  - **Nearest-Insufficient Counterexample (Scenario 7):** Solved 100.0% (navigates to farther sufficient resource).
  - **Novel Holdout 5x5 (Scenario 15):** Solved 100.0% (improved from 0.0%).
  - **Provably Infeasible Tasks (Scenarios 10, 11, 13):** Properly recognized without loops or false claims (0.0% success across all policies).
- **Tooling & Reproducibility:**
  - Implemented `worldsim/benchmark_phase12_2.py` and `run_phase12_2_benchmark.py`.
  - Saved machine-readable results in `benchmark_results_phase12_2.json` (Phase 12.1 results preserved in `benchmark_results.json`).

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 105 tests in 0.73s.
- `python -m compileall -q worldsim tests` — passed.
- `python run_phase12_2_benchmark.py` — passed.

Files changed / added:
- `worldsim/robust_policy.py` (new)
- `worldsim/benchmark_phase12_2.py` (new)
- `run_phase12_2_benchmark.py` (new)
- `worldsim/__init__.py` (updated exports)
- `tests/test_robust_learning.py` (new 16 tests)
- `benchmark_results_phase12_2.json` (new benchmark data)
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 12.1

Phase 12.1 conducts an independent scientific audit of Phase 12's generalization claims:
- **Audit Findings & Bug Fixes:**
  - Corrected an accounting bug in the exploratory benchmark that miscounted exhaustion whenever success rate wasn't 1.0. Implemented strict mutually exclusive per-episode accounting ($\text{Success} + \text{Exhaustion} + \text{Timeout} = 100.0\%$).
  - Eliminated training diversity disparity: implemented `train_tasks` on `LearningRunner`, benchmarking both Tabular Q and Linear Q under identical single-task and multi-task protocols.
  - Benchmarked both Unmasked Random (all 5 actions) and Masked Random (legal actions only).
  - Evaluated `EnergyPlannerPolicy` across all 10 held-out configurations individually rather than using a single point-estimate reference.
- **Matched Single-Task & Multi-Task Comparisons:**
  - *Condition A (Single-Task, Train-1):* Tabular Q: 100% familiar, 0.0% held-out avg. Linear Q: 100% familiar, 75.0%–87.5% held-out avg across budgets (300, 900, 1500 ep).
  - *Condition B (Multi-Task, 3 Envs):* Tabular Q: 100% familiar, 0.0% held-out avg. Linear Q: 100% familiar, 62.5%–75.0% held-out avg across budgets (300, 900, 1500 ep).
  - *Conclusion:* Tabular Q-learning completely fails to generalize (0.0%) across coordinate changes regardless of training diversity, whereas linear function approximation achieves robust zero-shot transfer (62.5%–87.5%).
- **Documented Representation Limitations & Failure Modes:**
  - *Reusable Resource Looping:* Linear model loops `Action.RECHARGE` when standing on reusable resources because training only used consumable resources.
  - *Sensory Horizon Blindness:* Linear model exhausts energy if resource is beyond Euclidean sensory radius ($r=2.0$) from the initial position.
  - *Feature Aliasing:* `FeatureExtractor` nearest-resource heuristic can favor a closer resource with insufficient recharge over a further resource with sufficient recharge.
  - *Infeasible Tasks:* Linear agent cleanly exhausts energy on impossible tasks without lookahead proof.
- **Reproducible Machine-Readable Tooling:**
  - Implemented `worldsim/benchmark.py` and `run_benchmark.py` generating `benchmark_results.json`.

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 89 tests in 0.74s.
- `python -m compileall -q worldsim tests` — passed.

Files changed / added:
- `worldsim/learning.py` (added `train_tasks` and `world` override to `evaluate`)
- `worldsim/benchmark.py` (new comprehensive benchmark runner)
- `run_benchmark.py` (new root execution script)
- `worldsim/__init__.py` (exported `run_benchmark_audit` and updated docstring)
- `tests/test_generalization_audit.py` (new 6 audit regression tests)
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 12

Phase 12 implements feature-based linear function approximation to overcome the generalization limitation of tabular Q-learning:
- **Feature Extraction (`FeatureExtractor`)**: Computes normalized, relative features strictly from permitted `Observation` fields: relative target distance/direction, energy deficit, resource proximity, and action feasibility.
- **Linear Q-Learner (`LinearQLearner`)**: Evaluates $Q(s, a) = \mathbf{w}^\top \boldsymbol{\phi}(s, a)$ with semi-gradient TD(0) updates, gradient clipping, exploration decay, and action feasibility masking.
- **Multi-Task Task Distribution Training (`LinearLearningRunner`)**: Supports round-robin task interleaving across diverse environments, preventing catastrophic forgetting and breaking collinearity between target and resource directions.
- **Generalization Across Benchmark Suite (Experiments A–F)**:
  - **Exp A (Familiar Config)**: Linear Q: 100.0% (6.00 steps), Tabular Q: 100.0% (6.00 steps), Planner: 100.0% (6.00 steps), Random: 0.0%.
  - **Exp B (Unseen Target (3,0))**: Linear Q: 100.0% (4.00 steps), Tabular Q: 0.0% (timeout 100%), Planner: 100.0% (3.00 steps), Random: 5.0%.
  - **Exp C (Unseen Start (1,1))**: Linear Q: 100.0% (6.00 steps), Tabular Q: 100.0% (6.00 steps), Planner: 100.0% (3.00 steps), Random: 7.0%.
  - **Exp D (Unseen Resource Layout (1,1))**: Linear Q: 100.0% (6.00 steps), Tabular Q: 0.0% (timeout 100%), Planner: 100.0% (6.00 steps), Random: 1.0%.
  - **Exp E1 (Combined Shift Solvable)**: Linear Q: 100.0% (6.00 steps), Tabular Q: 0.0% (timeout 100%), Planner: 100.0% (6.00 steps), Random: 1.0%.
  - **Exp E2 (Combined Shift Unsolvable)**: Linear Q: 0.0% (energy exhausted 100%), Tabular Q: 0.0% (timeout 100%), Planner: 0.0% (0 steps executed), Random: 0.0%.
  - **Exp F (Observation Limitations)**: Verified mathematically and empirically that when `observable_target=None`, all target features evaluate strictly to 0.0 with zero information leakage.

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 83 tests in 0.53s.
- `python -m compileall -q worldsim tests` — passed.

Files changed / added:
- `worldsim/linear_learning.py` (new)
- `worldsim/__init__.py`
- `tests/test_linear_learning.py` (new)
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 11

Phase 11 extends tabular Q-learning to energy-constrained navigation with consumable and reusable resources:
- **Observable Target Interface**: Added optional `observable_target` to `WorldConfig` and exposed `target` on `Observation`, enabling clean information-boundary-respecting experiments where the goal is explicitly observable.
- **Extended State Representation**: `QLearner` and `LearningConfig` support composite state keys `(x, y, energy, target, resources)` derived strictly from permitted `Observation` fields. Resource state tracks observable active coordinates, dynamically updating when consumable resources are depleted.
- **Recharge Action & Full Action Space**: Added `Action.RECHARGE` to the learner's action space alongside directional moves (`UP`, `DOWN`, `LEFT`, `RIGHT`).
- **Reward Shaping & Exploration Decay**: Added `failed_action_penalty` (penalizing invalid moves or invalid recharges), `recharge_cost`, `terminal_failure_penalty`, `epsilon_decay`, and `terminal_on_energy_exhaustion` to avoid futile steps when energy reaches 0 without a resource present.
- **Metrics & Observability**: `EpisodeResult` and `EvaluationResult` record `recharge_count`, `rejected_actions`, and `final_energy`.

Measured controlled experiment (4x4 world, start `(0, 0)`, target `(3, 2)`, Manhattan distance 5, `initial_energy=3`, `max_energy=6`, consumable resource at `(2, 0)` restoring 4):
- **Learned Q-Policy (1500 training episodes, 100 greedy evaluation episodes)**:
  - Success Rate: 100.0% (100/100)
  - Mean Steps (successes): 6.00 (optimal path: `RIGHT` -> `RIGHT` -> `RECHARGE` -> `DOWN` -> `RIGHT` -> `DOWN`)
  - Mean Cumulative Reward: 0.9500
  - Mean Remaining Energy: 2.00
  - Mean Recharge Actions: 1.00
  - Mean Rejected Actions: 0.00
- **Planner-Backed Policy (`EnergyPlannerPolicy`)**:
  - Success Rate: 100.0% (1/1)
  - Steps: 6 (optimal)
  - Cumulative Reward: 0.9500
  - Remaining Energy: 2
  - Recharge Actions: 1
  - Rejected Actions: 0
- **Energy-Aware Heuristic Policy (`EnergyAwarePolicy`)**:
  - Success Rate: 100.0% (1/1)
  - Steps: 6 (optimal)
  - Cumulative Reward: 0.9500
  - Remaining Energy: 2
  - Recharge Actions: 1
  - Rejected Actions: 0
- **Random Baseline (100 seeded episodes)**:
  - Success Rate: 0.0% (0/100)
  - Mean Steps: N/A
  - Mean Cumulative Reward: -0.2500
  - Mean Remaining Energy: 0.00
  - Mean Recharge Actions: 0.01
  - Mean Rejected Actions: 21.95

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 71 tests in 0.024s.
- `python -m compileall -q worldsim tests` — passed.

Files changed:
- `worldsim/models.py`
- `worldsim/config.py`
- `worldsim/perception.py`
- `worldsim/engine.py`
- `worldsim/learning.py`
- `worldsim/__init__.py`
- `tests/test_learning_energy.py`
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 10

Phase 10 adds energy-aware state-space planning and safe plan execution:
- **State-Space Planner (`plan_energy`)**: Solves `EnergyPlanningProblem` over the joint state space `(position, energy, active_resources)` via BFS. Finds optimal action paths respecting movement costs, maximum energy capacity, and consumable/reusable resource depletion.
- **Safe Plan Execution (`execute_energy_plan`)**: Validates that simulation dimensions, initial position, energy parameters, and resources strictly match the planning problem before stepping. Halts immediately if an action fails or actual state diverges from planned state, returning a structured `PlanExecutionResult`.
- **Planner-Backed Policy (`EnergyPlannerPolicy`)**: Operates within observation boundaries using only permitted `Observation` fields (`energy`, `observed_resources`). Replans dynamically as new resources are observed or consumed and never issues impossible actions.
- **Learning Architecture Integration**: `LearningConfig` includes an opt-in `include_energy: bool = False` flag; when enabled, `QLearner.state_key` returns `(x, y, energy)`. Full energy-aware reinforcement learning remains future work due to sample complexity across consumable resource states.

Measured controlled experiments across 6 benchmark scenarios:
- **Scenario 1 (Direct Reachable, $d=3$, $E=5$)**: Privileged Planner: 3 steps, $E_f=2$. Planner Policy: 3 steps, 100%. Heuristic: 3 steps, 100%. Energy-Ignorant: 3 steps, 100%. Random: 10% (5/50).
- **Scenario 2 (Single Recharge, $d=6$, $E=4$)**: Privileged Planner: 7 steps, 1 rech, $E_f=3$. Planner Policy: 7 steps, 100%. Heuristic: 7 steps, 100%. Energy-Ignorant: 0% (26 rej). Random: 0% (0/50).
- **Scenario 3 (Multiple Recharges, $d=10$, $E=4$)**: Privileged Planner: 12 steps, 2 rech, $E_f=2$. Planner Policy: 12 steps, 100%. Heuristic: 12 steps, 100%. Energy-Ignorant: 0% (26 rej). Random: 0% (0/50).
- **Scenario 4 (Unreachable Target, $d=8$, $E=3$)**: Privileged Planner: Found=False. Planner Policy: 0 steps (safe halt, avoids futile execution). Heuristic: 3 steps (stops at energy 0). Energy-Ignorant: 0% (27 rej). Random: 0%.
- **Scenario 5 (Consumable Depletion, $d=6$, $E=3$)**: Privileged Planner: 7 steps, 1 rech, $E_f=0$. Planner Policy: 7 steps, 100%. Heuristic: 7 steps, 100%. Energy-Ignorant: 0% (27 rej). Random: 0%.
- **Scenario 6 (Reusable Resource, $d=4$, $E=2$)**: Privileged Planner: 5 steps, 1 rech, $E_f=0$ (optimal). Planner Policy: 5 steps, 1 rech, 100% (optimal). Heuristic: 6 steps, 2 rech (greedy redundant recharge). Energy-Ignorant: 0% (28 rej). Random: 2% (1/50).

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 64 tests.
- `python -m compileall -q worldsim tests` — passed.

Files changed:
- `worldsim/planning.py`
- `worldsim/policy.py`
- `worldsim/learning.py`
- `worldsim/__init__.py`
- `tests/test_energy_planning.py`
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 9

Phase 9 implements an energy system, resource interactions, and energy-aware action planning:
- **Energy System**: Configured via `initial_energy` (default 10), `max_energy` (default 10), and `movement_energy_cost` (default 1). Movement consumes energy; insufficient energy or boundary collisions prevent movement without advancing the clock or consuming energy.
- **Resource Placement**: Explicit `Resource(position, recharge_amount, consumable)` instances placed in `WorldConfig.resources`. Consumable resources are removed upon recharge; reusable resources remain active. Recharging cannot exceed `max_energy`.
- **Extended Actions**: Added `Action.RECHARGE`. Valid only when positioned on an active resource cell; invalid recharge attempts are rejected without tick or energy changes.
- **Perception & Experience**: `Observation` exposes `energy` and `observed_resources` within sensory radius. `Experience` records `energy_before`, `energy_after`, and interaction `success`.
- **Decision Policies**: Implemented `EnergyAwarePolicy` in `worldsim.policy`, which monitors energy and observable resources, detouring to recharge when energy is insufficient to reach the target directly.
- **Multi-step Execution**: Extended `run_experiment` and added `run_policy_experiment` with `stop_on_rejected` option.

Measured controlled experiment (5x5 world, start `(0, 0)`, target `(4, 4)`, distance 8, initial energy 5, max 10, consumable resource at `(2, 1)` restoring 7 energy):
- **Energy-Ignorant Policy**: Task completed: False (0%). Exhausted energy at step 5 at position `(4, 1)`. 25 rejected moves, 0 final energy.
- **Energy-Aware Policy**: Task completed: True (100%). Detoured to `(2, 1)`, recharged to 9 energy, and reached `(4, 4)` in 9 steps. Final energy: 4. 0 rejected moves.
- **Random Baseline (100 seeded runs)**: Task completed: 1% (1/100). Mean final energy: 0.03, mean recharges: 0.21.

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 53 tests.
- `python -m compileall -q worldsim tests` — passed.

Files changed:
- `worldsim/models.py`
- `worldsim/actions.py`
- `worldsim/config.py`
- `worldsim/perception.py`
- `worldsim/experience.py`
- `worldsim/engine.py`
- `worldsim/experiment.py`
- `worldsim/policy.py`
- `worldsim/viewer.py`
- `worldsim/__init__.py`
- `tests/test_energy.py`
- `README.md`
- `WORLD_SIM_PLAN.md`
- `WORLD_SIM_HANDOFF.md`

## Latest update: Phase 8

Phase 8 adds standalone classical planning and plan execution in `worldsim.planning`: `PlanningProblem`, `PlanResult`, `plan(problem)`, and `execute_plan(config, result)`. Planning operates strictly on explicit external inputs (`width`, `height`, `start`, `target`) using breadth-first search (BFS). The planner is completely separate from simulation state, entity perception, and learning; plan execution runs actions strictly through the public `Simulation.step` API.

Expanded test suite in `tests/test_planning.py` covers:
- Start equals target returning an empty action tuple and empty execution.
- Shortest-path optimality across diverse Manhattan distances and directions.
- Dimension validation and boundary/out-of-bounds rejection.
- Unreachable/empty plan execution.
- Execution consistency between planning start and simulation start.
- Step-by-step adherence to simulation API contracts.

Measured controlled experiment (4x4 world, start `(0, 0)`, step cost `-0.01`, target reward `1.0`, max 30 steps, Q-learning 500 training episodes with seed `7`, evaluation 100 episodes with seed `99`):
- **Target `(1, 0)` (Manhattan dist 1):**
  - BFS Planner: 100% success (1/1), 1 step (optimal), cumulative reward 1.0000, 0 training episodes.
  - Tabular Q-Learning (trained for target): 100% success (100/100), mean 1.00 step, mean cumulative reward 1.0000.
  - Random Baseline: 76% success (76/100), mean 6.04 steps among successes, mean cumulative reward 0.6497.
- **Target `(2, 1)` (Manhattan dist 3):**
  - BFS Planner: 100% success (1/1), 3 steps (optimal), cumulative reward 0.9800, 0 training episodes.
  - Tabular Q-Learning (trained for target): 100% success (100/100), mean 3.00 steps, mean cumulative reward 0.9800.
  - Random Baseline: 76% success (76/100), mean 12.84 steps among successes, mean cumulative reward 0.5980.
- **Target `(0, 3)` (Manhattan dist 3):**
  - BFS Planner: 100% success (1/1), 3 steps (optimal), cumulative reward 0.9800, 0 training episodes.
  - Tabular Q-Learning (trained for target): 100% success (100/100), mean 3.00 steps, mean cumulative reward 0.9800.
  - Random Baseline: 48% success (48/100), mean 16.17 steps among successes, mean cumulative reward 0.2512.
- **Target `(3, 3)` (Manhattan dist 6):**
  - BFS Planner: 100% success (1/1), 6 steps (optimal), cumulative reward 0.9500, 0 training episodes.
  - Tabular Q-Learning (trained for target): 100% success (100/100), mean 6.00 steps, mean cumulative reward 0.9500.
  - Random Baseline: 38% success (38/100), mean 18.87 steps among successes, mean cumulative reward 0.1261.

**Generalization Limitation Analysis:**
BFS possesses privileged global target and world knowledge, allowing zero-shot optimal planning. Tabular Q-learning represents state solely as `(observed_x, observed_y)` and learns target-directed behavior purely via reward feedback during 500 training episodes. When a Q-learning policy trained on target `(3, 3)` is evaluated on targets `(1, 0)`, `(2, 1)`, or `(0, 3)` without retraining, it achieves 0/100 successes (mean cumulative reward -0.3000), confirming that target-unaware tabular policies cannot generalize across varying goals without retraining.

Files changed: `worldsim/planning.py`, `worldsim/__init__.py`, `tests/test_planning.py`, `README.md`, `WORLD_SIM_PLAN.md`, and `WORLD_SIM_HANDOFF.md`.

Tests run after implementation:
- `python -m unittest discover -s tests -v` — passed: 42 tests.
- `python -m compileall -q worldsim tests` — passed.
- Fixed-seed multi-target comparison experiment — passed with the metrics above.

## Latest update: Phase 7


Added `worldsim.memory.ExperienceMemory` and frozen `MemoryEntry`: permitted observation/action/reward/result-observation transitions, FIFO eviction, coordinate matching, tuple snapshots, and explicit clearing. It never reads target or private world state. Memory is opt-in through `LearningRunner(memory=...)`; training records/retrieves it without changing Q-learning, episode resets retain it, and evaluation writes neither memory nor Q-values. The full suite passed 36 tests and compilation passed. The fully observed control found no independent memory advantage over Q-learning. Files changed: `worldsim/memory.py`, `worldsim/learning.py`, `worldsim/__init__.py`, `tests/test_memory.py`, and these documents.

## Latest update: Phase 6

Phase 6 adds `worldsim.learning`: `LearningConfig`, a zero-initialized `QLearner`, `LearningRunner`, and immutable transition/episode/evaluation results. Learning is separate from the engine, actions, and viewer. The learner keys Q-values only by `Observation.entity_position`, so it uses permitted perception rather than hidden world data.

The external learning configuration defines fixed target, target reward, nonterminal step cost, alpha, gamma, epsilon, and episode limit. Reaching the target terminates the episode and omits future-value bootstrapping. Every episode starts with a fresh simulation; learned Q-values persist across `train()` calls. Evaluation uses no exploration and does not update Q-values; the random baseline uses a separately seeded policy.

Measured controlled experiment: 4x4 world, start `(0, 0)`, target `(3, 3)`, alpha `0.3`, gamma `0.9`, epsilon `0.2`, 30-step limit, 500 training episodes, training seed `7`, evaluation seed `99`. Greedy learned policy: 100/100 successes, mean 6.0 steps. Random baseline: 38/100 successes, mean 18.87 steps among successes. This demonstrates learning only for this defined toy target task.

Files changed: `worldsim/learning.py`, `worldsim/__init__.py`, `tests/test_learning.py`, `README.md`, `WORLD_SIM_PLAN.md`, and `WORLD_SIM_HANDOFF.md`.

Tests run after implementation:

- `python -m unittest discover -s tests -v` - passed: 33 tests.
- `python -m compileall -q worldsim tests` - passed.
- Fixed-seed learning/baseline experiment - passed with the metrics above.

## Latest update: Phase 5

Phase 5 adds the headless `run_experiment(config, actions)` API and frozen `ExperimentResult` in `worldsim.experiment`. A run creates its own private `Simulation`, snapshots the action iterable as a tuple, calls only `Simulation.step`, and returns the initial configuration/position, actions, ordered `StepResult` and `Experience` tuples, final position/tick, plus completion/error information.

The current simulation has no random behavior, so experiments intentionally take no seed and are reproducible from equivalent configuration and actions. Boundary actions preserve existing rejected-action semantics. An invalid action stops execution immediately; its index and error are returned, no later action runs, and it produces no experience record. The runner adds no UI, persistence, learning, or autonomous behavior.

Files changed: `worldsim/experiment.py`, `worldsim/__init__.py`, `tests/test_experiment.py`, `README.md`, `WORLD_SIM_PLAN.md`, and `WORLD_SIM_HANDOFF.md`.

Tests run after implementation:

- `python -m unittest discover -s tests -v` - passed: 28 tests.
- `python -m compileall -q worldsim tests` - passed.
- Experiment smoke check - passed: `RIGHT`, `DOWN` from `(1, 1)` completed at `(2, 2)`, tick `2`, with 2 experiences.

## Latest update: Phase 4

Phase 4 adds memory-only experience recording inside `Simulation.step()`. `Experience` is a frozen record containing `tick_before`, `observation_before`, `action`, `moved`, `position_before`, `position_after`, `observation_after`, and `tick_after`. It reuses the existing immutable domain models.

Every valid action request records one experience in chronological order: successful movement, boundary rejection, and paused rejection are all recorded. Unknown or malformed actions raise `ValueError` before any record or mutation. `Simulation.experiences` returns a tuple, protecting the private list; records and nested observations cannot be mutated.

The Tkinter UI is unchanged. History is in memory only and does not add learning, replay, persistence, or autonomous behavior.

Files changed: `worldsim/experience.py`, `worldsim/engine.py`, `worldsim/__init__.py`, `tests/test_experience.py`, `README.md`, `WORLD_SIM_PLAN.md`, and `WORLD_SIM_HANDOFF.md`.

Tests run after implementation:

- `python -m unittest discover -s tests -v` - passed: 23 tests.
- `python -m compileall -q worldsim tests` - passed.
- Experience smoke check - passed: default `Action.RIGHT` recorded action `right`, tick `0 -> 1`, and position `(10, 6) -> (11, 6)`.

## Latest update: Phase 3

Phase 3 adds controlled movement only. `worldsim.actions.Action` defines exactly `UP`, `DOWN`, `LEFT`, and `RIGHT`; `Simulation.step(action)` validates the enum, calculates one destination, validates it before assigning immutable state, and returns `StepResult(state, observation, moved)`. Success increments the tick once. Boundary attempts and all actions while paused preserve state and tick; unknown values raise `ValueError` before mutation.

The viewer has directional buttons and arrow-key bindings, both routed solely through `Simulation.step`. It redraws the returned state, so the existing perception tint updates with the entity.

Files changed: `worldsim/actions.py`, `worldsim/models.py`, `worldsim/engine.py`, `worldsim/viewer.py`, `worldsim/__init__.py`, `tests/test_actions.py`, `README.md`, and `WORLD_SIM_PLAN.md`.

Tests run after implementation:

- `python -m unittest discover -s tests -v` - passed: 17 tests.
- `python -m compileall -q worldsim tests` - passed.
- Action/perception smoke check - passed: `Action.RIGHT` moved the default entity to `(11, 6)`, advanced tick to `1`, and returned a fresh observation.

## Latest update: Phase 2

Phase 2 adds a configurable Euclidean sensory radius and `Simulation.observe()`. Its immutable `Observation` exposes the entity coordinate and only valid cells within `dx^2 + dy^2 <= radius^2`. `WorldViewer` tints exactly those observed cells before drawing the grid and entity. No movement, learning, resources, memory, or other Phase 3 behavior was added.

Files changed for this phase: `worldsim/config.py`, `worldsim/models.py`, `worldsim/engine.py`, `worldsim/perception.py`, `worldsim/viewer.py`, `worldsim/__init__.py`, `tests/test_perception.py`, `README.md`, and `WORLD_SIM_PLAN.md`.

Tests run after implementation:

- `python -m unittest discover -s tests -v` - passed: 12 tests.
- `python -m compileall -q worldsim tests` - passed.
- Perception import/smoke check - passed; the default radius returns 13 cells.

## Current state

Phase 1 baseline is implemented in an initially empty shared workspace. The headless engine initializes a bounded grid and one entity, increments a deterministic clock, and pauses/resumes. A Tkinter viewer displays the state and exposes a control button.

## Files changed

- `worldsim/config.py` — validated configuration
- `worldsim/models.py` — immutable world domain objects
- `worldsim/engine.py` — headless simulation and clock
- `worldsim/__init__.py`, `worldsim/viewer.py`, and `worldsim/__main__.py` — public API, visualization, and launch point
- `tests/test_simulation.py` — initialization, coordinates, placement, ticks, pause/resume, reproducibility
- `README.md`, `WORLD_SIM_PLAN.md`, `.gitignore` — instructions, architectural record, and generated-file exclusions

## Tests run

- `python -m unittest discover -s tests -v` — passed: 6 tests.
- `python -m compileall -q worldsim tests` — passed.
- `python -c "import tkinter; import worldsim"` — passed.

## Verification review (2026-10-21)

- No Antigravity findings are recorded in this handoff. Its only requested review area is viewer readability and the pause/resume affordance.
- Verified each Phase 1 acceptance item in `WORLD_SIM_PLAN.md`: configurable bounded grid; one ID-bearing entity at its configured position; headless deterministic clock; optional rendering adapter; pause/resume; tests; and run documentation.
- No confirmed Phase 1 defect was found. No implementation files were changed, and Phase 2 was not started.

## Outstanding issues

- The GUI needs a desktop/display environment for manual launch verification; engine tests do not require one.
- No future-life mechanics have been introduced.

## Suitable Antigravity tasks

- Review the running viewer’s readability, grid sizing, and pause/resume affordance.
- Propose visual refinements only; preserve the headless engine boundary and document any behavior-affecting proposal before implementation.
