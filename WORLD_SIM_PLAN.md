# WorldSim Plan

## Phase 14.4 status

Phase 14.4 (Decentralized Learning & Communicative Intent) introduces decentralized multi-agent reinforcement learning with linear function approximation and local intent communication under strict partial observability ($R \le 2.0$), while preserving the heuristic policy as the control baseline:
1. **Preflight Metric Audit (D01 Collision Telemetry Analysis):**
   - Conducted step-by-step diagnostic audit of Phase 14.3 D01 reported collisions: confirmed all 40 occurrences were swap conflict attempts on tick 5 between `agent_w` and `agent_e` ($2 \text{ agents} \times 1 \text{ tick} \times 20 \text{ episodes} = 40$).
   - Confirmed authoritative simulation engine prevented any physical overlap (zero illegal double occupancy, physical safety invariant held 100%).
   - Metric represents attempted conflicting actions evaluated by the engine, not realized penetrations.
2. **Decentralized Learning Architecture (`worldsim/decentralized_learning.py`):**
   - `DecentralizedLearningConfig`: learning rate 0.05, discount 0.95, gradient clipping 2.0, L2 regularization 1e-4, multi-objective reward weights.
   - `DecentralizedFeatureExtractor`: normalized relative feature representation within local sensory horizon ($R \le 2.0$), covering goal displacement, kinematic validity, energy margins, dynamic obstacle prediction, inactive obstacles, peer proximity, and local intent communication (`comm_peer_intent_vertex_conflict`, `comm_peer_swap_conflict`, `comm_priority_yield_detour`, `comm_my_priority_advance`).
   - `DecentralizedQLearner`: TD(0) linear function approximation with L2 weight decay, clipped TD errors, epsilon-greedy exploration, and JSON serialization.
   - `LearnedCoordinationPolicy`: policy wrapper adhering to WorldSim policy interface, incorporating agent-local memory (`AgentMemory`), legal action masking, and coordinator intent registration when communication is enabled.
   - `DecentralizedRewardEngine`: multi-objective reward function balancing individual target progress (+5.0), goal completion (+100.0), energy conservation (-0.5), collision avoidance (-15.0 swap/-15.0 vertex/-20.0 dynamic), safe yielding (-0.2), and repeated wait penalties (-4.0).
   - `train_decentralized_coordination`: multi-agent training runner across 5 diverse training environments using training seeds 42..91, strictly disjoint from evaluation seeds 500..519.
3. **Dedicated Unit Tests (`tests/test_decentralized_learning.py`):**
   - 8 comprehensive unit tests:
     - `test_feature_extractor_local_horizon_isolation`: Verifies features only use local information within $R \le 2.0$; distant peers and unobserved cells produce zero activations.
     - `test_intent_communication_toggle`: Peer intent features activate only when `enable_communication=True`.
     - `test_q_learner_td_update_and_serialization`: Weights update correctly via TD(0) and serialize/deserialize without loss.
     - `test_policy_action_selection_and_masking`: Masking strictly disallows illegal moves or moves without energy.
     - `test_reward_engine_penalties_and_incentives`: Verifies progress, collision, and energy exhaustion rewards.
     - `test_training_seed_separation`: Training seeds 42..91 strictly disjoint from evaluation seeds 500..519.
     - `test_symmetry_breaking_with_communication`: Communication breaks head-on deadlock and detours lower-priority agent.
     - `test_authoritative_simulation_physics_preservation`: Authoritative physics blocks swap collisions with zero physical overlap.
   - Full test suite: **232 passed in 8.63s** (`python -m pytest`).
4. **Comparative Benchmark Matrix (Seeds 500–519):**
   - **D01:** Heuristic 100% (14.0 steps, 40 swap attempts), Learned No-Comm 50.0% (10.0 steps, 920 conflicts), Learned Comm **100.0% (11.0 steps, 0 collisions)** — communication completely eliminated head-on swap conflict attempts!
   - **D02:** Heuristic 100% (14.0 steps), Learned No-Comm **100.0% (10.0 steps, 0 collisions)**, Learned Comm **100.0% (10.0 steps, 0 collisions)**.
   - **D03:** Heuristic 100% (13.0 steps), Learned No-Comm 0.0%, Learned Comm **100.0% (17.0 steps)**.
   - **D04:** Heuristic 100% (12.0 steps), Learned No-Comm 50.0% (75% agent comp), Learned Comm 0.0% (50% agent comp).
   - **U06 Regression:** Heuristic **100.0% completion, 0 exhaustions** (maintains >=95% requirement).
   - **U14 Regression:** Heuristic **100.0% completion, 0 exhaustions** (maintains >=95% requirement). Under Learned Comm, collision attempts dropped from 2410 down to **0**.
5. **Artifacts & Deliverables:**
   - `phase14_4_learned_coordination_eval.py`: Full evaluation harness.
   - `phase14_4_weights_no_comm.json` & `phase14_4_weights_comm.json`: Trained model weights.
   - `phase14_4_benchmark_results.json`: Machine-readable results across all 18 configurations.
   - `phase14_4_learned_coordination_report.md`: Detailed audit and comparative evaluation report.

## Phase 14.4 checklist

| Requirement | Status |
| --- | --- |
| Preflight metric audit: D01 40 collisions analyzed and classified as attempted swap conflicts | Complete |
| Documented POMDP state representation using local features ($R \le 2.0$) | Complete |
| Linear function-approximation Q-learner with TD(0) and weight decay | Complete |
| Communicative (`enable_communication=True`) and non-communicative (`False`) policy modes | Complete |
| Strict local information boundary ($R \le 2.0$) with no global trajectory leakage | Complete |
| Authoritative simulation physics preventing illegal double occupancy | Complete |
| Strict seed separation: Training (42..91) disjoint from Evaluation (500..519) | Complete |
| Multi-objective reward balancing progress, collision penalties, and team fairness | Complete |
| Dedicated unit tests `tests/test_decentralized_learning.py` (8 tests) | Complete |
| Full pytest test suite passing (232 passed) | Complete |
| Cross-process determinism verified across randomized `PYTHONHASHSEED` values | Complete |
| Comparative evaluation benchmark across D01–D04, U06, U14 on seeds 500–519 | Complete |
| U06 and U14 maintain 100.0% completion (>=95% requirement satisfied) | Complete |
| Heuristic baseline preserved as default policy | Complete |
| Structured results exported to `phase14_4_benchmark_results.json` | Complete |
| Comprehensive audit report exported to `phase14_4_learned_coordination_report.md` | Complete |
| Trained weights exported to JSON (`phase14_4_weights_no_comm.json`, `phase14_4_weights_comm.json`) | Complete |

## Phase 14.3 status

Phase 14.3 (Dynamic Obstacle Avoidance) introduces moving obstacles and local velocity-aware prediction and avoidance while preserving the working centralized baseline and decentralized local-horizon coordination modes:
1. **Dynamic Obstacles Architecture (`worldsim/dynamic_obstacles.py`):**
   - Configurable moving obstacles with deterministic movement patterns: `"linear_bounce"` (moves along velocity vector `(dx, dy)`, bouncing on grid boundaries, walls, resources, agents, and other obstacles), `"patrol"` (cycles or ping-pongs deterministically through waypoints), and `"fixed"`.
   - Immutable state tracking: `DynamicObstacleConfig` and `DynamicObstacleState` recording runtime position, velocity, waypoint index, and reversal state.
   - Authoritative simulation physics: `MultiAgentSimulation._resolve_conflicts` enforces strict non-overlapping physics across agents, dynamic obstacles, and resources, guaranteeing zero illegal double-occupancy under all conditions.
   - Comprehensive trajectory and telemetry tracking: `dynamic_obstacle_history` and `dynamic_obstacle_collisions` recorded per step.
2. **Local Prediction & Velocity-Aware Avoidance (`worldsim/robust_policy.py`, `worldsim/coordination.py`):**
   - In `AgentMemory`: Ingests moving obstacle positions and velocities strictly within local sensory radius ($R \le 2.0$), performs visual revalidation against current field of view to clear moved obstacles, tracks TTL expiration (`dynamic_obstacle_ttl = 6`), and computes predicted obstacle cells over a configurable short horizon (`get_predicted_dynamic_obstacle_cells(horizon=2)`).
   - In `RobustResourcePolicy`: In the presence of dynamic obstacles, permits `Action.RECHARGE` as an intentional wait in place; applies heavy penalty (`-200.0`) if an obstacle is moving into the agent's current position, and rewards safe waiting (`+15.0`) when crossing corridors or pad doorsteps while an obstacle traverses nearby.
   - Goal-aligned topological path planning: Refined BFS direction exploration to align with target displacement, preventing oscillatory backtracking onto recently vacated recharge pads while smoothly detouring laterally around moving obstacles.
   - Coordinator integration: `suggest_alternative_move` evaluates obstacle positions and predicted paths within local range, pruning collision-prone alternatives and allowing safe waiting unless an obstacle is advancing into the agent's cell.
3. **Dedicated Unit Tests (`tests/test_dynamic_obstacle_avoidance.py`):**
   - 8 comprehensive unit tests:
     - `test_deterministic_obstacle_trajectories`: Bit-for-bit trajectory determinism across repeated runs.
     - `test_obstacle_approaching_agent_destination`: Dynamic obstacle approaching planned next cell causes agent to yield safely.
     - `test_two_moving_obstacles_crossing_paths`: Deterministic bounce between intersecting moving obstacles.
     - `test_stale_observations_and_outside_communication_range`: Distant obstacles outside $R=2.0$ never leak to agent memory or coordinator.
     - `test_obstacle_blocked_by_wall_agent_and_resource`: Movement legality, boundary bouncing, and zero overlap invariants.
     - `test_safe_waiting_and_eventual_progress`: Safe waiting in place until a patrolling obstacle vacates a transit door.
     - `test_u06_quadrant_mesh_regression_coverage`: U06 retains 100.0% completion with 0 exhaustions.
     - `test_u14_blind_junction_regression_coverage`: U14 retains 100.0% completion with 0 exhaustions.
   - Full test suite: **224 passed in 8.29s** (`python -m pytest`).
4. **Controlled Benchmark Suite (D01–D04 across seeds 500–519):**
   - **D01 (Dual Patrol Chokepoint Crossing):** Static 100.0% (9.0 steps), Centralized 100.0% (14.0 steps, 0 dyn collisions), Decentralized 100.0% (14.0 steps, 0 dyn collisions).
   - **D02 (Dynamic Bottleneck Mesh):** Static 100.0% (13.0 steps), Centralized 100.0% (14.0 steps, 0 dyn collisions), Decentralized 100.0% (14.0 steps, 0 dyn collisions).
   - **D03 (Dynamic Obstacle Grid Mesh):** Static 100.0% (13.0 steps), Centralized 100.0% (13.0 steps, 0 dyn collisions), Decentralized 100.0% (13.0 steps, 0 dyn collisions).
   - **D04 (Resource Approach with Moving Obstacle):** Static 100.0% (13.5 steps), Centralized 100.0% (12.0 steps, 0 dyn collisions), Decentralized 100.0% (12.0 steps, 0 dyn collisions).
   - **U06 Regression:** 100.0% completion across seeds 500–519 (0 exhaustions).
   - **U14 Regression:** 100.0% completion across seeds 500–519 (0 exhaustions).
   - **Total Dynamic Collisions:** 0 across all evaluated dynamic modes and seeds.

## Phase 14.3 checklist

| Requirement | Status |
| --- | --- |
| Opt-in dynamic obstacle configuration (`dynamic_obstacles` in `MultiAgentWorldConfig`, static default) | Complete |
| Deterministic obstacle patterns (`linear_bounce`, `patrol`, `fixed`) with velocity tracking | Complete |
| Authoritative physical simulation engine preventing illegal overlaps (zero double-occupancy) | Complete |
| Local sensory horizon restriction ($R \le 2.0$) for dynamic obstacle observation in decentralized mode | Complete |
| Short-horizon (1–2 ticks) collision prediction and safe wait (`Action.RECHARGE`) / lateral detour | Complete |
| Stale dynamic obstacle memory visual revalidation and TTL expiration (`dynamic_obstacle_ttl = 6`) | Complete |
| Dedicated unit test suite `tests/test_dynamic_obstacle_avoidance.py` (8 tests) | Complete |
| Full pytest test suite passing (224 passed) | Complete |
| Cross-process determinism verified across randomized `PYTHONHASHSEED` values | Complete |
| Controlled dynamic benchmark suite (D01–D04) evaluated across seeds 500–519 (Static vs Cent vs Decent) | Complete |
| Structured results exported to `phase14_3_benchmark_results.json` | Complete |
| Comprehensive audit report exported to `phase14_3_dynamic_obstacle_report.md` | Complete |
| U06 and U14 retain 100.0% completion with 0 exhaustions | Complete |
| Zero changes to default centralized baseline behavior | Complete |

## Phase 14.2 Remediation status

Phase 14.2 Remediation resolves the two known decentralized-mode regressions (U06 and U14) while strictly preserving local sensory horizon boundaries ($R = 2.0$), authoritative physical simulation integrity, and 100% bit-for-bit backward compatibility with the centralized baseline:
1. **Targeted Fix A (U06 — Energy-Safe Resource Selection under Horizon Blindness):**
   - In `worldsim/coordination.py`, updated `MultiAgentCoordinator.get_resource_load`: in decentralized mode, when a queried resource is beyond the local communication radius ($R > 2.0$), it returns `None` rather than `0`, explicitly representing the queue load as unknown.
   - In `worldsim/robust_policy.py`, updated `ResourceUtilityEvaluator.compare_routes`: when resource queue load is unknown (`load is None`), applies a conservative load penalty (assuming at least 1 queued agent) and enforces a safety travel margin (`energy - travel_cost >= 2`).
   - Prevents agents with tight energy reserves from abandoning safe local resources to chase distant unobservable pads across the board.
   - Scenario U06 (8-Agent Staggered Quadrant Mesh) achieves **100.0% completion** across seeds 500–519 (mean 17.0 steps, 0 exhaustions, matching centralized baseline).
2. **Targeted Fix B (U14 — Local-Horizon Crossroad Junction Recognition & Doorway Yielding):**
   - In `worldsim/coordination.py`, updated `is_crossroad(p)`: recognizes internal 4-way crossroad junctions under local horizon ($R \le 2.0$) from doorway thresholds where `cardinals_open`, `known_diags >= 2`, and internal coordinates (`1 < p.x < width - 2`, `1 < p.y < height - 2`), preventing false positives on boundary cells, doorway exits, or outer walls.
   - Preserves traffic clearance right-of-way: vehicles inside the crossroad have right-of-way, while vehicles entering the crossroad yield at the entrance bottleneck.
   - Scenario U14 (Multi-Corner Blind Intersection) improves from **0.0% to 100.0% completion** across seeds 500–519 (mean 14.0 steps, 0 exhaustions, matching centralized baseline).
3. **Comprehensive Empirical Validation Across All 32 Scenarios:**
   - **Baseline Benchmark (16 Scenarios):** All 13 feasible baseline scenarios achieve **100.0% completion** in both Centralized and Decentralized modes. All 3 infeasible baseline scenarios (4A, 5A, 8) honestly achieve 0.0% completion in both modes.
   - **Unseen Generalization (16 Scenarios):** All 14 feasible unseen scenarios (U01–U10, U12–U15) achieve **100.0% completion** in both Centralized and Decentralized modes (0 regressions, 0 exhaustions). Both infeasible unseen scenarios (U11, U16) honestly achieve 0.0% completion in both modes.
   - Overall feasible completion rate across all 27 feasible scenarios: **100.0% Centralized, 100.0% Decentralized**.
4. **Test Suite & Verification:**
   - Dedicated unit test suite `tests/test_local_horizon_remediation.py` (6 tests covering reachable vs unreachable resources, unknown load handling, energy margins, partial pillar crossroad recognition, boundary cell discrimination, and end-to-end integration).
   - Full test suite: **216 passed in 6.75s** (`python -m pytest`).
   - Bit-for-bit trajectory equivalence across independent processes with randomized `PYTHONHASHSEED` (`0`, `42`, `12345`, `99999`).

## Phase 14.2 Remediation checklist

| Requirement | Status |
| --- | --- |
| Unknown queue load representation as `None` outside communication radius ($R=2.0$) | Complete |
| Conservative unknown load penalty and travel margin check in resource utility evaluation | Complete |
| Scenario U06 achieves $\ge 95\%$ (100.0%) completion with 0 exhaustions | Complete |
| Internal crossroad recognition under partial diagonal visibility (`known_diags >= 2`) | Complete |
| Exclusion of boundary and room doorway cells from false crossroad classification | Complete |
| Scenario U14 achieves $\ge 95\%$ (100.0%) completion with 0 exhaustions | Complete |
| 100% completion preserved across all 13 feasible baseline scenarios (Decentralized) | Complete |
| 100% completion preserved across all 14 feasible unseen scenarios (Decentralized) | Complete |
| All 5 infeasible scenarios honestly report 0.0% completion in both modes | Complete |
| Dedicated unit test suite `tests/test_local_horizon_remediation.py` (6 tests) | Complete |
| Full pytest test suite passing (216 passed) | Complete |
| Cross-process determinism verified across randomized `PYTHONHASHSEED` values | Complete |
| Zero changes to default centralized baseline behavior | Complete |

## Phase 14.2 status

Phase 14.2 (Local-Horizon Coordination) implements and benchmarks an experimental decentralized coordination mode where agents share intentions, coordinate right-of-way, and inspect resource load strictly within a local sensory radius of $R = 2.0$ (Euclidean distance $\le 2.0$), while preserving the existing centralized mode as the default baseline:
1. **Coordination Mode & Scoped Perception Architecture:**
   - Added `coordination_mode` (`"centralized"` vs `"decentralized"`, default `"centralized"`) and `communication_radius` (default `2.0`) to `MultiAgentWorldConfig` and `MultiAgentCoordinator`.
   - In decentralized mode, `get_visible_peers(agent_id, pos, current_positions)` scopes visible intentions and position updates strictly to peers within Euclidean distance $\le 2.0$.
   - Scoped obstacle awareness: In decentralized mode, agents plan and resolve conflicts strictly using their own local obstacle memory (`pol.memory.get_known_obstacles()` / `exhausted_positions`), avoiding leakage of the coordinator's global obstacle pool.
   - Preserved authoritative environment physics: physical bounces and collision resolution remain strictly enforced by the simulation engine, ensuring zero illegal double-occupancy or wall clipping when unobserved distant agents encounter one another.
2. **Comprehensive Empirical Benchmark (32 Scenarios):**
   - **Baseline Benchmark (16 Scenarios):** All 13 feasible baseline scenarios achieve **100.0% completion** in decentralized mode, matching centralized mode exactly. All 3 infeasible baseline scenarios (4A, 5A, 8) honestly achieve 0.0% completion with 100% exhaustion in both modes.
   - **Unseen Generalization (16 Scenarios):** 12 of 14 feasible unseen scenarios (U01–U05, U07–U10, U12–U13, U15) achieve **100.0% completion** in decentralized mode. Both infeasible unseen scenarios (U11, U16) honestly achieve 0.0% completion in both modes.
   - **Collision Separation:** Verified zero illegal movements across all 32 scenarios. Actual collisions (`vertex + swap`) are separated from benign blocked attempts (`occupied`).
   - True structural deadlocks remained 0 across all feasible baseline scenarios.
3. **Forensic Causal Analysis of Local-Horizon Regressions (< 95% Completion):**
   - **U06 (8-Agent Staggered Quadrant Mesh) — 0.0% Decentralized vs 100.0% Centralized:**
     - *Causal Mechanism:* **Resource Queue Blindness** (Invisible Distant Contention).
     - *Explanation:* In decentralized mode ($R=2.0$), agents in Quadrant 1 cannot observe Quadrant 4 agents committing to the distant recharge pad at (5, 5) ($\text{dist} = 5.0 > 2.0$). Seeing its local pad at (2, 2) occupied (`load = 1`), `agent_q1_a` falsely infers that (5, 5) is uncontested (`load = 0`). The raw static utility formula awards (5, 5) a higher unpenalized score (110.5 vs 69.0). `agent_q1_a` abandons its safe local doorstep with only 7 energy, ventures 7 steps across the board toward (5, 5), and exhausts mid-grid.
   - **U14 (Multi-Corner Blind Intersection) — 0.0% Decentralized vs 100.0% Centralized:**
     - *Causal Mechanism:* **Hidden Intentions & Stale/Localized Obstacle Beliefs** (Horizon Blindness at Blind Junctions).
     - *Explanation:* At the 1-cell intersection (3, 3), blind corner pillars exist at (2, 2), (4, 2), (2, 4), (4, 4). Centralized mode pooled all 4 pillars globally, allowing `is_crossroad` to recognize the junction and grant clearance right-of-way to the vehicle inside (3, 3). In decentralized mode ($R=2.0$), `agent_e` at (4, 3) cannot observe the opposite western pillars at (2, 2) and (2, 4) ($\text{dist} = \sqrt{5} \approx 2.24 > 2.0$). Because `is_crossroad` requires all 4 diagonal pillars, it evaluates to `False`. Failing to recognize the shared junction, `agent_e` asserts priority rotation against `agent_w`, resulting in opposing head-on bounce and cyclic deadlock at the neck.
4. **Dedicated Test Suite & Verification:**
   - Created `tests/test_local_horizon_coordination.py` (11 unit tests covering local scoping, intention hiding, horizon boundaries, obstacle memory isolation, TTL expiration, swap resolution, and physical bounce integrity).
   - Full test suite: **210 passed in 5.45s** (`python -m pytest`).

## Phase 14.2 checklist

| Requirement | Status |
| --- | --- |
| Explicit `coordination_mode` and `communication_radius` config in `MultiAgentWorldConfig` | Complete |
| Backward-compatible default `centralized` mode preserving 100% legacy behavior | Complete |
| Decentralized peer intention visibility restricted to Euclidean distance $\le 2.0$ | Complete |
| Decentralized local obstacle memory isolation (no pooling of global obstacle beliefs) | Complete |
| Decentralized resource queue/load query scoping | Complete |
| Authoritative simulation physics preventing double-occupancy and wall clipping | Complete |
| Dedicated unit test suite `tests/test_local_horizon_coordination.py` (11 tests) | Complete |
| Full 32-scenario evaluation harness (`phase14_2_local_horizon_eval.py`) | Complete |
| Structured results exported to `phase14_2_benchmark_results.json` | Complete |
| Comprehensive audit report exported to `phase14_2_local_horizon_report.md` | Complete |
| Collision breakdown separating actual collisions from blocked attempts | Complete |
| Forensic diagnosis of U06 (Resource Queue Blindness) and U14 (Horizon Blindness) | Complete |
| Full pytest test suite passing (210 passed) | Complete |

## Phase 14.1 status

Phase 14.1 (Determinism & Telemetry Cleanup) completes the foundational cleanup of the multi-agent coordination layer before decentralized learning, eliminating nondeterministic agent arbitration and enriching deadlock telemetry to differentiate between benign cooperative waiting and true structural deadlocks:
1. **Deterministic Agent Arbitration (`ResourceArbitrator`):**
   - Replaced Python's process-dependent `hash(aid)` tie-breaker under the `need_urgency` rule in `ResourceArbitrator` with the stable, deterministic `agent_val(aid)` mapping.
   - Codebase audit confirms zero built-in `hash()` invocations across the entire `worldsim/` package.
   - Verified cross-process determinism across independent subprocess invocations with randomized `PYTHONHASHSEED` (`0`, `999999`, `random`).
2. **Deadlock Telemetry & Classification Enriched (`DeadlockDetector` & `DeadlockEvent`):**
   - Added `classification` (`"cooperative_waiting"` vs `"structural_deadlock"`) and `waiting_reason` (`"resource_queue"`, `"bottleneck_yielding"`, `"resource_contention"`, `"doorway_contention"`, `"swap_yielding"`, `"unresolvable_block"`, `"reciprocal_swap_lock"`, `"vertex_contention_lock"`) to `DeadlockEvent`.
   - In `DeadlockDetector`, integrated spatial and coordinator awareness to distinguish intentional yielding and resource queueing from circular or unyielding physical blocks.
   - Added query properties: `cooperative_waiting_events`, `structural_deadlock_events`, `cooperative_waiting_count`, `structural_deadlock_count`, `unresolved_structural_deadlocks`, `unresolved_cooperative_waits`, and `waiting_reasons_tally`.
   - Preserved full backwards compatibility with all existing legacy benchmark assertions and counters (`detected_events`, `recovery_attempts`, `successful_recoveries`, `unresolved_deadlocks`).
3. **Empirical Verification on U09 & U14:**
   - Evaluated U09 (Scarce Resource Competition, Seeds 500–519): 100% completion (20/20 episodes, 0 exhaustions). All 167 detected events classified as `cooperative_waiting` (`resource_queue`: 83, `resource_contention`: 84); true `structural_deadlock`: 0.
   - Evaluated U14 (Multi-Corner Blind Junction, Seeds 500–519): 100% completion (20/20 episodes, 0 exhaustions). All 47 detected events classified as `cooperative_waiting` (`doorway_contention`: 40, `resource_queue`: 7); true `structural_deadlock`: 0.
   - Confirms that the apparent "deadlocks" reported during the Phase 13.7 audit in U09 and U14 are benign cooperative waiting and yielding that resolve naturally.
4. **Benchmark & Test Suite Verification:**
   - Full test suite: **199 passed in 5.40s** (`python -m pytest`), including 9 targeted regression tests in `tests/test_determinism_and_telemetry.py`.
   - 16-scenario baseline benchmark (`phase14_1_benchmark_results.json`): 100% bit-for-bit identical to Phase 13.7 baseline results, 0 regressions.

## Phase 14.1 checklist

| Requirement | Status |
| --- | --- |
| Deterministic agent priority mapping in `ResourceArbitrator` (`agent_val`) | Complete |
| Elimination of process-dependent `hash()` across entire `worldsim/` codebase | Complete |
| Subprocess cross-process determinism tests with varying `PYTHONHASHSEED` | Complete |
| Extended `DeadlockEvent` with `classification` and `waiting_reason` | Complete |
| Enhanced `DeadlockDetector` distinguishing cooperative waiting from structural deadlock | Complete |
| Dedicated unit test suite `tests/test_determinism_and_telemetry.py` (9 tests) | Complete |
| Backwards compatibility with legacy deadlock metrics and assertions | Complete |
| Telemetry analysis of U09 and U14 confirming 0 true structural deadlocks | Complete |
| Full 16-scenario baseline benchmark verified without regressions (`phase14_1_benchmark_results.json`) | Complete |
| Full pytest test suite passing (199 passed) | Complete |

## Phase 13.6 status

Phase 13.6 implements targeted, minimal, empirical fixes for generalization failures identified in the Phase 13.5 Unseen Scenario Audit, elevating all targeted scenarios to 100% completion while preserving 100% equivalence on the baseline Phase 13.5 16-scenario benchmark:
1. **Targeted Fix A (U03 — Static Obstacle Safety):**
   - Corrected coordinator candidate action filtering in `MultiAgentCoordinator.suggest_alternative_move` so that cells occupied by stationary, exhausted, or non-traversable agents (`obstacle_positions`, `oint is None`) are rejected from candidate sidesteps.
   - Scenario U03 (Cul-de-Sac Perimeter Bypass) achieves **100.0% overall completion** across seeds 500–519 (0 failed seeds). Dedicated test suite `tests/test_static_obstacle_safety.py` (7 tests pass).
2. **Targeted Fix B (U09 — Energy-Safe Resource Queuing):**
   - In `worldsim/robust_policy.py`, enforced energy safety checks before an agent commits to entering or advancing through an intermediate resource queue cell. Ensures $E \ge \text{cost\_to\_queue} + \text{queue\_wait\_buffer} + \text{cost\_from\_queue\_to\_resource}$.
   - In `worldsim/coordination.py`, added reservation fair release/reassignment when an agent cannot safely sustain its resource claim.
   - Scenario U09 (Scarce Resource Competition) achieves **100.0% overall completion** across seeds 500–519 (0 failed seeds). Dedicated test suite `tests/test_energy_safe_resource_queuing.py` (5 tests pass).
3. **Targeted Fix C (U14 — Energy-Safe Deadlock Detours):**
   - In `worldsim/coordination.py` and `worldsim/multi_agent.py`, introduced shared deterministic `agent_val` mapping for cardinal directions and hashed IDs, eliminating modulo collisions and process-randomized hash discrepancies.
   - Added crossroad exit right-of-way: vehicles clearing an intersection have priority over oncoming vehicles entering the intersection, preventing ping-pong limit cycles in single-cell necks.
   - Prioritized lateral sidesteps over corridor retreats in candidate action ranking (`lateral_bonus`), allowing oncoming traffic to pass without draining spare energy on corridor oscillations.
   - Eliminated unnecessary doorstep reversals in non-swap conflicts (`dist_d > cur_dist`), allowing safe zero-energy waiting at bottlenecks.
   - Scenario U14 (Multi-Corner Blind Junction) improves from **0.0% to 100.0% overall completion** across seeds 500–519 (100% per-agent success, 0 failed seeds). Dedicated test suite `tests/test_energy_safe_deadlock_detours.py` (7 tests pass).
4. **Targeted Fix D (U01 — Double Corridor Bottleneck):**
   - Verified that the foundational coordination architecture (deterministic agent valuation, doorway non-swap retreat prevention at 0 energy cost, lateral detour ranking over backwards reversals, and shared obstacle awareness) cleanly resolves 4-agent bottleneck contention at the central corridor portal.
   - Scenario U01 improves from **0.0% to 100.0% overall completion** across seeds 500–519 (20/20 episodes, 100% per-agent success, 9.5 mean steps, 0 exhaustions, 0 deadlocks, Jain's fairness index = 1.0).
   - Validated extended evaluation across 100 seeds (500–599) with 100.0% completion (100/100).
   - Dedicated test suite `tests/test_double_corridor_bottleneck.py` (9 tests pass).
5. **Comprehensive Unseen Generalization Matrix:**
   - All 14 feasible unseen scenarios (U01–U10, U12–U15) achieve **100.0% overall completion** across seeds 500–519.
   - Both mathematically infeasible scenarios (U11, U16) honestly report **0.0% overall completion**.
6. **Benchmark & Test Suite Verification:**
   - All 190 automated unit tests pass in 3.56s (`python -m pytest`).
   - All 16 baseline Phase 13.5 benchmark scenarios match expected outcomes with zero regressions.

## Phase 13.6 checklist

| Requirement | Status |
| --- | --- |
| Static obstacle safety check in coordinator sidestep selection (U03) | Complete |
| Energy-safe resource queuing & reservation arbitration (U09) | Complete |
| Deterministic agent priority mapping across simulation and coordinator (`agent_val`) | Complete |
| Lateral detour ranking over corridor reversals in swap conflicts | Complete |
| Prevention of doorstep retreat oscillations in non-swap conflicts | Complete |
| Crossroad junction clearing right-of-way | Complete |
| Double corridor bottleneck traversal & doorway yielding (U01) | Complete |
| Dedicated unit test suite `tests/test_double_corridor_bottleneck.py` (9 tests) | Complete |
| Full regression verification across U01, U03, U09, U14 (100.0% on seeds 500–519) | Complete |
| 100% completion across all 14 feasible unseen generalization scenarios | Complete |
| Zero regressions on Phase 13.5 16-scenario benchmark | Complete |

## Phase 13.5 status

Fixes for Reciprocal Target Lock and Sensory-Horizon Oscillation is complete and empirically validated. Phase 13.5 eliminates the two remaining Category 3 policy/coordination failure modes identified in Phase 13.4, elevating feasible benchmark scenarios to 100% completion while strictly preserving partial observability:
1. **Reciprocal Target Lock Resolution:**
   - In `worldsim/coordination.py`, `MultiAgentCoordinator` tracks `previous_intentions` across ticks, detects head-on swap conflicts, and bypasses destination locking when `is_swap_conflict` is True.
   - Evaluates priority rotation during swap conflicts; the higher-priority agent proceeds to target while the lower-priority agent sidesteps orthogonally into an adjacent cell.
   - Exempts completed agents (`status == 'reached_target'`) from blocking alternative moves, allowing active agents to sidestep into yielding goal cells.
   - Resolves Failure A in Scenario 4B: overall completion improves from **0.0% to 100.0%** (26.0 mean steps, 0 swap collisions).
2. **Sensory-Horizon Obstacle Memory & Fallback:**
   - In `worldsim/robust_policy.py`, `AgentMemory` maintains `remembered_obstacles` populated strictly from local observations.
   - Visual revalidation purges cleared obstacles upon direct re-observation within sensory range; bounded TTL (15 steps) expires stale obstacles if unobserved.
   - Energy-bounded BFS fallback: when an avoidance detour around completed agents exceeds remaining energy (`len(path) > energy`), planner falls back to routing through yielding completed agents rather than exhausting.
   - Resolves Failure B in Scenario 5B: overall completion improves from **50.0% to 100.0%** (32.5 mean steps, 0 ping-pong oscillations).
3. **Benchmark Verification Matrix (`benchmark_results_phase13_5.json`):**
   - All 13 feasible benchmark scenarios (1A, 1B, 1C, 1D, 2, 3A, 3B, 3C, 4B, 5B, 6, 7, 9) achieve **100.0% overall completion**.
   - All 3 infeasible configurations (4A, 5A, 8) honestly report 0.0% completion.
   - 100-episode extended evaluation maintains 100/100 completed episodes.
4. **Test Suite:** Dedicated unit test suite `tests/test_multi_agent_phase13_5.py` (8 tests). All 162 unit tests pass in 2.38s. Do not start Phase 14 without an agreed milestone.

## Phase 13.5 checklist

| Requirement | Status |
| --- | --- |
| Cross-tick intention memory (`previous_intentions`) in `MultiAgentCoordinator` | Complete |
| Reciprocal swap conflict detection in `suggest_alternative_move` | Complete |
| Priority-based orthogonal sidestep during swap deadlocks | Complete |
| Exemption of completed agents in coordinator conflict checks | Complete |
| Agent-owned local obstacle memory (`remembered_obstacles`) in `AgentMemory` | Complete |
| Visual revalidation upon re-entering sensory radius | Complete |
| Bounded TTL expiration (15 steps) for unobserved obstacles | Complete |
| Energy-bounded BFS fallback (`len(path) > energy`) in `RobustResourcePolicy` | Complete |
| Comprehensive benchmark runner (`worldsim/multi_agent_experiment_phase13_5.py`) | Complete |
| CLI execution script (`run_phase13_5_benchmark.py`) | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13_5.json`) | Complete |
| Dedicated unit test suite (`tests/test_multi_agent_phase13_5.py`, 162 tests passed) | Complete |
| Zero regressions across single-agent and prior multi-agent tests | Complete |

## Phase 13.4 status

Independent Validation & Robustness Audit is complete and empirically validated. Phase 13.4 confirms benchmark integrity, clarifies contention vs persistence accounting, evaluates 100-episode stability across unselected seeds, formally categorizes layout failures, and stress-tests resource fairness:
1. **Contention vs Deadlock Metric Resolution:** Proven that 520 collisions in Scenario 1 reflect transient, single-tick queue and doorstep contentions. Priority rotation resolves right-of-way on every tick, ensuring agents advance before persistent blocking can cause terminal failure. All 260 detected deadlocks were successfully recovered (`successful_recoveries = 260`, `unresolved_deadlocks = 0`). Starvation events (290) reflect temporary queue delays ($\ge 3$ blocked ticks); zero agents exhausted.
2. **100-Episode Extended Evaluation (Seeds 200–299):** Evaluated 100 consecutive episodes on the 8-agent bottleneck, achieving **100/100 (100.0%) overall completion**, 100.0% per-agent success, 28.50 mean steps (range 27–30 steps), 2,600 total collisions, 1,300 detected and resolved deadlocks (0 unresolved), and 0 exhaustions.
3. **Diverse Layout Generalization:** Evaluated permuted target assignments (100.0% overall completion) and opposing crossing walls (100.0% overall completion), confirming stability across varied spatial configurations.
4. **Formal Layout Failure Categorization:**
   - *Scenario 4A:* **Category 1 (Geometrically & Energetically Infeasible)**: Manhattan distance $d=7 > E=6$. Exhausts at step 6 under every possible policy.
   - *Scenario 4B:* **Category 3 (Policy / Coordination Breakdown)**: Active head-on swap lock at row $y=7$ between Agent 3 at $(4, 7)$ (target $(5, 7)$) and Agent 4 at $(5, 7)$ (target $(4, 7)$). Both are active and standing on each other's target cells; neither policy voluntarily yields away from its target cell.
   - *Scenario 5A:* **Category 1 & 2 (Infeasible & Undiscoverable)**: Distance $d=7 > E=6$ (Category 1); Euclidean distance to central resources is $5.0 > 2.0$ (sensory radius), making them undiscoverable (Category 2).
   - *Scenario 5B:* **Category 3 (Policy / Coordination Breakdown)**: Sensory-horizon limit cycle: Agent 1 ping-pongs between $(4, 7)$ and $(4, 6)$ because resting completed Agent 5 at $(6, 7)$ is visible from $(4, 7)$ ($d=2.0 \le 2.0$), triggering an upward detour, but falls outside the sensory horizon from $(4, 6)$ ($d \approx 2.24 > 2.0$), resetting memory and steering Agent 1 back DOWN to $(4, 7)$.
5. **Resource Fairness Stress Suite:** Fair-Balanced responds to urgency under asymmetric energy (e=1 vs e=5), rotates equitably across 3 claimants (Jain index = 1.000), and allocates to late-arriving urgent agents.
6. **Benchmark Suite & Tests:** Exported to `benchmark_results_phase13_4.json`. Dedicated unit test suite `tests/test_multi_agent_phase13_4.py`. All 154 unit tests pass in 1.66s. Do not start Phase 14 without an agreed milestone.

## Phase 13.4 checklist

| Requirement | Status |
| --- | --- |
| Independent verification and reproduction of Phase 13.3 benchmark results | Complete |
| Metric integrity audit: collision breakdown and transient vs persistence deadlock resolution | Complete |
| 100-episode extended evaluation across seeds 200–299 (100% completion confirmed) | Complete |
| Diverse layout generalization (permuted targets, opposing crossing walls) | Complete |
| Formal categorization of remaining layout failures (Scenarios 4A, 4B, 5A, 5B) | Complete |
| Diagnostic trace of policy/coordination breakdowns (head-on swap lock and sensory-horizon limit cycle) | Complete |
| Resource fairness stress testing: asymmetric urgency, 3-claimant rotation, staggered arrivals | Complete |
| Comprehensive audit benchmark runner (`worldsim/multi_agent_experiment_phase13_4.py`) | Complete |
| CLI execution script (`run_phase13_4_benchmark.py`) | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13_4.json`) | Complete |
| Dedicated unit test suite (`tests/test_multi_agent_phase13_4.py`, 154 tests passed) | Complete |
| Zero regressions across single-agent and prior multi-agent tests | Complete |

## Phase 13.3 status

Reliable Completion, Fairness & Benchmark Validation is complete and empirically validated. Phase 13.3 resolves the critical task completion bottleneck, eliminates permanent resource starvation, and establishes rigorous benchmark metric separation:
1. **Goal-Cell Yielding & Corridor Traversal:** `MultiAgentSimulation._resolve_conflicts` exempts completed agents (`status == 'reached_target'`) from triggering occupied stationary collisions. Inactive agents on their goals yield right-of-way, allowing active agents to traverse corridors and terminal cells without displacement.
2. **BFS Corridor Routing Fallback:** `RobustResourcePolicy` attempts strict obstacle avoidance, falling back to routing through yielding completed agents when narrow corridors or cul-de-sacs lack alternative paths.
3. **Deterministic Fair Rotation Tie-Breaking:** `ResourceArbitrator` under `fair_balanced` rotates tie-breaking deterministically modulo `len(claimants)` across ticks and seeds, guaranteeing 50/50 equitable access across episodes on consumable resources and 100% mutual completion on reusable resources.
4. **Metric Separation:** Every benchmark explicitly separates episode-level overall completion rate (100% of agents complete) from per-agent success rates and mean per-agent success.
5. **9-Scenario Benchmark Suite:** All 9 scenarios evaluated across recorded seeds and exported to `benchmark_results_phase13_3.json`:
   - 8-agent bottleneck improves from 0.0% to **100.0% overall completion** (20/20 episodes completed, 100% per-agent success).
   - 4-agent intersection maintains **100.0% overall completion**.
   - 2-agent resource competition proves starvation reduction (50/50 consumable, 100% reusable).
   - Terminal corridor stress test achieves **100.0% overall completion**.
   - All 146 unit tests pass in 0.77s. Do not start Phase 14 without an agreed milestone.

## Phase 13.3 checklist

| Requirement | Status |
| --- | --- |
| Goal-cell yielding in `MultiAgentSimulation._resolve_conflicts` | Complete |
| BFS corridor routing fallback through yielding completed agents in `RobustResourcePolicy` | Complete |
| Deterministic round-robin tie-breaking for `fair_balanced` in `ResourceArbitrator` | Complete |
| Optional persistent arbitrator support in `MultiAgentSimulation` | Complete |
| Metric separation: overall episode completion rate vs per-agent success rate | Complete |
| Comprehensive 9-scenario benchmark suite (`worldsim/multi_agent_experiment_phase13_3.py`) | Complete |
| Benchmark execution script (`run_phase13_3_benchmark.py`) | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13_3.json`) | Complete |
| Dedicated unit test suite (`tests/test_multi_agent_phase13_3.py`, 146 tests passed) | Complete |
| Zero regressions across single-agent and prior multi-agent tests | Complete |

## Phase 13.2 status

Resolve the Eight-Agent Bottleneck is complete and empirically validated. Phase 13.2 solves the multi-agent scaling bottleneck, queue exhaustion, and idle resource failures identified in Phase 13 and Phase 13.1:
1. **Multi-Recharge Resource Modeling:** `ResourceUtilityEvaluator` computes cumulative reachability for reusable resources (`energy_post = max_energy`), preventing false "insufficient" classifications that caused agents to wander and exhaust.
2. **Doorstep Queuing & Collision Arbitration:** `MultiAgentCoordinator.suggest_alternative_move` allows agents targeting resources or target waypoints to wait cost-free at the doorstep without being forced into energy-wasting detours.
3. **Energy-Aware Alternative Suggestions:** Coordinator validates `dist(d, target) <= energy - 1` before suggesting any alternative move, preventing fatal energy starvation.
4. **Persistent Resource Commitments & Fair Load Balancing:** Per-tick intentions reset cleanly while commitments persist across ticks, ensuring balanced utilization of all active resources without herds.
5. **Inactive Obstacle Routing:** Stationary completed agents are recognized as static obstacles; BFS path guidance navigates active agents safely around them.
6. **Benchmark & Test Suite:** All 6 controlled scenarios evaluated and exported to `benchmark_results_phase13_2.json`. Collisions in the 8-agent bottleneck dropped from 4,760 to 721 (85% reduction), swap collisions cut by 98% (880 down to 18), 4-agent intersection 100% completion maintained, and all 138 unit tests pass in 0.74s. Do not start Phase 14 without an agreed milestone.

## Phase 13.2 checklist

| Requirement | Status |
| --- | --- |
| Multi-recharge utility modeling for reusable resources in `ResourceUtilityEvaluator` | Complete |
| Coordinator doorstep queuing without fatal redirects away from resources/targets | Complete |
| Energy-aware alternative move filtering (`dist <= energy - 1`) | Complete |
| Persistent resource commitment tracking and balanced load arbitration | Complete |
| Stationary inactive agent obstacle detection and look-ahead BFS path guidance | Complete |
| Detailed diagnostic metrics (energy histories, time blocked, collision breakdown, unused resources) | Complete |
| Controlled 6-scenario benchmark suite (`worldsim/multi_agent_experiment_phase13_2.py`) | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13_2.json`) | Complete |
| Comprehensive test suite (`tests/test_multi_agent_phase13_2.py`, 138 tests passed) | Complete |

## Phase 13.1 status

Multi-agent coordination, resource allocation, and congestion control is complete and empirically validated. Phase 13.1 resolves the congestion bottlenecks and resource starvation failure modes identified in Phase 13:
1. **Deterministic Priority Rotation & Blocked Destination Memory:** `collision_rule='priority_rotation'` dynamically rotates vertex movement priority based on simulation tick, eliminating symmetric vertex bouncing. `RobustResourcePolicy` tracks blocked destination history in `AgentMemory`, temporarily penalizing repeatedly blocked cells unless they are the agent's target or an active resource.
2. **Fair Shared-Resource Allocation:** `ResourceArbitrator` supports `fair_balanced`, `need_urgency`, `priority_rotation`, and `starvation_prevention` rules, eliminating permanent lowest-ID starvation and arbitrating contested entries to shared resource cells.
3. **Deadlock Detection & Bounded Recovery:** `DeadlockDetector` identifies repeated swaps, occupied blocking, and vertex deadlocks using observable simulation events, tracking recovery attempts, successful recoveries, and unresolved deadlocks.
4. **Multi-Agent Coordination Layer:** `MultiAgentCoordinator` enables voluntary intention registration, alternative move suggestion, and resource load-balancing without violating observation boundaries or leaking private memory.
5. **Fairness & Starvation Monitoring:** `FairnessMonitor` logs per-agent steps, energy consumed, resource claims won/lost, energy gained, ticks blocked, and starvation events, computing Jain's Fairness Index.
6. **Benchmark & Test Suite:** Experiments A through J and 4 ablations executed and exported to `benchmark_results_phase13_1.json`. 4-agent intersection congestion is 100% resolved (Exp C: 0.0% -> 100.0%, cols 3,300 -> 0). 8-agent scaling bottleneck unlocked (5 of 8 agents reaching target). All 133 tests pass in 0.75s. Do not start Phase 14 without an agreed milestone.

## Phase 13.1 checklist

| Requirement | Status |
| --- | --- |
| Congestion-aware movement with deterministic priority rotation (`priority_rotation`) | Complete |
| Blocked destination memory and avoidance in `RobustResourcePolicy` | Complete |
| Fair, energy-aware resource arbitrator (`ResourceArbitrator`) with 5 rules | Complete |
| Bounded deadlock detector (`DeadlockDetector`) tracking swaps, blocking, vertex locks | Complete |
| Voluntary coordination broker (`MultiAgentCoordinator`) with intent de-confliction | Complete |
| Resource load-balancing across available sufficient resources | Complete |
| Fairness and starvation monitoring (`FairnessMonitor`) with Jain's index | Complete |
| 10 controlled experiments (Exp A through Exp J) evaluated and benchmarked | Complete |
| 4 ablation studies on 8-agent scaling scenario | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13_1.json`) | Complete |
| Viewer enhancements with deadlock outlines, blocked badges, and recovery HUD | Complete |
| Comprehensive test suite (`tests/test_multi_agent_coordination.py`, 133 tests passed) | Complete |

## Phase 13 status

Multi-agent simulation is complete and empirically validated. Phase 13 extends WorldSim from a single-agent framework to a multi-agent grid simulation where multiple autonomous entities operate simultaneously in the same environment:
1. **Simultaneous 5-Phase Tick Model:** Each tick freezes observations across all agents, formulates candidate action intents, resolves spatial conflicts deterministically via iterative stability, executes movements atomically, processes resource interactions with strict conservation (preventing duplicate energy extraction), and updates agent lifecycles (`active`, `reached_target`, `exhausted`).
2. **Deterministic Conflict Resolution:**
   - **Vertex collisions:** Two or more agents attempting to enter the same cell in the same tick are symmetrically bounced back to their respective origin coordinates (`collision_rule='bounce'`).
   - **Swap collisions:** Two agents attempting to swap positions across an edge are both bounced back to their origin coordinates.
   - **Occupied collisions:** An agent attempting to enter a cell occupied by a stationary agent is blocked and remains in place.
   - **Following ("train") movement:** An agent attempting to enter a cell currently vacated by another agent is allowed once the leading agent's movement is confirmed conflict-free.
3. **Atomic Shared Resource Depletion:** When a consumable resource is recharged, it is atomically depleted. If multiple agents attempt to recharge at the same resource simultaneously, exactly one agent acquires the energy while contentions are tracked and energy duplication is strictly prevented.
4. **Observation Isolation & Cooperative Sharing:** Each agent perceives the world strictly through its own bounded `MultiAgentObservation` ($r=2.0$). An opt-in cooperative discovery pool (`cooperative_sharing=True`) aggregates observed resources into `shared_discoveries`, enabling communication without violating individual observation boundaries.
5. **Heterogeneous Policy Assignment:** Any single-agent policy (`RobustResourcePolicy`, `LinearQLearner`, `QLearner`, `RandomPolicy`, or planners) can be assigned to individual agents.
6. **Multi-Agent Evaluation & Benchmark:** All 8 controlled experiments (Exp A through Exp H) pass and are exported to `benchmark_results_phase13.json`. Single-agent regression is 100% verified (Exp A). Scalability up to 8 simultaneous agents on an 8x8 grid demonstrates congestion and collision scaling.
7. **Viewer & Backward Compatibility:** `MultiAgentWorldViewer` provides interactive multi-agent visualization with distinct color palettes, per-agent energy tags, and status displays. All 116 automated tests pass in 0.75s, maintaining 100% backward compatibility with Phases 1–12.2. Do not start Phase 14 without an agreed milestone.

## Phase 13 checklist

| Requirement | Status |
| --- | --- |
| Stable agent identity and multi-agent world state (`AgentConfig`, `AgentState`, `MultiAgentWorldConfig`, `MultiAgentWorldState`) | Complete |
| Simultaneous 5-phase tick cycle with validation and lifecycle tracking | Complete |
| Deterministic conflict resolution (vertex bounce, swap bounce, occupied blocking, train movement) | Complete |
| Atomic shared resource consumption preventing duplicate energy extraction | Complete |
| Isolated per-agent observations with visible agent masking (`MultiAgentObservation`) | Complete |
| Opt-in cooperative discovery sharing (`cooperative_sharing=True/False`) | Complete |
| Flexible goal criteria (`individual`, `shared_any`, `shared_all`) | Complete |
| Per-agent policy assignment and heterogeneous policy support | Complete |
| Multi-agent metric calculation (success, steps, energy, collisions, contentions, Jain's fairness index) | Complete |
| 8 Controlled Experiments (Exp A to Exp H) executed and benchmarked | Complete |
| Machine-readable JSON benchmark export (`benchmark_results_phase13.json`) | Complete |
| Multi-agent GUI visualization (`MultiAgentWorldViewer`) | Complete |
| Dedicated multi-agent unit test suite (`tests/test_multi_agent.py`, 116 total tests passed) | Complete |

## Phase 12.2 status

Robust resource-aware learning is complete and validated. Phase 12.2 introduces four modular, observation-limited improvements (`RobustResourcePolicy`, `AgentMemory`, `RechargeEvaluator`, `ResourceUtilityEvaluator`, `ExplorationController`, `CycleDetector`):
1. Intelligent recharge control preventing repeated recharges at reusable resources or when energy is already sufficient for the target.
2. Utility-based resource selection calculating net energy margin and route feasibility, resolving the nearest-insufficient vs farther-sufficient counterexample.
3. Safe exploration beyond sensory range calculating unobserved grid frontiers and expanding coverage safely without hidden state access.
4. Failure recovery and cycle detection tracking action history, detecting loops/oscillations, and steering the agent toward productive alternatives.

Across a 15-scenario benchmark suite, Phase 12.2 Robust Linear Q achieves 100.0% success across all 12 feasible environments (improving from 75.0% for Phase 12.1 linear and 8.3% for tabular Q). Reusable-resource looping (Scenario 5) and sensory-horizon blindness (Scenario 6) are completely resolved (0.0% -> 100.0%). All 105 tests pass. Results are saved in `benchmark_results_phase12_2.json` and reproducible via `python run_phase12_2_benchmark.py`. Do not start Phase 13 without a new agreed milestone.

## Architecture

`WorldConfig` validates dimensions, initial placement, sensory radius, energy parameters (`initial_energy`, `max_energy`, `movement_energy_cost`), explicit `resources`, and optional `observable_target`. `Simulation` owns deterministic state, clock, pause state, energy tracking, and experience history. `Observation` exposes permitted entity coordinates, visible cells, current energy, observable resources within sensory radius, and optional observable target. `EnergyPlanningProblem` defines privileged known-model planning inputs for `plan_energy`. `execute_energy_plan` safely drives simulations through `Simulation.step`. `EnergyPlannerPolicy` provides observation-limited planner-backed decisions. `QLearner` and `LearningRunner` provide tabular reinforcement learning with multi-task `train_tasks` capability. `LinearQLearner`, `FeatureExtractor`, and `LinearLearningRunner` provide feature-based linear function approximation Q-learning with action feasibility masking, exploration decay, and task distribution training. `RobustResourcePolicy`, `AgentMemory`, `RechargeEvaluator`, `ResourceUtilityEvaluator`, `ExplorationController`, and `CycleDetector` provide modular robustness, intelligent recharge control, utility-based resource selection, safe exploration, and cycle recovery. `worldsim.benchmark` and `worldsim.benchmark_phase12_2` provide machine-readable benchmark suites and verified failure accounting.




## Phase 1 status

| Requirement | Status |
| --- | --- |
| Minimal configurable 2D grid | Complete |
| One uniquely identified, positioned entity | Complete |
| Headless clock/update loop | Complete |
| Basic grid visualization | Complete |
| Pause/resume control | Complete |
| Core automated tests | Complete |
| Setup/execution documentation | Complete |

## Phase 2 status

| Requirement | Status |
| --- | --- |
| Configurable Euclidean sensory radius | Complete |
| Bounded immutable observation interface | Complete |
| Perception visualization matching observation | Complete |
| Perception boundary/determinism/immutability tests | Complete |

## Phase 3 status

| Requirement | Status |
| --- | --- |
| Explicit four-direction action enum | Complete |
| Atomic validated action transitions | Complete |
| Boundary, invalid-action, and paused behavior | Complete |
| Fresh observations after movement | Complete |
| Manual buttons and arrow-key controls | Complete |
| Phase 1-3 automated tests | Complete |

## Phase 4 status

| Requirement | Status |
| --- | --- |
| Immutable `Experience` schema | Complete |
| Recording for successful and valid rejected actions | Complete |
| Read-only chronological in-memory history | Complete |
| History, nested-record, and prior-phase tests | Complete |
| UI unchanged | Complete |

## Phase 5 status

| Requirement | Status |
| --- | --- |
| Headless action-sequence experiment runner | Complete |
| Immutable experiment result and snapshots | Complete |
| Ordered results and experience collection | Complete |
| Reproducibility and invalid-action-stop tests | Complete |
| Existing UI unchanged | Complete |

## Phase 6 status

| Requirement | Status |
| --- | --- |
| Explicit target/reward learning environment | Complete |
| Zero-initialized tabular Q-learning | Complete |
| Seeded epsilon-greedy training and fresh episode resets | Complete |
| Non-mutating learned/random evaluation | Complete |
| Q-update, terminal, reproducibility, and prior-phase tests | Complete |

## Phase 7 status

| Requirement | Status |
| --- | --- |
| Bounded immutable FIFO transition memory | Complete |
| Permitted observed-coordinate matching | Complete |
| Opt-in integration in `LearningRunner` separate from Q-table | Complete |
| Persistence across training resets and non-mutating evaluation | Complete |
| Memory and separation automated tests | Complete |

## Phase 8 status

| Requirement | Status |
| --- | --- |
| Standalone BFS planner with explicit planning inputs | Complete |
| Plan execution through public simulation API | Complete |
| Boundary, invalid-dimension, and out-of-bounds validation | Complete |
| Execution consistency with simulation initial position | Complete |
| Comparative experiments across multiple targets (BFS vs Q-learning vs Random) | Complete |
| Automated tests and documentation | Complete |

## Phase 9 status

| Requirement | Status |
| --- | --- |
| Configurable energy system (`initial_energy`, `max_energy`, `movement_energy_cost`) | Complete |
| Explicit resource locations (`Resource`) and recharge mechanics | Complete |
| Extended action system (`Action.RECHARGE`) with atomic validation | Complete |
| Energy-aware decision policy (`EnergyAwarePolicy`) using permitted observations | Complete |
| Multi-step experiment execution with `stop_on_rejected` option | Complete |
| Integration with observation, experiences, planning, and viewer | Complete |
| Automated tests (53 total passed) and empirical validation | Complete |

## Phase 10 status

| Requirement | Status |
| --- | --- |
| State-space energy-aware planner (`plan_energy`, `EnergyPlanningProblem`) | Complete |
| Joint `(position, energy, active_resources)` graph search | Complete |
| Safe plan execution and start-state consistency validation (`execute_energy_plan`) | Complete |
| Observation-limited planner-backed policy (`EnergyPlannerPolicy`) with replanning | Complete |
| Opt-in `include_energy` support on `LearningConfig` / `QLearner` | Complete |
| 6 controlled benchmark scenarios comparing all 4 policies | Complete |
| Automated tests (64 total passed) and empirical validation | Complete |

## Phase 11 status

| Requirement | Status |
| --- | --- |
| Observable target interface (`WorldConfig.observable_target`, `Observation.target`) | Complete |
| Extended `QLearner` state representation (energy, observable target, active resources) | Complete |
| Extended action space (`Action.RECHARGE`) and failed-action penalties in RL | Complete |
| Exploration decay and terminal energy exhaustion semantics | Complete |
| Controlled benchmark comparing Q-learning, planners, heuristics, and random | Complete |
| Automated tests (71 total passed) and empirical validation | Complete |

## Phase 12 status

| Requirement | Status |
| --- | --- |
| Feature extractor on permitted observations and actions (`FeatureExtractor`) | Complete |
| Linear function-approximation Q-learner (`LinearQLearner`) | Complete |
| Action feasibility checks and action masking | Complete |
| Multi-task distribution training runner with task interleaving (`LinearLearningRunner`) | Complete |
| Controlled generalization benchmarks A–F across targets, starts, resources, and shifts | Complete |
| Automated tests (83 total passed) and empirical validation | Complete |

## Phase 12.1 status

| Requirement | Status |
| --- | --- |
| Independent audit of Phase 12 code and benchmark metrics | Complete |
| Corrected mutually exclusive exhaustion vs timeout metric accounting | Complete |
| Matched single-task training comparison across budgets (300, 900, 1500) | Complete |
| Matched multi-task training comparison across budgets (300, 900, 1500) | Complete |
| 10-Task Final Held-Out Benchmark Suite (8 feasible, 2 provably infeasible) | Complete |
| Ground-truth classical planner feasibility verifier | Complete |
| Regression test suite (89 total tests passed) and documentation | Complete |
| Standalone reproducible benchmark runner generating JSON results | Complete |

## Phase 12.2 status

| Requirement | Status |
| --- | --- |
| Intelligent recharge control (`RechargeEvaluator`, reusable loop fix) | Complete |
| Utility-based resource selection (`ResourceUtilityEvaluator`, counterexample fix) | Complete |
| Safe exploration beyond sensory range (`ExplorationController`, `AgentMemory`) | Complete |
| Failure recovery and cycle detection (`CycleDetector`) | Complete |
| Integrated robust policy wrapper (`RobustResourcePolicy`) | Complete |
| Dedicated test suite (`tests/test_robust_learning.py`, 105 total tests passed) | Complete |
| 15-scenario comprehensive benchmark suite (`run_phase12_2_benchmark.py`) | Complete |
| JSON benchmark export (`benchmark_results_phase12_2.json`) | Complete |

## Decisions

- Python standard library only: it provides a portable engine/test baseline and Tkinter visualization without third-party installs.
- `Action.UP`, `DOWN`, `LEFT`, and `RIGHT` are the only Phase 3 actions. They move exactly one cell in zero-based coordinates (`x` rightward, `y` downward).
- State objects are immutable so engine behavior is directly reproducible from identical configuration and action sequences.
- Perception uses `dx^2 + dy^2 <= sensory_radius^2`; cells are generated in row-major order only within world bounds and returned as an immutable tuple.
- `step` validates enum values before mutation. Successful moves advance exactly one tick; boundary moves and all moves while paused are rejected without changing state or tick.
- Every valid `step` attempt appends one immutable `Experience`; malformed actions raise before recording. `experiences` returns a tuple, and all nested domain data are immutable. History is deliberately memory-only and creates no learned behavior.
- `run_experiment` creates a fresh simulation for each run and calls only `step`. The current world is deterministic and has no RNG, so no seed is needed. Invalid actions stop an experiment at their index and are reported without swallowing or executing subsequent actions.
- Phase 6 randomness is isolated to a seeded learner policy. State is `(observed_x, observed_y)` only; target/reward live in explicit experiment configuration. Target transitions terminate with target reward; all nonterminal attempts receive step cost. Evaluation never calls Q-value update.
- Phase 7 persistent memory (`ExperienceMemory`) records immutable `MemoryEntry` tuples in FIFO order. Transition retrieval indexes only permitted entity coordinates `(x, y)` without inspecting private world state. Memory is decoupled from Q-learning equations and evaluation does not modify it.
- Phase 8 BFS planner operates purely on explicit planning inputs (`PlanningProblem`), remaining decoupled from the simulation engine. `execute_plan` passes actions strictly through `Simulation.step`. Privileged target information provided to the planner is explicitly distinguished from the target-unaware entity state representation used in Q-learning.
- Phase 9 energy consumes `movement_energy_cost` on successful moves; moves with insufficient energy or into boundaries are rejected with no clock or energy changes. Recharging is atomic on resource cells, capping energy at `max_energy` and consuming consumable resources. `EnergyAwarePolicy` uses only permitted `Observation` fields (`energy`, `observed_resources`) to plan detours to resources when energy is insufficient to reach the goal directly.
- Phase 10 energy planning uses BFS across `(position, energy, active_resources)` state keys. Privileged known-model planning (`plan_energy`) is treated as an optimal theoretical baseline; observation-limited agents (`EnergyPlannerPolicy`) construct plans strictly using discovered resources and current energy, replanning whenever observations reveal new or consumed resources. Tabular Q-learning retains backward-compatible `include_energy=False` default, leaving comprehensive energy-aware reinforcement learning as future work.
- Phase 11 reinforcement learning operates purely over permitted `Observation` properties. `QLearner` constructs state keys `(x, y, energy, target, active_resources)` without accessing internal simulation state. Failed action attempts (such as moving into boundaries, moving without energy, or recharging away from a resource) receive configurable penalties (`failed_action_penalty = -0.05`), steering the agent toward feasible, valid actions. Exploration decays smoothly (`epsilon_decay = 0.997`) down to `epsilon_min`. Early termination on permanent energy exhaustion prevents futile exploration.
- Phase 12 linear function approximation evaluates candidate actions via relative, normalized features ($\Delta x, \Delta y$, relative distance, energy deficit, resource proximity, feasibility). Tabular Q-learning fails to generalize because it hashes absolute coordinates, whereas linear Q-learning with relative features transfers zero-shot to unseen targets, starts, and resource layouts. In multi-task training, round-robin interleaving of tasks is required to prevent catastrophic forgetting and break collinearity between target and resource directions.
- Phase 12.1 benchmark audit enforces that benchmark results must be generated directly from running code without hardcoding. Metric accounting must be mutually exclusive ($\text{Success} + \text{Exhaustion} + \text{Timeout} = 100\%$). Representation limitations (reusable resource looping, sensory horizon blindness, nearest-resource aliasing) are documented as genuine boundaries of the linear model rather than obscured.
- Phase 13 multi-agent simulation uses a 5-phase simultaneous tick model with deterministic conflict resolution (symmetric vertex bounce, swap bounce, occupied blocking, train movement). Shared consumable resources are atomically depleted to prevent duplicate energy extraction. Agents perceive bounded local observations; cooperative discovery sharing is opt-in and does not leak private agent coordinates. Multi-agent evaluation tracks spatial contention, collision counts, individual and collective completion rates, and Jain's fairness index.

## Remaining work

- Manually verify controls and the multi-agent perception tint on a desktop Python installation if this workspace lacks a display.
- Do not start Phase 14 without a new agreed milestone.
