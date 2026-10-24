# WorldSim

## Phase 13.5 fixes for reciprocal target lock and sensory-horizon oscillation

Phase 13.5 resolves the two remaining Category 3 policy/coordination failure modes identified in Phase 13.4, elevating feasible benchmark scenarios to 100% overall completion without violating partial observability:

1. **Root Cause Analysis & Deterministic Reproduction:**
   - **Failure A — Reciprocal Target Lock:** In Scenario 4B (Relocated Resources, energy=7), Agent 3 at $(4, 7)$ targeted $(5, 7)$ and Agent 4 at $(5, 7)$ targeted $(4, 7)$. In `worldsim/coordination.py`, `suggest_alternative_move` unconditionally returned `None` when `pref_dest == target` or `pref_dest in resource_commitments`, preventing either agent from yielding during swap conflicts. Furthermore, completed agents were not exempted in alternative action conflict checks, blocking active agents from sidestepping into yielding completed cells.
   - **Failure B — Sensory-Horizon Oscillation:** In Scenario 5B (Asymmetric Starts, energy=8), an active agent at $(4, 7)$ targeting $(7, 7)$ detoured UP to $(4, 6)$ around a resting completed agent at $(6, 7)$. At $(4, 6)$, $(6, 7)$ was at Euclidean distance $d = \sqrt{2^2 + 1^2} \approx 2.24 > 2.0$, outside the sensory radius. Forgetting the obstacle, the agent turned back DOWN to $(4, 7)$, falling into an endless ping-pong limit cycle until timeout.

2. **Architectural Improvements:**
   - **Cross-Tick Intention Memory & Swap De-confliction (`worldsim/coordination.py`):**
     - Tracks `previous_intentions` across ticks to detect reciprocal swap intentions even before both agents have registered on the current tick.
     - Evaluates priority rotation during swap conflicts; the higher-priority agent proceeds directly to its target while the lower-priority agent sidesteps orthogonally.
     - Accepts `completed_positions` and exempts completed agents (`status == 'reached_target'`) from blocking candidate alternative actions.
     - Bypasses target and commitment retention when `is_swap_conflict` is True, allowing sidestepping away from target cells during swap deadlocks.
   - **Agent-Owned Local Obstacle Memory with Visual Revalidation (`worldsim/robust_policy.py`):**
     - `AgentMemory` maintains `remembered_obstacles: dict[Position, tuple[str, int]]` populated strictly from local observations.
     - **Visual Revalidation:** When an obstacle cell is re-observed within sensory range and found unoccupied, it is immediately purged from memory.
     - **Bounded TTL:** Stale obstacle memory expires after `obstacle_memory_ttl = 15` steps if never re-observed.
     - **Energy-Bounded BFS Fallback:** If an avoidance detour around completed agents would take more steps than remaining energy (`len(path) > energy`), the planner falls back to routing through yielding completed agents rather than starving on an impossible detour.

3. **Benchmark Verification Matrix (`benchmark_results_phase13_5.json`):**
   - **Scenario 4B (Relocated Resources, Energy=7):** Improves from **0.0% to 100.0% overall completion** (100% per-agent success, 26.0 mean steps, 0 swap deadlocks).
   - **Scenario 5B (Asymmetric Starts, Energy=8):** Improves from **50.0% to 100.0% overall completion** (100% per-agent success, 32.5 mean steps, 0 ping-pong oscillations).
   - **Scenarios 1A, 1B, 1C, 1D (8-Agent Bottleneck & Layouts):** Maintain **100.0% overall completion** (100-episode trial achieves 100/100 completed).
   - **Scenario 2 (4-Agent Intersection Regression):** Maintains **100.0% overall completion** (8.0 mean steps).
   - **Scenarios 3A, 3B, 3C (Resource Competition & Contention):** Maintain 100.0% success and fair allocation (Jain index = 1.000).
   - **Scenarios 6 & 7 (Goal-Cell Yielding & Swap Resolution):** Maintain **100.0% overall completion**.
   - **Scenarios 4A, 5A, 8 (Infeasible Environments):** Honestly report **0.0% completion** and 100% exhaustion without false claims.
   - **Scenario 9 (Single-Agent Baseline Regression):** Maintains **100.0% overall completion** (11.0 mean steps).

4. **Test Suite:**
   - Dedicated unit test suite `tests/test_multi_agent_phase13_5.py` (8 test cases).
   - All 162 unit tests pass in 2.38s (`python -m unittest discover -s tests -q`).
   - Zero byte-compilation errors (`python -m compileall -q worldsim tests`).

## Phase 13.4 independent validation and robustness audit

Phase 13.4 conducts an independent scientific audit of WorldSim Phase 13.3, verifying metric integrity, evaluating 100-episode stability across unselected seeds, formally categorizing remaining layout failure modes, and stress-testing multi-agent resource fairness under asymmetric claimant conditions:

1. **Audit Metric Integrity & Collision Breakdown:**
   - **Resolution of Contention vs Deadlock Accounting:** In Scenario 1 (8-Agent Bottleneck), the 520 total collisions (26 per episode) represent transient, single-tick queue and doorstep contentions. Priority rotation resolves right-of-way each tick, ensuring agents advance before persistent blocking can cause terminal failure.
   - **Deadlock Resolution Audit:** Across 20 episodes, 260 transient deadlock events were detected by `DeadlockDetector` (`persistence_threshold = 2`), and all 260 were successfully resolved (`successful_recoveries = 260`, `unresolved_deadlocks = 0`).
   - **Starvation Accounting:** The 290 recorded starvation events reflect temporary delays ($\ge 3$ consecutive ticks blocked in doorstep queues), but zero agents exhausted (`exhaustion_counts = 0`). All 8 agents reached their goals within 28.5 mean steps.

2. **100-Episode Extended Validation & Diverse Layout Generalization:**
   - **100-Episode Extended Evaluation (Seeds 200–299):** Evaluated 100 consecutive episodes on the 8-agent bottleneck. Achieved **100/100 (100.0%) overall completion**, 100.0% per-agent success, 28.50 mean steps (range 27–30 steps), 2,600 total collisions (26.0 per episode), 1,300 detected and resolved deadlocks (0 unresolved), and 0 exhaustions.
   - **Diverse Layout Permutations:**
     - Shuffled Target Assignments (Scenario 1C, seeds 300–319): Achieves **100.0% overall completion** with randomized agent-to-target pairings.
     - Opposing Crossing Walls (Scenario 1D, seeds 300–319): 4 West wall agents crossing paths with 4 East wall agents through the central corridor achieve **100.0% overall completion**, confirming multi-agent stability beyond corner-to-corner layouts.

3. **Rigorous Layout Failure Categorization & Diagnostic Breakdown:**
   - **Scenario 4A (Relocated Resources Baseline, energy=6):**
     - *Classification:* **Category 1 (Geometrically & Energetically Infeasible)**.
     - *Proof:* Agent 1 at $(0, 0)$ has Manhattan distance $d=7$ to resources at $(2, 5)$ and $(5, 2)$, and $d=14$ to target $(7, 7)$. Initial energy $E=6 < 7$. The agent exhausts at step 6 under every possible policy.
   - **Scenario 4B (Relocated Resources Feasible, energy=7):**
     - *Classification:* **Category 3 (Policy / Coordination Breakdown)**.
     - *Proof:* Solvable reachability (75.0% mean agent success), but fails overall completion due to an active head-on swap lock at row $y=7$. Agent 3 at $(4, 7)$ has target $(5, 7)$; Agent 4 at $(5, 7)$ has target $(4, 7)$. Because both agents are active and standing on each other's target cells, goal-cell yielding does not apply, and neither policy will voluntarily detour away from its target cell.
   - **Scenario 5A (Asymmetric Starts Baseline, energy=6):**
     - *Classification:* **Category 1 & 2 (Infeasible & Undiscoverable)**.
     - *Proof:* Agents 2 $(0, 7)$ and 3 $(7, 0)$ have $d=7$ to central resources $(3, 3)$ and $(4, 4)$, which exceeds initial energy 6 (Category 1). Furthermore, Euclidean distance $d = \sqrt{3^2 + 4^2} = 5.0 > 2.0$ (sensory radius), rendering the resources invisible from the start (Category 2).
   - **Scenario 5B (Asymmetric Starts Feasible, energy=8):**
     - *Classification:* **Category 3 (Policy / Coordination Breakdown)**.
     - *Proof:* Solvable with 0 exhaustions (93.8% mean agent success, 50.0% overall completion). In incomplete episodes (e.g. seed 101), Agent 1 ping-pongs between $(4, 7)$ and $(4, 6)$ due to a sensory-horizon limit cycle: resting completed Agent 5 at $(6, 7)$ is visible from $(4, 7)$ ($d=2.0 \le 2.0$), triggering a detour UP to $(4, 6)$; but from $(4, 6)$, $(6, 7)$ falls outside the sensory horizon ($d = \sqrt{2^2 + 1^2} \approx 2.24 > 2.0$), clearing the obstacle from observation and directing Agent 1 back DOWN to $(4, 7)$.

4. **Resource Fairness Stress Suite:**
   - **Asymmetric Energy Contention:** When Agent 1 has critical energy ($e=1$) and Agent 2 has abundant energy ($e=5$), `Fair-Balanced` responds to urgency and grants priority to Agent 1, whereas simple `Priority-Rotation` alternates blindly, causing unnecessary exhaustion.
   - **Tripartite Contention:** Fair-Balanced rotates equitably among 3 claimants ($1/3$ share each, Jain index = 1.000).
   - **Staggered Arrivals:** Correctly prioritizes late-arriving urgent agents on subsequent ticks.

5. **Benchmark Suite & Test Coverage:**
   - Machine-readable benchmark results exported to `benchmark_results_phase13_4.json` (16 audit scenarios, executable via `python run_phase13_4_benchmark.py`).
   - Dedicated unit test suite in `tests/test_multi_agent_phase13_4.py` (8 test cases).
   - All 154 unit tests pass in 1.66s (`python -m unittest discover -s tests -v`).
   - Zero byte-compilation errors (`python -m compileall -q worldsim tests`).

## Phase 13.3 reliable completion, fairness & benchmark validation

Phase 13.3 resolves the critical task completion bottleneck and establishes true resource fairness across WorldSim's multi-agent system:
1. **Audited Root Causes of Incomplete Completion:**
   - *Terminal Cell / Corridor Blocking ("Aisle vs Window Seat"):* In Phase 13.2, agents that reached their target (`status == "reached_target"`) became solid immovable obstacles. When front agents reached row 6 or earlier cells in row 7, they formed an impenetrable physical wall, completely sealing off the deeper target cells (e.g. (5, 7), (6, 7), (7, 7)) and guaranteeing 0.0% overall episode completion despite high per-agent success.
   - *Stationary Agent Collision Conflation:* Active agents attempting to pass through cells of completed agents suffered "occupied" collisions and were penalized by blocked-destination memory, causing them to reverse direction away from their goals.
   - *Arbitration Starvation & Masked Failure in Scenario 3:* Consumable single-resource arbitration reset per episode without inter-episode state, and hash-based tie-breaking permanently favored one agent across all seeds. Furthermore, reporting average per-agent success (50%) disguised the fact that one agent starved 100% of the time.
   - *Geometric Infeasibility in New Layouts (Scenarios 4 & 5):* Relocated resources in Scenario 4 placed Agent 1 at Manhattan distance 7 with only 6 initial energy; in Scenario 5, Agents 2 and 3 started at distance 7 from all central resources with only 6 initial energy. These were geometrically impossible without parameter normalization.
2. **Architectural Improvements:**
   - *Goal-Cell Yielding & Corridor Traversal:* `MultiAgentSimulation._resolve_conflicts` exempts agents with `status == "reached_target"` from blocking moving active agents with "occupied" collisions. Resting agents yield right-of-way, allowing active agents to traverse corridor and terminal cells to reach deeper goals without displacing completed agents.
   - *BFS Corridor Routing Fallback:* In `RobustResourcePolicy`, BFS path planning first attempts to route strictly around all other agents. If no open path exists (narrow corridor or cul-de-sac), it falls back to routing through yielding completed agents while avoiding genuinely impassable dead (exhausted) obstacles.
   - *Deterministic Fair Rotation Tie-Breaking:* `ResourceArbitrator` under `fair_balanced` rotates tie-breaking deterministically modulo `len(claimants)` across ticks/seeds, guaranteeing 50/50 equitable access across episodes for consumable resources (eliminating permanent Lowest-ID bias) and 100% mutual completion for reusable resources.
   - *Optional Persistent Arbitrator Support:* `MultiAgentSimulation` accepts an optional external `ResourceArbitrator`, enabling cross-episode starvation tracking.
3. **Comprehensive 9-Scenario Benchmark Matrix (`benchmark_results_phase13_3.json`):**
   - **Scenario 1 (Original 8-Agent Bottleneck):** Baseline achieved 0.0% overall completion (3,860 cols); Phase 13.1 achieved 0.0% overall completion (410 cols); Phase 13.3 achieves **100.0% overall completion** (20/20 episodes completed, 100% per-agent success, 28.5 mean steps, 520 cols).
   - **Scenario 2 (Four-Agent Cross Congestion Regression):** **100.0% overall completion** maintained (8.0 mean steps, 80 cols, 0 deadlocks).
   - **Scenario 3 (Two-Agent Resource Competition):**
     - Lowest-ID: Agent 1: 100.0%, Agent 2: 0.0% (100% starvation for Agent 2, Jain 0.500).
     - Priority-Rotation: Agent 1: 50.0%, Agent 2: 50.0% (equitable distribution).
     - Fair-Balanced: Agent 1: 50.0%, Agent 2: 50.0% (equitable distribution).
     - Reusable Resource: **100.0% overall completion** (both agents recharge, Jain 1.000).
   - **Scenario 4 (Eight-Agent Relocated Resources):** Baseline configuration confirms geometric infeasibility of Agent 1 (dist=7 > energy=6, 0.0% overall); feasible configuration (energy=7) elevates Agent 1 success to 50.0% and mean agent success to 75.0%.
   - **Scenario 5 (Eight-Agent Asymmetric Starts):** Baseline confirms perimeter agents exhaust before central discoveries arrive (40 exhaustions); feasible configuration (energy=8) achieves **50.0% overall completion** and 93.8% mean agent success (0 exhaustions).
   - **Scenario 6 (Completed Agent Goal-Cell Yielding in Narrow Corridor):** **100.0% overall completion** (5.0 steps, 0 occupied collisions).
   - **Scenario 7 (Deadlock & Head-On Swap Stress Test):** **100.0% overall completion** (0 deadlocks, 0 swaps).
   - **Scenario 8 (Intentionally Infeasible Environment):** 0.0% overall completion, 100.0% exhaustion honestly reported without false claims.
   - **Scenario 9 (Single-Agent Baseline Regression):** **100.0% overall completion** (11.0 steps), preserving single-agent backward compatibility.
4. **Test Suite:**
   - Dedicated unit test suite `tests/test_multi_agent_phase13_3.py` (8 test cases).
   - All 146 unit tests pass in 0.77s (`python -m unittest discover -s tests -q`).

## Phase 13.2 resolve the eight-agent bottleneck

Phase 13.2 resolves the multi-agent scaling bottleneck identified in Phase 13 and Phase 13.1, where 8 agents funneling into central resources suffered severe congestion, queue exhaustion, and idle resources:
1. **Diagnostic Root Cause Analysis:**
   - **Reusable Multi-Recharge Underestimation:** Phase 12.2/13.1 utility calculation evaluated single recharges (`gain = min(recharge_amount, max_energy - cur_e)`). For reusable resources requiring multiple recharges to cross the grid, this falsely marked the resource as "insufficient", causing agents to wander in `uncertain_explore` and exhaust.
   - **Coordinator Intent Hijacking & Fatal Detours:** The coordinator redirected agents attempting to enter an occupied resource pad away into neighboring cells. Under WorldSim rules, doorstep collisions consume zero energy, while forced redirects consume movement energy and push agents away into fatal dead ends.
   - **Energy-Blind Alternative Suggestions:** Alternative moves were suggested without verifying whether the resulting coordinate left sufficient energy to reach the target, inadvertently causing energy starvation.
   - **Resource Load Imbalance:** Lack of persistent cross-tick commitment tracking allowed all agents to herd to a single resource while other identical resources sat underutilized.
   - **Static Inactive Obstacles:** Once agents reached their goals or exhausted, they became stationary obstacles, creating physical bottlenecks that active agents needed to route around.
2. **Coordinated Architectural Solutions:**
   - **Multi-Recharge Capacity Modeling:** `ResourceUtilityEvaluator.evaluate_resource` models cumulative recharge capacity for non-consumable resources up to `max_energy`, correctly assessing multi-recharge feasibility.
   - **Doorstep Queuing & Collision Arbitration:** `MultiAgentCoordinator.suggest_alternative_move` permits agents targeting resources or target waypoints to queue at the doorstep, letting the engine's deterministic vertex arbitration resolve access cost-free.
   - **Energy-Aware Action Filtering:** Coordinator checks `dist(d, target) <= energy - 1` before suggesting any alternative move, eliminating fatal redirects.
   - **Persistent Resource Commitments & Load Balancing:** `register_intent` and `reset_tick` maintain resource commitments across ticks within each episode, balancing demand across all active resources via load penalties (`u.utility_score - (load * 30.0)`).
   - **Inactive Obstacle Routing:** `RobustResourcePolicy` distinguishes inactive stationary agents from active moving agents, employing dead-end look-ahead and BFS path guidance to route around completed agents.
3. **Controlled Benchmark Suite (Scenarios 1 through 6):**
   - **Scenario 1 (Original 8-Agent Bottleneck):** Baseline suffers 4,760 collisions and 2 unused resources. Phase 13.1 suffers 1,795 collisions and 880 swap locks. Phase 13.2 cuts collisions to 721 (85% reduction from baseline), reduces swap collisions by 98% (18 swaps), achieves 63.1%–87.5% agent success, and fully balances load across both resources (160 at (3, 3), 192 at (4, 4), 0 unused).
   - **Scenario 2 (Four-Agent Cross Congestion Regression):** 100.0% completion maintained with 10.0 mean duration and zero regressions.
   - **Scenario 3 (Two-Agent Resource Competition):** Verifies fair allocation under `lowest_id` vs `fair_balanced`.
   - **Scenario 4 (New 8-Agent Asymmetric Resource Layout):** Relocates resources to (2, 5) and (5, 2); achieves 62.5% mean agent success with zero unused resources.
   - **Scenario 5 (8-Agent Asymmetric Starts):** Agents starting across opposing corners achieve 75.0% mean agent success, 410 total collisions, and symmetric resource utilization (120 at (3, 3), 120 at (4, 4)).
   - **Scenario 6 (Intentionally Infeasible Environment):** 0.0% completion, 100.0% exhaustion honestly accounted without false claims.
   - Machine-readable benchmark results exported to `benchmark_results_phase13_2.json` (reproducible via `python run_phase13_2_benchmark.py`).
4. **Test Suite:**
   - Dedicated unit test suite `tests/test_multi_agent_phase13_2.py` verifies all coordination mechanisms.
   - All 138 unit tests pass in 0.74s (`python -m unittest discover -s tests -q`).

## Phase 13.1 multi-agent coordination, resource allocation, and congestion control

Phase 13.1 addresses the multi-agent scaling failures and bottlenecks identified in Phase 13 (particularly the 8-agent bottleneck where symmetric vertex bouncing and herd behavior caused severe congestion):
1. **Congestion-Aware Movement & Deterministic Priority Rotation:**
   - Replaces symmetric vertex bouncing with deterministic tick-based priority rotation (`collision_rule='priority_rotation'`), breaking symmetric vertex locks.
   - `RobustResourcePolicy` integrates agent-owned blocked-destination memory (`blocked_destinations`), temporarily penalizing repeatedly blocked cells unless they are the agent's goal or an available resource.
   - `MultiAgentCoordinator` offers voluntary intention exchange (`register_intent`, `suggest_alternative_move`), advising lower-priority agents to choose equivalent detours before conflicts occur.
2. **Fair, Energy-Aware Shared-Resource Allocation:**
   - `ResourceArbitrator` replaces permanent lowest-ID bias with four configurable allocation rules:
     - `lowest_id`: Phase 13 legacy alphabetical rule.
     - `priority_rotation`: Deterministic round-robin allocation rotating by tick.
     - `need_urgency`: Prioritizes agents with greatest energy deficit / lowest energy.
     - `starvation_prevention`: Prioritizes agents with highest consecutive contested losses.
     - `fair_balanced`: Synthesizes urgency and starvation compensation.
   - Eliminates permanent lowest-ID starvation; lower-priority agents receive access when needed.
3. **Deadlock Detection & Bounded Recovery Tracking:**
   - `DeadlockDetector` monitors simulation events to detect repeated swaps, persistent occupied blocking, and repeated vertex deadlocks beyond a persistence threshold.
   - Tracks deadlock events, recovery interventions, successful recoveries upon movement, and unresolved deadlocks.
4. **Fairness and Starvation Monitoring:**
   - `FairnessMonitor` tracks per-agent steps, energy consumed, resource claims won/lost, energy gained, ticks blocked, and starvation events.
   - Computes Jain's Fairness Index for `goal_completion`, `energy_gained`, `spatial_progress`, and `resource_access`.
5. **Controlled Benchmark Suite (Exp A through Exp J + 4 Ablations):**
   - **Exp A (Single-Agent Regression):** 100.0% completion, 7.0 steps, 0 collisions. Confirms identical performance to Phase 12.2 single-agent baseline.
   - **Exp B (Two-Agent Conflict):** 100.0% completion in both baseline and improved configurations.
   - **Exp C (Four-Agent Cross Congestion):** Baseline suffers 3,300 collisions and 0% completion; Phase 13.1 achieves 100.0% completion with 0 collisions.
   - **Exp D (Eight-Agent Bottleneck Scaling):** Baseline suffers 900 collisions; Phase 13.1 cuts collisions down to 500, with 5 of 8 agents reaching target (62.5% per-agent success).
   - **Exp E (Fair Resource Allocation):** Lowest-ID permanently starves agent 2 (100% vs 0%, Jain 0.50); Fair allocation gives priority to agent 2 (Jain 0.50/1.00 balanced across trials).
   - **Exp F (Shared Goals):** Any-agent criterion completes in 5.0 steps (100.0%).
   - **Exp G (Cooperative Discovery Sharing):** Isolated achieves 0.0% completion vs Shared achieves 100.0% completion.
   - **Exp H (Starvation Monitoring):** Detects 180 starvation events with 45 successful recoveries.
   - **Exp I (Provably Infeasible):** 0.0% completion, 100.0% exhaustion honestly verified.
   - **Exp J (Generalization Across 3 Distinct Layouts):** Evaluates Asymmetric Crossway (J1: 100.0%), Perimeter Diagonal (J2), and Dispersed Multi-Resource (J3).
   - **Ablation Studies:** Isolates congestion handling only (420 cols) vs resource allocation only (620 cols) vs full Phase 13.1 (500 cols).
   - Saved to `benchmark_results_phase13_1.json` (reproducible via `python run_phase13_1_benchmark.py`).
6. **GUI Viewer & Test Suite:**
   - `MultiAgentWorldViewer` visualizes deadlock cells with dashed red outlines, blocked agents with `[B]` badges, and deadlock recovery counters in the HUD.
   - Full test suite: 133 automated tests pass in 0.75s (`python -m unittest discover -s tests -v`).

## Phase 13 multi-agent simulation

Phase 13 extends WorldSim from a single-agent framework into a simultaneous multi-agent grid simulation where multiple autonomous entities operate simultaneously in the same environment:
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
7. **Viewer & Backward Compatibility:** `MultiAgentWorldViewer` provides interactive multi-agent visualization with distinct color palettes, per-agent energy tags, and status displays. All 116 automated tests pass in 0.75s, maintaining 100% backward compatibility with Phases 1–12.2.

## Phase 12.2 robust resource-aware learning

Phase 12.2 resolves the concrete failure modes uncovered during the Phase 12.1 audit by introducing a modular, observation-limited robustness layer (`RobustResourcePolicy`, `AgentMemory`, `RechargeEvaluator`, `ResourceUtilityEvaluator`, `ExplorationController`, `CycleDetector`):
1. **Intelligent Recharge Control:** Prevents infinite recharge loops at reusable resources by verifying whether energy is already at capacity or already sufficient to reach the target directly, or if an intermediate recharge is unnecessary when en route to a chosen waypoint.
2. **Utility-Based Resource Selection:** Replaces naive nearest-neighbor proximity with a transparent route utility calculation that accounts for travel distance, recharge amount, post-recharge capacity, and remaining distance to target, correctly resolving the counterexample where a nearest resource is insufficient but a farther resource is sufficient.
3. **Safe Exploration Beyond Sensory Range:** When a required resource is outside initial sensory radius $r=2.0$, agent-owned memory computes frontier coverage over unobserved grid cells, expanding sensory perception safely to discover resources without hidden state access.
4. **Failure Recovery and Cycle Detection:** A bounded recovery controller detects repeated recharges, 2-step ping-pong oscillations, repeated rejections, and unproductive position loops, penalizing cyclic actions and steering the agent toward alternatives.

Across a rigorous 15-scenario benchmark suite comparing 5 policies over 50 evaluation episodes each, **Phase 12.2 Robust Linear Q achieves 100.0% success across all 12 feasible tasks** (up from 75.0% for Phase 12.1 linear and 8.3% for tabular Q). Both reusable-resource looping (Scenario 5) and sensory-horizon blindness (Scenario 6) are completely solved (both improving from 0.0% to 100.0%), and provably infeasible environments (Scenarios 10, 11, 13) are recognized without infinite loops or false success claims. Benchmark results are preserved in `benchmark_results_phase12_2.json` and reproducible via `python run_phase12_2_benchmark.py`.

## Phase 12.1 generalization validation and benchmark audit

Phase 12.1 conducts an independent scientific audit of Phase 12's generalization claims. Through matched single-task and multi-task training across multiple budgets (300, 900, 1500 episodes) and evaluation on a 10-task final held-out test suite (8 feasible, 2 provably infeasible), the audit rigorously confirms that linear function-approximation Q-learning substantially outperforms tabular Q-learning in zero-shot generalization (75.0%–87.5% vs 0.0% on held-out tasks). Crucially, the audit uncovers and documents real representation limitations: reusable resource over-recharging, sensory horizon blindness under small radii, and nearest-resource feature aliasing. Provably infeasible environments lead to clean energy exhaustion rather than infinite loops. Machine-readable benchmarks are generated via `python run_benchmark.py`.

## Phase 12 generalizable reinforcement learning

Phase 12 addresses the generalization bottleneck of tabular Q-learning by introducing feature-based linear function approximation (`LinearQLearner`, `FeatureExtractor`, `LinearLearningRunner`, `LinearLearningConfig`). Rather than hashing absolute coordinates, the agent evaluates candidate actions using normalized, relative features derived strictly from permitted observations: relative target distance/direction, energy deficit, resource proximity, and action feasibility. Across controlled benchmarks with unseen targets, unseen starting points, unseen resource layouts, and combined distribution shifts, the function-approximation agent demonstrates 100% zero-shot transfer, matching optimal planner baselines without retraining.

## Phase 11 reinforcement learning for energy and resources

Phase 11 extends tabular Q-learning to energy-constrained navigation with consumable and reusable resources. `QLearner` state representation supports `(entity_x, entity_y, energy, observable_target, observed_resources)` tuples while strictly respecting observation boundaries. Actions include movement (`UP`, `DOWN`, `LEFT`, `RIGHT`) and `Action.RECHARGE`. With configurable exploration decay, failed action penalties, and terminal energy exhaustion, the agent learns through trial-and-error to recognize energy limits, navigate to resource cells, recharge, and reach goals without any hardcoded routing logic. In benchmark evaluation, the learned policy achieves 100% success matching the optimal step count (6 steps) of classical state-space planners.

## Phase 10 energy-aware planning and learning

Phase 10 extends planning to the combined $(position, energy, resource\_state)$ state space. `plan_energy(problem)` performs state-space search over explicit `EnergyPlanningProblem` inputs to find the shortest action sequence reaching the target while respecting movement costs, maximum energy, and consumable/reusable resource depletion. `execute_energy_plan(config, problem, plan_result)` verifies simulation-planning start consistency and halts on divergence. `EnergyPlannerPolicy` provides an observation-limited planner-backed agent that replans dynamically as resources are observed or consumed. Tabular Q-learning includes an opt-in `include_energy` state-key configuration.

## Phase 9 energy, resources, and action planning

WorldSim Phase 9 adds an energy system (`initial_energy`, `max_energy`, `movement_energy_cost`), explicit resource cells (`Resource(position, recharge_amount, consumable)`), an `Action.RECHARGE` action, and an `EnergyAwarePolicy` that chooses actions from permitted observations. Successful movement consumes energy; blocked moves, paused attempts, or moves with insufficient energy are rejected without advancing the clock or consuming energy. Recharging on a resource cell restores energy up to `max_energy` and consumes consumable resources. Multi-step experiments via `run_experiment(config, actions, stop_on_rejected=...)` and `run_policy_experiment(config, policy, max_steps=...)` record transitions and energy changes in immutable `Experience` tuples.

## Phase 8 standalone planning

`plan(problem)` computes a shortest-path action sequence using breadth-first search on an explicit `PlanningProblem(width, height, start, target)` without inspecting simulation internals. `execute_plan(config, plan_result)` runs the plan actions through the public `Simulation.step` API in a fresh simulation. The planner uses explicit, privileged target and boundary inputs to achieve zero-shot optimal paths (Manhattan distance) with 0 training episodes. In contrast, target-unaware tabular Q-learning requires 500 training episodes per target and cannot generalize across goals without retraining.

## Phase 7 persistent memory

`ExperienceMemory(capacity)` stores immutable observation, action, reward, and resulting-observation entries in FIFO order. `record`, `retrieve(observation)`, `entries`, and `clear()` are its public API. Retrieval matches only the permitted observed entity coordinate; memory has no target or private-world access. Pass memory to `LearningRunner(memory=...)` to opt in. Training records transitions and may reuse the highest-reward matching action; episode resets retain memory, while `clear()` removes it. Evaluation does not mutate memory or Q-values. Memory remains separate from the Q-table and does not change its update equation. No independent advantage over Q-learning was measured in the current fully observed toy task.

Phase 6 adds a small tabular Q-learning experiment layer to WorldSim. It remains a minimal grid simulation with bounded perception, controlled movement, in-memory experience records, and controlled experiments; it does not claim general intelligence, memory, planning, or survival behavior.

## Requirements

- Python 3.10+ with Tkinter (included by most standard Python installers)

## Run

From the repository root:

```powershell
python -m worldsim
```

The window shows a configurable grid, its one blue entity, and a subtle tint over cells that it can observe. The simulation clock updates every 350 ms. Use **Pause** and **Resume** to control the clock, then use the directional buttons or keyboard arrow keys to move one cell at a time.

## Test

```powershell
python -m unittest discover -s tests -v
```

## Configure a run

Create a `WorldConfig` and pass it to `Simulation`; the viewer remains separate:

```python
from worldsim import Action, Simulation, WorldConfig
from worldsim.viewer import WorldViewer

config = WorldConfig(width=30, height=18, entity_id="observer-1", initial_x=4, initial_y=7, sensory_radius=2.0)
WorldViewer(Simulation(config)).run()
```

Coordinates are zero-based: `(0, 0)` is the top-left cell, `x` grows rightward, and `y` grows downward. Thus `UP` reduces `y`, `DOWN` increases `y`, `LEFT` reduces `x`, and `RIGHT` increases `x`.

## Perception

Call `Simulation.observe()` to obtain the entity's immutable `Observation`. It includes the entity coordinate and a tuple of observed cells. A cell is included when its Euclidean distance from the entity is at most `sensory_radius`: `dx^2 + dy^2 <= radius^2`. Coordinates outside the world are never included.

## Controlled actions

Pass one of the four `Action` enum values (`UP`, `DOWN`, `LEFT`, `RIGHT`) to `Simulation.step(action)`. It returns an immutable `StepResult` containing the resulting state, fresh observation, and `moved` flag. A successful move advances the tick once. Boundary moves, and all moves while paused, return `moved=False` with no state or tick change. Any value other than an `Action` raises `ValueError` before state changes.

```python
result = simulation.step(Action.RIGHT)
```

## Experience history

Every valid `step()` request creates one immutable `Experience`, available in chronological order through `simulation.experiences`. History is a tuple, so callers cannot append, remove, reorder, or replace records. Each record contains:

- `tick_before`, `observation_before`, and `position_before`
- the supplied `action` and whether it `moved`
- `position_after`, `observation_after`, and `tick_after`

Successful moves record the state transition. Boundary and paused action attempts also record an experience with `moved=False` and unchanged position/tick. Unknown or malformed action values raise `ValueError` and create no record. Observations, positions, and records are immutable; history exists only for the active process and has no persistence or learning behavior.

## Controlled experiments

Use `run_experiment(config, actions)` to execute an ordered action sequence in a new, private simulation. It returns an immutable `ExperimentResult` containing the configuration, initial position, supplied action tuple, action results, experience history, final position/tick, completion state, and any error details.

```python
from worldsim import Action, WorldConfig, run_experiment

config = WorldConfig(width=4, height=4, initial_x=1, initial_y=1, sensory_radius=1.0)
result = run_experiment(config, (Action.RIGHT, Action.DOWN, Action.LEFT))
assert result.completed
assert result.final_position.x == 1
```

The current simulation has no random behavior, so experiments accept no seed and are reproducible from the same configuration and action sequence. Boundary actions retain normal `step()` behavior and appear in results/history with `moved=False`. If an invalid action occurs, the runner stops immediately without skipping it, returns `completed=False`, preserves all preceding results and experiences, and reports `error` plus `stopped_at_index`. Results are immutable snapshots and expose no mutable simulation instance or history.

## Tabular Q-learning experiment

`LearningRunner` is separate from the world engine and creates a fresh `Simulation` for each episode while retaining its `QLearner` Q-table. Its state key is only the entity coordinate from `Observation.entity_position`; it does not inspect a hidden grid. Missing Q-values initialize to `0.0`. The experiment target is an external, fixed cell—not an innate goal.

```python
from worldsim import LearningConfig, LearningRunner, Position, WorldConfig

world = WorldConfig(width=4, height=4, initial_x=0, initial_y=0)
learning = LearningConfig(target=Position(3, 3), epsilon=0.2, max_steps=30)
runner = LearningRunner(world, learning, seed=7)
runner.train(500)
evaluation = runner.evaluate(100, seed=99, policy="learned")
baseline = runner.evaluate(100, seed=99, policy="random")
```

`LearningConfig` explicitly defines the target, target reward, nonterminal step cost, learning rate, discount factor, epsilon, and step limit. Reaching the target gives `target_reward` and terminates that episode; every nonterminal action attempt receives `step_cost`. The update is `Q(s,a) += alpha * (reward + gamma * max Q(s',.) - Q(s,a))`; terminal transitions omit the future term. Evaluation uses no exploration and never modifies the Q-table. With the example configuration, 500 episodes, training seed `7`, and evaluation seed `99`, learned evaluation reached the target in 100/100 episodes (mean 6.0 steps), while the random baseline reached it in 38/100 (mean 18.87 successful-episode steps). This demonstrates the configured toy task only.

## BFS planning and execution

`PlanningProblem` defines explicit world dimensions, start position, and target position. `plan(problem)` returns a `PlanResult(found, actions)` containing the shortest action sequence to the target. `execute_plan(config, result)` steps through the action sequence in a new `Simulation(config)` and returns a tuple of `StepResult` instances.

```python
from worldsim import PlanningProblem, Position, WorldConfig, execute_plan, plan

problem = PlanningProblem(width=4, height=4, start=Position(0, 0), target=Position(3, 3))
result = plan(problem)
assert result.found
assert len(result.actions) == 6  # Optimal Manhattan distance |3-0| + |3-0|

config = WorldConfig(width=4, height=4, initial_x=0, initial_y=0)
outcomes = execute_plan(config, result)
assert outcomes[-1].state.entity.position == Position(3, 3)
```

In controlled experiments comparing BFS Planning against tabular Q-learning (500 training episodes) and a random baseline across multiple target coordinates:
- Target `(1, 0)` (Manhattan dist 1): BFS achieved 100% success in 1 step (optimal); Q-learning achieved 100% in 1.00 mean step; Random achieved 76% in 6.04 mean steps.
- Target `(2, 1)` (Manhattan dist 3): BFS achieved 100% success in 3 steps (optimal); Q-learning achieved 100% in 3.00 mean steps; Random achieved 76% in 12.84 mean steps.
- Target `(0, 3)` (Manhattan dist 3): BFS achieved 100% success in 3 steps (optimal); Q-learning achieved 100% in 3.00 mean steps; Random achieved 48% in 16.17 mean steps.
- Target `(3, 3)` (Manhattan dist 6): BFS achieved 100% success in 6 steps (optimal); Q-learning achieved 100% in 6.00 mean steps; Random achieved 38% in 18.87 mean steps.

BFS operates on privileged target and world coordinates with 0 training episodes. By contrast, tabular Q-learning learns only through reward signals for a fixed target position; when evaluated on alternative targets without retraining, it achieves 0% success.

## Energy, resources, and energy-aware policy

`WorldConfig` accepts `initial_energy` (default 10), `max_energy` (default 10), `movement_energy_cost` (default 1), and explicit `resources`:

```python
from worldsim import (
    Action,
    EnergyAwarePolicy,
    Position,
    Resource,
    WorldConfig,
    run_policy_experiment,
)

resource = Resource(position=Position(2, 1), recharge_amount=7, consumable=True)
config = WorldConfig(
    width=5,
    height=5,
    initial_x=0,
    initial_y=0,
    initial_energy=5,
    max_energy=10,
    movement_energy_cost=1,
    sensory_radius=3.0,
    resources=(resource,),
)
```

### Action Semantics and Rules
- **Movement (`UP`, `DOWN`, `LEFT`, `RIGHT`)**: Consumes `movement_energy_cost` on successful move. Blocked boundary moves, paused attempts, or attempts with `energy < movement_energy_cost` are rejected without advancing the clock or consuming energy (`moved=False`, `success=False`).
- **Recharge (`RECHARGE`)**: Valid only when the entity is on an active resource cell. Increases energy by `recharge_amount` up to `max_energy`, advances the tick once, and consumes the resource if `consumable=True` (`moved=False`, `success=True`). Attempting to recharge on a non-resource cell is rejected without clock or energy changes.

### Policy Comparison
In a 5x5 world with start `(0, 0)`, target `(4, 4)` (Manhattan distance 8), `initial_energy=5`, `max_energy=10`, and a consumable resource at `(2, 1)` restoring 7 energy:
- **Energy-Ignorant Policy**: Reaches energy exhaustion after 5 moves at `(4, 1)`, failing with 0% task completion and 25 rejected moves.
- **Energy-Aware Policy**: Evaluates energy deficit ($5 < 8$), detects the observable resource at `(2, 1)`, detours to recharge (raising energy to 9), and reaches `(4, 4)` in 9 total steps with 4 energy remaining (100% completion).
- **Random Baseline (100 seeded runs)**: Reaches the target in only 1/100 episodes (1% completion), with mean final energy 0.03 and 0.21 mean recharges.

## Energy-aware state-space planning

`EnergyPlanningProblem` models state-space planning over `(position, energy, active_resources)`:

```python
from worldsim import (
    EnergyPlannerPolicy,
    EnergyPlanningProblem,
    Position,
    Resource,
    WorldConfig,
    execute_energy_plan,
    plan_energy,
)

res = Resource(Position(2, 0), recharge_amount=5, consumable=True)
problem = EnergyPlanningProblem(
    width=5,
    height=5,
    start=Position(0, 0),
    target=Position(4, 2),
    initial_energy=4,
    max_energy=8,
    movement_energy_cost=1,
    resources=(res,),
)

plan_result = plan_energy(problem)
assert plan_result.found
assert plan_result.plan_length == 7  # 2 moves -> recharge -> 4 moves

config = WorldConfig(
    width=5,
    height=5,
    initial_x=0,
    initial_y=0,
    initial_energy=4,
    max_energy=8,
    movement_energy_cost=1,
    resources=(res,),
)
execution = execute_energy_plan(config, problem, plan_result)
assert execution.completed and execution.reached_target
```

### Information Boundaries
- **Privileged Known-Model Planning (`plan_energy`)**: Receives explicit target and global resource locations as an experimental optimal baseline.
- **Observation-Limited Policies (`EnergyPlannerPolicy`, `EnergyAwarePolicy`)**: Restricted to information present in `Observation` (`energy`, `observed_resources`). `EnergyPlannerPolicy` dynamically replans as new resources are discovered or consumed, and never issues impossible actions.

### Controlled Experiment Benchmarks (6 Scenarios)

| Scenario | Privileged BFS Planner | Planner Policy | Energy-Aware Heuristic | Energy-Ignorant | Random Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Direct Reachable** ($d=3$, $E=5$) | 3 steps, 0 rech, $E_f=2$ | 3 steps, 100% | 3 steps, 100% | 3 steps, 100% | 10% succ |
| **2. Single Recharge** ($d=6$, $E=4$) | 7 steps, 1 rech, $E_f=3$ | 7 steps, 100% | 7 steps, 100% | 0% (26 rej) | 0% succ |
| **3. Multiple Recharges** ($d=10$, $E=4$) | 12 steps, 2 rech, $E_f=2$ | 12 steps, 100% | 12 steps, 100% | 0% (26 rej) | 0% succ |
| **4. Unreachable Target** ($d=8$, $E=3$) | Found=False | 0 steps, 0 rej | 3 steps (0 rech) | 0% (27 rej) | 0% succ |
| **5. Consumable Depletion** ($d=6$, $E=3$) | 7 steps, 1 rech, $E_f=0$ | 7 steps, 100% | 7 steps, 100% | 0% (27 rej) | 0% succ |
| **6. Reusable Resource** ($d=4$, $E=2$) | 5 steps, 1 rech, $E_f=0$ | 5 steps, 100% | 6 steps, 2 rech | 0% (28 rej) | 2% succ |

Notice that in Scenario 6, `EnergyPlannerPolicy` found the strictly optimal 5-step plan (1 recharge), whereas the greedy heuristic took 6 steps due to an extra redundant recharge. In Scenario 4, `EnergyPlannerPolicy` safely avoided executing futile moves when the planner proved the target unreachable.

## Energy-aware reinforcement learning

`LearningConfig` supports energy-aware tabular Q-learning with fine-grained reward shaping, exploration decay, and state representation controls:

```python
from worldsim import (
    Action,
    LearningConfig,
    LearningRunner,
    Position,
    Resource,
    WorldConfig,
)

target = Position(3, 2)
res = Resource(position=Position(2, 0), recharge_amount=4, consumable=True)

world = WorldConfig(
    width=4,
    height=4,
    initial_x=0,
    initial_y=0,
    initial_energy=3,
    max_energy=6,
    movement_energy_cost=1,
    sensory_radius=3.0,
    resources=(res,),
    observable_target=target,
)

learning_cfg = LearningConfig(
    target=target,
    step_cost=-0.01,
    target_reward=1.0,
    failed_action_penalty=-0.05,
    recharge_cost=-0.01,
    learning_rate=0.3,
    discount_factor=0.9,
    epsilon=0.4,
    epsilon_decay=0.997,
    epsilon_min=0.02,
    max_steps=25,
    include_energy=True,
    include_resources=True,
    include_target=True,
    terminal_on_energy_exhaustion=True,
)

runner = LearningRunner(world, learning_cfg, seed=42)
runner.train(1500)
evaluation = runner.evaluate(100, seed=99, policy="learned")
assert evaluation.success_rate == 1.0
```

### State Space & Information Boundaries
The agent makes decisions using permitted `Observation` fields:
- `include_energy=True`: Includes entity's current energy in the state key.
- `include_target=True`: Includes `observation.target` if made observable via `WorldConfig.observable_target`.
- `include_resources=True`: Includes coordinates of observed active resources within sensory radius. Depletion of consumable resources upon recharge naturally updates the observed state.
- Composite state key: `(pos.x, pos.y, energy, (target.x, target.y), ((res.x, res.y), ...))`

### Controlled Benchmark Comparison
In a 4x4 grid with start `(0, 0)`, target `(3, 2)` (Manhattan distance 5), `initial_energy=3`, `max_energy=6`, `movement_energy_cost=1`, and a consumable resource at `(2, 0)` restoring 4 energy:

| Policy | Success Rate | Mean Steps | Cumulative Reward | Remaining Energy | Recharge Count | Rejected Actions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learned Q-Policy** | **100.0%** (100/100) | **6.00** | **0.9500** | **2.00** | **1.00** | **0.00** |
| **EnergyPlannerPolicy** | 100.0% (1/1) | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 |
| **EnergyAwarePolicy** | 100.0% (1/1) | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 |
| **Random Baseline** | 0.0% (0/100) | N/A | -0.2500 | 0.00 | 0.01 | 21.95 |

The learned Q-policy discovered through pure trial-and-error reinforcement learning the exact optimal 6-step trajectory (`RIGHT` -> `RIGHT` -> `RECHARGE` -> `DOWN` -> `RIGHT` -> `DOWN`), matching the privileged BFS state-space planner and energy heuristic without any hand-coded pathing or lookahead.

### Generalization Limitation
Tabular Q-learning associates state keys with expected returns for seen states. When evaluated on novel unvisited target or resource configurations without retraining, Q-values remain at their initial value (0.0), leading to arbitrary action choices. Classical planning (`plan_energy`) computes plans on-the-fly for any valid configuration without training episodes.

## Generalizable reinforcement learning (Phase 12)

Phase 12 solves the tabular generalization limitation by introducing **feature-based linear function approximation** (`LinearQLearner`, `FeatureExtractor`, `LinearLearningRunner`, `LinearLearningConfig`).

### Architecture and Feature Representation

Instead of memorizing absolute coordinate tuples `(x, y, energy, target, resources)`, `FeatureExtractor` extracts relative, normalized features strictly from permitted `Observation` fields:
1. **Action Feasibility & Legality:** `valid_move`, `valid_recharge`, `move_into_wall`, `move_no_energy`, `recharge_no_resource`, `invalid_action`.
2. **Target Navigation:** `target_dist_norm` ($d / D_{\text{max}}$), `move_closer_target`, `move_away_target`, `reaches_target`.
3. **Energy State & Deficit:** `energy_norm` ($E / E_{\text{max}}$), `energy_critical` ($E \le \text{cost}$), `energy_deficit` ($\max(0, d_{\text{target}} \cdot \text{cost} - E) / (D_{\text{max}} \cdot \text{cost})$), `energy_sufficient`.
4. **Resource Interaction & Detour:** `move_closer_resource`, `seek_res_when_deficient` ($\text{move\_closer\_resource} \times \text{energy\_deficit}$), `recharge_gain`, `recharge_when_needed`, `resource_consumable`.
5. **Action Priors & Bias:** One-hot action indicators and baseline bias.

The Q-value for candidate action $a$ given observation $s$ is:
$$Q(s, a; \mathbf{w}) = \sum_{i} w_i \phi_i(s, a)$$
Weights are updated via semi-gradient TD(0) with gradient clipping:
$$\delta = r + \gamma \max_{a'} Q(s', a'; \mathbf{w}) - Q(s, a; \mathbf{w})$$
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \text{clip}(\delta) \cdot \boldsymbol{\phi}(s, a)$$

### Usage Example

```python
from worldsim import (
    FeatureExtractor,
    LinearLearningConfig,
    LinearLearningRunner,
    LinearQLearner,
    Position,
    Resource,
    WorldConfig,
)

# Multi-task training set breaking collinearity between resource & target
env1 = WorldConfig(
    width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
    resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
    observable_target=Position(3, 2),
)
env2 = WorldConfig(
    width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
    resources=(Resource(Position(0, 2), recharge_amount=4, consumable=True),),
    observable_target=Position(3, 2),
)
env3 = WorldConfig(
    width=4, height=4, initial_x=0, initial_y=0, initial_energy=3, max_energy=6,
    resources=(Resource(Position(2, 0), recharge_amount=4, consumable=True),),
    observable_target=Position(1, 3),
)

config = LinearLearningConfig(learning_rate=0.08, discount_factor=0.95, max_steps=25)
runner = LinearLearningRunner(env1, config, seed=123)
runner.train_tasks([env1, env2, env3], episodes_per_task=300)

# Zero-shot evaluation on completely unseen target & resource configuration:
held_out_world = WorldConfig(
    width=4, height=4, initial_x=0, initial_y=1, initial_energy=3, max_energy=6,
    resources=(Resource(Position(1, 2), recharge_amount=4, consumable=True),),
    observable_target=Position(3, 3),
)
evaluation = runner.evaluate(100, world=held_out_world, seed=42)
assert evaluation.success_rate == 1.0
assert evaluation.steps_to_target[0] == 6  # Optimal path matching planner
```

### Controlled Benchmark Results (Experiments A–F)

Each condition was evaluated over 100 seeded evaluation episodes comparing:
1. **Linear Q-Learner:** Function-approximation RL agent using permitted observations.
2. **Tabular Q-Learner:** Tabular baseline trained on the familiar Phase 11 configuration.
3. **EnergyPlannerPolicy (Ref):** Privileged classical BFS planner baseline.
4. **Random Baseline:** Uniform random action baseline.

#### Experiment A: Familiar Configuration (Start (0,0), Target (3,2), Res (2,0), E=3)
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Familiar)** | **100.0%** | **6.00** | **0.9500** | **2.00** | **1.00** | **0.00** | 0.0% | 0.0% |
| **Tabular Q-Learner (Familiar)** | 100.0% | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 | 0.0% | 0.0% |
| **EnergyPlannerPolicy (Ref)** | 100.0% | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 0.0% | N/A | -1.3760 | 0.00 | 0.02 | 26.90 | 100.0% | 0.0% |

#### Experiment B: Unseen Target Location (Start (0,0), Target (3,0), Res (2,0))
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Zero-Shot)** | **100.0%** | **4.00** | **0.9700** | **4.00** | **1.00** | **0.00** | 0.0% | 0.0% |
| **Tabular Q-Learner (Zero-Shot)** | 0.0% | N/A | -0.2500 | 3.00 | 0.00 | 25.00 | 0.0% | 100.0% |
| **EnergyPlannerPolicy (Ref)** | 100.0% | 3.00 | 0.9800 | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 5.0% | 30.00 | -0.1915 | 0.00 | 0.02 | 26.90 | 95.0% | 0.0% |

#### Experiment C: Unseen Starting Position (Start (1,1), Target (3,2), Res (2,0))
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Zero-Shot)** | **100.0%** | **6.00** | **0.9500** | **2.00** | **1.00** | **0.00** | 0.0% | 0.0% |
| **Tabular Q-Learner (Zero-Shot)** | 100.0% | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 | 0.0% | 0.0% |
| **EnergyPlannerPolicy (Ref)** | 100.0% | 3.00 | 0.9800 | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 7.0% | 30.00 | 0.6327 | 0.00 | 0.05 | 26.75 | 93.0% | 0.0% |

#### Experiment D: Unseen Resource Layout (Start (0,0), Target (3,2), Res (1,1))
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Zero-Shot)** | **100.0%** | **6.00** | **0.9500** | **2.00** | **1.00** | **0.00** | 0.0% | 0.0% |
| **Tabular Q-Learner (Zero-Shot)** | 0.0% | N/A | -1.2500 | 3.00 | 0.00 | 25.00 | 0.0% | 100.0% |
| **EnergyPlannerPolicy (Ref)** | 100.0% | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 1.0% | 30.00 | -1.1229 | 0.00 | 0.08 | 26.60 | 99.0% | 0.0% |

#### Experiment E1: Combined Shift Solvable (Start (0,1), Target (3,3), Res (1,2), E=3)
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Zero-Shot)** | **100.0%** | **6.00** | **0.9500** | **2.00** | **1.00** | **0.00** | 0.0% | 0.0% |
| **Tabular Q-Learner (Zero-Shot)** | 0.0% | N/A | -0.2500 | 2.00 | 0.00 | 24.00 | 0.0% | 100.0% |
| **EnergyPlannerPolicy (Ref)** | 100.0% | 6.00 | 0.9500 | 2.00 | 1.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 1.0% | 30.00 | -1.1269 | 0.00 | 0.06 | 26.70 | 99.0% | 0.0% |

#### Experiment E2: Combined Shift Unsolvable (Start (0,0), Target (3,3), No Res, E=2)
| Policy | Success Rate | Mean Steps | Mean Reward | Final Energy | Recharges | Rejected | Failure: Exhausted | Failure: Timeout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Q-Learner (Zero-Shot)** | 0.0% | N/A | -0.1700 | 0.00 | 0.00 | 1.00 | 100.0% | 0.0% |
| **Tabular Q-Learner (Zero-Shot)** | 0.0% | N/A | -0.2500 | 2.00 | 0.00 | 25.00 | 0.0% | 100.0% |
| **EnergyPlannerPolicy (Ref)** | 0.0% | N/A | 0.0000 | 2.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| **Random Baseline** | 0.0% | N/A | -1.4200 | 0.00 | 0.00 | 28.00 | 100.0% | 0.0% |

#### Experiment F: Observation Limitations
When target information is omitted (`observable_target=None`), all target features (`target_dist_norm`, `move_closer_target`, `move_away_target`, `reaches_target`, `energy_deficit`) evaluate strictly to 0.0. The agent does not have access to hidden coordinates and cannot navigate to invisible goals.

### Scientific Findings and Limitations

1. **Tabular vs Function-Approximation Generalization:**
   Tabular Q-learning memorizes state-action pairs $(s, a)$. Any change in target coordinates, start coordinates, or resource coordinates creates unvisited state keys, resulting in complete failure (0% success). Linear function approximation maps relative observations to value estimates, achieving zero-shot generalization across held-out targets, starts, and resource layouts.
2. **Spurious Correlations in Single-Task Training:**
   If trained on only a single environment where target and resource lie along the same direction, linear models can develop collinear weights where `move_closer_target` and `move_closer_resource` confound. Interleaved multi-task training across a small set of diverse environments breaks this collinearity, enabling robust resource detouring.
3. **Linearity Boundary:**
   Linear models excel when features capture the key decision boundaries (e.g., `seek_res_when_deficient = move_closer_resource * energy_deficit`). In environments with non-convex obstacles (such as dead-end U-shaped walls), linear features would require higher-order spatial features or multi-step graph search to prevent local minima.

## Generalization Validation and Benchmark Audit (Phase 12.1)

Phase 12.1 provides an independent scientific audit of Phase 12's generalization claims.

### Audit Discoveries and Methodological Corrections

1. **Accounting Bug in Scratch Benchmark:** The exploratory benchmark script previously tallied energy exhaustion by checking `energy == 0 and not (success_rate == 1.0)`. In `worldsim.benchmark`, this was corrected to strict, per-episode mutually exclusive accounting:
   - Success Rate: $\text{episodes where } \text{reached\_target} = \text{True}$.
   - Exhaustion Rate: $\text{episodes where } \text{reached\_target} = \text{False} \text{ and } \text{final\_energy} = 0$.
   - Timeout Rate: $\text{episodes where } \text{reached\_target} = \text{False} \text{ and } \text{final\_energy} > 0$.
   - Strictly satisfies $\text{Success} + \text{Exhaustion} + \text{Timeout} = 100.0\%$.
2. **Fair Training Diversity:** The original comparison trained Linear Q-learning across 3 tasks while Tabular Q-learning was trained on 1 task. Phase 12.1 added `train_tasks` to `LearningRunner` and benchmarked both algorithms under identical single-task and multi-task protocols.
3. **Random Baseline Action Masking:** Clarified and benchmarked both Unmasked Random (sampling uniformly across all 5 actions) and Masked Random (sampling uniformly across legal actions).
4. **Planner Reference Comparison:** Evaluated `EnergyPlannerPolicy` across all 10 held-out configurations individually rather than using a single point-estimate reference.

### Controlled Comparisons Across Training Conditions and Budgets

#### Condition A: Matched Single-Task Training (Trained only on Train-1)
*Evaluated over 50 episodes on Familiar Env (Train-1) and across 8 Feasible Held-out Tasks (20 episodes each).*

| Budget | Tabular Q (Familiar) | Linear Q (Familiar) | Tabular Q (Held-out Avg) | Linear Q (Held-out Avg) |
| :---: | :---: | :---: | :---: | :---: |
| **300 ep** | 100.0% | 100.0% | **0.0%** | **75.0%** |
| **900 ep** | 100.0% | 100.0% | **0.0%** | **75.0%** |
| **1500 ep** | 100.0% | 100.0% | **0.0%** | **87.5%** |

#### Condition B: Matched Multi-Task Training (Trained on [Train-1, Train-2, Train-3])
*Evaluated over 50 episodes on Familiar Env (Train-1) and across 8 Feasible Held-out Tasks (20 episodes each).*

| Budget | Tabular Q (Familiar) | Linear Q (Familiar) | Tabular Q (Held-out Avg) | Linear Q (Held-out Avg) |
| :---: | :---: | :---: | :---: | :---: |
| **300 ep** | 100.0% | 100.0% | **0.0%** | **75.0%** |
| **900 ep** | 100.0% | 100.0% | **0.0%** | **75.0%** |
| **1500 ep** | 100.0% | 100.0% | **0.0%** | **62.5%** |

*Key Scientific Insight:* Tabular Q-learning achieves **0.0% generalization** even when trained across multi-task distributions because every novel coordinate combination produces an unvisited state key hash. Linear Q-learning achieves **62.5%–87.5% zero-shot generalization**.

### 10-Task Final Held-Out Evaluation Breakdown

Evaluated on 10 distinct, held-out configurations (8 feasible, 2 provably infeasible) using models trained under Condition B (900 episodes total):

| Task | Feasible? | Optimal Steps | Linear Q Success | Tabular Q Success | Planner Ref Success | Masked Random Success | Unmasked Random Success | Linear Q Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Test-1: Unseen Target (3,0)** | Yes | 3 | **100.0%** (4.0 steps) | 0.0% | 100.0% (3.0 steps) | 0.0% | 8.0% | None (100% success) |
| **Test-2: Inverted Start (3,3)->(0,0)** | Yes | 7 | **100.0%** (7.0 steps) | 0.0% | 100.0% (7.0 steps) | 0.0% | 0.0% | None (100% success) |
| **Test-3: Unseen Res (1,2)** | Yes | 6 | **100.0%** (6.0 steps) | 0.0% | 0.0%* | 4.0% | 0.0% | None (100% success) |
| **Test-4: Low Energy E0=2, Rech=6** | Yes | 6 | **100.0%** (6.0 steps) | 0.0% | 100.0% (6.0 steps) | 2.0% | 4.0% | None (100% success) |
| **Test-5: Reusable Res (1,1)** | Yes | 8 | **0.0%** | 0.0% | 100.0% (8.0 steps) | 0.0% | 0.0% | 100% Timeout (Recharging Loop) |
| **Test-6: 5x5 Grid Shift** | Yes | 8 | **0.0%** | 0.0% | 0.0%* | 0.0% | 0.0% | 100% Energy Exhaustion |
| **Test-7: High Deficit Detour** | Yes | 7 | **100.0%** (7.0 steps) | 0.0% | 100.0% (7.0 steps) | 0.0% | 0.0% | None (100% success) |
| **Test-8: Multiple Resources** | Yes | 8 | **100.0%** (8.0 steps) | 0.0% | 0.0%* | 0.0% | 0.0% | None (100% success) |
| **Test-9: Infeasible Low Res** | No | N/A | **0.0%** | 0.0% | 0.0% (0 steps) | 0.0% | 0.0% | 100% Energy Exhaustion |
| **Test-10: Infeasible No Res** | No | N/A | **0.0%** | 0.0% | 0.0% (0 steps) | 0.0% | 0.0% | 100% Energy Exhaustion |

*\*Note on Planner in Test-3, Test-6, and Test-8:* In Test-3 and Test-6, resources were outside default sensory radius $r=2.0$ from start, so observation-limited `EnergyPlannerPolicy` executed 0 actions. In Test-8, greedy replanning did not sequence multiple resources.

### Documented Representation Limitations & Failure Modes

1. **Reusable Resource Looping (Test-5):** When a resource is reusable (`consumable=False`), the resource remains present on the cell. Because training only used consumable resources, the linear model never learned to penalize recharging when energy is full, causing it to loop `Action.RECHARGE` until timeout.
2. **Sensory Horizon Blindness (Test-6):** On a larger grid (5x5) where the resource is beyond Euclidean sensory radius 2.0 from the start position ($d = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.24 > 2.0$), the agent cannot observe the resource and exhausts its energy attempting to reach the distant target directly.
3. **Nearest-Resource Feature Aliasing (Counterexample):** `FeatureExtractor` evaluates `move_closer_resource` based strictly on the nearest visible resource. If Resource A is at distance 1 with recharge amount 1 (insufficient to reach the target) and Resource B is at distance 2 with recharge amount 6 (sufficient), the linear features bias the agent toward Resource A, causing fatal energy exhaustion.
4. **Infeasibility vs Proving Infeasibility:** On provably impossible tasks (Tests 9 and 10), the linear agent does not have privileged global planning lookahead to prove infeasibility in advance; it moves toward the target and cleanly exhausts energy without hanging.

### Reproducing the Audit

Run the benchmark suite from the workspace root:

```powershell
python run_benchmark.py
```
or
```powershell
python -m worldsim.benchmark
```
This runs the full audit suite and saves machine-readable JSON metrics to `benchmark_results.json`.

## Multi-Agent Simulation (Phase 13)

Phase 13 introduces simultaneous multi-agent execution, deterministic collision handling, atomic resource competition, cooperative discovery sharing, and comprehensive benchmark evaluation.

### Multi-Agent Configuration and Execution

```python
from worldsim import (
    Action,
    AgentConfig,
    MultiAgentWorldConfig,
    MultiAgentSimulation,
    Position,
    Resource,
)
from worldsim.viewer import MultiAgentWorldViewer

config = MultiAgentWorldConfig(
    width=6,
    height=6,
    agents=(
        AgentConfig(id="agent_1", initial_x=0, initial_y=0, initial_energy=5, target=Position(5, 5)),
        AgentConfig(id="agent_2", initial_x=5, initial_y=0, initial_energy=5, target=Position(0, 5)),
    ),
    resources=(
        Resource(position=Position(2, 2), recharge_amount=6, consumable=False),
    ),
    max_energy=8,
    movement_energy_cost=1,
    cooperative_sharing=True,
    goal_criterion="individual",
)

sim = MultiAgentSimulation(config)
step_result = sim.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})
print("Agent 1 position:", step_result.agent_results["agent_1"].position_after)
```

### Deterministic Conflict Resolution Rules

1. **Vertex Collisions**: If two or more agents propose destinations that evaluate to the same cell $C$, all moving claimants bounce back symmetrically to their respective starting cells (`collision_type="vertex"`).
2. **Swap Collisions**: If Agent $A$ attempts to move to $P_B$ while Agent $B$ attempts to move to $P_A$, both movements are rejected and bounce back to their initial cells (`collision_type="swap"`).
3. **Occupied Collisions**: If Agent $A$ attempts to move to a cell occupied by a stationary Agent $B$, Agent $A$ is blocked and remains in place (`collision_type="occupied"`).
4. **Train / Following Movement**: If Agent $A$ moves to cell $P_B$ while Agent $B$ moves to cell $P_C$ (vacating $P_B$ without conflict), Agent $A$'s movement succeeds.

### Phase 13 Controlled Experiment Benchmark (8 Experiments)

Results from `benchmark_results_phase13.json` (50 episodes per experiment, reproducible via `python run_phase13_benchmark.py`):

| Experiment | Setup | Overall Completion | Per-Agent Success | Collisions (V/S/O) | Jain's Fairness | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp A: Single-Agent Control** | 1 agent baseline | **100.0%** | agent_1: 100.0% | 0 (0/0/0) | 1.00 | Confirms zero regression against Phase 12.2 single-agent baseline. |
| **Exp B: Multiple Independent** | 2 agents, separate goals & res | **100.0%** | agent_1: 100%, agent_2: 100% | 0 (0/0/0) | 1.00 | Parallel collision-free navigation to individual goals. |
| **Exp C: Resource Competition** | 2 agents, 1 consumable resource | **0.0%** | agent_1: 100%, agent_2: 0% | 0 (0/0/0) | 0.50 | Agent 1 reaches resource first; atomic depletion starves Agent 2. |
| **Exp D: Movement Conflicts** | 2 agents, head-on swap collision | **100.0%** | agent_1: 100%, agent_2: 100% | 200 (0/200/0) | 1.00 | Deterministic swap bounce prevents clipping; cycle recovery breaks deadlock. |
| **Exp E: Shared Goals** | 2 agents, shared goal | **100.0%** | agent_1: 0%, agent_2: 100% | 0 (0/0/0) | 0.50 | Any-agent completion: Agent 2 reaches target in 4 steps; episode terminates. |
| **Exp F1: Isolated Exploration** | 2 agents, isolated observations | **0.0%** | agent_1: 100%, agent_2: 0% | 0 (0/0/0) | 0.50 | Agent 2 cannot observe distant resource; fails from energy exhaustion. |
| **Exp F2: Cooperative Sharing** | 2 agents, shared discovery pool | **100.0%** | agent_1: 100%, agent_2: 100% | 0 (0/0/0) | 1.00 | Agent 1 discovers resource; broadcasts to Agent 2; both succeed. |
| **Exp G: Policy Comparison** | Robust Linear Q vs Random | **0.0%** | robust: 100%, random: 0% | 0 (0/0/0) | 0.50 | Robust policy reaches goal in 7 steps; Random policy exhausts energy. |
| **Exp H1: Scaling (1 Agent)** | 1 agent on 8x8 grid | **100.0%** | agent_1: 100% | 0 (0/0/0) | 1.00 | Baseline on 8x8 grid (16 steps). |
| **Exp H2: Scaling (2 Agents)** | 2 agents on 8x8 grid | **100.0%** | all 100% | 30 (0/0/30) | 1.00 | Mild congestion (30 occupied collisions). |
| **Exp H3: Scaling (4 Agents)** | 4 agents on 8x8 grid | **100.0%** | all 100% | 150 (60/0/90) | 1.00 | Moderate congestion; all agents reach goals in 18 steps. |
| **Exp H4: Scaling (8 Agents)** | 8 agents on 8x8 grid | **0.0%** | 4 agents 100%, 4 agents 0% | 2460 (720/0/1740) | 0.50 | High density causes severe bottleneck at central resources. |

### Reproducing the Multi-Agent Benchmark

```powershell
python run_phase13_benchmark.py
```
or
```powershell
python -m worldsim.multi_agent_experiment
```






