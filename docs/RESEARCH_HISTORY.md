# WorldSim: Research History & Phase-by-Phase Audit Archive

This document compiles the historical research logs, engineering milestones, mathematical proofs, and diagnostic audits conducted across WorldSim's development lifecycle (Phases 1 through 15).

---

## Phase 15.0: Live 2D World Viewer & Authoritative Simulation Interface

### Objective & Architecture
Build an authoritative, real-time 2D web viewer in Python without third-party web frameworks, providing sub-second position interpolation, interactive agent inspection, and full simulation playback controls.

1. **SimulationSession & REST API (`worldsim/web_viewer.py`):**
   - Implemented `SimulationSession` wrapping `MultiAgentSimulation` with thread-safe lock management.
   - Built zero-dependency HTTP server subclassing `http.server.ThreadingHTTPServer`.
   - Exposed REST API endpoints:
     - `GET /api/state`: Returns tick, scenario metadata, metrics, and agent snapshots.
     - `POST /api/step`: Advances simulation by one authoritative tick.
     - `POST /api/reset`: Resets simulation to tick 0 with clean seed.
     - `POST /api/select`: Dynamically loads any scenario (`D01`–`D04`, `U01`–`U16`, `B01`–`B02`) and policy (`hybrid_comm`, `remediated_comm`, `learned_comm`, `learned_no_comm`, `heuristic`, `centralized`).
     - `GET /api/health`: Health probe reporting server status.
2. **Frontend Architecture (`worldsim/static/index.html`):**
   - High-DPI HTML5 Canvas visualization with 60 FPS requestAnimationFrame loop.
   - Interactive Agent Inspector displaying real-time coordinates, energy bars, intent vectors, visible peers, and active memory.
   - Toggleable debug overlays: Breadcrumb trails, Sensory horizon ($R \le 2.0$), Intent vectors, Dynamic obstacle paths, Conflict markers.
3. **Verification & Audit:**
   - 252 automated pytest tests passing.
   - Headless Selenium browser automation verifying zero console errors, input field protection, and complete keyboard navigation.

---

## Phase 14.4: Decentralized Learning & Communicative Intent

### Objective
Introduce decentralized Q-learning under strict partial observability with opt-in local communicative intent ($R \le 2.0$).

1. **Problem Formulation:**
   - State space features: relative target distance/direction, normalized energy deficit, local peer intentions ($R \le 2.0$), and dynamic hazard proximity.
   - Action space: `UP`, `DOWN`, `LEFT`, `RIGHT`, `RECHARGE`, `STAY`.
2. **Key Findings:**
   - Communicative intent broadcast completely resolves vertex collision locks in dynamic patrol corridors (`D01`, `D03`), elevating completion from 0% to 100%.
   - In tight resource bottlenecks (`U14`), pure learned policies require safety arbitration to resolve doorstep contention, leading to the creation of `HybridCoordinationPolicy`.

---

## Phase 14.3: Dynamic Obstacle Avoidance

### Objective
Introduce moving hazard obstacles with linear bouncing trajectories and velocity-aware local avoidance.

1. **Mechanics & Invariants:**
   - Configurable moving obstacles with integer velocities `(vx, vy)`.
   - Engine enforces non-overlapping movement: dynamic obstacles bounce off walls and grid boundaries without overlapping agents or recharge pads.
   - Velocity-aware prediction: `predict_next_obstacle_state(obs, width, height, walls)` enables lookahead collision avoidance.

---

## Phase 14.2: Local-Horizon Coordination & Memory TTL

### Objective
Enforce strict sensory boundaries ($R \le 2.0$) on intention sharing and obstacle memory.

1. **Information Isolation:**
   - Peer intentions and obstacle detections are restricted to Euclidean distance $d \le 2.0$.
   - Global coordinator data structures are shielded from agent policy access.
2. **Memory TTL & Visual Revalidation:**
   - Local obstacle memory expires after bounded TTL ($15$ ticks) if not re-observed.
   - When an obstacle cell is re-observed within sensory range and found empty, it is purged immediately.

---

## Phase 13.0–13.7: Multi-Agent Scaling & Resource Fairness

1. **Simultaneous 5-Phase Tick Model:**
   - Perception freeze, candidate intent formulation, deterministic conflict resolution, atomic state update, and resource interaction.
2. **Deterministic Conflict Rules:**
   - Vertex collision bounce, swap collision bounce, occupied stationary blocking, and following (train) movement.
3. **Completed Agent Goal-Cell Yielding:**
   - Resting agents that reached their target yield right-of-way, allowing trailing active agents to traverse corridor bottlenecks.
4. **Fairness Arbitration:**
   - `ResourceArbitrator` with `fair_balanced` rule achieves a Jain Fairness Index of 1.000 across asymmetric claimant conditions.

---

## Phases 1.0–12.2: Foundations & Generalization

1. **Phase 12: Linear Function Approximation:**
   - Replaced tabular Q-learning with relative feature extraction (`LinearQLearner`), achieving zero-shot transfer across unseen targets and starts.
2. **Phase 11: Energy & Resource Reinforcement Learning:**
   - Integrated energy economics and consumable/reusable stations into trial-and-error navigation.
3. **Phase 10: State-Space BFS Planning:**
   - Formalized optimal planning across $(x, y, \text{energy}, \text{resources})$ state spaces.
4. **Phases 1–9: Deterministic Grid Physics:**
   - Built core immutable coordinate models, experience tuples, and observation mechanics.
