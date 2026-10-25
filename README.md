# WorldSim

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-279%20passed-success.svg?style=flat-square)](tests/)
[![Architecture](https://img.shields.io/badge/Dependencies-Zero%20External-orange.svg?style=flat-square)](worldsim/)
[![Viewer](https://img.shields.io/badge/Live%20Viewer-HTML5%20Canvas%202D-green.svg?style=flat-square)](run_viewer.py)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg?style=flat-square)](LICENSE)

**WorldSim** is an open-source, deterministic multi-agent simulation, multi-agent pathfinding (MAPF), and decentralized reinforcement learning research platform. It investigates how autonomous agents coordinate, navigate tight bottlenecks, budget constrained energy, and avoid dynamic hazards under **strict partial observability ($R \le 2.0$)** with short-range communicative intent.

Built with **zero third-party runtime dependencies** using the Python Standard Library, WorldSim features a real-time, interactive 2D web viewer (HTML5 Canvas), formal train/validation/test scenario isolation, a standard **Conflict-Based Search (CBS)** MAPF baseline, and a rigorous statistical benchmark reporting Wilson score confidence intervals across procedurally generated held-out environments.

---

## Table of Contents

- [Overview & Key Features](#overview--key-features)
- [Held-Out Generalization & MAPF Baselines](#held-out-generalization--mapf-baselines)
- [Live 2D Interactive Web Viewer](#live-2d-interactive-web-viewer)
- [System Architecture & Physics Engine](#system-architecture--physics-engine)
- [Coordination Policies](#coordination-policies)
- [Repository Organization](#repository-organization)
- [Quickstart & CLI Usage](#quickstart--cli-usage)
- [Cloud Hosting & Deployment](#cloud-hosting--deployment)
- [Research History & Documentation](#research-history--documentation)

---

## Overview & Key Features

* **Bit-for-Bit Determinism & Fingerprinting:** Movement resolution, collision handling, dynamic obstacle trajectories, and resource allocation are fully deterministic. Environments are fingerprinted with canonical SHA-256 hashes to guarantee genuine topological variation across seeds.
* **Strict Scenario Isolation (`Split.TEST`):** Genuinely held-out test scenarios are procedurally generated and isolated from training and hyperparameter tuning by hardcoded registry guards.
* **Conflict-Based Search (CBS) Baseline:** A zero-dependency Sharon et al. (2012/2015) MAPF solver featuring low-level Space-Time A*, high-level constraint tree branching, and dynamic obstacle scheduling.
* **Strict Partial Observability:** Agents make decisions using local observations within a localized Euclidean radius ($R \le 2.0$). Global maps and distant agent plans are invisible.
* **Decentralized Learning & Communicative Intent:** Peer agents exchange short-range intention vectors within sensory range ($R \le 2.0$) to negotiate bottlenecks and prevent collision deadlocks.
* **Dynamic Moving Hazards:** Moving obstacles with velocity vectors and predictive bounce dynamics navigate the grid; agents anticipate trajectories and replan dynamically.
* **Energy Economics & Recharging:** Agents must manage finite battery budgets, evaluate recharge station utilities, and navigate without suffering terminal exhaustion.
* **Statistical Rigor:** All binomial completion metrics feature exact Wilson score 95% confidence intervals, and step statistics are conditioned on successful episodes without hardcoded values.
* **Zero Runtime Dependencies:** The simulation engine, reinforcement learning agents, CBS solver, and web viewer run on vanilla Python (`http.server`, `math`, `heapq`, `dataclasses`, `json`, `hashlib`).

---

## Held-Out Generalization & MAPF Baselines

WorldSim implements a formal scenario split (`Split.TRAIN`, `Split.VAL`, `Split.TEST`). The held-out benchmark evaluates policies on 6 structurally novel procedural environments across 10 evaluation seeds (360 total episodes):

| Scenario ID | Name | Topology & Dynamic Features |
|---|---|---|
| `TEST_H01` | **Asymmetric Funnel Bottleneck** | Wide chamber funnels into constriction corridor ($W \in \{1, 2\}$) |
| `TEST_H02` | **Double-Chamber Doorway Swap** | Two rooms connected by a single doorway with asymmetric starting energies |
| `TEST_H03` | **Procedural Connected Maze** | Randomized obstacle fields with BFS-verified 4-corner reachability |
| `TEST_H04` | **Dynamic Hazard Highway** | Crossing two randomized counter-flowing patrol lanes |
| `TEST_H05` | **Three-Way T-Junction** | 3 agents converging simultaneously at an orthogonal 1-cell junction |
| `TEST_H06` | **Critical Recharge Alcove Chokepoint** | Energy budget requires dead-end alcove recharge detour to reach target |

Full empirical findings, confidence intervals, and raw records are in [`reports/HELD_OUT_BENCHMARK_REPORT.md`](reports/HELD_OUT_BENCHMARK_REPORT.md) and [`benchmarks/held_out_generalization_results.json`](benchmarks/held_out_generalization_results.json).

### Overall Held-Out Benchmark Results (10 Seeds per Scenario, 360 Episodes)

| Policy | Information Boundary | Overall Completion (95% CI) | Agent Success | Exhaustions | Timeouts | Mean Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **`hybrid_comm`** | Local Horizon ($R \le 2.0$) + Safety Guard | **98.3%** [91.1%, 99.7%] | **98.4%** | 1 | 0 | 6.4ms |
| **`cbs`** (MAPF) | Offline Space-Time A* + Constraint Tree | **83.3%** [72.0%, 90.7%] | 89.5% | 10 | 0 | 33.8ms |
| **`centralized`** | Omniscient Centralized Arbiter | **73.3%** [61.0%, 82.9%] | 82.6% | 0 | 16 | 14.9ms |
| **`heuristic`** | Robust BFS Graph Search | **73.3%** [61.0%, 82.9%] | 82.6% | 0 | 16 | 14.9ms |
| **`learned_comm`** | Decentralized Q-Learner + Intent Vectors | **70.0%** [57.5%, 80.1%] | 84.2% | 18 | 0 | 5.3ms |
| **`learned_no_comm`** | Decentralized Q-Learner (No Comm) | **16.7%** [9.3%, 28.0%] | 29.5% | 2 | 48 | 22.0ms |

### Key Scientific Insights
1. **Classical MAPF vs. Energy Constraints:** Standard CBS excels at geometric collision avoidance (100% on `TEST_H01`–`TEST_H05`), but fails on `TEST_H06` (0% completion, 10/10 exhaustions). Because classical MAPF assumes unconstrained energy, CBS selects the direct 7-step path without detouring to the recharge alcove. WorldSim's `hybrid_comm` recognizes battery exhaustion risk and achieves 100% completion by queuing at the recharge alcove.
2. **Doorway Swap Deadlocks:** In tight single-doorway swaps (`TEST_H02`), greedy heuristic and centralized arbiters suffer reciprocal blocking (0% completion). CBS (100%) and `learned_comm` / `hybrid_comm` (100%) successfully coordinate sidesteps and intent yielding.
3. **The Essential Role of Communication:** In narrow bottlenecks and dynamic highways, removing communicative intent (`learned_no_comm`) collapses performance to 16.7% due to endless step timeouts and head-on collisions.

---

## Live 2D Interactive Web Viewer

WorldSim includes an integrated, zero-dependency browser interface driven by an authoritative Python simulation session:

```
+-------------------------------------------------------------------------------+
|  WorldSim Live 2D Viewer | Scenario: [ D01 v ] | Policy: [ hybrid_comm v ]   |
+------------------------------------------------------+------------------------+
|                                                      | Agent Inspector        |
|   (0,0)                                              | ---------------------- |
|     +---+---+---+---+---+---+                        | ID: agent_1            |
|     | . | . | R | . | . | . |                        | Status: active         |
|     +---+---+---+---+---+---+                        | Pos: (3, 4)            |
|     | . |(A1)-->| . |[#]| . |  [#] Dynamic Hazard    | Energy: 7 / 10 [|||||] |
|     +---+---+---+---+---+---+                        | Intent: (3, 5) [DOWN]  |
|     | . | . | : | . | . | . |  (A1) Active Agent     | Visible Peers: agent_2 |
|     +---+---+---+---+---+---+                        | ---------------------- |
|     | . | . |(A2)<--| . | G |  (R)  Recharge Station | Telemetry              |
|     +---+---+---+---+---+---+                        | Step: 14  | Tick: 14   |
|                                (G)  Target Goal      | Collisions: 0          |
|   [< Play/Pause] [>| Step] [R Reset] Speed: [===|==] | Deadlocks: 0           |
+------------------------------------------------------+------------------------+
```

### Viewer Highlights
* **Authoritative Simulation Loop:** The browser never simulates physics; it queries Python REST endpoints (`/api/state`, `/api/step`, `/api/reset`, `/api/select`).
* **Interactive Agent Inspector:** Click any agent token on the grid to inspect its real-time energy, intent target, visible peers, and active memory.
* **Toggleable Debug Overlays:** Sensory horizon circles ($R \le 2.0$), peer-to-peer intent rays, dynamic hazard velocity vectors, breadcrumb trails, and conflict markers.

### Launching the Web Viewer
```bash
# Start viewer locally (opens http://localhost:8000 automatically)
python run_viewer.py

# Launch with custom scenario, policy, and port:
python run_viewer.py --scenario U14 --policy hybrid_comm --port 8080
```

#### Keyboard Shortcuts
| Key | Action | Key | Action |
| :---: | :--- | :---: | :--- |
| <kbd>Space</kbd> | Toggle Play / Pause | <kbd>H</kbd> | Toggle Sensory Horizon ($R \le 2.0$) |
| <kbd>N</kbd> / <kbd>→</kbd> | Single Step Forward | <kbd>I</kbd> | Toggle Intent Vectors |
| <kbd>R</kbd> | Reset Scenario | <kbd>O</kbd> | Toggle Dynamic Hazard Paths |
| <kbd>T</kbd> | Toggle Breadcrumb Trails | <kbd>C</kbd> | Toggle Conflict / Deadlock Markers |

---

## Empirical Multi-Seed Generalization Benchmark

To evaluate policy robustness and generalization, models were benchmarked across **6 diverse challenge environments** using **20 independent random seeds per scenario** (120 evaluation episodes per policy).

Full results and methodology are documented in [`reports/GENERALIZATION_STRESS_REPORT.md`](reports/GENERALIZATION_STRESS_REPORT.md) and [`benchmarks/multi_seed_generalization_results.json`](benchmarks/multi_seed_generalization_results.json).

### Comparative Performance Matrix (20 Seeds per Scenario)

| Scenario | Environment Topology & Challenge | `centralized` (Omniscient) | `hybrid_comm` (Learned + Safety) | `heuristic` (Robust BFS) | `learned_comm` (Decentralized + Intent) | `learned_no_comm` (No Comm Baseline) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D01** | Dual Patrol Crossing (Moving Hazards) | **100%** (14.0s) | **100%** (11.0s) | **100%** (14.0s) | **100%** (11.0s) | **0%** *(92 cols)* |
| **D03** | Dynamic Corridor Mesh (4 Moving Hazards) | **100%** (13.0s) | **100%** (18.0s) | **100%** (13.0s) | **100%** (18.0s) | **0%** *(188 cols)* |
| **U01** | Asymmetric Diagonal Crossing | 0% *(Deadlock)* | **100%** (11.0s) | 0% *(Deadlock)* | **100%** (11.0s) | 0% *(Deadlock)* |
| **U04** | Multi-Resource Separated Maze | **100%** (2.0s) | **100%** (2.0s) | **100%** (2.0s) | **100%** (2.0s) | 0% *(100 cols)* |
| **U06** | Severe 8-Agent Asymmetric Bottleneck | **100%** (19.0s) | 0% *(87.5% ind)* | **100%** (19.0s) | 0% *(Exhaustion)* | 0% *(Exhaustion)* |
| **U14** | Narrow Corridor Swap (Reusable Station) | 0% *(Deadlock)* | 0% *(50% ind)* | 0% *(Deadlock)* | 0% *(50% ind)* | 0% *(Deadlock)* |

### Scientific Insights & Failure Mode Analysis
1. **The Decisive Value of Communicative Intent:** In dynamic hazard scenarios (`D01`, `D03`), sharing short-range intent ($R \le 2.0$) elevated completion from **0% to 100%**, eliminating repeated vertex collisions.
2. **Where Reactive Learning Outperforms Heuristic Search:** In asymmetric crossing topologies (`U01`), both rule-based heuristic and centralized coordinators deadlocked in symmetric swaps. `learned_comm` resolved the deadlock dynamically in 11.0 steps.
3. **Where Planning Excels Over Reactive Models:** In dense 8-agent bottlenecks (`U06`), long-horizon energy management requires multi-step lookahead; heuristic graph search achieved 100% completion while pure reactive learning suffered premature energy exhaustion.

---

## System Architecture & Physics Engine

### Simultaneous 5-Phase Tick Model

Every simulation tick executes an atomic 5-phase lifecycle:
```mermaid
flowchart LR
    A["1. Perception Freeze\n(Local Horizon R<=2.0)"] --> B["2. Intent Formulation\n(Policy Action Choice)"]
    B --> C["3. Conflict Resolution\n(Authoritative Arbitration)"]
    C --> D["4. State Execution\n(Movement & Obstacles)"]
    D --> E["5. Resource Update\n(Depletion & Status)"]
```

### Deterministic Collision Rules
* **Vertex Collisions:** When two or more agents target the same cell in the same tick, moving claimants bounce back to their origins, or rotate right-of-way via tick-based priority.
* **Swap Collisions:** When two agents attempt to swap positions across an edge, both bounce back to prevent clipping.
* **Occupied Collisions:** Attempting to enter a cell occupied by a stationary or paused agent is blocked cost-free without energy penalty.
* **Goal-Cell Yielding:** Agents that reach their destination (`reached_target`) remain at their goal but yield right-of-way, allowing trailing active agents to traverse corridor bottlenecks.

---

## Coordination Policies

| Policy | Description | Information Boundary | Strengths |
| :--- | :--- | :---: | :--- |
| **`hybrid_comm`** *(Default)* | Learned Q-policy with safe recharge/detour arbitration and local intent communication | Local Horizon ($R \le 2.0$) | High safety, zero collisions, balances learned speed with safety guards. |
| **`cbs`** *(MAPF Baseline)* | Conflict-Based Search (Sharon et al. 2012) with Space-Time A* and constraint tree | Global Static Map + Dynamic Schedule | Optimal offline trajectories on static graphs; zero collisions. |
| **`learned_comm`** | Pure decentralized Q-learner using local peer intent vectors | Local Horizon ($R \le 2.0$) | Fast learned execution; excels at dynamic obstacle evasion and corridor swaps. |
| **`learned_no_comm`** | Non-communicative decentralized Q-learner | Local Sensory Only | Experimental control baseline measuring the empirical value of communication. |
| **`heuristic`** | Rule-based `RobustResourcePolicy` with BFS corridor routing and obstacle memory TTL | Local Sensory Only | Deterministic graph search; excels at complex multi-step energy routing. |
| **`centralized`** | Omniscient centralized multi-agent coordinator with global reservation table | Global Privileged | Upper-bound reference baseline. |

---

## Repository Organization

The repository is structured into modular packages, isolating experiments, benchmarks, and historical documentation from the core runtime engine:

```
SimWorld/
├── worldsim/                 # Core Python simulation engine (Zero External Dependencies)
│   ├── actions.py            # Movement and recharge action definitions
│   ├── cbs.py                # Conflict-Based Search (CBS) MAPF solver & Space-Time A*
│   ├── config.py             # World and simulation configurations
│   ├── coordination.py       # Multi-agent arbitration, deadlock detection & priority rules
│   ├── decentralized_learning.py # Q-learning models, feature extractors & hybrid policy
│   ├── dynamic_obstacles.py  # Moving hazard physics & trajectory prediction
│   ├── energy.py             # Energy management & recharge validation
│   ├── fingerprint.py        # Cryptographic SHA-256 canonical environment fingerprinting
│   ├── models.py             # Immutable Position, Resource, and Observation models
│   ├── multi_agent.py        # Simultaneous 5-phase simulation engine
│   ├── robust_policy.py      # BFS path planning, obstacle memory TTL & cycle breaker
│   ├── scenario_generator.py # Procedural held-out environment generator across 6 families
│   ├── scenario_registry.py  # Train / Validation / Test split isolation registry
│   ├── statistics.py         # Wilson score 95% CIs & continuous metric aggregations
│   ├── web_viewer.py         # Authoritative REST API & LiveViewerServer
│   └── static/
│       └── index.html        # High-DPI HTML5 Canvas live visualization SPA
│
├── experiments/              # Evaluation harnesses & benchmark runners
│   ├── eval_heldout.py       # Genuinely held-out test evaluation runner (Split.TEST)
│   ├── eval_generalization.py# Multi-seed empirical generalization benchmark
│   └── ...
│
├── benchmarks/               # Machine-readable JSON evaluation datasets & model weights
│   ├── held_out_generalization_results.json # Raw 360-episode records & statistical summaries
│   ├── multi_seed_generalization_results.json
│   └── ...
│
├── reports/                  # Markdown audit and verification reports
│   ├── HELD_OUT_BENCHMARK_REPORT.md # Formal held-out test split & CBS baseline report
│   ├── GENERALIZATION_STRESS_REPORT.md
│   └── ...
│
├── tests/                    # Comprehensive automated test suite (279 unit & integration tests)
├── docs/                     # Documentation archive & historical research audits
│   └── RESEARCH_HISTORY.md   # Chronological research logs (Phases 1 through 15)
│
├── run_viewer.py             # CLI launcher for the Live 2D Web Viewer
├── Dockerfile                # Production container deployment configuration
├── Procfile                  # Cloud web service process declaration (Render / Railway)
└── README.md
```

---

## Quickstart & CLI Usage

### Prerequisites
* **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13 tested).
* No external libraries required to run the simulation, CBS solver, or live viewer.

### 1. Launch the Live 2D Web Viewer
```bash
python run_viewer.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### 2. Run the Full Test Suite
Execute all 279 automated unit and integration tests:
```bash
python -m pytest
```
*Expected output: `279 passed in ~10s`.*

### 3. Run the Held-Out Generalization & MAPF Benchmark
Execute the complete held-out evaluation across 6 novel scenarios, 6 policies, and 10 evaluation seeds (360 episodes) with Wilson confidence intervals:
```bash
python -m experiments.eval_heldout --seeds 10
```

### 4. Run the Validation Benchmark
Reproduce the multi-seed empirical comparison across validation scenarios:
```bash
python -m experiments.eval_generalization
```

---

## Cloud Hosting & Deployment

WorldSim's viewer is a stateful simulation and can be deployed with one click to persistent web hosts:

### Deploying to Render.com (Recommended Free Hosting)
1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New +** → **Web Service**.
3. Select your repository (`WorldSim`).
4. Render detects the included [`Procfile`](Procfile) and [`Dockerfile`](Dockerfile) automatically.
5. Set **Start Command** (if manual):
   ```bash
   python run_viewer.py --host 0.0.0.0 --port $PORT --no-browser
   ```
6. Click **Create Web Service**. Render provides a free public HTTPS domain.

### Deploying via Docker
```bash
# Build Docker image
docker build -t worldsim .

# Run container mapping port 8000
docker run -p 8000:8000 worldsim
```

---

## Research History & Documentation

For detailed mathematical proofs, diagnostic logs, and phase-by-phase development audits from Phase 1.0 to Phase 15.0, refer to the documentation archive:
* [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md): Detailed phase-by-phase scientific audit archive.
* [`reports/GENERALIZATION_STRESS_REPORT.md`](reports/GENERALIZATION_STRESS_REPORT.md): Multi-seed statistical benchmark report.
* [`reports/PHASE15_VERIFICATION_REPORT.md`](reports/PHASE15_VERIFICATION_REPORT.md): End-to-end verification and headless browser audit.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
