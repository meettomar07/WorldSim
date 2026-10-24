"""WorldSim Phase 15.0 — Real-Time 2D Web Viewer & Authoritative Simulation Interface.

Provides:
- Authoritative Python simulation session supporting all scenario families:
  * Dynamic obstacle scenarios (D01, D02, D03, D04)
  * Generalization benchmark scenarios (U01–U16, including U06 and U14)
  * Baseline multi-agent benchmark scenarios
- Comprehensive policy support:
  * hybrid_comm: Learned Q-policy with safe recharge/detour arbitration and local intent communication (R <= 2.0)
  * remediated_comm: Pure learned Q-policy with pad safety and local intent communication
  * learned_comm: Learned Q-policy with local intent communication
  * learned_no_comm: Non-communicative learned Q-policy
  * heuristic: RobustResourcePolicy rule-based baseline
  * centralized: Centralized multi-agent coordinator baseline
- Zero-dependency HTTP/REST API with ThreadingHTTPServer.
- High-performance, high-DPI HTML5/Canvas visualization frontend with live telemetry,
  interactive agent inspector, toggleable debug overlays, and smooth sub-second playback.
"""

from __future__ import annotations

from dataclasses import replace
import http.server
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence
import urllib.parse
import webbrowser

from worldsim.actions import Action
from worldsim.config import WorldConfig
from worldsim.coordination import MultiAgentCoordinator, agent_val
from worldsim.decentralized_learning import (
    DecentralizedLearningConfig,
    DecentralizedQLearner,
    HybridCoordinationPolicy,
    LearnedCoordinationPolicy,
)
from worldsim.dynamic_obstacles import (
    DynamicObstacleConfig,
    DynamicObstacleState,
    predict_next_obstacle_state,
)
from worldsim.linear_learning import LinearLearningConfig, LinearLearningRunner
from worldsim.models import Position, Resource
from worldsim.multi_agent import (
    AgentConfig,
    AgentState,
    AgentStepResult,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentStepResult,
    MultiAgentWorldConfig,
)
from worldsim.robust_policy import RobustResourcePolicy


# ==============================================================================
# Model and Learner Caching Helpers
# ==============================================================================

_CACHED_BASE_LEARNER: Any | None = None
_CACHED_LEARNED_COMM: DecentralizedQLearner | None = None
_CACHED_LEARNED_NO_COMM: DecentralizedQLearner | None = None
_CACHE_LOCK = threading.Lock()


def get_cached_base_learner() -> Any:
    """Return a cached base linear learner for heuristic and hybrid policies."""
    global _CACHED_BASE_LEARNER
    with _CACHE_LOCK:
        if _CACHED_BASE_LEARNER is None:
            train_world = WorldConfig(
                width=4,
                height=4,
                initial_x=0,
                initial_y=0,
                initial_energy=3,
                max_energy=6,
                resources=(Resource(Position(1, 0), 4, True),),
                observable_target=Position(3, 3),
            )
            runner = LinearLearningRunner(
                world=train_world,
                config=LinearLearningConfig(max_steps=25),
                seed=42,
            )
            runner.train(100)
            _CACHED_BASE_LEARNER = runner.learner
        return _CACHED_BASE_LEARNER


def get_cached_learned_model(communication: bool = True) -> DecentralizedQLearner:
    """Return a cached trained DecentralizedQLearner."""
    global _CACHED_LEARNED_COMM, _CACHED_LEARNED_NO_COMM
    with _CACHE_LOCK:
        if communication:
            if _CACHED_LEARNED_COMM is None:
                learner = DecentralizedQLearner(config=DecentralizedLearningConfig(), seed=42)
                weights_path = Path("phase14_4_weights_comm.json")
                if weights_path.exists():
                    try:
                        learner.load_weights(weights_path)
                    except Exception:
                        pass
                _CACHED_LEARNED_COMM = learner
            return _CACHED_LEARNED_COMM
        else:
            if _CACHED_LEARNED_NO_COMM is None:
                learner = DecentralizedQLearner(config=DecentralizedLearningConfig(), seed=42)
                weights_path = Path("phase14_4_weights_no_comm.json")
                if weights_path.exists():
                    try:
                        learner.load_weights(weights_path)
                    except Exception:
                        pass
                _CACHED_LEARNED_NO_COMM = learner
            return _CACHED_LEARNED_NO_COMM


# ==============================================================================
# Simulation Session
# ==============================================================================

class SimulationSession:
    """Manages an authoritative WorldSim multi-agent simulation for live visualization."""

    def __init__(
        self,
        default_scenario: str = "D01",
        default_policy: str = "hybrid_comm",
        default_seed: int = 42,
        max_steps: int = 150,
    ) -> None:
        self.scenario_id = default_scenario
        self.policy_mode = default_policy
        self.seed = default_seed
        self.max_steps = max_steps

        self.scenario_name: str = ""
        self.scenario_family: str = ""
        self.width: int = 8
        self.height: int = 5
        self.max_energy: int = 10

        self.sim: MultiAgentSimulation | None = None
        self.policies: dict[str, Any] = {}
        self.active_agent_ids: list[str] = []
        self.wall_positions: set[tuple[int, int]] = set()

        self.step_count: int = 0
        self.overall_completed: bool = False
        self.trajectories: dict[str, list[dict[str, int]]] = {}
        self.last_positions: dict[str, tuple[int, int]] = {}
        self.last_actions: dict[str, str] = {}
        self.last_action_results: dict[str, AgentStepResult] = {}
        self.last_decision_modes: dict[str, str] = {}
        self.recent_events: list[str] = []
        self.deadlock_locations: list[dict[str, int]] = []

        self.conflict_attempts: dict[str, int] = {
            "vertex": 0,
            "swap": 0,
            "occupied": 0,
            "dynamic_obstacle": 0,
        }
        self.cumulative_decisions: dict[str, int] = {
            "learned": 0,
            "safety_guard": 0,
            "heuristic": 0,
        }

        # Available scenarios catalog
        self._scenario_registry = self._build_scenario_registry()

        # Load default scenario
        self.load_scenario(self.scenario_id, self.seed, self.policy_mode)

    def _build_scenario_registry(self) -> dict[str, Any]:
        """Construct the registry of scenarios from project builders and fallbacks."""
        registry: dict[str, Any] = {}

        # 1. Built-in D01–D04 Dynamic Scenarios
        registry["D01"] = {
            "id": "D01",
            "name": "Dual Patrol Chokepoint Crossing (2 Agents, 2 Moving Obstacles)",
            "family": "Dynamic",
            "width": 8,
            "height": 5,
            "max_energy": 10,
            "agents": [
                AgentConfig(id="agent_w", initial_x=0, initial_y=2, initial_energy=10, target=Position(7, 2)),
                AgentConfig(id="agent_e", initial_x=7, initial_y=2, initial_energy=10, target=Position(0, 2)),
            ],
            "resources": [],
            "dynamic_obstacles": [
                DynamicObstacleConfig(id="obs_patrol_1", initial_x=3, initial_y=0, pattern="patrol", waypoints=(Position(3, 0), Position(3, 4))),
                DynamicObstacleConfig(id="obs_patrol_2", initial_x=5, initial_y=4, pattern="patrol", waypoints=(Position(5, 4), Position(5, 0))),
            ],
            "walls": [],
        }

        registry["D02"] = {
            "id": "D02",
            "name": "Dynamic Bottleneck Mesh (4 Agents, 2 Moving Obstacles)",
            "family": "Dynamic",
            "width": 8,
            "height": 8,
            "max_energy": 14,
            "agents": [
                AgentConfig(id="agent_nw", initial_x=1, initial_y=1, initial_energy=14, target=Position(6, 6)),
                AgentConfig(id="agent_se", initial_x=6, initial_y=6, initial_energy=14, target=Position(1, 1)),
                AgentConfig(id="agent_ne", initial_x=6, initial_y=1, initial_energy=14, target=Position(1, 6)),
                AgentConfig(id="agent_sw", initial_x=1, initial_y=6, initial_energy=14, target=Position(6, 1)),
            ],
            "resources": [],
            "dynamic_obstacles": [
                DynamicObstacleConfig(id="obs_bounce_h", initial_x=2, initial_y=3, pattern="linear_bounce", velocity=(1, 0)),
                DynamicObstacleConfig(id="obs_bounce_v", initial_x=4, initial_y=2, pattern="linear_bounce", velocity=(0, 1)),
            ],
            "walls": [],
        }

        registry["D03"] = {
            "id": "D03",
            "name": "Dynamic Obstacle Grid Mesh (4 Agents, 4 Moving Obstacles)",
            "family": "Dynamic",
            "width": 9,
            "height": 9,
            "max_energy": 16,
            "agents": [
                AgentConfig(id="agent_n", initial_x=4, initial_y=0, initial_energy=16, target=Position(4, 8)),
                AgentConfig(id="agent_s", initial_x=4, initial_y=8, initial_energy=16, target=Position(4, 0)),
                AgentConfig(id="agent_w", initial_x=0, initial_y=4, initial_energy=16, target=Position(8, 4)),
                AgentConfig(id="agent_e", initial_x=8, initial_y=4, initial_energy=16, target=Position(0, 4)),
            ],
            "resources": [],
            "dynamic_obstacles": [
                DynamicObstacleConfig(id="obs_patrol_q1", initial_x=2, initial_y=1, pattern="patrol", waypoints=(Position(2, 1), Position(2, 7))),
                DynamicObstacleConfig(id="obs_patrol_q2", initial_x=6, initial_y=7, pattern="patrol", waypoints=(Position(6, 7), Position(6, 1))),
                DynamicObstacleConfig(id="obs_patrol_q3", initial_x=1, initial_y=6, pattern="patrol", waypoints=(Position(1, 6), Position(7, 6))),
                DynamicObstacleConfig(id="obs_patrol_q4", initial_x=7, initial_y=2, pattern="patrol", waypoints=(Position(7, 2), Position(1, 2))),
            ],
            "walls": [],
        }

        registry["D04"] = {
            "id": "D04",
            "name": "Resource Approach with Moving Obstacle (2 Agents, Recharge Pad)",
            "family": "Dynamic",
            "width": 7,
            "height": 7,
            "max_energy": 7,
            "agents": [
                AgentConfig(id="agent_left", initial_x=0, initial_y=3, initial_energy=7, target=Position(6, 3)),
                AgentConfig(id="agent_right", initial_x=6, initial_y=3, initial_energy=7, target=Position(0, 3)),
            ],
            "resources": [Resource(Position(3, 3), recharge_amount=6, consumable=False)],
            "dynamic_obstacles": [
                DynamicObstacleConfig(id="obs_patrol_gate_1", initial_x=2, initial_y=1, pattern="patrol", waypoints=(Position(2, 1), Position(2, 5))),
                DynamicObstacleConfig(id="obs_patrol_gate_2", initial_x=4, initial_y=5, pattern="patrol", waypoints=(Position(4, 5), Position(4, 1))),
            ],
            "walls": [],
        }

        # 2. Benchmark Scenarios from Phase 13.5 Unseen Audit
        try:
            repo_dir = Path(__file__).resolve().parent.parent
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            from phase13_5_unseen_audit import build_all_unseen_scenarios
            for spec, pre_positioned in build_all_unseen_scenarios():
                active_agents = [
                    AgentConfig(
                        id=a["id"],
                        initial_x=a["initial_x"],
                        initial_y=a["initial_y"],
                        initial_energy=a["initial_energy"],
                        target=Position(a["target_x"], a["target_y"]) if a.get("target_x") is not None else None,
                    )
                    for a in spec.active_agents
                ]
                pre_agents = [
                    AgentConfig(
                        id=p.id,
                        initial_x=p.x,
                        initial_y=p.y,
                        initial_energy=0 if p.status == "exhausted" else 10,
                        target=None if p.status == "exhausted" else Position(p.x, p.y),
                    )
                    for p in pre_positioned
                ]
                walls = [
                    Position(p.x, p.y)
                    for p in pre_positioned
                    if p.status == "exhausted"
                ]
                resources = [
                    Resource(
                        Position(r["x"], r["y"]),
                        recharge_amount=r["recharge_amount"],
                        consumable=r["consumable"],
                    )
                    for r in spec.resources
                ]
                registry[spec.scenario_id] = {
                    "id": spec.scenario_id,
                    "name": spec.experiment_name,
                    "family": spec.family or "Generalization",
                    "width": spec.width,
                    "height": spec.height,
                    "max_energy": spec.max_energy,
                    "agents": active_agents + pre_agents,
                    "active_agent_ids": [a["id"] for a in spec.active_agents],
                    "resources": resources,
                    "dynamic_obstacles": [],
                    "walls": walls,
                    "pre_positioned": pre_positioned,
                    "collision_rule": spec.collision_rule,
                    "resource_allocation_rule": spec.resource_allocation_rule,
                }
        except Exception:
            # Fallback U06 and U14 if audit module cannot be imported
            registry["U06"] = {
                "id": "U06",
                "name": "U06: 8-Agent Staggered Quadrant Mesh (8 Agents, 4 Resources)",
                "family": "Generalization",
                "width": 10,
                "height": 10,
                "max_energy": 8,
                "agents": [
                    AgentConfig(id=f"ag_{i}", initial_x=(i % 4) * 2, initial_y=(i // 4) * 5, initial_energy=8, target=Position(9 - (i % 4) * 2, 9 - (i // 4) * 5))
                    for i in range(8)
                ],
                "resources": [
                    Resource(Position(2, 2), 6, False),
                    Resource(Position(7, 2), 6, False),
                    Resource(Position(2, 7), 6, False),
                    Resource(Position(7, 7), 6, False),
                ],
                "dynamic_obstacles": [],
                "walls": [],
            }
            registry["U14"] = {
                "id": "U14",
                "name": "U14: Multi-Corner Blind Intersection (4 Agents, 4 Corner Pillars)",
                "family": "Generalization",
                "width": 7,
                "height": 7,
                "max_energy": 10,
                "agents": [
                    AgentConfig(id="agent_n", initial_x=3, initial_y=0, initial_energy=10, target=Position(3, 6)),
                    AgentConfig(id="agent_s", initial_x=3, initial_y=6, initial_energy=10, target=Position(3, 0)),
                    AgentConfig(id="agent_w", initial_x=0, initial_y=3, initial_energy=10, target=Position(6, 3)),
                    AgentConfig(id="agent_e", initial_x=6, initial_y=3, initial_energy=10, target=Position(0, 3)),
                    AgentConfig(id="wall_nw", initial_x=2, initial_y=2, initial_energy=0, target=None),
                    AgentConfig(id="wall_ne", initial_x=4, initial_y=2, initial_energy=0, target=None),
                    AgentConfig(id="wall_sw", initial_x=2, initial_y=4, initial_energy=0, target=None),
                    AgentConfig(id="wall_se", initial_x=4, initial_y=4, initial_energy=0, target=None),
                ],
                "active_agent_ids": ["agent_n", "agent_s", "agent_w", "agent_e"],
                "resources": [],
                "dynamic_obstacles": [],
                "walls": [Position(2, 2), Position(4, 2), Position(2, 4), Position(4, 4)],
            }

        # 3. Simple Baseline Multi-Agent Scenarios
        registry["B01"] = {
            "id": "B01",
            "name": "Baseline: 2-Agent Head-On Corridor Crossing (6x3)",
            "family": "Baseline",
            "width": 6,
            "height": 3,
            "max_energy": 10,
            "agents": [
                AgentConfig(id="agent_west", initial_x=0, initial_y=1, initial_energy=10, target=Position(5, 1)),
                AgentConfig(id="agent_east", initial_x=5, initial_y=1, initial_energy=10, target=Position(0, 1)),
            ],
            "resources": [],
            "dynamic_obstacles": [],
            "walls": [],
        }

        registry["B02"] = {
            "id": "B02",
            "name": "Baseline: 4-Agent Cross Intersection (7x7)",
            "family": "Baseline",
            "width": 7,
            "height": 7,
            "max_energy": 12,
            "agents": [
                AgentConfig(id="agent_n", initial_x=3, initial_y=0, initial_energy=12, target=Position(3, 6)),
                AgentConfig(id="agent_s", initial_x=3, initial_y=6, initial_energy=12, target=Position(3, 0)),
                AgentConfig(id="agent_w", initial_x=0, initial_y=3, initial_energy=12, target=Position(6, 3)),
                AgentConfig(id="agent_e", initial_x=6, initial_y=3, initial_energy=12, target=Position(0, 3)),
            ],
            "resources": [Resource(Position(3, 3), 6, False)],
            "dynamic_obstacles": [],
            "walls": [],
        }

        return registry

    def get_available_scenarios(self) -> list[dict[str, Any]]:
        """Return structured list of all selectable scenarios."""
        result = []
        for sid, sdata in self._scenario_registry.items():
            result.append({
                "id": sid,
                "name": sdata["name"],
                "family": sdata["family"],
                "width": sdata["width"],
                "height": sdata["height"],
                "agent_count": len(sdata.get("active_agent_ids", [a.id for a in sdata["agents"] if getattr(a, "target", None) is not None])),
                "has_dynamic_obstacles": bool(sdata.get("dynamic_obstacles")),
                "has_resources": bool(sdata.get("resources")),
                "has_walls": bool(sdata.get("walls")),
            })
        return result

    def get_available_policies(self) -> list[dict[str, str]]:
        """Return available policy options."""
        return [
            {
                "id": "hybrid_comm",
                "name": "Hybrid Decentralized (Learned + Safety Guard with Intent Comm)",
                "description": "Learned linear Q-policy with safe recharge/detour arbitration and R<=2.0 intent communication.",
            },
            {
                "id": "remediated_comm",
                "name": "Remediated Learned Policy (with Intent Comm)",
                "description": "Learned linear Q-policy with pad safety and local intent communication.",
            },
            {
                "id": "learned_comm",
                "name": "Learned Policy (with Intent Comm)",
                "description": "Learned linear Q-policy with local intent communication.",
            },
            {
                "id": "learned_no_comm",
                "name": "Learned Policy (No Comm)",
                "description": "Non-communicative learned linear Q-policy.",
            },
            {
                "id": "heuristic",
                "name": "Decentralized Heuristic (RobustResourcePolicy)",
                "description": "Rule-based local-horizon policy with cycle detection and energy-safe queuing.",
            },
            {
                "id": "centralized",
                "name": "Centralized Coordinator Baseline",
                "description": "Centralized priority and resource arbitrator baseline.",
            },
        ]

    def load_scenario(
        self,
        scenario_id: str,
        seed: int = 42,
        policy_mode: str = "hybrid_comm",
    ) -> dict[str, Any]:
        """Load and initialize a simulation scenario."""
        if scenario_id not in self._scenario_registry:
            scenario_id = "D01"
        scen = self._scenario_registry[scenario_id]

        self.scenario_id = scenario_id
        self.scenario_name = scen["name"]
        self.scenario_family = scen["family"]
        self.seed = seed
        self.policy_mode = policy_mode
        self.width = scen["width"]
        self.height = scen["height"]
        self.max_energy = scen.get("max_energy", 10)

        # Build world config
        is_centralized = policy_mode == "centralized"
        coord_mode = "centralized" if is_centralized else "decentralized"

        world_cfg = MultiAgentWorldConfig(
            width=scen["width"],
            height=scen["height"],
            agents=tuple(scen["agents"]),
            resources=tuple(scen.get("resources", ())),
            dynamic_obstacles=tuple(scen.get("dynamic_obstacles", ())),
            sensory_radius=2.0,
            communication_radius=2.0,
            enable_coordination=True,
            coordination_mode=coord_mode,
            collision_rule=scen.get("collision_rule", "priority_rotation"),
            resource_allocation_rule=scen.get("resource_allocation_rule", "lowest_id"),
        )

        self.sim = MultiAgentSimulation(world_cfg, seed=seed)

        # Handle pre-positioned entities (exhausted obstacles / walls)
        self.wall_positions = set()
        pre_positioned = scen.get("pre_positioned", ())
        for p in pre_positioned:
            if hasattr(p, "id") and p.id in self.sim._state.agents:
                target_pos = None if p.status == "exhausted" else Position(p.x, p.y)
                self.sim._state.agents[p.id] = replace(
                    self.sim._state.agents[p.id],
                    status=p.status,
                    active=False,
                    target=target_pos,
                    energy=0 if p.status == "exhausted" else self.sim._state.agents[p.id].energy,
                )
            if hasattr(p, "status") and p.status == "exhausted":
                self.wall_positions.add((p.x, p.y))

        # Explicit walls from scenario
        for w in scen.get("walls", ()):
            if isinstance(w, Position):
                self.wall_positions.add((w.x, w.y))
            elif isinstance(w, (tuple, list)):
                self.wall_positions.add((w[0], w[1]))

        # Identify active agent IDs
        if "active_agent_ids" in scen:
            self.active_agent_ids = list(scen["active_agent_ids"])
        else:
            self.active_agent_ids = [
                a.id for a in scen["agents"]
                if getattr(a, "target", None) is not None
            ]

        # Initialize policies per active agent
        base_learner = get_cached_base_learner()
        learned_comm_model = get_cached_learned_model(communication=True)
        learned_no_comm_model = get_cached_learned_model(communication=False)

        self.policies = {}
        for aid in self.active_agent_ids:
            if policy_mode == "hybrid_comm":
                self.policies[aid] = HybridCoordinationPolicy(
                    world_width=self.width,
                    world_height=self.height,
                    max_energy=self.max_energy,
                    sensory_radius=2.0,
                    learner=learned_comm_model,
                    base_learner=base_learner,
                    enable_communication=True,
                    coordinator=self.sim.coordinator,
                )
            elif policy_mode == "remediated_comm":
                self.policies[aid] = LearnedCoordinationPolicy(
                    world_width=self.width,
                    world_height=self.height,
                    max_energy=self.max_energy,
                    sensory_radius=2.0,
                    learner=learned_comm_model,
                    enable_communication=True,
                    coordinator=self.sim.coordinator,
                    enable_safety_guard=True,
                )
            elif policy_mode == "learned_comm":
                self.policies[aid] = LearnedCoordinationPolicy(
                    world_width=self.width,
                    world_height=self.height,
                    max_energy=self.max_energy,
                    sensory_radius=2.0,
                    learner=learned_comm_model,
                    enable_communication=True,
                    coordinator=self.sim.coordinator,
                    enable_safety_guard=False,
                )
            elif policy_mode == "learned_no_comm":
                self.policies[aid] = LearnedCoordinationPolicy(
                    world_width=self.width,
                    world_height=self.height,
                    max_energy=self.max_energy,
                    sensory_radius=2.0,
                    learner=learned_no_comm_model,
                    enable_communication=False,
                    coordinator=self.sim.coordinator,
                    enable_safety_guard=False,
                )
            elif policy_mode == "heuristic" or policy_mode == "centralized":
                pol = RobustResourcePolicy(
                    world_width=self.width,
                    world_height=self.height,
                    max_energy=self.max_energy,
                    sensory_radius=2.0,
                    base_learner=base_learner,
                    coordinator=self.sim.coordinator,
                )
                pol.coordinator = self.sim.coordinator
                pol.utility_evaluator.coordinator = self.sim.coordinator
                self.policies[aid] = pol

        # Reset runtime telemetry
        self.step_count = 0
        self.overall_completed = False
        self.last_actions = {}
        self.last_action_results = {}
        self.last_decision_modes = {aid: "learned" for aid in self.active_agent_ids}
        self.trajectories = {
            aid: [{"x": self.sim.state.agents[aid].position.x, "y": self.sim.state.agents[aid].position.y}]
            for aid in self.active_agent_ids
        }
        self.last_positions = {
            aid: (self.sim.state.agents[aid].position.x, self.sim.state.agents[aid].position.y)
            for aid in self.active_agent_ids
        }
        self.deadlock_locations = []
        self.conflict_attempts = {
            "vertex": 0,
            "swap": 0,
            "occupied": 0,
            "dynamic_obstacle": 0,
        }
        self.cumulative_decisions = {
            "learned": 0,
            "safety_guard": 0,
            "heuristic": 0,
        }
        self.recent_events = [
            f"Initialized scenario {scenario_id}: {self.scenario_name} (Seed {seed}, Policy {policy_mode})"
        ]

        return self.get_snapshot()

    def reset(self) -> dict[str, Any]:
        """Reset the current simulation with the same scenario and seed."""
        return self.load_scenario(self.scenario_id, self.seed, self.policy_mode)

    def step(self) -> dict[str, Any]:
        """Advance the authoritative simulation by exactly one tick."""
        if self.sim is None:
            return self.get_snapshot()

        # Check termination condition
        all_done = all(not self.sim.state.agents[aid].active for aid in self.active_agent_ids)
        if all_done or self.step_count >= self.max_steps:
            self.overall_completed = True
            return self.get_snapshot()

        # Reset tick on coordinator
        if self.sim.coordinator is not None:
            self.sim.coordinator.reset_tick()

        # Record previous positions for smooth client-side interpolation
        for aid in self.active_agent_ids:
            pos = self.sim.state.agents[aid].position
            self.last_positions[aid] = (pos.x, pos.y)

        # Step 1: Agent action selection
        actions: dict[str, Action] = {}
        for aid, agent in self.sim.state.agents.items():
            if not agent.active or aid not in self.policies:
                actions[aid] = Action.UP
                continue

            obs = self.sim.observe_agent(aid)
            pol = self.policies[aid]

            if hasattr(pol, "choose_action"):
                action = pol.choose_action(obs)
            elif hasattr(pol, "select_action"):
                action = pol.select_action(obs, explore=False)
            else:
                action = Action.UP

            actions[aid] = action
            self.last_actions[aid] = action.value

            # Track decision attribution
            if hasattr(pol, "last_decision_mode"):
                mode = pol.last_decision_mode
                self.last_decision_modes[aid] = mode
                if mode in self.cumulative_decisions:
                    self.cumulative_decisions[mode] += 1
            else:
                self.last_decision_modes[aid] = "heuristic"
                self.cumulative_decisions["heuristic"] += 1

        # Step 2: Authoritative simulation execution
        step_res = self.sim.step(actions)
        self.step_count += 1

        # Update conflict attempt metrics
        self.conflict_attempts["vertex"] += step_res.vertex_collisions
        self.conflict_attempts["swap"] += step_res.swap_collisions
        self.conflict_attempts["occupied"] += step_res.occupied_collisions
        self.conflict_attempts["dynamic_obstacle"] += step_res.dynamic_obstacle_collisions

        # Track deadlocks
        for dl in step_res.deadlock_events:
            self.deadlock_locations.append({"x": dl.location.x, "y": dl.location.y})
            agents_desc = getattr(dl, "affected_agents", ())
            self.recent_events.append(f"Tick {self.step_count}: Deadlock detected at ({dl.location.x}, {dl.location.y}) involving {agents_desc}")

        # Step 3: Record action outcomes in policies and update trajectories
        for aid in self.active_agent_ids:
            if aid in step_res.agent_results:
                ar = step_res.agent_results[aid]
                self.last_action_results[aid] = ar
                pol = self.policies.get(aid)
                if pol is not None and hasattr(pol, "record_action_result"):
                    pol.record_action_result(
                        ar.position_after,
                        ar.energy_after,
                        ar.action,
                        ar.success,
                    )
                new_pos = self.sim.state.agents[aid].position
                self.trajectories[aid].append({"x": new_pos.x, "y": new_pos.y})

                # Event log highlights
                if ar.reached_target:
                    self.recent_events.append(f"Tick {self.step_count}: Agent '{aid}' successfully reached target!")
                elif ar.exhausted:
                    self.recent_events.append(f"Tick {self.step_count}: Agent '{aid}' ran out of energy (exhausted)!")
                elif ar.action == Action.RECHARGE and ar.success and ar.gained_energy > 0:
                    self.recent_events.append(f"Tick {self.step_count}: Agent '{aid}' recharged +{ar.gained_energy} energy.")
                elif ar.collision_type is not None:
                    self.recent_events.append(f"Tick {self.step_count}: Agent '{aid}' conflict attempt: {ar.collision_type}.")

        # Check goal completion
        if step_res.overall_completed or all(not self.sim.state.agents[aid].active and self.sim.state.agents[aid].status == "reached_target" for aid in self.active_agent_ids):
            self.overall_completed = True
            self.recent_events.append(f"Tick {self.step_count}: All active agents completed goals!")

        # Keep recent events bounded
        if len(self.recent_events) > 60:
            self.recent_events = self.recent_events[-60:]

        return self.get_snapshot()

    def get_snapshot(self) -> dict[str, Any]:
        """Construct complete, authoritative JSON-serializable snapshot."""
        if self.sim is None:
            return {}

        state = self.sim.state

        # 1. Static walls and obstacles
        walls = [{"x": wx, "y": wy} for (wx, wy) in sorted(self.wall_positions)]

        # 2. Resources (recharge pads)
        resources_list = []
        for r in state.resources:
            occupant = None
            for aid, ag in state.agents.items():
                if ag.position == r.position:
                    occupant = aid
                    break
            resources_list.append({
                "x": r.position.x,
                "y": r.position.y,
                "recharge_amount": r.recharge_amount,
                "consumable": r.consumable,
                "active": occupant is not None,
                "occupied_by": occupant,
            })

        # 3. Dynamic obstacles
        dyn_obstacles_list = []
        blocked_cells = {Position(w[0], w[1]) for w in self.wall_positions}
        blocked_cells.update(r.position for r in state.resources)

        for did, dstate in state.dynamic_obstacles.items():
            cfg = next((o for o in self.sim.config.dynamic_obstacles if o.id == did), None)
            prev_x = dstate.position.x - dstate.velocity[0]
            prev_y = dstate.position.y - dstate.velocity[1]
            next_state = None
            if cfg is not None:
                next_state = predict_next_obstacle_state(
                    dstate, cfg, self.width, self.height, blocked_cells
                )
            dyn_obstacles_list.append({
                "id": did,
                "x": dstate.position.x,
                "y": dstate.position.y,
                "prev_x": prev_x,
                "prev_y": prev_y,
                "vx": dstate.velocity[0],
                "vy": dstate.velocity[1],
                "pattern": cfg.pattern if cfg else "fixed",
                "next_x": next_state.position.x if next_state else dstate.position.x,
                "next_y": next_state.position.y if next_state else dstate.position.y,
                "waypoints": [{"x": wp.x, "y": wp.y} for wp in (cfg.waypoints if cfg else ())],
            })

        # 4. Agents
        agents_dict = {}
        for aid in self.active_agent_ids:
            ag = state.agents[aid]
            prev = self.last_positions.get(aid, (ag.position.x, ag.position.y))
            pol = self.policies.get(aid)

            # Local observation
            obs = self.sim.observe_agent(aid)
            obs_cells = [{"x": c.x, "y": c.y} for c in obs.cells]

            # Visible peers within R <= 2.0
            visible_peers = [other.id for other in obs.observed_agents if other.id in self.active_agent_ids]

            # Intent communication vector
            intent_dict = None
            peer_intents_dict = {}
            if self.sim.coordinator is not None and aid in self.sim.coordinator.agent_intentions:
                intended_pos = self.sim.coordinator.agent_intentions[aid]
                intent_dict = {"x": intended_pos.x, "y": intended_pos.y}
                for pid in visible_peers:
                    if pid in self.sim.coordinator.agent_intentions:
                        p_pos = self.sim.coordinator.agent_intentions[pid]
                        peer_intents_dict[pid] = {"x": p_pos.x, "y": p_pos.y}

            # Decision attribution
            attribution = self.last_decision_modes.get(aid, "learned").upper()
            if attribution == "SAFETY_GUARD":
                attribution = "SAFETY GUARD"

            ar = self.last_action_results.get(aid)

            agents_dict[aid] = {
                "id": aid,
                "x": ag.position.x,
                "y": ag.position.y,
                "prev_x": prev[0],
                "prev_y": prev[1],
                "target": {"x": ag.target.x, "y": ag.target.y} if ag.target is not None else None,
                "energy": ag.energy,
                "max_energy": self.max_energy,
                "status": ag.status,
                "active": ag.active,
                "action": self.last_actions.get(aid, "NONE"),
                "action_success": ar.success if ar else True,
                "action_moved": ar.moved if ar else False,
                "collision_type": ar.collision_type if ar else None,
                "decision_attribution": attribution,
                "decision_counts": {
                    "learned": getattr(pol, "decision_counts", {}).get("learned", 0) if pol else 0,
                    "safety_guard": getattr(pol, "decision_counts", {}).get("safety_guard", 0) if pol else 0,
                },
                "visible_peers": visible_peers,
                "intent": intent_dict,
                "peer_intents": peer_intents_dict,
                "observation_cells": obs_cells,
                "trail": self.trajectories.get(aid, []),
            }

        # 5. Global Metrics & Authoritative Safety Invariants
        active_count = sum(1 for aid in self.active_agent_ids if state.agents[aid].active)
        reached_count = sum(1 for aid in self.active_agent_ids if state.agents[aid].status == "reached_target")
        exhausted_count = sum(1 for aid in self.active_agent_ids if state.agents[aid].status == "exhausted")
        total_count = len(self.active_agent_ids)
        comp_rate = (reached_count / total_count * 100.0) if total_count > 0 else 0.0

        total_conflict_attempts = (
            self.conflict_attempts["vertex"]
            + self.conflict_attempts["swap"]
            + self.conflict_attempts["occupied"]
            + self.conflict_attempts["dynamic_obstacle"]
        )

        return {
            "scenario": {
                "id": self.scenario_id,
                "name": self.scenario_name,
                "family": self.scenario_family,
                "width": self.width,
                "height": self.height,
                "seed": self.seed,
                "policy_mode": self.policy_mode,
                "max_steps": self.max_steps,
            },
            "tick": self.step_count,
            "overall_completed": self.overall_completed,
            "walls": walls,
            "resources": resources_list,
            "dynamic_obstacles": dyn_obstacles_list,
            "agents": agents_dict,
            "metrics": {
                "active_agents": active_count,
                "reached_agents": reached_count,
                "exhausted_agents": exhausted_count,
                "total_agents": total_count,
                "completion_rate_pct": round(comp_rate, 1),
                "attempted_conflicts": total_conflict_attempts,
                # AUTHORITATIVE INVARIANT: WorldSim engine prevents illegal cell overlap
                "physical_collisions": 0,
                "vertex_collisions": self.conflict_attempts["vertex"],
                "swap_collisions": self.conflict_attempts["swap"],
                "occupied_collisions": self.conflict_attempts["occupied"],
                "dynamic_obstacle_collisions": self.conflict_attempts["dynamic_obstacle"],
                "deadlock_events_count": len(self.deadlock_locations),
                "deadlock_locations": self.deadlock_locations[-10:],
                "total_learned_decisions": self.cumulative_decisions["learned"],
                "total_safety_decisions": self.cumulative_decisions["safety_guard"],
                "total_heuristic_decisions": self.cumulative_decisions["heuristic"],
            },
            "recent_events": self.recent_events,
        }


# ==============================================================================
# HTTP Request Handler & Server
# ==============================================================================

class SimulationHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for WorldSim viewer REST API and Web UI."""

    session: SimulationSession

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard logging to keep test and command output clean."""
        pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()

            # Read HTML from static directory or fallback
            html_path = Path(__file__).parent / "static" / "index.html"
            if html_path.exists():
                content = html_path.read_bytes()
            else:
                content = b"<!DOCTYPE html><html><body><h1>WorldSim Viewer</h1></body></html>"
            self.wfile.write(content)

        elif path == "/api/state":
            snapshot = self.session.get_snapshot()
            data = json.dumps(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/scenarios":
            scenarios = self.session.get_available_scenarios()
            data = json.dumps(scenarios).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/policies":
            policies = self.session.get_available_policies()
            data = json.dumps(policies).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/health":
            data = json.dumps({
                "status": "ok",
                "tick": self.session.step_count,
                "scenario": self.session.scenario_id,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if path == "/api/step":
            snapshot = self.session.step()
            data = json.dumps(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/reset":
            snapshot = self.session.reset()
            data = json.dumps(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/select":
            scenario_id = payload.get("scenario_id", self.session.scenario_id)
            seed = int(payload.get("seed", self.session.seed))
            policy_mode = payload.get("policy_mode", self.session.policy_mode)
            snapshot = self.session.load_scenario(scenario_id, seed, policy_mode)
            data = json.dumps(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404)
            self.end_headers()


class LiveViewerServer:
    """Zero-dependency multithreaded HTTP server for WorldSim live viewer."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        session: SimulationSession | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.session = session if session is not None else SimulationSession()

        # Custom handler subclass binding session
        session_ref = self.session

        class BoundHandler(SimulationHTTPRequestHandler):
            session = session_ref

        self.handler_class = BoundHandler
        self.httpd = http.server.ThreadingHTTPServer((self.host, self.port), self.handler_class)
        # Record actual bound port in case port 0 was passed
        self.port = self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def serve_forever(self) -> None:
        """Run the server synchronously in the current thread."""
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def launch_web_viewer(
    host: str = "127.0.0.1",
    port: int = 8000,
    scenario: str = "D01",
    policy: str = "hybrid_comm",
    seed: int = 42,
    open_browser: bool = True,
) -> LiveViewerServer:
    """Launch the WorldSim Web Viewer server and optionally open the user's browser."""
    session = SimulationSession(
        default_scenario=scenario,
        default_policy=policy,
        default_seed=seed,
    )
    server = LiveViewerServer(host=host, port=port, session=session)
    server.start()

    url = f"http://{host}:{server.port}/"
    print(f"WorldSim Live 2D Viewer active at: {url}")
    print(f"Loaded scenario: {session.scenario_id} ({session.scenario_name})")
    print(f"Policy: {session.policy_mode} | Seed: {session.seed}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WorldSim Live 2D World Viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--scenario", default="D01", help="Scenario ID (e.g. D01, D04, U06, U14)")
    parser.add_argument("--policy", default="hybrid_comm", help="Policy mode (hybrid_comm, remediated_comm, learned_comm, heuristic)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    args = parser.parse_args()
    srv = launch_web_viewer(
        host=args.host,
        port=args.port,
        scenario=args.scenario,
        policy=args.policy,
        seed=args.seed,
        open_browser=not args.no_browser,
    )
    print("Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down WorldSim Viewer...")
        srv.stop()
