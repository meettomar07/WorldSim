"""Multi-agent simulation environment, conflict resolution, and metrics for WorldSim Phase 13.

Supports:
- Multiple autonomous agents with unique identities, positions, energies, and policies.
- Two-phase tick execution with deterministic conflict resolution (vertex, swap, occupied).
- Atomic shared resource consumption and contention tracking (zero duplicate energy extraction).
- Configurable individual goals, shared goals, and opt-in cooperative discovery sharing.
- Comprehensive multi-agent metrics, experiment runners, and reproducible benchmarks.
"""

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from random import Random
import statistics
from typing import Any, Sequence

from .actions import Action, destination
from .coordination import (
    DeadlockDetector,
    DeadlockEvent,
    FairnessMonitor,
    MultiAgentCoordinator,
    ResourceArbitrator,
    agent_val,
    jains_fairness_index,
)
from .dynamic_obstacles import (
    DynamicObstacleConfig,
    DynamicObstacleState,
    predict_next_obstacle_state,
)
from .models import Position, Resource


# ==============================================================================
# Agent and World Configuration
# ==============================================================================


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Immutable initial configuration for a single agent."""

    id: str
    initial_x: int
    initial_y: int
    initial_energy: int = 10
    target: Position | None = None
    policy: Any | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Agent id must be a non-empty string.")
        if self.initial_energy < 0:
            raise ValueError("initial_energy must be non-negative.")


@dataclass(frozen=True, slots=True)
class MultiAgentWorldConfig:
    """Configuration for a multi-agent simulation environment."""

    width: int = 20
    height: int = 12
    agents: tuple[AgentConfig, ...] = ()
    resources: tuple[Resource, ...] = ()
    shared_target: Position | None = None
    sensory_radius: float = 2.0
    max_energy: int = 10
    movement_energy_cost: int = 1
    collision_rule: str = "bounce"  # "bounce", "priority", "priority_rotation"
    resource_allocation_rule: str = "lowest_id"  # "lowest_id", "priority_rotation", "need_urgency", "starvation_prevention", "fair_balanced"
    enable_coordination: bool = False
    cooperative_sharing: bool = False
    deadlock_threshold: int = 2
    congestion_safety_margin: int = 0
    goal_criterion: str = "individual"  # "individual", "shared_any", "shared_all", "shared_subset"
    required_goal_count: int = 1
    coordination_mode: str = "centralized"  # "centralized" or "decentralized"
    communication_radius: float = 2.0  # Communication/sensory radius for local horizon
    dynamic_obstacles: tuple[DynamicObstacleConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive.")
        if not self.agents:
            raise ValueError("Multi-agent configuration must have at least one agent.")
        if self.max_energy <= 0 or self.movement_energy_cost <= 0:
            raise ValueError("max_energy and movement_energy_cost must be positive.")
        if self.sensory_radius <= 0:
            raise ValueError("sensory_radius must be positive.")
        if self.communication_radius <= 0:
            raise ValueError("communication_radius must be positive.")
        if self.coordination_mode not in {"centralized", "decentralized"}:
            raise ValueError("coordination_mode must be 'centralized' or 'decentralized'.")
        if self.collision_rule not in {"bounce", "priority", "priority_rotation"}:
            raise ValueError("collision_rule must be 'bounce', 'priority', or 'priority_rotation'.")
        if self.resource_allocation_rule not in {
            "lowest_id",
            "priority_rotation",
            "need_urgency",
            "starvation_prevention",
            "fair_balanced",
        }:
            raise ValueError("Invalid resource_allocation_rule.")
        if self.goal_criterion not in {"individual", "shared_any", "shared_all", "shared_subset"}:
            raise ValueError("goal_criterion must be one of 'individual', 'shared_any', 'shared_all', 'shared_subset'.")

        # Validate agent IDs uniqueness
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Agent IDs must be unique; got {ids}")

        # Validate starting coordinates inside world and no overlapping starts
        seen_positions = set()
        for a in self.agents:
            if not (0 <= a.initial_x < self.width and 0 <= a.initial_y < self.height):
                raise ValueError(f"Agent {a.id} initial position ({a.initial_x}, {a.initial_y}) is out of bounds.")
            pos = (a.initial_x, a.initial_y)
            if pos in seen_positions:
                raise ValueError(f"Agents cannot start on the same cell; duplicate position {pos}.")
            seen_positions.add(pos)

        # Validate dynamic obstacles
        dyn_ids = [o.id for o in self.dynamic_obstacles]
        if len(dyn_ids) != len(set(dyn_ids)):
            raise ValueError(f"Dynamic obstacle IDs must be unique; got {dyn_ids}")
        for did in dyn_ids:
            if did in ids:
                raise ValueError(f"Dynamic obstacle ID '{did}' conflicts with agent ID.")

        res_positions = {(r.position.x, r.position.y) for r in self.resources}
        for o in self.dynamic_obstacles:
            pos = (o.initial_x, o.initial_y)
            if not (0 <= o.initial_x < self.width and 0 <= o.initial_y < self.height):
                raise ValueError(f"Dynamic obstacle {o.id} initial position {pos} is out of bounds.")
            if pos in seen_positions:
                raise ValueError(f"Dynamic obstacle {o.id} cannot start on an occupied agent cell: {pos}")
            if pos in res_positions:
                raise ValueError(f"Dynamic obstacle {o.id} cannot start on a resource pad: {pos}")
            seen_positions.add(pos)


# ==============================================================================
# Multi-Agent State and Observations
# ==============================================================================


@dataclass(frozen=True, slots=True)
class AgentState:
    """Current immutable runtime state of an individual agent."""

    id: str
    position: Position
    energy: int = 10
    target: Position | None = None
    active: bool = True
    status: str = "active"  # "active", "reached_target", "exhausted", "inactive"


@dataclass(frozen=True, slots=True)
class MultiAgentWorldState:
    """Immutable state snapshot of the entire multi-agent world."""

    width: int
    height: int
    tick: int
    agents: dict[str, AgentState]
    resources: tuple[Resource, ...] = ()
    shared_target: Position | None = None
    dynamic_obstacles: dict[str, DynamicObstacleState] = field(default_factory=dict)

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height


@dataclass(frozen=True, slots=True)
class MultiAgentObservation:
    """Bounded, duck-type compatible observation provided to an individual agent."""

    agent_id: str
    entity_position: Position
    cells: tuple[Position, ...]
    energy: int = 10
    observed_resources: tuple[Resource, ...] = ()
    target: Position | None = None
    observed_agents: tuple[AgentState, ...] = ()
    shared_discoveries: tuple[Resource, ...] = ()
    observed_dynamic_obstacles: tuple[DynamicObstacleState, ...] = ()


# ==============================================================================
# Step Results and Collision Metrics
# ==============================================================================


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    """Outcome of one tick for a single agent."""

    agent_id: str
    success: bool
    moved: bool
    action: Action
    position_before: Position
    position_after: Position
    energy_before: int
    energy_after: int
    consumed_energy: int
    gained_energy: int
    collision_type: str | None  # "vertex", "swap", "occupied", None
    resource_contention: bool
    reached_target: bool
    exhausted: bool
    observation: MultiAgentObservation


@dataclass(frozen=True, slots=True)
class MultiAgentStepResult:
    """Comprehensive outcome of a simultaneous tick across all agents."""

    tick: int
    state: MultiAgentWorldState
    agent_results: dict[str, AgentStepResult]
    vertex_collisions: int
    swap_collisions: int
    occupied_collisions: int
    resource_contentions: int
    resources_depleted: int
    overall_completed: bool
    deadlock_events: tuple[DeadlockEvent, ...] = ()
    blocked_agents: tuple[str, ...] = ()
    dynamic_obstacle_collisions: int = 0


# ==============================================================================
# Multi-Agent Simulation Engine
# ==============================================================================


class MultiAgentSimulation:
    """Deterministic, two-phase multi-agent simulation engine.

    Executes simultaneous actions:
    1. Pre-movement observations generated from a frozen world snapshot.
    2. Actions collected for all active agents.
    3. Deterministic conflict resolution (vertex bounce, swap bounce, occupied blocks).
    4. Atomic movement and resource updates (zero duplicate energy gain).
    5. Terminal status and goal completion evaluation.
    """

    def __init__(
        self,
        config: MultiAgentWorldConfig,
        seed: int | None = None,
        resource_arbitrator: ResourceArbitrator | None = None,
    ) -> None:
        self.config = config
        self._seed = seed if seed is not None else 0
        self._rng = Random(seed)

        # Initialize agents
        agents_dict: dict[str, AgentState] = {}
        for a in config.agents:
            target = a.target if a.target is not None else config.shared_target
            agents_dict[a.id] = AgentState(
                id=a.id,
                position=Position(a.initial_x, a.initial_y),
                energy=a.initial_energy,
                target=target,
                active=True,
                status="active",
            )

        # Initialize dynamic obstacles
        dynamic_obs_dict: dict[str, DynamicObstacleState] = {}
        for o in config.dynamic_obstacles:
            dynamic_obs_dict[o.id] = DynamicObstacleState(
                id=o.id,
                position=Position(o.initial_x, o.initial_y),
                velocity=o.velocity,
                waypoint_index=0,
                reverse_direction=False,
            )

        self._state = MultiAgentWorldState(
            width=config.width,
            height=config.height,
            tick=0,
            agents=agents_dict,
            resources=config.resources,
            shared_target=config.shared_target,
            dynamic_obstacles=dynamic_obs_dict,
        )

        self._paused = False
        self._shared_discoveries: set[Resource] = set()

        # Dynamic obstacle tracking
        self.dynamic_obstacle_history: list[dict[str, DynamicObstacleState]] = [dict(dynamic_obs_dict)]
        self.cumulative_dynamic_obstacle_collisions = 0

        # Cumulative event counters
        self.cumulative_vertex_collisions = 0
        self.cumulative_swap_collisions = 0
        self.cumulative_occupied_collisions = 0
        self.cumulative_resource_contentions = 0
        self.cumulative_resources_depleted = 0

        # Phase 13.1 & 13.3 Coordination, Deadlock & Fairness Components
        self.resource_arbitrator = (
            resource_arbitrator
            if resource_arbitrator is not None
            else ResourceArbitrator(rule=config.resource_allocation_rule)
        )
        self.deadlock_detector = DeadlockDetector(persistence_threshold=config.deadlock_threshold)
        self.coordinator = MultiAgentCoordinator(
            enabled=config.enable_coordination,
            world_width=config.width,
            world_height=config.height,
            coordination_mode=getattr(config, "coordination_mode", "centralized"),
            communication_radius=getattr(config, "communication_radius", 2.0),
        )
        self.fairness_monitor = FairnessMonitor(agent_ids=[a.id for a in config.agents])

    @property
    def deadlock_events(self) -> list[DeadlockEvent]:
        return self.deadlock_detector.detected_events

    @property
    def fairness_metrics(self) -> dict[str, Any]:
        return self.fairness_monitor.metrics

    def compute_jains_fairness(self, metric_name: str = "goal_completion") -> float:
        return self.fairness_monitor.compute_jains_fairness(metric_name)

    @property
    def obstacle_history(self) -> list[dict[str, DynamicObstacleState]]:
        """Return the chronological history of dynamic obstacle states."""
        return self.dynamic_obstacle_history

    @property
    def state(self) -> MultiAgentWorldState:
        return self._state

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def observe_agent(self, agent_id: str) -> MultiAgentObservation:
        """Construct bounded observation for the specified agent."""
        agent = self._state.agents[agent_id]
        origin = agent.position
        r_sq = self.config.sensory_radius * self.config.sensory_radius

        # Observed grid cells
        cells = tuple(
            Position(x, y)
            for y in range(self._state.height)
            for x in range(self._state.width)
            if (x - origin.x) ** 2 + (y - origin.y) ** 2 <= r_sq
        )

        # Observed resources
        observed_res = tuple(
            res for res in self._state.resources
            if (res.position.x - origin.x) ** 2 + (res.position.y - origin.y) ** 2 <= r_sq
        )

        # Observed other agents
        observed_agents = tuple(
            other for other_id, other in self._state.agents.items()
            if other_id != agent_id and (other.position.x - origin.x) ** 2 + (other.position.y - origin.y) ** 2 <= r_sq
        )

        # Observed dynamic obstacles within sensory radius R
        observed_dynamic = tuple(
            obs for obs in self._state.dynamic_obstacles.values()
            if (obs.position.x - origin.x) ** 2 + (obs.position.y - origin.y) ** 2 <= r_sq
        )

        shared = tuple(sorted(self._shared_discoveries, key=lambda r: (r.position.x, r.position.y))) if self.config.cooperative_sharing else ()

        return MultiAgentObservation(
            agent_id=agent_id,
            entity_position=origin,
            cells=cells,
            energy=agent.energy,
            observed_resources=observed_res,
            target=agent.target,
            observed_agents=observed_agents,
            shared_discoveries=shared,
            observed_dynamic_obstacles=observed_dynamic,
        )

    def observe_all(self) -> dict[str, MultiAgentObservation]:
        """Return observation dictionary for all registered agents."""
        return {agent_id: self.observe_agent(agent_id) for agent_id in self._state.agents}

    def _resolve_conflicts(
        self,
        intents: dict[str, tuple[Action, Position]],
    ) -> tuple[dict[str, Position], dict[str, str | None], dict[str, DynamicObstacleState], int, int, int, int]:
        """Deterministically resolve movement conflicts using iterative stability.

        Returns:
            final_destinations: agent_id -> final Position
            collision_types: agent_id -> collision type string or None
            next_obstacle_states: obstacle_id -> DynamicObstacleState
            vertex_collisions, swap_collisions, occupied_collisions, dynamic_obstacle_collisions
        """
        initial_pos = {aid: self._state.agents[aid].position for aid in self._state.agents}
        initial_obs_pos = {oid: self._state.dynamic_obstacles[oid].position for oid in self._state.dynamic_obstacles}
        tentative_dest = {aid: dest for aid, (_, dest) in intents.items()}
        collision_type: dict[str, str | None] = {aid: None for aid in self._state.agents}

        # Predict proposed states for dynamic obstacles
        obs_configs = {o.id: o for o in self.config.dynamic_obstacles}
        static_blocked_for_obstacles = set(r.position for r in self._state.resources)
        # Inactive or exhausted agents act as impassable walls
        static_blocked_for_obstacles.update(
            a.position for a in self._state.agents.values() if not a.active
        )

        tentative_obs_states: dict[str, DynamicObstacleState] = {}
        for oid, obs_state in self._state.dynamic_obstacles.items():
            tentative_obs_states[oid] = predict_next_obstacle_state(
                obs_state,
                obs_configs[oid],
                self.config.width,
                self.config.height,
                static_blocked_for_obstacles,
            )

        v_count = 0
        s_count = 0
        o_count = 0
        dyn_count = 0

        changed = True
        while changed:
            changed = False

            # Obstacle vs Obstacle conflicts
            # 1. Two obstacles claiming the same destination cell
            obs_claims: dict[Position, list[str]] = {}
            for oid, state in tentative_obs_states.items():
                obs_claims.setdefault(state.position, []).append(oid)
            for cell, claimants in obs_claims.items():
                if len(claimants) >= 2:
                    for oid in claimants:
                        if tentative_obs_states[oid].position != initial_obs_pos[oid]:
                            cur_v = tentative_obs_states[oid].velocity
                            tentative_obs_states[oid] = DynamicObstacleState(
                                id=oid,
                                position=initial_obs_pos[oid],
                                velocity=(-cur_v[0], -cur_v[1]),
                                waypoint_index=tentative_obs_states[oid].waypoint_index,
                                reverse_direction=not tentative_obs_states[oid].reverse_direction,
                            )
                            changed = True

            # 2. Two obstacles swap
            for o1 in list(tentative_obs_states):
                for o2 in list(tentative_obs_states):
                    if o1 < o2:
                        if (
                            tentative_obs_states[o1].position == initial_obs_pos[o2]
                            and tentative_obs_states[o2].position == initial_obs_pos[o1]
                            and initial_obs_pos[o1] != initial_obs_pos[o2]
                        ):
                            cur_v1 = tentative_obs_states[o1].velocity
                            cur_v2 = tentative_obs_states[o2].velocity
                            tentative_obs_states[o1] = DynamicObstacleState(
                                id=o1,
                                position=initial_obs_pos[o1],
                                velocity=(-cur_v1[0], -cur_v1[1]),
                                waypoint_index=tentative_obs_states[o1].waypoint_index,
                                reverse_direction=not tentative_obs_states[o1].reverse_direction,
                            )
                            tentative_obs_states[o2] = DynamicObstacleState(
                                id=o2,
                                position=initial_obs_pos[o2],
                                velocity=(-cur_v2[0], -cur_v2[1]),
                                waypoint_index=tentative_obs_states[o2].waypoint_index,
                                reverse_direction=not tentative_obs_states[o2].reverse_direction,
                            )
                            changed = True

            # Agent vs Dynamic Obstacle conflicts
            for aid in list(tentative_dest):
                a_dest = tentative_dest[aid]
                for oid, obs_state in list(tentative_obs_states.items()):
                    o_dest = obs_state.position
                    o_init = initial_obs_pos[oid]
                    a_init = initial_pos[aid]

                    # 1. Swap: agent tries to enter obstacle's cell, obstacle tries to enter agent's cell
                    if a_dest == o_init and o_dest == a_init and a_init != o_init:
                        tentative_dest[aid] = a_init
                        collision_type[aid] = "swap"
                        cur_v = obs_state.velocity
                        tentative_obs_states[oid] = DynamicObstacleState(
                            id=oid,
                            position=o_init,
                            velocity=(-cur_v[0], -cur_v[1]),
                            waypoint_index=obs_state.waypoint_index,
                            reverse_direction=not obs_state.reverse_direction,
                        )
                        dyn_count += 1
                        changed = True
                        continue

                    # 2. Vertex: agent and obstacle try to enter the same vacant cell
                    if a_dest == o_dest and a_dest != a_init and a_dest != o_init:
                        tentative_dest[aid] = a_init
                        collision_type[aid] = "vertex"
                        dyn_count += 1
                        changed = True
                        continue

                    # 3. Obstacle tries to move into agent's cell while agent is stationary / waiting
                    if o_dest == a_dest and a_dest == a_init:
                        cur_v = obs_state.velocity
                        tentative_obs_states[oid] = DynamicObstacleState(
                            id=oid,
                            position=o_init,
                            velocity=(-cur_v[0], -cur_v[1]),
                            waypoint_index=obs_state.waypoint_index,
                            reverse_direction=not obs_state.reverse_direction,
                        )
                        changed = True
                        continue

                    # 4. Agent tries to move into stationary obstacle's cell
                    if a_dest == o_init and o_dest == o_init and a_dest != a_init:
                        tentative_dest[aid] = a_init
                        collision_type[aid] = "occupied"
                        dyn_count += 1
                        changed = True
                        continue

            # 1. Vertex Collisions: 2+ agents claiming the same destination cell C
            dest_claims: dict[Position, list[str]] = {}
            for aid, dest in tentative_dest.items():
                dest_claims.setdefault(dest, []).append(aid)

            for cell, claimants in dest_claims.items():
                moving_claimants = [aid for aid in claimants if tentative_dest[aid] != initial_pos[aid]]
                if len(moving_claimants) >= 2:
                    matching_resource = next((r for r in self._state.resources if r.position == cell), None)
                    if matching_resource is not None and self.config.resource_allocation_rule != "lowest_id":
                        winner, losers = self.resource_arbitrator.arbitrate(
                            moving_claimants,
                            agent_energies={aid: self._state.agents[aid].energy for aid in moving_claimants},
                            agent_max_energies={aid: self.config.max_energy for aid in moving_claimants},
                            resource=matching_resource,
                            tick=self._state.tick + self._seed,
                        )
                        for aid in losers:
                            tentative_dest[aid] = initial_pos[aid]
                            collision_type[aid] = "vertex"
                            v_count += 1
                            changed = True
                        self.cumulative_resource_contentions += (len(moving_claimants) - 1)
                    elif self.config.collision_rule == "priority_rotation":
                        step_idx = self._state.tick + self._seed
                        def priority_key(a: str) -> int:
                            val = agent_val(a)
                            return (val + step_idx) % len(moving_claimants)
                        claimants_sorted = sorted(moving_claimants, key=priority_key)
                        winner = claimants_sorted[0]
                        losers = claimants_sorted[1:]
                        for aid in losers:
                            tentative_dest[aid] = initial_pos[aid]
                            collision_type[aid] = "vertex"
                            v_count += 1
                            changed = True
                    elif self.config.collision_rule == "priority":
                        claimants_sorted = sorted(moving_claimants)
                        winner = claimants_sorted[0]
                        losers = claimants_sorted[1:]
                        for aid in losers:
                            tentative_dest[aid] = initial_pos[aid]
                            collision_type[aid] = "vertex"
                            v_count += 1
                            changed = True
                    else:
                        # Symmetric bounce: all moving claimants are rejected back to initial position
                        for aid in moving_claimants:
                            tentative_dest[aid] = initial_pos[aid]
                            collision_type[aid] = "vertex"
                            v_count += 1
                            changed = True

            # 2. Swap Collisions: Agent A moves to P_B and Agent B moves to P_A
            for a_id in list(tentative_dest):
                for b_id in list(tentative_dest):
                    if a_id < b_id:
                        if (
                            tentative_dest[a_id] == initial_pos[b_id]
                            and tentative_dest[b_id] == initial_pos[a_id]
                            and initial_pos[a_id] != initial_pos[b_id]
                        ):
                            tentative_dest[a_id] = initial_pos[a_id]
                            tentative_dest[b_id] = initial_pos[b_id]
                            collision_type[a_id] = "swap"
                            collision_type[b_id] = "swap"
                            s_count += 2
                            changed = True

            # 3. Occupied Stationary Collisions: Agent A moves to cell of Agent B who is stationary
            for a_id, dest in list(tentative_dest.items()):
                if dest != initial_pos[a_id]:
                    # Find who is in 'dest'
                    occupant = next((b_id for b_id, b_dest in tentative_dest.items() if b_id != a_id and b_dest == dest and b_dest == initial_pos[b_id]), None)
                    if occupant is not None:
                        occupant_agent = self._state.agents[occupant]
                        if occupant_agent.status == "reached_target":
                            # Phase 13.3: Inactive agent on goal cell yields right-of-way so it does
                            # not permanently block corridors or terminal cells for active agents.
                            pass
                        else:
                            tentative_dest[a_id] = initial_pos[a_id]
                            collision_type[a_id] = "occupied"
                            o_count += 1
                            changed = True

        return tentative_dest, collision_type, tentative_obs_states, v_count, s_count, o_count, dyn_count

    def step(self, actions: dict[str, Action]) -> MultiAgentStepResult:
        """Execute one simultaneous tick across all agents.

        Phase 1: Freeze observations.
        Phase 2: Intent formulation & feasibility check.
        Phase 3: Conflict resolution.
        Phase 4: Atomic movements and resource interactions.
        Phase 5: State update & goal evaluation.
        """
        if self._paused:
            obs_map = self.observe_all()
            dummy_results = {
                aid: AgentStepResult(
                    agent_id=aid,
                    success=False,
                    moved=False,
                    action=actions.get(aid, Action.UP),
                    position_before=self._state.agents[aid].position,
                    position_after=self._state.agents[aid].position,
                    energy_before=self._state.agents[aid].energy,
                    energy_after=self._state.agents[aid].energy,
                    consumed_energy=0,
                    gained_energy=0,
                    collision_type=None,
                    resource_contention=False,
                    reached_target=False,
                    exhausted=False,
                    observation=obs_map[aid],
                )
                for aid in self._state.agents
            }
            return MultiAgentStepResult(
                tick=self._state.tick,
                state=self._state,
                agent_results=dummy_results,
                vertex_collisions=0,
                swap_collisions=0,
                occupied_collisions=0,
                resource_contentions=0,
                resources_depleted=0,
                overall_completed=False,
            )

        # Phase 1: Pre-movement observations
        pre_obs = self.observe_all()

        # Phase 2: Intent formulation
        intents: dict[str, tuple[Action, Position]] = {}
        for aid, agent in self._state.agents.items():
            act = actions.get(aid, Action.UP)
            if not isinstance(act, Action):
                raise ValueError(f"Action for agent {aid} must be an Action enum value; got {act!r}")

            if not agent.active:
                # Inactive / exhausted / completed agent remains in place
                intents[aid] = (act, agent.position)
                continue

            if act == Action.RECHARGE:
                intents[aid] = (act, agent.position)
            else:
                # Movement attempt
                if agent.energy < self.config.movement_energy_cost:
                    # Insufficient energy: cannot move
                    intents[aid] = (act, agent.position)
                else:
                    dest = destination(agent.position, act)
                    if not self._state.contains(dest):
                        # Out of bounds: cannot move
                        intents[aid] = (act, agent.position)
                    else:
                        intents[aid] = (act, dest)

        # Phase 3: Conflict resolution
        final_dests, collisions, next_obs_states, v_cols, s_cols, o_cols, dyn_cols = self._resolve_conflicts(intents)
        self.cumulative_vertex_collisions += v_cols
        self.cumulative_swap_collisions += s_cols
        self.cumulative_occupied_collisions += o_cols
        self.cumulative_dynamic_obstacle_collisions += dyn_cols

        # Phase 4: Atomic movements and resource interactions
        # First, process movements and energy costs
        interim_positions: dict[str, Position] = {}
        interim_energies: dict[str, int] = {}
        moved_flags: dict[str, bool] = {}
        success_flags: dict[str, bool] = {}

        for aid, agent in self._state.agents.items():
            act = actions.get(aid, Action.UP)
            start_pos = agent.position
            end_pos = final_dests[aid]
            cur_e = agent.energy

            if not agent.active:
                interim_positions[aid] = start_pos
                interim_energies[aid] = cur_e
                moved_flags[aid] = False
                success_flags[aid] = False
                continue

            if act != Action.RECHARGE:
                if end_pos != start_pos:
                    # Successful move
                    interim_positions[aid] = end_pos
                    interim_energies[aid] = cur_e - self.config.movement_energy_cost
                    moved_flags[aid] = True
                    success_flags[aid] = True
                else:
                    # Rejected move (blocked, boundary, collision, or no energy)
                    interim_positions[aid] = start_pos
                    interim_energies[aid] = cur_e
                    moved_flags[aid] = False
                    success_flags[aid] = False
            else:
                # Recharging handled below
                interim_positions[aid] = start_pos
                interim_energies[aid] = cur_e
                moved_flags[aid] = False
                success_flags[aid] = False

        # Next, process atomic resource consumption
        # Multiple agents attempting to consume the same resource cannot duplicate energy!
        remaining_resources = list(self._state.resources)
        recharged_energy: dict[str, int] = {aid: 0 for aid in self._state.agents}
        resource_contentions = 0
        resources_depleted_this_tick = 0

        # Group recharge attempts by resource position
        recharge_claims: dict[Position, list[str]] = {}
        for aid, agent in self._state.agents.items():
            act = actions.get(aid, Action.UP)
            if agent.active and act == Action.RECHARGE:
                recharge_claims.setdefault(interim_positions[aid], []).append(aid)

        for pos, claimants in recharge_claims.items():
            matching_res = next((r for r in remaining_resources if r.position == pos), None)
            if matching_res is None:
                # No resource here
                for aid in claimants:
                    success_flags[aid] = False
                continue

            if len(claimants) > 1:
                resource_contentions += (len(claimants) - 1)
                self.cumulative_resource_contentions += (len(claimants) - 1)

            # Deterministic consumption order: sorted by agent_id or fair arbitration
            if self.config.resource_allocation_rule == "lowest_id":
                sorted_claimants = sorted(claimants)
            else:
                winner, losers = self.resource_arbitrator.arbitrate(
                    claimants,
                    agent_energies=interim_energies,
                    agent_max_energies={aid: self.config.max_energy for aid in claimants},
                    resource=matching_res,
                    tick=self._state.tick,
                )
                sorted_claimants = [winner] + losers

            for idx, aid in enumerate(sorted_claimants):
                if matching_res is None or matching_res not in remaining_resources:
                    # Depleted by previous claimant in this tick!
                    success_flags[aid] = False
                    continue

                cur_e = interim_energies[aid]
                gain = min(matching_res.recharge_amount, self.config.max_energy - cur_e)
                interim_energies[aid] = cur_e + gain
                recharged_energy[aid] = gain
                success_flags[aid] = True

                if matching_res.consumable:
                    remaining_resources.remove(matching_res)
                    resources_depleted_this_tick += 1
                    self.cumulative_resources_depleted += 1
                    matching_res = None  # No longer available for other claimants

        # Phase 5: State update & goal evaluation
        new_agents: dict[str, AgentState] = {}
        step_results: dict[str, AgentStepResult] = {}

        for aid, agent in self._state.agents.items():
            pos = interim_positions[aid]
            e = interim_energies[aid]
            act = actions.get(aid, Action.UP)
            succ = success_flags[aid]
            moved = moved_flags[aid]

            reached = agent.target is not None and pos == agent.target
            exhausted = e == 0 and not any(r.position == pos for r in remaining_resources)

            if reached:
                new_status = "reached_target"
                new_active = False
                if self.coordinator is not None:
                    self.coordinator.clear_commitment(aid)
            elif exhausted:
                new_status = "exhausted"
                new_active = False
                if self.coordinator is not None:
                    self.coordinator.clear_commitment(aid)
            else:
                new_status = agent.status
                new_active = agent.active

            new_agents[aid] = AgentState(
                id=aid,
                position=pos,
                energy=e,
                target=agent.target,
                active=new_active,
                status=new_status,
            )

        # Update world state
        self._state = MultiAgentWorldState(
            width=self._state.width,
            height=self._state.height,
            tick=self._state.tick + 1,
            agents=new_agents,
            resources=tuple(remaining_resources),
            shared_target=self._state.shared_target,
            dynamic_obstacles=next_obs_states,
        )
        self.dynamic_obstacle_history.append(dict(next_obs_states))

        # Phase 6: Cooperative discovery sharing
        if self.config.cooperative_sharing:
            for aid in self._state.agents:
                obs = self.observe_agent(aid)
                for res in obs.observed_resources:
                    self._shared_discoveries.add(res)

        # Post-movement observations for step results
        post_obs = self.observe_all()

        # Evaluate overall task completion
        overall_completed = self._check_overall_completion()

        for aid, agent in self._state.agents.items():
            succ = success_flags[aid]
            moved = moved_flags[aid]
            step_results[aid] = AgentStepResult(
                agent_id=aid,
                success=succ,
                moved=moved,
                action=actions.get(aid, Action.UP),
                position_before=pre_obs[aid].entity_position,
                position_after=new_agents[aid].position,
                energy_before=pre_obs[aid].energy,
                energy_after=new_agents[aid].energy,
                consumed_energy=self.config.movement_energy_cost if moved else 0,
                gained_energy=recharged_energy[aid],
                collision_type=collisions[aid],
                resource_contention=len(recharge_claims.get(pre_obs[aid].entity_position, [])) > 1 and actions.get(aid) == Action.RECHARGE,
                reached_target=new_agents[aid].status == "reached_target",
                exhausted=new_agents[aid].status == "exhausted",
                observation=post_obs[aid],
            )

        # Deadlock detection and fairness monitoring
        intents_for_detector = {aid: (actions.get(aid, Action.UP), final_dests[aid]) for aid in self._state.agents}
        deadlocks = self.deadlock_detector.update_tick(
            tick=self._state.tick,
            agent_positions={aid: a.position for aid, a in self._state.agents.items()},
            intents=intents_for_detector,
            collisions=collisions,
            moved_flags=moved_flags,
            coordinator=self.coordinator,
            resources=tuple(r.position for r in self._state.resources),
        )

        recharge_outcomes = {}
        for pos, claimants in recharge_claims.items():
            for aid in claimants:
                recharge_outcomes[aid] = (recharged_energy.get(aid, 0) > 0)

        self.fairness_monitor.record_tick(
            tick=self._state.tick,
            agent_positions_before={aid: a.position for aid, a in self._state.agents.items()},
            agent_positions_after={aid: a.position for aid, a in new_agents.items()},
            agent_energies_before={aid: a.energy for aid, a in self._state.agents.items()},
            agent_energies_after={aid: a.energy for aid, a in new_agents.items()},
            actions=actions,
            success_flags=success_flags,
            collisions=collisions,
            recharge_claims=recharge_outcomes,
            energy_gained=recharged_energy,
        )

        for aid, a in new_agents.items():
            if a.status in {"reached_target", "exhausted"}:
                self.fairness_monitor.record_final(
                    aid,
                    completed=(a.status == "reached_target"),
                    exhausted=(a.status == "exhausted"),
                )

        return MultiAgentStepResult(
            tick=self._state.tick,
            state=self._state,
            agent_results=step_results,
            vertex_collisions=v_cols,
            swap_collisions=s_cols,
            occupied_collisions=o_cols,
            resource_contentions=resource_contentions,
            resources_depleted=resources_depleted_this_tick,
            overall_completed=overall_completed,
            deadlock_events=tuple(deadlocks),
            blocked_agents=tuple(aid for aid, col in collisions.items() if col is not None),
            dynamic_obstacle_collisions=dyn_cols,
        )

    def _check_overall_completion(self) -> bool:
        """Check if overall goal criterion is satisfied."""
        reached_count = sum(1 for a in self._state.agents.values() if a.status == "reached_target")
        total_agents = len(self._state.agents)

        if self.config.goal_criterion == "individual":
            return reached_count == total_agents
        elif self.config.goal_criterion == "shared_any":
            return reached_count >= 1
        elif self.config.goal_criterion == "shared_all":
            return reached_count == total_agents
        elif self.config.goal_criterion == "shared_subset":
            return reached_count >= self.config.required_goal_count
        return False
