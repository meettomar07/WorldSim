"""Robust resource-aware learning and policy control for WorldSim Phase 12.2.

Provides four modular, transparent improvements over Phase 12.1 linear learning:
1. Intelligent recharge control (RechargeEvaluator).
2. Utility-based resource selection (ResourceUtilityEvaluator).
3. Safe exploration beyond sensory range (ExplorationController & AgentMemory).
4. Failure recovery and cycle detection (CycleDetector).

All components strictly operate within agent-owned memory and permitted observations.
No hidden world state, private simulation state, or classical planners are accessed
during action selection.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from random import Random
import statistics
from typing import Any, Sequence

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .linear_learning import FeatureExtractor, LinearQLearner
from .models import Observation, Position, Resource


# ==============================================================================
# Agent-Owned Observation Memory
# ==============================================================================


@dataclass
class AgentMemory:
    """Agent-owned memory tracking observations, discovered resources, and history.

    Maintained strictly by the agent using only public Observation data.
    Does not inspect private simulation or engine state.
    """

    world_width: int
    world_height: int
    visited_positions: list[Position] = field(default_factory=list)
    visit_counts: dict[Position, int] = field(default_factory=dict)
    observed_cells: set[Position] = field(default_factory=set)
    known_resources: dict[Position, Resource] = field(default_factory=dict)
    depleted_resources: set[Position] = field(default_factory=set)
    action_history: deque[tuple[Position, int, Action, bool]] = field(default_factory=lambda: deque(maxlen=30))
    blocked_destinations: dict[Position, int] = field(default_factory=dict)
    remembered_obstacles: dict[Position, tuple[str, int]] = field(default_factory=dict)
    obstacle_memory_ttl: int = 15
    consecutive_recharges_at_pos: int = 0
    recovery_attempts: int = 0
    exploration_discoveries: int = 0
    step_count: int = 0

    def update(self, observation: Observation) -> None:
        """Update memory with the current step's observation."""
        pos = observation.entity_position
        self.visited_positions.append(pos)
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1
        self.observed_cells.update(observation.cells)

        # Update obstacle memory with local sensory observation
        observed_agents = getattr(observation, "observed_agents", ())
        observed_inactive = {
            other.position: other.status
            for other in observed_agents
            if not getattr(other, "active", True)
        }
        # Visual revalidation: if a cell is within sensory observation, update or clear remembered obstacle
        for cell in observation.cells:
            if cell in self.remembered_obstacles:
                if cell in observed_inactive:
                    self.remembered_obstacles[cell] = (observed_inactive[cell], self.step_count)
                else:
                    del self.remembered_obstacles[cell]

        # Record newly observed inactive obstacles
        for o_pos, o_status in observed_inactive.items():
            self.remembered_obstacles[o_pos] = (o_status, self.step_count)

        # Expire stale obstacle memory based on TTL
        expired = [
            p for p, (_, step) in self.remembered_obstacles.items()
            if self.step_count - step > self.obstacle_memory_ttl
        ]
        for p in expired:
            del self.remembered_obstacles[p]

        # Check for newly discovered resources
        for res in observation.observed_resources:
            if res.position not in self.known_resources and res.position not in self.depleted_resources:
                if self.step_count > 0:
                    self.exploration_discoveries += 1
                self.known_resources[res.position] = res

        # Ingest shared cooperative discoveries if present
        for res in getattr(observation, "shared_discoveries", ()):
            if res.position not in self.known_resources and res.position not in self.depleted_resources:
                self.known_resources[res.position] = res

        # Check for resource depletion at current position
        if pos in self.known_resources:
            res = self.known_resources[pos]
            if res.consumable and not any(r.position == pos for r in observation.observed_resources):
                self.depleted_resources.add(pos)
                del self.known_resources[pos]

        self.step_count += 1

    def record_action_result(self, pos: Position, energy: int, action: Action, success: bool) -> None:
        """Record the outcome of the action."""
        self.action_history.append((pos, energy, action, success))
        if action == Action.RECHARGE:
            self.consecutive_recharges_at_pos += 1
            if success and pos in self.known_resources and self.known_resources[pos].consumable:
                self.depleted_resources.add(pos)
                del self.known_resources[pos]
        else:
            self.consecutive_recharges_at_pos = 0
            # Track destination blocked status
            nx, ny = pos.x, pos.y
            if action == Action.UP:
                ny -= 1
            elif action == Action.DOWN:
                ny += 1
            elif action == Action.LEFT:
                nx -= 1
            elif action == Action.RIGHT:
                nx += 1
            target_cell = Position(nx, ny)
            if not success:
                self.blocked_destinations[target_cell] = self.blocked_destinations.get(target_cell, 0) + 1
            else:
                self.blocked_destinations.clear()

    def available_resources(self) -> tuple[Resource, ...]:
        """Return resources known to be active and not depleted."""
        return tuple(r for pos, r in self.known_resources.items() if pos not in self.depleted_resources)

    def get_remembered_inactive_obstacles(self) -> dict[Position, str]:
        """Return currently remembered inactive obstacles and their status."""
        return {pos: status for pos, (status, _) in self.remembered_obstacles.items()}

    def is_depleted(self, pos: Position) -> bool:
        """Return whether a resource at pos is known to be depleted."""
        return pos in self.depleted_resources

    def unobserved_cells(self) -> set[Position]:
        """Return grid coordinates that have not yet entered sensory observation."""
        all_cells = {
            Position(x, y)
            for x in range(self.world_width)
            for y in range(self.world_height)
        }
        return all_cells - self.observed_cells


# ==============================================================================
# Stage 1: Intelligent Recharge Control
# ==============================================================================


@dataclass(frozen=True, slots=True)
class RechargeAssessment:
    """Evaluation of whether recharging is beneficial and necessary."""

    is_beneficial: bool
    potential_gain: int
    reason: str


class RechargeEvaluator:
    """Determines whether Action.RECHARGE is useful and justified."""

    def __init__(
        self,
        max_energy: int,
        movement_energy_cost: int = 1,
        congestion_safety_margin: int = 0,
        coordinator: Any | None = None,
    ) -> None:
        self.max_energy = max_energy
        self.movement_energy_cost = movement_energy_cost
        self.congestion_safety_margin = congestion_safety_margin
        self.coordinator = coordinator

    def evaluate(
        self,
        observation: Observation,
        agent_memory: AgentMemory,
        recommended_target: Position | None = None,
    ) -> RechargeAssessment:
        """Assess whether recharging right now is beneficial."""
        pos = observation.entity_position
        energy = observation.energy

        # 1. Must be on an observed, active resource
        matching = [r for r in observation.observed_resources if r.position == pos and not agent_memory.is_depleted(pos)]
        if not matching:
            return RechargeAssessment(is_beneficial=False, potential_gain=0, reason="not_on_active_resource")

        resource = matching[0]

        # 2. Potential energy gain calculation
        gain = min(resource.recharge_amount, self.max_energy - energy)
        if gain <= 0:
            return RechargeAssessment(is_beneficial=False, potential_gain=0, reason="energy_already_at_maximum")

        # 3. Target requirement evaluation: is recharge needed to reach target?
        target = observation.target
        if target is not None:
            dist_target = abs(target.x - pos.x) + abs(target.y - pos.y)
            needed_energy = (dist_target * self.movement_energy_cost) + self.congestion_safety_margin
            if energy >= needed_energy:
                return RechargeAssessment(
                    is_beneficial=False,
                    potential_gain=gain,
                    reason="energy_already_sufficient_for_target",
                )

        # 4. If heading toward a specific recommended waypoint (e.g. a different, sufficient resource),
        # avoid unnecessary intermediate recharges if we can reach that waypoint directly and current resource is consumable.
        if resource.consumable and recommended_target is not None and pos != recommended_target:
            dist_waypoint = abs(recommended_target.x - pos.x) + abs(recommended_target.y - pos.y)
            if energy >= dist_waypoint * self.movement_energy_cost:
                return RechargeAssessment(
                    is_beneficial=False,
                    potential_gain=gain,
                    reason="intermediate_recharge_unnecessary",
                )

        # 5. Reusable resource loop prevention: repeated recharge without moving
        if not resource.consumable and agent_memory.consecutive_recharges_at_pos >= 1:
            if target is not None:
                dist_target = abs(target.x - pos.x) + abs(target.y - pos.y)
                needed_energy = (dist_target * self.movement_energy_cost) + self.congestion_safety_margin
                if energy >= needed_energy or energy >= self.max_energy or gain == 0:
                    return RechargeAssessment(
                        is_beneficial=False,
                        potential_gain=gain,
                        reason="reusable_loop_prevention",
                    )
            elif energy >= self.max_energy:
                return RechargeAssessment(
                    is_beneficial=False,
                    potential_gain=gain,
                    reason="reusable_already_full",
                )

        return RechargeAssessment(is_beneficial=True, potential_gain=gain, reason="beneficial_recharge")


# ==============================================================================
# Stage 2: Utility-Based Resource Selection
# ==============================================================================


@dataclass(frozen=True, slots=True)
class ResourceUtility:
    """Detailed utility assessment for an individual resource."""

    resource: Resource
    distance_to_agent: int
    energy_to_reach: int
    reachable_now: bool
    energy_after_recharge: int
    distance_to_target: int
    energy_to_target: int
    target_reachable_from_resource: bool
    net_energy_margin: int
    total_steps: int
    utility_score: float
    is_feasible_route: bool
    status: str


@dataclass(frozen=True, slots=True)
class RouteComparison:
    """Comparison between direct navigation to target vs resource detours."""

    direct_feasible: bool
    direct_distance: int
    direct_margin: int
    best_resource_utility: ResourceUtility | None
    recommended_target: Position | None
    decision: str  # "direct_to_target", "detour_to_resource", "uncertain_explore", "infeasible"


class ResourceUtilityEvaluator:
    """Transparent utility calculation evaluating routes to target via resources."""

    def __init__(
        self,
        max_energy: int,
        movement_energy_cost: int = 1,
        congestion_safety_margin: int = 0,
        coordinator: Any | None = None,
    ) -> None:
        self.max_energy = max_energy
        self.movement_energy_cost = movement_energy_cost
        self.congestion_safety_margin = congestion_safety_margin
        self.coordinator = coordinator

    def evaluate_resource(
        self,
        resource: Resource,
        pos: Position,
        energy: int,
        target: Position | None,
    ) -> ResourceUtility:
        """Compute utility metrics for visiting a single resource."""
        dist_r = abs(resource.position.x - pos.x) + abs(resource.position.y - pos.y)
        cost_r = dist_r * self.movement_energy_cost
        reachable_now = energy >= cost_r

        if reachable_now:
            energy_arrive = energy - cost_r
            if not resource.consumable:
                energy_post = self.max_energy
                gain = self.max_energy - energy_arrive
                num_recharges = max(1, (gain + resource.recharge_amount - 1) // resource.recharge_amount) if gain > 0 else 1
            else:
                gain = min(resource.recharge_amount, self.max_energy - energy_arrive)
                energy_post = energy_arrive + gain
                num_recharges = 1
        else:
            energy_arrive = -1
            gain = 0
            energy_post = -1
            num_recharges = 0

        if target is not None:
            dist_target = abs(target.x - resource.position.x) + abs(target.y - resource.position.y)
            cost_target = dist_target * self.movement_energy_cost
            target_reachable = reachable_now and (energy_post >= cost_target)
            net_margin = energy_post - cost_target if reachable_now else -999
            total_steps = dist_r + num_recharges + dist_target
        else:
            dist_target = 0
            cost_target = 0
            target_reachable = False
            net_margin = gain - cost_r
            total_steps = dist_r + num_recharges

        is_feasible = target_reachable if target is not None else reachable_now

        # Transparent utility scoring
        if is_feasible:
            # Positive utility: rewards surplus energy margin, penalizes unnecessary detour steps
            bonus = 30.0 if dist_r == 0 else 0.0
            score = 100.0 + bonus + (net_margin * 2.0) - (total_steps * 0.5)
            status = "sufficient"
        elif reachable_now:
            # Can reach resource, but target remains unreachable from there
            score = -50.0 + net_margin
            status = "insufficient"
        else:
            # Cannot even reach resource
            score = -100.0 - cost_r
            status = "unreachable"

        return ResourceUtility(
            resource=resource,
            distance_to_agent=dist_r,
            energy_to_reach=cost_r,
            reachable_now=reachable_now,
            energy_after_recharge=energy_post,
            distance_to_target=dist_target,
            energy_to_target=cost_target,
            target_reachable_from_resource=target_reachable,
            net_energy_margin=net_margin,
            total_steps=total_steps,
            utility_score=score,
            is_feasible_route=is_feasible,
            status=status,
        )

    def compare_routes(
        self,
        observation: Observation,
        agent_memory: AgentMemory,
    ) -> RouteComparison:
        """Compare direct target route against visiting available resources."""
        pos = observation.entity_position
        energy = observation.energy
        target = observation.target

        # 1. Direct path evaluation
        if target is not None:
            direct_dist = abs(target.x - pos.x) + abs(target.y - pos.y)
            direct_cost = direct_dist * self.movement_energy_cost
            direct_feasible = energy >= direct_cost
            direct_margin = energy - direct_cost
        else:
            direct_dist = 0
            direct_feasible = False
            direct_margin = -999

        # 2. Evaluate all known active resources
        resources = agent_memory.available_resources()
        utilities = [self.evaluate_resource(r, pos, energy, target) for r in resources]

        aid = getattr(observation, "agent_id", None)
        if self.coordinator is not None and len(utilities) > 1:
            def util_rank(u: ResourceUtility) -> float:
                load = self.coordinator.get_resource_load(u.resource.position, exclude_agent=aid)
                return u.utility_score - (load * 30.0)
            best_util = max(utilities, key=util_rank) if utilities else None
        else:
            best_util = max(utilities, key=lambda u: u.utility_score) if utilities else None

        # 3. Decision synthesis
        if direct_feasible:
            # Direct movement to target is fully feasible with current energy
            return RouteComparison(
                direct_feasible=True,
                direct_distance=direct_dist,
                direct_margin=direct_margin,
                best_resource_utility=best_util,
                recommended_target=target,
                decision="direct_to_target",
            )

        if best_util is not None and best_util.is_feasible_route:
            # Direct is infeasible, but a feasible detour exists
            return RouteComparison(
                direct_feasible=False,
                direct_distance=direct_dist,
                direct_margin=direct_margin,
                best_resource_utility=best_util,
                recommended_target=best_util.resource.position,
                decision="detour_to_resource",
            )

        # Neither direct nor any single known resource completes the route
        return RouteComparison(
            direct_feasible=False,
            direct_distance=direct_dist,
            direct_margin=direct_margin,
            best_resource_utility=best_util,
            recommended_target=None,
            decision="uncertain_explore",
        )


# ==============================================================================
# Stage 3: Safe Exploration Beyond Sensory Range
# ==============================================================================


class ExplorationController:
    """Guides safe exploration toward unobserved grid frontiers."""

    def __init__(
        self,
        world_width: int,
        world_height: int,
        movement_energy_cost: int = 1,
        sensory_radius: float = 2.0,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self.movement_energy_cost = movement_energy_cost
        self.sensory_radius_sq = sensory_radius * sensory_radius

    def evaluate_exploratory_actions(
        self,
        observation: Observation,
        agent_memory: AgentMemory,
        legal_actions: Sequence[Action],
    ) -> dict[Action, float]:
        """Score candidate movement actions by information gain and visit penalty."""
        pos = observation.entity_position
        energy = observation.energy
        target = observation.target
        unobserved = agent_memory.unobserved_cells()

        action_scores: dict[Action, float] = {}

        for action in legal_actions:
            if action == Action.RECHARGE:
                continue

            nx, ny = pos.x, pos.y
            if action == Action.UP:
                ny -= 1
            elif action == Action.DOWN:
                ny += 1
            elif action == Action.LEFT:
                nx -= 1
            elif action == Action.RIGHT:
                nx += 1

            npos = Position(nx, ny)

            # Calculate information gain: newly observable cells from npos
            info_gain = sum(
                1 for cell in unobserved
                if (cell.x - nx) ** 2 + (cell.y - ny) ** 2 <= self.sensory_radius_sq
            )

            # Novelty bonus and revisit penalty
            visits = agent_memory.visit_counts.get(npos, 0)
            score = (info_gain * 3.0) - (visits * 2.0)

            # Target direction heuristic: if target is known, bias toward target
            if target is not None:
                cur_dist = abs(target.x - pos.x) + abs(target.y - pos.y)
                new_dist = abs(target.x - nx) + abs(target.y - ny)
                if new_dist < cur_dist:
                    score += 1.0
                elif new_dist > cur_dist:
                    score -= 0.5

            action_scores[action] = score

        return action_scores


# ==============================================================================
# Stage 4: Failure Recovery and Cycle Detection
# ==============================================================================


@dataclass(frozen=True, slots=True)
class CycleDetectionResult:
    """Result of cycle analysis over recent action history."""

    detected: bool
    cycle_type: str  # "recharge_loop", "ping_pong", "position_loop", "repeated_rejection", "none"
    penalized_action: Action | None
    suggested_alternative: Action | None
    reason: str


class CycleDetector:
    """Detects unproductive action loops and oscillations."""

    def __init__(self, history_len: int = 12) -> None:
        self.history_len = history_len

    def detect(self, agent_memory: AgentMemory) -> CycleDetectionResult:
        """Analyze recent history for unproductive cyclic behavior."""
        hist = list(agent_memory.action_history)
        if len(hist) < 2:
            return CycleDetectionResult(False, "none", None, None, "insufficient_history")

        # 1. Repeated recharge loop: last 2 actions were RECHARGE
        if len(hist) >= 2:
            (p1, e1, a1, s1), (p2, e2, a2, s2) = hist[-2], hist[-1]
            if a1 == Action.RECHARGE and a2 == Action.RECHARGE:
                return CycleDetectionResult(
                    detected=True,
                    cycle_type="recharge_loop",
                    penalized_action=Action.RECHARGE,
                    suggested_alternative=None,
                    reason="consecutive_recharges_detected",
                )

        # 2. Repeated rejection: last 2 actions failed
        if len(hist) >= 2:
            (_, _, a1, s1), (_, _, a2, s2) = hist[-2], hist[-1]
            if not s1 and not s2 and a1 == a2:
                return CycleDetectionResult(
                    detected=True,
                    cycle_type="repeated_rejection",
                    penalized_action=a2,
                    suggested_alternative=None,
                    reason="action_repeatedly_rejected",
                )

        # 3. Ping-pong oscillation: pos0 -> pos1 -> pos0 -> pos1 (length 4)
        if len(hist) >= 4:
            (p0, _, a0, _), (p1, _, a1, _), (p2, _, a2, _), (p3, _, a3, _) = hist[-4:]
            if p0 == p2 and p1 == p3 and p0 != p1:
                # Agent is oscillating between p0 and p1
                return CycleDetectionResult(
                    detected=True,
                    cycle_type="ping_pong",
                    penalized_action=a3,
                    suggested_alternative=None,
                    reason="2_step_ping_pong_oscillation",
                )

        # 4. Position loop: repeated visits to current position without progress
        cur_pos = hist[-1][0]
        recent_positions = [h[0] for h in hist[-8:]]
        if recent_positions.count(cur_pos) >= 3:
            return CycleDetectionResult(
                detected=True,
                cycle_type="position_loop",
                penalized_action=hist[-1][2],
                suggested_alternative=None,
                reason="position_revisited_3_times_recently",
            )

        return CycleDetectionResult(False, "none", None, None, "normal_progression")


# ==============================================================================
# Robust Resource-Aware Policy Wrapper
# ==============================================================================


class RobustResourcePolicy:
    """Observation-limited robust policy integrating all four Phase 12.2 improvements.

    Wraps a base policy (or LinearQLearner) and evaluates:
    1. Intelligent recharge control (RechargeEvaluator).
    2. Utility-based resource selection (ResourceUtilityEvaluator).
    3. Safe exploration beyond sensory range (ExplorationController).
    4. Failure recovery and cycle detection (CycleDetector).

    Preserves strict observation boundaries and uses only agent-owned memory.
    """

    def __init__(
        self,
        world_width: int,
        world_height: int,
        max_energy: int,
        movement_energy_cost: int = 1,
        sensory_radius: float = 2.0,
        base_learner: LinearQLearner | None = None,
        max_recovery_budget: int = 15,
        congestion_safety_margin: int = 0,
        coordinator: Any | None = None,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self.max_energy = max_energy
        self.movement_energy_cost = movement_energy_cost
        self.sensory_radius = sensory_radius
        self.base_learner = base_learner
        # Ensure base learner feature extractor matches current world dimensions
        if self.base_learner is not None:
            fe = self.base_learner.feature_extractor
            if fe.world_width != world_width or fe.world_height != world_height:
                fe.world_width = world_width
                fe.world_height = world_height
                fe.max_dist = max(1, (world_width - 1) + (world_height - 1))
        self.max_recovery_budget = max_recovery_budget
        self.congestion_safety_margin = congestion_safety_margin
        self.coordinator = coordinator

        self.memory = AgentMemory(world_width=world_width, world_height=world_height)
        self.recharge_evaluator = RechargeEvaluator(
            max_energy=max_energy,
            movement_energy_cost=movement_energy_cost,
            congestion_safety_margin=congestion_safety_margin,
            coordinator=coordinator,
        )
        self.utility_evaluator = ResourceUtilityEvaluator(
            max_energy=max_energy,
            movement_energy_cost=movement_energy_cost,
            congestion_safety_margin=congestion_safety_margin,
            coordinator=coordinator,
        )
        self.exploration_controller = ExplorationController(
            world_width=world_width,
            world_height=world_height,
            movement_energy_cost=movement_energy_cost,
            sensory_radius=sensory_radius,
        )
        self.cycle_detector = CycleDetector()

    def reset_memory(self) -> None:
        """Reset agent memory for a new episode."""
        self.memory = AgentMemory(world_width=self.world_width, world_height=self.world_height)

    def get_legal_actions(self, observation: Observation) -> tuple[Action, ...]:
        """Determine legal actions from observation and geometry bounds."""
        pos = observation.entity_position
        energy = observation.energy
        legal: list[Action] = []

        if energy >= self.movement_energy_cost:
            if pos.y > 0:
                legal.append(Action.UP)
            if pos.y < self.world_height - 1:
                legal.append(Action.DOWN)
            if pos.x > 0:
                legal.append(Action.LEFT)
            if pos.x < self.world_width - 1:
                legal.append(Action.RIGHT)

        if any(r.position == pos and not self.memory.is_depleted(pos) for r in observation.observed_resources):
            legal.append(Action.RECHARGE)

        return tuple(legal) if legal else tuple(Action)

    def _find_path_bfs(
        self,
        start: Position,
        goal: Position,
        obstacles: set[Position],
    ) -> list[Position] | None:
        """Find shortest path from start to goal avoiding stationary obstacles."""
        if start == goal:
            return []
        queue = deque([[start]])
        seen = {start} | (obstacles - {goal})
        while queue:
            path = queue.popleft()
            curr = path[-1]
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = curr.x + dx, curr.y + dy
                if 0 <= nx < self.world_width and 0 <= ny < self.world_height:
                    nxt = Position(nx, ny)
                    if nxt == goal:
                        return path[1:] + [nxt]
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(path + [nxt])
        return None

    def choose_action(self, observation: Observation) -> Action:
        """Select an action using the integrated robustness framework."""
        pos = observation.entity_position
        energy = observation.energy
        target = observation.target

        # 1. Ingest observation into agent memory
        self.memory.update(observation)

        # Target reached check
        if target is not None and pos == target:
            return Action.UP  # Fallback terminal

        legal_actions = self.get_legal_actions(observation)
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 2. Cycle detection & recovery
        cycle_res = self.cycle_detector.detect(self.memory)
        penalized_action = cycle_res.penalized_action if cycle_res.detected else None

        # 3. Utility-based route & resource comparison
        route_comp = self.utility_evaluator.compare_routes(observation, self.memory)

        # 4. Intelligent recharge assessment
        recharge_assessment = self.recharge_evaluator.evaluate(
            observation, self.memory, recommended_target=route_comp.recommended_target
        )

        # 5. Exploratory scores if route is uncertain
        explore_scores: dict[Action, float] = {}
        if route_comp.decision == "uncertain_explore":
            explore_scores = self.exploration_controller.evaluate_exploratory_actions(
                observation, self.memory, legal_actions
            )

        # 6. Synthesize action scores
        best_action = legal_actions[0]
        best_score = float("-inf")

        observed_agents = getattr(observation, "observed_agents", ())
        observed_other_positions = {other.id: other.position for other in observed_agents}
        active_other_positions = {other.id: other.position for other in observed_agents if other.active}
        exhausted_positions = {other.position for other in observed_agents if other.status == "exhausted"}
        completed_positions = {other.position for other in observed_agents if other.status == "reached_target"}
        all_inactive_positions = {other.position for other in observed_agents if not other.active}

        # Incorporate remembered inactive obstacles from local observation history
        for rem_pos, rem_status in self.memory.get_remembered_inactive_obstacles().items():
            all_inactive_positions.add(rem_pos)
            if rem_status == "exhausted":
                exhausted_positions.add(rem_pos)
            elif rem_status == "reached_target":
                completed_positions.add(rem_pos)

        rec_target = route_comp.recommended_target
        bfs_next_cell = None
        if route_comp.decision == "direct_to_target" and rec_target is not None:
            # Phase 13.3: First attempt strictly avoiding all inactive obstacles
            path = self._find_path_bfs(pos, rec_target, all_inactive_positions)
            if not path or len(path) > energy:
                # Fallback: completed agents on goal cells yield right-of-way!
                # Only dead exhausted agents remain impassable physical obstacles.
                path = self._find_path_bfs(pos, rec_target, exhausted_positions)
            if path:
                bfs_next_cell = path[0]

        for action in legal_actions:
            score = 0.0

            # Apply Recharge Control
            if action == Action.RECHARGE:
                if not recharge_assessment.is_beneficial:
                    score -= 50.0  # Heavily penalize wasteful recharge
                else:
                    score += 35.0  # Strongly favor beneficial recharge
            else:
                nx, ny = pos.x, pos.y
                if action == Action.UP:
                    ny -= 1
                elif action == Action.DOWN:
                    ny += 1
                elif action == Action.LEFT:
                    nx -= 1
                elif action == Action.RIGHT:
                    nx += 1
                dest_pos = Position(nx, ny)

                is_goal_or_res = (target is not None and dest_pos == target) or any(
                    r.position == dest_pos for r in self.memory.available_resources()
                )

                # Waypoint & BFS guidance toward recommended target (target or resource)
                if bfs_next_cell is not None:
                    if dest_pos == bfs_next_cell:
                        score += 30.0
                    else:
                        cur_dist = abs(rec_target.x - pos.x) + abs(rec_target.y - pos.y)
                        new_dist = abs(rec_target.x - nx) + abs(rec_target.y - ny)
                        if new_dist < cur_dist:
                            score += 10.0
                        elif new_dist > cur_dist:
                            score -= 25.0
                elif rec_target is not None:
                    cur_dist = abs(rec_target.x - pos.x) + abs(rec_target.y - pos.y)
                    new_dist = abs(rec_target.x - nx) + abs(rec_target.y - ny)
                    if new_dist < cur_dist:
                        score += 25.0
                    elif new_dist > cur_dist:
                        score -= 25.0

                # Vital resource reachability protection: never step away if it makes the resource unreachable!
                if route_comp.decision == "detour_to_resource" and rec_target is not None:
                    dist_r = abs(rec_target.x - nx) + abs(rec_target.y - ny)
                    if dist_r > (energy - self.movement_energy_cost):
                        score -= 100.0

                # Permanent obstacle avoidance: dead exhausted agent is impassable
                if dest_pos in exhausted_positions:
                    score -= 60.0
                elif dest_pos in completed_positions and dest_pos != bfs_next_cell and not is_goal_or_res:
                    score -= 15.0

                # Congestion avoidance (gentle tie-breaker)
                if not is_goal_or_res and dest_pos in observed_other_positions.values():
                    score -= 5.0

                # Exploration guidance
                if action in explore_scores:
                    score += explore_scores[action]

                # Revisit penalty to prevent oscillating
                visits = self.memory.visit_counts.get(dest_pos, 0)
                score -= visits * 0.5

                # Congestion avoidance: penalize recently blocked destinations unless it's the target or a known resource
                if not is_goal_or_res:
                    blocked_times = self.memory.blocked_destinations.get(dest_pos, 0)
                    if blocked_times > 0:
                        score -= min(40.0, blocked_times * 15.0)

            # Apply Cycle Penalties
            if action == penalized_action and self.memory.recovery_attempts < self.max_recovery_budget:
                score -= 20.0
                self.memory.recovery_attempts += 1

            if score > best_score:
                best_score = score
                best_action = action

        # Coordinator de-confliction & intent registration if coordinator present
        if self.coordinator is not None:
            dest_map = {}
            for a in legal_actions:
                if a == Action.RECHARGE:
                    dest_map[a] = pos
                else:
                    nx, ny = pos.x, pos.y
                    if a == Action.UP:
                        ny -= 1
                    elif a == Action.DOWN:
                        ny += 1
                    elif a == Action.LEFT:
                        nx -= 1
                    elif a == Action.RIGHT:
                        nx += 1
                    dest_map[a] = Position(nx, ny)
            aid = getattr(observation, "agent_id", "agent")
            rec_target = route_comp.recommended_target
            target_for_coord = rec_target if rec_target is not None else getattr(observation, "target", None)
            alt = self.coordinator.suggest_alternative_move(
                aid,
                pos,
                legal_actions,
                dest_map,
                self.memory.step_count,
                preferred_action=best_action,
                current_positions=observed_other_positions,
                target=target_for_coord,
                energy=energy,
                completed_positions=completed_positions,
                obstacle_positions=exhausted_positions,
            )
            if alt is not None:
                best_action = alt

            dest = dest_map.get(best_action, pos)
            self.coordinator.register_intent(aid, dest, rec_target)

        return best_action
