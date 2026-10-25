"""Conflict-Based Search (CBS) Optimal Multi-Agent Pathfinding (MAPF) Solver.

Implements the standard Conflict-Based Search algorithm (Sharon et al., 2012, 2015)
with zero external runtime dependencies.

Components:
1. Low-level Solver: Space-Time A* with vertex constraints, edge constraints,
   and dynamic obstacle schedule avoidance.
2. High-level Solver: Best-first Constraint Tree (CT) search resolving vertex
   and edge-swap conflicts via binary constraint branching.
3. CBSPolicy: Integration policy executing CBS plans tick-by-tick within
   WorldSim's MultiAgentSimulation environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import time
from typing import Any, Sequence

from worldsim.actions import Action
from worldsim.dynamic_obstacles import (
    DynamicObstacleConfig,
    DynamicObstacleState,
    predict_next_obstacle_state,
)
from worldsim.models import Observation, Position, Resource
from worldsim.multi_agent import AgentConfig, MultiAgentObservation, MultiAgentWorldConfig


@dataclass(frozen=True)
class VertexConstraint:
    agent_id: str
    position: Position
    time_step: int


@dataclass(frozen=True)
class EdgeConstraint:
    agent_id: str
    from_pos: Position
    to_pos: Position
    time_step: int  # arrival time at to_pos


@dataclass(frozen=True)
class Conflict:
    agent_1: str
    agent_2: str
    time_step: int
    conflict_type: str  # "vertex" or "edge"
    position_1: Position  # for vertex: conflict cell; for edge: agent_1 from_pos
    position_2: Position  # for vertex: conflict cell; for edge: agent_1 to_pos (agent_2 from_pos)


@dataclass
class CTNode:
    """High-level Constraint Tree node."""
    constraints: dict[str, set[VertexConstraint | EdgeConstraint]]
    paths: dict[str, list[Position]]
    cost: int
    node_id: int = field(default=0)

    def __lt__(self, other: CTNode) -> bool:
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.node_id < other.node_id


@dataclass
class CBSSolution:
    success: bool
    paths: dict[str, list[Position]]
    cost: int  # sum of costs (SIC)
    makespan: int
    nodes_expanded: int
    runtime_ms: float
    timed_out: bool
    status: str  # "OPTIMAL", "TIMEOUT", "UNSOLVABLE", "MAX_EXPANSIONS"


class SpaceTimeAStar:
    """Low-level single-agent path planner in time-space under constraints."""

    def __init__(
        self,
        width: int,
        height: int,
        walls: set[Position],
        max_time: int = 100,
    ) -> None:
        self.width = width
        self.height = height
        self.walls = walls
        self.max_time = max_time

    @staticmethod
    def manhattan(p1: Position, p2: Position) -> int:
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def plan(
        self,
        agent_id: str,
        start: Position,
        goal: Position,
        constraints: set[VertexConstraint | EdgeConstraint],
        dynamic_obstacles_schedule: dict[int, set[Position]] | None = None,
        max_energy: int | None = None,
    ) -> list[Position] | None:
        """Find the shortest valid path for agent_id respecting constraints."""
        v_constraints: set[tuple[Position, int]] = set()
        e_constraints: set[tuple[Position, Position, int]] = set()

        for c in constraints:
            if c.agent_id != agent_id:
                continue
            if isinstance(c, VertexConstraint):
                v_constraints.add((c.position, c.time_step))
            elif isinstance(c, EdgeConstraint):
                e_constraints.add((c.from_pos, c.to_pos, c.time_step))

        # Earliest time after which goal must be permanently clear
        max_v_time = max((t for (p, t) in v_constraints if p == goal), default=-1)

        # Priority queue item: (f_score, h_score, g_time, position, path)
        open_list: list[tuple[int, int, int, int, Position, list[Position]]] = []
        visited: set[tuple[Position, int]] = set()

        h0 = self.manhattan(start, goal)
        heapq.heappush(open_list, (h0, h0, 0, 0, start, [start]))
        tie_counter = 0

        while open_list:
            f, h, g, _, curr_pos, path = heapq.heappop(open_list)

            # Goal test: reached goal and no subsequent vertex constraint at goal
            if curr_pos == goal and g > max_v_time:
                return path

            if g >= self.max_time:
                continue

            state_key = (curr_pos, g)
            if state_key in visited:
                continue
            visited.add(state_key)

            next_t = g + 1

            # Candidate moves: 4 orthogonal neighbors + STAY (wait)
            candidates = [
                Position(curr_pos.x, curr_pos.y - 1),  # UP
                Position(curr_pos.x, curr_pos.y + 1),  # DOWN
                Position(curr_pos.x - 1, curr_pos.y),  # LEFT
                Position(curr_pos.x + 1, curr_pos.y),  # RIGHT
                curr_pos,                              # STAY
            ]

            for next_pos in candidates:
                # 1. Grid boundaries
                if not (0 <= next_pos.x < self.width and 0 <= next_pos.y < self.height):
                    continue

                # 2. Static walls
                if next_pos in self.walls:
                    continue

                # 3. Energy constraint: movement consumes energy, staying does not
                if max_energy is not None:
                    moves_so_far = sum(1 for i in range(len(path) - 1) if path[i] != path[i + 1])
                    if next_pos != curr_pos:
                        moves_so_far += 1
                    if moves_so_far > max_energy:
                        continue

                # 4. Dynamic obstacles schedule
                if dynamic_obstacles_schedule and next_t in dynamic_obstacles_schedule:
                    if next_pos in dynamic_obstacles_schedule[next_t]:
                        continue

                # 5. Vertex constraint
                if (next_pos, next_t) in v_constraints:
                    continue

                # 6. Edge constraint (swap)
                if (curr_pos, next_pos, next_t) in e_constraints:
                    continue

                next_state_key = (next_pos, next_t)
                if next_state_key in visited:
                    continue

                next_h = self.manhattan(next_pos, goal)
                next_f = next_t + next_h
                tie_counter += 1
                heapq.heappush(
                    open_list,
                    (next_f, next_h, next_t, tie_counter, next_pos, path + [next_pos]),
                )

        return None


class CBSSolver:
    """Standard Conflict-Based Search Multi-Agent Pathfinding Solver."""

    def __init__(
        self,
        width: int,
        height: int,
        walls: set[Position] | None = None,
        max_time: int = 80,
        timeout_seconds: float = 5.0,
        max_expansions: int = 1500,
    ) -> None:
        self.width = width
        self.height = height
        self.walls = walls or set()
        self.max_time = max_time
        self.timeout_seconds = timeout_seconds
        self.max_expansions = max_expansions
        self.low_level = SpaceTimeAStar(width, height, self.walls, max_time)

    @staticmethod
    def detect_first_conflict(paths: dict[str, list[Position]]) -> Conflict | None:
        """Find the earliest vertex or edge conflict between any pair of agent paths."""
        agent_ids = list(paths.keys())
        max_len = max(len(p) for p in paths.values()) if paths else 0

        for t in range(1, max_len + 1):
            # Check vertex conflicts at time t
            positions_at_t: dict[Position, str] = {}
            for aid in agent_ids:
                p = paths[aid]
                pos_t = p[t] if t < len(p) else p[-1]
                if pos_t in positions_at_t:
                    other_aid = positions_at_t[pos_t]
                    return Conflict(
                        agent_1=other_aid,
                        agent_2=aid,
                        time_step=t,
                        conflict_type="vertex",
                        position_1=pos_t,
                        position_2=pos_t,
                    )
                positions_at_t[pos_t] = aid

            # Check edge swap conflicts between t-1 and t
            for i in range(len(agent_ids)):
                aid_1 = agent_ids[i]
                p1 = paths[aid_1]
                u1 = p1[t - 1] if (t - 1) < len(p1) else p1[-1]
                v1 = p1[t] if t < len(p1) else p1[-1]

                for j in range(i + 1, len(agent_ids)):
                    aid_2 = agent_ids[j]
                    p2 = paths[aid_2]
                    u2 = p2[t - 1] if (t - 1) < len(p2) else p2[-1]
                    v2 = p2[t] if t < len(p2) else p2[-1]

                    # Swap across edge: u1 -> v1 while u2 -> v2 where u1 == v2 and v1 == u2
                    if u1 == v2 and v1 == u2 and u1 != v1:
                        return Conflict(
                            agent_1=aid_1,
                            agent_2=aid_2,
                            time_step=t,
                            conflict_type="edge",
                            position_1=u1,
                            position_2=v1,
                        )

        return None

    @staticmethod
    def compute_sum_of_costs(paths: dict[str, list[Position]], goals: dict[str, Position]) -> int:
        """Compute Sum of Costs (SIC): steps until agent settles at goal."""
        sic = 0
        for aid, p in paths.items():
            goal = goals[aid]
            # find last index where agent was not at goal
            cost = 0
            for idx in range(len(p) - 1, -1, -1):
                if p[idx] != goal:
                    cost = idx + 1
                    break
            sic += cost
        return sic

    def solve(
        self,
        starts: dict[str, Position],
        goals: dict[str, Position],
        energy_budgets: dict[str, int] | None = None,
        dynamic_obstacles_schedule: dict[int, set[Position]] | None = None,
    ) -> CBSSolution:
        """Execute Conflict-Based Search and return optimal paths."""
        start_time = time.perf_counter()
        agent_ids = sorted(list(starts.keys()))

        # Root node: no constraints
        initial_constraints: dict[str, set[VertexConstraint | EdgeConstraint]] = {
            aid: set() for aid in agent_ids
        }
        initial_paths: dict[str, list[Position]] = {}

        for aid in agent_ids:
            energy_limit = energy_budgets.get(aid) if energy_budgets else None
            path = self.low_level.plan(
                aid,
                starts[aid],
                goals[aid],
                initial_constraints[aid],
                dynamic_obstacles_schedule,
                max_energy=energy_limit,
            )
            if path is None:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return CBSSolution(
                    success=False,
                    paths={},
                    cost=0,
                    makespan=0,
                    nodes_expanded=1,
                    runtime_ms=round(elapsed, 2),
                    timed_out=False,
                    status="UNSOLVABLE",
                )
            initial_paths[aid] = path

        initial_cost = self.compute_sum_of_costs(initial_paths, goals)
        root = CTNode(
            constraints=initial_constraints,
            paths=initial_paths,
            cost=initial_cost,
            node_id=0,
        )

        open_nodes: list[CTNode] = []
        heapq.heappush(open_nodes, root)
        node_counter = 0
        nodes_expanded = 0

        while open_nodes:
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.timeout_seconds:
                return CBSSolution(
                    success=False,
                    paths={},
                    cost=0,
                    makespan=0,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=round(elapsed * 1000.0, 2),
                    timed_out=True,
                    status="TIMEOUT",
                )

            # Check expansion limit
            if nodes_expanded >= self.max_expansions:
                return CBSSolution(
                    success=False,
                    paths={},
                    cost=0,
                    makespan=0,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=round(elapsed * 1000.0, 2),
                    timed_out=False,
                    status="MAX_EXPANSIONS",
                )

            curr_node = heapq.heappop(open_nodes)
            nodes_expanded += 1

            conflict = self.detect_first_conflict(curr_node.paths)
            if conflict is None:
                # Conflict-free optimal solution found
                makespan = max((len(p) - 1 for p in curr_node.paths.values()), default=0)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return CBSSolution(
                    success=True,
                    paths=curr_node.paths,
                    cost=curr_node.cost,
                    makespan=makespan,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=round(elapsed_ms, 2),
                    timed_out=False,
                    status="OPTIMAL",
                )

            # Branching on conflict: create 2 child nodes
            a1, a2 = conflict.agent_1, conflict.agent_2
            t = conflict.time_step

            if conflict.conflict_type == "vertex":
                branch_constraints = [
                    (a1, VertexConstraint(a1, conflict.position_1, t)),
                    (a2, VertexConstraint(a2, conflict.position_2, t)),
                ]
            else:  # edge swap: a1 moved u->v, a2 moved v->u
                u, v = conflict.position_1, conflict.position_2
                branch_constraints = [
                    (a1, EdgeConstraint(a1, u, v, t)),
                    (a2, EdgeConstraint(a2, v, u, t)),
                ]

            for constrained_agent, new_constraint in branch_constraints:
                child_constraints = {
                    aid: set(cons) for aid, cons in curr_node.constraints.items()
                }
                child_constraints[constrained_agent].add(new_constraint)

                child_paths = dict(curr_node.paths)
                energy_limit = energy_budgets.get(constrained_agent) if energy_budgets else None
                new_path = self.low_level.plan(
                    constrained_agent,
                    starts[constrained_agent],
                    goals[constrained_agent],
                    child_constraints[constrained_agent],
                    dynamic_obstacles_schedule,
                    max_energy=energy_limit,
                )

                if new_path is not None:
                    child_paths[constrained_agent] = new_path
                    child_cost = self.compute_sum_of_costs(child_paths, goals)
                    node_counter += 1
                    child_node = CTNode(
                        constraints=child_constraints,
                        paths=child_paths,
                        cost=child_cost,
                        node_id=node_counter,
                    )
                    heapq.heappush(open_nodes, child_node)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return CBSSolution(
            success=False,
            paths={},
            cost=0,
            makespan=0,
            nodes_expanded=nodes_expanded,
            runtime_ms=round(elapsed_ms, 2),
            timed_out=False,
            status="UNSOLVABLE",
        )


class CBSPolicy:
    """Agent policy that executes pre-computed Conflict-Based Search paths."""

    def __init__(
        self,
        agent_id: str,
        planned_path: list[Position] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.path = planned_path or []
        self.current_step = 0

    def set_path(self, path: list[Position]) -> None:
        self.path = list(path)
        self.current_step = 0

    def choose_action(self, observation: MultiAgentObservation | Observation) -> Action:
        """Select action following pre-computed CBS path."""
        curr_pos = observation.entity_position
        target_pos = observation.target

        # If goal reached or path exhausted, remain at target via RECHARGE (0,0 delta)
        if curr_pos == target_pos:
            return Action.RECHARGE

        self.current_step += 1
        if self.current_step >= len(self.path):
            return Action.RECHARGE

        next_pos = self.path[self.current_step]

        dx = next_pos.x - curr_pos.x
        dy = next_pos.y - curr_pos.y

        if dx == 1 and dy == 0:
            return Action.RIGHT
        elif dx == -1 and dy == 0:
            return Action.LEFT
        elif dx == 0 and dy == 1:
            return Action.DOWN
        elif dx == 0 and dy == -1:
            return Action.UP
        else:
            # Staying in place (delta 0, 0)
            return Action.RECHARGE


def build_dynamic_obstacle_schedule(
    dynamic_obstacles: Sequence[DynamicObstacleConfig],
    world_width: int,
    world_height: int,
    walls: set[Position],
    resources: Sequence[Resource] | None = None,
    max_time: int = 60,
) -> dict[int, set[Position]]:
    """Simulate future deterministic positions of dynamic obstacles up to max_time.

    Allows space-time A* to reserve hazard trajectories and avoid moving obstacles.
    """
    schedule: dict[int, set[Position]] = {t: set() for t in range(max_time + 1)}
    if not dynamic_obstacles:
        return schedule

    blocked = set(walls)
    if resources:
        for r in resources:
            blocked.add(r.position)

    curr_states: list[DynamicObstacleState] = [
        DynamicObstacleState(
            id=cfg.id,
            position=Position(cfg.initial_x, cfg.initial_y),
            velocity=cfg.velocity,
        )
        for cfg in dynamic_obstacles
    ]

    for t in range(max_time + 1):
        for s in curr_states:
            schedule[t].add(s.position)
        # Advance to t + 1
        next_states = []
        for s, cfg in zip(curr_states, dynamic_obstacles):
            nxt = predict_next_obstacle_state(s, cfg, world_width, world_height, blocked)
            next_states.append(nxt)
        curr_states = next_states

    return schedule
