"""Planning algorithms for WorldSim: geometric BFS and energy-aware state-space search."""

from collections import deque
from dataclasses import dataclass

from .actions import Action, destination
from .config import WorldConfig
from .engine import Simulation
from .models import Position, Resource, StepResult


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    """Explicit geometric planning inputs: grid dimensions, start coordinate, and target coordinate."""

    width: int
    height: int
    start: Position
    target: Position

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or not self.contains(self.start) or not self.contains(self.target):
            raise ValueError("dimensions must be positive and start/target must be in bounds.")

    def contains(self, position: Position) -> bool:
        """Return whether a zero-based coordinate belongs to this planning space."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Immutable result of a geometric planning query."""

    found: bool
    actions: tuple[Action, ...]


def plan(problem: PlanningProblem) -> PlanResult:
    """Find a shortest-path action sequence to the target using breadth-first search."""
    if problem.start == problem.target:
        return PlanResult(True, ())

    queue: deque[tuple[Position, tuple[Action, ...]]] = deque([(problem.start, ())])
    seen: set[Position] = {problem.start}

    while queue:
        position, path = queue.popleft()
        for action in Action:
            next_position = destination(position, action)
            if not problem.contains(next_position) or next_position in seen:
                continue
            next_path = path + (action,)
            if next_position == problem.target:
                return PlanResult(True, next_path)
            seen.add(next_position)
            queue.append((next_position, next_path))

    return PlanResult(False, ())


def execute_plan(config: WorldConfig, result: PlanResult) -> tuple[StepResult, ...]:
    """Execute a plan's actions through the simulation API in a new simulation."""
    simulation = Simulation(config)
    outcomes: list[StepResult] = []
    for action in result.actions:
        outcome = simulation.step(action)
        outcomes.append(outcome)
        if not outcome.moved:
            break
    return tuple(outcomes)


# ============================================================================
# Phase 10: Energy-Aware State-Space Planning
# ============================================================================


@dataclass(frozen=True, slots=True)
class EnergyPlanningProblem:
    """Privileged planning problem incorporating grid, energy bounds, and resources."""

    width: int
    height: int
    start: Position
    target: Position
    initial_energy: int = 10
    max_energy: int = 10
    movement_energy_cost: int = 1
    resources: tuple[Resource, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive.")
        if not self.contains(self.start) or not self.contains(self.target):
            raise ValueError("start and target must be in bounds.")
        if self.max_energy <= 0:
            raise ValueError("max_energy must be positive.")
        if not (0 <= self.initial_energy <= self.max_energy):
            raise ValueError("initial_energy must be in [0, max_energy].")
        if self.movement_energy_cost < 0:
            raise ValueError("movement_energy_cost must not be negative.")

        resources_tuple = tuple(self.resources)
        object.__setattr__(self, "resources", resources_tuple)

        for resource in resources_tuple:
            if not isinstance(resource, Resource):
                raise ValueError(f"expected Resource instance; got {resource!r}")
            if not self.contains(resource.position):
                raise ValueError(f"resource position {resource.position} is out of bounds.")
            if resource.recharge_amount <= 0:
                raise ValueError("resource recharge_amount must be positive.")

        if len({r.position for r in resources_tuple}) != len(resources_tuple):
            raise ValueError("duplicate resource positions are not allowed.")

    def contains(self, position: Position) -> bool:
        """Return whether a zero-based coordinate belongs to this planning space."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height


@dataclass(frozen=True, slots=True)
class EnergyPlanResult:
    """Immutable result of an energy-aware planning query."""

    found: bool
    actions: tuple[Action, ...]
    expected_final_position: Position | None
    expected_final_energy: int | None
    plan_length: int
    failure_reason: str | None


def plan_energy(problem: EnergyPlanningProblem) -> EnergyPlanResult:
    """Find a minimal-action sequence reaching the target considering energy and resources."""
    if problem.start == problem.target:
        return EnergyPlanResult(
            found=True,
            actions=(),
            expected_final_position=problem.start,
            expected_final_energy=problem.initial_energy,
            plan_length=0,
            failure_reason=None,
        )

    resource_map: dict[Position, Resource] = {r.position: r for r in problem.resources}
    initial_active = frozenset(r.position for r in problem.resources)

    # Search state: (position, energy, active_resources_frozenset)
    queue: deque[tuple[Position, int, frozenset[Position], tuple[Action, ...]]] = deque(
        [(problem.start, problem.initial_energy, initial_active, ())]
    )
    seen: set[tuple[Position, int, frozenset[Position]]] = {
        (problem.start, problem.initial_energy, initial_active)
    }

    while queue:
        pos, energy, active_res, path = queue.popleft()

        if pos == problem.target:
            return EnergyPlanResult(
                found=True,
                actions=path,
                expected_final_position=pos,
                expected_final_energy=energy,
                plan_length=len(path),
                failure_reason=None,
            )

        # 1. Movement transitions
        if energy >= problem.movement_energy_cost:
            next_energy = energy - problem.movement_energy_cost
            for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                next_pos = destination(pos, action)
                if not problem.contains(next_pos):
                    continue
                state_key = (next_pos, next_energy, active_res)
                if state_key not in seen:
                    seen.add(state_key)
                    queue.append((next_pos, next_energy, active_res, path + (action,)))

        # 2. Recharge transitions
        if pos in active_res:
            res = resource_map[pos]
            recharged_energy = min(problem.max_energy, energy + res.recharge_amount)
            if recharged_energy > energy:
                next_active = active_res - {pos} if res.consumable else active_res
                state_key = (pos, recharged_energy, next_active)
                if state_key not in seen:
                    seen.add(state_key)
                    queue.append((pos, recharged_energy, next_active, path + (Action.RECHARGE,)))

    return EnergyPlanResult(
        found=False,
        actions=(),
        expected_final_position=None,
        expected_final_energy=None,
        plan_length=0,
        failure_reason="target unreachable within available energy and resource constraints",
    )


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    """Outcome of safely executing an energy-aware plan through Simulation.step."""

    completed: bool
    reached_target: bool
    actions_executed: tuple[Action, ...]
    results: tuple[StepResult, ...]
    final_position: Position
    final_energy: int
    planned_length: int
    actual_steps: int
    divergence_reason: str | None


def execute_energy_plan(
    config: WorldConfig,
    problem: EnergyPlanningProblem,
    plan_result: EnergyPlanResult,
) -> PlanExecutionResult:
    """Safely execute an energy plan, verifying start state and halting on divergence."""
    if (
        config.width != problem.width
        or config.height != problem.height
        or config.initial_x != problem.start.x
        or config.initial_y != problem.start.y
    ):
        raise ValueError("simulation dimensions and initial position must match the planning problem.")
    if (
        config.initial_energy != problem.initial_energy
        or config.max_energy != problem.max_energy
        or config.movement_energy_cost != problem.movement_energy_cost
    ):
        raise ValueError("simulation energy configuration must match the planning problem.")
    if set(config.resources) != set(problem.resources):
        raise ValueError("simulation resources must match the planning problem.")

    if not plan_result.found:
        return PlanExecutionResult(
            completed=False,
            reached_target=False,
            actions_executed=(),
            results=(),
            final_position=problem.start,
            final_energy=problem.initial_energy,
            planned_length=0,
            actual_steps=0,
            divergence_reason=f"cannot execute unfound plan: {plan_result.failure_reason}",
        )

    simulation = Simulation(config)
    results: list[StepResult] = []
    actions_executed: list[Action] = []

    for index, action in enumerate(plan_result.actions):
        actions_executed.append(action)
        step_result = simulation.step(action)
        results.append(step_result)
        if not step_result.success:
            return PlanExecutionResult(
                completed=False,
                reached_target=simulation.state.entity.position == problem.target,
                actions_executed=tuple(actions_executed),
                results=tuple(results),
                final_position=simulation.state.entity.position,
                final_energy=simulation.state.entity.energy,
                planned_length=plan_result.plan_length,
                actual_steps=len(results),
                divergence_reason=f"action {action.value} failed at step {index + 1}",
            )

    reached = simulation.state.entity.position == problem.target
    divergence = None if reached else "executed all planned actions but entity did not reach target"
    return PlanExecutionResult(
        completed=True,
        reached_target=reached,
        actions_executed=tuple(actions_executed),
        results=tuple(results),
        final_position=simulation.state.entity.position,
        final_energy=simulation.state.entity.energy,
        planned_length=plan_result.plan_length,
        actual_steps=len(results),
        divergence_reason=divergence,
    )


