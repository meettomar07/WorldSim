"""Deterministic, headless controlled experiments for WorldSim."""

from dataclasses import dataclass
from typing import Any, Iterable

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .experience import Experience
from .models import Position, StepResult


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Immutable outcome of one ordered action-sequence experiment.

    The runner creates a private simulation and exposes only immutable snapshots.
    ``completed`` is false when an invalid action or rejected action (if
    ``stop_on_rejected`` is enabled) stops execution; in that case
    ``error`` and ``stopped_at_index`` identify the unexecuted action.
    """

    configuration: WorldConfig
    initial_position: Position
    actions: tuple[Action, ...]
    results: tuple[StepResult, ...]
    experiences: tuple[Experience, ...]
    final_position: Position
    final_tick: int
    completed: bool
    error: str | None
    stopped_at_index: int | None
    initial_energy: int = 10
    final_energy: int = 10


def run_experiment(
    config: WorldConfig,
    actions: Iterable[Action],
    stop_on_rejected: bool = False,
) -> ExperimentResult:
    """Execute actions in order through ``Simulation.step``.

    WorldSim currently has no randomness, so no seed is accepted or needed for
    reproducibility. Valid boundary and paused action attempts retain their normal
    simulation semantics. If ``stop_on_rejected`` is True, execution stops as soon
    as an action fails to move or recharge successfully. An invalid action
    (raising ValueError) always stops execution immediately.
    """
    supplied_actions = tuple(actions)
    simulation = Simulation(config)
    initial_position = simulation.state.entity.position
    initial_energy = simulation.state.entity.energy
    results: list[StepResult] = []

    for index, action in enumerate(supplied_actions):
        try:
            step_result = simulation.step(action)
            results.append(step_result)
            if stop_on_rejected and not step_result.success:
                return ExperimentResult(
                    configuration=config,
                    initial_position=initial_position,
                    actions=supplied_actions,
                    results=tuple(results),
                    experiences=simulation.experiences,
                    final_position=simulation.state.entity.position,
                    final_tick=simulation.state.tick,
                    completed=False,
                    error=f"action {action.value} was rejected",
                    stopped_at_index=index,
                    initial_energy=initial_energy,
                    final_energy=simulation.state.entity.energy,
                )
        except ValueError as error:
            return ExperimentResult(
                configuration=config,
                initial_position=initial_position,
                actions=supplied_actions,
                results=tuple(results),
                experiences=simulation.experiences,
                final_position=simulation.state.entity.position,
                final_tick=simulation.state.tick,
                completed=False,
                error=str(error),
                stopped_at_index=index,
                initial_energy=initial_energy,
                final_energy=simulation.state.entity.energy,
            )

    return ExperimentResult(
        configuration=config,
        initial_position=initial_position,
        actions=supplied_actions,
        results=tuple(results),
        experiences=simulation.experiences,
        final_position=simulation.state.entity.position,
        final_tick=simulation.state.tick,
        completed=True,
        error=None,
        stopped_at_index=None,
        initial_energy=initial_energy,
        final_energy=simulation.state.entity.energy,
    )


def run_policy_experiment(
    config: WorldConfig,
    policy: Any,
    max_steps: int = 30,
    stop_on_rejected: bool = False,
) -> ExperimentResult:
    """Execute actions selected dynamically by a decision policy."""
    simulation = Simulation(config)
    initial_position = simulation.state.entity.position
    initial_energy = simulation.state.entity.energy
    actions: list[Action] = []
    results: list[StepResult] = []

    for index in range(max_steps):
        obs = simulation.observe()
        action = policy.choose_action(obs) if hasattr(policy, "choose_action") else policy(obs)
        if action is None:
            break
        actions.append(action)
        try:
            step_result = simulation.step(action)
            results.append(step_result)
            if stop_on_rejected and not step_result.success:
                return ExperimentResult(
                    configuration=config,
                    initial_position=initial_position,
                    actions=tuple(actions),
                    results=tuple(results),
                    experiences=simulation.experiences,
                    final_position=simulation.state.entity.position,
                    final_tick=simulation.state.tick,
                    completed=False,
                    error=f"action {action.value} was rejected",
                    stopped_at_index=index,
                    initial_energy=initial_energy,
                    final_energy=simulation.state.entity.energy,
                )
        except ValueError as error:
            return ExperimentResult(
                configuration=config,
                initial_position=initial_position,
                actions=tuple(actions),
                results=tuple(results),
                experiences=simulation.experiences,
                final_position=simulation.state.entity.position,
                final_tick=simulation.state.tick,
                completed=False,
                error=str(error),
                stopped_at_index=index,
                initial_energy=initial_energy,
                final_energy=simulation.state.entity.energy,
            )

    return ExperimentResult(
        configuration=config,
        initial_position=initial_position,
        actions=tuple(actions),
        results=tuple(results),
        experiences=simulation.experiences,
        final_position=simulation.state.entity.position,
        final_tick=simulation.state.tick,
        completed=True,
        error=None,
        stopped_at_index=None,
        initial_energy=initial_energy,
        final_energy=simulation.state.entity.energy,
    )

