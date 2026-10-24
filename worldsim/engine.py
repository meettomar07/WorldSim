"""Deterministic, headless simulation engine for WorldSim."""

from .actions import Action, destination
from .config import WorldConfig
from .experience import Experience
from .models import Entity, Observation, Position, StepResult, WorldState
from .perception import observe


class Simulation:
    """Owns simulation state and clock; it has no UI or timing dependencies."""

    def __init__(self, config: WorldConfig) -> None:
        self._config = config
        self._state = WorldState(
            width=config.width,
            height=config.height,
            tick=0,
            entity=Entity(
                config.entity_id,
                Position(config.initial_x, config.initial_y),
                energy=config.initial_energy,
            ),
            resources=config.resources,
            observable_target=config.observable_target,
        )
        self._paused = False
        self._experiences: list[Experience] = []

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def experiences(self) -> tuple[Experience, ...]:
        """Return the chronological, read-only in-memory action history."""
        return tuple(self._experiences)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def observe(self) -> Observation:
        """Return the entity's bounded observation of the current world state."""
        return observe(self._state, self._config.sensory_radius)

    def step(self, action: Action) -> StepResult:
        """Attempt one external action and return the resulting state and observation.

        A valid action attempt always produces one immutable experience record.
        Successful moves consume movement energy and advance the tick once.
        Boundary attempts, attempts with insufficient energy, and all actions while
        paused are rejected without state, energy, or clock changes.
        RECHARGE restores energy up to max_energy if a resource is present.
        Unknown action values raise ``ValueError`` before any mutation or record.
        """
        if not isinstance(action, Action):
            raise ValueError(f"action must be an Action enum value; got {action!r}")

        tick_before = self._state.tick
        position_before = self._state.entity.position
        energy_before = self._state.entity.energy
        observation_before = self.observe()

        if self._paused:
            result = StepResult(self._state, self.observe(), moved=False, success=False)
        elif action == Action.RECHARGE:
            resource = next((r for r in self._state.resources if r.position == position_before), None)
            if resource is None:
                result = StepResult(self._state, self.observe(), moved=False, success=False)
            else:
                new_energy = min(self._config.max_energy, energy_before + resource.recharge_amount)
                new_resources = (
                    tuple(r for r in self._state.resources if r != resource)
                    if resource.consumable
                    else self._state.resources
                )
                self._state = WorldState(
                    width=self._state.width,
                    height=self._state.height,
                    tick=self._state.tick + 1,
                    entity=Entity(self._state.entity.id, position_before, energy=new_energy),
                    resources=new_resources,
                    observable_target=self._state.observable_target,
                )
                result = StepResult(self._state, self.observe(), moved=False, success=True)
        else:
            if energy_before < self._config.movement_energy_cost:
                result = StepResult(self._state, self.observe(), moved=False, success=False)
            else:
                next_position = destination(position_before, action)
                if not self._state.contains(next_position):
                    result = StepResult(self._state, self.observe(), moved=False, success=False)
                else:
                    new_energy = energy_before - self._config.movement_energy_cost
                    self._state = WorldState(
                        width=self._state.width,
                        height=self._state.height,
                        tick=self._state.tick + 1,
                        entity=Entity(self._state.entity.id, next_position, energy=new_energy),
                        resources=self._state.resources,
                        observable_target=self._state.observable_target,
                    )
                    result = StepResult(self._state, self.observe(), moved=True, success=True)

        self._experiences.append(
            Experience(
                tick_before=tick_before,
                observation_before=observation_before,
                action=action,
                moved=result.moved,
                position_before=position_before,
                position_after=result.state.entity.position,
                observation_after=result.observation,
                tick_after=result.state.tick,
                energy_before=energy_before,
                energy_after=result.state.entity.energy,
                success=result.success,
            )
        )
        return result

    def tick(self) -> WorldState:
        """Advance the clock once, unless paused, and return the new state."""
        if not self._paused:
            self._state = WorldState(
                width=self._state.width,
                height=self._state.height,
                tick=self._state.tick + 1,
                entity=self._state.entity,
                resources=self._state.resources,
                observable_target=self._state.observable_target,
            )
        return self._state

