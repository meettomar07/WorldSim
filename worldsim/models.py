"""Immutable domain objects, deliberately free of rendering concerns."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Resource:
    """An explicit, immutable energy resource placed at a grid coordinate."""

    position: Position
    recharge_amount: int = 5
    consumable: bool = True

    def __post_init__(self) -> None:
        if self.recharge_amount <= 0:
            raise ValueError("recharge_amount must be positive.")


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    position: Position
    energy: int = 10


@dataclass(frozen=True, slots=True)
class WorldState:
    width: int
    height: int
    tick: int
    entity: Entity
    resources: tuple[Resource, ...] = ()
    observable_target: Position | None = None

    def contains(self, position: Position) -> bool:
        """Return whether a zero-based coordinate belongs to this world."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height


@dataclass(frozen=True, slots=True)
class Observation:
    """An immutable, bounded view of the world available to the entity."""

    entity_position: Position
    cells: tuple[Position, ...]
    energy: int = 10
    observed_resources: tuple[Resource, ...] = ()
    target: Position | None = None



@dataclass(frozen=True, slots=True)
class StepResult:
    """Immutable result of one externally supplied action attempt."""

    state: WorldState
    observation: Observation
    moved: bool
    success: bool | None = None

    def __post_init__(self) -> None:
        if self.success is None:
            object.__setattr__(self, "success", self.moved)

