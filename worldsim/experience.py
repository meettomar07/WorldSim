"""Immutable records of externally supplied simulation actions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import Observation, Position

if TYPE_CHECKING:
    from .actions import Action


@dataclass(frozen=True, slots=True)
class Experience:
    """One complete, immutable record of a valid action attempt.

    Observations and positions are already immutable domain objects, so an
    experience can safely retain them without exposing mutable world state.
    Energy levels before and after the action and interaction success are also recorded.
    """

    tick_before: int
    observation_before: Observation
    action: "Action"
    moved: bool
    position_before: Position
    position_after: Position
    observation_after: Observation
    tick_after: int
    energy_before: int = 10
    energy_after: int = 10
    success: bool | None = None

    def __post_init__(self) -> None:
        if self.success is None:
            object.__setattr__(self, "success", self.moved)

