"""Explicit external actions supported by the Phase 3 simulation."""

from enum import Enum

from .models import Position


class Action(Enum):
    """Actions supported in the grid coordinate system.

    ``x`` increases to the right and ``y`` increases downward, so UP decreases
    ``y`` and LEFT decreases ``x``. RECHARGE interacts with resources in place.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    RECHARGE = "recharge"


def destination(origin: Position, action: Action) -> Position:
    """Calculate the one-cell destination for a validated action."""
    deltas = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
        Action.RECHARGE: (0, 0),
    }
    dx, dy = deltas[action]
    return Position(origin.x + dx, origin.y + dy)
