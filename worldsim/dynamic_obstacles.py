"""Dynamic obstacle definitions, deterministic movement rules, and trajectory tracking.

Supports:
- Declarative configuration of moving obstacles with distinct movement patterns:
  - "linear_bounce": Moves along velocity (dx, dy); bounces on walls/obstacles/resources.
  - "patrol": Cycles deterministically through a sequence of waypoints.
  - "fixed": Stationary obstacle with tracked state.
- Authoritative physical simulation: zero overlap with walls, resources, agents, or other obstacles.
- Trajectory recording of positions and velocities over time.
- Deterministic conflict resolution and prediction helpers for local-horizon avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .models import Position


@dataclass(frozen=True, slots=True)
class DynamicObstacleConfig:
    """Immutable configuration for a dynamic (moving) obstacle."""

    id: str
    initial_x: int
    initial_y: int
    pattern: str = "linear_bounce"  # "linear_bounce", "patrol", "fixed"
    velocity: tuple[int, int] = (0, 0)  # (dx, dy)
    waypoints: tuple[Position, ...] = ()  # For "patrol": sequence of waypoints
    wait_on_blocked: bool = True  # If blocked and cannot bounce, wait in place

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Dynamic obstacle id must be a non-empty string.")
        if self.pattern not in {"linear_bounce", "patrol", "fixed"}:
            raise ValueError(
                f"Unknown pattern '{self.pattern}'. Must be 'linear_bounce', 'patrol', or 'fixed'."
            )
        if self.pattern == "patrol" and not self.waypoints:
            raise ValueError("Pattern 'patrol' requires at least one waypoint.")


@dataclass(frozen=True, slots=True)
class DynamicObstacleState:
    """Immutable runtime state snapshot of a dynamic obstacle at a given tick."""

    id: str
    position: Position
    velocity: tuple[int, int] = (0, 0)
    waypoint_index: int = 0
    reverse_direction: bool = False


def predict_next_obstacle_state(
    obstacle: DynamicObstacleState,
    config: DynamicObstacleConfig,
    world_width: int,
    world_height: int,
    blocked_cells: set[Position],
) -> DynamicObstacleState:
    """Compute the next position and velocity for a dynamic obstacle deterministically.

    Guarantees:
    - Never steps out of bounds.
    - Never steps into blocked_cells (walls, resources, stationary agents, other obstacles).
    - If forward movement is blocked under linear_bounce, inverts velocity.
    - If blocked under patrol, waits in place until path clears.
    """
    pos = obstacle.position

    if config.pattern == "fixed":
        return DynamicObstacleState(
            id=obstacle.id,
            position=pos,
            velocity=(0, 0),
            waypoint_index=0,
            reverse_direction=False,
        )

    if config.pattern == "linear_bounce":
        vx, vy = obstacle.velocity
        if vx == 0 and vy == 0:
            vx, vy = config.velocity

        if vx == 0 and vy == 0:
            return DynamicObstacleState(
                id=obstacle.id,
                position=pos,
                velocity=(0, 0),
                waypoint_index=0,
                reverse_direction=False,
            )

        cand_x = pos.x + vx
        cand_y = pos.y + vy
        cand_pos = Position(cand_x, cand_y)

        # Check if candidate move is unblocked and in-bounds
        in_bounds = 0 <= cand_x < world_width and 0 <= cand_y < world_height
        if in_bounds and cand_pos not in blocked_cells:
            return DynamicObstacleState(
                id=obstacle.id,
                position=cand_pos,
                velocity=(vx, vy),
                waypoint_index=0,
                reverse_direction=False,
            )

        # Candidate is blocked! Invert velocity (bounce)
        rev_vx = -vx
        rev_vy = -vy
        rev_cand = Position(pos.x + rev_vx, pos.y + rev_vy)
        rev_in_bounds = 0 <= rev_cand.x < world_width and 0 <= rev_cand.y < world_height

        # On the bounce tick, the obstacle stays at pos with the inverted velocity
        # so agents observe the direction change and can react.
        return DynamicObstacleState(
            id=obstacle.id,
            position=pos,
            velocity=(rev_vx, rev_vy),
            waypoint_index=0,
            reverse_direction=not obstacle.reverse_direction,
        )

    if config.pattern == "patrol":
        waypoints = config.waypoints
        n = len(waypoints)
        idx = obstacle.waypoint_index
        rev = obstacle.reverse_direction

        # If currently at target waypoint, advance to next
        if pos == waypoints[idx]:
            if not rev:
                if idx + 1 < n:
                    idx = idx + 1
                else:
                    idx = max(0, idx - 1)
                    rev = True
            else:
                if idx - 1 >= 0:
                    idx = idx - 1
                else:
                    idx = min(n - 1, idx + 1)
                    rev = False

        target_wpt = waypoints[idx]
        dx = 1 if target_wpt.x > pos.x else (-1 if target_wpt.x < pos.x else 0)
        dy = 1 if target_wpt.y > pos.y else (-1 if target_wpt.y < pos.y else 0)

        # Attempt primary coordinate move
        if dx != 0:
            cand = Position(pos.x + dx, pos.y)
            vel = (dx, 0)
            if 0 <= cand.x < world_width and 0 <= cand.y < world_height and cand not in blocked_cells:
                return DynamicObstacleState(
                    id=obstacle.id,
                    position=cand,
                    velocity=vel,
                    waypoint_index=idx,
                    reverse_direction=rev,
                )

        if dy != 0:
            cand = Position(pos.x, pos.y + dy)
            vel = (0, dy)
            if 0 <= cand.x < world_width and 0 <= cand.y < world_height and cand not in blocked_cells:
                return DynamicObstacleState(
                    id=obstacle.id,
                    position=cand,
                    velocity=vel,
                    waypoint_index=idx,
                    reverse_direction=rev,
                )

        # Blocked: wait in place with zero velocity
        return DynamicObstacleState(
            id=obstacle.id,
            position=pos,
            velocity=(0, 0),
            waypoint_index=idx,
            reverse_direction=rev,
        )

    # Fallback
    return obstacle


def predict_obstacle_trajectory(
    obstacle: DynamicObstacleState,
    horizon: int = 2,
    world_width: int | None = None,
    world_height: int | None = None,
) -> list[Position]:
    """Compute near-future predicted positions for an obstacle over the given horizon."""
    predictions: list[Position] = []
    vx, vy = obstacle.velocity
    if vx == 0 and vy == 0:
        return [obstacle.position for _ in range(horizon)]

    curr = obstacle.position
    for _ in range(horizon):
        nxt = Position(curr.x + vx, curr.y + vy)
        if world_width is not None and not (0 <= nxt.x < world_width):
            vx = -vx
            nxt = Position(curr.x + vx, curr.y + vy)
        if world_height is not None and not (0 <= nxt.y < world_height):
            vy = -vy
            nxt = Position(curr.x + vx, curr.y + vy)
        predictions.append(nxt)
        curr = nxt
    return predictions

