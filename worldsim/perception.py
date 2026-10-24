"""Pure, bounded perception for the simulation."""

from .models import Observation, Position, WorldState


def observe(state: WorldState, radius: float) -> Observation:
    """Return cells and resources whose Euclidean distance from the entity is at most radius.

    A cell at ``(x, y)`` is observed when
    ``(x - entity_x)^2 + (y - entity_y)^2 <= radius^2``. Candidates are
    generated only inside the world's bounds and returned in row-major order.
    Resources within this radius are also included.
    """
    origin = state.entity.position
    radius_squared = radius * radius
    cells = tuple(
        Position(x, y)
        for y in range(state.height)
        for x in range(state.width)
        if (x - origin.x) ** 2 + (y - origin.y) ** 2 <= radius_squared
    )
    observed_resources = tuple(
        resource
        for resource in state.resources
        if (resource.position.x - origin.x) ** 2 + (resource.position.y - origin.y) ** 2 <= radius_squared
    )
    return Observation(
        entity_position=origin,
        cells=cells,
        energy=state.entity.energy,
        observed_resources=observed_resources,
        target=state.observable_target,
    )


