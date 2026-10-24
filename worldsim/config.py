"""Validated configuration for a WorldSim run."""

from dataclasses import dataclass

from .models import Position, Resource


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Dimensions, energy, resources, and initial state for the world.

    Coordinates are zero-based: x grows left-to-right and y grows top-to-bottom.
    """

    width: int = 20
    height: int = 12
    entity_id: str = "entity-0001"
    initial_x: int = 10
    initial_y: int = 6
    sensory_radius: float = 2.0
    initial_energy: int = 10
    max_energy: int = 10
    movement_energy_cost: int = 1
    resources: tuple[Resource, ...] = ()
    observable_target: Position | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World dimensions must both be positive.")
        if not self.entity_id:
            raise ValueError("entity_id must not be empty.")
        if not (0 <= self.initial_x < self.width):
            raise ValueError("initial_x must be inside the world bounds.")
        if not (0 <= self.initial_y < self.height):
            raise ValueError("initial_y must be inside the world bounds.")
        if self.sensory_radius < 0:
            raise ValueError("sensory_radius must not be negative.")
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
            if not (0 <= resource.position.x < self.width and 0 <= resource.position.y < self.height):
                raise ValueError(f"resource position {resource.position} is out of world bounds.")
            if resource.recharge_amount <= 0:
                raise ValueError("resource recharge_amount must be positive.")

        if len({r.position for r in resources_tuple}) != len(resources_tuple):
            raise ValueError("duplicate resource positions are not allowed.")

        if self.observable_target is not None:
            if not (0 <= self.observable_target.x < self.width and 0 <= self.observable_target.y < self.height):
                raise ValueError("observable_target must be inside the world bounds.")


