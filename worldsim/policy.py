"""Decision policies based on permitted entity observations."""

from random import Random
from typing import Sequence

from .actions import Action
from .models import Observation, Position, Resource


class EnergyAwarePolicy:
    """A heuristic policy that considers remaining energy and observable resources.

    It never inspects hidden simulation state or private configuration; it relies
    strictly on the permitted ``Observation``. If energy is insufficient to reach
    the intended target directly, and a resource is observed within sensory radius,
    it diverts to the resource to recharge before continuing toward the goal.
    If energy is below movement cost and no resource is available, it avoids impossible
    actions.
    """

    def __init__(
        self,
        target: Position | None = None,
        movement_cost: int = 1,
    ) -> None:
        self.target = target
        self.movement_cost = movement_cost

    def _step_towards(self, current: Position, destination: Position) -> Action | None:
        dx = destination.x - current.x
        dy = destination.y - current.y
        if dx > 0:
            return Action.RIGHT
        if dx < 0:
            return Action.LEFT
        if dy > 0:
            return Action.DOWN
        if dy < 0:
            return Action.UP
        return None

    def choose_action(self, observation: Observation) -> Action | None:
        pos = observation.entity_position
        if self.target is not None and pos == self.target:
            return None

        # Check if standing on an observed resource
        on_resource = any(r.position == pos for r in observation.observed_resources)
        if on_resource:
            if self.target is not None:
                dist = abs(self.target.x - pos.x) + abs(self.target.y - pos.y)
                needed = dist * self.movement_cost
                if observation.energy < needed:
                    return Action.RECHARGE
            else:
                return Action.RECHARGE

        # If energy cannot pay for movement and we are not on a resource, avoid impossible action
        if observation.energy < self.movement_cost:
            return None

        # If we have a target and current energy is insufficient or critically low:
        if self.target is not None:
            dist = abs(self.target.x - pos.x) + abs(self.target.y - pos.y)
            needed = dist * self.movement_cost
            if observation.energy <= needed and observation.observed_resources:
                # Target nearest observable resource
                nearest = min(
                    observation.observed_resources,
                    key=lambda r: abs(r.position.x - pos.x) + abs(r.position.y - pos.y),
                )
                if nearest.position == pos:
                    return Action.RECHARGE
                return self._step_towards(pos, nearest.position)
            return self._step_towards(pos, self.target)

        # If no target but resources are seen and energy is low:
        if observation.observed_resources:
            nearest = min(
                observation.observed_resources,
                key=lambda r: abs(r.position.x - pos.x) + abs(r.position.y - pos.y),
            )
            if nearest.position == pos:
                return Action.RECHARGE
            return self._step_towards(pos, nearest.position)

        return None


class EnergyIgnorantPolicy:
    """A policy that navigates toward a target without checking energy or resources."""

    def __init__(self, target: Position) -> None:
        self.target = target

    def choose_action(self, observation: Observation) -> Action | None:
        pos = observation.entity_position
        if pos == self.target:
            return None
        dx = self.target.x - pos.x
        dy = self.target.y - pos.y
        if dx > 0:
            return Action.RIGHT
        if dx < 0:
            return Action.LEFT
        if dy > 0:
            return Action.DOWN
        if dy < 0:
            return Action.UP
        return None


class RandomPolicy:
    """A seeded random policy choosing uniformly among actions."""

    def __init__(
        self,
        seed: int | None = None,
        actions: Sequence[Action] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.RECHARGE),
    ) -> None:
        self._rng = Random(seed)
        self.actions = tuple(actions)

    def choose_action(self, observation: Observation) -> Action:
        return self._rng.choice(self.actions)


class EnergyPlannerPolicy:
    """An observation-limited policy backed by the energy-aware planner.

    It only uses permitted Observation information (energy, position, and visible
    resources). When resources are observed or depleted, it replans dynamically.
    It never inspects hidden simulation state or issues actions known to be invalid.
    """

    def __init__(
        self,
        target: Position,
        world_width: int = 20,
        world_height: int = 12,
        max_energy: int = 10,
        movement_energy_cost: int = 1,
    ) -> None:
        self.target = target
        self.world_width = world_width
        self.world_height = world_height
        self.max_energy = max_energy
        self.movement_energy_cost = movement_energy_cost
        self._known_resources: dict[Position, Resource] = {}
        self._current_plan: list[Action] = []
        self._last_known_count = 0

    def choose_action(self, observation: Observation) -> Action | None:
        from .planning import EnergyPlanningProblem, plan_energy

        pos = observation.entity_position
        if pos == self.target:
            return None

        # Discover or update resources from current observation
        for r in observation.observed_resources:
            self._known_resources[r.position] = r

        # If standing on a consumed resource cell and it's no longer observed, remove it
        if pos in self._known_resources:
            if not any(r.position == pos for r in observation.observed_resources):
                del self._known_resources[pos]

        # Replan if no plan or resource knowledge changed
        replan_needed = (
            not self._current_plan or len(self._known_resources) != self._last_known_count
        )

        if replan_needed:
            self._last_known_count = len(self._known_resources)
            try:
                problem = EnergyPlanningProblem(
                    width=self.world_width,
                    height=self.world_height,
                    start=pos,
                    target=self.target,
                    initial_energy=observation.energy,
                    max_energy=self.max_energy,
                    movement_energy_cost=self.movement_energy_cost,
                    resources=tuple(self._known_resources.values()),
                )
                plan_result = plan_energy(problem)
                if plan_result.found:
                    self._current_plan = list(plan_result.actions)
                else:
                    # Target cannot be reached directly with known resources.
                    # Try planning to reach the closest known resource to recharge.
                    candidate_plans = []
                    for res_pos in self._known_resources:
                        if res_pos != pos:
                            sub_prob = EnergyPlanningProblem(
                                width=self.world_width,
                                height=self.world_height,
                                start=pos,
                                target=res_pos,
                                initial_energy=observation.energy,
                                max_energy=self.max_energy,
                                movement_energy_cost=self.movement_energy_cost,
                                resources=tuple(self._known_resources.values()),
                            )
                            sub_res = plan_energy(sub_prob)
                            if sub_res.found:
                                candidate_plans.append(sub_res)
                    if candidate_plans:
                        best_sub = min(candidate_plans, key=lambda p: p.plan_length)
                        self._current_plan = list(best_sub.actions) + [Action.RECHARGE]
                    else:
                        self._current_plan = []
            except ValueError:
                self._current_plan = []

        if self._current_plan:
            next_action = self._current_plan.pop(0)
            # Safety checks before issuing action
            if next_action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                if observation.energy < self.movement_energy_cost:
                    self._current_plan.clear()
                    return None
            elif next_action == Action.RECHARGE:
                if not any(r.position == pos for r in observation.observed_resources):
                    self._current_plan.clear()
                    return None
            return next_action

        return None

