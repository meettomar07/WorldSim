"""Small, explicit tabular Q-learning experiment layer for WorldSim."""

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .models import Observation, Position
from .memory import ExperienceMemory

StateKey = tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class LearningConfig:
    target: Position
    step_cost: float = -0.01
    target_reward: float = 1.0
    failed_action_penalty: float | None = None
    recharge_cost: float | None = None
    terminal_failure_penalty: float = 0.0
    learning_rate: float = 0.3
    discount_factor: float = 0.9
    epsilon: float = 0.2
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01
    max_steps: int = 30
    include_energy: bool = False
    include_resources: bool = False
    include_target: bool = False
    initial_q_value: float = 0.0
    terminal_on_energy_exhaustion: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1 or not 0 <= self.discount_factor <= 1:
            raise ValueError("learning_rate must be in (0, 1] and discount_factor in [0, 1].")
        if not 0 <= self.epsilon <= 1 or not 0 < self.epsilon_decay <= 1 or not 0 <= self.epsilon_min <= 1:
            raise ValueError("epsilon, epsilon_decay, and epsilon_min must be in valid ranges.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.target_reward <= 0 or self.step_cost >= 0:
            raise ValueError("target_reward must be positive and step_cost must be negative.")
        if self.failed_action_penalty is not None and self.failed_action_penalty > 0:
            raise ValueError("failed_action_penalty must be non-positive.")
        if self.terminal_failure_penalty > 0:
            raise ValueError("terminal_failure_penalty must be non-positive.")


@dataclass(frozen=True, slots=True)
class LearningTransition:
    state: StateKey
    action: Action
    reward: float
    next_state: StateKey
    terminal: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    transitions: tuple[LearningTransition, ...]
    reached_target: bool
    cumulative_reward: float
    recharge_count: int = 0
    rejected_actions: int = 0
    final_energy: int = 10

    @property
    def steps(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    policy: str
    episodes: int
    successes: int
    success_rate: float
    steps_to_target: tuple[int | None, ...]
    cumulative_rewards: tuple[float, ...]
    recharge_counts: tuple[int, ...] = ()
    rejected_counts: tuple[int, ...] = ()
    final_energies: tuple[int, ...] = ()


class QLearner:
    """Zero-initialized or custom Q-table with deterministic tie-breaking and seeded exploration."""

    def __init__(self, config: LearningConfig, seed: int | None = None) -> None:
        self.config = config
        self._values: dict[StateKey, dict[Action, float]] = {}
        self._random = Random(seed)
        self.epsilon = config.epsilon

    def state_key(self, observation: Observation) -> StateKey:
        pos = observation.entity_position
        parts: list[Any] = [pos.x, pos.y]
        if self.config.include_energy:
            parts.append(observation.energy)
        if self.config.include_target and observation.target is not None:
            parts.append((observation.target.x, observation.target.y))
        if self.config.include_resources:
            res_coords = tuple(sorted((r.position.x, r.position.y) for r in observation.observed_resources))
            parts.append(res_coords)
        return tuple(parts)

    def value(self, state: StateKey, action: Action) -> float:
        return self._values.get(state, {}).get(action, self.config.initial_q_value)

    @property
    def q_table(self) -> tuple[tuple[StateKey, tuple[tuple[Action, float], ...]], ...]:
        return tuple((state, tuple((action, self.value(state, action)) for action in Action)) for state in sorted(self._values))

    def select_action(self, state: StateKey, explore: bool = True) -> Action:
        eps = self.epsilon if explore else 0.0
        if explore and self._random.random() < eps:
            return self._random.choice(tuple(Action))
        return max(Action, key=lambda action: self.value(state, action))

    def update(self, state: StateKey, action: Action, reward: float, next_state: StateKey, terminal: bool) -> float:
        current = self.value(state, action)
        future = 0.0 if terminal else max(self.value(next_state, candidate) for candidate in Action)
        updated = current + self.config.learning_rate * (reward + self.config.discount_factor * future - current)
        self._values.setdefault(state, {})[action] = updated
        return updated

    def decay_epsilon(self) -> None:
        if self.config.epsilon_decay < 1.0:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)


class LearningRunner:
    """Creates fresh worlds per episode while retaining one learner and Q-table."""

    def __init__(self, world: WorldConfig, config: LearningConfig, seed: int | None = None, memory: ExperienceMemory | None = None) -> None:
        if not (0 <= config.target.x < world.width and 0 <= config.target.y < world.height):
            raise ValueError("target must be inside the world bounds.")
        self.world = world
        self.config = config
        self.learner = QLearner(config, seed)
        self.memory = memory

    def _episode(self, learn: bool, policy: str = "learned", rng: Random | None = None, world: WorldConfig | None = None) -> EpisodeResult:
        active_world = world or self.world
        target = active_world.observable_target if active_world.observable_target is not None else self.config.target
        simulation = Simulation(active_world)
        transitions: list[LearningTransition] = []
        reward_total = 0.0
        recharge_count = 0
        rejected_count = 0

        reached = simulation.observe().entity_position == target
        terminal_failure = False

        while not reached and not terminal_failure and len(transitions) < self.config.max_steps:
            observation = simulation.observe()
            state = self.learner.state_key(observation)
            matches = self.memory.retrieve(observation) if self.memory is not None else ()
            if policy == "random":
                action = rng.choice(tuple(Action)) if rng is not None else self.learner._random.choice(tuple(Action))
            elif learn and matches:
                action = max(matches, key=lambda entry: entry.reward).action
            else:
                action = self.learner.select_action(state, explore=learn)

            result = simulation.step(action)
            next_state = self.learner.state_key(result.observation)
            reached = result.observation.entity_position == target

            if action == Action.RECHARGE and result.success:
                recharge_count += 1
            if not result.success:
                rejected_count += 1

            if reached:
                reward = self.config.target_reward
                terminal = True
            elif not result.success:
                reward = (
                    self.config.failed_action_penalty
                    if self.config.failed_action_penalty is not None
                    else self.config.step_cost
                )
                if (
                    self.config.terminal_on_energy_exhaustion
                    and result.observation.energy == 0
                    and not any(r.position == result.observation.entity_position for r in result.observation.observed_resources)
                ):
                    terminal_failure = True
                    reward += self.config.terminal_failure_penalty
                    terminal = True
                else:
                    terminal = False
            else:
                if action == Action.RECHARGE:
                    reward = self.config.recharge_cost if self.config.recharge_cost is not None else self.config.step_cost
                else:
                    reward = self.config.step_cost
                terminal = False

            transition = LearningTransition(state, action, reward, next_state, terminal)
            transitions.append(transition)
            reward_total += reward

            if learn:
                self.learner.update(state, action, reward, next_state, terminal)
                if self.memory is not None:
                    self.memory.record(observation, action, reward, result.observation)

        final_energy = simulation.state.entity.energy
        return EpisodeResult(
            tuple(transitions),
            reached,
            reward_total,
            recharge_count=recharge_count,
            rejected_actions=rejected_count,
            final_energy=final_energy,
        )

    def train(self, episodes: int, world: WorldConfig | None = None) -> tuple[EpisodeResult, ...]:
        if episodes <= 0:
            raise ValueError("episodes must be positive.")
        w = world or self.world
        outcomes = []
        for _ in range(episodes):
            outcomes.append(self._episode(learn=True, world=w))
            self.learner.decay_epsilon()
        return tuple(outcomes)

    def train_tasks(
        self,
        tasks: Sequence[WorldConfig],
        episodes_per_task: int,
    ) -> tuple[EpisodeResult, ...]:
        """Train across a sequence of task environments using round-robin interleaving."""
        if episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive.")
        if not tasks:
            raise ValueError("tasks must not be empty.")
        outcomes = []
        total_episodes = len(tasks) * episodes_per_task
        for ep in range(total_episodes):
            task = tasks[ep % len(tasks)]
            outcomes.append(self._episode(learn=True, world=task))
            self.learner.decay_epsilon()
        return tuple(outcomes)

    def evaluate(self, episodes: int, world: WorldConfig | None = None, seed: int | None = None, policy: str = "learned") -> EvaluationResult:
        if episodes <= 0 or policy not in {"learned", "random"}:
            raise ValueError("episodes must be positive and policy must be 'learned' or 'random'.")
        w = world or self.world
        rng = Random(seed)
        outcomes = tuple(self._episode(learn=False, policy=policy, rng=rng, world=w) for _ in range(episodes))
        successes = sum(outcome.reached_target for outcome in outcomes)
        return EvaluationResult(
            policy=policy,
            episodes=episodes,
            successes=successes,
            success_rate=successes / episodes,
            steps_to_target=tuple(outcome.steps if outcome.reached_target else None for outcome in outcomes),
            cumulative_rewards=tuple(outcome.cumulative_reward for outcome in outcomes),
            recharge_counts=tuple(outcome.recharge_count for outcome in outcomes),
            rejected_counts=tuple(outcome.rejected_actions for outcome in outcomes),
            final_energies=tuple(outcome.final_energy for outcome in outcomes),
        )

