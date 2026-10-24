"""Feature-based linear function approximation reinforcement learning for WorldSim.

Implements transparent, deterministic, generalizable Q-learning over normalized
features derived strictly from permitted observations and candidate actions.
"""

from dataclasses import dataclass
from random import Random
from typing import Sequence

from .actions import Action
from .config import WorldConfig
from .engine import Simulation
from .learning import EpisodeResult, EvaluationResult, LearningTransition
from .models import Observation, Position, Resource


@dataclass(frozen=True, slots=True)
class LinearLearningConfig:
    """Hyperparameters and configuration for linear function-approximation Q-learning."""

    learning_rate: float = 0.05
    discount_factor: float = 0.95
    epsilon: float = 0.3
    epsilon_decay: float = 0.998
    epsilon_min: float = 0.02
    step_cost: float = -0.01
    target_reward: float = 1.0
    failed_action_penalty: float = -0.05
    recharge_cost: float = -0.01
    terminal_failure_penalty: float = -0.1
    max_steps: int = 30
    terminal_on_energy_exhaustion: bool = True
    gradient_clip: float = 2.0
    mask_invalid_actions: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")
        if not 0 <= self.epsilon <= 1 or not 0 < self.epsilon_decay <= 1 or not 0 <= self.epsilon_min <= 1:
            raise ValueError("epsilon, epsilon_decay, and epsilon_min must be in valid ranges.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.target_reward <= 0 or self.step_cost >= 0:
            raise ValueError("target_reward must be positive and step_cost must be negative.")
        if self.failed_action_penalty > 0:
            raise ValueError("failed_action_penalty must be non-positive.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")


class FeatureExtractor:
    """Extracts normalized, relative features from permitted observations and candidate actions.

    The feature extractor uses only permitted Observation fields (position, energy,
    cells, observed resources, and observable target) plus domain geometry bounds.
    It never accesses private simulation state, hidden maps, or planning algorithms.
    """

    def __init__(
        self,
        world_width: int = 20,
        world_height: int = 12,
        max_energy: int = 10,
        movement_energy_cost: int = 1,
    ) -> None:
        if world_width <= 0 or world_height <= 0:
            raise ValueError("world_width and world_height must be positive.")
        if max_energy <= 0 or movement_energy_cost <= 0:
            raise ValueError("max_energy and movement_energy_cost must be positive.")
        self.world_width = world_width
        self.world_height = world_height
        self.max_energy = max_energy
        self.movement_energy_cost = movement_energy_cost
        self.max_dist = max(1, (world_width - 1) + (world_height - 1))

    def extract(self, observation: Observation, action: Action) -> dict[str, float]:
        """Compute a dictionary of normalized feature values for (observation, action)."""
        features: dict[str, float] = {}
        pos = observation.entity_position
        energy = observation.energy
        target = observation.target
        resources = observation.observed_resources

        # 1. Base bias & action one-hot indicators
        features["bias"] = 0.1
        features[f"action_{action.value}"] = 1.0

        # Candidate next coordinates
        x, y = pos.x, pos.y
        if action == Action.UP:
            nx, ny = x, y - 1
        elif action == Action.DOWN:
            nx, ny = x, y + 1
        elif action == Action.LEFT:
            nx, ny = x - 1, y
        elif action == Action.RIGHT:
            nx, ny = x + 1, y
        else:  # Action.RECHARGE
            nx, ny = x, y

        # 2. Legality and boundary feasibility
        is_move = action != Action.RECHARGE
        in_bounds = 0 <= nx < self.world_width and 0 <= ny < self.world_height
        has_energy = energy >= self.movement_energy_cost
        on_resource = any(r.position == pos for r in resources)

        if is_move:
            is_valid = in_bounds and has_energy
            features["move_into_wall"] = 1.0 if not in_bounds else 0.0
            features["move_no_energy"] = 1.0 if (in_bounds and not has_energy) else 0.0
            features["invalid_action"] = 1.0 if not is_valid else 0.0
            features["valid_move"] = 1.0 if is_valid else 0.0
        else:
            is_valid = on_resource
            features["recharge_no_resource"] = 1.0 if not on_resource else 0.0
            features["invalid_action"] = 1.0 if not is_valid else 0.0
            features["valid_recharge"] = 1.0 if is_valid else 0.0

        # 3. Target navigation (relative direction and distance)
        if target is not None:
            cur_dist = abs(target.x - x) + abs(target.y - y)
            next_dist = abs(target.x - nx) + abs(target.y - ny)

            features["target_dist_norm"] = next_dist / self.max_dist
            if is_valid and is_move:
                if next_dist < cur_dist:
                    features["move_closer_target"] = 1.0
                elif next_dist > cur_dist:
                    features["move_away_target"] = 1.0
                if nx == target.x and ny == target.y:
                    features["reaches_target"] = 1.0

            needed_energy = cur_dist * self.movement_energy_cost
            deficit = max(0, needed_energy - energy) / (self.max_dist * self.movement_energy_cost)
            features["energy_deficit"] = deficit
            features["energy_sufficient"] = 1.0 if energy >= needed_energy else 0.0
        else:
            cur_dist = 0
            deficit = 0.0

        # 4. Energy state features
        features["energy_norm"] = energy / self.max_energy
        features["energy_critical"] = 1.0 if energy <= self.movement_energy_cost else 0.0

        # 5. Resource navigation & interaction features
        if resources:
            nearest_res = min(resources, key=lambda r: abs(r.position.x - x) + abs(r.position.y - y))
            r_cur_dist = abs(nearest_res.position.x - x) + abs(nearest_res.position.y - y)
            r_next_dist = abs(nearest_res.position.x - nx) + abs(nearest_res.position.y - ny)

            if is_valid and is_move:
                if r_next_dist < r_cur_dist:
                    features["move_closer_resource"] = 1.0
                    if deficit > 0:
                        features["seek_res_when_deficient"] = 1.0

            if action == Action.RECHARGE and is_valid:
                res_here = next(r for r in resources if r.position == pos)
                gain = min(res_here.recharge_amount, self.max_energy - energy) / self.max_energy
                features["recharge_gain"] = gain
                if deficit > 0 or energy <= self.movement_energy_cost * 2:
                    features["recharge_when_needed"] = 1.0
                if energy == self.max_energy:
                    features["recharge_wasteful"] = 1.0
                if res_here.consumable:
                    features["resource_consumable"] = 1.0

        return features


class LinearQLearner:
    """Linear function-approximation Q-learning agent with semi-gradient TD(0) updates."""

    def __init__(
        self,
        config: LinearLearningConfig,
        feature_extractor: FeatureExtractor,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.feature_extractor = feature_extractor
        self.weights: dict[str, float] = {}
        self._rng = Random(seed)
        self.epsilon = config.epsilon

    def q_value(self, observation: Observation, action: Action) -> float:
        """Compute Q(observation, action) = w^T phi(observation, action)."""
        feats = self.feature_extractor.extract(observation, action)
        return sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def get_legal_actions(self, observation: Observation) -> tuple[Action, ...]:
        """Determine legal actions from permitted observation and geometry bounds."""
        pos = observation.entity_position
        energy = observation.energy
        legal: list[Action] = []

        # Movement legality
        if energy >= self.feature_extractor.movement_energy_cost:
            if pos.y > 0:
                legal.append(Action.UP)
            if pos.y < self.feature_extractor.world_height - 1:
                legal.append(Action.DOWN)
            if pos.x > 0:
                legal.append(Action.LEFT)
            if pos.x < self.feature_extractor.world_width - 1:
                legal.append(Action.RIGHT)

        # Recharge legality
        if any(r.position == pos for r in observation.observed_resources):
            legal.append(Action.RECHARGE)

        return tuple(legal) if legal else tuple(Action)

    def select_action(self, observation: Observation, explore: bool = True) -> Action:
        """Select action via epsilon-greedy policy over candidate actions."""
        candidates = (
            self.get_legal_actions(observation)
            if self.config.mask_invalid_actions
            else tuple(Action)
        )
        eps = self.epsilon if explore else 0.0

        if explore and self._rng.random() < eps:
            return self._rng.choice(candidates)

        # Greedy choice with deterministic tie-breaking by Action enum order
        best_a = candidates[0]
        best_q = float("-inf")
        for a in candidates:
            q = self.q_value(observation, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        terminal: bool,
    ) -> float:
        """Update weights via semi-gradient TD(0) with gradient clipping."""
        current_q = self.q_value(observation, action)
        future_candidates = (
            self.get_legal_actions(next_observation)
            if self.config.mask_invalid_actions
            else tuple(Action)
        )
        future_q = (
            0.0
            if terminal
            else max(self.q_value(next_observation, a) for a in future_candidates)
        )
        target = reward + self.config.discount_factor * future_q
        td_error = target - current_q

        # Gradient clipping
        clip = self.config.gradient_clip
        clipped_error = max(-clip, min(clip, td_error))

        feats = self.feature_extractor.extract(observation, action)
        alpha = self.config.learning_rate
        for k, v in feats.items():
            if v != 0.0:
                self.weights[k] = self.weights.get(k, 0.0) + alpha * clipped_error * v

        return current_q + alpha * clipped_error

    def decay_epsilon(self) -> None:
        """Monotonically decay exploration rate epsilon toward epsilon_min."""
        if self.config.epsilon_decay < 1.0:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def copy_weights(self) -> dict[str, float]:
        """Return a copy of the current weight dictionary."""
        return dict(self.weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set the internal weight dictionary."""
        self.weights = dict(weights)


class LinearLearningRunner:
    """Executes training and evaluation for linear function-approximation learners.

    Supports single-world tasks as well as multi-task distributions across diverse
    start positions, target positions, energy levels, and resource configurations.
    """

    def __init__(
        self,
        world: WorldConfig,
        config: LinearLearningConfig,
        feature_extractor: FeatureExtractor | None = None,
        seed: int | None = None,
        learner: LinearQLearner | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.feature_extractor = feature_extractor or FeatureExtractor(
            world_width=world.width,
            world_height=world.height,
            max_energy=world.max_energy,
            movement_energy_cost=world.movement_energy_cost,
        )
        self.learner = learner or LinearQLearner(config, self.feature_extractor, seed=seed)

    def _episode(
        self,
        world: WorldConfig,
        learn: bool,
        policy: str = "learned",
        rng: Random | None = None,
    ) -> EpisodeResult:
        simulation = Simulation(world)
        transitions: list[LearningTransition] = []
        reward_total = 0.0
        recharge_count = 0
        rejected_count = 0
        target = world.observable_target

        reached = target is not None and simulation.observe().entity_position == target
        terminal_failure = False

        while not reached and not terminal_failure and len(transitions) < self.config.max_steps:
            observation = simulation.observe()

            if policy == "random":
                rand_rng = rng or self.learner._rng
                action = rand_rng.choice(tuple(Action))
            else:
                action = self.learner.select_action(observation, explore=learn)

            result = simulation.step(action)
            next_observation = result.observation
            reached = target is not None and next_observation.entity_position == target

            if action == Action.RECHARGE and result.success:
                recharge_count += 1
            if not result.success:
                rejected_count += 1

            if reached:
                reward = self.config.target_reward
                terminal = True
            elif not result.success:
                reward = self.config.failed_action_penalty
                if (
                    self.config.terminal_on_energy_exhaustion
                    and next_observation.energy == 0
                    and not any(r.position == next_observation.entity_position for r in next_observation.observed_resources)
                ):
                    terminal_failure = True
                    reward += self.config.terminal_failure_penalty
                    terminal = True
                else:
                    terminal = False
            else:
                reward = (
                    self.config.recharge_cost
                    if action == Action.RECHARGE
                    else self.config.step_cost
                )
                terminal = False

            # Dummy state keys for LearningTransition compatibility
            s_key = (observation.entity_position.x, observation.entity_position.y, observation.energy)
            ns_key = (next_observation.entity_position.x, next_observation.entity_position.y, next_observation.energy)
            transitions.append(LearningTransition(s_key, action, reward, ns_key, terminal))
            reward_total += reward

            if learn:
                self.learner.update(observation, action, reward, next_observation, terminal)

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
        """Train the agent for a number of episodes on the specified (or default) world."""
        if episodes <= 0:
            raise ValueError("episodes must be positive.")
        w = world or self.world
        outcomes = []
        for _ in range(episodes):
            outcomes.append(self._episode(w, learn=True))
            self.learner.decay_epsilon()
        return tuple(outcomes)

    def train_tasks(
        self,
        tasks: Sequence[WorldConfig],
        episodes_per_task: int,
    ) -> tuple[EpisodeResult, ...]:
        """Train across a sequence or distribution of task environments using round-robin interleaving."""
        if episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive.")
        if not tasks:
            raise ValueError("tasks must not be empty.")
        outcomes = []
        total_episodes = len(tasks) * episodes_per_task
        for ep in range(total_episodes):
            task = tasks[ep % len(tasks)]
            outcomes.append(self._episode(task, learn=True))
            self.learner.decay_epsilon()
        return tuple(outcomes)

    def evaluate(
        self,
        episodes: int,
        world: WorldConfig | None = None,
        seed: int | None = None,
        policy: str = "learned",
    ) -> EvaluationResult:
        """Evaluate agent performance without weight updates or exploration."""
        if episodes <= 0 or policy not in {"learned", "random"}:
            raise ValueError("episodes must be positive and policy must be 'learned' or 'random'.")
        w = world or self.world
        rng = Random(seed)
        outcomes = tuple(self._episode(w, learn=False, policy=policy, rng=rng) for _ in range(episodes))
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
