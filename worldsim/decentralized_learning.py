"""Decentralized Multi-Agent Reinforcement Learning with Local Intent Communication.

Implements Milestone 14.4:
- Interpretable, transparent decentralized Q-learning with linear function approximation.
- Strict local information boundaries: features derived solely from local observations
  within sensory and communication horizon R <= 2.0.
- Two distinct modes for empirical comparison:
  1. Non-communicative policy: observes only visible peer positions within R <= 2.0.
  2. Communicative policy: exchanges compact intent messages with peers within R <= 2.0.
- State representation covering relative goal displacement, energy margins, resource
  proximity, moving obstacle prediction, bottleneck detection, and peer intentions.
- Preserves authoritative simulation physics (zero illegal double occupancy).
- Training and evaluation separation: evaluation uses held-out seeds (500-519).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from random import Random
from typing import Any, Sequence

from .actions import Action
from .coordination import MultiAgentCoordinator, agent_val
from .dynamic_obstacles import DynamicObstacleState
from .models import Position, Resource
from .multi_agent import (
    AgentConfig,
    AgentState,
    MultiAgentObservation,
    MultiAgentSimulation,
    MultiAgentWorldConfig,
)
from .robust_policy import AgentMemory


@dataclass(frozen=True, slots=True)
class DecentralizedLearningConfig:
    """Hyperparameters and configuration for decentralized linear Q-learning."""

    learning_rate: float = 0.05
    discount_factor: float = 0.95
    epsilon_start: float = 0.30
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    gradient_clip: float = 2.0
    weight_decay: float = 1e-4

    # Reward weights
    target_reward: float = 100.0
    step_cost: float = -0.5
    move_energy_cost: float = -0.5
    progress_reward_scale: float = 5.0
    exhaustion_penalty: float = -100.0
    swap_collision_penalty: float = -15.0
    vertex_collision_penalty: float = -15.0
    dynamic_collision_penalty: float = -20.0
    blocked_occupied_penalty: float = -3.0
    beneficial_recharge_reward: float = 35.0
    safe_wait_reward: float = -0.2
    wasteful_wait_penalty: float = -5.0
    repeated_wait_penalty: float = -4.0
    fairness_bonus_scale: float = 2.0

    max_steps_per_episode: int = 35


class DecentralizedFeatureExtractor:
    """Extracts normalized, relative state-action features within local horizon R <= 2.0.

    Guarantees:
    - Never accesses private simulation state or distant agents beyond sensory horizon R <= 2.0.
    - When enable_communication=False, intent features are strictly disabled (0.0).
    - When enable_communication=True, peer intent features are populated strictly from
      peers within Euclidean distance <= 2.0.
    """

    def __init__(
        self,
        world_width: int,
        world_height: int,
        max_energy: int = 10,
        sensory_radius: float = 2.0,
        enable_communication: bool = False,
    ) -> None:
        self.world_width = max(1, world_width)
        self.world_height = max(1, world_height)
        self.max_energy = max(1, max_energy)
        self.sensory_radius = sensory_radius
        self.sensory_radius_sq = sensory_radius * sensory_radius
        self.enable_communication = enable_communication
        self.max_dist = max(1, (self.world_width - 1) + (self.world_height - 1))

    def extract(
        self,
        observation: MultiAgentObservation,
        action: Action,
        peer_intentions: dict[str, Position] | None = None,
        predicted_dyn_obstacles: set[Position] | None = None,
        inactive_obstacles: set[Position] | None = None,
        consecutive_waits: int = 0,
        tick: int = 0,
        known_resources: Sequence[Resource] | None = None,
    ) -> dict[str, float]:
        """Compute normalized feature vector for the given observation and candidate action."""
        features: dict[str, float] = {}

        pos = observation.entity_position
        energy = observation.energy
        target = observation.target
        resources = observation.observed_resources if observation.observed_resources else (tuple(known_resources) if known_resources else ())
        observed_peers = observation.observed_agents
        dyn_obstacles = getattr(observation, "observed_dynamic_obstacles", ())
        aid = getattr(observation, "agent_id", "agent")

        dyn_pred = predicted_dyn_obstacles if predicted_dyn_obstacles is not None else set()
        inactive_obs = inactive_obstacles if inactive_obstacles is not None else set()

        # Candidate next coordinates
        nx, ny = pos.x, pos.y
        if action == Action.UP:
            ny -= 1
        elif action == Action.DOWN:
            ny += 1
        elif action == Action.LEFT:
            nx -= 1
        elif action == Action.RIGHT:
            nx += 1
        dest_pos = Position(nx, ny)

        is_move = action != Action.RECHARGE
        in_bounds = 0 <= nx < self.world_width and 0 <= ny < self.world_height
        has_energy = energy >= 1
        on_active_resource = any(r.position == pos for r in resources)

        # 1. Base Bias & Action Indicator
        features["bias"] = 0.1
        if action == Action.RECHARGE:
            if on_active_resource:
                features["action_recharge_on_pad"] = 1.0
            else:
                features["action_wait_in_place"] = 1.0
        else:
            features[f"action_{action.value}"] = 1.0

        # 2. Kinematic & Boundary Feasibility
        if is_move:
            features["move_out_of_bounds"] = 1.0 if not in_bounds else 0.0
            features["move_no_energy"] = 1.0 if (in_bounds and not has_energy) else 0.0
            features["valid_move"] = 1.0 if (in_bounds and has_energy) else 0.0
        else:
            features["recharge_on_pad"] = 1.0 if on_active_resource else 0.0

        # 3. Goal Progress & Distance
        if target is not None:
            cur_dist = abs(target.x - pos.x) + abs(target.y - pos.y)
            new_dist = abs(target.x - nx) + abs(target.y - ny) if in_bounds else cur_dist + 2
            features["goal_dist_norm"] = new_dist / self.max_dist
            features["progress_to_goal"] = 1.0 if new_dist < cur_dist else (-1.0 if new_dist > cur_dist else 0.0)
            features["reaches_goal"] = 1.0 if dest_pos == target else 0.0
            features["goal_dx_norm"] = (target.x - nx) / self.max_dist
            features["goal_dy_norm"] = (target.y - ny) / self.max_dist
            features["energy_margin_to_goal"] = (energy - cur_dist) / self.max_energy

        # 4. Energy & Resource Planning
        features["energy_norm"] = energy / self.max_energy
        features["critically_low_energy"] = 1.0 if energy <= 2 else 0.0
        cur_dist = abs(target.x - pos.x) + abs(target.y - pos.y) if target is not None else 0
        energy_margin = energy - cur_dist if target is not None else 0
        needs_recharge = target is not None and energy_margin <= 2

        if resources:
            nearest_res = min(resources, key=lambda r: abs(r.position.x - pos.x) + abs(r.position.y - pos.y))
            r_cur = abs(nearest_res.position.x - pos.x) + abs(nearest_res.position.y - pos.y)
            r_new = abs(nearest_res.position.x - nx) + abs(nearest_res.position.y - ny) if in_bounds else r_cur
            features["dist_to_res_norm"] = r_new / self.max_dist
            features["progress_to_res"] = 1.0 if r_new < r_cur else (-1.0 if r_new > r_cur else 0.0)
            features["reaches_resource"] = 1.0 if (is_move and in_bounds and dest_pos == nearest_res.position) else 0.0

            if needs_recharge:
                features["low_energy_progress_to_res"] = 1.0 if r_new < r_cur else (-1.0 if r_new > r_cur else 0.0)
                features["low_energy_reaches_res"] = 1.0 if (is_move and in_bounds and dest_pos == nearest_res.position) else 0.0
                if on_active_resource and action == Action.RECHARGE:
                    features["urgent_recharge"] = 1.0

        # 5. Static & Inactive Obstacle Avoidance
        if dest_pos in inactive_obs:
            features["into_inactive_obstacle"] = 1.0

        # 6. Dynamic Obstacle Avoidance (Locally Observed within R <= 2.0)
        dyn_now = {o.position for o in dyn_obstacles}
        if dest_pos in dyn_now:
            features["into_dyn_obstacle_now"] = 1.0
        if dest_pos in dyn_pred:
            features["into_dyn_obstacle_pred"] = 1.0

        # Check if dynamic obstacle is crossing adjacent cells
        dyn_adjacent = any(
            abs(o.position.x - pos.x) + abs(o.position.y - pos.y) <= 1 for o in dyn_obstacles
        )
        if dyn_adjacent:
            features["dyn_obstacle_adjacent"] = 1.0

        # 7. Local Peer Observation (No Communication, R <= 2.0)
        active_peers = [p for p in observed_peers if p.active]
        features["visible_peer_count"] = len(active_peers) / 4.0
        peer_positions = {p.position for p in active_peers}

        if is_move and dest_pos in peer_positions:
            features["into_peer_current_cell"] = 1.0

        if active_peers:
            min_peer_dist = min(
                abs(p.position.x - pos.x) + abs(p.position.y - pos.y) for p in active_peers
            )
            features["min_peer_dist_norm"] = min_peer_dist / self.max_dist

            # Evasive lateral detour detection:
            # If an oncoming peer is directly ahead on same row or column, check if candidate move is lateral
            oncoming_ahead = any(
                (p.position.y == pos.y and abs(p.position.x - pos.x) <= 2)
                or (p.position.x == pos.x and abs(p.position.y - pos.y) <= 2)
                for p in active_peers
            )
            if oncoming_ahead and is_move and in_bounds and dest_pos not in peer_positions:
                is_lateral = (
                    any(p.position.y == pos.y and ny != pos.y for p in active_peers)
                    or any(p.position.x == pos.x and nx != pos.x for p in active_peers)
                )
                if is_lateral:
                    features["evasive_lateral_detour"] = 1.0

        # 8. Local Intent Communication (R <= 2.0, only when enabled)
        if self.enable_communication and peer_intentions:
            peer_intents = set(peer_intentions.values())
            if is_move and dest_pos in peer_intents:
                features["comm_peer_intent_vertex_conflict"] = 1.0

            # Check for swap conflict: peer currently at dest_pos intends to enter pos
            my_val = agent_val(aid)
            my_prio = ((my_val - 1) + tick) % 2
            for p_id, p_dest in peer_intentions.items():
                p_cur = next((p.position for p in active_peers if p.id == p_id), None)
                if p_cur is not None and is_move and dest_pos == p_cur and p_dest == pos:
                    features["comm_peer_swap_conflict"] = 1.0

                p_val = agent_val(p_id)
                p_prio = ((p_val - 1) + tick) % 2
                if p_dest == pos or p_dest == dest_pos:
                    if p_prio < my_prio:
                        features["comm_peer_has_priority"] = 1.0
                        if is_move and dest_pos not in peer_positions:
                            features["comm_priority_yield_detour"] = 1.0
                        elif action == Action.RECHARGE:
                            features["comm_yield_wait"] = 1.0
                    else:
                        features["comm_my_priority_advance"] = 1.0

            peer_intends_my_cell = any(p_dest == pos for p_dest in peer_intentions.values())
            if peer_intends_my_cell:
                features["comm_peer_intends_my_cell"] = 1.0

        # 9. Intentional Waiting vs. Wasteful Waiting & Consecutive Wait Penalty
        if action == Action.RECHARGE:
            if on_active_resource:
                if energy < self.max_energy:
                    features["safe_recharge"] = 1.0
            else:
                features["consecutive_waits"] = min(1.0, consecutive_waits / 3.0)
                has_conflict = (
                    pos in dyn_pred
                    or dyn_adjacent
                    or any(abs(p.position.x - pos.x) + abs(p.position.y - pos.y) <= 1 for p in active_peers)
                )
                if has_conflict:
                    if consecutive_waits == 0:
                        features["safe_intentional_wait"] = 1.0
                    else:
                        features["repeated_wait"] = 1.0
                else:
                    features["wasteful_wait"] = 1.0

        return features


class DecentralizedQLearner:
    """Linear function-approximation Q-learner for decentralized coordination policies."""

    def __init__(
        self,
        config: DecentralizedLearningConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config if config is not None else DecentralizedLearningConfig()
        self.rng = Random(seed)
        self.weights: dict[str, float] = {}
        self.epsilon = self.config.epsilon_start
        self.total_updates = 0

    def predict_q(self, features: dict[str, float]) -> float:
        """Compute Q(s, a) = w^T phi(s, a)."""
        q = 0.0
        for name, val in features.items():
            q += self.weights.get(name, 0.0) * val
        return q

    def select_action(
        self,
        observation: MultiAgentObservation,
        feature_extractor: DecentralizedFeatureExtractor,
        legal_actions: Sequence[Action],
        peer_intentions: dict[str, Position] | None = None,
        predicted_dyn_obstacles: set[Position] | None = None,
        inactive_obstacles: set[Position] | None = None,
        consecutive_waits: int = 0,
        tick: int = 0,
        known_resources: Sequence[Resource] | None = None,
        greedy: bool = False,
    ) -> tuple[Action, float, dict[str, float]]:
        """Select action using epsilon-greedy (training) or greedy (evaluation) policy."""
        if not legal_actions:
            legal_actions = (Action.UP,)

        # Compute features and Q-values for all legal candidate actions
        action_q: dict[Action, tuple[float, dict[str, float]]] = {}
        for a in legal_actions:
            feats = feature_extractor.extract(
                observation=observation,
                action=a,
                peer_intentions=peer_intentions,
                predicted_dyn_obstacles=predicted_dyn_obstacles,
                inactive_obstacles=inactive_obstacles,
                consecutive_waits=consecutive_waits,
                tick=tick,
                known_resources=known_resources,
            )
            q_val = self.predict_q(feats)
            action_q[a] = (q_val, feats)

        # Exploration vs. Exploitation
        if not greedy and self.rng.random() < self.epsilon:
            chosen = self.rng.choice(list(legal_actions))
            chosen_q, chosen_feats = action_q[chosen]
            return chosen, chosen_q, chosen_feats

        # Greedy selection with deterministic tie-breaking
        best_a = legal_actions[0]
        best_q = float("-inf")
        best_feats: dict[str, float] = {}

        for a in legal_actions:
            q_val, feats = action_q[a]
            if q_val > best_q:
                best_q = q_val
                best_a = a
                best_feats = feats

        return best_a, best_q, best_feats

    def update(
        self,
        features: dict[str, float],
        reward: float,
        next_max_q: float,
        current_q: float,
        done: bool,
    ) -> float:
        """Update linear weights via TD(0) gradient step: w <- w + alpha * delta * phi."""
        target = reward if done else reward + self.config.discount_factor * next_max_q
        td_error = target - current_q

        # Clip TD error for gradient stability
        clipped_td = max(-self.config.gradient_clip, min(self.config.gradient_clip, td_error))

        # Weight updates with L2 regularization
        lr = self.config.learning_rate
        decay = self.config.weight_decay
        for name, val in features.items():
            if val != 0.0:
                w = self.weights.get(name, 0.0)
                # Weight update with decay
                self.weights[name] = w * (1.0 - lr * decay) + (lr * clipped_td * val)

        self.total_updates += 1
        return td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each training episode."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save_weights(self, path: Path | str) -> None:
        """Save learned weights to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"weights": self.weights, "total_updates": self.total_updates}, f, indent=2)

    def load_weights(self, path: Path | str) -> None:
        """Load learned weights from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.weights = {k: float(v) for k, v in data["weights"].items()}
            self.total_updates = data.get("total_updates", 0)


class LearnedCoordinationPolicy:
    """Decentralized learned policy adhering to WorldSim policy interface.

    Supports:
    - enable_communication=False (Non-communicative learned policy).
    - enable_communication=True (Communicative learned policy with R <= 2.0 intent sharing).
    - Transparent feature evaluation and action selection.
    """

    def __init__(
        self,
        world_width: int,
        world_height: int,
        max_energy: int = 10,
        sensory_radius: float = 2.0,
        learner: DecentralizedQLearner | None = None,
        enable_communication: bool = False,
        coordinator: MultiAgentCoordinator | None = None,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self.max_energy = max_energy
        self.sensory_radius = sensory_radius
        self.enable_communication = enable_communication
        self.coordinator = coordinator

        self.feature_extractor = DecentralizedFeatureExtractor(
            world_width=world_width,
            world_height=world_height,
            max_energy=max_energy,
            sensory_radius=sensory_radius,
            enable_communication=enable_communication,
        )
        self.learner = learner if learner is not None else DecentralizedQLearner()
        self.memory = AgentMemory(world_width=world_width, world_height=world_height)

        # Step memory for TD updates
        self.last_observation: MultiAgentObservation | None = None
        self.last_action: Action | None = None
        self.last_features: dict[str, float] | None = None
        self.last_q: float = 0.0
        self.step_count = 0
        self.consecutive_waits = 0

    def get_legal_actions(self, observation: MultiAgentObservation) -> tuple[Action, ...]:
        """Compute legal action subset from geometry and energy."""
        pos = observation.entity_position
        energy = observation.energy
        legal: list[Action] = []

        if energy >= 1:
            if pos.y > 0:
                legal.append(Action.UP)
            if pos.y < self.world_height - 1:
                legal.append(Action.DOWN)
            if pos.x > 0:
                legal.append(Action.LEFT)
            if pos.x < self.world_width - 1:
                legal.append(Action.RIGHT)

        # Recharge allowed if on resource or if dynamic obstacles/peers observed
        on_res = any(r.position == pos for r in observation.observed_resources)
        has_dyn = bool(getattr(observation, "observed_dynamic_obstacles", ()))
        has_peers = bool(getattr(observation, "observed_agents", ()))
        if on_res or has_dyn or has_peers:
            legal.append(Action.RECHARGE)

        return tuple(legal) if legal else (Action.RECHARGE,)

    def choose_action(self, observation: MultiAgentObservation, greedy: bool = True) -> Action:
        """Select action using learned decentralized policy."""
        pos = observation.entity_position
        target = observation.target
        aid = getattr(observation, "agent_id", "agent")

        # Terminal target check
        if target is not None and pos == target:
            return Action.UP

        legal_actions = self.get_legal_actions(observation)
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Gather local peer intentions if communication is enabled
        peer_intentions: dict[str, Position] = {}
        if self.enable_communication and self.coordinator is not None:
            observed_other = {p.id: p.position for p in observation.observed_agents}
            visible_peers = self.coordinator.get_visible_peers(aid, pos, observed_other)
            peer_intentions = {
                oid: od
                for oid, od in self.coordinator.agent_intentions.items()
                if oid in visible_peers
            }

        # Update local memory strictly from agent observation
        self.memory.update(observation)

        # Inactive obstacles (exhausted peers) from observation and memory
        inactive_obs = set(self.memory.get_remembered_inactive_obstacles().keys())
        inactive_obs.update(p.position for p in observation.observed_agents if not p.active)

        # Dynamic obstacles from memory prediction and local observation
        predicted_dyn = self.memory.get_predicted_dynamic_obstacle_cells(horizon=2)
        for o in getattr(observation, "observed_dynamic_obstacles", ()):
            predicted_dyn.add(o.position)
            pred = Position(o.position.x + o.velocity[0], o.position.y + o.velocity[1])
            if 0 <= pred.x < self.world_width and 0 <= pred.y < self.world_height:
                predicted_dyn.add(pred)

        known_res = self.memory.available_resources()

        # Select action
        action, q_val, feats = self.learner.select_action(
            observation=observation,
            feature_extractor=self.feature_extractor,
            legal_actions=legal_actions,
            peer_intentions=peer_intentions,
            predicted_dyn_obstacles=predicted_dyn,
            inactive_obstacles=inactive_obs,
            consecutive_waits=self.consecutive_waits,
            tick=self.step_count,
            known_resources=known_res,
            greedy=greedy,
        )

        on_res = any(r.position == pos for r in observation.observed_resources) or any(r.position == pos for r in known_res)
        if action == Action.RECHARGE and not on_res:
            self.consecutive_waits += 1
        else:
            self.consecutive_waits = 0

        # Store for transition update
        self.last_observation = observation
        self.last_action = action
        self.last_features = feats
        self.last_q = q_val
        self.step_count += 1

        # Register intent with coordinator for peer communication if enabled
        if self.coordinator is not None:
            nx, ny = pos.x, pos.y
            if action == Action.UP:
                ny -= 1
            elif action == Action.DOWN:
                ny += 1
            elif action == Action.LEFT:
                nx -= 1
            elif action == Action.RIGHT:
                nx += 1
            dest = Position(nx, ny) if action != Action.RECHARGE else pos
            self.coordinator.register_intent(aid, dest, target, current_position=pos)

        return action

    def record_action_result(
        self,
        pos: Position,
        energy: int,
        action: Action,
        success: bool,
    ) -> None:
        """Record action outcome into agent memory."""
        self.memory.record_action_result(pos, energy, action, success)


class DecentralizedRewardEngine:
    """Calculates standardized, multi-objective learning rewards per agent transition."""

    def __init__(self, config: DecentralizedLearningConfig | None = None) -> None:
        self.config = config if config is not None else DecentralizedLearningConfig()

    def compute_reward(
        self,
        agent_id: str,
        obs_before: MultiAgentObservation,
        action: Action,
        pos_after: Position,
        energy_after: int,
        success: bool,
        collision_type: str | None,
        terminated: bool,
        all_agents_before: dict[str, AgentState],
        all_agents_after: dict[str, AgentState],
        dynamic_collisions: int = 0,
        consecutive_waits: int = 0,
    ) -> float:
        """Compute transparent scalar reward balancing individual progress, safety, and fairness."""
        reward = 0.0
        pos_before = obs_before.entity_position
        target = obs_before.target

        # 1. Step & Action Cost
        reward += self.config.step_cost
        if action != Action.RECHARGE:
            reward += self.config.move_energy_cost

        # 2. Target Progress & Completion
        if target is not None:
            dist_before = abs(target.x - pos_before.x) + abs(target.y - pos_before.y)
            dist_after = abs(target.x - pos_after.x) + abs(target.y - pos_after.y)
            delta = dist_before - dist_after
            reward += delta * self.config.progress_reward_scale

            if pos_after == target:
                reward += self.config.target_reward

        # 3. Energy Exhaustion Penalty
        if energy_after <= 0 and (target is None or pos_after != target):
            reward += self.config.exhaustion_penalty

        # 4. Collision Penalties
        if collision_type == "swap":
            reward += self.config.swap_collision_penalty
        elif collision_type == "vertex":
            reward += self.config.vertex_collision_penalty
        elif collision_type == "occupied":
            reward += self.config.blocked_occupied_penalty

        if dynamic_collisions > 0:
            reward += self.config.dynamic_collision_penalty

        # 5. Waiting & Recharge Rewards
        if action == Action.RECHARGE:
            on_res = any(r.position == pos_before for r in obs_before.observed_resources)
            if on_res and energy_after > obs_before.energy:
                reward += self.config.beneficial_recharge_reward
            else:
                # Intentional wait in place
                has_nearby_conflict = (
                    bool(getattr(obs_before, "observed_dynamic_obstacles", ()))
                    or bool(getattr(obs_before, "observed_agents", ()))
                )
                if has_nearby_conflict:
                    reward += self.config.safe_wait_reward
                    if consecutive_waits > 1:
                        reward += self.config.repeated_wait_penalty * (consecutive_waits - 1)
                else:
                    reward += self.config.wasteful_wait_penalty

        # 6. Team Progress / Fairness Component
        if all_agents_before and all_agents_after:
            active_peers = [
                aid for aid, ag in all_agents_after.items() if aid != agent_id and ag.active
            ]
            if active_peers:
                peer_progress = 0
                for paid in active_peers:
                    p_before = all_agents_before[paid]
                    p_after = all_agents_after[paid]
                    if p_before.target is not None:
                        d_b = abs(p_before.target.x - p_before.position.x) + abs(p_before.target.y - p_before.position.y)
                        d_a = abs(p_after.target.x - p_after.position.x) + abs(p_after.target.y - p_after.position.y)
                        if d_a < d_b:
                            peer_progress += 1
                reward += (peer_progress / len(active_peers)) * self.config.fairness_bonus_scale

        return reward


def build_training_scenarios() -> list[dict[str, Any]]:
    """Build a diverse suite of multi-agent training environments.

    Scenarios include:
    - Head-on crossing corridors (learning lateral detours and yielding).
    - Bottleneck chokepoints with moving obstacles (learning safe waiting).
    - Resource approach under energy constraints (learning recharge timing).
    """
    from .dynamic_obstacles import DynamicObstacleConfig

    scenarios: list[dict[str, Any]] = []

    # Train 1: 2-Agent Head-on Crossing (6x4 grid)
    scenarios.append({
        "name": "Train_HeadOn_Crossing",
        "width": 6,
        "height": 4,
        "agents": (
            AgentConfig(id="ag_w", initial_x=0, initial_y=1, initial_energy=10, target=Position(5, 1)),
            AgentConfig(id="ag_e", initial_x=5, initial_y=1, initial_energy=10, target=Position(0, 1)),
        ),
        "resources": (),
        "dynamic_obstacles": (),
    })

    # Train 2: Dynamic Obstacle Doorway Crossing (6x5 grid)
    scenarios.append({
        "name": "Train_DynObs_Crossing",
        "width": 6,
        "height": 5,
        "agents": (
            AgentConfig(id="ag_w", initial_x=0, initial_y=2, initial_energy=10, target=Position(5, 2)),
            AgentConfig(id="ag_e", initial_x=5, initial_y=2, initial_energy=10, target=Position(0, 2)),
        ),
        "resources": (),
        "dynamic_obstacles": (
            DynamicObstacleConfig(
                id="obs_gate",
                initial_x=3,
                initial_y=0,
                pattern="patrol",
                waypoints=(Position(3, 0), Position(3, 4)),
            ),
        ),
    })

    # Train 3: Resource Pad Approach with Tight Energy (7x5 grid)
    scenarios.append({
        "name": "Train_Resource_Approach",
        "width": 7,
        "height": 5,
        "agents": (
            AgentConfig(id="ag_left", initial_x=0, initial_y=2, initial_energy=5, target=Position(6, 2)),
            AgentConfig(id="ag_right", initial_x=6, initial_y=2, initial_energy=5, target=Position(0, 2)),
        ),
        "resources": (Resource(Position(3, 2), recharge_amount=6, consumable=False),),
        "dynamic_obstacles": (),
    })

    # Train 4: 4-Way Junction Intersection (7x7 grid)
    scenarios.append({
        "name": "Train_4Way_Intersection",
        "width": 7,
        "height": 7,
        "agents": (
            AgentConfig(id="ag_n", initial_x=3, initial_y=0, initial_energy=12, target=Position(3, 6)),
            AgentConfig(id="ag_s", initial_x=3, initial_y=6, initial_energy=12, target=Position(3, 0)),
            AgentConfig(id="ag_w", initial_x=0, initial_y=3, initial_energy=12, target=Position(6, 3)),
            AgentConfig(id="ag_e", initial_x=6, initial_y=3, initial_energy=12, target=Position(0, 3)),
        ),
        "resources": (),
        "dynamic_obstacles": (),
    })

    # Train 5: Off-Axis Recharge Detour under Energy Scarcity (7x6 grid)
    scenarios.append({
        "name": "Train_OffAxis_Recharge",
        "width": 7,
        "height": 6,
        "agents": (
            AgentConfig(id="ag_nw", initial_x=1, initial_y=1, initial_energy=6, target=Position(5, 4)),
            AgentConfig(id="ag_se", initial_x=5, initial_y=4, initial_energy=6, target=Position(1, 1)),
        ),
        "resources": (
            Resource(Position(2, 2), recharge_amount=6, consumable=False),
            Resource(Position(4, 3), recharge_amount=6, consumable=False),
        ),
        "dynamic_obstacles": (),
    })

    return scenarios


def train_decentralized_coordination(
    enable_communication: bool = False,
    training_seeds: Sequence[int] = tuple(range(42, 92)),  # 50 training seeds
    epochs: int = 5,
    max_steps_per_episode: int = 30,
    config: DecentralizedLearningConfig | None = None,
) -> DecentralizedQLearner:
    """Train a decentralized Q-learner across training scenarios using explicit training seeds.

    Guarantees:
    - Training seeds are strictly separate from evaluation seeds (500-519).
    - Returns trained DecentralizedQLearner with learned weight vector.
    """
    cfg = config if config is not None else DecentralizedLearningConfig(max_steps_per_episode=max_steps_per_episode)
    learner = DecentralizedQLearner(config=cfg, seed=training_seeds[0] if training_seeds else 42)
    reward_engine = DecentralizedRewardEngine(config=cfg)
    scenarios = build_training_scenarios()

    for epoch in range(epochs):
        for seed in training_seeds:
            for scen in scenarios:
                world_cfg = MultiAgentWorldConfig(
                    width=scen["width"],
                    height=scen["height"],
                    agents=tuple(scen["agents"]),
                    resources=tuple(scen.get("resources", ())),
                    dynamic_obstacles=tuple(scen.get("dynamic_obstacles", ())),
                    sensory_radius=2.0,
                    communication_radius=2.0,
                    enable_coordination=True,
                    coordination_mode="decentralized",
                    collision_rule="priority_rotation",
                )
                sim = MultiAgentSimulation(world_cfg, seed=seed)
                agent_ids = [a.id for a in scen["agents"]]

                policies = {
                    aid: LearnedCoordinationPolicy(
                        world_width=scen["width"],
                        world_height=scen["height"],
                        max_energy=10,
                        sensory_radius=2.0,
                        learner=learner,
                        enable_communication=enable_communication,
                        coordinator=sim.coordinator,
                    )
                    for aid in agent_ids
                }

                step = 0
                while step < max_steps_per_episode:
                    all_done = all(not sim.state.agents[aid].active for aid in agent_ids)
                    if all_done:
                        break

                    if sim.coordinator is not None:
                        sim.coordinator.reset_tick()

                    # Agents select actions (with exploration)
                    actions: dict[str, Action] = {}
                    obs_dict: dict[str, MultiAgentObservation] = {}
                    agents_before = {aid: sim.state.agents[aid] for aid in agent_ids}

                    for aid in agent_ids:
                        if sim.state.agents[aid].active:
                            obs = sim.observe_agent(aid)
                            obs_dict[aid] = obs
                            actions[aid] = policies[aid].choose_action(obs, greedy=False)
                        else:
                            actions[aid] = Action.UP

                    step_res = sim.step(actions)
                    step += 1
                    agents_after = {aid: sim.state.agents[aid] for aid in agent_ids}

                    # Perform TD updates for active agents
                    for aid in agent_ids:
                        if aid in obs_dict:
                            ag_res = step_res.agent_results[aid]
                            pol = policies[aid]
                            done = not sim.state.agents[aid].active
                            reward = reward_engine.compute_reward(
                                agent_id=aid,
                                obs_before=obs_dict[aid],
                                action=ag_res.action,
                                pos_after=ag_res.position_after,
                                energy_after=ag_res.energy_after,
                                success=ag_res.success,
                                collision_type=ag_res.collision_type,
                                terminated=done,
                                all_agents_before=agents_before,
                                all_agents_after=agents_after,
                                dynamic_collisions=step_res.dynamic_obstacle_collisions,
                                consecutive_waits=pol.consecutive_waits,
                            )
                            next_max_q = 0.0
                            if not done:
                                next_obs = sim.observe_agent(aid)
                                next_legal = pol.get_legal_actions(next_obs)
                                for na in next_legal:
                                    n_feats = pol.feature_extractor.extract(
                                        next_obs,
                                        na,
                                        consecutive_waits=pol.consecutive_waits,
                                        tick=step,
                                    )
                                    nq = learner.predict_q(n_feats)
                                    if nq > next_max_q:
                                        next_max_q = nq

                            if pol.last_features:
                                learner.update(
                                    features=pol.last_features,
                                    reward=reward,
                                    next_max_q=next_max_q,
                                    current_q=pol.last_q,
                                    done=done,
                                )

                    if step_res.overall_completed:
                        break

                learner.decay_epsilon()

    return learner

