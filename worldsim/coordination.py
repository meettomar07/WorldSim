"""Multi-agent coordination, congestion management, and fair resource allocation.

Phase 13.1 introduces:
1. Deterministic congestion resolution with priority rotation and blocked-cell avoidance.
2. Fair, energy-aware shared resource allocation preventing indefinite starvation.
3. Voluntary goal-aware and intent-aware multi-agent coordination.
4. Bounded deadlock detection and deterministic recovery.
5. Per-agent fairness and starvation monitoring with Jain's fairness index.

Strictly preserves observation boundaries, deterministic reproducibility, and single-agent
backward compatibility.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

from .actions import Action
from .models import Position, Resource


# ==============================================================================
# Metric Utilities: Jain's Fairness Index
# ==============================================================================


def jains_fairness_index(values: Sequence[float]) -> float:
    """Calculate Jain's fairness index for a sequence of non-negative values.

    J(x_1, ..., x_n) = (sum x_i)^2 / (n * sum (x_i^2))
    Returns 1.0 if all values are equal (or all zero).
    """
    if not values:
        return 1.0
    n = len(values)
    sum_val = float(sum(values))
    sum_sq = float(sum(v * v for v in values))
    if sum_sq == 0.0:
        return 1.0
    score = (sum_val * sum_val) / (n * sum_sq)
    return min(1.0, max(0.0, score))


# ==============================================================================
# Helper: Deterministic Agent Priority Hashing
# ==============================================================================

def agent_val(aid: str) -> int:
    """Deterministic integer value for an agent ID for priority ordering.

    Handles numbered IDs (e.g. 'agent_1'), cardinal directions (e.g. 'agent_n', 's'),
    and arbitrary strings via a deterministic polynomial hash (avoiding Python's
    randomized process-level hash seed).
    """
    digits = "".join(ch for ch in aid if ch.isdigit())
    if digits:
        return int(digits)
    cardinals = {"n": 0, "s": 1, "w": 2, "e": 3, "nw": 0, "ne": 1, "sw": 2, "se": 3}
    for k, v in cardinals.items():
        if aid == k or aid.endswith(f"_{k}"):
            return v
    h = 0
    for c in aid:
        h = (h * 31 + ord(c)) & 0xFFFFFF
    return h


# ==============================================================================
# Stage 2: Fair, Energy-Aware Resource Allocation
# ==============================================================================


@dataclass
class ResourceClaimRecord:
    """Historical record of an agent's resource claims and outcomes."""

    agent_id: str
    total_claims: int = 0
    successful_claims: int = 0
    contested_losses: int = 0
    total_energy_gained: int = 0
    consecutive_losses: int = 0


class ResourceArbitrator:
    """Arbitrates simultaneous resource recharge claims fairly and deterministically.

    Prevents multiple agents from receiving energy from the same consumable resource
    while eliminating permanent lowest-ID bias.
    """

    def __init__(self, rule: str = "fair_balanced") -> None:
        """Initialize arbitrator with configured allocation rule.

        Rules:
            'lowest_id': Phase 13 legacy rule (alphabetical ID).
            'priority_rotation': Deterministic round-robin rotation based on simulation tick.
            'need_urgency': Prioritizes agent with lowest energy / greatest deficit.
            'starvation_prevention': Prioritizes agent with highest recent contested losses.
            'fair_balanced': Combines urgency and starvation prevention.
        """
        self.rule = rule
        self.records: dict[str, ResourceClaimRecord] = {}

    def get_or_create_record(self, agent_id: str) -> ResourceClaimRecord:
        if agent_id not in self.records:
            self.records[agent_id] = ResourceClaimRecord(agent_id=agent_id)
        return self.records[agent_id]

    def arbitrate(
        self,
        claimants: list[str],
        agent_energies: dict[str, int],
        agent_max_energies: dict[str, int],
        resource: Resource,
        tick: int,
    ) -> tuple[str, list[str]]:
        """Select exactly one winner among claimants and return (winner, losers).

        Deterministic and reproducible.
        """
        if not claimants:
            raise ValueError("Cannot arbitrate empty claimants")
        if len(claimants) == 1:
            winner = claimants[0]
            rec = self.get_or_create_record(winner)
            rec.total_claims += 1
            rec.successful_claims += 1
            rec.consecutive_losses = 0
            gain = min(resource.recharge_amount, agent_max_energies.get(winner, 10) - agent_energies.get(winner, 0))
            rec.total_energy_gained += max(0, gain)
            return winner, []

        for aid in claimants:
            rec = self.get_or_create_record(aid)
            rec.total_claims += 1

        if self.rule == "lowest_id":
            sorted_claimants = sorted(claimants)
            winner = sorted_claimants[0]
        elif self.rule == "priority_rotation":
            # Deterministic tick-based rotation
            claimants_sorted = sorted(claimants)
            win_idx = tick % len(claimants_sorted)
            winner = claimants_sorted[win_idx]
        elif self.rule == "need_urgency":
            # Prioritize greatest energy deficit, break ties by lowest energy, then rotating tick
            def urgency_key(aid: str) -> tuple[int, int, int]:
                e = agent_energies.get(aid, 0)
                max_e = agent_max_energies.get(aid, 10)
                deficit = max_e - e
                # higher deficit first -> sort by -deficit, then e, then rotated id deterministically
                rot_tie = (agent_val(aid) + tick) % 1000
                return (-deficit, e, rot_tie)

            winner = min(claimants, key=urgency_key)
        elif self.rule == "starvation_prevention":
            # Prioritize agent with highest consecutive contested losses
            def starv_key(aid: str) -> tuple[int, int, str]:
                rec = self.get_or_create_record(aid)
                return (-rec.consecutive_losses, -rec.contested_losses, aid)

            winner = min(claimants, key=starv_key)
        else:  # 'fair_balanced'
            # Balanced score: urgency + starvation compensation
            # Phase 13.3: Deterministic round-robin tie-breaking rotates winner equitably on ties
            claimants_sorted = sorted(claimants)
            def balanced_key(aid: str) -> tuple[float, int]:
                e = agent_energies.get(aid, 0)
                max_e = agent_max_energies.get(aid, 10)
                deficit = max_e - e
                rec = self.get_or_create_record(aid)
                # Score: higher is better (negated for min)
                score = (deficit * 1.5) + (rec.consecutive_losses * 3.0) + (rec.contested_losses * 1.0)
                idx = claimants_sorted.index(aid)
                rot_tie = (idx - (tick % len(claimants_sorted))) % len(claimants_sorted)
                return (-score, rot_tie)

            winner = min(claimants, key=balanced_key)

        # Update records
        losers = sorted([aid for aid in claimants if aid != winner])
        w_rec = self.get_or_create_record(winner)
        w_rec.successful_claims += 1
        w_rec.consecutive_losses = 0
        gain = min(resource.recharge_amount, agent_max_energies.get(winner, 10) - agent_energies.get(winner, 0))
        w_rec.total_energy_gained += max(0, gain)

        for l_id in losers:
            l_rec = self.get_or_create_record(l_id)
            l_rec.contested_losses += 1
            l_rec.consecutive_losses += 1

        return winner, losers


# ==============================================================================
# Stage 4: Deadlock Detection & Bounded Recovery
# ==============================================================================


@dataclass(frozen=True, slots=True)
class DeadlockEvent:
    """Record of a detected deadlock condition."""

    tick: int
    deadlock_type: str  # "repeated_swap", "occupied_blocking", "vertex_deadlock", "starvation"
    affected_agents: tuple[str, ...]
    location: Position
    resolved: bool = False
    recovery_action_applied: str | None = None
    classification: str = "structural_deadlock"  # "structural_deadlock" or "cooperative_waiting"
    waiting_reason: str | None = None  # e.g. "resource_queue", "bottleneck_yielding", "circular_wait"


class DeadlockDetector:
    """Observable simulation deadlock detector with bounded recovery monitoring.

    Tracks spatial progress, repeated collisions, and position cycling across agents,
    distinguishing between benign cooperative waiting and true structural deadlocks.
    """

    def __init__(self, persistence_threshold: int = 2) -> None:
        self.persistence_threshold = persistence_threshold
        self.detected_events: list[DeadlockEvent] = []
        self.recovery_attempts: int = 0
        self.successful_recoveries: int = 0
        self.unresolved_deadlocks: int = 0
        self._unresolved_event_indices: set[int] = set()

        # Per-agent recent tracking
        self.recent_positions: dict[str, deque[Position]] = {}
        self.blocked_ticks: dict[str, int] = {}
        self.swap_history: deque[tuple[str, str, Position, Position]] = deque(maxlen=10)
        self.vertex_history: dict[Position, deque[tuple[int, tuple[str, ...]]]] = {}

    @property
    def cooperative_waiting_events(self) -> list[DeadlockEvent]:
        """Events classified as benign cooperative waiting (queueing or yielding)."""
        return [ev for ev in self.detected_events if ev.classification == "cooperative_waiting"]

    @property
    def structural_deadlock_events(self) -> list[DeadlockEvent]:
        """Events classified as true structural deadlocks (circular wait or unyielding lock)."""
        return [ev for ev in self.detected_events if ev.classification == "structural_deadlock"]

    @property
    def cooperative_waiting_count(self) -> int:
        return len(self.cooperative_waiting_events)

    @property
    def structural_deadlock_count(self) -> int:
        return len(self.structural_deadlock_events)

    @property
    def unresolved_structural_deadlocks(self) -> int:
        """Count of active unresolved events that are true structural deadlocks."""
        return sum(
            1 for idx in self._unresolved_event_indices
            if idx < len(self.detected_events) and self.detected_events[idx].classification == "structural_deadlock"
        )

    @property
    def unresolved_cooperative_waits(self) -> int:
        """Count of active unresolved events that are cooperative waits."""
        return sum(
            1 for idx in self._unresolved_event_indices
            if idx < len(self.detected_events) and self.detected_events[idx].classification == "cooperative_waiting"
        )

    @property
    def waiting_reasons_tally(self) -> dict[str, int]:
        """Distribution of waiting reasons across all detected events."""
        tally: dict[str, int] = {}
        for ev in self.detected_events:
            reason = ev.waiting_reason or "unknown"
            tally[reason] = tally.get(reason, 0) + 1
        return tally

    def update_tick(
        self,
        tick: int,
        agent_positions: dict[str, Position],
        intents: dict[str, tuple[Action, Position]],
        collisions: dict[str, str | None],
        moved_flags: dict[str, bool],
        coordinator: Any | None = None,
        resources: Sequence[Position] | Sequence[Resource] | None = None,
    ) -> list[DeadlockEvent]:
        """Analyze current tick outcomes and detect new or ongoing deadlocks."""
        # 0. Check if any previously unresolved deadlocks have now moved and recovered
        resolved_indices = set()
        for idx in self._unresolved_event_indices:
            ev = self.detected_events[idx]
            if any(moved_flags.get(aid, False) for aid in ev.affected_agents):
                self.successful_recoveries += 1
                resolved_indices.add(idx)
        self._unresolved_event_indices -= resolved_indices
        self.unresolved_deadlocks = len(self._unresolved_event_indices)

        resource_set: set[Position] = set()
        if resources is not None:
            for r in resources:
                if isinstance(r, Position):
                    resource_set.add(r)
                elif hasattr(r, "position"):
                    resource_set.add(r.position)
        if coordinator is not None:
            if hasattr(coordinator, "resource_commitments"):
                resource_set.update(coordinator.resource_commitments.values())
            if hasattr(coordinator, "resource_reservations"):
                resource_set.update(coordinator.resource_reservations.keys())

        def classify_occupied(aid: str, dest: Position) -> tuple[str, str]:
            if dest in resource_set or any(abs(dest.x - r.x) + abs(dest.y - r.y) <= 1 for r in resource_set):
                return ("cooperative_waiting", "resource_queue")
            if coordinator is not None and coordinator.resource_commitments.get(aid) in resource_set:
                return ("cooperative_waiting", "resource_queue")
            occupant = next((oid for oid, opos in agent_positions.items() if opos == dest and oid != aid), None)
            if occupant is not None:
                occ_intent_dest = intents.get(occupant, (None, None))[1]
                if moved_flags.get(occupant, False) or (occ_intent_dest is not None and occ_intent_dest != dest):
                    return ("cooperative_waiting", "bottleneck_yielding")
                if coordinator is not None and coordinator.agent_intentions.get(occupant) not in (None, dest):
                    return ("cooperative_waiting", "bottleneck_yielding")
                if coordinator is not None:
                    return ("cooperative_waiting", "bottleneck_yielding")
                return ("structural_deadlock", "unresolvable_block")
            return ("structural_deadlock", "unresolvable_block")

        def classify_swap(a1: str, a2: str, loc: Position) -> tuple[str, str]:
            if coordinator is not None:
                return ("cooperative_waiting", "swap_yielding")
            return ("structural_deadlock", "reciprocal_swap_lock")

        def classify_vertex(aid: str, dest: Position) -> tuple[str, str]:
            if dest in resource_set or any(abs(dest.x - r.x) + abs(dest.y - r.y) <= 1 for r in resource_set):
                return ("cooperative_waiting", "resource_contention")
            if coordinator is not None:
                return ("cooperative_waiting", "doorway_contention")
            return ("structural_deadlock", "vertex_contention_lock")

        new_events: list[DeadlockEvent] = []

        # 1. Update positions and blocked counters
        for aid, pos in agent_positions.items():
            self.recent_positions.setdefault(aid, deque(maxlen=8)).append(pos)
            if not moved_flags.get(aid, False) and collisions.get(aid) is not None:
                self.blocked_ticks[aid] = self.blocked_ticks.get(aid, 0) + 1
            elif moved_flags.get(aid, False):
                self.blocked_ticks[aid] = 0

        # 2. Check for Persistent Occupied Blocking (Agent blocked at same cell >= threshold)
        for aid, count in self.blocked_ticks.items():
            if count >= self.persistence_threshold and collisions.get(aid) == "occupied":
                pos = agent_positions[aid]
                dest = intents[aid][1]
                classification, reason = classify_occupied(aid, dest)
                event = DeadlockEvent(
                    tick=tick,
                    deadlock_type="occupied_blocking",
                    affected_agents=(aid,),
                    location=dest,
                    classification=classification,
                    waiting_reason=reason,
                )
                new_events.append(event)
                self.detected_events.append(event)

        # 3. Check for Swap Collisions between same pair
        swap_agents = [aid for aid, col in collisions.items() if col == "swap"]
        if len(swap_agents) >= 2:
            for i in range(len(swap_agents)):
                for j in range(i + 1, len(swap_agents)):
                    a1, a2 = swap_agents[i], swap_agents[j]
                    p1, p2 = agent_positions[a1], agent_positions[a2]
                    d1, d2 = intents[a1][1], intents[a2][1]
                    if d1 == p2 and d2 == p1:
                        # Recorded swap
                        recent_swaps = [s for s in self.swap_history if (s[0] == a1 and s[1] == a2) or (s[0] == a2 and s[1] == a1)]
                        if len(recent_swaps) >= 1:
                            classification, reason = classify_swap(a1, a2, p1)
                            event = DeadlockEvent(
                                tick=tick,
                                deadlock_type="repeated_swap",
                                affected_agents=(a1, a2),
                                location=p1,
                                classification=classification,
                                waiting_reason=reason,
                            )
                            new_events.append(event)
                            self.detected_events.append(event)
                        self.swap_history.append((a1, a2, p1, p2))

        # 4. Check for Repeated Vertex Collisions at same cell
        for aid, col in collisions.items():
            if col == "vertex":
                dest = intents[aid][1]
                v_queue = self.vertex_history.setdefault(dest, deque(maxlen=5))
                v_queue.append((tick, (aid,)))
                recent_v = [v for v in v_queue if tick - v[0] <= 2]
                if len(recent_v) >= self.persistence_threshold:
                    classification, reason = classify_vertex(aid, dest)
                    event = DeadlockEvent(
                        tick=tick,
                        deadlock_type="vertex_deadlock",
                        affected_agents=(aid,),
                        location=dest,
                        classification=classification,
                        waiting_reason=reason,
                    )
                    new_events.append(event)
                    self.detected_events.append(event)

        if new_events:
            self.recovery_attempts += len(new_events)
            for ev in new_events:
                self._unresolved_event_indices.add(self.detected_events.index(ev))
            self.unresolved_deadlocks = len(self._unresolved_event_indices)

        return new_events

    def record_recovery_outcome(self, event: DeadlockEvent, resolved: bool) -> None:
        """Record whether a deadlock event was successfully resolved."""
        if resolved:
            self.successful_recoveries += 1
        else:
            self.unresolved_deadlocks += 1


# ==============================================================================
# Stage 1 & 3: Multi-Agent Coordinator & Congestion Management
# ==============================================================================


@dataclass
class CoordinationSignal:
    """Permitted coordination signals exchanged between agents."""

    intended_destination: Position | None = None
    targeted_resource: Position | None = None
    congested_cells: tuple[Position, ...] = ()
    yield_requested: bool = False


class MultiAgentCoordinator:
    """Coordination broker enabling voluntary multi-agent cooperation.

    Strictly observation-bounded: only exchanges permitted public intentions,
    such as intended next cells and targeted resources, to:
    1. Pre-empt vertex and swap collisions by advising lower-priority agents to choose
       an equal or near-equal detour action.
    2. Load-balance resource detours across multiple available resources.
    3. Share congestion / deadlock warnings and provide actionable recovery behaviors.
    """

    def __init__(
        self,
        enabled: bool = True,
        world_width: int | None = None,
        world_height: int | None = None,
        coordination_mode: str = "centralized",
        communication_radius: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.world_width = world_width
        self.world_height = world_height
        self.coordination_mode = coordination_mode
        self.communication_radius = communication_radius
        self.agent_intentions: dict[str, Position] = {}
        self.previous_intentions: dict[str, Position] = {}
        self.agent_positions: dict[str, Position] = {}
        self.previous_positions: dict[str, Position] = {}
        self.resource_reservations: dict[Position, str] = {}
        self.resource_commitments: dict[str, Position] = {}
        self.congested_cells: set[Position] = set()
        self.known_obstacles: set[Position] = set()

    def reset(self) -> None:
        """Reset coordinator state between episodes or explicit resets."""
        self.agent_intentions.clear()
        self.previous_intentions.clear()
        self.agent_positions.clear()
        self.previous_positions.clear()
        self.resource_reservations.clear()
        self.resource_commitments.clear()
        self.congested_cells.clear()
        self.known_obstacles.clear()

    def reset_tick(self) -> None:
        """Clear per-tick intentions while preserving persistent resource commitments."""
        self.previous_intentions = dict(self.agent_intentions)
        self.previous_positions = dict(self.agent_positions)
        self.agent_intentions.clear()

    def register_intent(
        self,
        agent_id: str,
        destination: Position,
        targeted_resource: Position | None = None,
        current_position: Position | None = None,
    ) -> None:
        """Register an agent's intended movement and resource destination for the tick."""
        if not self.enabled:
            return
        self.agent_intentions[agent_id] = destination
        if current_position is not None:
            self.agent_positions[agent_id] = current_position
        if targeted_resource is not None:
            self.resource_commitments[agent_id] = targeted_resource
            if targeted_resource not in self.resource_reservations:
                self.resource_reservations[targeted_resource] = agent_id
        else:
            self.clear_commitment(agent_id)

    def is_in_range(self, p1: Position, p2: Position) -> bool:
        """Check if Euclidean distance between p1 and p2 is <= communication_radius."""
        return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 <= (self.communication_radius ** 2 + 1e-9)

    def get_visible_peers(
        self,
        agent_id: str,
        pos: Position,
        current_positions: dict[str, Position] | None = None,
    ) -> set[str]:
        """Return the set of peer agent IDs whose intentions are visible to agent_id at pos."""
        if self.coordination_mode == "centralized":
            peers = set(self.agent_intentions.keys())
            if hasattr(self, "previous_intentions"):
                peers.update(self.previous_intentions.keys())
            if current_positions:
                peers.update(current_positions.keys())
            peers.discard(agent_id)
            return peers

        visible: set[str] = set()
        if current_positions:
            for oid, opos in current_positions.items():
                if oid != agent_id and self.is_in_range(pos, opos):
                    visible.add(oid)

        candidates = set(self.agent_positions.keys())
        if hasattr(self, "previous_positions"):
            candidates.update(self.previous_positions.keys())

        for oid in candidates:
            if oid == agent_id or oid in visible:
                continue
            opos = self.agent_positions.get(oid)
            if opos is None and hasattr(self, "previous_positions"):
                opos = self.previous_positions.get(oid)
            if opos is not None and self.is_in_range(pos, opos):
                visible.add(oid)

        return visible

    def clear_commitment(self, agent_id: str) -> None:
        """Clear persistent resource commitment when agent arrives, completes, or exhausts."""
        old_res = self.resource_commitments.pop(agent_id, None)
        if old_res is not None and self.resource_reservations.get(old_res) == agent_id:
            self.resource_reservations.pop(old_res, None)
            # Reassign reservation to another agent committed to this resource, if any
            for other_aid, other_res in self.resource_commitments.items():
                if other_res == old_res:
                    self.resource_reservations[old_res] = other_aid
                    break

    def suggest_alternative_move(
        self,
        agent_id: str,
        pos: Position,
        candidate_actions: Sequence[Action],
        destination_map: dict[Action, Position],
        tick: int,
        preferred_action: Action | None = None,
        current_positions: dict[str, Position] | None = None,
        target: Position | None = None,
        num_agents: int = 8,
        energy: int | None = None,
        completed_positions: Sequence[Position] | set[Position] | None = None,
        obstacle_positions: Sequence[Position] | set[Position] | None = None,
        dynamic_obstacles: Sequence[Any] | None = None,
    ) -> Action | None:
        """Suggest an alternative action if the preferred action would cause a conflict."""
        if not self.enabled or not candidate_actions:
            return None

        # Identify preferred action (defaults to candidate_actions[0] if omitted)
        pref = preferred_action if preferred_action is not None else candidate_actions[0]
        if pref == Action.RECHARGE:
            return None  # Never hijack an intended in-place recharge

        pref_dest = destination_map.get(pref, pos)
        if pref_dest == pos:
            return None

        self.agent_positions[agent_id] = pos

        comp_set = set(completed_positions) if completed_positions is not None else set()
        obs_set = set(obstacle_positions) if obstacle_positions is not None else set()

        if self.coordination_mode == "centralized":
            if not hasattr(self, "known_obstacles"):
                self.known_obstacles = set()
            if obs_set:
                self.known_obstacles.update(obs_set)
            all_obs = self.known_obstacles.union(obs_set)
        else:
            # Decentralized mode: strictly use agent's own local obstacle memory
            all_obs = obs_set

        visible_peers = self.get_visible_peers(agent_id, pos, current_positions)
        visible_intentions = {
            oid: od for oid, od in self.agent_intentions.items() if oid in visible_peers
        }
        visible_prev_intentions = {
            oid: od for oid, od in getattr(self, "previous_intentions", {}).items() if oid in visible_peers
        }

        # Check for swap conflict with currently occupied cell
        is_swap_conflict = False
        swap_partner = None
        if current_positions:
            for oid, opos in current_positions.items():
                if oid != agent_id and opos == pref_dest and opos not in comp_set:
                    if self.coordination_mode == "centralized" or oid in visible_peers:
                        oint = visible_intentions.get(oid)
                        prev_oint = visible_prev_intentions.get(oid)
                        if oint == pos or prev_oint == pos:
                            is_swap_conflict = True
                            swap_partner = oid
                            break

        # Never redirect away from a resource pad or target waypoint unless in swap conflict
        if self.coordination_mode == "centralized":
            relevant_commitments = set(self.resource_commitments.values())
            relevant_reservations = set(self.resource_reservations.keys())
        else:
            relevant_commitments = {
                res for aid, res in self.resource_commitments.items() if aid in visible_peers
            }
            relevant_reservations = {
                res for res, aid in self.resource_reservations.items() if aid in visible_peers
            }

        if (pref_dest in relevant_commitments or pref_dest in relevant_reservations) and not is_swap_conflict:
            return None
        if target is not None and pref_dest == target and not is_swap_conflict:
            return None

        my_val = agent_val(agent_id)
        rot_base = max(2, num_agents)
        my_prio = ((my_val - 1) + tick) % rot_base

        # Check if the preferred action causes a conflict
        has_conflict = False

        dyn_current_cells: set[Position] = set()
        dyn_predicted_cells: set[Position] = set()
        if dynamic_obstacles:
            for obs in dynamic_obstacles:
                obs_pos = getattr(obs, "position", None)
                if obs_pos is None:
                    continue
                if self.coordination_mode == "centralized" or self.is_in_range(pos, obs_pos):
                    dyn_current_cells.add(obs_pos)
                    vel = getattr(obs, "velocity", (0, 0))
                    nxt = Position(obs_pos.x + vel[0], obs_pos.y + vel[1])
                    dyn_predicted_cells.add(nxt)
                    all_obs.add(obs_pos)
                    all_obs.add(nxt)

        if pref_dest in dyn_current_cells or pref_dest in dyn_predicted_cells:
            has_conflict = True

        def is_crossroad(p: Position) -> bool:
            diags = (
                Position(p.x - 1, p.y - 1),
                Position(p.x + 1, p.y - 1),
                Position(p.x - 1, p.y + 1),
                Position(p.x + 1, p.y + 1),
            )
            cardinals = (
                Position(p.x, p.y - 1),
                Position(p.x, p.y + 1),
                Position(p.x - 1, p.y),
                Position(p.x + 1, p.y),
            )
            if self.coordination_mode == "centralized":
                return all(d in all_obs for d in diags) and all(c not in all_obs for c in cardinals)
            cardinals_open = all(
                c not in all_obs and (self.world_width is None or 0 <= c.x < self.world_width) and (self.world_height is None or 0 <= c.y < self.world_height)
                for c in cardinals
            )
            if not cardinals_open:
                return False
            known_diags = sum(1 for d in diags if d in all_obs)
            if known_diags == 4:
                return True
            if known_diags >= 2:
                if self.world_width is not None and (p.x <= 1 or p.x >= self.world_width - 2):
                    return False
                if self.world_height is not None and (p.y <= 1 or p.y >= self.world_height - 2):
                    return False
                return True
            return False

        if is_swap_conflict and swap_partner is not None:
            if is_crossroad(pos):
                return None  # Vehicle clearing crossroad has right-of-way
            elif is_crossroad(pref_dest):
                has_conflict = True  # Vehicle entering crossroad yields to occupant
            else:
                eff_target = target if target is not None else self.resource_commitments.get(agent_id)
                my_has_lateral = False
                for a in candidate_actions:
                    if a in (pref, Action.RECHARGE):
                        continue
                    d = destination_map.get(a, pos)
                    if d not in all_obs and d not in comp_set and (self.world_width is None or 0 <= d.x < self.world_width) and (self.world_height is None or 0 <= d.y < self.world_height):
                        if (pos.x == pref_dest.x and d.x != pos.x) or (pos.y == pref_dest.y and d.y != pos.y):
                            my_has_lateral = True
                            break

                partner_has_lateral = False
                p_pos = pref_dest
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    pd = Position(p_pos.x + dx, p_pos.y + dy)
                    if pd != pos and pd not in all_obs and pd not in comp_set:
                        if (self.world_width is None or 0 <= pd.x < self.world_width) and (self.world_height is None or 0 <= pd.y < self.world_height):
                            if (pos.x == p_pos.x and pd.x != p_pos.x) or (pos.y == p_pos.y and pd.y != p_pos.y):
                                partner_has_lateral = True
                                break

                if my_has_lateral and not partner_has_lateral:
                    has_conflict = True
                elif partner_has_lateral and not my_has_lateral:
                    return None
                else:
                    partner_prio = ((agent_val(swap_partner) - 1) + tick) % rot_base
                    if partner_prio <= my_prio:
                        has_conflict = True
                    else:
                        return None  # Higher priority proceeds

        # 1. Vertex conflict with registered intents
        if not has_conflict:
            for oid, od in visible_intentions.items():
                if oid != agent_id and od == pref_dest:
                    oprio = ((agent_val(oid) - 1) + tick) % rot_base
                    if oprio <= my_prio:
                        has_conflict = True
                        break

        # 2. Swap conflict or occupied conflict with another agent
        if not has_conflict and current_positions:
            for oid, opos in current_positions.items():
                if oid != agent_id and opos == pref_dest and opos not in comp_set:
                    if self.coordination_mode == "centralized" or oid in visible_peers:
                        oint = visible_intentions.get(oid)
                        if oint == pos:
                            oprio = ((agent_val(oid) - 1) + tick) % rot_base
                            if oprio <= my_prio:
                                has_conflict = True
                                break
                        elif oint == pref_dest:
                            has_conflict = True
                            break

        if not has_conflict:
            return None

        eff_target = target if target is not None else self.resource_commitments.get(agent_id)
        cur_dist = abs(eff_target.x - pos.x) + abs(eff_target.y - pos.y) if eff_target is not None else 0

        # Preferred action conflicts! Filter candidate actions for safe alternatives
        safe_actions: list[Action] = []
        for a in candidate_actions:
            if a == pref:
                continue
            if a == Action.RECHARGE:
                if pos in dyn_predicted_cells:
                    continue  # Unsafe to wait if dynamic obstacle is moving directly into pos!
                safe_actions.append(a)
                continue

            # In-bounds and destination mapping check
            if a not in destination_map or destination_map[a] is None:
                continue
            d = destination_map[a]
            if d == pos:  # A non-recharge action must change position
                continue
            if d.x < 0 or d.y < 0:
                continue
            if self.world_width is not None and d.x >= self.world_width:
                continue
            if self.world_height is not None and d.y >= self.world_height:
                continue

            if d in self.congested_cells:
                continue
            if d in all_obs:
                continue
            if any(oid != agent_id and od == d for oid, od in visible_intentions.items()):
                continue

            s_conf = False
            if current_positions:
                for oid, opos in current_positions.items():
                    if oid != agent_id and opos == d and opos not in comp_set:
                        if self.coordination_mode == "centralized" or oid in visible_peers:
                            oint = visible_intentions.get(oid)
                            # Stationary / exhausted (oint is None) or staying / swapping
                            if oint is None or oint == pos or oint == d:
                                s_conf = True
                                break
            if s_conf:
                continue

            # In non-swap conflicts, avoid taking wasteful moves that increase distance away from target
            if not is_swap_conflict and eff_target is not None:
                dist_d = abs(eff_target.x - d.x) + abs(eff_target.y - d.y)
                if dist_d > cur_dist:
                    continue

            # Ensure alternative move does not cause fatal energy starvation
            if eff_target is not None and energy is not None:
                dist_d = abs(eff_target.x - d.x) + abs(eff_target.y - d.y)
                if dist_d > (energy - 1):
                    continue

            # If agent has a committed resource, ensure move does not make that resource unreachable
            targeted_res = self.resource_commitments.get(agent_id)
            if targeted_res is not None and energy is not None and targeted_res != eff_target:
                dist_to_res = abs(targeted_res.x - d.x) + abs(targeted_res.y - d.y)
                if dist_to_res > (energy - 1):
                    continue

            safe_actions.append(a)

        if not safe_actions:
            return None

        # Rank safe alternatives by proximity to goal/target or committed resource, favoring lateral detours
        rank_target = target if target is not None else self.resource_commitments.get(agent_id)
        if rank_target is not None:
            def alt_rank(act: Action) -> tuple[float, int]:
                if act == Action.RECHARGE:
                    return (999.0, 0)
                d = destination_map[act]
                dist = abs(rank_target.x - d.x) + abs(rank_target.y - d.y)
                is_lateral = (pos.x == pref_dest.x and d.x != pos.x) or (pos.y == pref_dest.y and d.y != pos.y)
                lateral_bonus = -3.0 if is_lateral else 0.0
                return (dist + lateral_bonus, 0)
            safe_actions.sort(key=alt_rank)

        return safe_actions[0]

    def get_recovery_action(
        self,
        agent_id: str,
        pos: Position,
        legal_actions: Sequence[Action],
        destination_map: dict[Action, Position],
        blocked_cell: Position | None = None,
        partner_id: str | None = None,
        tick: int = 0,
        num_agents: int = 8,
        current_positions: dict[str, Position] | None = None,
        completed_positions: Sequence[Position] | set[Position] | None = None,
        obstacle_positions: Sequence[Position] | set[Position] | None = None,
        dynamic_obstacles: Sequence[Any] | None = None,
    ) -> Action | None:
        """Provide an actionable sidestep or yield behavior to resolve detected deadlocks."""
        if not self.enabled or not legal_actions:
            return None

        my_val = agent_val(agent_id)
        rot_base = max(2, num_agents)
        my_prio = ((my_val - 1) + tick) % rot_base

        if partner_id is not None:
            partner_val = agent_val(partner_id)
            partner_prio = ((partner_val - 1) + tick) % rot_base
            if my_prio < partner_prio:
                return None  # Higher priority proceeds

        comp_set = set(completed_positions) if completed_positions is not None else set()
        obs_set = set(obstacle_positions) if obstacle_positions is not None else set()

        visible_peers = self.get_visible_peers(agent_id, pos, current_positions)
        visible_intentions = {
            oid: od for oid, od in self.agent_intentions.items() if oid in visible_peers
        }

        orthogonal_candidates: list[Action] = []
        if blocked_cell is not None:
            if blocked_cell.x == pos.x:  # Vertical block -> try horizontal sidestep
                for a in (Action.RIGHT, Action.LEFT):
                    if a in legal_actions and destination_map.get(a) not in self.congested_cells:
                        orthogonal_candidates.append(a)
            elif blocked_cell.y == pos.y:  # Horizontal block -> try vertical sidestep
                for a in (Action.DOWN, Action.UP):
                    if a in legal_actions and destination_map.get(a) not in self.congested_cells:
                        orthogonal_candidates.append(a)

        for cand in orthogonal_candidates:
            if cand not in destination_map or destination_map[cand] is None:
                continue
            d = destination_map[cand]
            if d.x < 0 or d.y < 0:
                continue
            if self.world_width is not None and d.x >= self.world_width:
                continue
            if self.world_height is not None and d.y >= self.world_height:
                continue
            if d in obs_set:
                continue
            if any(oid != agent_id and od == d for oid, od in visible_intentions.items()):
                continue
            if current_positions:
                s_conf = False
                for oid, opos in current_positions.items():
                    if oid != agent_id and opos == d and opos not in comp_set:
                        if self.coordination_mode == "centralized" or oid in visible_peers:
                            oint = visible_intentions.get(oid)
                            if oint is None or oint == pos or oint == d:
                                s_conf = True
                                break
                if s_conf:
                    continue
            return cand

        if Action.RECHARGE in legal_actions:
            return Action.RECHARGE

        return None

    def get_resource_load(
        self,
        resource_pos: Position,
        exclude_agent: str | None = None,
        querying_agent_pos: Position | None = None,
    ) -> int | None:
        """Return number of agents currently committed to or targeting this resource.

        In decentralized mode, returns None if the resource is outside the querying agent's
        local communication horizon, indicating that the queue load is unknown.
        """
        if not self.enabled:
            return 0
        if self.coordination_mode == "centralized":
            return sum(
                1 for aid, pos in self.resource_commitments.items()
                if aid != exclude_agent and pos == resource_pos
            )
        if querying_agent_pos is None:
            return None
        # In decentralized mode: if resource itself is beyond communication range, load is unknown
        if not self.is_in_range(querying_agent_pos, resource_pos):
            return None
        load = 0
        for aid, pos in self.resource_commitments.items():
            if aid == exclude_agent or pos != resource_pos:
                continue
            aid_pos = self.agent_positions.get(aid)
            if aid_pos is not None and self.is_in_range(querying_agent_pos, aid_pos):
                load += 1
        return load

    def mark_congested(self, pos: Position) -> None:
        """Mark a cell as a congested chokepoint."""
        self.congested_cells.add(pos)

    def clear_congestion(self, pos: Position) -> None:
        self.congested_cells.discard(pos)


# ==============================================================================
# Stage 5: Fairness & Starvation Monitor
# ==============================================================================


@dataclass
class AgentFairnessMetrics:
    """Per-agent fairness and execution tracking metrics."""

    agent_id: str
    goal_completed: bool = False
    energy_exhausted: bool = False
    steps_taken: int = 0
    ticks_blocked: int = 0
    ticks_to_first_move: int | None = None
    total_energy_consumed: int = 0
    resource_claims_attempted: int = 0
    resource_claims_won: int = 0
    resource_claims_lost: int = 0
    energy_gained_from_resources: int = 0
    vertex_conflicts: int = 0
    swap_conflicts: int = 0
    occupied_conflicts: int = 0
    starvation_events: int = 0
    recovery_interventions: int = 0


class FairnessMonitor:
    """Monitors per-agent execution, detects starvation, and computes Jain's index."""

    def __init__(self, agent_ids: Sequence[str]) -> None:
        self.metrics: dict[str, AgentFairnessMetrics] = {
            aid: AgentFairnessMetrics(agent_id=aid) for aid in agent_ids
        }
        self._consecutive_blocked: dict[str, int] = {aid: 0 for aid in agent_ids}
        self.energy_history: dict[str, list[int]] = {aid: [] for aid in agent_ids}
        self.resource_load_distribution: dict[Position, int] = {}

    def record_tick(
        self,
        tick: int,
        agent_positions_before: dict[str, Position],
        agent_positions_after: dict[str, Position],
        agent_energies_before: dict[str, int],
        agent_energies_after: dict[str, int],
        actions: dict[str, Action],
        success_flags: dict[str, bool],
        collisions: dict[str, str | None],
        recharge_claims: dict[str, bool],  # agent_id -> won (True) or lost (False)
        energy_gained: dict[str, int],
    ) -> None:
        """Update metrics for one completed tick."""
        for aid, m in self.metrics.items():
            succ = success_flags.get(aid, False)
            col = collisions.get(aid)
            pos_before = agent_positions_before.get(aid)
            pos_after = agent_positions_after.get(aid)

            if succ and pos_before != pos_after:
                m.steps_taken += 1
                if m.ticks_to_first_move is None:
                    m.ticks_to_first_move = tick
                self._consecutive_blocked[aid] = 0
            elif not succ and col is not None:
                m.ticks_blocked += 1
                self._consecutive_blocked[aid] += 1
                if self._consecutive_blocked[aid] >= 3:
                    m.starvation_events += 1

            if col == "vertex":
                m.vertex_conflicts += 1
            elif col == "swap":
                m.swap_conflicts += 1
            elif col == "occupied":
                m.occupied_conflicts += 1

            # Energy
            e_before = agent_energies_before.get(aid, 0)
            e_after = agent_energies_after.get(aid, 0)
            self.energy_history.setdefault(aid, []).append(e_after)

            gain = energy_gained.get(aid, 0)
            spent = max(0, (e_before + gain) - e_after)
            m.total_energy_consumed += spent
            m.energy_gained_from_resources += gain

            if gain > 0 and pos_after is not None:
                self.resource_load_distribution[pos_after] = self.resource_load_distribution.get(pos_after, 0) + 1

            if aid in recharge_claims:
                m.resource_claims_attempted += 1
                if recharge_claims[aid]:
                    m.resource_claims_won += 1
                else:
                    m.resource_claims_lost += 1

    def get_unused_resources_count(self, all_resources: Sequence[Resource]) -> int:
        """Return the count of world resources that were never recharged at."""
        return sum(1 for r in all_resources if self.resource_load_distribution.get(r.position, 0) == 0)

    def record_intervention(self, agent_id: str) -> None:
        if agent_id in self.metrics:
            self.metrics[agent_id].recovery_interventions += 1

    def record_final(self, agent_id: str, completed: bool, exhausted: bool) -> None:
        if agent_id in self.metrics:
            self.metrics[agent_id].goal_completed = completed
            self.metrics[agent_id].energy_exhausted = exhausted

    def compute_jains_fairness(self, metric_name: str = "goal_completion") -> float:
        """Compute Jain's fairness index for a specified non-negative metric.

        Supported metric names:
            'goal_completion': 1.0 for completed, 0.0 for not.
            'energy_gained': Total energy gained from resources.
            'spatial_progress': Steps taken.
            'resource_access': Successful claims count.
        """
        vals: list[float] = []
        for m in self.metrics.values():
            if metric_name == "goal_completion":
                vals.append(1.0 if m.goal_completed else 0.0)
            elif metric_name == "energy_gained":
                vals.append(float(m.energy_gained_from_resources))
            elif metric_name == "spatial_progress":
                vals.append(float(m.steps_taken))
            elif metric_name == "resource_access":
                vals.append(float(m.resource_claims_won))
            else:
                vals.append(float(m.steps_taken))
        return jains_fairness_index(vals)
