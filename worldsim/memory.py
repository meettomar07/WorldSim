"""Bounded, immutable transition memory using observed state matching."""
from collections import deque
from dataclasses import dataclass

from .actions import Action
from .models import Observation


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    observation: Observation
    action: Action
    reward: float
    resulting_observation: Observation


class ExperienceMemory:
    """FIFO memory; entries match when observed entity coordinates are equal."""
    def __init__(self, capacity: int = 100) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._entries: deque[MemoryEntry] = deque(maxlen=capacity)

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def record(self, observation: Observation, action: Action, reward: float, resulting_observation: Observation) -> None:
        self._entries.append(MemoryEntry(observation, action, reward, resulting_observation))

    def retrieve(self, observation: Observation) -> tuple[MemoryEntry, ...]:
        position = observation.entity_position
        return tuple(entry for entry in self._entries if entry.observation.entity_position == position)

    def clear(self) -> None:
        self._entries.clear()
