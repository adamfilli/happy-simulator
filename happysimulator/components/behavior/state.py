"""Internal state model for behavioral agents.

Tracks needs, mood, beliefs, knowledge, and episodic memory. State
values decay over simulated time to model natural drift (needs grow,
mood returns to neutral).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """A single episodic memory recorded by an agent.

    Attributes:
        time: Simulation time (seconds) when the event occurred.
        event_type: Type label of the triggering event.
        source: Name of the entity or agent that originated the event.
        valence: Emotional valence of the memory (-1.0 negative to 1.0 positive).
        details: Arbitrary extra information.
    """

    time: float
    event_type: str
    source: str = ""
    valence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Mutable internal state of a behavioral agent.

    All scalar fields are bounded to [0, 1] except beliefs which range
    [-1, 1] (opinion strength).

    Attributes:
        satisfaction: Overall satisfaction level (0-1).
        energy: Energy/motivation level (0-1).
        mood: Current mood (0=negative, 0.5=neutral, 1=positive).
        beliefs: Topic-keyed opinion values (-1 to 1).
        needs: Named need levels (0=satisfied, 1=urgent).
        knowledge: Set of known facts/topics.
    """

    satisfaction: float = 0.5
    energy: float = 1.0
    mood: float = 0.5
    beliefs: dict[str, float] = field(default_factory=dict)
    needs: dict[str, float] = field(default_factory=dict)
    knowledge: set[str] = field(default_factory=set)
    _memories: deque[Memory] = field(default_factory=lambda: deque(maxlen=100))

    def add_memory(self, memory: Memory) -> None:
        """Record a new memory, evicting the oldest if at capacity."""
        self._memories.append(memory)

    def recent_memories(self, n: int = 5) -> list[Memory]:
        """Return the *n* most recent memories (newest first)."""
        items = list(self._memories)
        items.reverse()
        return items[:n]

    def average_recent_valence(self, n: int = 5) -> float:
        """Mean valence of the *n* most recent memories."""
        recent = self.recent_memories(n)
        if not recent:
            return 0.0
        return sum(m.valence for m in recent) / len(recent)

    def decay(self, dt_seconds: float) -> None:
        """Apply time-based decay to needs, mood, and energy.

        - Needs drift upward (grow more urgent) at 0.01/s.
        - Mood drifts toward neutral (0.5) at 0.02/s.
        - Energy drifts downward at 0.005/s.

        Args:
            dt_seconds: Elapsed simulation time in seconds.
        """
        if dt_seconds <= 0:
            return

        # Needs drift up
        for k in list(self.needs.keys()):
            self.needs[k] = min(1.0, self.needs[k] + 0.01 * dt_seconds)

        # Mood drifts to neutral
        drift_rate = 0.02 * dt_seconds
        if self.mood > 0.5:
            self.mood = max(0.5, self.mood - drift_rate)
        elif self.mood < 0.5:
            self.mood = min(0.5, self.mood + drift_rate)

        # Energy drifts down
        self.energy = max(0.0, self.energy - 0.005 * dt_seconds)
