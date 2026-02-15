"""Garbage collector pause injection model.

Simulates GC pauses that affect application entities. Different GC
strategies model the tradeoff between pause duration and frequency:

- StopTheWorld: Long, infrequent pauses that halt all processing.
- ConcurrentGC: Shorter, more frequent pauses with some concurrent work.
- GenerationalGC: Frequent minor collections with rare major collections.

The GarbageCollector self-schedules periodic collection events and injects
delays into a target entity's processing pipeline.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GC strategies
# ---------------------------------------------------------------------------


class GCStrategy(ABC):
    """Strategy defining GC pause characteristics."""

    @abstractmethod
    def pause_duration_s(self, heap_pressure: float) -> float:
        """Return pause duration in seconds for current heap pressure.

        Args:
            heap_pressure: Fraction of heap used (0.0 to 1.0).
        """
        ...

    @abstractmethod
    def collection_interval_s(self) -> float:
        """Return seconds between collections."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this GC strategy."""
        ...


class StopTheWorld(GCStrategy):
    """Full heap collection with long pauses.

    Args:
        base_pause_s: Base pause duration (default 50ms).
        interval_s: Seconds between collections (default 10s).
        pressure_multiplier: How much heap pressure extends pause (default 3.0).
    """

    def __init__(
        self,
        *,
        base_pause_s: float = 0.05,
        interval_s: float = 10.0,
        pressure_multiplier: float = 3.0,
    ) -> None:
        self._base_pause_s = base_pause_s
        self._interval_s = interval_s
        self._pressure_multiplier = pressure_multiplier

    def pause_duration_s(self, heap_pressure: float) -> float:
        # Pause grows linearly with heap pressure
        multiplier = 1.0 + self._pressure_multiplier * heap_pressure
        jitter = 0.8 + 0.4 * random.random()
        return self._base_pause_s * multiplier * jitter

    def collection_interval_s(self) -> float:
        return self._interval_s

    @property
    def name(self) -> str:
        return "StopTheWorld"


class ConcurrentGC(GCStrategy):
    """Mostly concurrent collection with short pauses.

    Most work happens concurrently; only marking/remarking phases pause.

    Args:
        pause_s: Pause duration per cycle (default 5ms).
        interval_s: Seconds between collections (default 2s).
    """

    def __init__(
        self,
        *,
        pause_s: float = 0.005,
        interval_s: float = 2.0,
    ) -> None:
        self._pause_s = pause_s
        self._interval_s = interval_s

    def pause_duration_s(self, heap_pressure: float) -> float:
        # Small jitter
        jitter = 0.9 + 0.2 * random.random()
        return self._pause_s * jitter

    def collection_interval_s(self) -> float:
        return self._interval_s

    @property
    def name(self) -> str:
        return "ConcurrentGC"


class GenerationalGC(GCStrategy):
    """Generational collection with minor and major cycles.

    Minor collections are fast and frequent. Major collections are slower
    and triggered based on heap pressure.

    Args:
        minor_pause_s: Minor GC pause duration (default 2ms).
        major_pause_s: Major GC pause duration (default 30ms).
        minor_interval_s: Seconds between minor collections (default 1s).
        major_threshold: Heap pressure threshold for major GC (default 0.75).
    """

    def __init__(
        self,
        *,
        minor_pause_s: float = 0.002,
        major_pause_s: float = 0.03,
        minor_interval_s: float = 1.0,
        major_threshold: float = 0.75,
    ) -> None:
        self._minor_pause_s = minor_pause_s
        self._major_pause_s = major_pause_s
        self._minor_interval_s = minor_interval_s
        self._major_threshold = major_threshold
        self._collections_since_major: int = 0

    def pause_duration_s(self, heap_pressure: float) -> float:
        self._collections_since_major += 1
        if heap_pressure >= self._major_threshold:
            self._collections_since_major = 0
            jitter = 0.8 + 0.4 * random.random()
            return self._major_pause_s * jitter
        jitter = 0.9 + 0.2 * random.random()
        return self._minor_pause_s * jitter

    def collection_interval_s(self) -> float:
        return self._minor_interval_s

    @property
    def name(self) -> str:
        return "GenerationalGC"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GCStats:
    """Frozen snapshot of garbage collector statistics.

    Attributes:
        collections: Total collection cycles.
        total_pause_s: Cumulative pause time.
        max_pause_s: Longest single pause.
        min_pause_s: Shortest single pause.
        minor_collections: Minor GC cycles (GenerationalGC only).
        major_collections: Major GC cycles (GenerationalGC only).
        strategy_name: Name of the active GC strategy.
    """

    collections: int = 0
    total_pause_s: float = 0.0
    max_pause_s: float = 0.0
    min_pause_s: float = 0.0
    minor_collections: int = 0
    major_collections: int = 0
    strategy_name: str = ""

    @property
    def avg_pause_s(self) -> float:
        return self.total_pause_s / self.collections if self.collections else 0.0


# ---------------------------------------------------------------------------
# GarbageCollector entity
# ---------------------------------------------------------------------------


_GC_COLLECT = "_gc_collect"


class GarbageCollector(Entity):
    """GC pause injection model.

    Self-schedules periodic collection events and injects pauses that
    block processing of the target entity. The ``pause()`` generator
    method can be called from any entity's handle_event to inject a
    GC-aware delay.

    Args:
        name: Entity name.
        strategy: GC strategy. Defaults to GenerationalGC.
        heap_pressure: Fixed heap pressure (0.0-1.0). If None, grows
            over time from 0.3 to 0.9 based on collection count.

    Example::

        gc = GarbageCollector("jvm_gc", strategy=StopTheWorld())
        sim = Simulation(entities=[gc, ...], ...)

        # In another entity's handle_event:
        yield from gc.pause()  # inject GC pause if due
    """

    def __init__(
        self,
        name: str,
        *,
        strategy: GCStrategy | None = None,
        heap_pressure: float | None = None,
    ) -> None:
        super().__init__(name)
        self._strategy = strategy or GenerationalGC()
        self._fixed_pressure = heap_pressure
        self._collection_count: int = 0
        self._paused: bool = False

        # Stats
        self._total_pause_s: float = 0.0
        self._max_pause_s: float = 0.0
        self._min_pause_s: float = float("inf")
        self._minor_collections: int = 0
        self._major_collections: int = 0

    @property
    def collection_count(self) -> int:
        """Total number of GC collections performed."""
        return self._collection_count

    @property
    def stats(self) -> GCStats:
        """Frozen snapshot of GC statistics."""
        return GCStats(
            collections=self._collection_count,
            total_pause_s=self._total_pause_s,
            max_pause_s=self._max_pause_s,
            min_pause_s=self._min_pause_s if self._collection_count > 0 else 0.0,
            minor_collections=self._minor_collections,
            major_collections=self._major_collections,
            strategy_name=self._strategy.name,
        )

    def _heap_pressure(self) -> float:
        """Current heap pressure."""
        if self._fixed_pressure is not None:
            return self._fixed_pressure
        # Simulated pressure: grows over time, sawtooth pattern
        base = 0.3 + 0.6 * min(1.0, self._collection_count / 50.0)
        return min(0.95, base)

    def _schedule_next(self) -> Event:
        """Create the next scheduled GC event."""
        interval = self._strategy.collection_interval_s()
        return Event(
            time=self.now + interval,
            event_type=_GC_COLLECT,
            target=self,
        )

    def _do_collect(self) -> float:
        """Perform a collection and return pause duration."""
        pressure = self._heap_pressure()
        pause = self._strategy.pause_duration_s(pressure)

        self._collection_count += 1
        self._total_pause_s += pause

        if pause > self._max_pause_s:
            self._max_pause_s = pause
        if pause < self._min_pause_s:
            self._min_pause_s = pause

        # Track minor vs major for generational
        if isinstance(self._strategy, GenerationalGC):
            if pressure >= self._strategy._major_threshold:
                self._major_collections += 1
            else:
                self._minor_collections += 1

        logger.debug(
            "[%s] GC collection #%d: pause=%.4fs, pressure=%.2f",
            self.name,
            self._collection_count,
            pause,
            pressure,
        )
        return pause

    def pause(self) -> Generator[float, None, float]:
        """Inject a GC pause at the current time.

        Call from any entity's handle_event to simulate GC overhead.
        Returns the pause duration.
        """
        pause = self._do_collect()
        yield pause
        return pause

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Handle scheduled GC collection events."""
        if event.event_type == _GC_COLLECT:
            pause = self._do_collect()
            yield pause
            return [self._schedule_next()]
        return []

    def prime(self) -> Event:
        """Create the initial GC scheduling event.

        Call this and pass to ``sim.schedule()`` to start the GC cycle.
        Requires the entity to have a clock (be registered with a Simulation).
        """
        return Event(
            time=self.now,
            event_type=_GC_COLLECT,
            target=self,
        )

    def __repr__(self) -> str:
        return (
            f"GarbageCollector('{self.name}', strategy={self._strategy.name}, "
            f"collections={self._collection_count})"
        )
