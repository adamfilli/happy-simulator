"""Cache warming implementation.

Pre-populates cache during cold start to improve initial performance.
Supports configurable warming rate and progress tracking.

Example:
    from happysimulator.components.datastore import (
        KVStore, CachedStore, LRUEviction, CacheWarmer
    )

    backing = KVStore(name="db")
    cache = CachedStore(name="cache", backing_store=backing, ...)

    warmer = CacheWarmer(
        name="warmer",
        cache=cache,
        keys_to_warm=["user:1", "user:2", "config:main"],
        warmup_rate=100.0,  # 100 keys/second
    )

    # Start warming as part of simulation
    initial_event = warmer.start_warming()
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


@dataclass(frozen=True)
class CacheWarmerStats:
    """Statistics tracked by CacheWarmer."""

    keys_to_warm: int = 0
    keys_warmed: int = 0
    keys_failed: int = 0  # Keys that couldn't be found
    warmup_time_seconds: float = 0.0


class CacheWarmer(Entity):
    """Pre-populates cache during cold start.

    Fetches specified keys from the backing store and loads them into
    the cache before normal traffic begins. This reduces cache misses
    during the initial warm-up period.

    The warmer processes keys at a configurable rate to avoid
    overwhelming the backing store.

    Attributes:
        name: Entity name for identification.
        progress: Fraction of warming complete (0.0 to 1.0).
        is_complete: Whether warming has finished.
    """

    def __init__(
        self,
        name: str,
        cache: Entity,
        keys_to_warm: list[str] | Callable[[], list[str]],
        warmup_rate: float = 100.0,
        warmup_latency: float = 0.001,
    ):
        """Initialize the cache warmer.

        Args:
            name: Name for this warmer entity.
            cache: The cache to warm (must have a get method).
            keys_to_warm: List of keys or callable that returns keys.
            warmup_rate: Keys to warm per second.
            warmup_latency: Simulated latency per key fetch in seconds.

        Raises:
            ValueError: If parameters are invalid.
        """
        if warmup_rate <= 0:
            raise ValueError(f"warmup_rate must be > 0, got {warmup_rate}")
        if warmup_latency < 0:
            raise ValueError(f"warmup_latency must be >= 0, got {warmup_latency}")

        super().__init__(name)
        self._cache = cache
        self._keys_provider = keys_to_warm
        self._warmup_rate = warmup_rate
        self._warmup_latency = warmup_latency

        # State
        self._keys: list[str] = []
        self._current_index = 0
        self._started = False
        self._completed = False
        self._start_time: Instant | None = None
        self._end_time: Instant | None = None

        # Statistics
        self._keys_to_warm = 0
        self._keys_warmed = 0
        self._keys_failed = 0
        self._warmup_time_seconds = 0.0

    @property
    def stats(self) -> CacheWarmerStats:
        """Frozen snapshot of cache warmer statistics."""
        return CacheWarmerStats(
            keys_to_warm=self._keys_to_warm,
            keys_warmed=self._keys_warmed,
            keys_failed=self._keys_failed,
            warmup_time_seconds=self._warmup_time_seconds,
        )

    @property
    def progress(self) -> float:
        """Fraction of warming complete (0.0 to 1.0)."""
        if not self._keys:
            return 1.0 if self._completed else 0.0
        return self._current_index / len(self._keys)

    @property
    def is_complete(self) -> bool:
        """Whether warming has finished."""
        return self._completed

    @property
    def is_started(self) -> bool:
        """Whether warming has started."""
        return self._started

    @property
    def warmup_rate(self) -> float:
        """Keys warmed per second."""
        return self._warmup_rate

    def get_keys_to_warm(self) -> list[str]:
        """Get the list of keys to warm.

        Resolves the keys provider if it's a callable.

        Returns:
            List of keys to warm.
        """
        if callable(self._keys_provider):
            return self._keys_provider()
        return self._keys_provider

    def start_warming(self) -> Event:
        """Start the cache warming process.

        Returns:
            Initial event to begin warming.
        """
        self._keys = self.get_keys_to_warm()
        self._current_index = 0
        self._started = True
        self._completed = False

        self._keys_to_warm = len(self._keys)
        self._keys_warmed = 0
        self._keys_failed = 0

        # Create initial warming event
        return Event(
            time=Instant.Epoch,  # Will be scheduled at current time
            event_type="cache_warm",
            target=self,
            context={"action": "warm_next"},
        )

    def warm_keys(self, now: Instant) -> Generator[float]:
        """Warm all keys.

        This is a generator that yields delays while warming.

        Args:
            now: Current simulation time.

        Yields:
            Delays between key fetches.
        """
        self._start_time = now
        inter_key_delay = 1.0 / self._warmup_rate

        for key in self._keys:
            # Fetch key into cache
            try:
                gen = self._cache.get(key)
                value = None
                try:
                    while True:
                        delay = next(gen)
                        yield delay
                except StopIteration as e:
                    value = e.value

                if value is not None:
                    self._keys_warmed += 1
                else:
                    self._keys_failed += 1
            except Exception:
                self._keys_failed += 1

            self._current_index += 1

            # Rate limit
            yield inter_key_delay

        self._completed = True
        if self._clock:
            self._end_time = self._clock.now
            if self._start_time:
                self._warmup_time_seconds = (self._end_time - self._start_time).to_seconds()

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Handle warming events.

        Args:
            event: The warming event.

        Yields:
            Delays for key fetches.

        Returns:
            Empty list when warming is complete.
        """
        if event.context.get("action") == "warm_next":
            yield from self.warm_keys(event.time)

        return []
