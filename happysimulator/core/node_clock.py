"""Per-node clocks with skew and drift models.

In real distributed systems, every node has its own clock with skew and drift.
NodeClock transforms true simulation time into perceived local time, enabling
modeling of clock-sensitive protocols (leader election, lease expiry, cache TTLs,
distributed tracing).

Key insight: NodeClock is a *view layer* over the shared Clock. Events are still
ordered by true simulation time (the global Clock). NodeClock only transforms the
read side — what an entity perceives as "now". This avoids causality issues.

Usage::

    from happysimulator import NodeClock, FixedSkew, LinearDrift, Duration

    # Fixed offset: node clock is 50ms ahead of true time
    clock = NodeClock(FixedSkew(Duration.from_seconds(0.05)))

    # Drifting clock: 1000 ppm = 1ms drift per second
    clock = NodeClock(LinearDrift(rate_ppm=1000))

    # In an entity:
    class RaftNode(Entity):
        def __init__(self, name, node_clock):
            super().__init__(name)
            self._node_clock = node_clock

        def set_clock(self, clock):
            super().set_clock(clock)
            self._node_clock.set_clock(clock)

        @property
        def local_now(self):
            return self._node_clock.now
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from happysimulator.core.clock import Clock
from happysimulator.core.temporal import Duration, Instant


@runtime_checkable
class ClockModel(Protocol):
    """Protocol for time transformation models.

    A ClockModel transforms true simulation time into perceived local time.
    Implementations define how a node's clock deviates from the global clock.
    """

    def read(self, true_time: Instant) -> Instant:
        """Transform true simulation time into perceived local time.

        Args:
            true_time: The actual simulation time from the global clock.

        Returns:
            The time as perceived by the node.
        """
        ...


class FixedSkew:
    """Constant time offset — clock is always ahead or behind by a fixed amount.

    A positive offset means the clock reads ahead of true time (fast clock).
    A negative offset means the clock reads behind true time (slow clock).

    Args:
        offset: The constant offset to apply. Positive = ahead, negative = behind.
    """

    def __init__(self, offset: Duration):
        self._offset = offset

    def read(self, true_time: Instant) -> Instant:
        """Return true_time shifted by the fixed offset."""
        return true_time + self._offset

    @property
    def offset(self) -> Duration:
        """The fixed offset applied by this model."""
        return self._offset


class LinearDrift:
    """Clock that runs faster or slower than true time, accumulating drift.

    Drift is specified in parts per million (ppm). A rate of 1000 ppm means
    the clock gains 1ms per second of elapsed true time. Negative ppm means
    the clock runs slow.

    The drift accumulates linearly: at true time T seconds from epoch,
    the perceived time is T + (T * rate_ppm / 1_000_000) seconds.

    Args:
        rate_ppm: Drift rate in parts per million. Positive = fast, negative = slow.
    """

    def __init__(self, rate_ppm: float):
        self._rate_ppm = rate_ppm

    def read(self, true_time: Instant) -> Instant:
        """Return true_time with accumulated linear drift."""
        elapsed_ns = true_time.nanoseconds
        drift_ns = int(elapsed_ns * self._rate_ppm / 1_000_000)
        return Instant(elapsed_ns + drift_ns)

    @property
    def rate_ppm(self) -> float:
        """The drift rate in parts per million."""
        return self._rate_ppm


class NodeClock:
    """Per-node clock that transforms true simulation time via a ClockModel.

    NodeClock wraps a base Clock (the shared simulation clock) and applies
    a ClockModel to transform what the node perceives as "now". The base
    clock is injected via set_clock(), typically forwarded from Entity.set_clock().

    This is a plain class, NOT an Entity. Entities hold a NodeClock reference
    and forward clock injection to it.

    Args:
        model: The clock model defining how time is transformed.
            If None, the node clock returns true time (identity).
    """

    def __init__(self, model: ClockModel | None = None):
        self._model = model
        self._clock: Clock | None = None

    def set_clock(self, clock: Clock) -> None:
        """Inject the base simulation clock.

        Args:
            clock: The shared simulation clock to read true time from.
        """
        self._clock = clock

    @property
    def now(self) -> Instant:
        """The perceived local time, transformed by the clock model.

        Returns true time if no model is set.

        Raises:
            RuntimeError: If accessed before clock injection.
        """
        if self._clock is None:
            raise RuntimeError("NodeClock has no base clock — call set_clock() first.")
        true_time = self._clock.now
        if self._model is None:
            return true_time
        return self._model.read(true_time)

    @property
    def model(self) -> ClockModel | None:
        """The clock model, or None for identity."""
        return self._model
