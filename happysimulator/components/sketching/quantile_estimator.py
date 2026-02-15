"""Quantile estimator entity for latency percentile tracking.

Specialized entity wrapper for T-Digest that provides convenient access
to latency percentiles (p50, p95, p99, p999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.sketching.tdigest import TDigest

if TYPE_CHECKING:
    from collections.abc import Callable

    from happysimulator.core.event import Event


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    """Summary of key latency percentiles."""

    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    p999: float
    min: float | None
    max: float | None
    count: int


class QuantileEstimator(Entity):
    """Entity wrapper for T-Digest quantile/percentile estimation.

    Tracks latency values from events and provides percentile queries.
    Ideal for monitoring latency distributions during simulation.

    Args:
        name: Entity name for logging.
        value_extractor: Function to extract latency value from event.
        compression: T-Digest compression factor (higher = more accuracy).
        seed: Random seed for reproducibility.

    Example:
        # Track response latencies
        estimator = QuantileEstimator(
            name="latency_tracker",
            value_extractor=lambda e: e.context.get("response_time"),
            compression=100,
        )

        # After simulation
        print(f"p99 latency: {estimator.percentile(99)}")
        print(estimator.summary())
    """

    def __init__(
        self,
        name: str,
        value_extractor: Callable[[Event], float | None],
        compression: float = 100.0,
        seed: int | None = None,
    ):
        """Initialize QuantileEstimator.

        Args:
            name: Entity name.
            value_extractor: Function to extract latency value from Event.
            compression: T-Digest compression factor.
            seed: Random seed (passed to TDigest for API consistency).
        """
        super().__init__(name)
        self._tdigest = TDigest(compression=compression, seed=seed)
        self._value_extractor = value_extractor
        self._events_processed = 0

    @property
    def compression(self) -> float:
        """T-Digest compression factor."""
        return self._tdigest.compression

    @property
    def events_processed(self) -> int:
        """Number of events processed."""
        return self._events_processed

    @property
    def sample_count(self) -> int:
        """Number of latency samples collected."""
        return self._tdigest.item_count

    def handle_event(self, event: Event) -> list[Event]:
        """Process an event by updating the T-Digest.

        Args:
            event: Incoming event.

        Returns:
            Empty list (estimator is a sink).
        """
        value = self._value_extractor(event)

        if value is not None:
            self._tdigest.add(value)

        self._events_processed += 1
        return []

    def quantile(self, q: float) -> float:
        """Estimate value at a given quantile.

        Args:
            q: Quantile to estimate (0.0 to 1.0).

        Returns:
            Estimated value at the quantile.
        """
        return self._tdigest.quantile(q)

    def percentile(self, p: float) -> float:
        """Estimate value at a given percentile.

        Args:
            p: Percentile to estimate (0 to 100).

        Returns:
            Estimated value at the percentile.
        """
        return self._tdigest.percentile(p)

    def cdf(self, value: float) -> float:
        """Estimate CDF at a value.

        Args:
            value: Value to get CDF for.

        Returns:
            Estimated probability that a sample <= value.
        """
        return self._tdigest.cdf(value)

    @property
    def min(self) -> float | None:
        """Minimum value observed."""
        return self._tdigest.min

    @property
    def max(self) -> float | None:
        """Maximum value observed."""
        return self._tdigest.max

    def summary(self) -> LatencyPercentiles:
        """Get summary of key percentiles.

        Returns:
            LatencyPercentiles with p50, p75, p90, p95, p99, p999.
        """
        if self._tdigest.item_count == 0:
            return LatencyPercentiles(
                p50=0.0,
                p75=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                p999=0.0,
                min=None,
                max=None,
                count=0,
            )

        return LatencyPercentiles(
            p50=self.percentile(50),
            p75=self.percentile(75),
            p90=self.percentile(90),
            p95=self.percentile(95),
            p99=self.percentile(99),
            p999=self.percentile(99.9),
            min=self._tdigest.min,
            max=self._tdigest.max,
            count=self._tdigest.item_count,
        )

    def clear(self) -> None:
        """Clear the T-Digest and reset counters."""
        self._tdigest.clear()
        self._events_processed = 0
