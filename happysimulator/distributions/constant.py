"""Deterministic latency distribution.

ConstantLatency always returns the same latency value. Use for
predictable, reproducible scenarios or as a baseline for testing.
"""

import logging

from happysimulator.core.temporal import Duration, Instant
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


class ConstantLatency(LatencyDistribution):
    """Latency distribution that always returns the same value.

    Every call to get_latency() returns the configured latency exactly.
    No randomness is involved.

    Use for deterministic tests or when modeling fixed processing delays.
    """

    def __init__(self, latency: Duration | float):
        """Initialize with a fixed latency value.

        Args:
            latency: The constant latency as Duration or seconds (float).
        """
        super().__init__(latency)
        logger.debug("ConstantLatency created: latency=%.6fs", self._mean_latency)

    def get_latency(self, current_time: Instant) -> Duration:
        """Return the constant latency value."""
        return Duration.from_seconds(self._mean_latency)
