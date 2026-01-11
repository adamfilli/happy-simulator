"""Exponentially distributed latency.

ExponentialLatency samples from an exponential distribution with the
configured mean. The exponential distribution models memoryless waiting
times and is commonly used for service times in queuing theory.
"""

import logging
import random

from happysimulator.core.instant import Instant
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


class ExponentialLatency(LatencyDistribution):
    """Latency distribution sampling from an exponential distribution.

    Uses random.expovariate() to generate exponentially distributed
    latencies with the specified mean. The exponential distribution has
    the memoryless property: the remaining wait time has the same
    distribution regardless of time already waited.

    Samples have high variance (coefficient of variation = 1), so values
    can range from near-zero to several multiples of the mean.
    """

    def __init__(self, mean_latency: Instant):
        """Initialize with mean latency (expected value of distribution)."""
        super().__init__(mean_latency)
        self._lambda = 1 / self._mean_latency
        logger.debug("ExponentialLatency created: mean=%.6fs lambda=%.6f", self._mean_latency, self._lambda)

    def get_latency(self, current_time: Instant) -> Instant:
        """Sample a random latency from the exponential distribution."""
        sample = random.expovariate(self._lambda)
        logger.debug("ExponentialLatency sampled: %.6fs (mean=%.6fs)", sample, self._mean_latency)
        return Instant.from_seconds(sample)