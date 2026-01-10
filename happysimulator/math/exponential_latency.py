"""Exponentially distributed latency.

ExponentialLatency samples from an exponential distribution with the
configured mean. The exponential distribution models memoryless waiting
times and is commonly used for service times in queuing theory.
"""

import random

from happysimulator.math.latency_distribution import LatencyDistribution
from happysimulator.utils.instant import Instant


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

    def get_latency(self, current_time: Instant) -> Instant:
        """Sample a random latency from the exponential distribution."""
        return Instant.from_seconds(random.expovariate(self._lambda))