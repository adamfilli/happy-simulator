import random

from archive.latency_distribution import LatencyDistribution
from happysimulator.utils.instant import Instant


class ExponentialLatency(LatencyDistribution):
    def __init__(self, mean_latency: Instant):
        super().__init__(mean_latency)
        self._lambda = 1 / self._mean_latency

    def get_latency(self, current_time: Instant) -> Instant:
        return Instant.from_seconds(random.expovariate(self._lambda))