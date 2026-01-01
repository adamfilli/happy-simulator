import random

from archive.latency_distribution import LatencyDistribution
from happysimulator.utils.instant import Instant


class NormalLatency(LatencyDistribution):
    def __init__(self, mean_latency: Instant, std_dev: Instant):
        super().__init__(mean_latency)
        self._std_dev = std_dev.to_seconds()

    def get_latency(self, current_time: Instant) -> Instant:
        generated_latency_seconds = random.gauss(self._mean_latency, self._std_dev)
        return Instant.from_seconds(max(0.0, generated_latency_seconds))