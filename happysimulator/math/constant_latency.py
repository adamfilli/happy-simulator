from happysimulator.math.latency_distribution import LatencyDistribution
from happysimulator.utils.instant import Instant


class ConstantLatency(LatencyDistribution):
    def __init__(self, latency: Instant):
        super().__init__(latency)

    def get_latency(self, current_time: Instant) -> Instant:
        return Instant.from_seconds(self._mean_latency)
