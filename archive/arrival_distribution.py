from enum import Enum
import random

from happysimulator.utils.instant import Instant


class ArrivalDistribution(Enum):
    CONSTANT = 1
    POISSON = 2

    def get_next_arrival_time(self, current_time: Instant, rate_per_second: float) -> Instant:
        if self == ArrivalDistribution.CONSTANT:
            return current_time + (1.0 / rate_per_second)
        elif self == ArrivalDistribution.POISSON:
            return current_time + Instant.from_seconds(random.expovariate(rate_per_second))
        else:
            raise NotImplementedError("This ArrivalDistribution type not implemented")
        