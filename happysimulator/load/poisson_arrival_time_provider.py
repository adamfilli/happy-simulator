import math
import numpy as np

from happysimulator.load.arrival_time_provider import ArrivalTimeProvider

class PoissonArrivalTimeProvider(ArrivalTimeProvider):
    """
    Generates events according to a Non-Homogeneous Poisson Process.
    Inter-arrival 'mass' is exponentially distributed.
    """
    def _get_target_integral_value(self) -> float:
        # Generate standard exponential random variable (mean=1)
        # Formula: -ln(U) where U is uniform(0,1)
        return -math.log(1.0 - np.random.random())