"""Poisson process arrival time provider for stochastic event generation.

Events arrive according to a non-homogeneous Poisson process. The target
integral value follows an exponential distribution (mean=1), which produces
the characteristic memoryless inter-arrival times of a Poisson process.

For a constant rate r, expected inter-arrival time is 1/r seconds, but
actual intervals vary randomly according to the exponential distribution.
"""

import math
import numpy as np

from happysimulator.load.arrival_time_provider import ArrivalTimeProvider


class PoissonArrivalTimeProvider(ArrivalTimeProvider):
    """Stochastic event arrival following a Poisson process.

    Uses exponentially distributed target integral values. Combined with
    a rate profile, this produces a non-homogeneous Poisson process where
    the instantaneous rate can vary over time.

    Use this for realistic load modeling where arrival times have natural
    variability (e.g., user requests, network traffic).
    """

    def _get_target_integral_value(self) -> float:
        """Return exponential random variable for Poisson inter-arrivals."""
        return -math.log(1.0 - np.random.random())