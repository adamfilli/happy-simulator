"""Interface for time-varying rate functions.

Profiles define how a rate (e.g., request rate, failure rate) varies over
simulation time. Used by ArrivalTimeProviders to compute event spacing and
by other components to model time-dependent behavior.
"""

from abc import ABC, abstractmethod

from ..utils.instant import Instant


class Profile(ABC):
    """Abstract base class for time-varying rate functions.

    Implement get_rate() to define how the rate changes over time.
    Common patterns:
    - Constant rate: return same value regardless of time
    - Step function: return different values for different time ranges
    - Ramp: linearly interpolate between rates
    - Periodic: model daily/hourly traffic patterns
    """

    @abstractmethod
    def get_rate(self, time: Instant) -> float:
        """Return the rate at the given simulation time.

        Args:
            time: The simulation time to query.

        Returns:
            The rate value (interpretation depends on usage context).
        """
        pass