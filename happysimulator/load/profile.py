"""Interface for time-varying rate functions.

Profiles define how a rate (e.g., request rate, failure rate) varies over
simulation time. Used by ArrivalTimeProviders to compute event spacing and
by other components to model time-dependent behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from happysimulator.core.instant import Instant


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


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Profile that returns a constant rate regardless of time.

    Args:
        rate: The constant rate value (e.g., requests per second).
    """

    rate: float

    def get_rate(self, time: Instant) -> float:
        return self.rate


@dataclass(frozen=True)
class LinearRampProfile(Profile):
    """Load profile that ramps linearly from start_rate to end_rate.

    Args:
        duration_s: Time over which to ramp.
        start_rate: Initial rate (e.g., requests per second).
        end_rate: Final rate (e.g., requests per second).
    """

    duration_s: float
    start_rate: float
    end_rate: float

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        if t <= 0:
            return self.start_rate
        if t >= self.duration_s:
            return self.end_rate

        # Linear interpolation
        fraction = t / self.duration_s
        return self.start_rate + fraction * (self.end_rate - self.start_rate)
