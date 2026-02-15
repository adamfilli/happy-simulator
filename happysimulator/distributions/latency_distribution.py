"""Base class for latency distributions.

LatencyDistribution provides an interface for sampling latency values
in simulation. Implementations can be constant (deterministic) or
follow statistical distributions (exponential, normal, etc.).

Latency distributions support arithmetic operators for adjusting
the mean, returning modified copies.
"""

import copy
from abc import ABC, abstractmethod

from happysimulator.core.temporal import Duration, Instant


class LatencyDistribution(ABC):
    """Abstract base class for latency sampling.

    Subclasses implement get_latency() to return sampled delay values.
    The mean_latency parameter configures the expected average latency;
    actual samples may vary based on the distribution type.

    Supports +/- operators to adjust the mean latency, returning new
    instances (original is unchanged).

    Attributes:
        _mean_latency: Mean latency in seconds (stored as float for calculations).
    """

    def __init__(self, mean_latency: Duration | float):
        """Initialize with a mean latency value.

        Args:
            mean_latency: Expected mean latency as Duration or seconds (float).
        """
        if isinstance(mean_latency, Duration):
            self._mean_latency = mean_latency.to_seconds()
        else:
            self._mean_latency = float(mean_latency)

    @abstractmethod
    def get_latency(self, current_time: Instant) -> Duration:
        """Sample a latency value.

        Args:
            current_time: Current simulation time (for time-varying distributions).

        Returns:
            Sampled latency as a Duration.
        """

    def __add__(self, additional: float) -> "LatencyDistribution":
        """Return a copy with increased mean latency."""
        new_instance = copy.deepcopy(self)
        new_instance._mean_latency += additional
        return new_instance

    def __sub__(self, subtraction: float) -> "LatencyDistribution":
        """Return a copy with decreased mean latency."""
        new_instance = copy.deepcopy(self)
        new_instance._mean_latency -= subtraction
        return new_instance
