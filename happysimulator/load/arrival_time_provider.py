"""Base class for computing event arrival times from rate profiles.

Uses numerical integration to find when the accumulated rate (area under
the rate curve) reaches a target value. Subclasses define how that target
is determined:
- ConstantArrivalTimeProvider: target = 1.0 (deterministic spacing)
- PoissonArrivalTimeProvider: target = exponential random (stochastic)

This approach handles non-homogeneous (time-varying) rate profiles.
"""

import logging
from abc import ABC, abstractmethod

from happysimulator.core.temporal import Instant
from happysimulator.load.profile import ConstantRateProfile, Profile
from happysimulator.numerics import brentq, integrate_adaptive_simpson

logger = logging.getLogger(__name__)


class ArrivalTimeProvider(ABC):
    """Computes arrival times by integrating a rate profile.

    Finds the time t such that the integral of the rate from current_time
    to t equals a target value. The target value determines the arrival
    distribution (constant for deterministic, exponential for Poisson).

    Uses scipy's numerical integration and root-finding for accuracy with
    arbitrary rate profiles.

    Attributes:
        profile: Rate function over time.
        current_time: Time of the last arrival (updated after each call).
    """

    def __init__(self, profile: Profile, start_time: Instant):
        self.profile = profile
        self.current_time = start_time
        self._is_constant_rate = isinstance(profile, ConstantRateProfile)
        self._constant_rate: float = profile.rate if self._is_constant_rate else 0.0

    @abstractmethod
    def _get_target_integral_value(self) -> float:
        """Return the target area under the rate curve for the next event.

        - Return 1.0 for deterministic arrivals (constant rate spacing)
        - Return exponential random for Poisson arrivals
        """

    def next_arrival_time(self) -> Instant:
        """Compute the next event arrival time.

        Integrates the rate profile forward until the accumulated area
        reaches the target value from _get_target_integral_value().

        Returns:
            The next arrival time.

        Raises:
            RuntimeError: If the rate is zero indefinitely or optimization fails.
        """
        target_area = self._get_target_integral_value()
        t_start_sec = self.current_time.to_seconds()

        # Fast path for ConstantRateProfile: O(1) direct calculation
        if self._is_constant_rate:
            rate = self._constant_rate
            if rate <= 0:
                raise RuntimeError("Cannot compute arrival with zero or negative rate")
            t_next = t_start_sec + target_area / rate
            self.current_time = Instant.from_seconds(t_next)
            logger.debug(
                "Next arrival computed (fast path): time=%.6f target_area=%.4f", t_next, target_area
            )
            return self.current_time

        # General numerical solution for arbitrary profiles
        def rate_fn(t_seconds: float) -> float:
            return self.profile.get_rate(Instant.from_seconds(t_seconds))

        def objective_func(t_candidate_sec: float) -> float:
            current_area, _ = integrate_adaptive_simpson(
                rate_fn, t_start_sec, t_candidate_sec, tol=1e-10
            )
            return current_area - target_area

        # Smart initial guess based on instantaneous rate
        current_rate = rate_fn(t_start_sec)

        if current_rate > 0:
            # Linear prediction: Time = Area / Rate
            # Multiply by 2.0 to be optimistic and try to bracket immediately
            estimated_delay = (target_area / current_rate) * 2.0
            # Clamp to reasonable bounds
            estimated_delay = max(1e-9, min(estimated_delay, 3600.0))
            t_high = t_start_sec + estimated_delay
        else:
            # Rate is 0? Fallback to probe for future rate increases
            t_high = t_start_sec + 0.1

        # Bracket search with geometric expansion
        t_low = t_start_sec
        max_iter = 50
        found_bracket = False

        for _ in range(max_iter):
            val = objective_func(t_high)

            if val > 0:
                found_bracket = True
                break

            # Geometric expansion: double the window
            step = max(1e-6, t_high - t_low)
            t_high += step * 2.0

        if not found_bracket:
            logger.error(
                "Could not bracket arrival time: target_area=%.4f start=%.4f",
                target_area,
                t_start_sec,
            )
            raise RuntimeError(
                f"Could not find event with target area {target_area} starting at {t_start_sec}"
            )

        # Root finding using Brent's method
        result = brentq(objective_func, t_low, t_high)

        if result.converged:
            self.current_time = Instant.from_seconds(result.root)
            logger.debug(
                "Next arrival computed: time=%.6f target_area=%.4f", result.root, target_area
            )
            return self.current_time
        logger.error("Root-finding failed for arrival time optimization")
        raise RuntimeError("Optimization for next arrival time failed.")
