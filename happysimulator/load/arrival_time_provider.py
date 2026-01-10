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
import scipy.integrate as integrate
import scipy.optimize as optimize

from happysimulator.load.profile import Profile
from happysimulator.utils.instant import Instant

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

    @abstractmethod
    def _get_target_integral_value(self) -> float:
        """Return the target area under the rate curve for the next event.

        - Return 1.0 for deterministic arrivals (constant rate spacing)
        - Return exponential random for Poisson arrivals
        """
        pass

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
        
        # 1. Bridge to floats
        t_start_sec = self.current_time.to_seconds()

        # 2. Wrapper for Scipy
        def rate_fn_for_scipy(t_seconds: float) -> float:
            return self.profile.get_rate(Instant.from_seconds(t_seconds))

        # 3. Integral Equation
        def objective_func(t_candidate_sec: float) -> float:
            # Note: limit=50 improves performance for simple profiles by restricting subdivision depth
            current_area, _ = integrate.quad(rate_fn_for_scipy, t_start_sec, t_candidate_sec, limit=50)
            return current_area - target_area

        # 4. Smart Initial Guess (The Fix)
        # Get the instantaneous rate to predict where the target area might be reached.
        current_rate = rate_fn_for_scipy(t_start_sec)
        
        if current_rate > 0:
            # Linear prediction: Time = Area / Rate
            # We multiply by 2.0 to be "optimistic" and try to bracket it immediately.
            estimated_delay = (target_area / current_rate) * 2.0
            
            # Clamp: Don't let the guess be too microscopic (e.g. < 1ns) or Scipy might underflow
            # Don't let it be too huge (e.g. > 1 hour) if rate is tiny
            estimated_delay = max(1e-9, min(estimated_delay, 3600.0))
            
            t_high = t_start_sec + estimated_delay
        else:
            # Rate is 0? Fallback to a small default step to probe for future rate increases.
            t_high = t_start_sec + 0.1

        # 5. Bracket Search (Expansion)
        t_low = t_start_sec
        max_iter = 50
        found_bracket = False
        
        for _ in range(max_iter):
            val = objective_func(t_high)
            
            if val > 0:
                found_bracket = True
                break
            
            # Geometric Expansion: Double the window if we haven't found the event yet.
            # (e.g. Rate dropped unexpectedly)
            step = max(1e-6, t_high - t_low)
            t_high += step * 2.0
            
        if not found_bracket:
            logger.error(
                "Could not bracket arrival time: target_area=%.4f start=%.4f",
                target_area, t_start_sec
            )
            raise RuntimeError(f"Could not find event event with target area {target_area} starting at {t_start_sec}")

        # 6. Root Finding
        result = optimize.root_scalar(objective_func, bracket=[t_low, t_high], method='brentq')

        if result.converged:
            self.current_time = Instant.from_seconds(result.root)
            logger.debug(
                "Next arrival computed: time=%.6f target_area=%.4f",
                result.root, target_area
            )
            return self.current_time
        else:
            logger.error("Root-finding failed for arrival time optimization")
            raise RuntimeError("Optimization for next arrival time failed.")