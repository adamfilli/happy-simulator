from abc import ABC, abstractmethod
import scipy.integrate as integrate
import scipy.optimize as optimize

from happysimulator.load.profile import Profile
from happysimulator.utils.instant import Instant

class ArrivalTimeProvider(ABC):
    def __init__(self, profile: Profile, start_time: Instant):
        self.profile = profile
        self.current_time = start_time

    @abstractmethod
    def _get_target_integral_value(self) -> float:
        """
        Returns the amount of 'probability mass' (area under the rate curve) 
        to accumulate before the next event occurs.
        """
        pass

    def next_arrival_time(self) -> Instant:
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
            raise RuntimeError(f"Could not find event event with target area {target_area} starting at {t_start_sec}")

        # 6. Root Finding
        result = optimize.root_scalar(objective_func, bracket=[t_low, t_high], method='brentq')
        
        if result.converged:
            self.current_time = Instant.from_seconds(result.root)
            return self.current_time
        else:
            raise RuntimeError("Optimization for next arrival time failed.")