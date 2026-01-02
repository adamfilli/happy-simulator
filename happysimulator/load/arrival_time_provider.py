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
        Subclasses must implement this to define the distribution.
        Returns the amount of 'probability mass' to integrate over.
        """
        pass

    def next_arrival_time(self) -> Instant:
        """
        Calculates the absolute time of the next event.
        """
        # 1. Get the "fuel" (target area) from the specific distribution strategy
        target_area = self._get_target_integral_value()
        
        # 2. Define the integral equation: Integral(t_curr to t_next) = target_area
        def objective_func(t_candidate):
            # integrate.quad returns (value, error)
            area, _ = integrate.quad(self.profile.get_rate, self.current_time, t_candidate)
            return area - target_area

        # 3. Bracket Search (Find a generic range [t_curr, t_high] that contains the answer)
        t_low = self.current_time
        t_high = self.current_time + 1.0  # Initial guess
        
        max_iter = 50
        found_bracket = False
        
        for _ in range(max_iter):
            val = objective_func(t_high)
            if val > 0:
                found_bracket = True
                break
            # Look further ahead (double the window)
            # Handle t=0 edge case by ensuring minimal step
            step = max(0.1, t_high - self.current_time)
            t_high += step * 2.0
            
        if not found_bracket:
            raise RuntimeError("Rate profile is effectively zero; cannot find next event.")

        # 4. Root Finding (Pinpoint exact time)
        result = optimize.root_scalar(objective_func, bracket=[t_low, t_high], method='brentq')
        
        if result.converged:
            self.current_time = result.root
            return self.current_time
        else:
            raise RuntimeError("Optimization for next arrival time failed.")