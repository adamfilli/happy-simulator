from abc import ABC, abstractmethod
import math
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

# --- Type Alias for clarity ---
Instant = float

# --- 1. The Profile Interface (Your definition) ---
class Profile(ABC):
    @abstractmethod
    def get_rate(self, time: Instant) -> float:
        """Returns the rate (events/unit time) at the specific time."""
        pass

# --- 2. The Abstract Base Provider ---
class ArrivalTimeProvider(ABC):
    def __init__(self, profile: Profile):
        self.profile = profile
        self.current_time = 0.0

    @abstractmethod
    def _get_target_integral_value(self) -> float:
        """
        Subclasses must implement this to define the distribution.
        Returns the amount of 'probability mass' to integrate over.
        """
        pass

    def next_arrival_time(self) -> float:
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

# --- 3. The Concrete Implementations ---

class PoissonArrivalTimeProvider(ArrivalTimeProvider):
    """
    Generates events according to a Non-Homogeneous Poisson Process.
    Inter-arrival 'mass' is exponentially distributed.
    """
    def _get_target_integral_value(self) -> float:
        # Generate standard exponential random variable (mean=1)
        # Formula: -ln(U) where U is uniform(0,1)
        return -math.log(1.0 - np.random.random())

class ConstantArrivalTimeProvider(ArrivalTimeProvider):
    """
    Generates deterministic events. 
    The system accumulates exactly 1.0 unit of 'rate' before triggering.
    If rate is constant r=2, events happen every 0.5s exactly.
    """
    def _get_target_integral_value(self) -> float:
        return 1.0

# --- 4. Example Usage ---

# Define a profile: Rate = t (Linear increasing)
class LinearProfile(Profile):
    def get_rate(self, time: Instant) -> float:
        return time if time > 0 else 0.0

# Setup
profile = LinearProfile()
poisson_provider = PoissonArrivalTimeProvider(profile)
constant_provider = ConstantArrivalTimeProvider(profile)

print(f"{'Event':<5} | {'Poisson Time':<15} | {'Constant Time':<15}")
print("-" * 45)

# Reset times for comparison (creating new instances is cleaner usually, 
# but here we just run them side-by-side)
for i in range(100):
    t_p = poisson_provider.next_arrival_time()
    t_c = constant_provider.next_arrival_time()
    
    print(f"{i+1:<5} | {t_p:<15.4f} | {t_c:<15.4f}")