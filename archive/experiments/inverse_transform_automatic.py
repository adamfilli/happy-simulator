import math
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

class UniversalSource:
    def __init__(self, rate_function):
        self.rate_function = rate_function
        self.current_time = 0.0
        
    def get_next_arrival_time(self):
        """
        Numerically solves for the next arrival time for ANY rate function.
        Solves: Integral(rate_func, t_curr, t_next) = E
        """
        # 1. Generate the required "probability mass" (Fuel)
        E = -math.log(1.0 - np.random.random())
        
        # 2. Define the objective function for the solver.
        # We want to find 't' such that: Integral(t_curr, t) - E = 0
        def objective_func(t_candidate):
            # Calculate area under curve from current_time to t_candidate
            # integrate.quad returns (value, error), we need [0]
            area, _ = integrate.quad(self.rate_function, self.current_time, t_candidate)
            return area - E

        # 3. Bracket Search
        # We need to find a time interval [a, b] where the solution exists.
        # 'a' is obviously current_time (where area = 0, so result is -E).
        # We need to find 'b' where area > E.
        t_low = self.current_time
        t_high = self.current_time + 1.0 # Initial guess: 1 second later
        
        # Expand the window until we capture enough area
        # Safety: limit iterations to prevent infinite loops if rate is 0 forever
        max_iter = 100 
        for _ in range(max_iter):
            val = objective_func(t_high)
            if val > 0: # We found a point where Area > E
                break
            # If not enough area yet, double the window and try again
            t_high = self.current_time + (t_high - self.current_time) * 2
        else:
            raise RuntimeError("Rate function is effectively zero; cannot find next event.")

        # 4. Root Finding (The Numerical Inversion)
        # We use Brent's method, which is fast and robust for this type of function
        result = optimize.root_scalar(objective_func, bracket=[t_low, t_high], method='brentq')
        
        if result.converged:
            self.current_time = result.root
            return result.root
        else:
            raise RuntimeError("Optimization failed to converge")

# --- TEST CASE 1: The Stepwise Function ---
# Let's define the complex stepwise function you asked for earlier:
# 0-3s: Rate 0
# 3-5s: Rate 10
# 5-8s: Rate 0
# >8s : Rate 5 (added this so simulation continues)
def my_stepwise_rate(t):
    if 0 <= t < 3:
        return 0.0
    elif 3 <= t < 5:
        return 10.0
    elif 5 <= t < 8:
        return 0.0
    else:
        return 5.0

print("--- Simulating Stepwise Function ---")
source_step = UniversalSource(my_stepwise_rate)
events = []
for i in range(10):
    t = source_step.get_next_arrival_time()
    events.append(t)
    print(f"Event {i+1}: {t:.4f}")

# Verify the gap logic
# We expect first event > 3.0 (skipping the first 0-rate zone)
# We expect a gap between 5.0 and 8.0 where no events happen

print("\n--- TEST CASE 2: The Linear Function (3t) ---")
# Verifying it matches our previous manual math
source_linear = UniversalSource(lambda t: 3*t)
for i in range(5):
    t = source_linear.get_next_arrival_time()
    print(f"Event {i+1}: {t:.4f} (Rate is {3*t:.2f})")