import math
import random
import collections

# The problem with this one is the inverse transform is solved manually,
# see inverse_transform_automatic for the general solution

class VariableRateSource:
    def __init__(self):
        self.current_time = 0.0
        
    def get_next_arrival_time(self):
        E = -math.log(1.0 - random.random())
        # Inverse transform for rate(t) = t -> Integral is 0.5*t^2
        term_inside = (self.current_time ** 2) + (2.0 * E)
        next_time = math.sqrt(term_inside)
        self.current_time = next_time
        return next_time

# --- Analysis Setup ---

source = VariableRateSource()
events = []

# We use a deque to store the last N timestamps (Rolling Window)
window_size = 5
window = collections.deque(maxlen=window_size)

print(f"{'Event':<5} | {'Time':<8} | {'Theor. Rate':<12} | {'Observed Rate (Roll 5)':<22} | {'Gap'}")
print("-" * 75)

# Increased range to 20 to see the "rolling" stats stabilize
for i in range(20):
    t = source.get_next_arrival_time()
    events.append(t)
    window.append(t)
    
    # 1. Theoretical Instantaneous Rate is simply 't'
    theoretical_rate = t
    
    # 2. Empirical Rolling Rate
    # Calculated as: (Events in Window) / (Time Duration of Window)
    if len(window) == window_size:
        # Time taken for the last 5 events = (Current Time) - (Time of oldest event in window)
        time_delta = window[-1] - window[0]
        # We have (window_size - 1) intervals between these 5 points
        count = window_size - 1
        observed_rate = count / time_delta
        obs_str = f"{observed_rate:.4f}"
    else:
        observed_rate = 0.0
        obs_str = "..." # Not enough data yet

    # Gap for visual reference
    gap = events[-1] - events[-2] if len(events) > 1 else 0

    print(f"{i+1:<5} | {t:<8.4f} | {theoretical_rate:<12.4f} | {obs_str:<22} | {gap:.4f}")