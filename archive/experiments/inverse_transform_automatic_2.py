import math
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# --- 1. The Universal Source (Numerical Inverse Transform) ---
class UniversalSource:
    def __init__(self, rate_function):
        self.rate_function = rate_function
        self.current_time = 0.0
        
    def get_next_arrival_time(self):
        E = -math.log(1.0 - np.random.random())
        
        def objective_func(t_candidate):
            area, _ = integrate.quad(self.rate_function, self.current_time, t_candidate)
            return area - E

        # Bracket Search
        t_low = self.current_time
        t_high = self.current_time + 0.5 
        
        # Expand window
        max_iter = 100 
        for _ in range(max_iter):
            val = objective_func(t_high)
            if val > 0:
                break
            t_high = self.current_time + (t_high - self.current_time) * 2
        else:
            return None # Should handle gracefully if rate stays 0 forever

        # Root Finding
        try:
            result = optimize.root_scalar(objective_func, bracket=[t_low, t_high], method='brentq')
        except ValueError:
            return None # Handle edge cases
            
        if result.converged:
            self.current_time = result.root
            return result.root
        return None

# --- 2. The Rate Function Definition ---
def stepwise_rate(t):
    # 0-3s: Rate 0
    # 3-5s: Rate 10
    # 5-8s: Rate 0
    # 8-10s: Rate 5
    if 0 <= t < 3:
        return 0.0
    elif 3 <= t < 5:
        return 10.0
    elif 5 <= t < 8:
        return 0.0
    elif 8 <= t <= 10:
        return 5.0
    else:
        return 0.0

# --- 3. Simulation Loop ---
# We run multiple "trials" to get statistically significant data for the plot
num_trials = 10
all_events = []

print(f"Running {num_trials} simulations over 0-10s range...")

for _ in range(num_trials):
    source = UniversalSource(stepwise_rate)
    while True:
        t = source.get_next_arrival_time()
        if t is None or t > 10:
            break
        all_events.append(t)

# --- 4. Analysis & Plotting ---

# Create bins: 0, 1, 2, ... 10
bins = np.arange(0, 11, 1)

# Calculate Empirical Rate
# Histogram counts = number of events in that bin across ALL trials
# To get Rate (events/sec), we divide by (number of trials * bin width)
counts, bin_edges = np.histogram(all_events, bins=bins)
bin_width = bins[1] - bins[0]
empirical_rate = counts / (num_trials * bin_width)

# Prepare Theoretical Curve for Plotting
t_vals = np.linspace(0, 10, 1000)
rate_vals = [stepwise_rate(t) for t in t_vals]

# Plot
plt.figure(figsize=(12, 6))

# Plot Theoretical Rate (Line)
plt.plot(t_vals, rate_vals, 'r-', linewidth=2, label='Theoretical Rate (Input Function)')

# Plot Empirical Rate (Bars)
# We use step plot or bar chart to represent buckets
plt.bar(bins[:-1], empirical_rate, width=bin_width, align='edge', alpha=0.4, color='blue', edgecolor='black', label='Empirical Observed Rate')

plt.title('Verification: Theoretical Input vs. Simulated Output')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Rate (Events / sec)')
plt.xticks(bins)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# Print detailed comparison
print("\n--- Bucket Analysis ---")
print(f"{'Time Bucket':<15} | {'Count':>8} | {'Empirical Rate':<15} | {'Theoretical Rate':<18} | {'Expected Count':>14} | {'Diff':>18}")
print("-" * 115)
for i in range(len(bins)-1):
    # Theoretical rate is tricky for buckets that cross transitions (like 3-4s is mixed if we had finer buckets)
    # But here our buckets align with transitions for simplicity (except we need to look at the midpoint)
    midpoint = (bins[i] + bins[i+1]) / 2
    theo = stepwise_rate(midpoint)
    emp_rate = empirical_rate[i]
    count = int(counts[i])
    expected_count = theo * bin_width * num_trials
    diff = count - expected_count
    if expected_count != 0:
        diff_pct = (diff / expected_count) * 100.0
        diff_str = f"{diff:+.1f} ({diff_pct:+.1f}%)"
    else:
        diff_str = f"{diff:+.1f} (n/a)"
    print(f"{bins[i]:.1f}s - {bins[i+1]:.1f}s | {count:8d} | {emp_rate:<15.4f} | {theo:<18.2f} | {expected_count:14.1f} | {diff_str:>18}")