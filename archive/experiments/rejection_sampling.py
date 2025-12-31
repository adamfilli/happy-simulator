import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. TARGET DISTRIBUTION f(x)
# Let's define a weird bimodal-like distribution (unnormalized for this example)
def target_pdf(x):
    # A mix of exponentials and sine waves, just to be difficult
    return 0.3 * np.exp(-0.2 * x**2) + 0.7 * np.exp(-0.2 * (x - 4)**2)

# 2. PROPOSAL DISTRIBUTION g(x)
# We will use a Normal distribution centered at 2 with a wide variance
# to ensure we cover the target.
prop_mu, prop_sigma = 2, 4
proposal_dist = stats.norm(prop_mu, prop_sigma)

def proposal_pdf(x):
    return proposal_dist.pdf(x)

# 3. DETERMINE SCALING CONSTANT M
# M must be large enough so M * g(x) >= f(x) everywhere.
# In production, you'd find this via optimization. Here, we estimate it visually/safely.
M = 5.5 

def rejection_sampling(n_samples):
    samples = []
    
    while len(samples) < n_samples:
        # Step A: Sample candidate from proposal g(x)
        x_candidate = proposal_dist.rvs()
        
        # Step B: Sample uniform u for the vertical check
        u = np.random.uniform(0, 1)
        
        # Step C: Acceptance Condition
        # We accept if u is under the ratio of Target / (M * Proposal)
        acceptance_prob = target_pdf(x_candidate) / (M * proposal_pdf(x_candidate))
        
        if u < acceptance_prob:
            samples.append(x_candidate)
            
    return np.array(samples)

# Generate and Plot
samples = rejection_sampling(100000)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Generated Samples')

# Plot the theoretical curves for comparison
x_vals = np.linspace(-10, 15, 1000)
plt.plot(x_vals, target_pdf(x_vals)/np.trapz(target_pdf(x_vals), x_vals), 'r--', label='Target PDF (Normalized)')
plt.plot(x_vals, M * proposal_pdf(x_vals), 'g:', label='Envelope M*g(x)')

plt.legend()
plt.title("Rejection Sampling in Action")
plt.show()