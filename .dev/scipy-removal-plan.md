# Scipy Removal Development Plan

## Overview

This document outlines the plan to remove the `scipy` dependency from happy-simulator. Currently, scipy is only used in `happysimulator/load/arrival_time_provider.py` for two functions:

1. **`scipy.integrate.quad`** - Numerical integration of rate functions
2. **`scipy.optimize.root_scalar`** - Root finding using Brent's method

The goal is to replace these with pure Python/numpy implementations that maintain the same accuracy and reliability.

## Current Usage Analysis

### Location
`happysimulator/load/arrival_time_provider.py` lines 14-15, 75, 121

### What It Does
The `ArrivalTimeProvider` computes arrival times by solving this equation:

```
∫[t_start → t_arrival] rate(t) dt = target_area
```

Where:
- `rate(t)` is the rate profile function (e.g., constant, ramp, spike)
- `target_area` is 1.0 for deterministic arrivals, or exponential random for Poisson

This requires:
1. **Integration**: Computing `∫ rate(t) dt` from `t_start` to `t_candidate`
2. **Root finding**: Finding `t_arrival` where the integral equals `target_area`

### Current Implementation Flow
1. Define `objective_func(t) = integral(rate, t_start, t) - target_area`
2. Bracket the root using geometric expansion
3. Use Brent's method to find where `objective_func(t) = 0`

---

## Implementation Strategy

### Option 1: Analytical Solutions for Common Profiles (Recommended)

For the most common profile types, we can compute closed-form solutions that are faster and more accurate than numerical methods:

| Profile Type | Integral Formula | Root Formula |
|-------------|------------------|--------------|
| `ConstantRateProfile` | `rate × Δt` | `t_start + target_area / rate` |
| `LinearRampProfile` | Trapezoidal area | Quadratic formula |
| `SpikeProfile` | Piecewise constant | Piecewise linear solve |

**For arbitrary profiles**, fall back to numerical methods.

### Option 2: Numerical Implementation (Fallback)

Implement:
1. **Adaptive Simpson's Rule** for integration
2. **Brent's Method** for root finding

---

## Detailed Design

### Phase 1: Create Integration Module

**File**: `happysimulator/numerics/integration.py`

```python
"""Numerical integration methods for rate profiles."""

from typing import Callable

def integrate_adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> tuple[float, float]:
    """Adaptive Simpson's rule integration.

    Uses recursive subdivision to achieve desired tolerance.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        tol: Absolute error tolerance
        max_depth: Maximum recursion depth

    Returns:
        Tuple of (integral_value, error_estimate)
    """
    ...
```

**Algorithm**: Adaptive Simpson's rule
- Compute Simpson's rule on [a, b]: `S1 = (b-a)/6 * (f(a) + 4*f(m) + f(b))`
- Compute on [a, m] and [m, b]: `S2 = S_left + S_right`
- If `|S2 - S1| < 15 * tol`: return `S2 + (S2 - S1)/15` (Richardson extrapolation)
- Else: recurse on each half with `tol/2`

### Phase 2: Create Root Finding Module

**File**: `happysimulator/numerics/root_finding.py`

```python
"""Root finding algorithms."""

from dataclasses import dataclass
from typing import Callable

@dataclass
class RootResult:
    """Result of root finding."""
    root: float
    converged: bool
    iterations: int
    function_calls: int

def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-12,
    rtol: float = 4 * 2.220446049250313e-16,  # 4 * machine epsilon
    maxiter: int = 100,
) -> RootResult:
    """Find root of f in bracket [a, b] using Brent's method.

    Combines bisection, secant, and inverse quadratic interpolation.
    Guaranteed to converge if f(a) and f(b) have opposite signs.

    Args:
        f: Continuous function
        a, b: Bracket with f(a) * f(b) < 0
        xtol: Absolute tolerance
        rtol: Relative tolerance
        maxiter: Maximum iterations

    Returns:
        RootResult with the root and convergence info

    Raises:
        ValueError: If f(a) and f(b) have the same sign
    """
    ...
```

**Algorithm**: Brent's method
1. Start with bracket [a, b] where f(a) * f(b) < 0
2. Each iteration, choose between:
   - Inverse quadratic interpolation (if 3 distinct points)
   - Secant method (linear interpolation)
   - Bisection (fallback)
3. Accept step only if it stays within bracket and converges faster than bisection
4. Stop when |b - a| < tol

### Phase 3: Profile-Specific Optimizations

**File**: `happysimulator/load/arrival_time_provider.py` (modified)

Add fast paths for common profiles:

```python
def _solve_constant_rate(self, target_area: float, t_start: float, rate: float) -> float:
    """O(1) solution for constant rate: t = t_start + target_area / rate"""
    if rate <= 0:
        raise RuntimeError("Cannot compute arrival with zero or negative rate")
    return t_start + target_area / rate

def _solve_linear_ramp(
    self,
    target_area: float,
    t_start: float,
    profile: LinearRampProfile
) -> float:
    """O(1) solution for linear ramp using quadratic formula.

    For rate(t) = r0 + (r1 - r0) * t / T:
    ∫[0→t] rate(s) ds = r0*t + (r1-r0)*t²/(2T) = target_area

    Solve: (r1-r0)/(2T) * t² + r0 * t - target_area = 0
    """
    ...
```

### Phase 4: Refactor ArrivalTimeProvider

**File**: `happysimulator/load/arrival_time_provider.py`

```python
from happysimulator.numerics.integration import integrate_adaptive_simpson
from happysimulator.numerics.root_finding import brentq
from happysimulator.load.profile import ConstantRateProfile, LinearRampProfile, SpikeProfile

class ArrivalTimeProvider(ABC):
    def next_arrival_time(self) -> Instant:
        target_area = self._get_target_integral_value()
        t_start_sec = self.current_time.to_seconds()

        # Fast path for common profiles
        if isinstance(self.profile, ConstantRateProfile):
            t_next = self._solve_constant_rate(target_area, t_start_sec, self.profile.rate)
            self.current_time = Instant.from_seconds(t_next)
            return self.current_time

        if isinstance(self.profile, LinearRampProfile):
            t_next = self._solve_linear_ramp(target_area, t_start_sec, self.profile)
            self.current_time = Instant.from_seconds(t_next)
            return self.current_time

        # Fallback to numerical solution for arbitrary profiles
        return self._solve_numerical(target_area, t_start_sec)

    def _solve_numerical(self, target_area: float, t_start_sec: float) -> Instant:
        """Numerical solution using adaptive integration and Brent's method."""
        def rate_fn(t: float) -> float:
            return self.profile.get_rate(Instant.from_seconds(t))

        def objective(t_candidate: float) -> float:
            integral, _ = integrate_adaptive_simpson(rate_fn, t_start_sec, t_candidate)
            return integral - target_area

        # Bracket finding (same logic as current)
        t_low = t_start_sec
        t_high = self._find_upper_bracket(t_start_sec, target_area, objective)

        # Root finding
        result = brentq(objective, t_low, t_high)

        if result.converged:
            self.current_time = Instant.from_seconds(result.root)
            return self.current_time
        else:
            raise RuntimeError("Root finding failed to converge")
```

---

## Testing Strategy

### Unit Tests for Numerical Methods

**File**: `tests/unit/numerics/test_integration.py`

```python
class TestAdaptiveSimpson:
    def test_constant_function(self):
        """∫[0→1] 5 dx = 5"""
        result, _ = integrate_adaptive_simpson(lambda x: 5.0, 0, 1)
        assert abs(result - 5.0) < 1e-10

    def test_linear_function(self):
        """∫[0→1] x dx = 0.5"""
        result, _ = integrate_adaptive_simpson(lambda x: x, 0, 1)
        assert abs(result - 0.5) < 1e-10

    def test_quadratic_function(self):
        """∫[0→1] x² dx = 1/3"""
        result, _ = integrate_adaptive_simpson(lambda x: x**2, 0, 1)
        assert abs(result - 1/3) < 1e-10

    def test_sine_function(self):
        """∫[0→π] sin(x) dx = 2"""
        import math
        result, _ = integrate_adaptive_simpson(math.sin, 0, math.pi)
        assert abs(result - 2.0) < 1e-10

    def test_exponential_function(self):
        """∫[0→1] e^x dx = e - 1"""
        import math
        result, _ = integrate_adaptive_simpson(math.exp, 0, 1)
        assert abs(result - (math.e - 1)) < 1e-10

    def test_discontinuous_function(self):
        """Step function: ∫[0→2] step(x-1) dx = 1"""
        def step(x):
            return 1.0 if x >= 1.0 else 0.0
        result, _ = integrate_adaptive_simpson(step, 0, 2, tol=1e-6)
        assert abs(result - 1.0) < 1e-4  # Lower tolerance for discontinuity

    def test_matches_scipy(self):
        """Verify results match scipy.integrate.quad for various functions."""
        import scipy.integrate as scipy_int

        test_cases = [
            (lambda x: x**3, 0, 2),
            (lambda x: 1/(1 + x**2), 0, 10),
            (lambda x: math.exp(-x**2), -2, 2),
        ]

        for f, a, b in test_cases:
            our_result, _ = integrate_adaptive_simpson(f, a, b)
            scipy_result, _ = scipy_int.quad(f, a, b)
            assert abs(our_result - scipy_result) < 1e-8
```

**File**: `tests/unit/numerics/test_root_finding.py`

```python
class TestBrentq:
    def test_linear_root(self):
        """f(x) = x - 2, root at x = 2"""
        result = brentq(lambda x: x - 2, 0, 5)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-12

    def test_quadratic_root(self):
        """f(x) = x² - 4, root at x = 2"""
        result = brentq(lambda x: x**2 - 4, 1, 3)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-12

    def test_transcendental_root(self):
        """f(x) = cos(x) - x, root near 0.739"""
        import math
        result = brentq(lambda x: math.cos(x) - x, 0, 1)
        assert result.converged
        assert abs(result.root - 0.7390851332151607) < 1e-10

    def test_near_boundary(self):
        """Root very close to bracket boundary."""
        result = brentq(lambda x: x - 0.001, 0, 1)
        assert result.converged
        assert abs(result.root - 0.001) < 1e-12

    def test_invalid_bracket_raises(self):
        """Should raise when f(a) and f(b) have same sign."""
        with pytest.raises(ValueError):
            brentq(lambda x: x**2 + 1, -1, 1)  # No real roots

    def test_matches_scipy(self):
        """Verify results match scipy.optimize.brentq."""
        import scipy.optimize as scipy_opt

        test_cases = [
            (lambda x: x**3 - 2*x - 5, 2, 3),
            (lambda x: math.exp(x) - 10, 2, 3),
            (lambda x: x * math.log(x) - 1, 1, 3),
        ]

        for f, a, b in test_cases:
            our_result = brentq(f, a, b)
            scipy_result = scipy_opt.brentq(f, a, b)
            assert abs(our_result.root - scipy_result) < 1e-10
```

### Arrival Time Provider Tests

**File**: `tests/unit/load/test_arrival_time_provider.py`

```python
class TestArrivalTimeProviderNoScipy:
    """Tests verifying arrival time calculations match scipy-based implementation."""

    def test_constant_rate_deterministic(self):
        """With rate=10/s and target=1.0, events arrive every 0.1s."""
        profile = ConstantRateProfile(rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        times = [provider.next_arrival_time().to_seconds() for _ in range(10)]

        expected = [0.1 * (i + 1) for i in range(10)]
        for actual, exp in zip(times, expected):
            assert abs(actual - exp) < 1e-10

    def test_constant_rate_various_rates(self):
        """Test multiple constant rates."""
        for rate in [1.0, 5.0, 10.0, 100.0, 1000.0]:
            profile = ConstantRateProfile(rate=rate)
            provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

            t = provider.next_arrival_time().to_seconds()
            expected = 1.0 / rate
            assert abs(t - expected) < 1e-10, f"Failed for rate={rate}"

    def test_linear_ramp_up(self):
        """Linear ramp from 0 to 100 over 10 seconds."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=0.0, end_rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        # First arrival: ∫[0→t] 10*s ds = 5*t² = 1.0 → t = sqrt(0.2) ≈ 0.447s
        t1 = provider.next_arrival_time().to_seconds()
        expected_t1 = (2.0 / 10.0) ** 0.5  # sqrt(2 * target / slope)
        assert abs(t1 - expected_t1) < 1e-6

    def test_linear_ramp_down(self):
        """Linear ramp from 100 to 10 over 5 seconds."""
        profile = LinearRampProfile(duration_s=5.0, start_rate=100.0, end_rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        t1 = provider.next_arrival_time().to_seconds()
        # Should be close to 1/100 = 0.01s but slightly longer due to decreasing rate
        assert 0.009 < t1 < 0.015

    def test_spike_profile(self):
        """Spike profile with baseline=10, spike=100."""
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=100.0,
            warmup_s=1.0,
            spike_duration_s=0.5,
        )
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        # First 10 events during warmup (rate=10, so 0.1s apart)
        warmup_times = []
        for _ in range(10):
            warmup_times.append(provider.next_arrival_time().to_seconds())

        # All should be during warmup phase
        assert all(t < 1.0 for t in warmup_times)

        # Events should be approximately 0.1s apart
        for i in range(1, len(warmup_times)):
            delta = warmup_times[i] - warmup_times[i-1]
            assert abs(delta - 0.1) < 1e-6

    def test_comparison_with_scipy_implementation(self):
        """Run same scenarios with scipy and our implementation, compare results."""
        # This test should be run during development to validate
        # After scipy is removed, this becomes a reference for expected values

        test_profiles = [
            ConstantRateProfile(rate=50.0),
            LinearRampProfile(duration_s=10.0, start_rate=10.0, end_rate=100.0),
            SpikeProfile(baseline_rate=20.0, spike_rate=200.0, warmup_s=2.0, spike_duration_s=1.0),
        ]

        for profile in test_profiles:
            provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
            times = [provider.next_arrival_time().to_seconds() for _ in range(20)]

            # Store these as expected values (captured from scipy implementation)
            # Then compare against our implementation
            ...
```

### Integration Tests

**File**: `tests/integration/test_arrival_time_no_scipy.py`

```python
class TestArrivalTimeIntegration:
    """Integration tests verifying arrival times in full simulation."""

    def test_constant_rate_event_count(self):
        """10 req/s over 10s should produce ~100 events."""
        profile = ConstantRateProfile(rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        counter = EventCounter("counter")
        event_provider = SimpleEventProvider(counter)

        source = Source(
            name="test_source",
            event_provider=event_provider,
            arrival_time_provider=provider,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[counter],
        )
        sim.run()

        assert 99 <= counter.count <= 101  # Allow for boundary effects

    def test_poisson_statistics(self):
        """Poisson arrivals should have exponential inter-arrival times."""
        import numpy as np

        profile = ConstantRateProfile(rate=100.0)
        provider = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)

        # Generate 1000 inter-arrival times
        times = []
        for _ in range(1001):
            times.append(provider.next_arrival_time().to_seconds())

        inter_arrivals = np.diff(times)

        # Mean should be ~1/rate = 0.01
        mean_ia = np.mean(inter_arrivals)
        assert 0.008 < mean_ia < 0.012

        # Coefficient of variation should be ~1 for exponential
        cv = np.std(inter_arrivals) / mean_ia
        assert 0.9 < cv < 1.1

    def test_ramp_profile_event_density(self):
        """Events should be denser toward end of ramp-up."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=10.0, end_rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        times = []
        for _ in range(200):
            t = provider.next_arrival_time().to_seconds()
            if t > 10.0:
                break
            times.append(t)

        # Count events in first half vs second half of time period
        first_half = sum(1 for t in times if t < 5.0)
        second_half = sum(1 for t in times if 5.0 <= t < 10.0)

        # Second half should have more events due to higher rate
        assert second_half > first_half * 1.5
```

### Regression Tests (Capture Current Behavior)

**File**: `tests/regression/test_arrival_time_regression.py`

```python
"""Regression tests capturing expected arrival times from scipy implementation.

These values were captured from the working scipy implementation and should
remain constant after scipy removal.
"""

import pytest
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.profile import ConstantRateProfile, LinearRampProfile
from happysimulator.core.temporal import Instant


class TestArrivalTimeRegression:
    """Golden value tests for arrival times."""

    # Values captured from scipy implementation (2026-01-31)
    CONSTANT_RATE_50_FIRST_10 = [
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
        0.199999999,
    ]

    CONSTANT_RATE_100_FIRST_10 = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.099999999,
    ]

    LINEAR_RAMP_10_100_FIRST_10 = [
        0.095864499,
        0.184655976,
        0.267741515,
        0.346097448,
        0.420449859,
        0.49135612,
        0.55925515,
        0.624499924,
        0.687379335,
        0.748133387,
    ]

    LINEAR_RAMP_100_10_FIRST_10 = [
        0.010004504,
        0.020018032,
        0.030040609,
        0.040072259,
        0.050113007,
        0.060162878,
        0.070221897,
        0.080290089,
        0.090367479,
        0.100454092,
    ]

    SPIKE_PROFILE_FIRST_30 = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.799999999, 0.899999998, 0.999999998,
        1.099999998, 1.199999998, 1.299999998, 1.399999998, 1.499999998, 1.599999998,
        1.699999998, 1.799999998, 1.899999998, 1.999999997, 2.009999997, 2.019999996,
        2.029999996, 2.039999995, 2.049999994, 2.059999994, 2.069999993, 2.079999993,
        2.089999992, 2.099999991,
    ]

    def test_constant_rate_50_regression(self):
        """Verify constant rate=50 arrival times match captured values."""
        profile = ConstantRateProfile(rate=50.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for expected in self.CONSTANT_RATE_50_FIRST_10:
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, f"Expected {expected}, got {actual}"

    def test_constant_rate_100_regression(self):
        """Verify constant rate=100 arrival times match captured values."""
        profile = ConstantRateProfile(rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for expected in self.CONSTANT_RATE_100_FIRST_10:
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, f"Expected {expected}, got {actual}"

    def test_linear_ramp_up_regression(self):
        """Verify linear ramp-up arrival times match captured values."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=10.0, end_rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for expected in self.LINEAR_RAMP_10_100_FIRST_10:
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, f"Expected {expected}, got {actual}"

    def test_linear_ramp_down_regression(self):
        """Verify linear ramp-down arrival times match captured values."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=100.0, end_rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for expected in self.LINEAR_RAMP_100_10_FIRST_10:
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, f"Expected {expected}, got {actual}"

    def test_spike_profile_regression(self):
        """Verify spike profile arrival times match captured values."""
        profile = SpikeProfile(baseline_rate=10.0, spike_rate=100.0, warmup_s=2.0, spike_duration_s=1.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for expected in self.SPIKE_PROFILE_FIRST_30:
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, f"Expected {expected}, got {actual}"
```

---

## File Organization

```
happysimulator/
├── numerics/
│   ├── __init__.py
│   ├── integration.py      # Adaptive Simpson's rule
│   └── root_finding.py     # Brent's method
├── load/
│   ├── arrival_time_provider.py  # Modified to use numerics/
│   └── ...

tests/
├── unit/
│   ├── numerics/
│   │   ├── __init__.py
│   │   ├── test_integration.py
│   │   └── test_root_finding.py
│   └── load/
│       └── test_arrival_time_provider.py
├── integration/
│   └── test_arrival_time_no_scipy.py
└── regression/
    └── test_arrival_time_regression.py
```

---

## Implementation Phases

### Phase 1: Capture Regression Values
**Before** removing scipy, run the current implementation and capture exact values:

```python
# Script to capture regression values
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.profile import ConstantRateProfile, LinearRampProfile, SpikeProfile
from happysimulator.core.temporal import Instant

profiles = [
    ("constant_50", ConstantRateProfile(rate=50.0)),
    ("constant_100", ConstantRateProfile(rate=100.0)),
    ("ramp_10_100", LinearRampProfile(10.0, 10.0, 100.0)),
    ("ramp_100_10", LinearRampProfile(10.0, 100.0, 10.0)),
    ("spike", SpikeProfile(10.0, 100.0, 2.0, 1.0)),
]

for name, profile in profiles:
    provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
    times = [provider.next_arrival_time().to_seconds() for _ in range(20)]
    print(f"{name}: {times}")
```

### Phase 2: Implement Numerical Methods
1. Create `happysimulator/numerics/` package
2. Implement `integrate_adaptive_simpson()`
3. Implement `brentq()`
4. Write unit tests comparing against scipy

### Phase 3: Implement Fast Paths
1. Add `_solve_constant_rate()` - O(1) direct calculation
2. Add `_solve_linear_ramp()` - O(1) quadratic formula
3. Add fast path dispatch in `next_arrival_time()`

### Phase 4: Integrate and Test
1. Update `arrival_time_provider.py` to use new implementations
2. Remove scipy imports
3. Run regression tests
4. Run full test suite

### Phase 5: Remove Dependency
1. Remove `scipy` from `pyproject.toml`
2. Update documentation
3. Verify clean install works

---

## Risks and Mitigations

### Risk: Numerical Accuracy
**Mitigation**:
- Use adaptive algorithms with configurable tolerance
- Compare results against scipy in tests
- Use analytical solutions where possible

### Risk: Performance Regression
**Mitigation**:
- Analytical solutions are O(1), faster than scipy
- Profile before/after to verify
- Adaptive integration only for complex profiles

### Risk: Edge Cases
**Mitigation**:
- Comprehensive test suite
- Handle zero/negative rates
- Handle very small/large time intervals
- Test with extreme rate profiles

### Risk: Floating Point Issues
**Mitigation**:
- Use same tolerances as scipy defaults
- Document precision limitations
- Test near machine epsilon boundaries

---

## Success Criteria

1. All existing tests pass
2. No scipy import in codebase
3. Regression tests pass with <1e-8 tolerance
4. Performance same or better for common profiles
5. Clean `pip install` without scipy

---

## References

- [Brent's Method (Wikipedia)](https://en.wikipedia.org/wiki/Brent%27s_method)
- [Adaptive Simpson's Rule](https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method)
- [scipy.integrate.quad source](https://github.com/scipy/scipy/blob/main/scipy/integrate/_quadpack_py.py)
- [scipy.optimize.brentq source](https://github.com/scipy/scipy/blob/main/scipy/optimize/_zeros_py.py)

---

## Appendix: Brent's Method Implementation Reference

The core algorithm for reference (simplified):

```python
def brentq(f, a, b, xtol=1e-12, maxiter=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa

    c, fc = a, fa
    d = e = b - a

    for _ in range(maxiter):
        if abs(b - a) < xtol:
            return b

        # Try inverse quadratic interpolation
        if fa != fc and fb != fc:
            s = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb))
        else:
            # Secant method
            s = b - fb*(b-a)/(fb-fa)

        # Acceptance conditions (ensure convergence)
        if not ((3*a+b)/4 < s < b or (3*a+b)/4 > s > b):
            s = (a + b) / 2  # Bisection fallback

        fs = f(s)
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa

    return b
```
