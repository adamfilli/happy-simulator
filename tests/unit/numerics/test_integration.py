"""Unit tests for adaptive Simpson's rule integration."""

import math

import pytest

from happysimulator.numerics.integration import integrate_adaptive_simpson


class TestAdaptiveSimpson:
    """Tests for integrate_adaptive_simpson."""

    def test_constant_function(self):
        """Integral of constant 5 from 0 to 1 is 5."""
        result, _ = integrate_adaptive_simpson(lambda x: 5.0, 0, 1)
        assert abs(result - 5.0) < 1e-10

    def test_linear_function(self):
        """Integral of x from 0 to 1 is 0.5."""
        result, _ = integrate_adaptive_simpson(lambda x: x, 0, 1)
        assert abs(result - 0.5) < 1e-10

    def test_quadratic_function(self):
        """Integral of x^2 from 0 to 1 is 1/3."""
        result, _ = integrate_adaptive_simpson(lambda x: x**2, 0, 1)
        assert abs(result - 1 / 3) < 1e-10

    def test_cubic_function(self):
        """Integral of x^3 from 0 to 2 is 4."""
        result, _ = integrate_adaptive_simpson(lambda x: x**3, 0, 2)
        assert abs(result - 4.0) < 1e-10

    def test_sine_function(self):
        """Integral of sin(x) from 0 to pi is 2."""
        result, _ = integrate_adaptive_simpson(math.sin, 0, math.pi)
        assert abs(result - 2.0) < 1e-10

    def test_cosine_function(self):
        """Integral of cos(x) from 0 to pi/2 is 1."""
        result, _ = integrate_adaptive_simpson(math.cos, 0, math.pi / 2)
        assert abs(result - 1.0) < 1e-10

    def test_exponential_function(self):
        """Integral of e^x from 0 to 1 is e - 1."""
        result, _ = integrate_adaptive_simpson(math.exp, 0, 1)
        assert abs(result - (math.e - 1)) < 1e-10

    def test_reciprocal_function(self):
        """Integral of 1/(1 + x^2) from 0 to 1 is pi/4."""
        result, _ = integrate_adaptive_simpson(lambda x: 1 / (1 + x**2), 0, 1)
        assert abs(result - math.pi / 4) < 1e-10

    def test_gaussian(self):
        """Integral of e^(-x^2) from -2 to 2 is approximately sqrt(pi)*erf(2)."""
        result, _ = integrate_adaptive_simpson(lambda x: math.exp(-(x**2)), -2, 2)
        expected = math.sqrt(math.pi) * math.erf(2)
        assert abs(result - expected) < 1e-8

    def test_zero_width_interval(self):
        """Integral over zero-width interval is 0."""
        result, error = integrate_adaptive_simpson(lambda x: x**2, 1, 1)
        assert result == 0.0
        assert error == 0.0

    def test_reversed_bounds(self):
        """Integral with reversed bounds gives negative result."""
        result_forward, _ = integrate_adaptive_simpson(lambda x: x**2, 0, 1)
        result_backward, _ = integrate_adaptive_simpson(lambda x: x**2, 1, 0)
        assert abs(result_forward + result_backward) < 1e-10

    def test_large_interval(self):
        """Integration over larger interval."""
        result, _ = integrate_adaptive_simpson(lambda x: x, 0, 100)
        assert abs(result - 5000.0) < 1e-6

    def test_narrow_peak(self):
        """Integration of narrow Gaussian peak."""
        # Gaussian with sigma = 0.1 centered at 0.5
        sigma = 0.1

        def narrow_gaussian(x: float) -> float:
            return math.exp(-((x - 0.5) ** 2) / (2 * sigma**2))

        result, _ = integrate_adaptive_simpson(narrow_gaussian, 0, 1, tol=1e-8)
        # Expected: sqrt(2*pi) * sigma * (erf((0.5)/(sigma*sqrt(2))) - erf((-0.5)/(sigma*sqrt(2)))) / 2
        # For sigma=0.1, the peak is mostly within [0, 1], so result ≈ sqrt(2*pi) * 0.1 ≈ 0.2507
        expected = sigma * math.sqrt(2 * math.pi) * math.erf(0.5 / (sigma * math.sqrt(2)))
        assert abs(result - expected) < 1e-6

    def test_piecewise_constant_approximation(self):
        """Integration of step function (requires adaptation)."""
        # Step function: 0 for x < 0.5, 1 for x >= 0.5
        def step(x: float) -> float:
            return 1.0 if x >= 0.5 else 0.0

        result, _ = integrate_adaptive_simpson(step, 0, 1, tol=1e-6)
        # Should be approximately 0.5
        assert abs(result - 0.5) < 1e-3


class TestIntegrationRateProfiles:
    """Tests simulating rate profile integration for arrival time calculations."""

    def test_constant_rate_integral(self):
        """Integral of constant rate 50 from 0 to 0.02 should be 1.0."""
        rate = 50.0
        result, _ = integrate_adaptive_simpson(lambda t: rate, 0, 0.02)
        assert abs(result - 1.0) < 1e-10

    def test_linear_ramp_integral(self):
        """Integral of linear ramp from 10 to 100 over 10s at early time."""
        # rate(t) = 10 + 9*t for t in [0, 10]
        def ramp(t: float) -> float:
            return 10.0 + 9.0 * t

        # Analytical: integral from 0 to T = 10*T + 4.5*T^2
        t_end = 0.5
        expected = 10.0 * t_end + 4.5 * t_end**2
        result, _ = integrate_adaptive_simpson(ramp, 0, t_end)
        assert abs(result - expected) < 1e-10

    def test_varying_rate_integral(self):
        """Integration of time-varying rate profile."""
        # Sinusoidal rate: rate(t) = 50 + 20*sin(2*pi*t)
        def sinusoidal_rate(t: float) -> float:
            return 50.0 + 20.0 * math.sin(2 * math.pi * t)

        # Integral from 0 to 1: 50*1 + 20*(-cos(2*pi)/2*pi + cos(0)/2*pi) = 50
        result, _ = integrate_adaptive_simpson(sinusoidal_rate, 0, 1)
        assert abs(result - 50.0) < 1e-8
