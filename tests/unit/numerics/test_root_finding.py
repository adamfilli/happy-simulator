"""Unit tests for Brent's root finding method."""

import math

import pytest

from happysimulator.numerics.root_finding import RootResult, brentq


class TestBrentq:
    """Tests for brentq root finding."""

    def test_linear_root(self):
        """Find root of f(x) = x - 2 at x = 2."""
        result = brentq(lambda x: x - 2, 0, 5)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-12

    def test_quadratic_root_positive(self):
        """Find root of f(x) = x^2 - 4 at x = 2."""
        result = brentq(lambda x: x**2 - 4, 1, 3)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-11

    def test_quadratic_root_negative(self):
        """Find root of f(x) = x^2 - 4 at x = -2."""
        result = brentq(lambda x: x**2 - 4, -3, -1)
        assert result.converged
        assert abs(result.root + 2.0) < 1e-11

    def test_cubic_root(self):
        """Find root of f(x) = x^3 - 2x - 5 near x ≈ 2.0946."""
        result = brentq(lambda x: x**3 - 2 * x - 5, 2, 3)
        assert result.converged
        # Verify by evaluating function at root
        assert abs(result.root**3 - 2 * result.root - 5) < 1e-10

    def test_transcendental_root(self):
        """Find root of f(x) = cos(x) - x near x ≈ 0.739."""
        result = brentq(lambda x: math.cos(x) - x, 0, 1)
        assert result.converged
        expected = 0.7390851332151607
        assert abs(result.root - expected) < 1e-10

    def test_exponential_root(self):
        """Find root of f(x) = e^x - 10 at x = ln(10)."""
        result = brentq(lambda x: math.exp(x) - 10, 2, 3)
        assert result.converged
        assert abs(result.root - math.log(10)) < 1e-12

    def test_log_root(self):
        """Find root of f(x) = ln(x) - 1 at x = e."""
        result = brentq(lambda x: math.log(x) - 1, 2, 3)
        assert result.converged
        assert abs(result.root - math.e) < 1e-12

    def test_near_lower_boundary(self):
        """Root very close to lower bracket boundary."""
        result = brentq(lambda x: x - 0.001, 0, 1)
        assert result.converged
        assert abs(result.root - 0.001) < 1e-11

    def test_near_upper_boundary(self):
        """Root very close to upper bracket boundary."""
        result = brentq(lambda x: x - 0.999, 0, 1)
        assert result.converged
        assert abs(result.root - 0.999) < 1e-12

    def test_root_at_zero(self):
        """Root at x = 0."""
        result = brentq(lambda x: x, -1, 1)
        assert result.converged
        assert abs(result.root) < 1e-12

    def test_invalid_bracket_same_sign_raises(self):
        """Should raise ValueError when f(a) and f(b) have same sign."""
        with pytest.raises(ValueError, match="opposite signs"):
            brentq(lambda x: x**2 + 1, -1, 1)  # No real roots

    def test_invalid_bracket_both_positive_raises(self):
        """Should raise ValueError when both values are positive."""
        with pytest.raises(ValueError, match="opposite signs"):
            brentq(lambda x: x**2, 1, 2)  # Both f(1)=1 and f(2)=4 are positive

    def test_result_dataclass_fields(self):
        """Verify RootResult has expected fields."""
        result = brentq(lambda x: x - 1, 0, 2)
        assert isinstance(result, RootResult)
        assert hasattr(result, "root")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "function_calls")

    def test_function_call_count(self):
        """Verify function call count is reasonable."""
        result = brentq(lambda x: x - 1, 0, 2)
        assert result.function_calls >= 2  # At least evaluate at both brackets
        assert result.function_calls < 50  # Should converge quickly for simple case

    def test_iteration_count(self):
        """Verify iteration count is reasonable."""
        result = brentq(lambda x: x**3 - x - 2, 1, 2)
        assert result.iterations < 20  # Should converge in fewer iterations


class TestBrentqArrivalTimeScenarios:
    """Tests simulating arrival time root finding scenarios."""

    def test_constant_rate_arrival(self):
        """Find arrival time for constant rate 50, target area 1.0."""
        # For constant rate r, integral from 0 to t is r*t = 1.0
        # So root is at t = 1/r = 0.02
        rate = 50.0
        target = 1.0

        def objective(t: float) -> float:
            return rate * t - target

        result = brentq(objective, 0.001, 1.0)
        assert result.converged
        assert abs(result.root - 0.02) < 1e-11

    def test_linear_ramp_arrival(self):
        """Find arrival time for linear ramp rate."""
        # rate(t) = 10 + 9*t for t in [0, 10]
        # Integral from 0 to t: 10*t + 4.5*t^2 = target
        target = 1.0

        def objective(t: float) -> float:
            return 10.0 * t + 4.5 * t**2 - target

        result = brentq(objective, 0.001, 1.0)
        assert result.converged
        # Solve quadratic: 4.5*t^2 + 10*t - 1 = 0
        # t = (-10 + sqrt(100 + 18)) / 9 ≈ 0.09586
        expected = (-10 + math.sqrt(118)) / 9
        assert abs(result.root - expected) < 1e-10

    def test_high_rate_small_interval(self):
        """Root finding for high rate (short inter-arrival time)."""
        rate = 1000.0
        target = 1.0

        def objective(t: float) -> float:
            return rate * t - target

        result = brentq(objective, 1e-6, 1.0)
        assert result.converged
        assert abs(result.root - 0.001) < 1e-12

    def test_low_rate_large_interval(self):
        """Root finding for low rate (long inter-arrival time)."""
        rate = 0.1
        target = 1.0

        def objective(t: float) -> float:
            return rate * t - target

        result = brentq(objective, 0.1, 100.0)
        assert result.converged
        assert abs(result.root - 10.0) < 1e-10

    def test_objective_function_with_integration(self):
        """Combined integration and root finding scenario."""
        from happysimulator.numerics.integration import integrate_adaptive_simpson

        # Rate profile: constant 50
        def rate(t: float) -> float:
            return 50.0

        target = 1.0
        t_start = 0.0

        def objective(t_candidate: float) -> float:
            integral, _ = integrate_adaptive_simpson(rate, t_start, t_candidate)
            return integral - target

        result = brentq(objective, 0.001, 1.0)
        assert result.converged
        assert abs(result.root - 0.02) < 1e-10
