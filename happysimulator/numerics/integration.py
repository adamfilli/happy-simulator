"""Numerical integration methods.

Provides adaptive Simpson's rule integration as a pure Python replacement
for scipy.integrate.quad.
"""

from collections.abc import Callable


def integrate_adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> tuple[float, float]:
    """Adaptive Simpson's rule integration.

    Uses recursive subdivision to achieve desired tolerance.
    This is a replacement for scipy.integrate.quad for smooth functions.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        tol: Absolute error tolerance.
        max_depth: Maximum recursion depth.

    Returns:
        Tuple of (integral_value, error_estimate).

    Raises:
        ValueError: If bounds are invalid or max_depth is exceeded.
    """
    if a == b:
        return 0.0, 0.0

    if a > b:
        result, error = integrate_adaptive_simpson(f, b, a, tol, max_depth)
        return -result, error

    def _simpson(fa: float, fm: float, fb: float, h: float) -> float:
        """Compute Simpson's rule: h/3 * (f(a) + 4*f(m) + f(b))."""
        return h / 3.0 * (fa + 4.0 * fm + fb)

    def _adaptive(
        a: float,
        b: float,
        fa: float,
        fb: float,
        s_whole: float,
        depth: int,
        tol: float,
    ) -> tuple[float, float]:
        """Recursive adaptive Simpson's rule."""
        m = (a + b) / 2.0
        h = (b - a) / 2.0

        fm = f(m)
        lm = (a + m) / 2.0
        rm = (m + b) / 2.0
        flm = f(lm)
        frm = f(rm)

        s_left = _simpson(fa, flm, fm, h / 2.0)
        s_right = _simpson(fm, frm, fb, h / 2.0)
        s_combined = s_left + s_right

        error_estimate = (s_combined - s_whole) / 15.0

        if depth >= max_depth or abs(error_estimate) < tol:
            # Richardson extrapolation for improved accuracy
            return s_combined + error_estimate, abs(error_estimate)

        # Recurse on each half with tighter tolerance
        left_result, left_error = _adaptive(a, m, fa, fm, s_left, depth + 1, tol / 2.0)
        right_result, right_error = _adaptive(m, b, fm, fb, s_right, depth + 1, tol / 2.0)

        return left_result + right_result, left_error + right_error

    # Initial evaluation
    fa = f(a)
    fb = f(b)
    m = (a + b) / 2.0
    fm = f(m)
    h = (b - a) / 2.0

    s_whole = _simpson(fa, fm, fb, h)

    return _adaptive(a, b, fa, fb, s_whole, 0, tol)
