"""Root finding algorithms.

Provides Brent's method as a pure Python replacement for scipy.optimize.brentq.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class RootResult:
    """Result of root finding.

    Attributes:
        root: The found root value.
        converged: Whether the algorithm converged.
        iterations: Number of iterations used.
        function_calls: Number of function evaluations.
    """

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
        f: Continuous function to find root of.
        a: Lower bracket bound.
        b: Upper bracket bound.
        xtol: Absolute tolerance for root.
        rtol: Relative tolerance for root.
        maxiter: Maximum number of iterations.

    Returns:
        RootResult with the root and convergence info.

    Raises:
        ValueError: If f(a) and f(b) have the same sign.
    """
    func_calls = 0

    def eval_f(x: float) -> float:
        nonlocal func_calls
        func_calls += 1
        return f(x)

    fa = eval_f(a)
    fb = eval_f(b)

    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs, got f({a})={fa}, f({b})={fb}"
        )

    # Ensure |f(b)| <= |f(a)| so b is the best estimate
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for iteration in range(maxiter):
        # Check for convergence
        tol = 2.0 * rtol * abs(b) + xtol
        m = (c - b) / 2.0

        if abs(m) <= tol or fb == 0:
            return RootResult(
                root=b, converged=True, iterations=iteration, function_calls=func_calls
            )

        # Decide between interpolation and bisection
        if abs(e) >= tol and abs(fa) > abs(fb):
            # Try interpolation
            s = fb / fa

            if a == c:
                # Linear interpolation (secant method)
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            # Adjust signs
            if p > 0:
                q = -q
            else:
                p = -p

            # Accept interpolation if it's good enough
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                # Fall back to bisection
                d = m
                e = m
        else:
            # Bisection
            d = m
            e = m

        # Update a (the previous best estimate)
        a = b
        fa = fb

        # Update b (the current best estimate)
        if abs(d) > tol:
            b = b + d
        elif m > 0:
            b = b + tol
        else:
            b = b - tol

        fb = eval_f(b)

        # Maintain the bracket: f(b) and f(c) have opposite signs
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        elif abs(fc) < abs(fb):
            # Swap so b remains the best estimate
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

    # Did not converge within maxiter
    return RootResult(
        root=b, converged=False, iterations=maxiter, function_calls=func_calls
    )
