"""Numerical methods for simulation computations.

This module provides pure Python implementations of numerical algorithms,
replacing scipy dependencies for:
- Numerical integration (Adaptive Simpson's rule)
- Root finding (Brent's method)
"""

from happysimulator.numerics.integration import integrate_adaptive_simpson
from happysimulator.numerics.root_finding import RootResult, brentq

__all__ = [
    "RootResult",
    "brentq",
    "integrate_adaptive_simpson",
]
