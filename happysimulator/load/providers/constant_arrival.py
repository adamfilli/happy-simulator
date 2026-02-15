"""Deterministic arrival time provider for constant-rate event generation.

Events arrive at regular intervals determined by the rate. With a constant
rate of r events/second, events occur every 1/r seconds. For time-varying
rates, events occur when the accumulated rate integral reaches 1.0.
"""

from happysimulator.load.arrival_time_provider import ArrivalTimeProvider


class ConstantArrivalTimeProvider(ArrivalTimeProvider):
    """Deterministic event arrival with regular spacing.

    Uses target integral value of 1.0, meaning each event occurs when the
    accumulated rate reaches 1.0. With constant rate r, this yields events
    every 1/r seconds.

    Use this for reproducible tests and scenarios where exact timing matters.
    """

    def _get_target_integral_value(self) -> float:
        """Return 1.0 for deterministic arrival spacing."""
        return 1.0
