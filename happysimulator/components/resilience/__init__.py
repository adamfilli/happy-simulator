"""Resilience patterns for fault tolerance.

This module provides components for building resilient systems that
handle failures gracefully.

Components:
    CircuitBreaker: Fails fast when downstream is unhealthy
    Bulkhead: Isolates resources to prevent cascade failures
    TimeoutWrapper: Wraps services with timeout handling
    Fallback: Provides fallback behavior on failure
    Hedge: Sends redundant requests to reduce tail latency

Example:
    from happysimulator.components.resilience import (
        CircuitBreaker,
        Bulkhead,
        TimeoutWrapper,
        Fallback,
        Hedge,
    )

    # Compose multiple resilience patterns
    hedge = Hedge("hedge", backend, hedge_delay=0.050)
    timeout = TimeoutWrapper("timeout", hedge, timeout=5.0)
    breaker = CircuitBreaker("breaker", timeout, failure_threshold=5)
    bulkhead = Bulkhead("bulkhead", breaker, max_concurrent=10)
"""

from happysimulator.components.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerStats,
)
from happysimulator.components.resilience.bulkhead import (
    Bulkhead,
    BulkheadStats,
)
from happysimulator.components.resilience.timeout import (
    TimeoutWrapper,
    TimeoutStats,
)
from happysimulator.components.resilience.fallback import (
    Fallback,
    FallbackStats,
)
from happysimulator.components.resilience.hedge import (
    Hedge,
    HedgeStats,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerStats",
    # Bulkhead
    "Bulkhead",
    "BulkheadStats",
    # Timeout
    "TimeoutWrapper",
    "TimeoutStats",
    # Fallback
    "Fallback",
    "FallbackStats",
    # Hedge
    "Hedge",
    "HedgeStats",
]
