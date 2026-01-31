"""Rate limiter components for controlling request throughput.

This module provides various rate limiting algorithms:
- FixedWindowRateLimiter: Simple fixed time window limiting
- AdaptiveRateLimiter: AIMD-based self-tuning rate limiter
- DistributedRateLimiter: Coordinated limiting across multiple instances

For basic rate limiting, see also:
- TokenBucketRateLimiter: Classic token bucket (allows bursting)
- LeakyBucketRateLimiter: Strict output rate (no bursting)
- SlidingWindowRateLimiter: Sliding window log algorithm

Example:
    from happysimulator.components.rate_limiter import (
        FixedWindowRateLimiter,
        AdaptiveRateLimiter,
    )

    # Simple fixed window: 100 requests per second
    limiter = FixedWindowRateLimiter(
        name="api_limit",
        downstream=server,
        requests_per_window=100,
        window_size=1.0,
    )

    # Self-tuning adaptive limiter
    adaptive = AdaptiveRateLimiter(
        name="adaptive",
        downstream=server,
        initial_rate=100.0,
        min_rate=10.0,
        max_rate=1000.0,
    )
"""

from happysimulator.components.rate_limiter.fixed_window import (
    FixedWindowRateLimiter,
    FixedWindowStats,
)
from happysimulator.components.rate_limiter.adaptive import (
    AdaptiveRateLimiter,
    AdaptiveRateLimiterStats,
    RateAdjustmentReason,
    RateSnapshot,
)
from happysimulator.components.rate_limiter.distributed import (
    DistributedRateLimiter,
    DistributedRateLimiterStats,
)

__all__ = [
    # Fixed Window
    "FixedWindowRateLimiter",
    "FixedWindowStats",
    # Adaptive
    "AdaptiveRateLimiter",
    "AdaptiveRateLimiterStats",
    "RateAdjustmentReason",
    "RateSnapshot",
    # Distributed
    "DistributedRateLimiter",
    "DistributedRateLimiterStats",
]
