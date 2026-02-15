"""Rate limiter components for controlling request throughput.

This module provides:

**Policies** (pure algorithms, not Entities):
- TokenBucketPolicy: Classic token bucket (allows bursting)
- LeakyBucketPolicy: Strict output rate (no bursting)
- SlidingWindowPolicy: Sliding window log algorithm
- FixedWindowPolicy: Fixed time window counter
- AdaptivePolicy: AIMD-based self-tuning rate limiter

**Entity** (simulation actor):
- RateLimitedEntity: Generic Entity that wraps any policy with a FIFO queue

**Distributed** (unchanged, uses generator yields for I/O):
- DistributedRateLimiter: Coordinated limiting across multiple instances

Example:
    from happysimulator.components.rate_limiter import (
        RateLimitedEntity,
        TokenBucketPolicy,
        FixedWindowPolicy,
    )

    # Token bucket with queuing
    limiter = RateLimitedEntity(
        name="api_limit",
        downstream=server,
        policy=TokenBucketPolicy(capacity=10.0, refill_rate=5.0),
    )

    # Fixed window with queuing
    limiter = RateLimitedEntity(
        name="api_limit",
        downstream=server,
        policy=FixedWindowPolicy(requests_per_window=100, window_size=1.0),
    )
"""

from happysimulator.components.rate_limiter.distributed import (
    DistributedRateLimiter,
    DistributedRateLimiterStats,
)
from happysimulator.components.rate_limiter.inductor import (
    Inductor,
    InductorStats,
)
from happysimulator.components.rate_limiter.null import NullRateLimiter
from happysimulator.components.rate_limiter.policy import (
    AdaptivePolicy,
    FixedWindowPolicy,
    LeakyBucketPolicy,
    RateAdjustmentReason,
    RateLimiterPolicy,
    RateSnapshot,
    SlidingWindowPolicy,
    TokenBucketPolicy,
)
from happysimulator.components.rate_limiter.rate_limited_entity import (
    RateLimitedEntity,
    RateLimitedEntityStats,
)

__all__ = [
    "AdaptivePolicy",
    # Distributed (unchanged)
    "DistributedRateLimiter",
    "DistributedRateLimiterStats",
    "FixedWindowPolicy",
    # Inductor (burst suppression)
    "Inductor",
    "InductorStats",
    "LeakyBucketPolicy",
    # Null (pass-through)
    "NullRateLimiter",
    "RateAdjustmentReason",
    # Entity
    "RateLimitedEntity",
    "RateLimitedEntityStats",
    # Protocol
    "RateLimiterPolicy",
    "RateSnapshot",
    "SlidingWindowPolicy",
    # Policies
    "TokenBucketPolicy",
]
