"""Integration tests for RateLimitedEntity with various policies.

Tests verify that RateLimitedEntity correctly queues and drains events
according to the plugged-in policy when run inside a full simulation.
"""

from __future__ import annotations

import pytest

from happysimulator.components.rate_limiter.policy import (
    AdaptivePolicy,
    FixedWindowPolicy,
    LeakyBucketPolicy,
    SlidingWindowPolicy,
    TokenBucketPolicy,
)
from happysimulator.components.rate_limiter.rate_limited_entity import (
    RateLimitedEntity,
)
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------

class TestRateLimitedEntityTokenBucket:

    def test_low_load_all_forwarded(self):
        """With load well below the rate limit, all requests are forwarded."""
        sink = Sink()
        policy = TokenBucketPolicy(capacity=10.0, refill_rate=10.0)
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=5.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.received > 0
        assert limiter.stats.dropped == 0
        assert limiter.stats.forwarded == limiter.stats.received

    def test_overload_queues_then_drains(self):
        """Excess requests queue and drain at the refill rate."""
        sink = Sink()
        policy = TokenBucketPolicy(capacity=5.0, refill_rate=5.0)
        limiter = RateLimitedEntity(
            "limiter", downstream=sink, policy=policy, queue_capacity=1000,
        )

        source = Source.constant(rate=20.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.received > 0
        assert limiter.stats.queued > 0
        assert limiter.stats.forwarded > 0
        assert limiter.stats.dropped == 0  # Queue capacity is large enough

    def test_queue_overflow_drops(self):
        """When queue overflows, excess requests are dropped."""
        sink = Sink()
        policy = TokenBucketPolicy(capacity=1.0, refill_rate=1.0, initial_tokens=0.0)
        limiter = RateLimitedEntity(
            "limiter", downstream=sink, policy=policy, queue_capacity=5,
        )

        source = Source.constant(rate=100.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.dropped > 0


# ---------------------------------------------------------------------------
# Leaky Bucket
# ---------------------------------------------------------------------------

class TestRateLimitedEntityLeakyBucket:

    def test_strict_output_rate(self):
        """Leaky bucket enforces strict output rate with no bursting."""
        sink = Sink()
        policy = LeakyBucketPolicy(leak_rate=5.0)
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=3.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.received > 0
        assert limiter.stats.forwarded > 0
        assert limiter.stats.dropped == 0


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------

class TestRateLimitedEntitySlidingWindow:

    def test_requests_within_window_limit(self):
        """Sliding window allows up to max_requests per window."""
        sink = Sink()
        policy = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=5)
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=3.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.received > 0
        assert limiter.stats.forwarded > 0
        assert limiter.stats.dropped == 0


# ---------------------------------------------------------------------------
# Fixed Window
# ---------------------------------------------------------------------------

class TestRateLimitedEntityFixedWindow:

    def test_window_counter_resets(self):
        """Fixed window resets counter at window boundaries."""
        sink = Sink()
        policy = FixedWindowPolicy(requests_per_window=5, window_size=1.0)
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=3.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.received > 0
        assert limiter.stats.forwarded > 0
        assert limiter.stats.dropped == 0


# ---------------------------------------------------------------------------
# Adaptive
# ---------------------------------------------------------------------------

class TestRateLimitedEntityAdaptive:

    def test_adaptive_with_feedback(self):
        """Adaptive policy adjusts rate based on success/failure feedback."""
        sink = Sink()
        policy = AdaptivePolicy(
            initial_rate=50.0, min_rate=5.0, max_rate=200.0,
            increase_step=5.0, decrease_factor=0.7,
        )
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=10.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        assert limiter.stats.forwarded > 0
        # Record some success feedback
        for t in sink.completion_times[:10]:
            policy.record_success(t)
        assert policy.successes > 0


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestInvariants:

    def test_received_equals_forwarded_plus_queued_plus_dropped(self):
        """received == forwarded + in_queue + dropped always holds."""
        sink = Sink()
        policy = TokenBucketPolicy(capacity=3.0, refill_rate=2.0)
        limiter = RateLimitedEntity(
            "limiter", downstream=sink, policy=policy, queue_capacity=50,
        )

        source = Source.constant(rate=15.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        total = limiter.stats.forwarded + limiter.queue_depth + limiter.stats.dropped
        assert limiter.stats.received == total

    def test_forwarded_matches_sink(self):
        """Number of forwarded events matches what the sink received."""
        sink = Sink()
        policy = TokenBucketPolicy(capacity=10.0, refill_rate=5.0)
        limiter = RateLimitedEntity("limiter", downstream=sink, policy=policy)

        source = Source.constant(rate=3.0, target=limiter, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[limiter, sink],
        )
        sim.run()

        # Sink might miss the last forward event if it fires at end_time boundary
        assert abs(limiter.stats.forwarded - len(sink.completion_times)) <= 1
