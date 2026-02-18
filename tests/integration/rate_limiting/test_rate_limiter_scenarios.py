"""Parametrized unit-style tests for rate limiter scenarios."""

from __future__ import annotations

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter import (
    LeakyBucketPolicy,
    RateLimitedEntity,
    TokenBucketPolicy,
)
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


def _make_rate_limiter(policy, queue_capacity=100):
    """Create a rate limiter with a simulation for unit-style testing."""
    sink = Sink("sink")
    rate_limiter = RateLimitedEntity(
        name="limiter",
        downstream=sink,
        policy=policy,
        queue_capacity=queue_capacity,
    )
    Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[],
        entities=[rate_limiter, sink],
    )
    return rate_limiter, sink


@pytest.mark.parametrize(
    "policy,queue_capacity,event_times,forward_to_sink,expected",
    [
        pytest.param(
            TokenBucketPolicy(capacity=5.0, refill_rate=2.0, initial_tokens=5.0),
            100,
            [i * 0.1 for i in range(5)],
            True,
            {"forwarded": 5, "dropped": 0, "sink_count": 5},
            id="token_bucket_basic",
        ),
        pytest.param(
            TokenBucketPolicy(capacity=5.0, refill_rate=1.0, initial_tokens=0.0),
            100,
            [0.0],
            False,
            {"queued": 1, "dropped": 0, "sink_count": 0},
            id="token_bucket_empty_bucket",
        ),
        pytest.param(
            LeakyBucketPolicy(leak_rate=2.0),
            100,
            [0.0, 0.0],
            True,
            {"forwarded": 1, "queued": 1},
            id="leaky_bucket_basic",
        ),
        pytest.param(
            LeakyBucketPolicy(leak_rate=1.0),
            3,
            [0.0] * 5,
            False,
            {"forwarded": 1, "dropped": 1, "queue_depth": 3},
            id="leaky_bucket_full_queue",
        ),
    ],
)
def test_rate_limiter_scenarios(policy, queue_capacity, event_times, forward_to_sink, expected):
    """Parametrized test covering basic rate limiter scenarios."""
    rate_limiter, sink = _make_rate_limiter(policy, queue_capacity)

    for time_s in event_times:
        event = Event(time=Instant.from_seconds(time_s), event_type="Request", target=rate_limiter)
        result = rate_limiter.handle_event(event)
        if forward_to_sink:
            for evt in result:
                if evt.target is sink:
                    sink.handle_event(evt)

    for attr, val in expected.items():
        if attr == "sink_count":
            assert len(sink.completion_times) == val, f"sink_count: expected {val}"
        elif attr == "queue_depth":
            assert rate_limiter.queue_depth == val, f"queue_depth: expected {val}"
        else:
            assert getattr(rate_limiter.stats, attr) == val, f"{attr}: expected {val}"
