"""Tests for sliding window rate limiter with visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter import RateLimitedEntity, SlidingWindowPolicy
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

from .rate_limiter_helpers import (
    ConstantRateProfile,
    LinearRampProfile,
    StepRateProfile,
    bin_counts,
    linspace,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "test_name,profile,window_size_seconds,max_requests",
    [
        ("sliding_constant_within_limit", ConstantRateProfile(rate=3.0), 1.0, 5),
        (
            "sliding_ramp_exceeds_limit",
            LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=20.0),
            1.0,
            5,
        ),
        ("sliding_step_overload", StepRateProfile(t_switch_s=30.0, low=3.0, high=15.0), 1.0, 5),
        (
            "sliding_large_window",
            LinearRampProfile(t_end_s=60.0, start_rate=5.0, end_rate=30.0),
            2.0,
            10,
        ),
    ],
)
def test_sliding_window_with_profile(
    test_name: str,
    profile,
    window_size_seconds: float,
    max_requests: int,
    test_output_dir: "Path",
):
    """Test sliding window rate limiter behavior with various load profiles."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    sink = Sink("sink")
    policy = SlidingWindowPolicy(window_size_seconds=window_size_seconds, max_requests=max_requests)
    rate_limiter = RateLimitedEntity(
        name="sliding_window_limiter", downstream=sink, policy=policy, queue_capacity=10000,
    )

    effective_rate_limit = max_requests / window_size_seconds

    source = Source.with_profile(
        profile=profile, target=rate_limiter, poisson=False, name=f"RequestSource_{test_name}",
    )

    sim = Simulation(
        start_time=Instant.Epoch, end_time=end_time, sources=[source], entities=[rate_limiter, sink],
    )
    sim.run()

    received_s = [t.to_seconds() for t in rate_limiter.received_times]
    forwarded_s = [t.to_seconds() for t in rate_limiter.forwarded_times]

    total = rate_limiter.stats.forwarded + rate_limiter.queue_depth + rate_limiter.stats.dropped
    assert rate_limiter.stats.received == total
    assert len(received_s) >= 50, f"Expected at least 50 requests, got {len(received_s)}"

    print(f"\n=== {test_name} (Sliding Window) ===")
    print(
        f"Window: {window_size_seconds}s, Max requests: {max_requests} (effective rate: {effective_rate_limit}/s)"
    )
    print(f"Received:  {rate_limiter.stats.received}")
    print(f"Forwarded: {rate_limiter.stats.forwarded}")
    print(f"Queued:    {rate_limiter.queue_depth}")
    print(f"Dropped:   {rate_limiter.stats.dropped}")

    sample_times = linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    bin_s = 1.0
    recv_centers, recv_rates = bin_counts(received_s, duration_s, bin_s)
    fwd_centers, fwd_rates = bin_counts(forwarded_s, duration_s, bin_s)

    fig1, ax_rate = plt.subplots(figsize=(12, 5))
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile (target)", linewidth=2)
    ax_rate.axhline(
        y=effective_rate_limit, color="r", linestyle="--",
        label=f"Rate limit ({max_requests}/{window_size_seconds}s = {effective_rate_limit}/s)",
        linewidth=2,
    )
    if recv_centers:
        ax_rate.step(recv_centers, recv_rates, where="mid", alpha=0.7, label="Received (binned)", color="gray")
    if fwd_centers:
        ax_rate.step(fwd_centers, fwd_rates, where="mid", alpha=0.9, label="Forwarded (binned)", color="green")

    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title(
        f"Sliding Window Rate Limiter: {test_name}\n(window={window_size_seconds}s, max={max_requests})"
    )
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    fig1.tight_layout()
    fig1.savefig(test_output_dir / "rate_comparison.png", dpi=150)
    plt.close(fig1)

    print(f"Saved plots/data for {test_name} to: {test_output_dir}")


def test_sliding_window_basic_functionality():
    """Basic test for SlidingWindowPolicy + RateLimitedEntity."""
    sink = Sink("sink")
    policy = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=3)
    rate_limiter = RateLimitedEntity(
        name="limiter", downstream=sink, policy=policy, queue_capacity=100,
    )

    Simulation(start_time=Instant.Epoch, duration=10.0, sources=[], entities=[rate_limiter, sink])

    for _ in range(3):
        event = Event(time=Instant.Epoch, event_type="Request", target=rate_limiter)
        result = rate_limiter.handle_event(event)
        for evt in result:
            if evt.target is sink:
                sink.handle_event(evt)

    assert rate_limiter.stats.forwarded == 3
    assert rate_limiter.stats.dropped == 0
    assert len(sink.completion_times) == 3

    event = Event(time=Instant.Epoch, event_type="Request", target=rate_limiter)
    rate_limiter.handle_event(event)
    assert rate_limiter.stats.queued == 1


def test_sliding_window_empty():
    """Test that requests are allowed when window is empty."""
    sink = Sink("sink")
    policy = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=5)
    rate_limiter = RateLimitedEntity(
        name="limiter", downstream=sink, policy=policy, queue_capacity=100,
    )

    Simulation(start_time=Instant.Epoch, duration=10.0, sources=[], entities=[rate_limiter, sink])

    event = Event(time=Instant.Epoch, event_type="Request", target=rate_limiter)
    rate_limiter.handle_event(event)

    assert rate_limiter.stats.forwarded == 1
    assert rate_limiter.stats.dropped == 0
