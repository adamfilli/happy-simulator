"""Tests for leaky bucket rate limiter with visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter import LeakyBucketPolicy, RateLimitedEntity
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

from .rate_limiter_helpers import (
    ConstantRateProfile,
    LinearRampProfile,
    StepRateProfile,
    bin_counts,
    linspace,
    write_csv,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "test_name,profile,leak_rate",
    [
        ("leaky_constant_within_limit", ConstantRateProfile(rate=3.0), 5.0),
        (
            "leaky_ramp_exceeds_limit",
            LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=20.0),
            5.0,
        ),
        ("leaky_step_overload", StepRateProfile(t_switch_s=30.0, low=3.0, high=15.0), 5.0),
    ],
)
def test_leaky_bucket_with_profile(
    test_name: str,
    profile,
    leak_rate: float,
    test_output_dir: "Path",
):
    """Test leaky bucket rate limiter behavior with various load profiles."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    sink = Sink("sink")
    policy = LeakyBucketPolicy(leak_rate=leak_rate)
    rate_limiter = RateLimitedEntity(
        name="leaky_rate_limiter", downstream=sink, policy=policy, queue_capacity=10000,
    )

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

    print(f"\n=== {test_name} (Leaky Bucket) ===")
    print(f"Received:  {rate_limiter.stats.received}")
    print(f"Forwarded: {rate_limiter.stats.forwarded}")
    print(f"In queue:  {rate_limiter.queue_depth}")
    print(f"Dropped:   {rate_limiter.stats.dropped}")

    write_csv(
        test_output_dir / "events_received.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(received_s)),
    )
    write_csv(
        test_output_dir / "events_forwarded.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(forwarded_s)),
    )

    sample_times = linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    bin_s = 1.0
    recv_centers, recv_rates = bin_counts(received_s, duration_s, bin_s)
    fwd_centers, fwd_rates = bin_counts(forwarded_s, duration_s, bin_s)

    fig1, ax_rate = plt.subplots(figsize=(12, 5))
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile (target)", linewidth=2)
    ax_rate.axhline(
        y=leak_rate, color="r", linestyle="--", label=f"Leak rate ({leak_rate}/s)", linewidth=2
    )
    if recv_centers:
        ax_rate.step(recv_centers, recv_rates, where="mid", alpha=0.7, label="Received (binned)", color="gray")
    if fwd_centers:
        ax_rate.step(fwd_centers, fwd_rates, where="mid", alpha=0.9, label="Forwarded (binned)", color="green")

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title(f"Leaky Bucket Rate Limiter: {test_name}\n(leak_rate={leak_rate}/s)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)
    ax_rate.set_xlabel("Time (s)")

    fig1.tight_layout()
    fig1.savefig(test_output_dir / "rate_comparison.png", dpi=150)
    plt.close(fig1)

    print(f"Saved plots/data for {test_name} to: {test_output_dir}")
