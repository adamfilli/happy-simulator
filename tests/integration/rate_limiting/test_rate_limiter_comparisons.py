"""Comparison tests across different rate limiter types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter import (
    LeakyBucketPolicy,
    RateLimitedEntity,
    SlidingWindowPolicy,
    TokenBucketPolicy,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

from .rate_limiter_helpers import LinearRampProfile, bin_counts, linspace

if TYPE_CHECKING:
    from pathlib import Path


def test_leaky_bucket_vs_token_bucket_comparison(test_output_dir: Path):
    """Compare leaky bucket vs token bucket behavior with the same load profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)
    profile = LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=15.0)
    rate_limit = 5.0

    # Token Bucket
    token_sink = Sink("token_sink")
    token_policy = TokenBucketPolicy(capacity=10.0, refill_rate=rate_limit)
    token_limiter = RateLimitedEntity(
        name="token_limiter", downstream=token_sink, policy=token_policy, queue_capacity=10000,
    )
    token_source = Source.with_profile(
        profile=profile, target=token_limiter, poisson=False, name="TokenSource",
    )

    # Leaky Bucket
    leaky_sink = Sink("leaky_sink")
    leaky_policy = LeakyBucketPolicy(leak_rate=rate_limit)
    leaky_limiter = RateLimitedEntity(
        name="leaky_limiter", downstream=leaky_sink, policy=leaky_policy, queue_capacity=10000,
    )
    leaky_source = Source.with_profile(
        profile=profile, target=leaky_limiter, poisson=False, name="LeakySource",
    )

    token_sim = Simulation(
        start_time=Instant.Epoch, end_time=end_time,
        sources=[token_source], entities=[token_limiter, token_sink],
    )
    token_sim.run()

    leaky_sim = Simulation(
        start_time=Instant.Epoch, end_time=end_time,
        sources=[leaky_source], entities=[leaky_limiter, leaky_sink],
    )
    leaky_sim.run()

    token_fwd_s = [t.to_seconds() for t in token_limiter.forwarded_times]
    leaky_fwd_s = [t.to_seconds() for t in leaky_limiter.forwarded_times]

    sample_times = linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    bin_s = 1.0
    token_centers, token_rates = bin_counts(token_fwd_s, duration_s, bin_s)
    leaky_centers, leaky_rates = bin_counts(leaky_fwd_s, duration_s, bin_s)

    fig, (ax_rate, ax_cum) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile", linewidth=2, alpha=0.7)
    ax_rate.axhline(
        y=rate_limit, color="k", linestyle="--", label=f"Rate limit ({rate_limit}/s)", linewidth=2
    )
    if token_centers:
        ax_rate.step(token_centers, token_rates, where="mid", label="Token bucket (forwarded)", color="green", linewidth=1.5)
    if leaky_centers:
        ax_rate.step(leaky_centers, leaky_rates, where="mid", label="Leaky bucket (forwarded)", color="purple", linewidth=1.5)

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title("Token Bucket vs Leaky Bucket Rate Limiter Comparison\n(same rate limit, same load profile)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    if token_fwd_s:
        ax_cum.step(
            [0.0, *token_fwd_s], [0, *list(range(1, len(token_fwd_s) + 1))],
            where="post", label=f"Token bucket (total: {len(token_fwd_s)})", color="green", linewidth=2,
        )
    if leaky_fwd_s:
        ax_cum.step(
            [0.0, *leaky_fwd_s], [0, *list(range(1, len(leaky_fwd_s) + 1))],
            where="post", label=f"Leaky bucket (total: {len(leaky_fwd_s)})", color="purple", linewidth=2,
        )

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative forwarded")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(test_output_dir / "token_vs_leaky_comparison.png", dpi=150)
    plt.close(fig)

    print("\n=== Token Bucket vs Leaky Bucket Comparison ===")
    print(f"Token Bucket: forwarded={token_limiter.stats.forwarded}, queued={token_limiter.queue_depth}")
    print(f"Leaky Bucket: forwarded={leaky_limiter.stats.forwarded}, queued={leaky_limiter.queue_depth}")
    print(f"Saved comparison plot to: {test_output_dir}")


def test_all_rate_limiters_comparison(test_output_dir: Path):
    """Compare all three rate limiter types with the same load profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)
    profile = LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=15.0)
    rate_limit = 5.0

    # Token Bucket
    token_sink = Sink("token_sink")
    token_policy = TokenBucketPolicy(capacity=10.0, refill_rate=rate_limit)
    token_limiter = RateLimitedEntity(
        name="token_limiter", downstream=token_sink, policy=token_policy, queue_capacity=10000,
    )
    token_source = Source.with_profile(
        profile=profile, target=token_limiter, poisson=False, name="TokenSource",
    )

    # Leaky Bucket
    leaky_sink = Sink("leaky_sink")
    leaky_policy = LeakyBucketPolicy(leak_rate=rate_limit)
    leaky_limiter = RateLimitedEntity(
        name="leaky_limiter", downstream=leaky_sink, policy=leaky_policy, queue_capacity=10000,
    )
    leaky_source = Source.with_profile(
        profile=profile, target=leaky_limiter, poisson=False, name="LeakySource",
    )

    # Sliding Window
    sliding_sink = Sink("sliding_sink")
    sliding_policy = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=int(rate_limit))
    sliding_limiter = RateLimitedEntity(
        name="sliding_limiter", downstream=sliding_sink, policy=sliding_policy, queue_capacity=10000,
    )
    sliding_source = Source.with_profile(
        profile=profile, target=sliding_limiter, poisson=False, name="SlidingSource",
    )

    # Run simulations
    for sources, entities in [
        ([token_source], [token_limiter, token_sink]),
        ([leaky_source], [leaky_limiter, leaky_sink]),
        ([sliding_source], [sliding_limiter, sliding_sink]),
    ]:
        sim = Simulation(
            start_time=Instant.Epoch, end_time=end_time, sources=sources, entities=entities,
        )
        sim.run()

    token_fwd_s = [t.to_seconds() for t in token_limiter.forwarded_times]
    leaky_fwd_s = [t.to_seconds() for t in leaky_limiter.forwarded_times]
    sliding_fwd_s = [t.to_seconds() for t in sliding_limiter.forwarded_times]

    sample_times = linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    bin_s = 1.0
    token_centers, token_rates = bin_counts(token_fwd_s, duration_s, bin_s)
    leaky_centers, leaky_rates = bin_counts(leaky_fwd_s, duration_s, bin_s)
    sliding_centers, sliding_rates = bin_counts(sliding_fwd_s, duration_s, bin_s)

    fig, (ax_rate, ax_cum) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile", linewidth=2, alpha=0.7)
    ax_rate.axhline(
        y=rate_limit, color="k", linestyle="--", label=f"Rate limit ({rate_limit}/s)", linewidth=2
    )
    if token_centers:
        ax_rate.step(token_centers, token_rates, where="mid", label="Token bucket", color="green", linewidth=1.5)
    if leaky_centers:
        ax_rate.step(leaky_centers, leaky_rates, where="mid", label="Leaky bucket", color="purple", linewidth=1.5)
    if sliding_centers:
        ax_rate.step(sliding_centers, sliding_rates, where="mid", label="Sliding window", color="orange", linewidth=1.5)

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title("All Rate Limiters Comparison\n(Token Bucket vs Leaky Bucket vs Sliding Window)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    if token_fwd_s:
        ax_cum.step(
            [0.0, *token_fwd_s], [0, *list(range(1, len(token_fwd_s) + 1))],
            where="post", label=f"Token bucket (total: {len(token_fwd_s)})", color="green", linewidth=2,
        )
    if leaky_fwd_s:
        ax_cum.step(
            [0.0, *leaky_fwd_s], [0, *list(range(1, len(leaky_fwd_s) + 1))],
            where="post", label=f"Leaky bucket (total: {len(leaky_fwd_s)})", color="purple", linewidth=2,
        )
    if sliding_fwd_s:
        ax_cum.step(
            [0.0, *sliding_fwd_s], [0, *list(range(1, len(sliding_fwd_s) + 1))],
            where="post", label=f"Sliding window (total: {len(sliding_fwd_s)})", color="orange", linewidth=2,
        )

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative forwarded")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(test_output_dir / "all_rate_limiters_comparison.png", dpi=150)
    plt.close(fig)

    print("\n=== All Rate Limiters Comparison ===")
    print(f"Token Bucket:   forwarded={token_limiter.stats.forwarded}, queued={token_limiter.queue_depth}")
    print(f"Leaky Bucket:   forwarded={leaky_limiter.stats.forwarded}, queued={leaky_limiter.queue_depth}")
    print(f"Sliding Window: forwarded={sliding_limiter.stats.forwarded}, queued={sliding_limiter.queue_depth}")
    print(f"Saved comparison plot to: {test_output_dir}")
