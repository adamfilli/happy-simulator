"""Tests for rate limiter entities with visualization.

These tests verify rate limiting behavior and generate plots showing:
- Incoming request rate vs forwarded/dropped rates
- Token bucket level / queue depth / window count over time
- Cumulative requests (received, forwarded, dropped)

Supported rate limiters:
- TokenBucketRateLimiter: Classic token bucket algorithm
- LeakyBucketRateLimiter: Queue-based leaky bucket algorithm
- SlidingWindowRateLimiter: Sliding window log algorithm

Run:
    pytest tests/test_rate_limiter.py -v

Output:
    test_output/test_rate_limiter/<test_name>/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pytest

from happysimulator.entities.entity import Entity
from happysimulator.entities.token_bucket_rate_limiter import TokenBucketRateLimiter
from happysimulator.entities.leaky_bucket_rate_limiter import LeakyBucketRateLimiter
from happysimulator.entities.sliding_window_rate_limiter import SlidingWindowRateLimiter
from happysimulator.events.event import Event
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.simulation import Simulation
from happysimulator.utils.instant import Instant


# --- Test Entities ---


class TimeSeriesCounterEntity(Entity):
    """Entity that records when it handled events (acts as a sink)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.handled_times: list[Instant] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.handled_times.append(event.time)
        return []


# --- Test Events and Providers ---


class RequestEvent(Event):
    """A simple request event targeting the rate limiter."""

    def __init__(self, time: Instant, target: Entity):
        super().__init__(time=time, event_type="Request", target=target)


class RequestProvider(EventProvider):
    """Generates request events targeting the rate limiter."""

    def __init__(self, rate_limiter: TokenBucketRateLimiter):
        super().__init__()
        self._rate_limiter = rate_limiter

    def get_events(self, time: Instant) -> List[Event]:
        return [RequestEvent(time, self._rate_limiter)]


# --- Profile Definitions ---


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant arrival rate profile."""
    rate: float

    def get_rate(self, time: Instant) -> float:  # noqa: ARG002
        return float(self.rate)


@dataclass(frozen=True)
class LinearRampProfile(Profile):
    """Linear ramp from start_rate to end_rate over t_end_s seconds."""
    t_end_s: float
    start_rate: float
    end_rate: float

    def get_rate(self, time: Instant) -> float:
        t = max(0.0, min(time.to_seconds(), self.t_end_s))
        if self.t_end_s <= 0:
            return float(self.end_rate)
        frac = t / self.t_end_s
        return float(self.start_rate + frac * (self.end_rate - self.start_rate))


@dataclass(frozen=True)
class StepRateProfile(Profile):
    """Step function that switches from low to high rate at t_switch_s."""
    t_switch_s: float
    low: float
    high: float

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        return float(self.low if t < self.t_switch_s else self.high)


# --- Utility Functions ---


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Generate n evenly spaced values from start to stop."""
    if n <= 1:
        return [float(start)]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    """Write data to a CSV file."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _bin_counts(times_s: list[float], duration_s: float, bin_s: float) -> tuple[list[float], list[float]]:
    """Bin event times and return (bin_centers, events_per_second)."""
    if duration_s <= 0 or bin_s <= 0:
        return [], []

    n_bins = int(duration_s // bin_s) + 1
    counts = [0] * n_bins

    for t in times_s:
        if t < 0:
            continue
        idx = int(t // bin_s)
        if 0 <= idx < n_bins:
            counts[idx] += 1

    centers = [(i + 0.5) * bin_s for i in range(n_bins)]
    rates = [c / bin_s for c in counts]
    return centers, rates


# --- Test Cases ---


@pytest.mark.parametrize(
    "test_name,profile,bucket_capacity,refill_rate",
    [
        # Low constant load - should be fully forwarded
        ("constant_within_limit", ConstantRateProfile(rate=5.0), 20.0, 10.0),
        # Ramp up exceeding rate limit
        ("ramp_exceeds_limit", LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=20.0), 10.0, 5.0),
        # Step function - sudden increase
        ("step_overload", StepRateProfile(t_switch_s=30.0, low=3.0, high=15.0), 10.0, 5.0),
        # High constant load - significant dropping
        ("constant_overload", ConstantRateProfile(rate=20.0), 10.0, 5.0),
    ],
)
def test_rate_limiter_with_profile(
    test_name: str,
    profile: Profile,
    bucket_capacity: float,
    refill_rate: float,
    test_output_dir: Path,
):
    """Test rate limiter behavior with various load profiles and visualize results."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    # Create sink and rate limiter
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = TokenBucketRateLimiter(
        name="rate_limiter",
        downstream=sink,
        capacity=bucket_capacity,
        refill_rate=refill_rate,
    )

    # Create source targeting the rate limiter
    provider = RequestProvider(rate_limiter)
    arrival_provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(
        name=f"RequestSource_{test_name}",
        event_provider=provider,
        arrival_time_provider=arrival_provider,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[rate_limiter, sink],
    )
    sim.run()

    # Extract time series data
    received_s = [t.to_seconds() for t in rate_limiter.received_times]
    forwarded_s = [t.to_seconds() for t in rate_limiter.forwarded_times]
    dropped_s = [t.to_seconds() for t in rate_limiter.dropped_times]
    token_times = [(t.to_seconds(), level) for t, level in rate_limiter.token_levels]

    # Verify basic invariants
    assert rate_limiter.stats.requests_received == rate_limiter.stats.requests_forwarded + rate_limiter.stats.requests_dropped
    assert rate_limiter.stats.requests_forwarded == len(sink.handled_times)
    assert len(received_s) > 0, "Should have received some requests"

    # Print summary
    print(f"\n=== {test_name} ===")
    print(f"Received:  {rate_limiter.stats.requests_received}")
    print(f"Forwarded: {rate_limiter.stats.requests_forwarded}")
    print(f"Dropped:   {rate_limiter.stats.requests_dropped}")
    print(f"Drop rate: {rate_limiter.stats.requests_dropped / rate_limiter.stats.requests_received * 100:.1f}%")

    # --- Save CSV data ---
    _write_csv(
        test_output_dir / "events_received.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(received_s)),
    )
    _write_csv(
        test_output_dir / "events_forwarded.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(forwarded_s)),
    )
    _write_csv(
        test_output_dir / "events_dropped.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(dropped_s)),
    )
    _write_csv(
        test_output_dir / "token_levels.csv",
        header=["time_s", "tokens"],
        rows=token_times,
    )

    # --- Sample profile for plotting ---
    sample_times = _linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    # --- Plot 1: Rate comparison (profile vs realized received/forwarded/dropped) ---
    bin_s = 1.0
    recv_centers, recv_rates = _bin_counts(received_s, duration_s, bin_s)
    fwd_centers, fwd_rates = _bin_counts(forwarded_s, duration_s, bin_s)
    drop_centers, drop_rates = _bin_counts(dropped_s, duration_s, bin_s)

    fig1, (ax_rate, ax_tokens) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # Rate comparison
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile (target)", linewidth=2)
    ax_rate.axhline(y=refill_rate, color="r", linestyle="--", label=f"Rate limit ({refill_rate}/s)", linewidth=2)
    if recv_centers:
        ax_rate.step(recv_centers, recv_rates, where="mid", alpha=0.7, label="Received (binned)", color="gray")
    if fwd_centers:
        ax_rate.step(fwd_centers, fwd_rates, where="mid", alpha=0.9, label="Forwarded (binned)", color="green")
    if drop_centers:
        ax_rate.step(drop_centers, drop_rates, where="mid", alpha=0.9, label="Dropped (binned)", color="red")

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title(f"Rate Limiter: {test_name}\n(capacity={bucket_capacity}, refill={refill_rate}/s)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    # Token level over time
    if token_times:
        token_t, token_v = zip(*token_times)
        ax_tokens.plot(token_t, token_v, "m-", alpha=0.7, label="Token level", linewidth=1)
        ax_tokens.axhline(y=bucket_capacity, color="orange", linestyle=":", label=f"Capacity ({bucket_capacity})")
        ax_tokens.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="Min required (1)")

    ax_tokens.set_xlabel("Time (s)")
    ax_tokens.set_ylabel("Tokens")
    ax_tokens.legend(loc="upper right")
    ax_tokens.grid(True, alpha=0.3)
    ax_tokens.set_ylim(bottom=0)

    fig1.tight_layout()
    fig1.savefig(test_output_dir / "rate_and_tokens.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Cumulative counts ---
    fig2, ax_cum = plt.subplots(figsize=(12, 5))

    # Step plots for cumulative counts
    if received_s:
        ax_cum.step([0.0] + received_s, [0] + list(range(1, len(received_s) + 1)),
                    where="post", label="Received", color="gray", alpha=0.7)
    if forwarded_s:
        ax_cum.step([0.0] + forwarded_s, [0] + list(range(1, len(forwarded_s) + 1)),
                    where="post", label="Forwarded", color="green", linewidth=2)
    if dropped_s:
        ax_cum.step([0.0] + dropped_s, [0] + list(range(1, len(dropped_s) + 1)),
                    where="post", label="Dropped", color="red", linewidth=2)

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative events")
    ax_cum.set_title(f"Cumulative Events: {test_name}")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "cumulative_events.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: Instantaneous rate over time (scatter) ---
    fig3, ax_scatter = plt.subplots(figsize=(12, 5))

    if received_s:
        ax_scatter.scatter(received_s, [1] * len(received_s), alpha=0.3, s=10, c="gray", label="Received")
    if forwarded_s:
        ax_scatter.scatter(forwarded_s, [2] * len(forwarded_s), alpha=0.5, s=15, c="green", label="Forwarded")
    if dropped_s:
        ax_scatter.scatter(dropped_s, [3] * len(dropped_s), alpha=0.5, s=15, c="red", label="Dropped")

    ax_scatter.set_yticks([1, 2, 3])
    ax_scatter.set_yticklabels(["Received", "Forwarded", "Dropped"])
    ax_scatter.set_xlabel("Time (s)")
    ax_scatter.set_title(f"Event Timeline: {test_name}")
    ax_scatter.grid(True, alpha=0.3, axis="x")

    fig3.tight_layout()
    fig3.savefig(test_output_dir / "event_timeline.png", dpi=150)
    plt.close(fig3)

    print(f"Saved plots/data for {test_name} to: {test_output_dir}")


def test_rate_limiter_basic_functionality():
    """Basic unit test for TokenBucketRateLimiter without visualization."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = TokenBucketRateLimiter(
        name="limiter",
        downstream=sink,
        capacity=5.0,
        refill_rate=2.0,
        initial_tokens=5.0,
    )

    # Helper to dispatch forwarded events to sink
    def dispatch_events(events: list[Event]) -> None:
        for evt in events:
            if evt.target is sink:
                sink.handle_event(evt)

    # First 5 requests should be forwarded (consuming initial tokens)
    for i in range(5):
        event = RequestEvent(time=Instant.from_seconds(i * 0.1), target=rate_limiter)
        result = rate_limiter.handle_event(event)
        dispatch_events(result)

    assert rate_limiter.stats.requests_forwarded == 5
    assert rate_limiter.stats.requests_dropped == 0
    assert len(sink.handled_times) == 5

    # Next request at the exact same time as the last one should be dropped.
    # Using the same timestamp avoids any refill and makes this deterministic.
    event = RequestEvent(time=Instant.from_seconds(0.4), target=rate_limiter)
    result = rate_limiter.handle_event(event)
    dispatch_events(result)
    assert rate_limiter.stats.requests_dropped == 1

    # After 1 second, 2 tokens should have been added (refill_rate=2.0)
    event = RequestEvent(time=Instant.from_seconds(1.5), target=rate_limiter)
    result = rate_limiter.handle_event(event)
    dispatch_events(result)
    assert len(result) == 1  # Should be forwarded
    assert rate_limiter.stats.requests_forwarded == 6


def test_rate_limiter_empty_bucket():
    """Test that requests are dropped when bucket is empty."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = TokenBucketRateLimiter(
        name="limiter",
        downstream=sink,
        capacity=5.0,
        refill_rate=1.0,
        initial_tokens=0.0,  # Start empty
    )

    # Request should be dropped since no tokens available
    event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
    result = rate_limiter.handle_event(event)

    assert len(result) == 0
    assert rate_limiter.stats.requests_dropped == 1
    assert len(sink.handled_times) == 0


# =============================================================================
# Leaky Bucket Rate Limiter Tests
# =============================================================================


class LeakyBucketRequestProvider(EventProvider):
    """Generates request events targeting the leaky bucket rate limiter."""

    def __init__(self, rate_limiter: LeakyBucketRateLimiter):
        super().__init__()
        self._rate_limiter = rate_limiter

    def get_events(self, time: Instant) -> List[Event]:
        return [RequestEvent(time, self._rate_limiter)]


@pytest.mark.parametrize(
    "test_name,profile,bucket_capacity,leak_rate",
    [
        # Low constant load - should be fully forwarded (with delay)
        ("leaky_constant_within_limit", ConstantRateProfile(rate=3.0), 20, 5.0),
        # Ramp up exceeding rate limit
        ("leaky_ramp_exceeds_limit", LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=20.0), 15, 5.0),
        # Step function - sudden increase
        ("leaky_step_overload", StepRateProfile(t_switch_s=30.0, low=3.0, high=15.0), 15, 5.0),
        # High constant load - significant dropping
        ("leaky_constant_overload", ConstantRateProfile(rate=20.0), 10, 5.0),
    ],
)
def test_leaky_bucket_with_profile(
    test_name: str,
    profile: Profile,
    bucket_capacity: int,
    leak_rate: float,
    test_output_dir: Path,
):
    """Test leaky bucket rate limiter behavior with various load profiles and visualize results."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    # Create sink and rate limiter
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = LeakyBucketRateLimiter(
        name="leaky_rate_limiter",
        downstream=sink,
        capacity=bucket_capacity,
        leak_rate=leak_rate,
    )

    # Create source targeting the rate limiter
    provider = LeakyBucketRequestProvider(rate_limiter)
    arrival_provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(
        name=f"RequestSource_{test_name}",
        event_provider=provider,
        arrival_time_provider=arrival_provider,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[rate_limiter, sink],
    )
    sim.run()

    # Extract time series data
    received_s = [t.to_seconds() for t in rate_limiter.received_times]
    forwarded_s = [t.to_seconds() for t in rate_limiter.forwarded_times]
    dropped_s = [t.to_seconds() for t in rate_limiter.dropped_times]
    queue_times = [(t.to_seconds(), depth) for t, depth in rate_limiter.queue_depths]

    # Verify basic invariants
    # Note: For leaky bucket, requests can still be in queue at simulation end
    total_accounted = rate_limiter.stats.requests_forwarded + rate_limiter.stats.requests_dropped + rate_limiter.queue_depth
    assert rate_limiter.stats.requests_received == total_accounted, (
        f"received={rate_limiter.stats.requests_received} != "
        f"forwarded={rate_limiter.stats.requests_forwarded} + "
        f"dropped={rate_limiter.stats.requests_dropped} + "
        f"in_queue={rate_limiter.queue_depth}"
    )
    # Depending on simulation end-time semantics, a leak event can be processed slightly
    # past end_time, causing the limiter to count a forward while the corresponding
    # downstream event is left unprocessed in the heap. Allow a small off-by-one here.
    forwarded = rate_limiter.stats.requests_forwarded
    handled = len(sink.handled_times)
    assert handled <= forwarded
    assert (forwarded - handled) <= 1
    assert len(received_s) > 0, "Should have received some requests"

    # Print summary
    print(f"\n=== {test_name} (Leaky Bucket) ===")
    print(f"Received:  {rate_limiter.stats.requests_received}")
    print(f"Forwarded: {rate_limiter.stats.requests_forwarded}")
    print(f"Dropped:   {rate_limiter.stats.requests_dropped}")
    print(f"In queue:  {rate_limiter.queue_depth}")
    print(f"Drop rate: {rate_limiter.stats.requests_dropped / rate_limiter.stats.requests_received * 100:.1f}%")

    # --- Save CSV data ---
    _write_csv(
        test_output_dir / "events_received.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(received_s)),
    )
    _write_csv(
        test_output_dir / "events_forwarded.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(forwarded_s)),
    )
    _write_csv(
        test_output_dir / "events_dropped.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(dropped_s)),
    )
    _write_csv(
        test_output_dir / "queue_depths.csv",
        header=["time_s", "queue_depth"],
        rows=queue_times,
    )

    # --- Sample profile for plotting ---
    sample_times = _linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    # --- Plot 1: Rate comparison (profile vs realized received/forwarded/dropped) ---
    bin_s = 1.0
    recv_centers, recv_rates = _bin_counts(received_s, duration_s, bin_s)
    fwd_centers, fwd_rates = _bin_counts(forwarded_s, duration_s, bin_s)
    drop_centers, drop_rates = _bin_counts(dropped_s, duration_s, bin_s)

    fig1, (ax_rate, ax_queue) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # Rate comparison
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile (target)", linewidth=2)
    ax_rate.axhline(y=leak_rate, color="r", linestyle="--", label=f"Leak rate ({leak_rate}/s)", linewidth=2)
    if recv_centers:
        ax_rate.step(recv_centers, recv_rates, where="mid", alpha=0.7, label="Received (binned)", color="gray")
    if fwd_centers:
        ax_rate.step(fwd_centers, fwd_rates, where="mid", alpha=0.9, label="Forwarded (binned)", color="green")
    if drop_centers:
        ax_rate.step(drop_centers, drop_rates, where="mid", alpha=0.9, label="Dropped (binned)", color="red")

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title(f"Leaky Bucket Rate Limiter: {test_name}\n(capacity={bucket_capacity}, leak_rate={leak_rate}/s)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    # Queue depth over time
    if queue_times:
        queue_t, queue_d = zip(*queue_times)
        ax_queue.step(queue_t, queue_d, "m-", where="post", alpha=0.7, label="Queue depth", linewidth=1)
        ax_queue.axhline(y=bucket_capacity, color="orange", linestyle=":", label=f"Capacity ({bucket_capacity})")

    ax_queue.set_xlabel("Time (s)")
    ax_queue.set_ylabel("Queue Depth")
    ax_queue.legend(loc="upper right")
    ax_queue.grid(True, alpha=0.3)
    ax_queue.set_ylim(bottom=0)

    fig1.tight_layout()
    fig1.savefig(test_output_dir / "rate_and_queue.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Cumulative counts ---
    fig2, ax_cum = plt.subplots(figsize=(12, 5))

    # Step plots for cumulative counts
    if received_s:
        ax_cum.step([0.0] + received_s, [0] + list(range(1, len(received_s) + 1)),
                    where="post", label="Received", color="gray", alpha=0.7)
    if forwarded_s:
        ax_cum.step([0.0] + forwarded_s, [0] + list(range(1, len(forwarded_s) + 1)),
                    where="post", label="Forwarded", color="green", linewidth=2)
    if dropped_s:
        ax_cum.step([0.0] + dropped_s, [0] + list(range(1, len(dropped_s) + 1)),
                    where="post", label="Dropped", color="red", linewidth=2)

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative events")
    ax_cum.set_title(f"Cumulative Events (Leaky Bucket): {test_name}")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "cumulative_events.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: Event timeline (scatter) ---
    fig3, ax_scatter = plt.subplots(figsize=(12, 5))

    if received_s:
        ax_scatter.scatter(received_s, [1] * len(received_s), alpha=0.3, s=10, c="gray", label="Received")
    if forwarded_s:
        ax_scatter.scatter(forwarded_s, [2] * len(forwarded_s), alpha=0.5, s=15, c="green", label="Forwarded")
    if dropped_s:
        ax_scatter.scatter(dropped_s, [3] * len(dropped_s), alpha=0.5, s=15, c="red", label="Dropped")

    ax_scatter.set_yticks([1, 2, 3])
    ax_scatter.set_yticklabels(["Received", "Forwarded", "Dropped"])
    ax_scatter.set_xlabel("Time (s)")
    ax_scatter.set_title(f"Event Timeline (Leaky Bucket): {test_name}")
    ax_scatter.grid(True, alpha=0.3, axis="x")

    fig3.tight_layout()
    fig3.savefig(test_output_dir / "event_timeline.png", dpi=150)
    plt.close(fig3)

    print(f"Saved plots/data for {test_name} to: {test_output_dir}")


def test_leaky_bucket_basic_functionality():
    """Basic unit test for LeakyBucketRateLimiter without visualization."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = LeakyBucketRateLimiter(
        name="limiter",
        downstream=sink,
        capacity=5,
        leak_rate=2.0,  # 2 requests per second = 0.5s per request
    )

    # Send 3 requests at t=0 - all should be queued
    for _ in range(3):
        event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
        rate_limiter.handle_event(event)

    assert rate_limiter.stats.requests_received == 3
    assert rate_limiter.stats.requests_dropped == 0
    assert rate_limiter.queue_depth == 3
    assert len(sink.handled_times) == 0  # Nothing forwarded yet (leak events not processed)


def test_leaky_bucket_full_queue():
    """Test that requests are dropped when queue is full."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = LeakyBucketRateLimiter(
        name="limiter",
        downstream=sink,
        capacity=3,
        leak_rate=1.0,
    )

    # Fill the queue
    for _ in range(3):
        event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
        rate_limiter.handle_event(event)

    assert rate_limiter.queue_depth == 3
    assert rate_limiter.stats.requests_dropped == 0

    # Next request should be dropped
    event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
    rate_limiter.handle_event(event)

    assert rate_limiter.stats.requests_dropped == 1
    assert rate_limiter.queue_depth == 3  # Still 3 (didn't add the dropped one)


def test_leaky_bucket_vs_token_bucket_comparison(test_output_dir: Path):
    """Compare leaky bucket vs token bucket behavior with the same load profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)
    profile = LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=15.0)

    # Common parameters
    rate_limit = 5.0  # 5 requests per second

    # --- Token Bucket Setup ---
    token_sink = TimeSeriesCounterEntity("token_sink")
    token_limiter = TokenBucketRateLimiter(
        name="token_limiter",
        downstream=token_sink,
        capacity=10.0,
        refill_rate=rate_limit,
    )
    token_provider = RequestProvider(token_limiter)
    token_arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    token_source = Source(
        name="TokenSource",
        event_provider=token_provider,
        arrival_time_provider=token_arrival,
    )

    # --- Leaky Bucket Setup ---
    leaky_sink = TimeSeriesCounterEntity("leaky_sink")
    leaky_limiter = LeakyBucketRateLimiter(
        name="leaky_limiter",
        downstream=leaky_sink,
        capacity=10,
        leak_rate=rate_limit,
    )
    leaky_provider = LeakyBucketRequestProvider(leaky_limiter)
    leaky_arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    leaky_source = Source(
        name="LeakySource",
        event_provider=leaky_provider,
        arrival_time_provider=leaky_arrival,
    )

    # Run simulations
    token_sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[token_source],
        entities=[token_limiter, token_sink],
    )
    token_sim.run()

    leaky_sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[leaky_source],
        entities=[leaky_limiter, leaky_sink],
    )
    leaky_sim.run()

    # Extract data
    token_fwd_s = [t.to_seconds() for t in token_limiter.forwarded_times]
    leaky_fwd_s = [t.to_seconds() for t in leaky_limiter.forwarded_times]

    # Sample profile
    sample_times = _linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    # Bin the forwarded rates
    bin_s = 1.0
    token_centers, token_rates = _bin_counts(token_fwd_s, duration_s, bin_s)
    leaky_centers, leaky_rates = _bin_counts(leaky_fwd_s, duration_s, bin_s)

    # --- Comparison Plot ---
    fig, (ax_rate, ax_cum) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # Rate comparison
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile", linewidth=2, alpha=0.7)
    ax_rate.axhline(y=rate_limit, color="k", linestyle="--", label=f"Rate limit ({rate_limit}/s)", linewidth=2)
    if token_centers:
        ax_rate.step(token_centers, token_rates, where="mid", label="Token bucket (forwarded)", color="green", linewidth=1.5)
    if leaky_centers:
        ax_rate.step(leaky_centers, leaky_rates, where="mid", label="Leaky bucket (forwarded)", color="purple", linewidth=1.5)

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title("Token Bucket vs Leaky Bucket Rate Limiter Comparison\n(same rate limit, same load profile)")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    # Cumulative comparison
    if token_fwd_s:
        ax_cum.step([0.0] + token_fwd_s, [0] + list(range(1, len(token_fwd_s) + 1)),
                    where="post", label=f"Token bucket (total: {len(token_fwd_s)})", color="green", linewidth=2)
    if leaky_fwd_s:
        ax_cum.step([0.0] + leaky_fwd_s, [0] + list(range(1, len(leaky_fwd_s) + 1)),
                    where="post", label=f"Leaky bucket (total: {len(leaky_fwd_s)})", color="purple", linewidth=2)

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative forwarded")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(test_output_dir / "token_vs_leaky_comparison.png", dpi=150)
    plt.close(fig)

    # Print comparison
    print("\n=== Token Bucket vs Leaky Bucket Comparison ===")
    print(f"Token Bucket: forwarded={token_limiter.stats.requests_forwarded}, dropped={token_limiter.stats.requests_dropped}")
    print(f"Leaky Bucket: forwarded={leaky_limiter.stats.requests_forwarded}, dropped={leaky_limiter.stats.requests_dropped}")
    print(f"Saved comparison plot to: {test_output_dir}")


# =============================================================================
# Sliding Window Rate Limiter Tests
# =============================================================================


class SlidingWindowRequestProvider(EventProvider):
    """Generates request events targeting the sliding window rate limiter."""

    def __init__(self, rate_limiter: SlidingWindowRateLimiter):
        super().__init__()
        self._rate_limiter = rate_limiter

    def get_events(self, time: Instant) -> List[Event]:
        return [RequestEvent(time, self._rate_limiter)]


@pytest.mark.parametrize(
    "test_name,profile,window_size_seconds,max_requests",
    [
        # Low constant load - should be fully forwarded
        ("sliding_constant_within_limit", ConstantRateProfile(rate=3.0), 1.0, 5),
        # Ramp up exceeding rate limit
        ("sliding_ramp_exceeds_limit", LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=20.0), 1.0, 5),
        # Step function - sudden increase
        ("sliding_step_overload", StepRateProfile(t_switch_s=30.0, low=3.0, high=15.0), 1.0, 5),
        # High constant load - significant dropping
        ("sliding_constant_overload", ConstantRateProfile(rate=20.0), 1.0, 5),
        # Larger window with more requests allowed
        ("sliding_large_window", LinearRampProfile(t_end_s=60.0, start_rate=5.0, end_rate=30.0), 2.0, 10),
    ],
)
def test_sliding_window_with_profile(
    test_name: str,
    profile: Profile,
    window_size_seconds: float,
    max_requests: int,
    test_output_dir: Path,
):
    """Test sliding window rate limiter behavior with various load profiles and visualize results."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    # Create sink and rate limiter
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = SlidingWindowRateLimiter(
        name="sliding_window_limiter",
        downstream=sink,
        window_size_seconds=window_size_seconds,
        max_requests=max_requests,
    )

    # Effective rate limit for comparison
    effective_rate_limit = max_requests / window_size_seconds

    # Create source targeting the rate limiter
    provider = SlidingWindowRequestProvider(rate_limiter)
    arrival_provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(
        name=f"RequestSource_{test_name}",
        event_provider=provider,
        arrival_time_provider=arrival_provider,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[rate_limiter, sink],
    )
    sim.run()

    # Extract time series data
    received_s = [t.to_seconds() for t in rate_limiter.received_times]
    forwarded_s = [t.to_seconds() for t in rate_limiter.forwarded_times]
    dropped_s = [t.to_seconds() for t in rate_limiter.dropped_times]
    window_times = [(t.to_seconds(), count) for t, count in rate_limiter.window_counts]

    # Verify basic invariants
    assert rate_limiter.stats.requests_received == rate_limiter.stats.requests_forwarded + rate_limiter.stats.requests_dropped
    assert rate_limiter.stats.requests_forwarded == len(sink.handled_times)
    assert len(received_s) > 0, "Should have received some requests"

    # Print summary
    print(f"\n=== {test_name} (Sliding Window) ===")
    print(f"Window: {window_size_seconds}s, Max requests: {max_requests} (effective rate: {effective_rate_limit}/s)")
    print(f"Received:  {rate_limiter.stats.requests_received}")
    print(f"Forwarded: {rate_limiter.stats.requests_forwarded}")
    print(f"Dropped:   {rate_limiter.stats.requests_dropped}")
    print(f"Drop rate: {rate_limiter.stats.requests_dropped / rate_limiter.stats.requests_received * 100:.1f}%")

    # --- Save CSV data ---
    _write_csv(
        test_output_dir / "events_received.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(received_s)),
    )
    _write_csv(
        test_output_dir / "events_forwarded.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(forwarded_s)),
    )
    _write_csv(
        test_output_dir / "events_dropped.csv",
        header=["index", "time_s"],
        rows=((i, t) for i, t in enumerate(dropped_s)),
    )
    _write_csv(
        test_output_dir / "window_counts.csv",
        header=["time_s", "window_count"],
        rows=window_times,
    )

    # --- Sample profile for plotting ---
    sample_times = _linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    # --- Plot 1: Rate comparison (profile vs realized received/forwarded/dropped) ---
    bin_s = 1.0
    recv_centers, recv_rates = _bin_counts(received_s, duration_s, bin_s)
    fwd_centers, fwd_rates = _bin_counts(forwarded_s, duration_s, bin_s)
    drop_centers, drop_rates = _bin_counts(dropped_s, duration_s, bin_s)

    fig1, (ax_rate, ax_window) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # Rate comparison
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile (target)", linewidth=2)
    ax_rate.axhline(y=effective_rate_limit, color="r", linestyle="--",
                    label=f"Rate limit ({max_requests}/{window_size_seconds}s = {effective_rate_limit}/s)", linewidth=2)
    if recv_centers:
        ax_rate.step(recv_centers, recv_rates, where="mid", alpha=0.7, label="Received (binned)", color="gray")
    if fwd_centers:
        ax_rate.step(fwd_centers, fwd_rates, where="mid", alpha=0.9, label="Forwarded (binned)", color="green")
    if drop_centers:
        ax_rate.step(drop_centers, drop_rates, where="mid", alpha=0.9, label="Dropped (binned)", color="red")

    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title(f"Sliding Window Rate Limiter: {test_name}\n(window={window_size_seconds}s, max={max_requests})")
    ax_rate.legend(loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(bottom=0)

    # Window count over time
    if window_times:
        window_t, window_c = zip(*window_times)
        ax_window.plot(window_t, window_c, "m-", alpha=0.7, label="Window count", linewidth=1)
        ax_window.axhline(y=max_requests, color="orange", linestyle=":", label=f"Max requests ({max_requests})")

    ax_window.set_xlabel("Time (s)")
    ax_window.set_ylabel("Requests in Window")
    ax_window.legend(loc="upper right")
    ax_window.grid(True, alpha=0.3)
    ax_window.set_ylim(bottom=0)

    fig1.tight_layout()
    fig1.savefig(test_output_dir / "rate_and_window.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Cumulative counts ---
    fig2, ax_cum = plt.subplots(figsize=(12, 5))

    # Step plots for cumulative counts
    if received_s:
        ax_cum.step([0.0] + received_s, [0] + list(range(1, len(received_s) + 1)),
                    where="post", label="Received", color="gray", alpha=0.7)
    if forwarded_s:
        ax_cum.step([0.0] + forwarded_s, [0] + list(range(1, len(forwarded_s) + 1)),
                    where="post", label="Forwarded", color="green", linewidth=2)
    if dropped_s:
        ax_cum.step([0.0] + dropped_s, [0] + list(range(1, len(dropped_s) + 1)),
                    where="post", label="Dropped", color="red", linewidth=2)

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative events")
    ax_cum.set_title(f"Cumulative Events (Sliding Window): {test_name}")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "cumulative_events.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: Event timeline (scatter) ---
    fig3, ax_scatter = plt.subplots(figsize=(12, 5))

    if received_s:
        ax_scatter.scatter(received_s, [1] * len(received_s), alpha=0.3, s=10, c="gray", label="Received")
    if forwarded_s:
        ax_scatter.scatter(forwarded_s, [2] * len(forwarded_s), alpha=0.5, s=15, c="green", label="Forwarded")
    if dropped_s:
        ax_scatter.scatter(dropped_s, [3] * len(dropped_s), alpha=0.5, s=15, c="red", label="Dropped")

    ax_scatter.set_yticks([1, 2, 3])
    ax_scatter.set_yticklabels(["Received", "Forwarded", "Dropped"])
    ax_scatter.set_xlabel("Time (s)")
    ax_scatter.set_title(f"Event Timeline (Sliding Window): {test_name}")
    ax_scatter.grid(True, alpha=0.3, axis="x")

    fig3.tight_layout()
    fig3.savefig(test_output_dir / "event_timeline.png", dpi=150)
    plt.close(fig3)

    print(f"Saved plots/data for {test_name} to: {test_output_dir}")


def test_sliding_window_basic_functionality():
    """Basic unit test for SlidingWindowRateLimiter without visualization."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = SlidingWindowRateLimiter(
        name="limiter",
        downstream=sink,
        window_size_seconds=1.0,
        max_requests=3,
    )

    # Helper to dispatch forwarded events to sink
    def dispatch_events(events: list[Event]) -> None:
        for evt in events:
            if evt.target is sink:
                sink.handle_event(evt)

    # First 3 requests at t=0 should be forwarded
    for _ in range(3):
        event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
        result = rate_limiter.handle_event(event)
        dispatch_events(result)

    assert rate_limiter.stats.requests_forwarded == 3
    assert rate_limiter.stats.requests_dropped == 0
    assert len(sink.handled_times) == 3

    # 4th request at same time should be dropped (window full)
    event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
    result = rate_limiter.handle_event(event)
    dispatch_events(result)
    assert rate_limiter.stats.requests_dropped == 1

    # Request at t=0.5 should still be dropped (within window)
    event = RequestEvent(time=Instant.from_seconds(0.5), target=rate_limiter)
    result = rate_limiter.handle_event(event)
    dispatch_events(result)
    assert rate_limiter.stats.requests_dropped == 2

    # Request at t=1.1 should be allowed (old requests pruned)
    event = RequestEvent(time=Instant.from_seconds(1.1), target=rate_limiter)
    result = rate_limiter.handle_event(event)
    dispatch_events(result)
    assert len(result) == 1
    assert rate_limiter.stats.requests_forwarded == 4


def test_sliding_window_empty():
    """Test that requests are allowed when window is empty."""
    sink = TimeSeriesCounterEntity("sink")
    rate_limiter = SlidingWindowRateLimiter(
        name="limiter",
        downstream=sink,
        window_size_seconds=1.0,
        max_requests=5,
    )

    # First request should always be allowed
    event = RequestEvent(time=Instant.Epoch, target=rate_limiter)
    result = rate_limiter.handle_event(event)

    assert len(result) == 1
    assert rate_limiter.stats.requests_forwarded == 1
    assert rate_limiter.stats.requests_dropped == 0


def test_all_rate_limiters_comparison(test_output_dir: Path):
    """Compare all three rate limiter types with the same load profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)
    profile = LinearRampProfile(t_end_s=60.0, start_rate=2.0, end_rate=15.0)

    # Common parameters - all configured for ~5 requests/second limit
    rate_limit = 5.0

    # --- Token Bucket Setup ---
    token_sink = TimeSeriesCounterEntity("token_sink")
    token_limiter = TokenBucketRateLimiter(
        name="token_limiter",
        downstream=token_sink,
        capacity=10.0,
        refill_rate=rate_limit,
    )
    token_provider = RequestProvider(token_limiter)
    token_arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    token_source = Source(
        name="TokenSource",
        event_provider=token_provider,
        arrival_time_provider=token_arrival,
    )

    # --- Leaky Bucket Setup ---
    leaky_sink = TimeSeriesCounterEntity("leaky_sink")
    leaky_limiter = LeakyBucketRateLimiter(
        name="leaky_limiter",
        downstream=leaky_sink,
        capacity=10,
        leak_rate=rate_limit,
    )
    leaky_provider = LeakyBucketRequestProvider(leaky_limiter)
    leaky_arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    leaky_source = Source(
        name="LeakySource",
        event_provider=leaky_provider,
        arrival_time_provider=leaky_arrival,
    )

    # --- Sliding Window Setup ---
    sliding_sink = TimeSeriesCounterEntity("sliding_sink")
    sliding_limiter = SlidingWindowRateLimiter(
        name="sliding_limiter",
        downstream=sliding_sink,
        window_size_seconds=1.0,
        max_requests=int(rate_limit),
    )
    sliding_provider = SlidingWindowRequestProvider(sliding_limiter)
    sliding_arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    sliding_source = Source(
        name="SlidingSource",
        event_provider=sliding_provider,
        arrival_time_provider=sliding_arrival,
    )

    # Run simulations
    token_sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[token_source],
        entities=[token_limiter, token_sink],
    )
    token_sim.run()

    leaky_sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[leaky_source],
        entities=[leaky_limiter, leaky_sink],
    )
    leaky_sim.run()

    sliding_sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[sliding_source],
        entities=[sliding_limiter, sliding_sink],
    )
    sliding_sim.run()

    # Extract data
    token_fwd_s = [t.to_seconds() for t in token_limiter.forwarded_times]
    leaky_fwd_s = [t.to_seconds() for t in leaky_limiter.forwarded_times]
    sliding_fwd_s = [t.to_seconds() for t in sliding_limiter.forwarded_times]

    # Sample profile
    sample_times = _linspace(0.0, duration_s, 601)
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    # Bin the forwarded rates
    bin_s = 1.0
    token_centers, token_rates = _bin_counts(token_fwd_s, duration_s, bin_s)
    leaky_centers, leaky_rates = _bin_counts(leaky_fwd_s, duration_s, bin_s)
    sliding_centers, sliding_rates = _bin_counts(sliding_fwd_s, duration_s, bin_s)

    # --- Comparison Plot ---
    fig, (ax_rate, ax_cum) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

    # Rate comparison
    ax_rate.plot(sample_times, sample_rates, "b-", label="Load profile", linewidth=2, alpha=0.7)
    ax_rate.axhline(y=rate_limit, color="k", linestyle="--", label=f"Rate limit ({rate_limit}/s)", linewidth=2)
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

    # Cumulative comparison
    if token_fwd_s:
        ax_cum.step([0.0] + token_fwd_s, [0] + list(range(1, len(token_fwd_s) + 1)),
                    where="post", label=f"Token bucket (total: {len(token_fwd_s)})", color="green", linewidth=2)
    if leaky_fwd_s:
        ax_cum.step([0.0] + leaky_fwd_s, [0] + list(range(1, len(leaky_fwd_s) + 1)),
                    where="post", label=f"Leaky bucket (total: {len(leaky_fwd_s)})", color="purple", linewidth=2)
    if sliding_fwd_s:
        ax_cum.step([0.0] + sliding_fwd_s, [0] + list(range(1, len(sliding_fwd_s) + 1)),
                    where="post", label=f"Sliding window (total: {len(sliding_fwd_s)})", color="orange", linewidth=2)

    ax_cum.set_xlabel("Time (s)")
    ax_cum.set_ylabel("Cumulative forwarded")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(test_output_dir / "all_rate_limiters_comparison.png", dpi=150)
    plt.close(fig)

    # Print comparison
    print("\n=== All Rate Limiters Comparison ===")
    print(f"Token Bucket:   forwarded={token_limiter.stats.requests_forwarded}, dropped={token_limiter.stats.requests_dropped}")
    print(f"Leaky Bucket:   forwarded={leaky_limiter.stats.requests_forwarded}, dropped={leaky_limiter.stats.requests_dropped}")
    print(f"Sliding Window: forwarded={sliding_limiter.stats.requests_forwarded}, dropped={sliding_limiter.stats.requests_dropped}")
    print(f"Saved comparison plot to: {test_output_dir}")

