"""Integration tests for the Inductor with visualizations.

Generates PNG plots and CSV data in ``test_output/`` demonstrating
the inductor's burst suppression behaviour under various load profiles,
and comparing it against traditional rate limiters.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.rate_limiter.inductor import Inductor
from happysimulator.components.rate_limiter.policy import (
    LeakyBucketPolicy,
    TokenBucketPolicy,
)
from happysimulator.components.rate_limiter.rate_limited_entity import (
    RateLimitedEntity,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.profile import (
    ConstantRateProfile,
    LinearRampProfile,
    Profile,
)
from happysimulator.load.source import Source

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Custom profiles (local to tests)
# =============================================================================


@dataclass(frozen=True)
class StepRateProfile(Profile):
    """Rate jumps from ``low`` to ``high`` at ``step_time``."""

    low: float = 10.0
    high: float = 50.0
    step_time: float = 20.0

    def get_rate(self, time: Instant) -> float:
        return self.low if time.to_seconds() < self.step_time else self.high


@dataclass(frozen=True)
class PeriodicBurstProfile(Profile):
    """Alternates between base rate and burst rate with a fixed period."""

    base_rate: float = 10.0
    burst_rate: float = 80.0
    period: float = 10.0
    burst_duration: float = 2.0

    def get_rate(self, time: Instant) -> float:
        t_in_period = time.to_seconds() % self.period
        return self.burst_rate if t_in_period < self.burst_duration else self.base_rate


# =============================================================================
# Helpers
# =============================================================================


def _bin_events(times: list[Instant], bin_width: float) -> tuple[list[float], list[float]]:
    """Bin event times into fixed-width buckets and return (midpoints, rates)."""
    if not times:
        return [], []
    seconds = [t.to_seconds() for t in times]
    t_max = max(seconds)
    n_bins = max(1, int(t_max / bin_width) + 1)
    counts = [0] * n_bins
    for s in seconds:
        idx = min(int(s / bin_width), n_bins - 1)
        counts[idx] += 1
    midpoints = [(i + 0.5) * bin_width for i in range(n_bins)]
    rates = [c / bin_width for c in counts]
    return midpoints, rates


def _profile_line(
    profile: Profile, end_s: float, n_points: int = 300
) -> tuple[list[float], list[float]]:
    """Sample a profile to produce a line for plotting."""
    ts = [i * end_s / n_points for i in range(n_points + 1)]
    rs = [profile.get_rate(Instant.from_seconds(t)) for t in ts]
    return ts, rs


def _save_csv(path: Path, columns: dict[str, list]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = list(columns.keys())
        writer.writerow(headers)
        rows = zip(*columns.values(), strict=False)
        for row in rows:
            writer.writerow(row)


# =============================================================================
# test_inductor_burst_suppression
# =============================================================================

_PROFILES: dict[str, tuple[Profile, float]] = {
    "constant_10": (ConstantRateProfile(rate=10.0), 30.0),
    "linear_ramp_5_to_50": (
        LinearRampProfile(duration_s=60.0, start_rate=5.0, end_rate=50.0),
        60.0,
    ),
    "step_10_to_50": (StepRateProfile(low=10.0, high=50.0, step_time=20.0), 60.0),
    "periodic_burst": (
        PeriodicBurstProfile(base_rate=10.0, burst_rate=80.0, period=10.0, burst_duration=2.0),
        60.0,
    ),
}


@pytest.mark.parametrize("profile_name", list(_PROFILES.keys()))
def test_inductor_burst_suppression(profile_name: str, test_output_dir: Path):
    """Run an inductor under various profiles and visualize rate smoothing."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile, end_s = _PROFILES[profile_name]
    tau = 2.0

    sink = Sink()
    inductor = Inductor("inductor", downstream=sink, time_constant=tau)
    source = Source.with_profile(profile, target=inductor, poisson=False, name="src")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=end_s,
        sources=[source],
        entities=[inductor, sink],
    )
    sim.run()

    assert inductor.stats.received > 0
    assert inductor.stats.forwarded > 0

    # Bin events
    bin_w = 1.0
    in_t, in_r = _bin_events(inductor.received_times, bin_w)
    out_t, out_r = _bin_events(inductor.forwarded_times, bin_w)
    prof_t, prof_r = _profile_line(profile, end_s)
    est_t = [t.to_seconds() for t, _ in inductor.rate_history]
    est_r = [r for _, r in inductor.rate_history]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prof_t, prof_r, "--", color="gray", linewidth=1, label="Load profile")
    ax.plot(in_t, in_r, alpha=0.5, label="Input rate (binned)")
    ax.plot(out_t, out_r, alpha=0.7, linewidth=2, label="Output rate (binned)")
    ax.plot(est_t, est_r, ":", linewidth=2, label=f"EWMA estimate (τ={tau}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (events/s)")
    ax.set_title(f"Inductor burst suppression — {profile_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(test_output_dir / f"inductor_{profile_name}.png", dpi=150)
    plt.close(fig)

    # CSV
    _save_csv(test_output_dir / f"inductor_{profile_name}_input.csv", {"time": in_t, "rate": in_r})
    _save_csv(
        test_output_dir / f"inductor_{profile_name}_output.csv", {"time": out_t, "rate": out_r}
    )


# =============================================================================
# test_inductor_time_constants
# =============================================================================


def test_inductor_time_constants(test_output_dir: Path):
    """Compare three time constants under the periodic burst profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = PeriodicBurstProfile(base_rate=10.0, burst_rate=80.0, period=10.0, burst_duration=2.0)
    end_s = 60.0
    taus = [1.0, 3.0, 10.0]
    bin_w = 1.0

    fig, axes = plt.subplots(len(taus), 1, figsize=(12, 4 * len(taus)), sharex=True)
    prof_t, prof_r = _profile_line(profile, end_s)

    for ax, tau in zip(axes, taus, strict=False):
        sink = Sink()
        inductor = Inductor(f"ind_tau{tau}", downstream=sink, time_constant=tau)
        source = Source.with_profile(profile, target=inductor, poisson=False, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=end_s,
            sources=[source],
            entities=[inductor, sink],
        )
        sim.run()

        out_t, out_r = _bin_events(inductor.forwarded_times, bin_w)
        est_t = [t.to_seconds() for t, _ in inductor.rate_history]
        est_r = [r for _, r in inductor.rate_history]

        ax.plot(prof_t, prof_r, "--", color="gray", linewidth=1, label="Load profile")
        ax.plot(out_t, out_r, alpha=0.6, label="Output rate")
        ax.plot(est_t, est_r, ":", linewidth=2, label="EWMA estimate")
        ax.set_ylabel("Rate (events/s)")
        ax.set_title(f"τ = {tau}s")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Inductor time constant comparison — periodic bursts", fontsize=14)
    fig.tight_layout()
    fig.savefig(test_output_dir / "inductor_time_constants.png", dpi=150)
    plt.close(fig)


# =============================================================================
# test_inductor_vs_rate_limiters_comparison
# =============================================================================


def test_inductor_vs_rate_limiters_comparison(test_output_dir: Path):
    """Grand comparison under periodic bursts: traditional limiters vs inductors."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = PeriodicBurstProfile(base_rate=10.0, burst_rate=80.0, period=10.0, burst_duration=2.0)
    end_s = 60.0
    bin_w = 1.0

    configs: list[tuple[str, object]] = []

    # Token bucket
    sink_tb = Sink()
    tb = RateLimitedEntity(
        "token_bucket",
        downstream=sink_tb,
        policy=TokenBucketPolicy(capacity=20.0, refill_rate=10.0),
        queue_capacity=10_000,
    )
    configs.append(("TokenBucket (cap=20, refill=10)", tb))

    # Leaky bucket
    sink_lb = Sink()
    lb = RateLimitedEntity(
        "leaky_bucket",
        downstream=sink_lb,
        policy=LeakyBucketPolicy(leak_rate=10.0),
        queue_capacity=10_000,
    )
    configs.append(("LeakyBucket (rate=10)", lb))

    # Inductor τ=2
    sink_i2 = Sink()
    i2 = Inductor("inductor_t2", downstream=sink_i2, time_constant=2.0)
    configs.append(("Inductor (τ=2s)", i2))

    # Inductor τ=5
    sink_i5 = Sink()
    i5 = Inductor("inductor_t5", downstream=sink_i5, time_constant=5.0)
    configs.append(("Inductor (τ=5s)", i5))

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 4 * len(configs)), sharex=True)
    prof_t, prof_r = _profile_line(profile, end_s)

    for ax, (label, entity) in zip(axes, configs, strict=False):
        source = Source.with_profile(profile, target=entity, poisson=False, name=f"src_{label}")

        entities_list = [entity]
        # Add the downstream sink
        if hasattr(entity, "_downstream"):
            entities_list.append(entity._downstream)
        elif hasattr(entity, "downstream"):
            entities_list.append(entity.downstream)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=end_s,
            sources=[source],
            entities=entities_list,
        )
        sim.run()

        # Get forwarded times
        if hasattr(entity, "forwarded_times"):
            fwd_times = entity.forwarded_times
        else:
            fwd_times = []

        # Get received times
        if hasattr(entity, "received_times"):
            recv_times = entity.received_times
        else:
            recv_times = []

        in_t, in_r = _bin_events(recv_times, bin_w)
        out_t, out_r = _bin_events(fwd_times, bin_w)

        ax.plot(prof_t, prof_r, "--", color="gray", linewidth=1, label="Load profile")
        ax.plot(in_t, in_r, alpha=0.3, label="Input")
        ax.plot(out_t, out_r, linewidth=2, label="Output")

        # Show EWMA estimate for inductors
        if isinstance(entity, Inductor) and entity.rate_history:
            est_t = [t.to_seconds() for t, _ in entity.rate_history]
            est_r = [r for _, r in entity.rate_history]
            ax.plot(est_t, est_r, ":", linewidth=2, label="EWMA est.")

        stats = entity.stats
        ax.set_ylabel("Rate (events/s)")
        ax.set_title(
            f"{label}  [fwd={stats.forwarded}, drop={stats.dropped}, q={entity.queue_depth}]"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Rate limiter comparison — periodic bursts", fontsize=14)
    fig.tight_layout()
    fig.savefig(test_output_dir / "comparison_periodic_burst.png", dpi=150)
    plt.close(fig)


# =============================================================================
# test_inductor_vs_rate_limiters_ramp
# =============================================================================


def test_inductor_vs_rate_limiters_ramp(test_output_dir: Path):
    """Comparison under linear ramp: traditional limiters flatline, inductor follows."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = LinearRampProfile(duration_s=60.0, start_rate=5.0, end_rate=50.0)
    end_s = 60.0
    bin_w = 1.0

    configs: list[tuple[str, object]] = []

    # Token bucket
    sink_tb = Sink()
    tb = RateLimitedEntity(
        "token_bucket",
        downstream=sink_tb,
        policy=TokenBucketPolicy(capacity=20.0, refill_rate=10.0),
        queue_capacity=10_000,
    )
    configs.append(("TokenBucket (cap=20, refill=10)", tb))

    # Leaky bucket
    sink_lb = Sink()
    lb = RateLimitedEntity(
        "leaky_bucket",
        downstream=sink_lb,
        policy=LeakyBucketPolicy(leak_rate=10.0),
        queue_capacity=10_000,
    )
    configs.append(("LeakyBucket (rate=10)", lb))

    # Inductor τ=2
    sink_i2 = Sink()
    i2 = Inductor("inductor_t2", downstream=sink_i2, time_constant=2.0)
    configs.append(("Inductor (τ=2s)", i2))

    # Inductor τ=5
    sink_i5 = Sink()
    i5 = Inductor("inductor_t5", downstream=sink_i5, time_constant=5.0)
    configs.append(("Inductor (τ=5s)", i5))

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 4 * len(configs)), sharex=True)
    prof_t, prof_r = _profile_line(profile, end_s)

    for ax, (label, entity) in zip(axes, configs, strict=False):
        source = Source.with_profile(profile, target=entity, poisson=False, name=f"src_{label}")

        entities_list = [entity]
        if hasattr(entity, "_downstream"):
            entities_list.append(entity._downstream)
        elif hasattr(entity, "downstream"):
            entities_list.append(entity.downstream)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=end_s,
            sources=[source],
            entities=entities_list,
        )
        sim.run()

        if hasattr(entity, "forwarded_times"):
            fwd_times = entity.forwarded_times
        else:
            fwd_times = []

        if hasattr(entity, "received_times"):
            recv_times = entity.received_times
        else:
            recv_times = []

        in_t, in_r = _bin_events(recv_times, bin_w)
        out_t, out_r = _bin_events(fwd_times, bin_w)

        ax.plot(prof_t, prof_r, "--", color="gray", linewidth=1, label="Load profile")
        ax.plot(in_t, in_r, alpha=0.3, label="Input")
        ax.plot(out_t, out_r, linewidth=2, label="Output")

        if isinstance(entity, Inductor) and entity.rate_history:
            est_t = [t.to_seconds() for t, _ in entity.rate_history]
            est_r = [r for _, r in entity.rate_history]
            ax.plot(est_t, est_r, ":", linewidth=2, label="EWMA est.")

        stats = entity.stats
        ax.set_ylabel("Rate (events/s)")
        ax.set_title(
            f"{label}  [fwd={stats.forwarded}, drop={stats.dropped}, q={entity.queue_depth}]"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Rate limiter comparison — linear ramp 5→50 req/s", fontsize=14)
    fig.tight_layout()
    fig.savefig(test_output_dir / "comparison_linear_ramp.png", dpi=150)
    plt.close(fig)
