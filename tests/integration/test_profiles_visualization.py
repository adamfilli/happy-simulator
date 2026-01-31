"""Tests that generate plots for different load profiles.

These tests are intentionally "visual":
- Run a deterministic simulation (ConstantArrivalTimeProvider integrates the rate curve).
- Record a time series of handled events.
- Save raw data (CSV) + matplotlib charts under test_output/.

Run:
    pytest tests/test_profiles_visualization.py -v

Output:
    test_output/test_profiles_visualization/<test_name>/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TimeSeriesCounterEntity(Entity):
    """Entity that records when it handled events."""

    def __init__(self, name: str):
        super().__init__(name)
        self.handled_times: list[Instant] = []

    def handle_event(self, event: Event):
        self.handled_times.append(event.time)
        return []


class PingEvent(Event):
    def __init__(self, time: Instant, counter: TimeSeriesCounterEntity):
        super().__init__(time=time, event_type="Ping", target=counter)


class PingProvider(EventProvider):
    def __init__(self, counter: TimeSeriesCounterEntity):
        super().__init__()
        self._counter = counter

    def get_events(self, time: Instant) -> List[Event]:
        return [PingEvent(time, self._counter)]


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    rate: float

    def get_rate(self, time: Instant) -> float:  # noqa: ARG002
        return float(self.rate)


@dataclass(frozen=True)
class StepRateProfile(Profile):
    t_switch_s: float
    low: float
    high: float

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        return float(self.low if t < self.t_switch_s else self.high)


@dataclass(frozen=True)
class LinearRampProfile(Profile):
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
class SinusoidProfile(Profile):
    base: float
    amplitude: float
    period_s: float
    min_rate: float = 0.001

    def get_rate(self, time: Instant) -> float:
        import math

        t = time.to_seconds()
        if self.period_s <= 0:
            return max(float(self.min_rate), float(self.base))
        raw = self.base + self.amplitude * math.sin(2.0 * math.pi * (t / self.period_s))
        return max(float(self.min_rate), float(raw))


def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n <= 1:
        return [float(start)]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _bin_counts(times_s: list[float], duration_s: float, bin_s: float) -> tuple[list[float], list[float]]:
    """Returns (bin_centers, events_per_second)."""
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


@pytest.mark.parametrize(
    "profile_name,profile",
    [
        ("constant_1hz", ConstantRateProfile(rate=1.0)),
        ("step_1_to_3", StepRateProfile(t_switch_s=30.0, low=1.0, high=3.0)),
        ("ramp_0.5_to_3", LinearRampProfile(t_end_s=60.0, start_rate=0.5, end_rate=3.0)),
        ("sinusoid", SinusoidProfile(base=2.0, amplitude=1.0, period_s=20.0, min_rate=0.2)),
    ],
)
def test_profile_generates_expected_shape(profile_name: str, profile: Profile, test_output_dir: Path):
    """Generate event time series and plots for a given profile."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration_s = 60.0
    end_time = Instant.from_seconds(duration_s)

    counter = TimeSeriesCounterEntity("counter")
    provider = PingProvider(counter)
    arrival_time_provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

    source = Source(name=f"PingSource_{profile_name}", event_provider=provider, arrival_time_provider=arrival_time_provider)

    sim = Simulation(start_time=Instant.Epoch, end_time=end_time, sources=[source], entities=[counter])
    sim.run()

    times_s = [t.to_seconds() for t in counter.handled_times]

    # Basic sanity checks (test should pass deterministically)
    assert times_s == sorted(times_s)
    assert all(0.0 <= t <= duration_s + 1e-6 for t in times_s)
    assert len(times_s) > 0

    # --- Save raw time series
    _write_csv(
        test_output_dir / "events.csv",
        header=["index", "time_s", "dt_s"],
        rows=(
            (i, t, (t - times_s[i - 1]) if i > 0 else t)
            for i, t in enumerate(times_s)
        ),
    )

    # --- Save expected profile samples
    sample_times = _linspace(0.0, duration_s, 601)  # 0.1s resolution
    sample_rates = [profile.get_rate(Instant.from_seconds(t)) for t in sample_times]

    _write_csv(
        test_output_dir / "profile_rate.csv",
        header=["time_s", "rate_per_s"],
        rows=((t, r) for t, r in zip(sample_times, sample_rates, strict=False)),
    )

    # --- Plot 1: profile rate + cumulative handled events
    fig, (ax_rate, ax_count) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7))

    ax_rate.plot(sample_times, sample_rates)
    ax_rate.set_ylabel("rate (events / s)")
    ax_rate.set_title(f"Profile: {profile_name}")
    ax_rate.grid(True)

    if times_s:
        # Step plot of cumulative count; include t=0 with count=0
        ax_count.step([0.0] + times_s, [0] + list(range(1, len(times_s) + 1)), where="post")
    ax_count.set_xlabel("time (s)")
    ax_count.set_ylabel("cumulative events")
    ax_count.grid(True)

    fig.tight_layout()
    fig.savefig(test_output_dir / "profile_and_cumulative.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: binned realized rate vs expected profile
    bin_s = 1.0
    centers, realized_rates = _bin_counts(times_s, duration_s=duration_s, bin_s=bin_s)

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sample_times, sample_rates, label="expected rate", linewidth=2)
    if centers:
        ax.step(centers, realized_rates, where="mid", label="realized (binned)")
    ax.set_title(f"Rate comparison (bin={bin_s}s): {profile_name}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("events / s")
    ax.grid(True)
    ax.legend()

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "expected_vs_realized_rate.png", dpi=150)
    plt.close(fig2)

    # Make it easy to find output from test logs
    print(f"\nSaved plots/data for {profile_name} to: {test_output_dir}")
