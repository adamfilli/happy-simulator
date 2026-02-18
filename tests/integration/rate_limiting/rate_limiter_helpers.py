"""Shared utilities for rate limiter tests.

Profiles, binning helpers, and CSV writer used across
token bucket, leaky bucket, and sliding window test modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.temporal import Instant
from happysimulator.load.profile import Profile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# --- Profile Definitions ---


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant arrival rate profile."""

    rate: float

    def get_rate(self, time: Instant) -> float:
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


def linspace(start: float, stop: float, n: int) -> list[float]:
    """Generate n evenly spaced values from start to stop."""
    if n <= 1:
        return [float(start)]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    """Write data to a CSV file."""
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def bin_counts(
    times_s: list[float], duration_s: float, bin_s: float
) -> tuple[list[float], list[float]]:
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
