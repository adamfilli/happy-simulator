"""Phase detection for time-series simulation data.

Identifies regime changes where behavior shifts significantly,
such as transitions from stable to degraded states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from happysimulator.instrumentation.data import Data


@dataclass
class Phase:
    """A detected phase/regime in the time series."""

    start_s: float
    end_s: float
    mean: float
    std: float
    label: str  # "stable", "degraded", "recovering", "overloaded"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "mean": self.mean,
            "std": self.std,
            "label": self.label,
        }


def detect_phases(
    data: Data,
    *,
    window_s: float = 5.0,
    threshold: float = 2.0,
) -> list[Phase]:
    """Detect phases where behavior changes significantly.

    Uses a simple change-point detection approach: bucket the data into
    windows, compute per-window statistics, and identify transitions
    where the mean shifts by more than `threshold` standard deviations
    from the previous phase.

    Args:
        data: Time-series data to analyze.
        window_s: Window size for bucketing.
        threshold: Number of standard deviations for a phase transition.

    Returns:
        Ordered list of Phase objects with start/end times and summary
        statistics. Returns empty list if fewer than 2 windows of data.
    """
    if data.count() < 2:
        return []

    bucketed = data.bucket(window_s)
    if len(bucketed) < 1:
        return []

    times = bucketed.times()
    means = bucketed.means()

    if len(times) < 2:
        label = _classify_phase(means[0], means[0], 0.0)
        return [
            Phase(
                start_s=times[0],
                end_s=times[0] + window_s,
                mean=means[0],
                std=0.0,
                label=label,
            )
        ]

    # Group consecutive windows into phases using change-point detection
    phases: list[Phase] = []
    phase_start_idx = 0
    phase_values: list[float] = [means[0]]

    for i in range(1, len(means)):
        phase_mean = sum(phase_values) / len(phase_values)
        phase_std = _pstdev(phase_values) if len(phase_values) > 1 else 0.0

        # Check if this window represents a significant shift
        effective_std = max(phase_std, abs(phase_mean) * 0.1) if phase_mean != 0 else 1.0
        deviation = abs(means[i] - phase_mean) / effective_std

        if deviation > threshold:
            # End current phase, start new one
            overall_mean = sum(phase_values) / len(phase_values)
            overall_std = _pstdev(phase_values) if len(phase_values) > 1 else 0.0
            baseline = means[0]  # Use first window as baseline
            label = _classify_phase(overall_mean, baseline, overall_std)

            phases.append(
                Phase(
                    start_s=times[phase_start_idx],
                    end_s=times[i],
                    mean=overall_mean,
                    std=overall_std,
                    label=label,
                )
            )
            phase_start_idx = i
            phase_values = [means[i]]
        else:
            phase_values.append(means[i])

    # Close the last phase
    overall_mean = sum(phase_values) / len(phase_values)
    overall_std = _pstdev(phase_values) if len(phase_values) > 1 else 0.0
    baseline = means[0]
    label = _classify_phase(overall_mean, baseline, overall_std)

    phases.append(
        Phase(
            start_s=times[phase_start_idx],
            end_s=times[-1] + window_s,
            mean=overall_mean,
            std=overall_std,
            label=label,
        )
    )

    return phases


def _classify_phase(mean: float, baseline: float, std: float) -> str:
    """Classify a phase based on its mean relative to baseline."""
    if baseline == 0:
        if mean == 0:
            return "stable"
        return "degraded"

    ratio = mean / baseline
    if ratio < 1.5:
        return "stable"
    if ratio < 3.0:
        return "degraded"
    return "overloaded"


def _pstdev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5
