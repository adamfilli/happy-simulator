"""Rules-based recommendations engine for simulation results.

Analyzes SimulationResult data to generate actionable suggestions
about capacity, architecture, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from happysimulator.ai.result import SimulationResult


@dataclass
class Recommendation:
    """An actionable suggestion based on simulation analysis."""

    category: str  # "capacity", "architecture", "configuration"
    description: str
    confidence: str  # "high", "medium", "low"
    suggested_change: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "suggested_change": self.suggested_change,
        }


def generate_recommendations(result: SimulationResult) -> list[Recommendation]:
    """Analyze results and suggest improvements.

    Rules:
    - Queue saturation: queue depth growing over time -> more capacity
    - Underutilization: low utilization -> fewer servers
    - Tail latency: high p99/p50 ratio -> investigate variance
    - Phase transitions: degraded phases detected -> capacity planning
    """
    recommendations: list[Recommendation] = []

    _check_queue_saturation(result, recommendations)
    _check_tail_latency(result, recommendations)
    _check_phase_transitions(result, recommendations)
    _check_underutilization(result, recommendations)

    return recommendations


def _check_queue_saturation(
    result: SimulationResult, recommendations: list[Recommendation]
) -> None:
    """Detect growing queue depth indicating saturation."""
    for name, data in result.queue_depth.items():
        if data.count() < 20:
            continue

        # Compare first 20% vs last 20% of the simulation
        times = data.times()
        if not times:
            continue
        duration = times[-1] - times[0]
        if duration <= 0:
            continue

        early_end = times[0] + duration * 0.2
        late_start = times[0] + duration * 0.8

        early = data.between(times[0], early_end)
        late = data.between(late_start, times[-1])

        if early.count() == 0 or late.count() == 0:
            continue

        early_mean = early.mean()
        late_mean = late.mean()

        # Queue depth more than doubled and is meaningfully non-zero
        if late_mean > max(early_mean * 2, 5):
            recommendations.append(
                Recommendation(
                    category="capacity",
                    description=(
                        f"Queue depth for '{name}' is growing over time "
                        f"(early mean: {early_mean:.1f}, late mean: {late_mean:.1f}), "
                        f"indicating the system is saturated."
                    ),
                    confidence="high",
                    suggested_change=(
                        "Increase service capacity (more servers or higher concurrency) "
                        "or reduce arrival rate."
                    ),
                )
            )


def _check_tail_latency(result: SimulationResult, recommendations: list[Recommendation]) -> None:
    """Detect high tail latency ratio."""
    if result.latency is None or result.latency.count() < 20:
        return

    p50 = result.latency.percentile(0.50)
    p99 = result.latency.percentile(0.99)

    if p50 <= 0:
        return

    ratio = p99 / p50
    if ratio > 10:
        recommendations.append(
            Recommendation(
                category="configuration",
                description=(
                    f"Tail latency is very high relative to median: "
                    f"p99={p99:.4f}s is {ratio:.0f}x the p50={p50:.4f}s. "
                    f"This suggests high variance or occasional queueing delays."
                ),
                confidence="medium",
                suggested_change=(
                    "Investigate sources of variance: service time distribution, "
                    "queue buildup during bursts, or resource contention. "
                    "Consider adding concurrency or using a less variable service time."
                ),
            )
        )


def _check_phase_transitions(
    result: SimulationResult, recommendations: list[Recommendation]
) -> None:
    """Detect degraded phases from the analysis."""
    if not result.analysis.phases:
        return

    for metric_name, phases in result.analysis.phases.items():
        for phase in phases:
            if phase.label in ("degraded", "overloaded"):
                recommendations.append(
                    Recommendation(
                        category="capacity",
                        description=(
                            f"Metric '{metric_name}' entered a '{phase.label}' phase "
                            f"from t={phase.start_s:.1f}s to t={phase.end_s:.1f}s "
                            f"(mean={phase.mean:.4f})."
                        ),
                        confidence="high",
                        suggested_change=(
                            f"Plan capacity to handle load levels that cause the "
                            f"transition around t={phase.start_s:.1f}s. "
                            f"Consider auto-scaling or load shedding."
                        ),
                    )
                )
                break  # One recommendation per metric


def _check_underutilization(
    result: SimulationResult, recommendations: list[Recommendation]
) -> None:
    """Detect underutilized servers from queue depth data."""
    for name, data in result.queue_depth.items():
        if data.count() < 20:
            continue

        mean_depth = data.mean()
        max_depth = data.max()

        # If queue is essentially always empty, system may be overprovisioned
        if mean_depth < 0.5 and max_depth < 3:
            recommendations.append(
                Recommendation(
                    category="capacity",
                    description=(
                        f"Queue '{name}' is nearly always empty "
                        f"(mean depth: {mean_depth:.2f}, max: {max_depth:.1f}), "
                        f"suggesting the system is overprovisioned."
                    ),
                    confidence="low",
                    suggested_change=(
                        "Consider reducing server count or concurrency to save resources, "
                        "unless headroom is intentional for burst handling."
                    ),
                )
            )
