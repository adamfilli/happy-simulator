"""Rich simulation result wrappers with comparison and AI-friendly output.

Works with any simulation — not tied to a specific builder pattern.
Wraps SimulationSummary + SimulationAnalysis + raw metric Data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from happysimulator.analysis.report import SimulationAnalysis, analyze

if TYPE_CHECKING:
    from happysimulator.instrumentation.data import Data
    from happysimulator.instrumentation.summary import SimulationSummary


@dataclass
class MetricDiff:
    """Difference between a single metric across two simulation runs."""

    name: str
    mean_a: float
    mean_b: float
    mean_change_pct: float
    p99_a: float
    p99_b: float
    p99_change_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "mean_change_pct": round(self.mean_change_pct, 1),
            "p99_a": round(self.p99_a, 6),
            "p99_b": round(self.p99_b, 6),
            "p99_change_pct": round(self.p99_change_pct, 1),
        }


@dataclass
class SimulationComparison:
    """Side-by-side comparison of two simulation runs."""

    result_a: SimulationResult
    result_b: SimulationResult
    metric_diffs: dict[str, MetricDiff] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_a": self.result_a.to_dict(),
            "result_b": self.result_b.to_dict(),
            "metric_diffs": {name: diff.to_dict() for name, diff in self.metric_diffs.items()},
        }

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Format comparison as structured text for AI consumption."""
        lines: list[str] = []
        lines.append("## Simulation Comparison")
        lines.append("")
        lines.append("| Metric | Run A | Run B | Change |")
        lines.append("|--------|-------|-------|--------|")

        for name, diff in self.metric_diffs.items():
            sign = "+" if diff.mean_change_pct >= 0 else ""
            lines.append(
                f"| {name} (mean) | {diff.mean_a:.4f}s | {diff.mean_b:.4f}s "
                f"| {sign}{diff.mean_change_pct:.1f}% |"
            )
            sign99 = "+" if diff.p99_change_pct >= 0 else ""
            lines.append(
                f"| {name} (p99) | {diff.p99_a:.4f}s | {diff.p99_b:.4f}s "
                f"| {sign99}{diff.p99_change_pct:.1f}% |"
            )

        # Summary and throughput
        a_summary = self.result_a.summary
        b_summary = self.result_b.summary
        a_eps = a_summary.events_per_second
        b_eps = b_summary.events_per_second
        if a_eps > 0:
            eps_change = ((b_eps - a_eps) / a_eps) * 100
            sign_t = "+" if eps_change >= 0 else ""
            lines.append(
                f"| throughput | {a_eps:.1f}/s | {b_eps:.1f}/s | {sign_t}{eps_change:.1f}% |"
            )

        lines.append("")

        # Key differences
        key_diffs: list[str] = []
        for name, diff in self.metric_diffs.items():
            if abs(diff.p99_change_pct) > 10:
                direction = "lower" if diff.p99_change_pct < 0 else "higher"
                key_diffs.append(
                    f"- Run B has {abs(diff.p99_change_pct):.0f}% {direction} "
                    f"tail latency (p99) for {name}"
                )
            if abs(diff.mean_change_pct) > 20:
                direction = "lower" if diff.mean_change_pct < 0 else "higher"
                key_diffs.append(
                    f"- {name} mean is {abs(diff.mean_change_pct):.0f}% {direction} in Run B"
                )

        if key_diffs:
            lines.append("## Key Differences")
            lines.extend(key_diffs)
            lines.append("")

        return "\n".join(lines)


@dataclass
class SimulationResult:
    """Rich simulation result with analysis, comparison, and AI-friendly output.

    Works with any simulation — not tied to a specific builder pattern.
    """

    summary: SimulationSummary
    analysis: SimulationAnalysis
    latency: Data | None = None
    queue_depth: dict[str, Data] = field(default_factory=dict)
    throughput: Data | None = None
    recommendations: list[Any] = field(default_factory=list)

    @classmethod
    def from_run(
        cls,
        summary: SimulationSummary,
        *,
        latency: Data | None = None,
        queue_depth: dict[str, Data] | None = None,
        throughput: Data | None = None,
        **named_metrics: Data,
    ) -> SimulationResult:
        """Create a SimulationResult by running analyze() automatically.

        Args:
            summary: SimulationSummary from Simulation.run().
            latency: Latency time-series data (e.g., from LatencyTracker.data).
            queue_depth: Queue depth data keyed by server name.
            throughput: Throughput time-series data.
            **named_metrics: Additional named Data objects to analyze.
        """
        # Pick first queue_depth for the analyzer (it expects a single Data)
        qd = queue_depth or {}
        first_qd = next(iter(qd.values()), None) if qd else None

        analysis = analyze(
            summary,
            latency=latency,
            queue_depth=first_qd,
            throughput=throughput,
            **named_metrics,
        )

        result = cls(
            summary=summary,
            analysis=analysis,
            latency=latency,
            queue_depth=qd,
            throughput=throughput,
        )

        # Generate recommendations
        from happysimulator.ai.insights import generate_recommendations

        result.recommendations = generate_recommendations(result)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Structured data for programmatic access."""
        result = self.analysis.to_dict()
        if self.recommendations:
            result["recommendations"] = [r.to_dict() for r in self.recommendations]
        return result

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Generate AI-optimized summary text.

        Includes analysis output plus recommendations.
        """
        parts: list[str] = [self.analysis.to_prompt_context(max_tokens=max_tokens)]

        if self.recommendations:
            rec_lines = ["## Recommendations"]
            for rec in self.recommendations:
                rec_lines.append(f"- [{rec.confidence}] **{rec.category}**: {rec.description}")
                if rec.suggested_change:
                    rec_lines.append(f"  Suggested: {rec.suggested_change}")
            rec_lines.append("")
            parts.append("\n".join(rec_lines))

        return "\n".join(parts)

    def compare(self, other: SimulationResult) -> SimulationComparison:
        """Compare this result with another."""
        diffs: dict[str, MetricDiff] = {}

        # Compare latency if both have it
        if (
            self.latency is not None
            and other.latency is not None
            and self.latency.count() > 0
            and other.latency.count() > 0
        ):
            mean_a = self.latency.mean()
            mean_b = other.latency.mean()
            p99_a = self.latency.percentile(0.99)
            p99_b = other.latency.percentile(0.99)
            diffs["latency"] = MetricDiff(
                name="latency",
                mean_a=mean_a,
                mean_b=mean_b,
                mean_change_pct=_pct_change(mean_a, mean_b),
                p99_a=p99_a,
                p99_b=p99_b,
                p99_change_pct=_pct_change(p99_a, p99_b),
            )

        # Compare queue depths
        common_keys = set(self.queue_depth.keys()) & set(other.queue_depth.keys())
        for key in sorted(common_keys):
            data_a = self.queue_depth[key]
            data_b = other.queue_depth[key]
            if data_a.count() > 0 and data_b.count() > 0:
                mean_a = data_a.mean()
                mean_b = data_b.mean()
                p99_a = data_a.percentile(0.99)
                p99_b = data_b.percentile(0.99)
                diffs[f"queue_depth_{key}"] = MetricDiff(
                    name=f"queue_depth_{key}",
                    mean_a=mean_a,
                    mean_b=mean_b,
                    mean_change_pct=_pct_change(mean_a, mean_b),
                    p99_a=p99_a,
                    p99_b=p99_b,
                    p99_change_pct=_pct_change(p99_a, p99_b),
                )

        return SimulationComparison(
            result_a=self,
            result_b=other,
            metric_diffs=diffs,
        )


@dataclass
class SweepResult:
    """Results from a parametric sweep across multiple simulation runs."""

    parameter_name: str
    parameter_values: list[Any]
    results: list[SimulationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "results": [r.to_dict() for r in self.results],
        }

    def best_by(self, metric: str = "latency", stat: str = "p99") -> SimulationResult:
        """Find the result with the best (lowest) value for a metric.

        Args:
            metric: "latency" or a queue_depth key.
            stat: "p99", "mean", "p50", etc.

        Returns:
            The SimulationResult with the lowest value for the given metric+stat.
        """

        def _get_value(result: SimulationResult) -> float:
            if metric == "latency" and result.latency is not None:
                data = result.latency
            elif metric in result.queue_depth:
                data = result.queue_depth[metric]
            else:
                return float("inf")
            if data.count() == 0:
                return float("inf")
            if stat == "p99":
                return data.percentile(0.99)
            if stat == "p50":
                return data.percentile(0.50)
            if stat == "mean":
                return data.mean()
            return data.percentile(0.99)

        return min(self.results, key=_get_value)

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Format sweep results as a table for AI consumption."""
        lines: list[str] = []
        lines.append(f"## Parameter Sweep: {self.parameter_name}")
        lines.append("")

        # Build header
        header = f"| {self.parameter_name} | latency_mean | latency_p99 |"
        separator = "|" + "---|" * 3

        # Add queue depth columns if present
        qd_keys: list[str] = []
        for r in self.results:
            for key in r.queue_depth:
                if key not in qd_keys:
                    qd_keys.append(key)
        for key in qd_keys:
            header += f" qd_{key}_mean |"
            separator += "---|"

        header += " throughput |"
        separator += "---|"

        lines.append(header)
        lines.append(separator)

        # Track for saturation detection
        prev_latency_p99 = None

        for val, result in zip(self.parameter_values, self.results, strict=False):
            row = f"| {val} |"

            if result.latency is not None and result.latency.count() > 0:
                lat_mean = result.latency.mean()
                lat_p99 = result.latency.percentile(0.99)
                row += f" {lat_mean:.4f}s | {lat_p99:.4f}s |"

                # Detect saturation
                if prev_latency_p99 is not None and lat_p99 > prev_latency_p99 * 5:
                    row += "  <-- saturation"
                prev_latency_p99 = lat_p99
            else:
                row += " - | - |"

            for key in qd_keys:
                if key in result.queue_depth and result.queue_depth[key].count() > 0:
                    qd_mean = result.queue_depth[key].mean()
                    row += f" {qd_mean:.1f} |"
                else:
                    row += " - |"

            row += f" {result.summary.events_per_second:.1f}/s |"
            lines.append(row)

        lines.append("")

        # Observations
        observations: list[str] = []
        if len(self.results) >= 2:
            latencies = []
            for r in self.results:
                if r.latency is not None and r.latency.count() > 0:
                    latencies.append(r.latency.percentile(0.99))
                else:
                    latencies.append(None)

            # Detect saturation point
            for i in range(1, len(latencies)):
                if (
                    latencies[i] is not None
                    and latencies[i - 1] is not None
                    and latencies[i] > latencies[i - 1] * 5
                ):
                    observations.append(
                        f"- System saturates between {self.parameter_name}="
                        f"{self.parameter_values[i - 1]} and "
                        f"{self.parameter_name}={self.parameter_values[i]}"
                    )
                    ratio = latencies[i] / latencies[i - 1] if latencies[i - 1] > 0 else 0
                    observations.append(
                        f"- At {self.parameter_name}={self.parameter_values[i]}, "
                        f"p99 latency increases {ratio:.0f}x"
                    )
                    break

        if observations:
            lines.append("## Observations")
            lines.extend(observations)
            lines.append("")

        return "\n".join(lines)


def _pct_change(a: float, b: float) -> float:
    """Calculate percentage change from a to b."""
    if a == 0:
        return 0.0 if b == 0 else float("inf")
    return ((b - a) / abs(a)) * 100
