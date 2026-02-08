"""Structured simulation analysis optimized for AI/LLM consumption.

The analyze() function combines phase detection, anomaly detection, and
metric summaries into a single SimulationAnalysis result. The
to_prompt_context() method formats the analysis for inclusion in LLM prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.summary import SimulationSummary
from happysimulator.instrumentation.recorder import InMemoryTraceRecorder
from happysimulator.analysis.phases import Phase, detect_phases


@dataclass
class MetricSummary:
    """Pre-computed statistics for a named metric."""
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    by_phase: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "p50": round(self.p50, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
        }
        if self.by_phase:
            result["by_phase"] = self.by_phase
        return result


@dataclass
class Anomaly:
    """A detected anomaly with context for reasoning."""
    time_s: float
    metric: str
    description: str
    severity: str  # "info", "warning", "critical"
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "metric": self.metric,
            "description": self.description,
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class CausalChain:
    """A chain of causally-related observations."""
    trigger_description: str
    effects: list[str]
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger_description,
            "effects": self.effects,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class SimulationAnalysis:
    """Structured analysis of a simulation run, designed for AI/LLM consumption."""
    summary: SimulationSummary
    phases: dict[str, list[Phase]] = field(default_factory=dict)
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    anomalies: list[Anomaly] = field(default_factory=list)
    causal_chains: list[CausalChain] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "phases": {
                name: [p.to_dict() for p in phases]
                for name, phases in self.phases.items()
            },
            "metrics": {
                name: ms.to_dict() for name, ms in self.metrics.items()
            },
            "anomalies": [a.to_dict() for a in self.anomalies],
            "causal_chains": [c.to_dict() for c in self.causal_chains],
        }

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Format as structured text for inclusion in an LLM prompt.

        Automatically truncates/summarizes to fit within token budget.
        Prioritizes anomalies and phase transitions over steady-state data.

        Args:
            max_tokens: Approximate token budget (1 token ~= 4 chars).
        """
        max_chars = max_tokens * 4
        sections: list[str] = []

        # Summary (always included)
        sections.append("## Simulation Summary")
        sections.append(f"- Duration: {self.summary.duration_s:.2f}s")
        sections.append(f"- Events processed: {self.summary.total_events_processed}")
        sections.append(f"- Events/sec: {self.summary.events_per_second:.1f}")
        sections.append(f"- Wall clock: {self.summary.wall_clock_seconds:.3f}s")
        sections.append("")

        # Anomalies (high priority)
        if self.anomalies:
            sections.append("## Anomalies Detected")
            for a in self.anomalies:
                sections.append(f"- [{a.severity}] t={a.time_s:.1f}s: {a.description}")
            sections.append("")

        # Causal chains (high priority)
        if self.causal_chains:
            sections.append("## Causal Chains")
            for chain in self.causal_chains:
                sections.append(f"- Trigger: {chain.trigger_description}")
                for effect in chain.effects:
                    sections.append(f"  -> {effect}")
                sections.append(f"  Duration: {chain.duration_s:.1f}s")
            sections.append("")

        # Phase analysis
        if self.phases:
            sections.append("## Phase Analysis")
            for metric_name, phases in self.phases.items():
                sections.append(f"### {metric_name}")
                for p in phases:
                    sections.append(
                        f"- [{p.label}] {p.start_s:.1f}s-{p.end_s:.1f}s: "
                        f"mean={p.mean:.4f}, std={p.std:.4f}"
                    )
            sections.append("")

        # Metric summaries (lower priority, truncate if needed)
        if self.metrics:
            metric_section = ["## Metrics"]
            for name, ms in self.metrics.items():
                metric_section.append(
                    f"- {name}: mean={ms.mean:.4f}, p50={ms.p50:.4f}, "
                    f"p95={ms.p95:.4f}, p99={ms.p99:.4f}, n={ms.count}"
                )
                if ms.by_phase:
                    for phase_data in ms.by_phase:
                        metric_section.append(
                            f"    [{phase_data.get('label', '?')}] "
                            f"mean={phase_data.get('mean', 0):.4f}"
                        )
            metric_section.append("")

            # Check if adding metrics would exceed budget
            current = "\n".join(sections)
            metrics_text = "\n".join(metric_section)
            if len(current) + len(metrics_text) < max_chars:
                sections.extend(metric_section)

        # Entity summaries
        if self.summary.entities:
            entity_section = ["## Entities"]
            for name, es in self.summary.entities.items():
                line = f"- {name} ({es.entity_type}): {es.events_handled} events"
                if es.queue_stats:
                    qs = es.queue_stats
                    line += f" | queue: accepted={qs.total_accepted}, dropped={qs.total_dropped}"
                entity_section.append(line)
            entity_section.append("")

            current = "\n".join(sections)
            entity_text = "\n".join(entity_section)
            if len(current) + len(entity_text) < max_chars:
                sections.extend(entity_section)

        result = "\n".join(sections)
        if len(result) > max_chars:
            result = result[:max_chars - 20] + "\n\n[truncated]"
        return result


def analyze(
    summary: SimulationSummary,
    *,
    latency: Data | None = None,
    queue_depth: Data | None = None,
    throughput: Data | None = None,
    trace_recorder: InMemoryTraceRecorder | None = None,
    phase_window_s: float = 5.0,
    phase_threshold: float = 2.0,
    anomaly_threshold: float = 3.0,
    **named_metrics: Data,
) -> SimulationAnalysis:
    """Run full analysis pipeline on simulation results.

    Combines phase detection, anomaly detection, and causal chain
    extraction into a single structured result.

    Args:
        summary: SimulationSummary from Simulation.run().
        latency: Latency time-series data (e.g., from LatencyTracker.data).
        queue_depth: Queue depth time-series data (e.g., from Probe).
        throughput: Throughput time-series data (e.g., from ThroughputTracker.data).
        trace_recorder: Optional InMemoryTraceRecorder for causal analysis.
        phase_window_s: Window size for phase detection.
        phase_threshold: Sensitivity for phase transitions.
        anomaly_threshold: Sensitivity for anomaly detection (std devs).
        **named_metrics: Additional named Data objects to analyze.

    Returns:
        SimulationAnalysis with all detected patterns.
    """
    all_metrics: dict[str, Data] = {}
    if latency is not None:
        all_metrics["latency"] = latency
    if queue_depth is not None:
        all_metrics["queue_depth"] = queue_depth
    if throughput is not None:
        all_metrics["throughput"] = throughput
    all_metrics.update(named_metrics)

    # Phase detection
    phases: dict[str, list[Phase]] = {}
    for name, data in all_metrics.items():
        detected = detect_phases(data, window_s=phase_window_s, threshold=phase_threshold)
        if detected:
            phases[name] = detected

    # Metric summaries
    metric_summaries: dict[str, MetricSummary] = {}
    for name, data in all_metrics.items():
        if data.count() == 0:
            continue

        # Compute per-phase breakdowns
        by_phase: list[dict[str, Any]] = []
        if name in phases:
            for phase in phases[name]:
                phase_data = data.between(phase.start_s, phase.end_s)
                if phase_data.count() > 0:
                    by_phase.append({
                        "label": phase.label,
                        "start_s": phase.start_s,
                        "end_s": phase.end_s,
                        "mean": phase_data.mean(),
                        "p50": phase_data.percentile(0.50),
                        "p99": phase_data.percentile(0.99),
                    })

        metric_summaries[name] = MetricSummary(
            name=name,
            count=data.count(),
            mean=data.mean(),
            std=data.std(),
            min=data.min(),
            max=data.max(),
            p50=data.percentile(0.50),
            p95=data.percentile(0.95),
            p99=data.percentile(0.99),
            by_phase=by_phase,
        )

    # Anomaly detection
    anomalies = _detect_anomalies(all_metrics, phases, anomaly_threshold)

    # Causal chain detection
    causal_chains = _detect_causal_chains(all_metrics, phases, anomalies)

    return SimulationAnalysis(
        summary=summary,
        phases=phases,
        metrics=metric_summaries,
        anomalies=anomalies,
        causal_chains=causal_chains,
    )


def _detect_anomalies(
    metrics: dict[str, Data],
    phases: dict[str, list[Phase]],
    threshold: float,
) -> list[Anomaly]:
    """Detect anomalies by looking for values far from phase means."""
    anomalies: list[Anomaly] = []

    for name, data in metrics.items():
        if data.count() < 10:
            continue

        overall_mean = data.mean()
        overall_std = data.std()
        if overall_std == 0:
            continue

        # Use bucketed data to detect per-window anomalies
        bucketed = data.bucket(window_s=5.0)
        for t, m in zip(bucketed.times(), bucketed.means()):
            deviation = abs(m - overall_mean) / overall_std
            if deviation > threshold:
                severity = "critical" if deviation > threshold * 2 else "warning"
                anomalies.append(Anomaly(
                    time_s=t,
                    metric=name,
                    description=(
                        f"{name} at t={t:.1f}s: mean={m:.4f} "
                        f"({deviation:.1f}x std from overall mean {overall_mean:.4f})"
                    ),
                    severity=severity,
                    context={
                        "window_mean": round(m, 6),
                        "overall_mean": round(overall_mean, 6),
                        "overall_std": round(overall_std, 6),
                        "deviation_stds": round(deviation, 2),
                    },
                ))

    # Sort by time
    anomalies.sort(key=lambda a: a.time_s)
    return anomalies


def _detect_causal_chains(
    metrics: dict[str, Data],
    phases: dict[str, list[Phase]],
    anomalies: list[Anomaly],
) -> list[CausalChain]:
    """Detect causal chains by correlating phase transitions across metrics.

    Looks for patterns like: throughput spike -> queue buildup -> latency increase.
    """
    chains: list[CausalChain] = []

    # Look for queue_depth degradation followed by latency degradation
    qd_phases = phases.get("queue_depth", [])
    lat_phases = phases.get("latency", [])

    for qp in qd_phases:
        if qp.label in ("degraded", "overloaded"):
            # Find a latency phase that starts around the same time
            for lp in lat_phases:
                if lp.label in ("degraded", "overloaded"):
                    # If queue degradation starts before or near latency degradation
                    if abs(qp.start_s - lp.start_s) < 15.0:
                        effects = [
                            f"Queue depth entered '{qp.label}' state (mean={qp.mean:.2f})",
                            f"Latency entered '{lp.label}' state (mean={lp.mean:.4f}s)",
                        ]
                        start = min(qp.start_s, lp.start_s)
                        end = max(qp.end_s, lp.end_s)
                        chains.append(CausalChain(
                            trigger_description=f"System degradation starting at t={start:.1f}s",
                            effects=effects,
                            duration_s=end - start,
                        ))
                        break  # Only one chain per queue phase

    return chains
