"""Analysis tools for reasoning about simulation results.

This package provides utilities for understanding simulation behavior:

- **phases**: Detect regime changes in time-series data
- **trace_analysis**: Reconstruct event lifecycles from trace data
- **report**: Structured analysis output optimized for AI/LLM consumption
"""

from happysimulator.analysis.phases import Phase, detect_phases
from happysimulator.analysis.trace_analysis import EventLifecycle, trace_event_lifecycle
from happysimulator.analysis.report import (
    Anomaly,
    CausalChain,
    MetricSummary,
    SimulationAnalysis,
    analyze,
)
from happysimulator.analysis.causal_graph import (
    CausalGraph,
    CausalNode,
    build_causal_graph,
)

__all__ = [
    "Phase",
    "detect_phases",
    "EventLifecycle",
    "trace_event_lifecycle",
    "Anomaly",
    "CausalChain",
    "MetricSummary",
    "SimulationAnalysis",
    "analyze",
    "CausalGraph",
    "CausalNode",
    "build_causal_graph",
]
