"""AI integration layer for happysimulator.

Provides rich result wrappers, comparison tools, and recommendations
for simulation output analysis.
"""

from happysimulator.ai.insights import Recommendation, generate_recommendations
from happysimulator.ai.result import (
    MetricDiff,
    SimulationComparison,
    SimulationResult,
    SweepResult,
)

__all__ = [
    "MetricDiff",
    "Recommendation",
    "SimulationComparison",
    "SimulationResult",
    "SweepResult",
    "generate_recommendations",
]
