"""Entity wrappers for sketching algorithms.

This module provides simulation-aware wrappers around the pure sketching
algorithms. These wrappers are Entities that can be used as event sinks
or processors in simulations.

The wrappers:
- Extract values from events using configurable extractors
- Update sketches on each event
- Provide query methods for accessing sketch statistics
- Support time-windowed analysis (via clock integration)

Example:
    from happysimulator.components.sketching import TopKCollector

    # Track top customers by request count
    collector = TopKCollector(
        name="customer_tracker",
        k=100,
        value_extractor=lambda e: e.context.get("customer_id"),
    )

    # Use in simulation
    sim = Simulation(
        sources=[traffic_source],
        entities=[..., collector],
        end_time=Instant.from_seconds(60),
    )
"""

from happysimulator.components.sketching.sketch_collector import SketchCollector
from happysimulator.components.sketching.topk_collector import TopKCollector
from happysimulator.components.sketching.quantile_estimator import QuantileEstimator

__all__ = [
    "SketchCollector",
    "TopKCollector",
    "QuantileEstimator",
]
