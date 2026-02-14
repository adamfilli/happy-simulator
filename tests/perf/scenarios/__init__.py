"""Scenario registry for performance benchmarks."""

from tests.perf.scenarios import (
    cancellation,
    generator_heavy,
    instrumented,
    large_heap,
    memory_footprint,
    throughput,
)

SCENARIOS: dict[str, object] = {
    "throughput": throughput,
    "generator_heavy": generator_heavy,
    "instrumented": instrumented,
    "memory_footprint": memory_footprint,
    "large_heap": large_heap,
    "cancellation": cancellation,
}
