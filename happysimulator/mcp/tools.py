"""Simulation tool implementations for the MCP server.

These functions contain the actual simulation logic and can be imported
without the mcp SDK installed. The server.py module wraps them in MCP
tool definitions.
"""

from __future__ import annotations

import json
import random
from typing import Any

from happysimulator.ai.result import SimulationResult, SweepResult
from happysimulator.components.server import Server as SimServer
from happysimulator.core.simulation import Simulation
from happysimulator.distributions.exponential import ExponentialLatency
from happysimulator.instrumentation.collectors import LatencyTracker
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.source import Source


def run_queue_simulation(
    arrival_rate: float,
    service_rate: float,
    servers: int = 1,
    duration: float = 100.0,
    seed: int | None = None,
) -> SimulationResult:
    """Build and run an M/M/1 or M/M/c queue simulation."""
    if seed is not None:
        random.seed(seed)

    tracker = LatencyTracker("Sink")
    server = SimServer(
        "Server",
        concurrency=servers,
        service_time=ExponentialLatency(1.0 / service_rate),
        downstream=tracker,
    )
    source = Source.poisson(rate=arrival_rate, target=server)
    depth_probe, depth_data = Probe.on(server, "depth", interval=0.5)

    summary = Simulation(
        duration=duration,
        sources=[source],
        entities=[server, tracker],
        probes=[depth_probe],
    ).run()

    return SimulationResult.from_run(
        summary,
        latency=tracker.data,
        queue_depth={"Server": depth_data},
    )


def run_pipeline_simulation(
    stages: list[dict[str, Any]],
    source_rate: float,
    duration: float = 100.0,
    seed: int | None = None,
    poisson: bool = True,
) -> SimulationResult:
    """Build and run a multi-stage pipeline simulation."""
    if seed is not None:
        random.seed(seed)

    tracker = LatencyTracker("Sink")

    # Build stages in reverse so each points to the next
    entities: list[Any] = [tracker]
    probes: list[Any] = []
    depth_data: dict[str, Any] = {}
    downstream = tracker

    for stage_config in reversed(stages):
        name = stage_config.get("name", "Server")
        concurrency = stage_config.get("concurrency", 1)
        service_time_s = stage_config.get("service_time", 0.01)

        server = SimServer(
            name,
            concurrency=concurrency,
            service_time=ExponentialLatency(service_time_s),
            downstream=downstream,
        )
        probe, data = Probe.on(server, "depth", interval=0.5)
        probes.append(probe)
        depth_data[name] = data
        entities.append(server)
        downstream = server

    # Source targets the first stage (last built)
    first_stage = downstream
    if poisson:
        source = Source.poisson(rate=source_rate, target=first_stage)
    else:
        source = Source.constant(rate=source_rate, target=first_stage)

    summary = Simulation(
        duration=duration,
        sources=[source],
        entities=entities,
        probes=probes,
    ).run()

    return SimulationResult.from_run(
        summary,
        latency=tracker.data,
        queue_depth=depth_data,
    )


def format_response(result: SimulationResult) -> str:
    """Format a SimulationResult as JSON with prompt_context and data."""
    return json.dumps({
        "prompt_context": result.to_prompt_context(),
        "data": result.to_dict(),
    }, indent=2, default=str)


DISTRIBUTIONS_INFO = [
    {
        "name": "ConstantLatency",
        "description": "Fixed service time",
        "parameters": {"latency_s": "Service time in seconds"},
        "example": "ConstantLatency(0.01) -> always 10ms",
    },
    {
        "name": "ExponentialLatency",
        "description": "Exponentially distributed service time (memoryless)",
        "parameters": {"mean_s": "Mean service time in seconds"},
        "example": "ExponentialLatency(0.1) -> mean 100ms",
    },
    {
        "name": "UniformDistribution",
        "description": "Uniformly distributed between min and max",
        "parameters": {"low": "Minimum value", "high": "Maximum value"},
        "example": "UniformDistribution(0.01, 0.1) -> 10-100ms",
    },
    {
        "name": "PercentileFittedLatency",
        "description": "Fit a distribution to observed percentile data",
        "parameters": {"percentiles": "Dict of {percentile: value}"},
        "example": 'PercentileFittedLatency({0.5: 0.01, 0.99: 0.1})',
    },
]


def format_distributions(distributions: list[dict] | None = None) -> str:
    """Format distribution info as markdown."""
    distributions = distributions or DISTRIBUTIONS_INFO
    lines = ["## Available Service Time Distributions", ""]
    for d in distributions:
        lines.append(f"### {d['name']}")
        lines.append(d["description"])
        lines.append(f"Example: `{d['example']}`")
        lines.append("")
    return "\n".join(lines)
