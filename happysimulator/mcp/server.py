"""MCP server exposing happysimulator as callable tools.

Requires the ``mcp`` package: ``pip install mcp``

Usage:
    python -m happysimulator.mcp
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from happysimulator.ai.result import SweepResult
from happysimulator.mcp.tools import (
    DISTRIBUTIONS_INFO,
    format_distributions,
    format_response,
    run_pipeline_simulation,
    run_queue_simulation,
)

mcp_server = Server("happysimulator")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="simulate_queue",
            description=(
                "Run an M/M/1 or M/M/c queue simulation. "
                "Models a server pool with exponential service times "
                "and Poisson arrivals. Returns latency, queue depth, "
                "and throughput analysis with recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arrival_rate": {
                        "type": "number",
                        "description": "Mean arrivals per second (Poisson)",
                    },
                    "service_rate": {
                        "type": "number",
                        "description": "Mean completions per second per server",
                    },
                    "servers": {
                        "type": "integer",
                        "description": "Number of servers (default 1 for M/M/1)",
                        "default": 1,
                    },
                    "duration": {
                        "type": "number",
                        "description": "Simulation duration in seconds (default 100)",
                        "default": 100,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                },
                "required": ["arrival_rate", "service_rate"],
            },
        ),
        Tool(
            name="simulate_pipeline",
            description=(
                "Run a multi-stage pipeline simulation. "
                "Each stage is a server with configurable concurrency "
                "and service time. Returns per-stage queue depth "
                "and end-to-end latency analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stages": {
                        "type": "array",
                        "description": "Pipeline stages in order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "concurrency": {
                                    "type": "integer",
                                    "default": 1,
                                },
                                "service_time": {
                                    "type": "number",
                                    "description": "Mean service time in seconds",
                                },
                            },
                            "required": ["name", "service_time"],
                        },
                    },
                    "source_rate": {
                        "type": "number",
                        "description": "Arrival rate in events/sec",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Simulation duration in seconds (default 100)",
                        "default": 100,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                    "poisson": {
                        "type": "boolean",
                        "description": "Use Poisson arrivals (default true)",
                        "default": True,
                    },
                },
                "required": ["stages", "source_rate"],
            },
        ),
        Tool(
            name="sweep_parameter",
            description=(
                "Run a parametric sweep of an M/M/c queue across multiple "
                "values of a parameter. Returns a comparison table showing "
                "how latency and queue depth change. Useful for finding "
                "saturation points and optimal configurations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "description": "Parameter to sweep",
                        "enum": ["arrival_rate", "service_rate", "servers"],
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Values to sweep over",
                    },
                    "arrival_rate": {
                        "type": "number",
                        "description": "Base arrival rate (required unless sweeping arrival_rate)",
                    },
                    "service_rate": {
                        "type": "number",
                        "description": "Base service rate (required unless sweeping service_rate)",
                    },
                    "servers": {
                        "type": "integer",
                        "description": "Base server count (default 1)",
                        "default": 1,
                    },
                    "duration": {
                        "type": "number",
                        "description": "Simulation duration in seconds (default 100)",
                        "default": 100,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                },
                "required": ["parameter", "values"],
            },
        ),
        Tool(
            name="compare_scenarios",
            description=(
                "Compare two queue simulation configurations side by side. "
                "Returns a diff table showing changes in latency, queue depth, "
                "and throughput between the two scenarios."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scenario_a": {
                        "type": "object",
                        "description": "First scenario config",
                        "properties": {
                            "arrival_rate": {"type": "number"},
                            "service_rate": {"type": "number"},
                            "servers": {"type": "integer", "default": 1},
                            "duration": {"type": "number", "default": 100},
                        },
                        "required": ["arrival_rate", "service_rate"],
                    },
                    "scenario_b": {
                        "type": "object",
                        "description": "Second scenario config",
                        "properties": {
                            "arrival_rate": {"type": "number"},
                            "service_rate": {"type": "number"},
                            "servers": {"type": "integer", "default": 1},
                            "duration": {"type": "number", "default": 100},
                        },
                        "required": ["arrival_rate", "service_rate"],
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                },
                "required": ["scenario_a", "scenario_b"],
            },
        ),
        Tool(
            name="list_distributions",
            description=(
                "List available service time distributions and their parameters. "
                "Useful for understanding what distributions can be used "
                "in simulation configurations."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "simulate_queue":
        result = run_queue_simulation(
            arrival_rate=arguments["arrival_rate"],
            service_rate=arguments["service_rate"],
            servers=arguments.get("servers", 1),
            duration=arguments.get("duration", 100),
            seed=arguments.get("seed"),
        )
        return [TextContent(type="text", text=format_response(result))]

    if name == "simulate_pipeline":
        result = run_pipeline_simulation(
            stages=arguments["stages"],
            source_rate=arguments["source_rate"],
            duration=arguments.get("duration", 100),
            seed=arguments.get("seed"),
            poisson=arguments.get("poisson", True),
        )
        return [TextContent(type="text", text=format_response(result))]

    if name == "sweep_parameter":
        param = arguments["parameter"]
        values = arguments["values"]
        base_arrival = arguments.get("arrival_rate", 10)
        base_service = arguments.get("service_rate", 12)
        base_servers = arguments.get("servers", 1)
        duration = arguments.get("duration", 100)
        seed = arguments.get("seed")

        results = []
        for val in values:
            kwargs = {
                "arrival_rate": base_arrival,
                "service_rate": base_service,
                "servers": base_servers,
                "duration": duration,
                "seed": seed,
            }
            kwargs[param] = int(val) if param == "servers" else val
            results.append(run_queue_simulation(**kwargs))

        sweep = SweepResult(
            parameter_name=param,
            parameter_values=values,
            results=results,
        )
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "prompt_context": sweep.to_prompt_context(),
                        "data": sweep.to_dict(),
                    },
                    indent=2,
                    default=str,
                ),
            )
        ]

    if name == "compare_scenarios":
        seed = arguments.get("seed")
        a_cfg = arguments["scenario_a"]
        b_cfg = arguments["scenario_b"]

        result_a = run_queue_simulation(
            arrival_rate=a_cfg["arrival_rate"],
            service_rate=a_cfg["service_rate"],
            servers=a_cfg.get("servers", 1),
            duration=a_cfg.get("duration", 100),
            seed=seed,
        )
        result_b = run_queue_simulation(
            arrival_rate=b_cfg["arrival_rate"],
            service_rate=b_cfg["service_rate"],
            servers=b_cfg.get("servers", 1),
            duration=b_cfg.get("duration", 100),
            seed=seed,
        )
        comparison = result_a.compare(result_b)
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "prompt_context": comparison.to_prompt_context(),
                        "data": comparison.to_dict(),
                    },
                    indent=2,
                    default=str,
                ),
            )
        ]

    if name == "list_distributions":
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "prompt_context": format_distributions(),
                        "data": {"distributions": DISTRIBUTIONS_INFO},
                    },
                    indent=2,
                ),
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


async def run_server():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )
