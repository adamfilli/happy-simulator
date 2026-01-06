"""Tracing infrastructure for simulation engine instrumentation.

This module provides engine-level tracing (heap operations, simulation loop events)
separate from application-level tracing (which lives on Event.context["trace"]).
"""

from happysimulator.tracing.recorder import (
    TraceRecorder,
    InMemoryTraceRecorder,
    NullTraceRecorder,
)

__all__ = [
    "TraceRecorder",
    "InMemoryTraceRecorder",
    "NullTraceRecorder",
]
