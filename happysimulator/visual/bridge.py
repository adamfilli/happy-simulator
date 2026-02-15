"""SimulationBridge: mediator between the simulation and the API layer.

Wraps Simulation + sim.control for the visual debugger, providing
state snapshots, event recording, and topology tracking.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from happysimulator.visual.code_debugger import CodeBreakpoint, CodeDebugger
from happysimulator.visual.serializers import is_internal_event, serialize_entity, serialize_event
from happysimulator.visual.topology import discover

if TYPE_CHECKING:
    from happysimulator.core.event import Event
    from happysimulator.core.simulation import Simulation
    from happysimulator.visual.dashboard import Chart


@dataclass
class RecordedEvent:
    time_s: float
    event_type: str
    target_name: str
    source_name: str | None
    event_id: int
    is_internal: bool
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "time_s": self.time_s,
            "event_type": self.event_type,
            "target_name": self.target_name,
            "source_name": self.source_name,
            "event_id": self.event_id,
            "is_internal": self.is_internal,
        }
        if self.context:
            d["context"] = self.context
        return d


@dataclass
class RecordedLog:
    time_s: float | None
    wall_time: str
    level: str
    logger_name: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "wall_time": self.wall_time,
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
        }


class _BridgeLogHandler(logging.Handler):
    """Captures log records from the happysimulator logger hierarchy."""

    def __init__(self, bridge: SimulationBridge) -> None:
        super().__init__(level=logging.DEBUG)
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            time_s: float | None = None
            with contextlib.suppress(Exception):
                time_s = self._bridge._sim._current_time.to_seconds()

            logger_name = record.name
            if logger_name.startswith("happysimulator."):
                logger_name = logger_name[len("happysimulator.") :]

            entry = RecordedLog(
                time_s=time_s,
                wall_time=datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3],
                level=record.levelname,
                logger_name=logger_name,
                message=self.format(record),
            )

            with self._bridge._lock:
                self._bridge._log_buffer.append(entry)
                self._bridge._new_logs_buffer.append(entry)
        except Exception:
            self.handleError(record)


class SimulationBridge:
    """Wraps a Simulation for the visual debugger API."""

    MAX_EVENT_LOG = 5000
    MAX_LOG_BUFFER = 5000
    # PERF: Each entity accumulates up to MAX_HISTORY_SAMPLES snapshots, each a
    # full serialized dict (~5-10 keys).  With 20 entities this caps at ~50 MB.
    # Future optimisation: store only numeric fields at write time instead of the
    # full dict, and consider adaptive snapshot intervals for long-running sims.
    MAX_HISTORY_SAMPLES = 10_000

    def __init__(self, sim: Simulation, charts: list | None = None) -> None:

        self._sim = sim
        self._charts: list[Chart] = charts or []
        self._topology = discover(sim)
        self._event_log: deque[RecordedEvent] = deque(maxlen=self.MAX_EVENT_LOG)
        self._last_handler_name: str | None = None
        self._new_events_buffer: list[RecordedEvent] = []
        self._new_edges_buffer: list[dict[str, str]] = []
        self._log_buffer: deque[RecordedLog] = deque(maxlen=self.MAX_LOG_BUFFER)
        self._new_logs_buffer: list[RecordedLog] = []
        self._event_counter: int = 0
        self._edge_counts: dict[tuple[str, str], int] = {}
        self._edge_timestamps: dict[tuple[str, str], deque] = {}
        self._topology_node_ids: set[str] = {n.id for n in self._topology.nodes}
        self._entity_history: dict[str, list[tuple[float, dict]]] = {}
        self._last_snapshot_time: float = -1.0
        self._lock = threading.Lock()

        # Code debugger — injected into the simulation for ProcessContinuation access
        self._code_debugger = CodeDebugger()
        sim._code_debugger = self._code_debugger

        # Install event hook
        sim.control.on_event(self._on_event)

        # Install log handler on the happysimulator root logger
        self._hs_logger = logging.getLogger("happysimulator")
        self._prev_log_level = self._hs_logger.level
        if self._hs_logger.level > logging.DEBUG or self._hs_logger.level == logging.NOTSET:
            self._hs_logger.setLevel(logging.DEBUG)
        self._log_handler = _BridgeLogHandler(self)
        self._hs_logger.addHandler(self._log_handler)

    @staticmethod
    def _serialize_context(ctx: dict[str, Any]) -> dict[str, Any]:
        """Convert event context to JSON-safe dict."""
        from happysimulator.core.temporal import Instant

        result: dict[str, Any] = {}
        for key, val in ctx.items():
            if isinstance(val, Instant):
                result[key] = val.to_seconds()
            elif isinstance(val, dict):
                result[key] = SimulationBridge._serialize_context(val)
            elif isinstance(val, list):
                serialized = []
                for item in val:
                    if isinstance(item, dict):
                        serialized.append(SimulationBridge._serialize_context(item))
                    elif isinstance(item, Instant):
                        serialized.append(item.to_seconds())
                    elif isinstance(item, (int, float, str, bool, type(None))):
                        serialized.append(item)
                    else:
                        serialized.append(str(item))
                result[key] = serialized
            elif isinstance(val, (int, float, str, bool, type(None))):
                result[key] = val
            else:
                result[key] = str(val)
        return result

    def _on_event(self, event: Event) -> None:
        """Hook called after each event is processed."""
        self._event_counter += 1
        target_name = getattr(event.target, "name", type(event.target).__name__)

        ctx = self._serialize_context(event.context) if event.context else None

        recorded = RecordedEvent(
            time_s=event.time.to_seconds(),
            event_type=event.event_type,
            target_name=target_name,
            source_name=self._last_handler_name,
            event_id=self._event_counter,
            is_internal=is_internal_event(event.event_type),
            context=ctx,
        )

        with self._lock:
            self._event_log.append(recorded)
            self._new_events_buffer.append(recorded)

        # Track edge flow (resolve sub-entity names to topology-level parents)
        resolved_target = self._resolve_to_topology_node(target_name)
        resolved_source = (
            self._resolve_to_topology_node(self._last_handler_name)
            if self._last_handler_name
            else None
        )
        if resolved_source and resolved_source != resolved_target:
            edge_key = (resolved_source, resolved_target)
            self._edge_counts[edge_key] = self._edge_counts.get(edge_key, 0) + 1
            if edge_key not in self._edge_timestamps:
                self._edge_timestamps[edge_key] = deque(maxlen=1000)
            self._edge_timestamps[edge_key].append(event.time.to_seconds())

        self._last_handler_name = target_name

        # --- Entity state history snapshots ---
        # PERF: The time check (float subtraction + comparison) runs on every
        # event but is negligible.  The expensive part — iterating all entities
        # and calling serialize_entity() — only fires every 0.1 s of sim time.
        time_s = event.time.to_seconds()
        if time_s - self._last_snapshot_time >= 0.1:
            self._last_snapshot_time = time_s
            for entity in (
                list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes)
            ):
                name = getattr(entity, "name", type(entity).__name__)
                snapshot = serialize_entity(entity)
                history = self._entity_history.setdefault(name, [])
                history.append((time_s, snapshot))
                # PERF: Halving discards every other sample, losing temporal
                # resolution on older data.  A ring-buffer or logarithmic
                # downsampling would preserve more detail at the edges.
                if len(history) > self.MAX_HISTORY_SAMPLES:
                    self._entity_history[name] = history[::2]

    def _resolve_to_topology_node(self, name: str) -> str:
        """Map a sub-entity name (e.g. 'Server.worker') to its topology-level parent ('Server')."""
        if name in self._topology_node_ids:
            return name
        # Walk up dotted segments: Server.driver.foo → Server.driver → Server
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in self._topology_node_ids:
                return candidate
        return name

    def get_current_time_s(self) -> float:
        """Return current simulation time in seconds."""
        return self._sim._current_time.to_seconds()

    def get_topology(self) -> dict:
        """Return the current topology as a JSON-safe dict."""
        return self._topology.to_dict()

    def get_state(self) -> dict[str, Any]:
        """Return full simulation state snapshot."""
        from happysimulator.core.temporal import Instant

        state = self._sim.control.get_state()
        entity_states: dict[str, Any] = {}

        for entity in (
            list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes)
        ):
            name = getattr(entity, "name", type(entity).__name__)
            entity_states[name] = serialize_entity(entity)

        upcoming: list[dict] = []
        if state.is_paused:
            with contextlib.suppress(RuntimeError):
                upcoming.extend(serialize_event(evt) for evt in self._sim.control.peek_next(10))

        end_time = self._sim._end_time
        end_time_s = end_time.to_seconds() if end_time != Instant.Infinity else None

        return {
            "time_s": state.current_time.to_seconds(),
            "events_processed": state.events_processed,
            "heap_size": state.heap_size,
            "is_paused": state.is_paused,
            "is_running": state.is_running,
            "is_complete": state.is_complete,
            "entities": entity_states,
            "upcoming": upcoming,
            "end_time_s": end_time_s,
        }

    def _clear_new_buffers(self) -> None:
        """Clear the new-event, new-edge, and new-log buffers under the lock."""
        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

    def _drain_new_buffers(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Drain and return contents of the new-event/edge/log buffers."""
        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            new_logs = [l.to_dict() for l in self._new_logs_buffer]
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()
        return new_events, new_edges, new_logs

    def step(self, count: int = 1) -> dict[str, Any]:
        """Step the simulation and return new state + processed events."""
        self._clear_new_buffers()

        self._sim.control.step(count)
        state = self.get_state()
        new_events, new_edges, new_logs = self._drain_new_buffers()

        # Drain code traces
        code_traces = [t.to_dict() for t in self._code_debugger.drain_completed_traces()]

        result: dict[str, Any] = {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
            "new_logs": new_logs,
            "edge_stats": self.get_edge_stats(),
        }
        if code_traces:
            result["code_traces"] = code_traces
        return result

    def run_to(self, time_s: float) -> dict[str, Any]:
        """Run the simulation until the given time, then return state."""
        from happysimulator.core.control.breakpoints import TimeBreakpoint
        from happysimulator.core.temporal import Instant

        self._clear_new_buffers()

        target = Instant.from_seconds(time_s)
        self._sim.control.add_breakpoint(TimeBreakpoint(time=target, one_shot=True))
        self._sim.control.resume()

        state = self.get_state()
        new_events, new_edges, new_logs = self._drain_new_buffers()
        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
            "new_logs": new_logs,
            "edge_stats": self.get_edge_stats(),
        }

    def run_to_event(self, event_number: int) -> dict[str, Any]:
        """Run the simulation until the given event number, then return state."""
        from happysimulator.core.control.breakpoints import EventCountBreakpoint

        self._clear_new_buffers()

        self._sim.control.add_breakpoint(EventCountBreakpoint(count=event_number, one_shot=True))
        self._sim.control.resume()

        state = self.get_state()
        new_events, new_edges, new_logs = self._drain_new_buffers()
        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
            "new_logs": new_logs,
            "edge_stats": self.get_edge_stats(),
        }

    def reset(self) -> dict[str, Any]:
        """Reset the simulation and return the initial state."""
        self._code_debugger.reset()
        self._sim.control.reset()
        self._event_log.clear()
        self._log_buffer.clear()
        self._event_counter = 0
        self._last_handler_name = None
        self._edge_counts.clear()
        self._edge_timestamps.clear()
        self._entity_history.clear()
        self._last_snapshot_time = -1.0
        self._topology = discover(self._sim)
        self._topology_node_ids = {n.id for n in self._topology.nodes}
        for chart in self._charts:
            chart.data.clear()

        # Re-prime: pause + run
        self._sim.control.pause()
        self._sim.run()

        return self.get_state()

    def close(self) -> None:
        """Remove the log handler and restore logger level."""
        self._hs_logger.removeHandler(self._log_handler)
        self._hs_logger.setLevel(self._prev_log_level)

    def get_event_log(self, last_n: int = 100) -> list[dict]:
        """Return the last N recorded events."""
        with self._lock:
            events = list(self._event_log)
        return [e.to_dict() for e in events[-last_n:]]

    def get_timeseries(
        self, probe_name: str, start_s: float | None = None, end_s: float | None = None
    ) -> dict[str, Any]:
        """Return time series data for a named probe."""
        from happysimulator.instrumentation.probe import Probe

        for probe in self._sim._probes:
            if isinstance(probe, Probe) and probe.name == probe_name:
                data = probe.data_sink
                if start_s is not None or end_s is not None:
                    s = start_s if start_s is not None else 0.0
                    e = end_s if end_s is not None else float("inf")
                    data = data.between(s, e)
                return {
                    "name": probe.name,
                    "metric": probe.metric,
                    "target": probe.target.name,
                    "times": data.times(),
                    "values": data.raw_values(),
                }
        return {"name": probe_name, "metric": "", "target": "", "times": [], "values": []}

    def list_probes(self) -> list[dict[str, str]]:
        """Return metadata for all registered probes."""
        from happysimulator.instrumentation.probe import Probe

        result = [
            {
                "name": probe.name,
                "metric": probe.metric,
                "target": probe.target.name,
            }
            for probe in self._sim._probes
            if isinstance(probe, Probe)
        ]
        return result

    def get_chart_configs(self) -> list[dict]:
        """Return display config for all predefined charts."""
        return [chart.to_config() for chart in self._charts]

    def get_chart_data(
        self, chart_id: str, start_s: float | None = None, end_s: float | None = None
    ) -> dict[str, Any]:
        """Return time series data for a predefined chart by ID."""
        for chart in self._charts:
            if chart.chart_id == chart_id:
                result = chart.get_data(start_s=start_s, end_s=end_s)
                result["chart_id"] = chart_id
                result["config"] = chart.to_config()
                return result
        return {"chart_id": chart_id, "times": [], "values": [], "config": {}}

    # --- Entity history ---

    def get_entity_history(self, entity_name: str) -> dict[str, Any]:
        """Return per-metric time series extracted from snapshot history."""
        snapshots = self._entity_history.get(entity_name, [])
        if not snapshots:
            return {"entity": entity_name, "metrics": {}}

        # Collect all numeric metric names from the first snapshot
        metric_names = [k for k, v in snapshots[0][1].items() if isinstance(v, (int, float))]

        metrics: dict[str, dict[str, list]] = {}
        for name in metric_names:
            times: list[float] = []
            values: list[float] = []
            for t, state in snapshots:
                val = state.get(name)
                if isinstance(val, (int, float)):
                    times.append(t)
                    values.append(val)
            metrics[name] = {"times": times, "values": values}

        return {"entity": entity_name, "metrics": metrics}

    # --- Edge stats ---

    def get_edge_stats(self) -> dict[str, Any]:
        """Return per-edge throughput stats with 5s sliding window."""
        current_time = self._sim._current_time.to_seconds()
        window = 5.0
        stats = {}
        for (src, tgt), timestamps in self._edge_timestamps.items():
            recent = sum(1 for t in timestamps if t >= current_time - window)
            rate = recent / window if window > 0 else 0
            key = f"{src}->{tgt}"
            stats[key] = {
                "source": src,
                "target": tgt,
                "count": self._edge_counts.get((src, tgt), 0),
                "rate": round(rate, 2),
            }
        return stats

    # --- Code Debug Methods ---

    def get_entity_source(self, entity_name: str) -> dict[str, Any] | None:
        """Return source code for an entity's handler method."""
        for entity in (
            list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes)
        ):
            name = getattr(entity, "name", type(entity).__name__)
            if name == entity_name:
                location = self._code_debugger.get_source(entity)
                if location:
                    return location.to_dict()
                return None
        return None

    def activate_code_debug(self, entity_name: str) -> dict[str, Any]:
        """Activate code-level debugging for an entity."""
        # Pre-cache source
        for entity in (
            list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes)
        ):
            name = getattr(entity, "name", type(entity).__name__)
            if name == entity_name:
                self._code_debugger.get_source(entity)
                break
        self._code_debugger.activate_entity(entity_name)
        return self._code_debugger.get_state()

    def deactivate_code_debug(self, entity_name: str) -> dict[str, Any]:
        """Deactivate code-level debugging for an entity."""
        self._code_debugger.deactivate_entity(entity_name)
        return self._code_debugger.get_state()

    def set_code_breakpoint(self, entity_name: str, line_number: int) -> dict[str, Any]:
        """Set a code breakpoint on a specific line."""
        bp = CodeBreakpoint(entity_name=entity_name, line_number=line_number)
        bp_id = self._code_debugger.add_breakpoint(bp)
        return {"id": bp_id, **bp.to_dict()}

    def remove_code_breakpoint(self, bp_id: str) -> bool:
        """Remove a code breakpoint by ID."""
        return self._code_debugger.remove_breakpoint(bp_id)

    def get_code_debug_state(self) -> dict[str, Any]:
        """Return the full code debug state."""
        return self._code_debugger.get_state()

    def code_continue(self) -> None:
        """Continue from a code breakpoint pause."""
        self._code_debugger.code_continue()

    def code_step(self) -> None:
        """Step to the next line."""
        self._code_debugger.code_step()

    def code_step_over(self) -> None:
        """Step over — next line in the same frame."""
        self._code_debugger.code_step_over()

    def code_step_out(self) -> None:
        """Step out — continue until the current frame returns."""
        self._code_debugger.code_step_out()
