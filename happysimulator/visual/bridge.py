"""SimulationBridge: mediator between the simulation and the API layer.

Wraps Simulation + sim.control for the visual debugger, providing
state snapshots, event recording, and topology tracking.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from happysimulator.core.event import Event
from happysimulator.visual.serializers import serialize_entity, serialize_event, is_internal_event
from happysimulator.visual.topology import Topology, discover

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation


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

    def __init__(self, bridge: "SimulationBridge") -> None:
        super().__init__(level=logging.DEBUG)
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            time_s: float | None = None
            try:
                time_s = self._bridge._sim._current_time.to_seconds()
            except Exception:
                pass

            logger_name = record.name
            if logger_name.startswith("happysimulator."):
                logger_name = logger_name[len("happysimulator."):]

            entry = RecordedLog(
                time_s=time_s,
                wall_time=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
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

    def __init__(self, sim: "Simulation", charts: list | None = None) -> None:
        from happysimulator.visual.dashboard import Chart
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
        self._entity_history: dict[str, dict[str, list[tuple[float, float]]]] = {}
        self._last_snapshot_time: float = -1.0
        self._snapshot_interval: float = 0.1
        self._lock = threading.Lock()

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

        # Track edge flow
        if self._last_handler_name and self._last_handler_name != target_name:
            edge_key = (self._last_handler_name, target_name)
            self._edge_counts[edge_key] = self._edge_counts.get(edge_key, 0) + 1
            if edge_key not in self._edge_timestamps:
                self._edge_timestamps[edge_key] = deque(maxlen=1000)
            self._edge_timestamps[edge_key].append(event.time.to_seconds())

        self._last_handler_name = target_name

        # Periodic entity state snapshots
        time_s = event.time.to_seconds()
        if time_s - self._last_snapshot_time >= self._snapshot_interval:
            self._snapshot_entities(time_s)
            self._last_snapshot_time = time_s

    def get_topology(self) -> dict:
        """Return the current topology as a JSON-safe dict."""
        return self._topology.to_dict()

    def get_state(self) -> dict[str, Any]:
        """Return full simulation state snapshot."""
        from happysimulator.core.temporal import Instant

        state = self._sim.control.get_state()
        entity_states: dict[str, Any] = {}

        for entity in list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes):
            name = getattr(entity, "name", type(entity).__name__)
            entity_states[name] = serialize_entity(entity)

        upcoming: list[dict] = []
        if state.is_paused:
            try:
                for evt in self._sim.control.peek_next(10):
                    upcoming.append(serialize_event(evt))
            except RuntimeError:
                pass

        return {
            "time_s": state.current_time.to_seconds(),
            "events_processed": state.events_processed,
            "heap_size": state.heap_size,
            "is_paused": state.is_paused,
            "is_running": state.is_running,
            "is_complete": state.is_complete,
            "entities": entity_states,
            "upcoming": upcoming,
            "end_time_s": self._sim._end_time.to_seconds() if self._sim._end_time != Instant.Infinity else None,
        }

    def step(self, count: int = 1) -> dict[str, Any]:
        """Step the simulation and return new state + processed events."""
        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

        self._sim.control.step(count)
        state = self.get_state()

        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            new_logs = [l.to_dict() for l in self._new_logs_buffer]
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
            "new_logs": new_logs,
            "edge_stats": self.get_edge_stats(),
        }

    def run_to(self, time_s: float) -> dict[str, Any]:
        """Run the simulation until the given time, then return state."""
        from happysimulator.core.control.breakpoints import TimeBreakpoint
        from happysimulator.core.temporal import Instant

        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

        target = Instant.from_seconds(time_s)
        self._sim.control.add_breakpoint(TimeBreakpoint(time=target, one_shot=True))
        self._sim.control.resume()

        state = self.get_state()

        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            new_logs = [l.to_dict() for l in self._new_logs_buffer]
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

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

        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

        self._sim.control.add_breakpoint(
            EventCountBreakpoint(count=event_number, one_shot=True)
        )
        self._sim.control.resume()

        state = self.get_state()

        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            new_logs = [l.to_dict() for l in self._new_logs_buffer]
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()
            self._new_logs_buffer.clear()

        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
            "new_logs": new_logs,
            "edge_stats": self.get_edge_stats(),
        }

    def reset(self) -> dict[str, Any]:
        """Reset the simulation and return the initial state."""
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

        # Clear chart data so it doesn't accumulate across runs
        for chart in self._charts:
            chart.data.clear()
        # Clear probe data sinks
        from happysimulator.instrumentation.probe import Probe
        for probe in self._sim._probes:
            if isinstance(probe, Probe):
                probe.data_sink.clear()

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

    def get_timeseries(self, probe_name: str, start_s: float | None = None, end_s: float | None = None) -> dict[str, Any]:
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

        result = []
        for probe in self._sim._probes:
            if isinstance(probe, Probe):
                result.append({
                    "name": probe.name,
                    "metric": probe.metric,
                    "target": probe.target.name,
                })
        return result

    def get_chart_configs(self) -> list[dict]:
        """Return display config for all predefined charts."""
        return [chart.to_config() for chart in self._charts]

    def get_chart_data(self, chart_id: str, start_s: float | None = None, end_s: float | None = None) -> dict[str, Any]:
        """Return time series data for a predefined chart by ID."""
        for chart in self._charts:
            if chart.chart_id == chart_id:
                result = chart.get_data(start_s=start_s, end_s=end_s)
                result["chart_id"] = chart_id
                result["config"] = chart.to_config()
                return result
        return {"chart_id": chart_id, "times": [], "values": [], "config": {}}

    # --- Breakpoint CRUD ---

    def list_breakpoints_json(self) -> list[dict[str, Any]]:
        """Serialize all breakpoints to JSON-safe dicts."""
        from happysimulator.core.control.breakpoints import (
            TimeBreakpoint, EventCountBreakpoint, EventTypeBreakpoint, MetricBreakpoint,
        )
        result = []
        for bp_id, bp in self._sim.control.list_breakpoints():
            info: dict[str, Any] = {"id": bp_id, "one_shot": bp.one_shot}
            if isinstance(bp, TimeBreakpoint):
                info["type"] = "time"
                info["time_s"] = bp.time.to_seconds()
            elif isinstance(bp, EventCountBreakpoint):
                info["type"] = "event_count"
                info["count"] = bp.count
            elif isinstance(bp, EventTypeBreakpoint):
                info["type"] = "event_type"
                info["event_type"] = bp.event_type
            elif isinstance(bp, MetricBreakpoint):
                info["type"] = "metric"
                info["entity_name"] = bp.entity_name
                info["attribute"] = bp.attribute
                info["operator"] = bp.operator
                info["threshold"] = bp.threshold
            else:
                info["type"] = "custom"
                info["description"] = str(bp)
            result.append(info)
        return result

    def add_breakpoint_from_json(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a breakpoint from a JSON body and return its info."""
        from happysimulator.core.control.breakpoints import (
            TimeBreakpoint, EventCountBreakpoint, EventTypeBreakpoint, MetricBreakpoint,
        )
        from happysimulator.core.temporal import Instant

        bp_type = body.get("type", "")
        one_shot = body.get("one_shot", True)

        if bp_type == "time":
            bp = TimeBreakpoint(time=Instant.from_seconds(body["time_s"]), one_shot=one_shot)
        elif bp_type == "event_count":
            bp = EventCountBreakpoint(count=body["count"], one_shot=one_shot)
        elif bp_type == "event_type":
            bp = EventTypeBreakpoint(event_type=body["event_type"], one_shot=one_shot)
        elif bp_type == "metric":
            bp = MetricBreakpoint(
                entity_name=body["entity_name"],
                attribute=body["attribute"],
                operator=body["operator"],
                threshold=body["threshold"],
                one_shot=one_shot,
            )
        else:
            raise ValueError(f"Unknown breakpoint type: {bp_type}")

        bp_id = self._sim.control.add_breakpoint(bp)
        return {"id": bp_id, "type": bp_type}

    def remove_breakpoint(self, bp_id: str) -> None:
        """Remove a breakpoint by ID."""
        self._sim.control.remove_breakpoint(bp_id)

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._sim.control.clear_breakpoints()

    # --- Edge stats ---

    def get_edge_stats(self) -> dict[str, Any]:
        """Return per-edge throughput stats."""
        current_time = self._sim._current_time.to_seconds()
        window = 5.0  # look back 5 sim-seconds
        stats = {}
        for (src, tgt), timestamps in self._edge_timestamps.items():
            recent = sum(1 for t in timestamps if t >= current_time - window)
            rate = recent / window if window > 0 else 0
            key = f"{src}->{tgt}"
            stats[key] = {"source": src, "target": tgt, "count": self._edge_counts.get((src, tgt), 0), "rate": round(rate, 2)}
        return stats

    # --- Entity state history ---

    def _snapshot_entities(self, time_s: float) -> None:
        """Record numeric entity state values."""
        for entity in list(self._sim._sources) + list(self._sim._entities) + list(self._sim._probes):
            name = getattr(entity, "name", type(entity).__name__)
            serialized = serialize_entity(entity)
            if name not in self._entity_history:
                self._entity_history[name] = {}
            for key, val in serialized.items():
                if isinstance(val, (int, float)):
                    if key not in self._entity_history[name]:
                        self._entity_history[name][key] = []
                    history = self._entity_history[name][key]
                    history.append((time_s, float(val)))
                    # Cap at 10000 samples
                    if len(history) > 10000:
                        # Downsample: keep every other point for older data
                        half = len(history) // 2
                        self._entity_history[name][key] = history[:half:2] + history[half:]

    def get_entity_history(self, entity_name: str, metric: str | None = None) -> dict[str, Any]:
        """Return entity state history."""
        history = self._entity_history.get(entity_name, {})
        if metric:
            data = history.get(metric, [])
            return {
                "entity": entity_name,
                "metrics": {metric: {"times": [t for t, _ in data], "values": [v for _, v in data]}},
            }
        metrics = {}
        for key, data in history.items():
            metrics[key] = {"times": [t for t, _ in data], "values": [v for _, v in data]}
        return {"entity": entity_name, "metrics": metrics}

    # --- Time access ---

    def get_current_time_s(self) -> float:
        """Return current simulation time in seconds."""
        return self._sim._current_time.to_seconds()
