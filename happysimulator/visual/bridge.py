"""SimulationBridge: mediator between the simulation and the API layer.

Wraps Simulation + sim.control for the visual debugger, providing
state snapshots, event recording, and topology tracking.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
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
    event_id: str
    is_internal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "event_type": self.event_type,
            "target_name": self.target_name,
            "source_name": self.source_name,
            "event_id": self.event_id,
            "is_internal": self.is_internal,
        }


class SimulationBridge:
    """Wraps a Simulation for the visual debugger API."""

    MAX_EVENT_LOG = 5000

    def __init__(self, sim: "Simulation") -> None:
        self._sim = sim
        self._topology = discover(sim)
        self._event_log: deque[RecordedEvent] = deque(maxlen=self.MAX_EVENT_LOG)
        self._last_handler_name: str | None = None
        self._new_events_buffer: list[RecordedEvent] = []
        self._new_edges_buffer: list[dict[str, str]] = []
        self._lock = threading.Lock()

        # Install event hook
        sim.control.on_event(self._on_event)

    def _on_event(self, event: Event) -> None:
        """Hook called after each event is processed."""
        target_name = getattr(event.target, "name", type(event.target).__name__)

        recorded = RecordedEvent(
            time_s=event.time.to_seconds(),
            event_type=event.event_type,
            target_name=target_name,
            source_name=self._last_handler_name,
            event_id=str(event._id),
            is_internal=is_internal_event(event.event_type),
        )

        with self._lock:
            self._event_log.append(recorded)
            self._new_events_buffer.append(recorded)

        self._last_handler_name = target_name

    def get_topology(self) -> dict:
        """Return the current topology as a JSON-safe dict."""
        return self._topology.to_dict()

    def get_state(self) -> dict[str, Any]:
        """Return full simulation state snapshot."""
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
        }

    def step(self, count: int = 1) -> dict[str, Any]:
        """Step the simulation and return new state + processed events."""
        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()

        self._sim.control.step(count)
        state = self.get_state()

        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()

        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
        }

    def run_to(self, time_s: float) -> dict[str, Any]:
        """Run the simulation until the given time, then return state."""
        from happysimulator.core.control.breakpoints import TimeBreakpoint
        from happysimulator.core.temporal import Instant

        with self._lock:
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()

        target = Instant.from_seconds(time_s)
        self._sim.control.add_breakpoint(TimeBreakpoint(time=target, one_shot=True))
        self._sim.control.resume()

        state = self.get_state()

        with self._lock:
            new_events = [e.to_dict() for e in self._new_events_buffer]
            new_edges = list(self._new_edges_buffer)
            self._new_events_buffer.clear()
            self._new_edges_buffer.clear()

        return {
            "state": state,
            "new_events": new_events,
            "new_edges": new_edges,
        }

    def reset(self) -> dict[str, Any]:
        """Reset the simulation and return the initial state."""
        self._sim.control.reset()
        self._event_log.clear()
        self._last_handler_name = None
        self._topology = discover(self._sim)

        # Re-prime: pause + run
        self._sim.control.pause()
        self._sim.run()

        return self.get_state()

    def get_event_log(self, last_n: int = 100) -> list[dict]:
        """Return the last N recorded events."""
        with self._lock:
            events = list(self._event_log)
        return [e.to_dict() for e in events[-last_n:]]

    def get_timeseries(self, probe_name: str) -> dict[str, Any]:
        """Return time series data for a named probe."""
        from happysimulator.instrumentation.probe import Probe

        for probe in self._sim._probes:
            if isinstance(probe, Probe) and probe.name == probe_name:
                samples = probe.data_sink.values
                return {
                    "name": probe.name,
                    "metric": probe.metric,
                    "target": probe.target.name,
                    "times": [t for t, _ in samples],
                    "values": [v for _, v in samples],
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
