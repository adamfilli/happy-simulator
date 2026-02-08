"""AI-powered simulation analysis workflow.

Demonstrates the full pipeline:
1. Run a simulation
2. Analyze results with detect_phases(), anomaly detection, causal chains
3. Generate structured output for LLM consumption via to_prompt_context()

This example runs a simple M/M/1 queue with a load spike to create
interesting behavior for analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    LatencyTracker,
    PoissonArrivalTimeProvider,
    Probe,
    Profile,
    QueuedResource,
    Simulation,
    Source,
)
from happysimulator.analysis import analyze, detect_phases


# =============================================================================
# Simple spike profile for analysis demonstration
# =============================================================================


@dataclass(frozen=True)
class SpikeLoadProfile(Profile):
    """Simple profile: steady state -> spike -> steady state."""
    base_rate: float = 5.0
    spike_rate: float = 20.0
    spike_start: float = 30.0
    spike_end: float = 40.0

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        if self.spike_start <= t < self.spike_end:
            return self.spike_rate
        return self.base_rate


class SimpleServer(QueuedResource):
    """M/M/1 server with exponential service times."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.stats_processed: int = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        service_time = random.expovariate(10.0)  # mean 0.1s
        yield service_time, None
        self.stats_processed += 1
        return [Event(
            time=self.now,
            event_type="Completed",
            target=self.downstream,
            context=event.context,
        )]


class SimpleProvider(EventProvider):
    def __init__(self, target: Entity):
        self._target = target

    def get_events(self, time: Instant) -> list[Event]:
        return [Event(
            time=time,
            event_type="Request",
            target=self._target,
            context={"created_at": time},
        )]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    random.seed(42)

    # Build pipeline
    sink = LatencyTracker("Sink")
    server = SimpleServer("Server", downstream=sink)

    queue_data = Data()
    queue_probe = Probe(
        target=server, metric="depth", data=queue_data,
        interval=0.5, start_time=Instant.Epoch,
    )

    profile = SpikeLoadProfile()
    provider = SimpleProvider(server)
    arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Load", event_provider=provider, arrival_time_provider=arrival)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(80.0),
        sources=[source],
        entities=[server, sink],
        probes=[queue_probe],
    )

    # 1. Run simulation
    summary = sim.run()

    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(summary)

    # 2. Detect phases in latency data
    print("\n" + "=" * 60)
    print("PHASE DETECTION")
    print("=" * 60)
    phases = detect_phases(sink.data, window_s=5.0)
    for p in phases:
        print(f"  [{p.label:>10}] {p.start_s:.0f}s - {p.end_s:.0f}s  "
              f"mean={p.mean:.4f}s  std={p.std:.4f}s")

    # 3. Run full analysis pipeline
    print("\n" + "=" * 60)
    print("FULL ANALYSIS")
    print("=" * 60)
    analysis = analyze(
        summary,
        latency=sink.data,
        queue_depth=queue_data,
    )

    # Show anomalies
    if analysis.anomalies:
        print(f"\nAnomalies detected: {len(analysis.anomalies)}")
        for a in analysis.anomalies:
            print(f"  [{a.severity}] t={a.time_s:.0f}s: {a.description}")
    else:
        print("\nNo anomalies detected.")

    # Show causal chains
    if analysis.causal_chains:
        print(f"\nCausal chains: {len(analysis.causal_chains)}")
        for chain in analysis.causal_chains:
            print(f"  Trigger: {chain.trigger_description}")
            for effect in chain.effects:
                print(f"    -> {effect}")
    else:
        print("\nNo causal chains detected.")

    # 4. Generate LLM-optimized context
    print("\n" + "=" * 60)
    print("LLM PROMPT CONTEXT (for feeding to an AI model)")
    print("=" * 60)
    prompt_context = analysis.to_prompt_context(max_tokens=1000)
    print(prompt_context)

    # 5. Show JSON output
    print("\n" + "=" * 60)
    print("JSON OUTPUT (for programmatic consumption)")
    print("=" * 60)
    import json
    d = analysis.to_dict()
    # Truncate for display
    print(json.dumps(d, indent=2, default=str)[:2000])
    if len(json.dumps(d, default=str)) > 2000:
        print("  ... [truncated]")


if __name__ == "__main__":
    main()
