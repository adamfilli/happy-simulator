"""
Test: Increasing concurrency under overload.

Scenario:
- A ConcurrencyTracker processes requests with 1 second latency.
- Load arrives at 2 requests/second for 60 seconds.
- Since we add 2 requests/sec but only complete 1/sec, concurrency grows by ~1/sec.
- Expected: concurrency reaches ~60 by the end of the simulation.
"""

from pathlib import Path

import pytest

from happysimulator.instrumentation.probe import Probe
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class ConcurrencyTrackerEntity(Entity):
    """Entity that tracks concurrent in-flight requests via a generator process.

    Each request takes 1 second to process. The entity is single-threaded,
    meaning only one request can be actively processing at a time. Other
    requests must wait for the lock, causing concurrency to build up.
    """

    def __init__(self):
        super().__init__("concurrencytracker")
        self.concurrency = 0
        self.requests_started = 0
        self.requests_completed = 0
        self._is_processing = False  # Lock for single-threaded processing

    def handle_event(self, event: Event):
        self.requests_started += 1
        self.concurrency += 1

        # Wait for the lock (single-threaded processing constraint)
        while self._is_processing:
            yield 0.01, None  # Poll until lock is available

        # Acquire lock and process
        self._is_processing = True
        yield 1, None  # Simulate 1 second of processing time
        self._is_processing = False  # Release lock

        self.requests_completed += 1
        self.concurrency -= 1
        return []


class ConstantTwoPerSecondProfile(Profile):
    """Returns a rate of 2.0 events per second for 60 seconds."""

    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 2.0
        else:
            return 0


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    """Helper to write CSV files for test output."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_increasing_concurrency_under_overload(test_output_dir: Path):
    """
    Verifies that concurrency grows when load exceeds processing capacity.

    With 2 requests/sec arriving and each taking 1 second to process,
    we can only complete 1 request per second per concurrent slot.
    Initially concurrency=0, so:
    - At t=0.5: 1st request arrives, concurrency=1
    - At t=1.0: 2nd request arrives, concurrency=2
    - At t=1.5: 1st request completes, 3rd arrives, concurrency=2

    Since arrival rate (2/s) > completion rate (~1/s initially),
    concurrency grows roughly linearly until we reach ~60 by t=60s.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A. CONFIGURATION
    sim_duration = 60.0
    probe_interval = 0.5  # Sample concurrency every 0.5 seconds

    # Setup the concurrency tracking entity
    tracker = ConcurrencyTrackerEntity()

    # Create the event Source (generates 2 events/sec)
    profile = ConstantTwoPerSecondProfile()
    event_source = Source.with_profile(
        profile=profile, target=tracker,
        poisson=False, name="RequestSource",
    )

    # Create the Probe to measure concurrency
    concurrency_probe, concurrency_data = Probe.on(tracker, "concurrency", interval=probe_interval)

    # B. INITIALIZATION
    sim = Simulation(
        sources=[event_source],
        entities=[tracker],
        probes=[concurrency_probe],
        duration=sim_duration
    )

    # C. EXECUTION
    sim.run()

    # D. ASSERTIONS

    # Verify the probe collected data
    samples = concurrency_data.values
    assert len(samples) > 0, "Probe should have collected at least one sample"

    times = [s[0] for s in samples]
    values = [s[1] for s in samples]

    assert times == sorted(times), "Samples should be in chronological order"
    assert all(v >= 0 for v in values), "Concurrency should never be negative"

    # We expect ~120 requests started (2/sec * 60 sec)
    assert 118 <= tracker.requests_started <= 122, \
        f"Expected ~120 requests started, got {tracker.requests_started}"

    # Final concurrency should be around 60 (±5 for timing tolerance)
    # because we started 120 and completed ~60 (1/sec for 60 seconds)
    final_concurrency = tracker.concurrency
    max_concurrency = max(values)

    # Allow some tolerance since the exact timing depends on event ordering
    assert 50 <= max_concurrency <= 70, \
        f"Expected max concurrency around 60, got {max_concurrency}"

    # Verify concurrency increased over time (not flat)
    early_samples = [v for t, v in samples if t < 10]
    late_samples = [v for t, v in samples if t > 50]

    avg_early = sum(early_samples) / len(early_samples) if early_samples else 0
    avg_late = sum(late_samples) / len(late_samples) if late_samples else 0

    assert avg_late > avg_early, \
        f"Concurrency should increase over time. Early avg: {avg_early:.1f}, Late avg: {avg_late:.1f}"

    # E. VISUALIZATION

    # Save raw data as CSV
    _write_csv(
        test_output_dir / "increasing_concurrency_samples.csv",
        header=["time_s", "concurrency"],
        rows=[(t, v) for t, v in samples]
    )

    # Plot 1: Concurrency over time showing the growth
    fig, (ax_conc, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # Time series plot
    ax_conc.step(times, values, where="post", linewidth=1.5, color="crimson")
    ax_conc.fill_between(times, values, step="post", alpha=0.3, color="crimson")

    # Add theoretical line: concurrency grows by ~1 per second
    theoretical_times = [0, sim_duration]
    theoretical_concurrency = [0, sim_duration]  # 1 extra request/sec accumulates
    ax_conc.plot(theoretical_times, theoretical_concurrency,
                 color="blue", linestyle="--", linewidth=2, alpha=0.7,
                 label="Theoretical (arrival - completion rate)")

    ax_conc.set_xlabel("Time (s)")
    ax_conc.set_ylabel("Concurrency")
    ax_conc.set_title("Concurrency Under Overload (2 req/s arrival, 1 sec processing)")
    ax_conc.grid(True, alpha=0.3)
    ax_conc.set_xlim(0, sim_duration)
    ax_conc.set_ylim(bottom=0)
    ax_conc.legend()

    # Histogram of concurrency values
    ax_hist.hist(values, bins=30, edgecolor="black", alpha=0.7, color="crimson")
    ax_hist.set_xlabel("Concurrency Level")
    ax_hist.set_ylabel("Frequency (# samples)")
    ax_hist.set_title("Distribution of Concurrency Levels")
    ax_hist.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(test_output_dir / "increasing_concurrency_plot.png", dpi=150)
    plt.close(fig)

    # Plot 2: Annotated version with key metrics
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    ax2.step(times, values, where="post", linewidth=1.5, color="crimson", label="Measured Concurrency")
    ax2.fill_between(times, values, step="post", alpha=0.2, color="crimson")

    # Annotations
    ax2.axhline(y=max_concurrency, color="darkred", linestyle=":", alpha=0.7,
                label=f"Max concurrency = {max_concurrency}")
    ax2.axhline(y=avg_late, color="orange", linestyle="--", alpha=0.7,
                label=f"Late avg (t>50s) = {avg_late:.1f}")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Concurrent Requests")
    ax2.set_title(f"Overload Test: {tracker.requests_started} started, {tracker.requests_completed} completed")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, sim_duration)
    ax2.set_ylim(bottom=0)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "increasing_concurrency_annotated.png", dpi=150)
    plt.close(fig2)

    print(f"\nSaved plots/data to: {test_output_dir}")
    print(f"  - Total samples collected: {len(samples)}")
    print(f"  - Max concurrency observed: {max_concurrency}")
    print(f"  - Final concurrency: {final_concurrency}")
    print(f"  - Requests: {tracker.requests_started} started, {tracker.requests_completed} completed")
    print(f"  - Early avg concurrency (t<10s): {avg_early:.1f}")
    print(f"  - Late avg concurrency (t>50s): {avg_late:.1f}")
