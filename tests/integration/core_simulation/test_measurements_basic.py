from pathlib import Path

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source


class ConcurrencyTrackerEntity(Entity):
    """Entity that tracks concurrent in-flight requests via a generator process."""

    def __init__(self):
        super().__init__("concurrencytracker")
        self.concurrency = 0
        self.first_counter = 0
        self.second_counter = 0

    def handle_event(self, event: Event):
        self.first_counter += 1
        self.concurrency += 1
        yield 1, None  # Simulate 1 second of processing time
        self.second_counter += 1
        self.concurrency -= 1
        return []


class ConstantOneProfile(Profile):
    """Returns a rate of 1.0 event per second."""

    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 1.0
        return 0


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    """Helper to write CSV files for test output."""
    import csv

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# --- 2. The Test Case ---


def test_basic_concurrency_probe(test_output_dir: Path):
    """
    Verifies that a Probe correctly measures concurrency on a ConcurrencyTrackerEntity.

    The entity increments concurrency at the start of handling an event, then
    yields for 1 second (simulating processing), then decrements concurrency.

    With events arriving at 1/sec and each taking 1 second to process, we expect
    concurrency to hover around 1 (brief overlap possible due to timing).
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A. CONFIGURATION
    sim_duration = 60.0
    probe_interval = 0.1  # Sample concurrency every 0.1 second

    # Setup the concurrency tracking entity
    tracker = ConcurrencyTrackerEntity()

    # Create the event Source
    profile = ConstantOneProfile()
    event_source = Source.with_profile(
        profile=profile,
        target=tracker,
        event_type="Ping",
        poisson=False,
        name="PingSource",
    )

    # Create the Probe to measure concurrency
    concurrency_probe, concurrency_data = Probe.on(tracker, "concurrency", interval=probe_interval)

    # B. INITIALIZATION
    sim = Simulation(
        sources=[event_source],
        entities=[tracker],
        probes=[concurrency_probe],
        duration=sim_duration,
    )

    # C. EXECUTION
    sim.run()

    # D. ASSERTIONS

    # Verify the probe collected data
    samples = concurrency_data.values
    assert len(samples) > 0, "Probe should have collected at least one sample"

    # Verify sample structure (time, value)
    times = [s[0] for s in samples]
    values = [s[1] for s in samples]

    assert times == sorted(times), "Samples should be in chronological order"
    assert all(v >= 0 for v in values), "Concurrency should never be negative"

    # With 1 event/sec and 1 sec processing, concurrency should peak around 1-2
    max_concurrency = max(values)
    assert max_concurrency >= 1, f"Expected max concurrency >= 1, got {max_concurrency}"

    # Verify the tracker processed around 60 events
    assert 59 <= tracker.first_counter <= 61, (
        f"Expected around 60 events started, got {tracker.first_counter}"
    )
    assert 59 <= tracker.second_counter <= 61, (
        f"Expected around 60 events completed, got {tracker.second_counter}"
    )

    # E. VISUALIZATION

    # Save raw data as CSV
    _write_csv(
        test_output_dir / "concurrency_samples.csv",
        header=["time_s", "concurrency"],
        rows=[(t, v) for t, v in samples],
    )

    # Plot 1: Concurrency over time
    fig, (ax_conc, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # Time series plot
    ax_conc.step(times, values, where="post", linewidth=1, color="steelblue")
    ax_conc.fill_between(times, values, step="post", alpha=0.3, color="steelblue")
    ax_conc.set_xlabel("Time (s)")
    ax_conc.set_ylabel("Concurrency")
    ax_conc.set_title("Concurrency Over Time (1 event/sec, 1 sec processing)")
    ax_conc.grid(True, alpha=0.3)
    ax_conc.set_xlim(0, sim_duration)
    ax_conc.set_ylim(bottom=0)

    # Histogram of concurrency values
    ax_hist.hist(
        values, bins=range(int(max(values)) + 3), edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax_hist.set_xlabel("Concurrency Level")
    ax_hist.set_ylabel("Frequency (# samples)")
    ax_hist.set_title("Distribution of Concurrency Levels")
    ax_hist.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(test_output_dir / "concurrency_plot.png", dpi=150)
    plt.close(fig)

    # Plot 2: Cumulative events started vs completed over time
    # (This requires additional tracking, so we'll just note the final counts)
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.step(times, values, where="post", linewidth=1.5, label="Concurrency")
    ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Expected (rate × latency)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Concurrent Requests")
    ax2.set_title("Concurrency Tracking with Probe")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, sim_duration)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "concurrency_with_expected.png", dpi=150)
    plt.close(fig2)

    print(f"\nSaved plots/data to: {test_output_dir}")
    print(f"  - Total samples collected: {len(samples)}")
    print(f"  - Max concurrency observed: {max_concurrency}")
    print(f"  - Events started: {tracker.first_counter}, completed: {tracker.second_counter}")
