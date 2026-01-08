"""
Test: QueueingServer with queue depth monitoring.

Scenario:
- A QueueingServer processes requests with 1 second latency and concurrency=1.
- Load arrives at 2 requests/second for 60 seconds.
- Since we can only process 1 req/sec, the queue depth grows over time.
- Expected: queue depth reaches ~60 by the end of the simulation.
"""

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Generic, TypeVar, Optional, List

import pytest

from happysimulator.data.data import Data
from happysimulator.data.probe import Probe
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.simulation import Simulation
from happysimulator.utils.instant import Instant

T = TypeVar('T')

class QueuePolicy(ABC, Generic[T]):
    @abstractmethod
    def push(self, item: T) -> bool:
        """Add item. Return False if rejected (full)."""
        pass

    @abstractmethod
    def pop(self) -> Optional[T]:
        """Remove and return next item, or None if empty."""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class FIFOQueue(QueuePolicy):
    def __init__(self, capacity: int = float('inf')):
        self.capacity = capacity
        self._queue = deque()

    def push(self, item) -> bool:
        if len(self._queue) >= self.capacity:
            return False # Drop (or handle rejection)
        self._queue.append(item)
        return True

    def pop(self):
        if not self._queue: return None
        return self._queue.popleft() # FIFO
        
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
    
class QueueingServer(Entity):
    def __init__(self, name: str, service_time: float, concurrency: int = 1, queue_capacity: int = 100):
        super().__init__(name)
        self.service_time = service_time
        self.concurrency_limit = concurrency
        
        # Strategy Pattern: Easily swap this for PriorityQueue later
        self.queue = FIFOQueue(capacity=queue_capacity)
        
        self.active_workers = 0
        self.requests_dropped = 0
        self.requests_completed = 0
    
    @property
    def queue_depth(self) -> int:
        """Current number of items waiting in the queue."""
        return len(self.queue)
    
    @property
    def in_flight(self) -> int:
        """Total requests in the system (queued + being processed)."""
        return len(self.queue) + self.active_workers

    def handle_event(self, event: Event):
        # 1. Enqueue the new work
        accepted = self.queue.push(event)
        
        if not accepted:
            self.requests_dropped += 1
            # Optional: Return a "Dropped" event or log it
            return []

        # 2. Check if we can start a worker immediately
        if self.active_workers < self.concurrency_limit:
            self.active_workers += 1
            # Start the generator loop!
            return self._worker_process()
            
        # 3. If all workers busy, return nothing.
        # The event is safely in self.queue and will be handled later.
        return []

    def _worker_process(self):
        """
        A daemon-like process that keeps working until the queue is empty.
        This avoids the overhead of restarting the generator for every request.
        """
        while not self.queue.is_empty():
            # A. Pick next job
            request_event = self.queue.pop()
            
            # B. Process it (Yield Delay)
            # You can make this dynamic based on the request_event payload if needed
            yield self.service_time
            
            # C. Finish job
            self.requests_completed += 1
            
            # (Optional) Emit Side Effects (Response Event)
            # response = Event(time=..., target=request_event.source, ...)
            # yield 0, [response]

        # D. No more work? Retire this worker.
        self.active_workers -= 1


# --- Test Components ---

class RequestEvent(Event):
    """A request event targeting a QueueingServer."""
    
    def __init__(self, time: Instant, target: Entity):
        super().__init__(time=time, event_type="Request", target=target, callback=None)


class ConstantTwoPerSecondProfile(Profile):
    """Returns a rate of 2.0 events per second for 60 seconds."""
    
    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 2.0
        else:
            return 0


class RequestProvider(EventProvider):
    """Provides RequestEvents targeting a QueueingServer."""
    
    def __init__(self, server: QueueingServer):
        super().__init__()
        self.server = server
    
    def get_events(self, time: Instant) -> List[Event]:
        return [RequestEvent(time, self.server)]


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    """Helper to write CSV files for test output."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# --- Test Case ---

def test_queueing_server_queue_depth(test_output_dir: Path):
    """
    Verifies that queue depth grows when load exceeds processing capacity.
    
    With 2 requests/sec arriving and 1 worker processing at 1 sec/request,
    we can only complete 1 request per second. The queue depth grows by ~1/sec.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A. CONFIGURATION
    sim_duration = 60.0
    probe_interval = 0.5  # Sample every 0.5 seconds
    
    # Setup the queueing server (single worker, 1 sec processing, large queue)
    server = QueueingServer(
        name="TestServer",
        service_time=1.0,
        concurrency=1,
        queue_capacity=200  # Large enough to not drop requests
    )
    
    # Setup the data sinks for probes
    queue_depth_data = Data()
    in_flight_data = Data()
    
    # Setup the Source components (generates 2 events/sec)
    profile = ConstantTwoPerSecondProfile()
    provider = RequestProvider(server)
    arrival_time_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
    
    # Create the event Source
    event_source = Source(
        name="RequestSource",
        event_provider=provider,
        arrival_time_provider=arrival_time_provider
    )
    
    # Create Probes to measure queue depth and total in-flight
    queue_depth_probe = Probe(
        target=server,
        metric="queue_depth",
        data=queue_depth_data,
        interval=probe_interval,
        start_time=Instant.Epoch
    )
    
    in_flight_probe = Probe(
        target=server,
        metric="in_flight",
        data=in_flight_data,
        interval=probe_interval,
        start_time=Instant.Epoch
    )

    # B. INITIALIZATION
    sim = Simulation(
        sources=[event_source],
        entities=[server],
        probes=[queue_depth_probe, in_flight_probe],
        end_time=Instant.from_seconds(sim_duration)
    )

    # C. EXECUTION
    sim.run()

    # D. ASSERTIONS
    
    # Verify probes collected data
    queue_samples = queue_depth_data.values
    in_flight_samples = in_flight_data.values
    
    assert len(queue_samples) > 0, "Queue depth probe should have collected samples"
    assert len(in_flight_samples) > 0, "In-flight probe should have collected samples"
    
    queue_times = [s[0] for s in queue_samples]
    queue_values = [s[1] for s in queue_samples]
    
    in_flight_times = [s[0] for s in in_flight_samples]
    in_flight_values = [s[1] for s in in_flight_samples]
    
    # Verify no requests were dropped (queue was large enough)
    assert server.requests_dropped == 0, f"Expected no drops, got {server.requests_dropped}"
    
    # We expect ~120 requests to arrive (2/sec * 60 sec)
    total_arrived = server.requests_completed + len(server.queue) + server.active_workers
    assert 118 <= total_arrived <= 122, f"Expected ~120 requests arrived, got {total_arrived}"
    
    # Queue depth should grow over time
    max_queue_depth = max(queue_values)
    max_in_flight = max(in_flight_values)
    
    assert max_queue_depth >= 50, f"Expected max queue depth >= 50, got {max_queue_depth}"
    assert max_in_flight >= 50, f"Expected max in-flight >= 50, got {max_in_flight}"
    
    # Verify queue depth increased over time
    early_queue = [v for t, v in queue_samples if t < 10]
    late_queue = [v for t, v in queue_samples if t > 50]
    
    avg_early_queue = sum(early_queue) / len(early_queue) if early_queue else 0
    avg_late_queue = sum(late_queue) / len(late_queue) if late_queue else 0
    
    assert avg_late_queue > avg_early_queue, \
        f"Queue depth should grow. Early avg: {avg_early_queue:.1f}, Late avg: {avg_late_queue:.1f}"
    
    # E. VISUALIZATION
    
    # Save raw data as CSV
    _write_csv(
        test_output_dir / "queue_depth_samples.csv",
        header=["time_s", "queue_depth"],
        rows=[(t, v) for t, v in queue_samples]
    )
    _write_csv(
        test_output_dir / "in_flight_samples.csv",
        header=["time_s", "in_flight"],
        rows=[(t, v) for t, v in in_flight_samples]
    )
    
    # Plot 1: Queue depth and in-flight over time
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    
    # Queue depth plot
    ax1.step(queue_times, queue_values, where="post", linewidth=1.5, color="steelblue", label="Queue Depth")
    ax1.fill_between(queue_times, queue_values, step="post", alpha=0.3, color="steelblue")
    
    # Theoretical line
    theoretical_times = [0, sim_duration]
    theoretical_depth = [0, sim_duration]  # ~1 extra request/sec accumulates in queue
    ax1.plot(theoretical_times, theoretical_depth, 
             color="red", linestyle="--", linewidth=2, alpha=0.7,
             label="Theoretical growth (~1/sec)")
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Queue Depth")
    ax1.set_title("Queue Depth Under Overload (2 req/s arrival, 1 sec processing)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, sim_duration)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    
    # In-flight plot (queue + active workers)
    ax2.step(in_flight_times, in_flight_values, where="post", linewidth=1.5, color="darkorange", label="In-Flight (queue + processing)")
    ax2.fill_between(in_flight_times, in_flight_values, step="post", alpha=0.3, color="darkorange")
    ax2.step(queue_times, queue_values, where="post", linewidth=1.5, color="steelblue", alpha=0.7, label="Queue Depth")
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Total In-Flight vs Queue Depth")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, sim_duration)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(test_output_dir / "queue_depth_plot.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: Histogram of queue depths
    fig2, (ax_hist1, ax_hist2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    ax_hist1.hist(queue_values, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax_hist1.set_xlabel("Queue Depth")
    ax_hist1.set_ylabel("Frequency (# samples)")
    ax_hist1.set_title("Distribution of Queue Depths")
    ax_hist1.grid(True, alpha=0.3, axis="y")
    
    ax_hist2.hist(in_flight_values, bins=30, edgecolor="black", alpha=0.7, color="darkorange")
    ax_hist2.set_xlabel("In-Flight Count")
    ax_hist2.set_ylabel("Frequency (# samples)")
    ax_hist2.set_title("Distribution of In-Flight Requests")
    ax_hist2.grid(True, alpha=0.3, axis="y")
    
    fig2.tight_layout()
    fig2.savefig(test_output_dir / "queue_distributions.png", dpi=150)
    plt.close(fig2)
    
    print(f"\nSaved plots/data to: {test_output_dir}")
    print(f"  - Total queue samples: {len(queue_samples)}")
    print(f"  - Max queue depth: {max_queue_depth}")
    print(f"  - Max in-flight: {max_in_flight}")
    print(f"  - Requests completed: {server.requests_completed}")
    print(f"  - Requests dropped: {server.requests_dropped}")
    print(f"  - Early avg queue depth (t<10s): {avg_early_queue:.1f}")
    print(f"  - Late avg queue depth (t>50s): {avg_late_queue:.1f}")

