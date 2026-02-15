"""Retrying client with timeout-based retries against a queued server.

This example demonstrates a client that:
1. Sends requests to a queued server
2. Starts a timeout timer for each request
3. If the server completes before timeout: success, cancel the timer
4. If timeout fires before completion: retry the request

The server has exponential service times. With mean=100ms and timeout=70ms,
approximately 50% of requests will timeout (since P(X > t) = e^(-t/mean)).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RETRYING CLIENT SIMULATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐      Request        ┌─────────────────────────────────┐
   │   Source    │─────────────────────►│         Retrying Client         │
   │  (Poisson)  │                      │                                 │
   └─────────────┘                      │  • Tracks in-flight requests    │
                                        │  • Schedules timeout events     │
                                        │  • Retries on timeout           │
                                        │  • Records success on complete  │
                                        └───────────────┬─────────────────┘
                                                        │
                                           Send request │
                                                        ▼
                                        ┌─────────────────────────────────┐
                                        │         Queued Server           │
                                        │  ┌─────────┐   ┌─────────────┐  │
                                        │  │  Queue  │──►│   Server    │  │
                                        │  │ (FIFO)  │   │ (Exp ~100ms)│  │
                                        │  └─────────┘   └─────────────┘  │
                                        └───────────────┬─────────────────┘
                                                        │
                                           Completion   │
                                                        ▼
                                        ┌─────────────────────────────────┐
                                        │       Retrying Client           │
                                        │  (receives completion event)    │
                                        └─────────────────────────────────┘
```

## Timeout Math

For exponential service time with mean μ and timeout t:
- P(timeout) = P(X > t) = e^(-t/μ)
- With μ=100ms and t=70ms: P(timeout) = e^(-0.7) ≈ 0.50

"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    Probe,
    QueuedResource,
    Simulation,
    Source,
)


# =============================================================================
# Queued Server with Exponential Service Time
# =============================================================================


class QueuedServer(QueuedResource):
    """A queued server with exponential service times.

    Sends completion events back to the originating client.
    Tracks service time latency for each processed request.
    """

    def __init__(
        self,
        name: str,
        *,
        mean_service_time_s: float = 0.1,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.mean_service_time_s = mean_service_time_s
        self.stats_processed: int = 0

        # Latency time series: service time for each completed request
        self.completion_times: list[Instant] = []
        self.service_times_s: list[float] = []

    def has_capacity(self) -> bool:
        return True

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process request with exponential service time, then send completion."""
        service_time = random.expovariate(1.0 / self.mean_service_time_s)
        yield service_time, None

        self.stats_processed += 1

        # Record service time latency
        self.completion_times.append(self.now)
        self.service_times_s.append(service_time)

        # Send completion back to client
        client: Entity = event.context.get("client")
        if client is None:
            return []

        completion = Event(
            time=self.now,
            event_type="Completion",
            target=client,
            context={
                "request_id": event.context.get("request_id"),
                "original_created_at": event.context.get("created_at"),
                "service_time_s": service_time,
            },
        )
        return [completion]

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, service_times_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.service_times_s)


# =============================================================================
# Retrying Client
# =============================================================================


@dataclass
class InFlightRequest:
    """Tracks state for an in-flight request."""
    request_id: int
    created_at: Instant
    attempt: int
    timeout_event_id: int  # To match timeout events


class RetryingClient(Entity):
    """Client that sends requests with timeout-based retries.

    For each request:
    1. Send to server with a unique request_id
    2. Schedule a timeout event
    3. On completion: record success, ignore future timeout
    4. On timeout (if not completed): retry up to max_retries
    """

    def __init__(
        self,
        name: str,
        *,
        server: Entity,
        timeout_s: float = 0.07,
        max_retries: int = 3,
    ):
        super().__init__(name)
        self.server = server
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        # Track in-flight requests by request_id
        self._in_flight: dict[int, InFlightRequest] = {}
        self._next_timeout_id: int = 0

        # Stats
        self.stats_requests_received: int = 0
        self.stats_attempts_sent: int = 0
        self.stats_completions: int = 0
        self.stats_timeouts: int = 0
        self.stats_retries: int = 0
        self.stats_gave_up: int = 0

        # Latency tracking (from original creation to completion)
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.attempts_per_request: list[int] = []

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting.

        Latencies are end-to-end: from original request creation to successful completion.
        """
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)

    def goodput_time_series(self, bucket_size_s: float = 1.0) -> tuple[list[float], list[int]]:
        """Return (bucket_start_times_s, completions_per_bucket) for plotting goodput.

        Goodput is the number of successfully completed requests per time bucket.
        This represents useful work completed, excluding timed-out attempts.

        Args:
            bucket_size_s: Size of each time bucket in seconds.

        Returns:
            Tuple of (bucket_times, completion_counts) for plotting.
        """
        if not self.completion_times:
            return [], []

        buckets: dict[int, int] = defaultdict(int)
        for t in self.completion_times:
            bucket = int(t.to_seconds() / bucket_size_s)
            buckets[bucket] += 1

        sorted_buckets = sorted(buckets.keys())
        bucket_times = [b * bucket_size_s for b in sorted_buckets]
        counts = [buckets[b] for b in sorted_buckets]

        return bucket_times, counts

    def handle_event(self, event: Event) -> list[Event]:
        """Handle incoming request, completion, or timeout events."""
        event_type = event.event_type

        if event_type == "NewRequest":
            return self._handle_new_request(event)
        elif event_type == "Completion":
            return self._handle_completion(event)
        elif event_type == "Timeout":
            return self._handle_timeout(event)

        return []

    def _handle_new_request(self, event: Event) -> list[Event]:
        """Handle a new request from the source."""
        self.stats_requests_received += 1
        request_id = event.context.get("request_id", self.stats_requests_received)

        return self._send_request(request_id, event.time, attempt=1)

    def _send_request(self, request_id: int, created_at: Instant, attempt: int) -> list[Event]:
        """Send a request to the server and schedule a timeout."""
        self.stats_attempts_sent += 1
        self._next_timeout_id += 1
        timeout_id = self._next_timeout_id

        # Track this request
        self._in_flight[request_id] = InFlightRequest(
            request_id=request_id,
            created_at=created_at,
            attempt=attempt,
            timeout_event_id=timeout_id,
        )

        # Create request event for server
        server_request = Event(
            time=self.now,
            event_type="Request",
            target=self.server,
            context={
                "request_id": request_id,
                "created_at": created_at,
                "attempt": attempt,
                "client": self,
            },
        )

        # Schedule timeout
        timeout_event = Event(
            time=self.now + Instant.from_seconds(self.timeout_s),
            event_type="Timeout",
            target=self,
            context={
                "request_id": request_id,
                "timeout_id": timeout_id,
            },
        )

        return [server_request, timeout_event]

    def _handle_completion(self, event: Event) -> list[Event]:
        """Handle completion from server."""
        request_id = event.context.get("request_id")

        if request_id not in self._in_flight:
            # Already timed out and retried, or duplicate completion
            return []

        in_flight = self._in_flight.pop(request_id)
        self.stats_completions += 1

        # Record latency from original creation time
        original_created_at = event.context.get("original_created_at", in_flight.created_at)
        latency_s = (event.time - original_created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)
        self.attempts_per_request.append(in_flight.attempt)

        return []

    def _handle_timeout(self, event: Event) -> list[Event]:
        """Handle timeout - retry if the request is still in flight."""
        request_id = event.context.get("request_id")
        timeout_id = event.context.get("timeout_id")

        if request_id not in self._in_flight:
            # Already completed, ignore this timeout
            return []

        in_flight = self._in_flight[request_id]

        # Check if this timeout matches the current attempt
        if in_flight.timeout_event_id != timeout_id:
            # Stale timeout from a previous attempt
            return []

        self.stats_timeouts += 1

        # Remove from in-flight
        del self._in_flight[request_id]

        # Check retry limit
        if in_flight.attempt >= self.max_retries:
            self.stats_gave_up += 1
            return []

        # Retry
        self.stats_retries += 1
        return self._send_request(
            request_id,
            in_flight.created_at,
            attempt=in_flight.attempt + 1,
        )


# =============================================================================
# Event Provider for the Source
# =============================================================================


class ClientRequestProvider(EventProvider):
    """Generates request events targeting the retrying client."""

    def __init__(self, client: RetryingClient, *, stop_after: Instant | None = None):
        self._client = client
        self._stop_after = stop_after
        self._request_id: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self._request_id += 1
        return [
            Event(
                time=time,
                event_type="NewRequest",
                target=self._client,
                context={"request_id": self._request_id},
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the retrying client simulation."""
    client: RetryingClient
    server: QueuedServer
    queue_depth_data: Data
    requests_generated: int


def run_retrying_client_simulation(
    *,
    duration_s: float = 30.0,
    drain_s: float = 5.0,
    arrival_rate: float = 5.0,
    mean_service_time_s: float = 0.1,
    timeout_s: float = 0.07,
    max_retries: int = 3,
    probe_interval_s: float = 0.1,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the retrying client simulation.

    Args:
        duration_s: How long to generate load
        drain_s: Extra time for in-flight requests to complete
        arrival_rate: New requests per second
        mean_service_time_s: Mean server processing time
        timeout_s: Client timeout before retry
        max_retries: Maximum retry attempts per request
        probe_interval_s: Queue depth sampling interval
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Create server
    server = QueuedServer(
        name="Server",
        mean_service_time_s=mean_service_time_s,
    )

    # Create client
    client = RetryingClient(
        name="Client",
        server=server,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )

    # Create queue depth probe

    queue_probe, queue_depth_data = Probe.on(server, "depth", interval=probe_interval_s)

    # Create source
    stop_after = Instant.from_seconds(duration_s)
    provider = ClientRequestProvider(client, stop_after=stop_after)
    profile = ConstantRateProfile(rate=arrival_rate)
    arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[client, server],
        probes=[queue_probe],
    )
    sim.run()

    return SimulationResult(
        client=client,
        server=server,
        queue_depth_data=queue_depth_data,
        requests_generated=provider._request_id,
    )


def print_summary(result: SimulationResult, timeout_s: float, mean_service_s: float) -> None:
    """Print summary statistics."""
    import math

    client = result.client

    print("\n" + "=" * 60)
    print("RETRYING CLIENT SIMULATION RESULTS")
    print("=" * 60)

    # Expected timeout rate for exponential distribution
    expected_timeout_rate = math.exp(-timeout_s / mean_service_s)

    print(f"\nConfiguration:")
    print(f"  Mean service time: {mean_service_s * 1000:.0f}ms")
    print(f"  Client timeout: {timeout_s * 1000:.0f}ms")
    print(f"  Max retries: {client.max_retries}")
    print(f"  Expected timeout rate: {expected_timeout_rate * 100:.1f}%")

    print(f"\nRequest Flow:")
    print(f"  Requests generated (by source): {result.requests_generated}")
    print(f"  Requests received (by client):  {client.stats_requests_received}")
    print(f"  Attempts sent to server:        {client.stats_attempts_sent}")
    print(f"  Server processed:               {result.server.stats_processed}")

    print(f"\nOutcomes:")
    print(f"  Completions (success):  {client.stats_completions}")
    print(f"  Timeouts:               {client.stats_timeouts}")
    print(f"  Retries:                {client.stats_retries}")
    print(f"  Gave up (max retries):  {client.stats_gave_up}")

    if client.stats_attempts_sent > 0:
        actual_timeout_rate = client.stats_timeouts / client.stats_attempts_sent
        print(f"\nActual timeout rate: {actual_timeout_rate * 100:.1f}%")

    if client.stats_requests_received > 0:
        success_rate = client.stats_completions / client.stats_requests_received
        retry_amplification = client.stats_attempts_sent / client.stats_requests_received
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Retry amplification: {retry_amplification:.2f}x")

    # Goodput statistics
    goodput_times, goodput_counts = client.goodput_time_series(bucket_size_s=1.0)
    if goodput_counts:
        avg_goodput = sum(goodput_counts) / len(goodput_counts)
        max_goodput = max(goodput_counts)
        min_goodput = min(goodput_counts)
        print(f"\nGoodput (successful completions/second):")
        print(f"  Average: {avg_goodput:.1f} req/s")
        print(f"  Min:     {min_goodput} req/s")
        print(f"  Max:     {max_goodput} req/s")

    # Server latency statistics
    server = result.server
    if server.service_times_s:
        sorted_service_times = sorted(server.service_times_s)
        avg = sum(sorted_service_times) / len(sorted_service_times)
        p50 = sorted_service_times[len(sorted_service_times) // 2]
        p99_idx = int(len(sorted_service_times) * 0.99)
        p99 = sorted_service_times[min(p99_idx, len(sorted_service_times) - 1)]

        print(f"\nServer Service Time (processing only):")
        print(f"  Average: {avg * 1000:.1f}ms")
        print(f"  p50:     {p50 * 1000:.1f}ms")
        print(f"  p99:     {p99 * 1000:.1f}ms")
        print(f"  Max:     {max(sorted_service_times) * 1000:.1f}ms")

    if client.latencies_s:
        sorted_latencies = sorted(client.latencies_s)
        avg = sum(sorted_latencies) / len(sorted_latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

        print(f"\nEnd-to-End Latency (from creation to completion):")
        print(f"  Average: {avg * 1000:.1f}ms")
        print(f"  p50:     {p50 * 1000:.1f}ms")
        print(f"  p99:     {p99 * 1000:.1f}ms")
        print(f"  Max:     {max(sorted_latencies) * 1000:.1f}ms")

    if client.attempts_per_request:
        attempt_counts = defaultdict(int)
        for attempts in client.attempts_per_request:
            attempt_counts[attempts] += 1

        print(f"\nAttempts per successful request:")
        for attempts in sorted(attempt_counts.keys()):
            count = attempt_counts[attempts]
            pct = count / len(client.attempts_per_request) * 100
            print(f"  {attempts} attempt(s): {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    client = result.client
    server = result.server

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Queue depth over time
    ax = axes[0, 0]
    q_times = [t for (t, _) in result.queue_depth_data.values]
    q_depths = [v for (_, v) in result.queue_depth_data.values]
    ax.plot(q_times, q_depths, 'b-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Queue Depth')
    ax.set_title('Server Queue Depth Over Time')
    ax.grid(True, alpha=0.3)

    # Server service time over time (binned by second)
    ax = axes[0, 1]
    server_times_s, server_latencies_s = server.latency_time_series_seconds()

    # Bin service times by second and compute average
    service_buckets: dict[int, list[float]] = defaultdict(list)
    for t, lat in zip(server_times_s, server_latencies_s):
        bucket = int(t)
        service_buckets[bucket].append(lat)

    service_bucket_times = sorted(service_buckets.keys())
    service_bucket_avg = [sum(service_buckets[b]) / len(service_buckets[b]) * 1000
                         for b in service_bucket_times]

    ax.plot(service_bucket_times, service_bucket_avg, 'g-', linewidth=1.5, marker='o', markersize=3)
    ax.axhline(y=server.mean_service_time_s * 1000, color='r', linestyle='--',
               label=f'Expected Mean ({server.mean_service_time_s * 1000:.0f}ms)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Avg Service Time (ms)')
    ax.set_title('Server Service Time Over Time (1s avg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Client end-to-end latency over time (binned by second)
    ax = axes[1, 0]
    client_times_s, client_latencies_s = client.latency_time_series_seconds()

    # Bin latencies by second and compute average
    latency_buckets: dict[int, list[float]] = defaultdict(list)
    for t, lat in zip(client_times_s, client_latencies_s):
        bucket = int(t)
        latency_buckets[bucket].append(lat)

    bucket_times = sorted(latency_buckets.keys())
    bucket_avg_latencies = [sum(latency_buckets[b]) / len(latency_buckets[b]) * 1000
                           for b in bucket_times]

    ax.plot(bucket_times, bucket_avg_latencies, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Avg Latency (ms)')
    ax.set_title('Client End-to-End Latency Over Time (1s avg)')
    ax.grid(True, alpha=0.3)

    # Goodput over time
    ax = axes[1, 1]
    goodput_times, goodput_counts = client.goodput_time_series(bucket_size_s=1.0)
    ax.plot(goodput_times, goodput_counts, 'g-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Completions / second')
    ax.set_title('Goodput Over Time (successful completions per second)')
    ax.grid(True, alpha=0.3)

    # Latency histograms (server vs client)
    ax = axes[2, 0]
    ax.hist([lat * 1000 for lat in server_latencies_s], bins=50, alpha=0.5,
            label='Server Service Time', color='green', edgecolor='darkgreen')
    ax.hist([lat * 1000 for lat in client_latencies_s], bins=50, alpha=0.5,
            label='Client End-to-End', color='blue', edgecolor='darkblue')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Latency Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Attempts distribution
    ax = axes[2, 1]
    attempt_counts = defaultdict(int)
    for attempts in client.attempts_per_request:
        attempt_counts[attempts] += 1

    attempts_list = sorted(attempt_counts.keys())
    counts = [attempt_counts[a] for a in attempts_list]
    ax.bar(attempts_list, counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Attempts')
    ax.set_ylabel('Successful Requests')
    ax.set_title('Attempts per Successful Request')
    ax.set_xticks(attempts_list)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_dir / "retrying_client_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'retrying_client_results.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrying client simulation")
    parser.add_argument("--duration", type=float, default=30.0, help="Load duration (s)")
    parser.add_argument("--drain", type=float, default=5.0, help="Drain time (s)")
    parser.add_argument("--rate", type=float, default=5.0, help="Arrival rate (req/s)")
    parser.add_argument("--service-time", type=float, default=0.1, help="Mean service time (s)")
    parser.add_argument("--timeout", type=float, default=0.07, help="Client timeout (s)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/retrying_client", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running retrying client simulation...")
    print(f"  Duration: {args.duration}s + {args.drain}s drain")
    print(f"  Arrival rate: {args.rate} req/s")
    print(f"  Mean service time: {args.service_time * 1000:.0f}ms")
    print(f"  Timeout: {args.timeout * 1000:.0f}ms")

    result = run_retrying_client_simulation(
        duration_s=args.duration,
        drain_s=args.drain,
        arrival_rate=args.rate,
        mean_service_time_s=args.service_time,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        seed=seed,
    )

    print_summary(result, args.timeout, args.service_time)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
