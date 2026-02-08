# GC-Induced Metastable Collapse Example Design

> **Status:** Design complete, ready for implementation

## Overview

Create `examples/gc_caused_collapse.py` demonstrating how a GC pause exceeding client timeout triggers non-recoverable metastable collapse, even at moderate utilization.

**Key Insight**: Unlike load-spike metastability (which requires high utilization), GC-induced collapse occurs at moderate load because:
1. GC pause > client timeout causes ALL in-flight requests to timeout
2. Retries create a "retry storm" that overwhelms recovery capacity
3. Queue compounds faster than it drains, leading to collapse

## Architecture

```
    REQUEST FLOW (70% utilization baseline)

    ┌─────────────┐      Request        ┌─────────────────────────────────┐
    │   Source    │─────────────────────►│         Retrying Client         │
    │  (Poisson)  │                      │                                 │
    │  7 req/s    │                      │  Timeout: 200ms                 │
    └─────────────┘                      │  Max Retries: 3                 │
                                         │  Retry Delay: 50ms              │
                                         └───────────────┬─────────────────┘
                                                         │
                                                         ▼
                                         ┌─────────────────────────────────┐
                                         │        GC-Aware Server          │
                                         │  ┌─────────┐   ┌─────────────┐  │
                                         │  │  Queue  │──►│   Worker    │  │
                                         │  │ (FIFO)  │   │ (Exp ~100ms)│  │
                                         │  └─────────┘   └─────────────┘  │
                                         │         ↑                       │
                                         │    GC Pause: 500ms every 10s   │
                                         └───────────────┬─────────────────┘
                                                         │
                                                         ▼
                                         ┌─────────────────────────────────┐
                                         │              Sink               │
                                         │  (Tracks latency, goodput)      │
                                         └─────────────────────────────────┘
```

## Collapse Mechanism

**Before GC** (stable):
- Arrival: 7 req/s, Service: 10 req/s
- 30% headroom, queue nearly empty

**During GC** (500ms):
- Server STOPS (processing = 0)
- ~3.5 new requests queue up
- ALL in-flight timeout (500ms > 200ms timeout)
- Clients retry timed-out requests

**After GC**:
- Queue = original arrivals + retry storm
- Drain rate = 10 - 7 = 3 req/s (slow)
- Before queue clears, NEXT GC hits
- Queue compounds → collapse

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Arrival rate | 7 req/s | 70% utilization - clearly below saturation |
| Service rate | 10 req/s | Mean service time 100ms |
| Client timeout | 200ms | 2x mean service - reasonable for normal ops |
| GC duration | 500ms | 2.5x timeout - guarantees all in-flight timeout |
| GC interval | 10s | Short enough to observe multiple GCs |
| Max retries | 3 | Amplifies load but finite |
| Retry delay | 50ms | Fixed, contributes to retry storm |

## Files to Create/Modify

### New: `examples/gc_caused_collapse.py`

**Structure** (following existing patterns):
1. Docstring with ASCII architecture diagram
2. `GCServer` class extending `QueuedResource`
3. `RetryingClientWithStats` class (adapted from retrying_client.py)
4. `GCCollapseProfile` dataclass
5. `SimulationResult` and `ComparisonResult` dataclasses
6. `run_gc_collapse_simulation()` function
7. `run_comparison()` function (with retries vs without)
8. `visualize_results()` function
9. `print_summary()` function
10. Entry point with argparse

### New: `tests/integration/test_gc_caused_collapse.py`

Integration test that:
- Runs comparison scenario
- Asserts collapse occurs with retries
- Asserts recovery occurs without retries
- Saves visualization artifacts

## Key Classes

### GCServer

```python
class GCServer(QueuedResource):
    """Server that experiences periodic stop-the-world GC pauses.

    Simulates stop-the-world garbage collection where all processing halts
    for the duration of the GC pause. This can trigger metastable failure
    if GC duration exceeds client timeout.

    GC Schedule:
    - First GC at `gc_start_time`
    - Subsequent GCs at `gc_start_time + n * gc_interval`
    - Each GC pauses processing for `gc_duration`
    """

    def __init__(
        self,
        name: str,
        *,
        mean_service_time_s: float = 0.1,
        gc_interval_s: float = 10.0,
        gc_duration_s: float = 0.5,
        gc_start_time_s: float = 10.0,
        downstream: Entity | None = None,
    ): ...

    def _get_gc_pause_remaining(self) -> float:
        """Calculate remaining GC pause time if currently in a GC window.

        Returns 0 if not in a GC pause, otherwise returns seconds remaining.
        """
        current_time = self.now.to_seconds()

        if current_time < self.gc_start_time_s:
            return 0.0

        time_since_first_gc = current_time - self.gc_start_time_s
        gc_cycle_position = time_since_first_gc % self.gc_interval_s

        if gc_cycle_position < self.gc_duration_s:
            return self.gc_duration_s - gc_cycle_position

        return 0.0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        # Check if we're entering a GC pause
        gc_remaining = self._get_gc_pause_remaining()
        if gc_remaining > 0:
            self.stats_gc_pauses += 1
            yield gc_remaining, None  # Server completely stops

        # Normal service time
        service_time = random.expovariate(1.0 / self.mean_service_time_s)
        yield service_time, None

        # Return completion event
        if self.downstream is None:
            return []

        return [Event(
            time=self.now,
            event_type="Completed",
            target=self.downstream,
            context=event.context,
        )]
```

### RetryingClientWithStats

Adapted from `examples/retrying_client.py` with additional tracking for:
- Per-request attempt counts over time
- Retry amplification ratio over time
- In-flight requests over time

```python
class RetryingClientWithStats(Entity):
    """Client with timeout-based retries and detailed statistics.

    Tracks:
    - Per-request attempt counts
    - Timeout timing relative to request creation
    - Running averages for retry amplification
    """

    def __init__(
        self,
        name: str,
        *,
        server: Entity,
        timeout_s: float = 0.2,
        max_retries: int = 3,
        retry_delay_s: float = 0.05,
        retry_enabled: bool = True,  # Toggle for comparison
    ):
        super().__init__(name)
        self.server = server
        self.timeout_s = timeout_s
        self.max_retries = max_retries if retry_enabled else 0
        self.retry_delay_s = retry_delay_s

        # Core tracking
        self._in_flight: dict[int, InFlightRequest] = {}
        self._next_timeout_id: int = 0

        # Stats
        self.stats_requests_received: int = 0
        self.stats_attempts_sent: int = 0
        self.stats_completions: int = 0
        self.stats_timeouts: int = 0
        self.stats_retries: int = 0
        self.stats_gave_up: int = 0

        # Latency tracking
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.attempts_per_request: list[int] = []

        # Time series for visualization
        self.attempts_by_time: list[tuple[float, int]] = []  # (time_s, attempt_count)
```

## Simulation Phases

1. **Warm-up (0-10s)**: Baseline at 70% utilization, no GC yet
2. **First GC (10-10.5s)**: Triggers initial retry storm
3. **Recovery Attempt (10.5-20s)**: Server tries to drain queue
4. **GC Cycles (20-60s)**: Repeated GCs compound the problem
5. **Observation (60-100s)**: Sustained collapse (with retries) vs recovery (without)

## Comparison Scenarios

Run two scenarios with identical random seed:

1. **WITH RETRIES**: `retry_enabled=True, max_retries=3`
   - Expected: Metastable collapse, queue grows unbounded

2. **WITHOUT RETRIES**: `retry_enabled=False` (or `max_retries=0`)
   - Expected: Brief spikes, full recovery between GCs

## Visualizations

**Figure 1: Overview (3 subplots, shared x-axis)**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Plot 1: GC Events Timeline                                                 │
│  - Vertical bars showing GC pause windows (shaded regions)                  │
│  - Annotations for GC duration                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Plot 2: Queue Depth Over Time                                              │
│  - With retries (red) vs Without retries (green)                            │
│  - Shows queue compounding with retries                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Plot 3: Goodput Over Time                                                  │
│  - With retries vs Without retries                                          │
│  - Shows goodput collapse with retries                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Figure 2: Analysis (2x2 grid)**
```
┌───────────────────────────────────┬───────────────────────────────────────┐
│  Plot 1: Retry Amplification      │  Plot 2: Latency Distribution         │
│  - attempts/request over time     │  - Histogram: with vs without retries │
│  - Shows increasing amplification │  - Shows tail latency explosion       │
├───────────────────────────────────┼───────────────────────────────────────┤
│  Plot 3: Cumulative Timeouts      │  Plot 4: Summary Statistics           │
│  - Cumulative timeout count       │  - Text box with key metrics          │
│  - Inflection point at collapse   │  - Comparison table                   │
└───────────────────────────────────┴───────────────────────────────────────┘
```

## Expected Console Output

```
=======================================================================
GC-INDUCED METASTABLE COLLAPSE SIMULATION
=======================================================================

Configuration:
  Service capacity: 10 req/s (mean service time = 100ms)
  Arrival rate: 7 req/s (70% utilization)
  Client timeout: 200ms
  GC interval: 10s
  GC duration: 500ms (2.5x timeout)
  Max retries: 3

SCENARIO 1: WITH RETRIES
-----------------------------------------------------------------------
  Requests generated: 840
  Successful completions: 523
  Timeouts: 1247
  Retry amplification: 2.84x
  Final queue depth: 47
  Average latency: 3.2s (expected: 0.1s)
  p99 latency: 12.4s

SCENARIO 2: WITHOUT RETRIES (NoRetry policy)
-----------------------------------------------------------------------
  Requests generated: 840
  Successful completions: 798
  Timeouts: 42
  Retry amplification: 1.00x
  Final queue depth: 0
  Average latency: 0.18s
  p99 latency: 0.62s

ANALYSIS:
-----------------------------------------------------------------------
The GC-induced collapse demonstrates how retries amplify transient
failures into sustained metastable failure:

1. With retries: 2.84x request amplification caused queue to compound
   after each GC, never recovering to steady state.

2. Without retries: System experiences brief latency spikes during GC
   but returns to steady state between GC events.

3. Recovery would require reducing load to ~30% utilization (70%
   reduction from normal), matching the theoretical requirement
   to overcome 2.84x amplification.
=======================================================================
```

## Verification

After implementation, run:
```bash
# Run example
python examples/gc_caused_collapse.py --output output/gc_collapse

# Run tests
pytest tests/integration/test_gc_caused_collapse.py -v
```

Expected outcomes:
- WITH retries: Final queue depth > 50, success rate < 70%
- WITHOUT retries: Final queue depth = 0, success rate > 95%
- Visualizations clearly show divergent behavior after first GC

## Reference Files

- `examples/m_m_1_queue.py` - Primary pattern for example structure and visualization
- `examples/retrying_client.py` - Client with timeout/retry pattern to adapt
- `happysimulator/components/queued_resource.py` - Base class for GCServer
- `happysimulator/components/client/retry.py` - RetryPolicy implementations
- `tests/integration/test_resilience_visualization.py` - Test pattern with visualization output

## Implementation Checklist

- [ ] Create `GCServer` class extending `QueuedResource`
- [ ] Create `RetryingClientWithStats` with retry toggle
- [ ] Create `LatencyTrackingSink` (or reuse from m_m_1_queue.py)
- [ ] Create `GCCollapseProfile` dataclass
- [ ] Implement `run_gc_collapse_simulation()` function
- [ ] Implement `run_comparison()` for with/without retries
- [ ] Implement visualization functions
- [ ] Implement `print_summary()` with analysis
- [ ] Add argparse CLI
- [ ] Create integration test
