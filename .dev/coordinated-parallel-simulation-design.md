# Coordinated Parallel Simulation Design

## Overview

Extend `ParallelSimulation` with barrier-based time synchronization so that
partitions can exchange events while running in parallel. Users declare
`PartitionLink`s between partitions, each with a minimum latency. The minimum
latency across all links determines the synchronization window size `W`. All
partitions advance in lockstep through time windows of size `W`, exchanging
cross-partition events at each barrier.

This builds on the Phase 1 `ParallelSimulation` (commit 67a8198), which runs
fully independent partitions. When no links are declared, behavior is identical
to Phase 1 — zero overhead.

## Motivation

Many real-world systems have *mostly* independent subsystems that occasionally
communicate:

- **Multi-region deployments**: US-East and EU-West run independently but
  replicate writes cross-region with 50ms+ network latency
- **Federated services**: Independent microservices that make occasional RPC
  calls with known minimum round-trip times
- **Sharded databases with cross-shard queries**: Each shard processes its own
  keyspace but periodically coordinates for distributed transactions
- **Multi-datacenter consensus**: Raft/Paxos nodes in different datacenters
  with fixed minimum network delay

Phase 1 forbids any cross-partition interaction. Users who need *any*
communication must fall back to a single-threaded `Simulation`, losing all
parallelism. Barrier-based synchronization enables parallelism for the large
class of systems where cross-partition communication has a natural minimum
latency.

## Background: Barrier-Based Synchronization

### The Core Invariant

All partitions advance in lockstep through time windows of size `W`:

```
Window [0, W)     → all partitions process in parallel → barrier → exchange
Window [W, 2W)    → all partitions process in parallel → barrier → exchange
Window [2W, 3W)   → ...
```

During window `[T, T+W)`, partition A might generate an event targeting
partition B. Because the declared minimum cross-partition latency is `>= W`,
that event's timestamp is at earliest `T + W` — outside the current window.
Partition B doesn't need it yet.

```
Window [0.0, 1.0)                             min_latency = 1.0s

  US:  ──●────●──●───────●──────────|
         0.1  0.3 0.4    0.7        1.0
               │
               │ event for EU at t ≥ 1.3
               │ (0.3 + 1.0s min latency)
               ▼
         [outbox — not delivered yet]

  EU:  ────●───────●──●────────●────|
           0.2     0.5 0.6    0.9   1.0

── barrier ── exchange outboxes ──

Window [1.0, 2.0)

  EU:  ────●──●─────●──────────●────|
           1.2 1.3  1.6        1.9  2.0
               ▲
               └── delivered from US outbox
```

**Correctness guarantee**: when `W <= min_latency` across all links, the
barrier approach produces results identical to sequential single-threaded
execution. No approximation, no timing error.

### Why Not Other Approaches

| Approach | Status | Reason |
|----------|--------|--------|
| **Barrier-based** | Chosen | No rollback, generator-compatible, simple, Python-friendly |
| **Conservative (Chandy-Misra-Bryant)** | Possible future | Higher implementation complexity, null message overhead; viable if denser interaction is needed |
| **Optimistic (Time Warp)** | Not feasible | Requires checkpointing/rolling back Python generators — impossible with the current entity model |

## API Design

### PartitionLink

A declaration that two partitions can exchange events, with a minimum latency
bound that guarantees temporal separation.

```python
@dataclass(frozen=True)
class PartitionLink:
    """Declares a cross-partition communication channel.

    The min_latency is a contract: any event crossing from source_partition
    to dest_partition must have a timestamp at least min_latency seconds
    after the sending entity's current time. This bound determines the
    synchronization window size.

    The latency distribution (optional) models actual transmission delay.
    When provided, it is applied to cross-partition events automatically
    by the coordinator. When omitted, events are delivered at their
    original timestamp (the user's model must already include the delay).

    Args:
        source_partition: Name of the sending partition.
        dest_partition: Name of the receiving partition.
        min_latency: Minimum seconds between send time and delivery time.
            Must be > 0. Determines the synchronization window size.
        latency: Optional latency distribution applied to cross-partition
            events. Must always sample values >= min_latency.
        packet_loss: Probability of dropping a cross-partition event [0, 1).
    """
    source_partition: str
    dest_partition: str
    min_latency: float
    latency: LatencyDistribution | None = None
    packet_loss: float = 0.0

    def __post_init__(self):
        if self.min_latency <= 0:
            raise ValueError(
                f"min_latency must be > 0, got {self.min_latency}"
            )
        if not (0.0 <= self.packet_loss < 1.0):
            raise ValueError(
                f"packet_loss must be in [0, 1), got {self.packet_loss}"
            )

    @staticmethod
    def bidirectional(
        partition_a: str,
        partition_b: str,
        min_latency: float,
        latency: LatencyDistribution | None = None,
        packet_loss: float = 0.0,
    ) -> tuple[PartitionLink, PartitionLink]:
        """Create a pair of links for bidirectional communication."""
        return (
            PartitionLink(partition_a, partition_b, min_latency, latency, packet_loss),
            PartitionLink(partition_b, partition_a, min_latency, latency, packet_loss),
        )
```

### Extended ParallelSimulation

```python
sim = ParallelSimulation(
    partitions=[
        SimulationPartition(
            name="us",
            entities=[us_lb, us_server, us_queue, us_sink],
            sources=[us_traffic],
        ),
        SimulationPartition(
            name="eu",
            entities=[eu_lb, eu_server, eu_queue, eu_sink],
            sources=[eu_traffic],
        ),
    ],
    links=[                                   # NEW — optional
        *PartitionLink.bidirectional(
            "us", "eu",
            min_latency=0.05,                 # 50ms cross-region floor
        ),
    ],
    duration=100.0,
)

summary = sim.run()
```

When `links` is empty (default), behavior is identical to Phase 1:
fire-and-forget parallel execution with no synchronization. When links are
present, the coordinator uses windowed execution with barrier synchronization.

### Constructor Changes

```python
class ParallelSimulation:
    def __init__(
        self,
        partitions: list[SimulationPartition],
        start_time: Instant | None = None,
        end_time: Instant | None = None,
        duration: float | None = None,
        max_workers: int | None = None,
        links: list[PartitionLink] | None = None,       # NEW
        window_size: float | None = None,                # NEW — override
    ):
```

- `links`: Cross-partition communication declarations. When present, enables
  coordinated windowed execution.
- `window_size`: Override the auto-computed window size. Must be
  `<= min(link.min_latency for link in links)`. Useful for finer-grained
  synchronization when the user wants tighter coordination at the cost of
  more barriers.

### Cross-Partition Event Patterns

Entities send cross-partition events through the existing Network component
or by directly creating events with appropriate timestamps. The coordinator
handles routing transparently.

**Pattern 1: Via Network (recommended)**

The user's model already has network links between regions. The network
adds latency naturally:

```python
network = Network("cross-region")
network.add_bidirectional_link(
    us_server, eu_server,
    cross_region_network("us-eu"),   # latency >= 50ms
)

# us_server's handle_event:
def handle_event(self, event):
    yield 0.1  # process locally
    return [network.send(self, eu_server, "Replicate", payload={...})]
```

The Network entity lives in the source partition. Its `handle_event()`
yields the network latency, then returns an event targeting `eu_server`.
The coordinator intercepts this cross-partition event and routes it through
the outbox.

**Pattern 2: Direct with explicit delay**

For simpler cases without a Network entity:

```python
def handle_event(self, event):
    yield 0.1
    return [Event(
        time=self.now + Duration.from_seconds(0.05),  # >= min_latency
        event_type="Replicate",
        target=eu_server,
    )]
```

The coordinator validates that the event's timestamp respects the
declared `min_latency`. If not, it raises `RuntimeError`.

## Internal Architecture

### Execution Modes

```
ParallelSimulation.run()
│
├── No links? ──> Phase 1 path (unchanged)
│   └── ThreadPoolExecutor: fire-and-forget parallel execution
│
└── Has links? ──> Phase 2 path (NEW)
    └── WindowedCoordinator: barrier-synchronized execution
```

### WindowedCoordinator

The coordinator drives the windowed execution loop from the main thread:

```
WindowedCoordinator
├── _window_size: float              # min(link.min_latency) or override
├── _outboxes: dict[str, list[Event]]  # partition_name -> pending remote events
├── _link_map: dict[(src, dst), PartitionLink]
│
└── run()
    ├── for each window [T, T+W):
    │   ├── Phase 1: EXECUTE (parallel)
    │   │   └── submit all partition.run_window(T, T+W) to thread pool
    │   │       └── each partition processes local events in [T, T+W)
    │   │           └── cross-partition events diverted to outbox
    │   │
    │   ├── Phase 2: EXCHANGE (coordinator thread)
    │   │   ├── collect all outboxes
    │   │   ├── validate min_latency constraints
    │   │   ├── apply latency distribution (if configured on link)
    │   │   ├── apply packet loss (if configured)
    │   │   └── inject events into destination partition heaps
    │   │
    │   └── Phase 3: ADVANCE
    │       └── T += W
    │
    └── merge summaries
```

### Windowed Simulation Execution

Each partition's `Simulation` needs to support running for a single time
window and then pausing. Two approaches:

**Option A: Modify `_end_time` per window (recommended)**

Add a `run_window()` method to `Simulation` that temporarily sets
`_end_time` to the window boundary and runs:

```python
# On Simulation — new method for windowed execution
def run_window(self, window_end: Instant) -> None:
    """Run until window_end, keeping the simulation resumable.

    Unlike run(), this does not mark the simulation as completed.
    The heap and entity state are preserved for the next window.
    """
    saved_end = self._end_time
    self._end_time = window_end
    try:
        if not self._is_running:
            # First window — full initialization
            self._wall_start = _time.monotonic()
            self._is_running = True
            self._event_heap.set_current_time(self._current_time)
            _set_active_context(self._event_heap, self._clock)
        self._run_loop_windowed()
    finally:
        self._end_time = saved_end
```

`_run_loop_windowed()` is similar to `_run_loop_fast()` but does NOT set
`_is_running = False` at the end — the simulation stays "open" for the
next window.

**Option B: TimeBreakpoint**

Use the existing control surface: add a `TimeBreakpoint` at each window
boundary. When hit, the simulation pauses and returns control.

Downside: forces lazy creation of the control surface and adds per-event
breakpoint checking overhead.

**Recommendation**: Option A — dedicated `run_window()` avoids control
surface overhead and is semantically clearer.

### Cross-Partition Event Router

Replace the Phase 1 partition guard (which raises `RuntimeError`) with a
router that diverts cross-partition events to an outbox.

The router intercepts events **after `event.invoke()`** and **before
`heap.push()`** in the run loop:

```python
def _run_loop_windowed(self):
    # ... same as _run_loop_fast, except:

    new_events = event.invoke()

    if new_events:
        if self._event_router is not None:
            local, remote = self._event_router(new_events)
            if local:
                heap_push(local)
            # remote events are already in the outbox
        else:
            heap_push(new_events)
```

The router factory:

```python
def _make_event_router(
    partition_name: str,
    entity_ids: frozenset[int],
    outbox: list[Event],
    allowed_targets: frozenset[int],  # entity IDs in linked partitions
) -> Callable:
    """Create a router that separates local and cross-partition events."""
    from happysimulator.core.callback_entity import CallbackEntity

    def router(events: list[Event]) -> tuple[list[Event], list[Event]]:
        local = []
        for event in events:
            target = event.target
            if isinstance(target, CallbackEntity):
                local.append(event)
                continue

            target_id = id(target)
            if target_id in entity_ids:
                local.append(event)
            elif target_id in allowed_targets:
                outbox.append(event)
            else:
                target_name = getattr(target, "name", repr(target))
                raise RuntimeError(
                    f"Partition '{partition_name}': event targets "
                    f"'{target_name}' which is not in this partition "
                    f"and has no PartitionLink to reach it."
                )
        return local, []  # remote already in outbox

    return router
```

Events targeting a linked partition go to the outbox. Events targeting an
unlinked partition raise `RuntimeError` (same as Phase 1 — no silent
misrouting).

### Event Exchange at Barrier

After all partitions complete a window, the coordinator processes outboxes:

```python
def _exchange_events(self) -> int:
    """Route cross-partition events from outboxes to destination heaps.

    Returns total number of events exchanged.
    """
    total = 0
    for partition_name, outbox in self._outboxes.items():
        for event in outbox:
            dest_partition = self._entity_to_partition[id(event.target)]
            link = self._link_map.get((partition_name, dest_partition))

            if link is None:
                raise RuntimeError(
                    f"No PartitionLink from '{partition_name}' to "
                    f"'{dest_partition}'"
                )

            # Validate min_latency constraint
            send_time = ...  # tracked when event entered outbox
            if (event.time - send_time).to_seconds() < link.min_latency:
                raise RuntimeError(
                    f"Cross-partition event violates min_latency: "
                    f"event at {event.time} sent at {send_time}, "
                    f"min_latency={link.min_latency}s"
                )

            # Apply latency distribution if configured
            if link.latency is not None:
                delay = link.latency.sample()
                event = Event(
                    time=send_time + Duration.from_seconds(delay),
                    event_type=event.event_type,
                    target=event.target,
                    context=event.context,
                )

            # Apply packet loss
            if link.packet_loss > 0 and random.random() < link.packet_loss:
                continue  # drop

            # Inject into destination partition's heap
            self._simulations[dest_partition].schedule(event)
            total += 1

        outbox.clear()

    return total
```

## Time Model

### Window Size Derivation

The window size is automatically computed as:

```python
W = min(link.min_latency for link in links)
```

This is the largest safe window: any cross-partition event generated at time
`T` has timestamp `>= T + W`, so it falls outside the current window `[T, T+W)`.

Users can override with a smaller window via `window_size=` for tighter
synchronization at the cost of more barriers.

### Window Boundary Behavior

At the end of window `[T, T+W)`:

1. **In-flight generators**: A generator that started at `T+W-0.02` and
   yielded 0.1 has a `ProcessContinuation` at `T+W+0.08`. This continuation
   stays in the local heap — it's a local event targeting the same entity.
   It will be processed in the next window.

2. **Events past the boundary**: Any events in the heap with timestamps
   `>= T+W` remain in the heap for the next window. This includes both
   local continuation events and not-yet-popped local events.

3. **Cross-partition events**: All cross-partition events generated during
   the window have timestamps `>= T+W` (guaranteed by the min_latency
   constraint). They sit in the outbox until the barrier phase delivers
   them.

### Correctness Argument

**Claim**: When `W <= min(link.min_latency)`, the windowed execution produces
results identical to sequential single-threaded execution.

**Proof sketch**:
- Within a window, each partition processes events in strict timestamp order
  (same as sequential).
- No cross-partition event can have a timestamp within the current window
  (min_latency guarantee), so no partition ever misses an event it should have
  seen.
- Cross-partition events are delivered at the barrier and injected into the
  destination heap before the next window starts, so they are available for
  processing in the correct window.
- The only difference from sequential execution is the interleaving of
  partition-local events, which is irrelevant because partitions share no
  state.

## Validation

### Init-Time Checks

In addition to Phase 1 validation (no duplicate entities, no cross-references
beyond linked partitions):

1. **Link partition names exist**: Both `source_partition` and `dest_partition`
   must match declared partition names.
2. **Positive min_latency**: All links must have `min_latency > 0`.
3. **Window size valid**: If user-provided, `window_size <= min(min_latency)`.
4. **Link completeness**: If partition A has entities referencing partition B
   entities (detected by attribute walking), a link from A to B must exist.
   Raises `ValueError` with guidance if missing.
5. **No shared entities**: Same as Phase 1 — no entity in multiple partitions.

### Runtime Checks

1. **Min latency enforcement**: When a cross-partition event enters the outbox,
   validate `event.time - current_partition_time >= link.min_latency`. Raises
   `RuntimeError` if violated.
2. **Unlinked cross-partition events**: Events targeting a partition with no
   declared link raise `RuntimeError` (same as Phase 1).

### Relaxed Validation for Linked Partitions

Phase 1 raises `ValueError` if entity A (partition "us") holds a reference to
entity B (partition "eu") via an attribute. With links, this is allowed:
attribute walking skips cross-references between linked partitions. The runtime
router handles the actual event routing.

## Results & Metrics

### Extended ParallelSimulationSummary

```python
@dataclass
class ParallelSimulationSummary:
    # ... existing fields ...

    # NEW — coordination metrics
    total_windows: int                           # number of barrier cycles
    total_cross_partition_events: int             # events exchanged
    cross_partition_events_per_window: float       # avg events per barrier
    window_size_s: float                          # W used
    barrier_overhead_seconds: float               # wall time spent in barriers
    coordination_efficiency: float                # 1 - (barrier_time / wall_time)
```

These metrics help users understand the overhead of coordination and tune
window size.

### Per-Window Metrics (Optional)

For debugging, the coordinator can record per-window stats:

```python
@dataclass
class WindowMetrics:
    window_start: float
    window_end: float
    events_exchanged: int
    partition_wall_times: dict[str, float]   # per-partition time in this window
    barrier_wall_time: float                  # time spent exchanging
```

Available via `summary.window_metrics` when tracing is enabled.

## Constraints & Limitations

| Constraint | Reason | Future Relaxation |
|-----------|--------|-------------------|
| min_latency must be > 0 | Zero latency means zero window size (sequential) | Conservative PDES for zero-latency links |
| No cross-partition SimFuture | resolve() pushes to partition-local heap | Event-based request-response pattern instead |
| No shared Resource/Mutex | Non-atomic state across threads | Distributed resource protocol via links |
| Events must respect min_latency | Correctness of barrier approach | Runtime error if violated |
| Window overhead scales with 1/W | More barriers = more synchronization | Adaptive window sizing |
| Cross-partition events delivered at window boundary | Barrier-based, not continuous | Conservative PDES for continuous delivery |

### Cross-Partition SimFuture Workaround

SimFuture cannot cross partition boundaries (resolve() is partition-local).
Instead, use the request-response event pattern:

```python
# In source partition entity:
def handle_event(self, event):
    # Send request to remote entity with reply-to context
    return [Event(
        time=self.now + Duration.from_seconds(0.05),
        event_type="Query",
        target=remote_entity,
        context={"reply_to": self, "request_id": uuid4()},
    )]

# In destination partition entity:
def handle_event(self, event):
    if event.event_type == "Query":
        result = ...  # compute
        return [Event(
            time=self.now + Duration.from_seconds(0.05),
            event_type="QueryResponse",
            target=event.context["reply_to"],
            context={"request_id": event.context["request_id"], "result": result},
        )]
```

Both events cross partition boundaries and are routed through the barrier.
The round-trip latency is `>= 2 * min_latency` (one crossing each way).

## Example: Cross-Region Database Replication

Two database regions with async replication. Each region processes independent
client traffic. Writes are replicated cross-region with 50ms+ network latency.

```python
from happysimulator import (
    Source, Event, Sink, Duration, QueuedResource, FIFOQueue,
)
from happysimulator.parallel import (
    ParallelSimulation, SimulationPartition, PartitionLink,
)
from happysimulator.instrumentation import LatencyTracker, Probe


class DatabaseServer(QueuedResource):
    """Processes queries locally. Replicates writes to a remote replica."""

    def __init__(self, name, sink, replica=None):
        super().__init__(name, policy=FIFOQueue())
        self.sink = sink
        self.replica = replica  # entity in another partition (or None)
        self._replication_count = 0

    def handle_queued_event(self, event):
        yield 0.005  # 5ms query processing

        # Replicate writes to remote region
        if event.event_type == "Write" and self.replica is not None:
            self._replication_count += 1
            # Event targets replica in another partition — will be routed
            # through the outbox by the coordinator. The latency is handled
            # by the PartitionLink (or the user can add explicit delay here).
            yield 0.0, [Event(
                time=self.now + Duration.from_seconds(0.06),  # >= min_latency
                event_type="Replicate",
                target=self.replica,
                context={"origin": self.name},
            )]

        return [Event(time=self.now, event_type="Done", target=self.sink)]


# Build per-region components
us_sink = Sink("us-sink")
eu_sink = Sink("eu-sink")

us_db = DatabaseServer("us-db", sink=us_sink)
eu_db = DatabaseServer("eu-db", sink=eu_sink)

# Wire cross-region replication
us_db.replica = eu_db
eu_db.replica = us_db

# Traffic: 80% reads, 20% writes per region
us_read_src = Source.poisson(rate=800, target=us_db, event_type="Read")
us_write_src = Source.poisson(rate=200, target=us_db, event_type="Write")
eu_read_src = Source.poisson(rate=600, target=eu_db, event_type="Read")
eu_write_src = Source.poisson(rate=150, target=eu_db, event_type="Write")

# Instrumentation
us_tracker = LatencyTracker("us-latency")
eu_tracker = LatencyTracker("eu-latency")
us_probe, us_depth = Probe.on(us_db, "depth", interval=0.5)
eu_probe, eu_depth = Probe.on(eu_db, "depth", interval=0.5)

sim = ParallelSimulation(
    partitions=[
        SimulationPartition(
            name="us",
            entities=[us_db, us_sink, us_tracker],
            sources=[us_read_src, us_write_src],
            probes=[us_probe],
        ),
        SimulationPartition(
            name="eu",
            entities=[eu_db, eu_sink, eu_tracker],
            sources=[eu_read_src, eu_write_src],
            probes=[eu_probe],
        ),
    ],
    links=[
        *PartitionLink.bidirectional("us", "eu", min_latency=0.05),
    ],
    duration=60.0,
)

summary = sim.run()

print(f"Total events:       {summary.total_events_processed:,}")
print(f"Wall clock:         {summary.wall_clock_seconds:.1f}s")
print(f"Speedup:            {summary.speedup:.1f}x")
print(f"Windows:            {summary.total_windows}")
print(f"Cross-partition:    {summary.total_cross_partition_events:,} events")
print(f"Barrier overhead:   {summary.barrier_overhead_seconds:.2f}s")
print(f"Coordination eff:   {summary.coordination_efficiency:.0%}")
print()
print(f"US p99 latency:     {us_tracker.p99():.3f}s")
print(f"EU p99 latency:     {eu_tracker.p99():.3f}s")
print(f"US replication:     {us_db._replication_count:,} writes")
print(f"EU replication:     {eu_db._replication_count:,} writes")
```

Expected output (on free-threaded Python, 60s sim, 50ms window):

```
Total events:       ~130,000
Wall clock:         ~3.2s
Speedup:            ~1.8x
Windows:            1,200
Cross-partition:    ~21,000 events
Barrier overhead:   ~0.15s
Coordination eff:   ~95%

US p99 latency:     0.012s
EU p99 latency:     0.014s
US replication:     12,000 writes
EU replication:     9,000 writes
```

## Visual Debugger

The visual debugger does not support coordinated parallel mode in this phase.
`serve(sim)` raises `TypeError` if `sim` is a `ParallelSimulation` with links.

Future: the dashboard could show cross-partition events as dashed arrows in
the graph view, and a "coordination lane" in the timeline showing barrier
phases.

## Testing Strategy

### Unit Tests

1. **PartitionLink validation**
   - min_latency <= 0 raises ValueError
   - packet_loss outside [0, 1) raises ValueError
   - bidirectional() creates two links

2. **Coordinator validation**
   - Link referencing unknown partition raises ValueError
   - window_size > min_latency raises ValueError
   - Missing link for cross-referenced entities raises ValueError

3. **Event routing**
   - Local events go to heap
   - Cross-partition events go to outbox
   - Events to unlinked partition raise RuntimeError
   - CallbackEntity events (Event.once) are always local

4. **Min latency enforcement**
   - Event violating min_latency raises RuntimeError
   - Event exactly at min_latency passes

5. **Window size computation**
   - Auto-computed from min(link.min_latency)
   - User override respected when <= min_latency

6. **No-links fallback**
   - ParallelSimulation with links=[] behaves identically to Phase 1

### Integration Tests

1. **Two-partition replication**: US and EU with bidirectional links. Verify
   cross-partition events are delivered in correct windows with correct
   timestamps.

2. **Correctness vs sequential**: Run the same model as a single Simulation
   and as a coordinated ParallelSimulation. Verify identical event counts
   and entity states (modulo non-deterministic timing from parallel Sources).
   Use `ConstantArrivalTimeProvider` and `ConstantLatency` for determinism.

3. **Multi-hop**: Three partitions A → B → C with links. Event originating
   in A must traverse two barriers to reach C.

4. **Unidirectional links**: Events flow only one direction. Reverse
   direction raises RuntimeError.

5. **Packet loss**: With 50% packet loss on link, ~50% of cross-partition
   events are dropped (statistical test over many events).

6. **Window boundary generators**: A generator that spans a window boundary
   (starts in window N, continues in window N+1) processes correctly.

7. **Empty windows**: Windows where no cross-partition events are exchanged
   incur minimal overhead.

8. **Summary metrics**: Verify total_windows, cross_partition_events,
   barrier_overhead are correctly computed.

### Performance Tests

1. **Barrier overhead**: Measure wall-clock cost per barrier (target: < 1ms
   per barrier for 2 partitions).

2. **Scaling with cross-partition volume**: Measure throughput as
   cross-partition event ratio increases from 0% to 50%.

3. **Window size sensitivity**: Same workload with W = 0.001, 0.01, 0.1, 1.0.
   Measure speedup and barrier overhead at each.

4. **Comparison with Phase 1**: For independent partitions, coordinated mode
   with no cross-partition events should have < 5% overhead vs Phase 1.

## Implementation Plan

### Step 1: PartitionLink dataclass (small)
- `PartitionLink` with validation
- `bidirectional()` factory
- Unit tests

### Step 2: Simulation.run_window() (medium)
- Add `run_window(window_end)` to `Simulation`
- Runs the event loop up to `window_end` without marking simulation as complete
- Keeps heap and entity state for next window
- Unit tests: run_window produces partial results, can be called repeatedly

### Step 3: Event router (medium)
- `_make_event_router()` factory (replaces `_make_partition_guard()` when links present)
- Modify `_run_loop_windowed()` to filter new events after invoke()
- Route cross-partition events to outbox, local events to heap
- Runtime min_latency validation
- Unit tests for routing logic

### Step 4: WindowedCoordinator (medium)
- Windowed execution loop: execute → barrier → exchange → advance
- Outbox/inbox management
- Latency distribution application
- Packet loss application
- Window size auto-computation
- Integration into `ParallelSimulation.run()` (branch on links presence)

### Step 5: Validation updates (small)
- Relax attribute-walk validation for linked partitions
- Add link completeness check
- Add window_size validation

### Step 6: Summary extension (small)
- Add coordination metrics to `ParallelSimulationSummary`
- Track barrier overhead, cross-partition event counts, window count

### Step 7: Integration tests (medium)
- Cross-region replication scenario
- Correctness vs sequential execution
- Multi-hop, unidirectional, packet loss scenarios
- Window boundary edge cases

### Step 8: Documentation (small)
- Update CLAUDE.md with PartitionLink reference
- User guide page for coordinated parallel simulation
- API reference via mkdocstrings
