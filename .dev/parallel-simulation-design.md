# Parallel Simulation Design

## Overview

A `ParallelSimulation` mode that runs independent groups of entities ("partitions") on
separate threads for true parallel execution. Users explicitly declare which entities
form each partition. Each partition runs a full, independent `Simulation` instance
internally. No cross-partition entity references or events are permitted.

This is the medium-term approach: simple, safe, and high-ROI for workloads with
natural parallelism (independent fleet nodes, isolated regions, replicated shards).

## Motivation

Many real-world simulations contain naturally independent subsystems:

- **Fleet simulations**: 100 identical worker nodes processing independent request streams
- **Regional deployments**: US-East, EU-West, AP-South operating independently
- **Sharded databases**: Each shard processes its own partition of the keyspace
- **A/B testing**: Control and experiment groups with separate traffic
- **Parameter sweeps**: Same topology with different configurations

Today, all events flow through a single heap on a single thread. A simulation with
N independent subsystems runs N times slower than it needs to.

## API Design

### Core Types

```python
from happysimulator.parallel import ParallelSimulation, SimulationPartition

# Explicitly declare independent partitions
sim = ParallelSimulation(
    partitions=[
        SimulationPartition(
            name="region-us",
            entities=[us_server, us_queue, us_sink],
            sources=[us_traffic],
            probes=[us_depth_probe],
        ),
        SimulationPartition(
            name="region-eu",
            entities=[eu_server, eu_queue, eu_sink],
            sources=[eu_traffic],
            probes=[eu_depth_probe],
        ),
    ],
    duration=100.0,              # shared across all partitions
    # OR: end_time=Instant.from_seconds(100.0)
)

summary = sim.run()              # returns ParallelSimulationSummary
```

### SimulationPartition Dataclass

```python
@dataclass
class SimulationPartition:
    """Declaration of an independent entity group for parallel execution.

    All entities, sources, and probes within a partition execute sequentially
    on a single thread, exactly as in a normal Simulation. Partitions execute
    in parallel with each other.

    Constraints:
    - No entity in this partition may hold a reference to an entity in
      another partition (validated at init time, enforced at runtime).
    - No Source may target an entity outside this partition.
    - No shared Resource, Mutex, Network, or other synchronization
      primitive may span partitions.
    """
    name: str
    entities: list[Simulatable] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    probes: list[Source] = field(default_factory=list)
    fault_schedule: FaultSchedule | None = None
    trace_recorder: TraceRecorder | None = None

    # Optional per-partition overrides
    start_time: Instant | None = None  # defaults to ParallelSimulation's start_time
    end_time: Instant | None = None    # defaults to ParallelSimulation's end_time
```

### ParallelSimulationSummary

```python
@dataclass
class ParallelSimulationSummary:
    """Merged summary from all partitions."""

    # Aggregate metrics
    duration_s: float                      # max across partitions
    total_events_processed: int            # sum
    events_per_second: float               # sum (aggregate throughput)
    wall_clock_seconds: float              # max (slowest partition)

    # Per-partition detail
    partitions: dict[str, SimulationSummary]  # name -> individual summary
    entities: dict[str, EntitySummary]        # merged (disjoint by construction)

    # Parallel-specific metrics
    partition_wall_times: dict[str, float]    # name -> wall seconds
    speedup: float                            # sequential_total / wall_clock_seconds
    parallelism_efficiency: float             # speedup / num_partitions

    def to_dict(self) -> dict: ...
```

### Convenience: From Existing Simulation

For simulations already built with the standard API, provide a conversion path:

```python
# Build entities as usual
nodes = [Node(f"node-{i}") for i in range(100)]
sources = [Source.poisson(rate=10, target=n) for n in nodes]

# Convert to parallel — user declares the grouping
sim = ParallelSimulation.from_groups(
    groups={f"node-{i}": ([nodes[i]], [sources[i]]) for i in range(100)},
    duration=100.0,
)
```

### Schedule Events

```python
# Schedule into a specific partition
sim.schedule(events, partition="region-us")

# Schedule before run (events must target entities within the named partition)
sim.schedule_all({
    "region-us": [bootstrap_event_us],
    "region-eu": [bootstrap_event_eu],
})
```

### Control Surface

Each partition has its own `SimulationControl`. The `ParallelSimulation` exposes
an aggregate control that delegates:

```python
# Pause all partitions
sim.control.pause()

# Pause a specific partition
sim.partitions["region-us"].control.pause()

# Breakpoints are per-partition
sim.partitions["region-us"].control.add_breakpoint(
    TimeBreakpoint(time=Instant.from_seconds(50.0))
)
```

Note: Interactive control (step, pause, breakpoints) in parallel mode is inherently
more complex. For the medium-term, keep it simple:
- `pause()` sends a pause signal to all partition threads
- `step()` steps one event in *one* partition (round-robin or earliest-next-event)
- Breakpoints are per-partition only

## Internal Architecture

### Composition Over New Machinery

The key insight is that each partition IS a `Simulation`. We reuse the entire existing
engine — heap, clock, entity dispatch, generator/continuation machinery — unchanged.
`ParallelSimulation` is a thin coordinator over N `Simulation` instances.

```
ParallelSimulation
├── _validate_partitions()       # init-time cross-reference check
├── _build_simulations()         # create N Simulation instances
├── run()                        # submit to thread pool, collect results
│   ├── ThreadPoolExecutor(max_workers=N)
│   │   ├── partition_1.run()    # standard Simulation.run()
│   │   ├── partition_2.run()    # standard Simulation.run()
│   │   └── partition_N.run()    # standard Simulation.run()
│   └── _merge_summaries()       # aggregate results
└── partitions: dict[str, Simulation]
```

### Threading Model

```python
import concurrent.futures

class ParallelSimulation:
    def run(self) -> ParallelSimulationSummary:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._simulations)
        ) as executor:
            futures = {
                executor.submit(sim.run): name
                for name, sim in self._simulations.items()
            }

            results: dict[str, SimulationSummary] = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                results[name] = future.result()  # propagates exceptions

        return self._merge_summaries(results)
```

**Why threads, not processes:**
- Entities can't be pickled (generators, closures, lambda callbacks)
- Thread creation is lightweight
- Python 3.13+ free-threaded mode (PEP 703) removes the GIL, enabling true parallelism
- With the GIL present, threads still provide correct isolation (just no speedup)
- Users who need GIL-free execution opt into free-threaded Python at the interpreter level

**GIL detection and user guidance:**

```python
import sys

def run(self) -> ParallelSimulationSummary:
    if len(self._simulations) > 1:
        # Detect GIL status (Python 3.13+)
        gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
        if gil_enabled:
            logger.warning(
                "ParallelSimulation: GIL is enabled. Partitions will run "
                "concurrently but not in parallel. For true parallelism, "
                "use free-threaded Python: python3.13t"
            )
    ...
```

### Context Variable Safety

`SimFuture` uses `contextvars.ContextVar` for the active heap/clock references.
`concurrent.futures.ThreadPoolExecutor` copies context to worker threads by default
(Python 3.12+). Each partition's `Simulation.run()` calls `_set_active_context()`
which sets the contextvar for that thread. This is safe because:

1. Each thread sets its own contextvar values
2. `contextvars` are per-thread (or per-context in asyncio)
3. A `SimFuture.resolve()` in partition A pushes to partition A's heap only

**One concern**: The `_active_code_debugger` in `event.py` is a module-level global,
not a contextvar. This would need to become a contextvar too, or be disabled in
parallel mode.

### Global Event Counter

`event.py` uses `_global_event_counter = count()` (module-level `itertools.count`).
Under CPython's GIL, `count.__next__()` is atomic. Under free-threaded Python, it is
NOT guaranteed atomic.

Options:
1. **Per-partition counters**: Each Simulation gets its own counter. Sort order only
   matters within a partition, so this is sufficient.
2. **Atomic counter**: Use `threading.Lock` around `count.__next__()`. Simple but adds
   contention on a hot path.
3. **Leave it**: Under GIL, it's safe. Under free-threaded, minor non-determinism in
   tie-breaking order — acceptable since partitions are independent.

Recommendation: Option 1 (per-partition counters) for correctness. Requires making
`_global_event_counter` an instance variable on `EventHeap` or `Simulation`, and
passing it into `Event.__init__()` or using a contextvar.

## Validation

### Init-Time Validation

The most critical safety check. Cross-partition references would cause silent
correctness bugs (entity reads stale clock, heap races, etc.).

```python
def _validate_partitions(self) -> None:
    """Verify no entity appears in multiple partitions and no cross-references exist."""

    # 1. Build partition membership: entity -> partition_name
    membership: dict[int, str] = {}  # id(entity) -> partition_name
    all_entity_ids: dict[int, str] = {}  # id(entity) -> entity.name

    for part in self._partitions:
        for entity in part.entities:
            eid = id(entity)
            if eid in membership:
                raise ValueError(
                    f"Entity '{entity.name}' appears in both partition "
                    f"'{membership[eid]}' and '{part.name}'"
                )
            membership[eid] = part.name
            all_entity_ids[eid] = getattr(entity, 'name', repr(entity))

        # Sources must target entities in this partition
        for source in part.sources:
            target = getattr(source, '_target', None) or \
                     getattr(source, 'target', None)
            if target is not None:
                tid = id(target)
                if tid in membership and membership[tid] != part.name:
                    raise ValueError(
                        f"Source '{source.name}' in partition '{part.name}' "
                        f"targets entity '{all_entity_ids[tid]}' in partition "
                        f"'{membership[tid]}'"
                    )

    # 2. Walk entity attributes for cross-references
    for part in self._partitions:
        for entity in part.entities:
            self._check_entity_references(entity, part.name, membership, all_entity_ids)

def _check_entity_references(
    self, entity, partition_name, membership, all_entity_ids,
    _visited=None, _depth=0, _max_depth=3,
) -> None:
    """Walk entity attributes looking for references to entities in other partitions."""
    if _depth >= _max_depth:
        return
    if _visited is None:
        _visited = set()
    eid = id(entity)
    if eid in _visited:
        return
    _visited.add(eid)

    for attr_name in vars(entity):
        value = getattr(entity, attr_name, None)
        if isinstance(value, Entity):
            vid = id(value)
            if vid in membership and membership[vid] != partition_name:
                raise ValueError(
                    f"Entity '{entity.name}' in partition '{partition_name}' "
                    f"references entity '{all_entity_ids[vid]}' in partition "
                    f"'{membership[vid]}' via attribute '{attr_name}'"
                )
            # Recurse into sub-entities (e.g., QueuedResource internals)
            if vid in membership and membership[vid] == partition_name:
                self._check_entity_references(
                    value, partition_name, membership, all_entity_ids,
                    _visited, _depth + 1, _max_depth,
                )

        # Also check lists/tuples of entities (common pattern: self.replicas = [...])
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Entity):
                    iid = id(item)
                    if iid in membership and membership[iid] != partition_name:
                        raise ValueError(
                            f"Entity '{entity.name}' in partition '{partition_name}' "
                            f"references entity '{all_entity_ids[iid]}' in partition "
                            f"'{membership[iid]}' via attribute '{attr_name}'"
                        )
```

### Runtime Validation (Safety Net)

Even with init-time checks, dynamic event creation could produce cross-partition
events (e.g., a callback that captures an entity from another partition). Add a
runtime guard in the event dispatch path.

This requires each partition's entities to be registered in a set for O(1) lookup:

```python
class _PartitionAwareSimulation(Simulation):
    """Simulation subclass that rejects events targeting outside entities."""

    def __init__(self, partition_name: str, entity_ids: frozenset[int], **kwargs):
        super().__init__(**kwargs)
        self._partition_name = partition_name
        self._entity_ids = entity_ids

    def _run_loop(self) -> SimulationSummary:
        # Override to add target validation before invoke
        while self._event_heap.has_events() and self._end_time >= self._current_time:
            # ... (same as Simulation._run_loop) ...

            event = self._event_heap.pop()

            # PARTITION GUARD
            target_id = id(event.target)
            if target_id not in self._entity_ids:
                raise RuntimeError(
                    f"Partition '{self._partition_name}': event '{event.event_type}' "
                    f"targets entity '{getattr(event.target, 'name', '?')}' which is "
                    f"not in this partition. Cross-partition events are not supported."
                )

            # ... rest of loop ...
```

**Performance note**: `id()` lookup in a `frozenset` is O(1). The overhead per event
is negligible compared to `handle_event()` dispatch.

Alternatively (to avoid overriding `_run_loop`), this could be implemented as a
control hook (`sim.control.on_event(...)`) but that forces lazy creation of the
control surface, adding overhead. A cleaner approach: add an optional
`_event_validator` callback to the base `Simulation` class.

## Time Model

Each partition has its own `Clock` instance, advancing independently. There is no
global time synchronization because partitions are truly independent.

```
Partition A:  t=0 ──> t=3.2 ──> t=7.1 ──> t=100.0  (done)
Partition B:  t=0 ──> t=1.5 ──> t=4.8 ──> t=99.3   (heap empty, done)
Partition C:  t=0 ──> t=2.0 ──> t=5.5 ──> t=100.0  (done)
              ↑                                       ↑
          all start at                          all bounded by
          start_time                            end_time
```

Each partition's simulation terminates independently based on its own heap and the
shared `end_time`.

## Results Merging

```python
def _merge_summaries(
    self, results: dict[str, SimulationSummary]
) -> ParallelSimulationSummary:
    wall_times = {
        name: summary.wall_clock_seconds
        for name, summary in results.items()
    }
    total_wall = max(wall_times.values())
    sequential_estimate = sum(wall_times.values())

    # Merge entity summaries (disjoint sets)
    merged_entities: dict[str, EntitySummary] = {}
    for summary in results.values():
        merged_entities.update(summary.entities)

    return ParallelSimulationSummary(
        duration_s=max(s.duration_s for s in results.values()),
        total_events_processed=sum(s.total_events_processed for s in results.values()),
        events_per_second=sum(s.events_per_second for s in results.values()),
        wall_clock_seconds=total_wall,
        partitions=results,
        entities=merged_entities,
        partition_wall_times=wall_times,
        speedup=sequential_estimate / total_wall if total_wall > 0 else 1.0,
        parallelism_efficiency=(
            (sequential_estimate / total_wall) / len(results)
            if total_wall > 0 else 1.0
        ),
    )
```

## Instrumentation & Probes

Probes and data collectors are per-partition and work unchanged:

```python
# Before creating partitions:
us_probe, us_data = Probe.on(us_server, "depth", interval=0.1)
eu_probe, eu_data = Probe.on(eu_server, "depth", interval=0.1)

sim = ParallelSimulation(
    partitions=[
        SimulationPartition("us", entities=[us_server, ...], probes=[us_probe]),
        SimulationPartition("eu", entities=[eu_server, ...], probes=[eu_probe]),
    ],
    duration=100.0,
)
sim.run()

# Access data per-partition as usual
print(us_data.mean())   # works — data object was populated by us_probe's partition
print(eu_data.p99())    # works — independent data object
```

`Data` objects are written by the probe thread and read by the user after `run()`
completes. No cross-thread access during execution.

## Visual Debugger

For the medium-term, the visual debugger does **not** support parallel mode.
`serve(sim)` raises `TypeError` if `sim` is a `ParallelSimulation`.

Future: The dashboard could show partitions as separate lanes/tabs, each with its
own event log, graph view, and chart dashboard.

## Constraints & Limitations

| Constraint | Reason | Future Relaxation |
|-----------|--------|-------------------|
| No shared entities across partitions | Clock, heap, generator safety | Conservative PDES with inter-partition messaging |
| No cross-partition SimFuture | resolve() pushes to partition-local heap | Future bridge mechanism |
| No shared Resource/Mutex/Network | Non-atomic state, waiter queues | Synchronized mailbox pattern |
| No visual debugger in parallel mode | Bridge assumes single simulation | Multi-partition dashboard |
| True parallelism requires free-threaded Python | GIL prevents thread-level parallelism | Python ecosystem moving toward GIL-free |
| No interactive step-through across partitions | Time is per-partition | Global virtual time coordinator |

## Future Extensions

### Phase 2: Cross-Partition Event Bridge

For workloads where partitions occasionally communicate (e.g., independent nodes
that share a load balancer), add a `PartitionBridge`:

```python
bridge = PartitionBridge(
    name="cross-region",
    latency=ConstantLatency(0.05),  # minimum 50ms between regions
)

sim = ParallelSimulation(
    partitions=[
        SimulationPartition("us", entities=[us_lb, us_server, ...], ...),
        SimulationPartition("eu", entities=[eu_lb, eu_server, ...], ...),
    ],
    bridges=[
        bridge.connect("us", "eu"),  # bidirectional, 50ms minimum latency
    ],
    duration=100.0,
)
```

The bridge provides conservative PDES synchronization:
- Each partition reports its LBTS (Lower Bound on Time Stamp)
- Cross-partition events are buffered until the destination's time catches up
- The minimum bridge latency provides lookahead for the conservative algorithm
- Partitions never need to rollback (conservative guarantee)

### Phase 3: Auto-Partitioning

Analyze entity wiring at init time to automatically discover independent subgraphs:

```python
sim = Simulation(entities=[...], sources=[...])
sim.run(parallel=True)  # auto-detect partitions from entity reference graph
```

This requires building a dependency graph from entity attributes and finding
weakly connected components. Entities connected through shared Resources, Networks,
or SimFuture patterns would be placed in the same partition.

## Example: Independent Fleet Nodes

The motivating use case — 100 identical servers processing independent traffic:

```python
from happysimulator import (
    Source, Instant, QueuedResource, FIFOQueue, Sink, Event,
)
from happysimulator.parallel import ParallelSimulation, SimulationPartition
from happysimulator.instrumentation import Probe, LatencyTracker

# Build 100 independent server pipelines
partitions = []
trackers = []

for i in range(100):
    server = MyServer(f"server-{i}")
    sink = Sink(f"sink-{i}")
    tracker = LatencyTracker(f"latency-{i}")
    source = Source.poisson(rate=50, target=server, event_type="Request")
    probe, data = Probe.on(server, "depth", interval=0.5)

    trackers.append(tracker)
    partitions.append(SimulationPartition(
        name=f"node-{i}",
        entities=[server, sink, tracker],
        sources=[source],
        probes=[probe],
    ))

sim = ParallelSimulation(partitions=partitions, duration=60.0)
summary = sim.run()

print(f"Total events: {summary.total_events_processed:,}")
print(f"Wall clock:   {summary.wall_clock_seconds:.1f}s")
print(f"Speedup:      {summary.speedup:.1f}x over sequential")
print(f"Efficiency:   {summary.parallelism_efficiency:.0%}")

# Per-node analysis still works
for t in trackers[:5]:
    print(f"  {t.name}: p99={t.p99():.3f}s, mean={t.mean_latency():.3f}s")
```

Expected output (on 8-core free-threaded Python):
```
Total events: 3,000,000
Wall clock:   8.2s
Speedup:      7.4x over sequential
Efficiency:   92%
```

## Testing Strategy

### Unit Tests

1. **Validation tests**: Cross-partition references detected at init
   - Entity in two partitions -> ValueError
   - Source targeting wrong partition -> ValueError
   - Entity attribute referencing other partition -> ValueError
   - List attribute containing other partition's entity -> ValueError

2. **Single-partition equivalence**: `ParallelSimulation` with 1 partition produces
   identical results to `Simulation` with the same entities

3. **Determinism**: Same seed + same partitions = same results (per-partition)

4. **Summary merging**: Verify aggregate metrics are correctly computed

### Integration Tests

1. **N independent M/M/1 queues**: Run N partitions, verify each produces correct
   latency statistics matching analytical M/M/1 results

2. **Unbalanced partitions**: One partition with heavy load, others light — verify
   wall clock is bounded by the slowest partition

3. **GIL detection warning**: Verify warning is emitted when GIL is enabled

4. **Runtime guard**: Event targeting wrong partition raises RuntimeError

### Performance Tests

1. **Scaling benchmark**: 1, 2, 4, 8, 16 partitions of identical work — measure
   speedup curve

2. **Overhead benchmark**: Single partition via `ParallelSimulation` vs `Simulation`
   — measure coordinator overhead (should be < 1%)

## Implementation Plan

### Step 1: Core Types (small)
- `SimulationPartition` dataclass
- `ParallelSimulationSummary` dataclass
- `ParallelSimulation` class skeleton

### Step 2: Validation (medium)
- `_validate_partitions()` init-time checker
- Cross-reference attribute walker
- Unit tests for all validation paths

### Step 3: Execution (medium)
- `_build_simulations()` — create N `Simulation` instances from partitions
- `run()` — ThreadPoolExecutor dispatch and result collection
- `_merge_summaries()` — aggregate results
- GIL detection and warning

### Step 4: Runtime Safety (small)
- Runtime target validation (event targets must be in-partition)
- Either via `_PartitionAwareSimulation` subclass or validator callback

### Step 5: Per-Partition Event Counter (small)
- Make `_global_event_counter` partition-local for free-threaded safety
- Option: contextvar-based counter, or pass counter through EventHeap

### Step 6: Schedule API (small)
- `schedule(events, partition=name)` method
- `schedule_all(dict)` convenience method

### Step 7: Tests (medium)
- Full unit test suite for validation
- Integration tests with M/M/1 queues
- Performance benchmarks

### Step 8: Documentation (small)
- Add to CLAUDE.md quick reference
- User guide page in docs/guides/
- API reference via mkdocstrings

### Step 9: Code Debugger Fix (small)
- Convert `_active_code_debugger` from module global to contextvar
- Or disable in parallel mode
