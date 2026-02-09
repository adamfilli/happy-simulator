# Happy Simulator: DES Improvement Recommendations

## Context

Happy-simulator is a well-engineered discrete-event simulation library with a clean core (pop-invoke-push loop, Entity actors, generator-based processes, nanosecond time). The user wants to simulate software systems at all scales: individual components and OS-level models through to Kafka, consensus protocols, and large distributed architectures.

After thorough codebase review, I've identified **8 high-impact improvements** organized into three tiers. Each addresses a real gap that limits the kinds of systems you can naturally model today.

---

## Tier 1: Core Engine (benefits every simulation)

### 1. Event Cancellation

**Gap:** Once an event is scheduled, it can never be removed from the heap. Every timeout, retry timer, lease renewal, election timer, and heartbeat in real systems is *cancellable* — you set a timer, and if the expected thing happens first, you cancel it. Today entities must check stale preconditions inside `handle_event()`, which wastes heap space and muddies intent.

**Impact:** Touches nearly every distributed system model. Without this, modeling Raft election timeouts, TCP retransmit timers, Kafka session timeouts, or even simple HTTP request timeouts requires workaround patterns.

**API sketch:**
```python
# Schedule a cancellable event
timeout_event = Event(time=now + 5.0, event_type="ElectionTimeout", target=self)
sim.schedule(timeout_event)

# Cancel it when we hear from the leader
timeout_event.cancel()  # Marks as cancelled; heap skips it on pop

# In the simulation loop: cancelled events are silently discarded on pop
# Optional: heap.cancel(event) for O(1) lookup via event._id
```

**Implementation:** Add `_cancelled: bool` field to `Event`, add `cancel()` method. In `Simulation.run()`, after `pop()`, check `if event._cancelled: continue`. This is the "lazy deletion" pattern — simple, no heap restructuring needed. The `EventHeap` can optionally track cancelled count for diagnostics.

**Effort:** Small. ~20 lines in event.py + simulation.py.

---

### 2. SimFuture — Yield on Events, Not Just Delays

**Gap:** This is the single biggest ergonomic limitation. Generators can only `yield float` (a delay). They *cannot* yield "wait until this response arrives" or "wait until this resource is available." This forces every request-response interaction into callback chains or state machines, which is unnatural for modeling protocols.

Compare the current way to model a request-response:

```python
# TODAY: Callback-based, fragmented across methods
class Client(Entity):
    def __init__(self):
        self._pending = {}  # Manual state tracking

    def handle_event(self, event):
        if event.event_type == "SendRequest":
            req_id = uuid4()
            self._pending[req_id] = event.context
            return [Event(time=self.now, event_type="Request", target=self.server,
                         context={"reply_to": self, "req_id": req_id})]

        elif event.event_type == "Response":
            original = self._pending.pop(event.context["req_id"])
            # ... finally process response
```

vs. what becomes possible with SimFuture:

```python
# WITH SIMFUTURE: Sequential, natural
class Client(Entity):
    def handle_event(self, event):
        response = yield SimFuture()  # Pause until resolved
        # response is the value passed to future.resolve(value)
        latency = self.now - event.time
```

**How it works:** A `SimFuture` is an object that a generator can yield. When yielded, the ProcessContinuation *parks* instead of scheduling a time-based resume. When some other entity calls `future.resolve(value)`, the continuation is re-scheduled at the current simulation time, and the resolved value is sent into the generator via `generator.send(value)`.

**What this enables:**
- Natural request-response modeling (RPC, HTTP, gRPC)
- `yield resource.acquire()` — wait for shared resource availability
- `yield lock.acquire()` — wait for mutex
- `yield any_of(timeout_future, response_future)` — first-to-complete races (critical for timeout patterns)
- `yield all_of(ack1, ack2, ack3)` — quorum waits (critical for consensus)
- `yield channel.receive()` — message passing between processes

**Implementation notes:**
- `SimFuture` holds a reference to the `ProcessContinuation` that yielded it
- `future.resolve(value)` schedules the continuation at current time, passing value via `gen.send(value)`
- `future.fail(exception)` schedules the continuation with `gen.throw(exception)`
- `any_of(*futures)` / `all_of(*futures)` are composite futures
- This is exactly how SimPy's `Environment.event()` works, adapted to happy-simulator's Entity model
- ProcessContinuation's `invoke()` already calls `next(gen)` — needs to change to `gen.send(resolved_value)` when a future is involved

**Effort:** Medium. Core is ~100 lines (SimFuture class + ProcessContinuation changes). The combinators (`any_of`, `all_of`) add another ~50. But this is foundational — almost every Tier 2 suggestion becomes dramatically simpler with this in place.

---

### 3. Shared Resources (CPU, Memory, Disk, Bandwidth)

**Gap:** No way to model shared, contended resources. QueuedResource handles per-entity concurrency (one server with N slots), but there's no way to say "these 5 servers share a 10 Gbps network link" or "these processes share 8 CPU cores." In real systems, contention for shared resources (CPU, memory bandwidth, disk I/O, network bandwidth) drives most performance behavior.

**API sketch:**
```python
resource = Resource("cpu_cores", capacity=8)

class Worker(Entity):
    def handle_event(self, event):
        # Acquire 2 CPU cores (blocks if unavailable)
        grant = yield resource.acquire(amount=2)
        yield 0.1  # Do work
        grant.release()  # Return the cores

# With SimFuture, acquire() returns a SimFuture that resolves
# when capacity is available. Without SimFuture, it uses the
# callback/queue pattern like QueuedResource.
```

**Why this matters:**
- OS simulation: processes competing for CPU time slices, memory pages, disk bandwidth
- Distributed systems: network bandwidth shared across connections, connection pools, thread pools
- Kafka: broker disk I/O shared across partitions, network shared across consumers

**Builds on:** SimFuture (for the `yield resource.acquire()` pattern). Can also work without SimFuture using the QueueDriver callback pattern, but much less ergonomic.

**Effort:** Medium. Core Resource class ~80 lines. Pairs naturally with existing QueuedResource as a higher-level composition.

---

## Tier 2: Distributed Systems Primitives

### 4. Per-Node Clocks and Clock Models

**Gap:** All entities share one `Clock`. In real distributed systems, every node has its own clock with drift and skew. This matters enormously for:
- Leader election (Raft heartbeat timeouts depend on local clocks)
- Lease expiry (node A's lease expires at a different real time than node B thinks)
- Cache TTLs (different nodes expire cached entries at different times)
- Distributed tracing (clock skew corrupts trace ordering)

**API sketch:**
```python
from happysimulator import NodeClock, ClockModel

# Fixed skew
node_a_clock = NodeClock(skew=Duration.from_millis(50))  # 50ms ahead

# Drifting clock (1ms drift per simulated second)
node_b_clock = NodeClock(model=LinearDrift(rate_ppm=1000))

# NTP-corrected (periodic corrections with jitter)
node_c_clock = NodeClock(model=NTPCorrected(correction_interval=30.0, jitter_ms=5.0))

# Entities use their node clock
class RaftNode(Entity):
    def __init__(self, name, clock_model):
        super().__init__(name)
        self._node_clock = clock_model

    @property
    def local_now(self) -> Instant:
        """What this node thinks the time is."""
        return self._node_clock.read(self.now)  # Transform true time → local time
```

**Key insight:** The simulation's `Clock` remains the source of truth (events are ordered by true time). `NodeClock` is a *view* that transforms true time into what a node perceives. This keeps the core engine unchanged while enabling clock-sensitive protocols.

**Effort:** Small-medium. NodeClock is a thin wrapper (~40 lines). Clock models (linear drift, NTP) are ~30 lines each.

---

### 5. Network Topology and Partitions

**Gap:** `NetworkLink` models individual point-to-point links, but there's no graph abstraction. You can't say "partition nodes {A, B} from {C, D, E}" — you'd have to manually find and modify every link. For simulating distributed systems, network partitions are the most important failure mode.

**API sketch:**
```python
from happysimulator import NetworkTopology, Partition

# Build topology
topo = NetworkTopology()
topo.add_node("broker-1")
topo.add_node("broker-2")
topo.add_node("broker-3")
topo.connect("broker-1", "broker-2", profile=datacenter_network())
topo.connect("broker-2", "broker-3", profile=datacenter_network())
topo.connect("broker-1", "broker-3", profile=datacenter_network())

# Inject partition
partition = topo.partition(["broker-1"], ["broker-2", "broker-3"])
# All links between the two groups silently drop packets

# Heal partition
partition.heal()

# Asymmetric partition (A can send to B, but not receive)
topo.partition(["broker-1"], ["broker-2"], asymmetric=True)

# Entities route through topology
class Broker(Entity):
    def send(self, dest_name: str, message: Event):
        link = self.topology.link_to(dest_name)
        return [Event(time=self.now, target=link, ...)]
```

**Effort:** Medium. Topology is a graph wrapper (~100 lines) over existing NetworkLink. Partition is a set operation that toggles links.

---

### 6. Fault Injection Framework

**Gap:** No unified way to inject failures. Individual components have ad-hoc failure modes (NetworkLink has loss probability, CircuitBreaker has failure thresholds), but there's no way to say "at t=30s, crash node X and restart it at t=45s" or "inject 5% packet loss between regions A and B from t=20 to t=40."

**API sketch:**
```python
from happysimulator import FaultSchedule, faults

schedule = FaultSchedule()

# Node crash/restart
schedule.add(faults.CrashNode("broker-1", at=30.0, restart_at=45.0))

# Network degradation
schedule.add(faults.InjectLatency("broker-1", "broker-2",
    extra_ms=500, start=20.0, end=40.0))

# Disk slowdown
schedule.add(faults.SlowDisk("broker-1", iops_factor=0.1, start=25.0, end=35.0))

# Random failures (Jepsen-style)
schedule.add(faults.RandomPartition(
    nodes=["b1", "b2", "b3", "b4", "b5"],
    mtbf=60.0,  # Mean time between failures
    mttr=10.0,  # Mean time to repair
))

sim = Simulation(sources=[...], entities=[...], fault_schedule=schedule)
```

**Implementation:** FaultSchedule generates `Event.once()` events at the specified times that modify entity/link state. Node crash = remove from topology + discard queued events. Restart = re-add + trigger recovery logic.

**Effort:** Medium-large. The framework is ~100 lines, but each fault type needs integration with the components it affects (NetworkLink, Node, Resources).

---

### 7. Logical Clocks

**Gap:** No Lamport timestamps, vector clocks, or hybrid logical clocks (HLC). These are fundamental to distributed systems — they establish causal ordering without relying on synchronized physical clocks. Every consensus protocol, distributed database, and event-sourcing system uses some form of logical time.

**API sketch:**
```python
from happysimulator import LamportClock, VectorClock, HybridLogicalClock

# Lamport: simple monotonic counter
clock = LamportClock()
clock.tick()           # Local event
clock.send()           # Returns timestamp to include in message
clock.receive(ts)      # Updates clock to max(local, received) + 1

# Vector clock: per-node counters
vc = VectorClock(node_id="broker-1", nodes=["broker-1", "broker-2", "broker-3"])
vc.tick()
vc.send()              # Returns full vector
vc.receive(remote_vc)  # Element-wise max + increment local
vc.happened_before(other_vc) -> bool  # Causal ordering

# HLC: physical + logical (best of both worlds)
hlc = HybridLogicalClock(node_id="broker-1", physical_clock=node_clock)
hlc.now()              # Returns HLCTimestamp(physical, logical, node_id)
hlc.send()
hlc.receive(remote_hlc)
```

**Why these matter:**
- Lamport: ordering events in distributed logs (Kafka offset ordering across partitions)
- Vector clocks: conflict detection in replicated datastores (Dynamo, Riak)
- HLC: CockroachDB, Spanner-style transaction ordering

**Effort:** Small. Pure algorithms, no simulation integration needed. ~50 lines each. They're used *by* entities, not by the engine.

---

## Tier 3: Quality of Life

### 8. Causal Event Graph

**Gap:** Events have UUIDs and trace spans, but there's no first-class "event A caused event B" relationship. When debugging a complex simulation (why did this Raft node start an election? what request triggered this queue overflow?), you need to trace causality chains. Currently this requires reading trace spans manually.

**API sketch:**
```python
# Automatic: track parent-child relationships
new_event = Event(
    time=self.now,
    event_type="Response",
    target=client,
    context={**event.context},  # Inherits parent's trace context
)
# The context already carries "id" — we'd add "parent_id" automatically
# when an event is created inside handle_event()

# Query after simulation
graph = sim.causal_graph()
chain = graph.ancestors(event_id)     # What caused this?
descendants = graph.descendants(event_id)  # What did this cause?
graph.critical_path()                  # Longest causal chain
```

**Implementation:** During `event.invoke()`, set a thread-local "current event ID". When new Events are created, auto-populate `context["parent_id"]`. Post-simulation, build the graph from context.

**Effort:** Small. ~30 lines for tracking, ~50 for the graph query API.

---

## Implementation Order

I'd recommend this sequence, where each phase unlocks the next:

| Phase | Item | Unlocks |
|-------|------|---------|
| 1 | Event Cancellation ✅ | Clean timeout/timer patterns everywhere |
| 2 | SimFuture ✅ | Natural request-response, resource acquire, lock acquire, quorum waits |
| 3 | Logical Clocks | Pure algorithms, no engine changes needed |
| 4 | Per-Node Clocks ✅ | Clock-sensitive protocol modeling |
| 5 | Shared Resources ✅ | CPU/memory/disk contention (uses SimFuture) |
| 6 | Network Topology | Graph abstraction over existing NetworkLink |
| 7 | Fault Injection | Node crash, partition, degradation (uses Topology) |
| 8 | Causal Event Graph | Debugging and analysis |

Phases 1-2 are foundational and transform what's expressible. Phases 3-4 are independent and can be done in any order. Phases 5-7 build on earlier phases. Phase 8 is independent and useful at any time.

---

## What's Already Strong (no changes needed)

- Core pop-invoke-push loop is clean and correct
- Nanosecond time precision prevents float non-determinism
- Generator-based process model is elegant (just needs SimFuture extension)
- Zero-cost simulation control (lazy creation + guarded checks)
- Component library coverage (queues, rate limiters, network, caching, sketching)
- Instrumentation (Data, Probe, LatencyTracker, BucketedData)
- Load generation ergonomics (Source factories, profiles)
