# CLAUDE.md

Discrete-event simulation library for Python 3.13+.

## Quick Reference

| Aspect | Summary |
|--------|---------|
| **Core Loop** | `EventHeap` pop → `Entity.handle_event()` → schedule returned `Event`s |
| **Key Invariant** | Events always have a `target` (Entity); use `Event.once()` for function-based dispatch |
| **Time** | Use `Instant.from_seconds(n)`, not raw floats |
| **Generators** | Yield delays (float seconds) or `SimFuture`; return events on completion |
| **Load Gen** | `Source.poisson(rate=10, target=server)` for quick setup |
| **Resources** | `Resource("cpu", capacity=8)` + `yield resource.acquire(2)` |
| **Network** | `Network` topology + `partition()`/`.heal()` for failures |
| **Clocks** | `NodeClock(FixedSkew(...))` for skew/drift; `LamportClock`, `VectorClock`, `HybridLogicalClock` for causal ordering |
| **Control** | `sim.control.pause()` / `.step()` / `.add_breakpoint()` |
| **Testing** | Use `Source.constant()` or `ConstantArrivalTimeProvider` for deterministic timing |

## Development Commands

```bash
pytest -q                                    # all tests (~2755, ~68s)
pytest tests/integration/test_queue.py -q    # single file
python examples/m_m_1_queue.py               # run example
```

## Reading Order

1. `happysimulator/__init__.py` → public API surface
2. `core/instant.py` → `core/event.py` → `core/entity.py` → `core/simulation.py` → `core/sim_future.py`
3. `core/control/control.py` → `core/control/breakpoints.py`
4. `load/source.py` → `components/queue.py` → `components/queued_resource.py`
5. `examples/m_m_1_queue.py` + `tests/integration/`
6. `.dev/COMPONENTLIB.md` for design philosophy

---

## Core Abstractions

### Instant & Duration

```python
t = Instant.from_seconds(1.5)       # Instant.Epoch = t0, Instant.Infinity = never
t2 = t + Duration.from_seconds(0.5) # arithmetic returns Duration
```

### Event

Every Event must have a `target` Entity. Use `Event.once()` for function-based dispatch.

```python
Event(time=Instant.from_seconds(1.0), event_type="Request", target=server, context={...})
Event.once(time=Instant.from_seconds(1.0), event_type="Ping", fn=lambda e: print("pong"))
```

### Entity

Stateful actors. `handle_event()` returns `None | Event | list[Event] | Generator`.

```python
class Server(Entity):
    def handle_event(self, event) -> Generator[float, None, list[Event]]:
        yield 0.1                    # pause 100ms
        yield 0.01, [Event(...)]     # pause AND schedule side-effect NOW
        response = yield future      # park until future.resolve(value)
        return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

**Yield forms**: `yield delay` | `yield delay, event(s)` | `yield future`

### SimFuture

Pause generators until external conditions are met.

```python
# Request-response
future = SimFuture()
yield 0.0, [Event(time=self.now, event_type="Req", target=server, context={"reply": future})]
response = yield future  # parks until server calls future.resolve(value)

# Timeout race
idx, value = yield any_of(response_future, timeout_future)  # (index, value) of first

# Quorum wait
results = yield all_of(f1, f2, f3)  # [value1, value2, value3]
```

- Each SimFuture can only be yielded by one generator
- Pre-resolved futures resume immediately

---

## Load Generation

```python
Source.constant(rate=10, target=server, event_type="Request")          # deterministic
Source.poisson(rate=10, target=server, event_type="Request")           # stochastic
Source.with_profile(profile=MyProfile(), target=server, poisson=True)  # custom
# All accept stop_after (float seconds or Instant)
```

Full constructor for advanced cases (custom `EventProvider`):
```python
Source(name="Traffic", event_provider=my_provider, arrival_time_provider=ConstantArrivalTimeProvider(...))
```

## Components

### QueuedResource

Combines queue + processing. Override `handle_queued_event()` and optionally `has_capacity()`.

```python
class MyServer(QueuedResource):
    def __init__(self, name, downstream, concurrency=1):
        super().__init__(name, policy=FIFOQueue())
        self.downstream, self.concurrency, self._in_flight = downstream, concurrency, 0

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event):
        self._in_flight += 1
        try:
            yield 0.1
        finally:
            self._in_flight -= 1
        return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

### Sink & Counter

```python
sink = Sink()                    # tracks latency from context['created_at']
sink.events_received             # count
sink.latency_stats()             # {count, avg, min, max, p50, p99}

counter = Counter()              # counts by event type
counter.total / counter.by_type
```

### Resource (Contended Capacity)

```python
cpu = Resource("cpu_cores", capacity=8)  # must register with Simulation(entities=[cpu, ...])

# In entity:
grant = yield cpu.acquire(amount=2)  # blocks via SimFuture if unavailable
yield 0.1
grant.release()                       # idempotent, wakes waiters (FIFO)

grant = cpu.try_acquire(amount=2)     # non-blocking: Grant | None
cpu.available / cpu.utilization / cpu.waiters / cpu.stats
```

### Inductor (Burst Suppression)

EWMA-based smoothing with **no throughput cap** — resists rate *changes*, not absolute rate.

```python
inductor = Inductor(name="Smoother", downstream=server, time_constant=1.0, queue_capacity=10000)
inductor.stats / inductor.estimated_rate / inductor.queue_depth
```

### Industrial Components

Package: `happysimulator.components.industrial`

```python
from happysimulator.components.industrial import (
    BalkingQueue, RenegingQueuedResource,
    ConveyorBelt, InspectionStation, BatchProcessor,
    ShiftSchedule, ShiftedServer, Shift,
    BreakdownScheduler, InventoryBuffer, AppointmentScheduler,
    ConditionalRouter, PerishableInventory, PooledCycleResource,
    GateController, SplitMerge, PreemptibleResource, PreemptibleGrant,
)
```

| Component | Base | Description |
|-----------|------|-------------|
| `BalkingQueue` | `QueuePolicy` | Wraps any queue policy; rejects arrivals when depth >= threshold. `BalkingQueue(inner, balk_threshold=5, balk_probability=1.0)` |
| `RenegingQueuedResource` | `QueuedResource` | Abstract; checks `(now - created_at) > patience` on dequeue, routes expired to `reneged_target`. Subclass implements `_handle_served_event()` |
| `ConveyorBelt` | `Entity` | Fixed transit time between stations. `ConveyorBelt(name, transit_time, downstream, capacity=None)` |
| `InspectionStation` | `QueuedResource` | Probabilistic pass/fail routing. `InspectionStation(name, inspection_time, pass_rate, pass_target, fail_target)` |
| `BatchProcessor` | `Entity` | Accumulates items until `batch_size` or `timeout_s`, processes as batch. `BatchProcessor(name, batch_size, process_time, downstream, timeout_s=None)` |
| `ShiftSchedule` + `ShiftedServer` | `QueuedResource` | Time-varying capacity via `Shift(start_s, end_s, capacity)` schedule |
| `BreakdownScheduler` | `Entity` | Random UP/DOWN cycles on target; sets `target._broken`. `BreakdownScheduler(name, target, mean_time_to_failure, mean_repair_time)` |
| `InventoryBuffer` | `Entity` | `(s, Q)` reorder policy. `InventoryBuffer(name, initial_stock, reorder_point, reorder_quantity, supplier, lead_time)` |
| `AppointmentScheduler` | `Entity` | Fixed-time arrivals with `no_show_rate`. `scheduler.start_events()` returns initial events |
| `ConditionalRouter` | `Entity` | Declarative routing via ordered `(predicate, target)` list. Factory: `ConditionalRouter.by_context_field(name, field, mapping, default)` |
| `PerishableInventory` | `Entity` | Inventory with shelf life; periodic spoilage sweeps remove expired items. `PerishableInventory(name, initial_stock, shelf_life_s, spoilage_check_interval_s, reorder_point, ...)` |
| `PooledCycleResource` | `Entity` | Pool of N identical units with fixed cycle time. `PooledCycleResource(name, pool_size, cycle_time, downstream, queue_capacity=0)` |
| `GateController` | `Entity` | Opens/closes on schedule or programmatically; queues arrivals when closed. `GateController(name, downstream, schedule=[(open_s, close_s)], initially_open=True)` |
| `SplitMerge` | `Entity` | Fan-out to N targets, `all_of` wait, merge results downstream. Targets resolve `context["reply_future"]`. `SplitMerge(name, targets, downstream)` |
| `PreemptibleResource` | `Entity` | Priority-based resource with preemption. `acquire(amount, priority, preempt=True, on_preempt=callback)` returns `SimFuture[PreemptibleGrant]` |

**Examples** (20 industrial simulations in `examples/`): `bank_branch.py`, `manufacturing_line.py`, `hospital_er.py`, `call_center.py`, `grocery_store.py`, `car_wash.py`, `restaurant.py`, `supply_chain.py`, `warehouse_fulfillment.py`, `parking_lot.py`, `coffee_shop.py`, `drive_through.py`, `laundromat.py`, `pharmacy.py`, `theme_park.py`, `airport_terminal.py`, `hotel_operations.py`, `blood_bank.py`, `elevator_system.py`, `urgent_care.py`

### Network Topology & Partitions

```python
network = Network(name="cluster")
network.add_bidirectional_link(node_a, node_b, datacenter_network("link_ab"))
event = network.send(node_a, node_c, "Request", payload={...})

partition = network.partition([node_a], [node_c], asymmetric=False)  # returns handle
partition.heal()              # selective heal
network.heal_partition()      # heal ALL

# Condition factories: local_network(), datacenter_network(), cross_region_network(),
# internet_network(), satellite_network(), lossy_network(), slow_network(), mobile_3g/4g_network()
```

### Per-Node Clocks

Events ordered by true time; NodeClock transforms the *read* side only.

```python
NodeClock(FixedSkew(Duration.from_seconds(-0.05)))  # constant offset
NodeClock(LinearDrift(rate_ppm=1000))                # accumulating drift
# Use self.now for scheduling (true time), self.local_now for decisions (perceived time)
```

### Logical Clocks

Pure algorithms, not Entities. Stored as entity fields.

```python
clock = LamportClock()                                    # tick(), send(), receive(ts)
vc = VectorClock("node-1", ["node-1", "node-2", "node-3"])  # happened_before(), is_concurrent()
hlc = HybridLogicalClock("node-1", physical_clock=node_clock)  # now(), send(), receive(ts)
```

---

## Observability & Analysis

### Data Class

```python
data.between(30.0, 60.0).mean()     # slice + aggregate
data.percentile(0.99)               # p99
data.bucket(window_s=1.0)           # BucketedData with .times(), .means(), .p99s(), .to_dict()
data.rate(window_s=1.0)             # count/sec per window
```

### Collectors

```python
sink = LatencyTracker("Sink")       # .p50(), .p99(), .mean_latency(), .data, .summary()
tp = ThroughputTracker("Tp")        # .throughput(window_s=1.0)
```

### SimulationSummary

`sim.run()` returns `SimulationSummary` with `.duration_s`, `.total_events_processed`, `.events_per_second`, `.wall_clock_seconds`, `.entities`, `.to_dict()`.

### Analysis

```python
from happysimulator.analysis import analyze, detect_phases
phases = detect_phases(data, window_s=5.0, threshold=2.0)
analysis = analyze(sim.summary, latency=tracker.data, queue_depth=probe_data)
analysis.to_prompt_context(max_tokens=2000)  # LLM-optimized output
```

---

## Simulation Control

Lazy-created via `sim.control` — zero overhead when unused.

```python
sim.control.pause()                          # pause before/during run
sim.control.step(5)                          # process N events
sim.control.resume()                         # resume to completion
state = sim.control.get_state()              # .current_time, .events_processed, .is_paused
sim.control.peek_next(3)                     # upcoming events (while paused)
sim.control.reset()                          # clear heap, re-prime sources (not entity state)
```

### Breakpoints

```python
sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(30.0)))  # one-shot
sim.control.add_breakpoint(EventCountBreakpoint(count=1000))                  # one-shot
sim.control.add_breakpoint(ConditionBreakpoint(fn=lambda ctx: ..., description="..."))
sim.control.add_breakpoint(MetricBreakpoint(entity_name="Server", attribute="depth", operator="gt", threshold=100))
sim.control.add_breakpoint(EventTypeBreakpoint(event_type="Timeout"))
# Management: remove_breakpoint(id), list_breakpoints(), clear_breakpoints()
```

### Hooks

```python
sim.control.on_event(lambda event: ...)          # fires every event, no pause
sim.control.on_time_advance(lambda t: ...)
sim.control.remove_hook(hook_id)
```

---

## Logging

Silent by default. Enable explicitly:

```python
happysimulator.enable_console_logging(level="DEBUG")
happysimulator.enable_file_logging("sim.log", max_bytes=10_000_000)
happysimulator.enable_json_logging()
happysimulator.configure_from_env()  # HS_LOGGING, HS_LOG_FILE, HS_LOG_JSON
happysimulator.set_module_level("core.simulation", "DEBUG")
```

---

## Testing Patterns

- **Unit**: `tests/unit/` | **Integration**: `tests/integration/`
- Deterministic: `ConstantArrivalTimeProvider` + `ConstantLatency(0.1)`
- Fixtures: `test_output_dir` (per-test artifacts), `timestamped_output_dir` (historical)
- Visualization: `pytest.importorskip("matplotlib")`, `matplotlib.use("Agg")`
- Set `random.seed(42)` or use `seed=` on distributions for reproducibility

---

## Key Directories

```
happysimulator/
├── core/           # instant, event, entity, simulation, sim_future, callback_entity,
│   │               # node_clock, logical_clocks, clock, protocols
│   └── control/    # SimulationControl, breakpoints, state
├── load/           # Source, profiles, EventProvider, ArrivalTimeProvider
├── components/     # queue, queued_resource, common (Sink/Counter), random_router,
│   │               # rate_limiter/, network/, server/, client/, resilience/,
│   │               # messaging/, datastore/, sync/, queue_policies/
│   └── industrial/ # BalkingQueue, RenegingQueuedResource, ConveyorBelt,
│                   # InspectionStation, BatchProcessor, ShiftSchedule,
│                   # BreakdownScheduler, InventoryBuffer, AppointmentScheduler,
│                   # ConditionalRouter, PerishableInventory, PooledCycleResource,
│                   # GateController, SplitMerge, PreemptibleResource
├── distributions/  # ConstantLatency, ExponentialLatency, ZipfDistribution, Uniform
├── instrumentation/# Data, LatencyTracker, ThroughputTracker, Probe
└── utils/
tests/              # unit/, integration/, conftest.py
examples/           # m_m_1_queue.py, basic_client_server.py, ...
.dev/               # Design documents (COMPONENTLIB.md, *-design.md)
```

---

## Design Principles

- **Composition over inheritance** — combine smaller entities
- **Protocol-based** — `Simulatable` protocol for duck-typing; clock injection via `set_clock()`
- **Generator-friendly** — express delays with `yield` in `handle_event()`
- **Python 3.13+** — type hints, dataclasses, `|` union syntax
- **Google-style docstrings** — comment the "why", document yield semantics

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Event must have a 'target'" | Add `target=` or use `Event.once()` |
| Generator not progressing | Yield `float` (seconds), not `Instant`; check `end_time` |
| Non-deterministic tests | Use `ConstantArrivalTimeProvider`, set `random.seed(42)` |
| Queue grows forever | Ensure arrival rate < service rate |
| Events not processed | Register entities in `Simulation(entities=[...])` |

Debug: `happysimulator.enable_console_logging("DEBUG")` or `Probe(target=server, metric="depth", data=data, interval=0.1)`

---

## .dev Documentation

Create design docs for major features/architecture decisions. Template: Overview, Motivation, Requirements, Design, Examples, Testing, Alternatives, Implementation Plan. Name as `feature-name-design.md`.

## Skills

| Skill | Description |
|-------|-------------|
| `/commit` | Create a git commit |
| `/commit-push-pr` | Commit, push, and open a PR |
| `/code-review` | Review a PR |
| `/line-count` | Count lines of code (source, tests, examples) |
| `/update-claudemd` | Review changes and update CLAUDE.md |
| `/update-pypi` | Bump version for PyPI release |
