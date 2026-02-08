# happy-simulator: Discrete-Event Simulation for Distributed Systems Engineering

> **Target venue**: SoftwareX (Elsevier)
> **Format**: Original Software Publication, 3,000–6,000 words
> **Status**: Detailed outline with content sketches

---

## Section 1: Introduction (~600 words)

### Hook

> A single garbage collection pause lasting one second caused permanent system
> collapse in our simulation—despite the server running at only 70% utilization
> with a generous 500ms client timeout. The 30% spare capacity was not enough.
> Retry amplification turned a transient disruption into a self-sustaining
> feedback loop that no amount of waiting would resolve.

### Problem Statement

Software engineers designing distributed systems cannot reason effectively about
emergent behavior before deployment. Systems exhibit:

- **Feedback loops**: Client retries amplify server overload, turning transient
  spikes into sustained collapse (metastable failures)
- **Nonlinear phase transitions**: M/M/1 queue length grows as ρ²/(1−ρ); at 90%
  utilization, a 5% load increase doubles expected queue length
- **Cascading failures**: A single slow component (GC pause, cache miss storm,
  network partition) can propagate through retry chains, load balancers, and
  shared resources to collapse an entire service mesh

These behaviors emerge from the interaction of individually well-understood
components and resist prediction from component specifications alone.

### Gap in Existing Tools

| Tool category | Limitation for software engineers |
|:---|:---|
| Analytical queueing theory | Cannot model feedback loops, finite buffers, adaptive policies |
| Load testing / chaos engineering | Requires deployed system; late in development cycle; non-exhaustive |
| SimPy, salabim (Python DES) | Generic process/resource primitives; no distributed-systems vocabulary |
| ns-3, OMNeT++ (network simulators) | Packet-level granularity; too low-level for application reasoning |

### Contributions

This paper presents `happy-simulator`, a discrete-event simulation library for
Python 3.13+ that addresses this gap. Our contributions are:

1. **A generator-based DES engine** with nanosecond-precision integer time,
   uniform target-based event dispatch, and natural expression of multi-step
   processes via Python generators.

2. **A component library of 100+ distributed-systems primitives** organized into
   13 categories (queue policies, cache eviction, resilience patterns, rate
   limiters, load balancers, streaming algorithms, etc.) that encode established
   production-systems patterns as composable simulation building blocks.

3. **Three case studies** demonstrating that simulation reveals non-obvious
   emergent behaviors—metastable failure from retry amplification, GC-induced
   collapse at moderate utilization, and cache cold-start dynamics with
   Zipf-distributed workloads—that are difficult to predict analytically and
   dangerous to discover in production.

---

## Section 2: Related Work (~600 words)

### 2.1 Discrete-Event Simulation Libraries

**SimPy** (Python). The most widely used Python DES library. Provides
`Environment`, `Process`, and `Resource` primitives. Processes use `yield
env.timeout(delay)` for delays. Strengths: mature, well-documented, large
community. Limitations for our use case: no built-in distributed-systems
components; users must implement caches, circuit breakers, retry logic, queue
management policies, etc. from scratch. Time is floating-point, susceptible to
accumulation errors.

**salabim** (Python). Alternative Python DES with animation support. Similar
process/resource model to SimPy with additional features for manufacturing
(conveyors, batching). Not oriented toward software systems modeling.

**DESMO-J** (Java). Java-based DES framework with extensive model library.
Primarily used in logistics and manufacturing simulation. Java ecosystem limits
accessibility for software engineers accustomed to Python.

**AnyLogic, Arena** (commercial). Powerful GUI-based simulation tools used in
operations research. Commercial licensing, proprietary, and oriented toward
manufacturing/logistics rather than software architecture.

### 2.2 Network Simulators

**ns-3** and **OMNeT++** simulate network behavior at the packet level with
detailed protocol stacks. While powerful for network protocol research, they
operate at a granularity inappropriate for reasoning about application-level
concerns: service-to-service interaction patterns, cache hit rates, circuit
breaker state machines, retry policies, and queue management algorithms.

### 2.3 Chaos Engineering

Tools like Netflix's Chaos Monkey and Gremlin inject faults into running
production or staging systems. This approach is valuable but complementary to
simulation: it requires a deployed system, cannot exhaustively explore parameter
spaces, and carries operational risk. Simulation operates at design time, before
code is written or infrastructure provisioned.

### 2.4 Positioning

`happy-simulator` occupies a distinct niche: **application-level DES with
distributed-systems vocabulary**. It operates at the same abstraction level as
architectural diagrams—servers, clients, caches, load balancers, message
queues—rather than packets or factory floors. The component library encodes
patterns from production systems engineering (circuit breakers, bulkheads, CoDel
queues, token bucket rate limiters) as composable primitives, reducing the
modeling effort required to build realistic simulations.

---

## Section 3: Architecture (~1,200 words)

### 3.1 Core Simulation Loop

The simulation engine implements a standard pop-invoke-push event loop:

1. Pop the earliest event from a min-heap (`EventHeap`)
2. Advance the simulation clock to the event's timestamp
3. Dispatch the event to its target entity via `entity.handle_event(event)`
4. Push any returned follow-up events onto the heap
5. Repeat until `end_time` or heap exhaustion

**Content sketch**: Describe the `Simulation` class, `EventHeap`, and dispatch
mechanism. Include ASCII architecture diagram from CLAUDE.md. Emphasize
simplicity—the entire core loop is ~100 lines of Python.

**Figure 1**: Architecture diagram showing EventHeap → Event → Entity →
Result → EventHeap cycle.

### 3.2 Time Representation

Time is represented as 64-bit integer nanoseconds via the `Instant` class.

**Key design rationale**:
- Floating-point accumulation errors cause event-ordering bugs in long
  simulations (e.g., events at t=1000000.1 and t=1000000.1 may compare unequal)
- Integer arithmetic is exact: `Instant.from_seconds(1.5)` stores 1,500,000,000
  nanoseconds
- Comparison operators work without epsilon tolerances
- `Duration` class for time spans with arithmetic operations

### 3.3 Generator-Based Process Model

Entity behavior is expressed as Python generators:

```python
def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
    yield 0.05   # 50ms network latency
    yield 0.10   # 100ms processing time
    return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

**Content sketch**:
- Each `yield` pauses the process for the specified duration (in seconds)
- The runtime wraps generators as `ProcessContinuation` events that reschedule
  after each yield
- Generators can yield `(delay, side_effect_events)` tuples for immediate side
  effects during multi-step processes
- Compare with SimPy's approach (`yield env.timeout()` + `yield resource.request()`)
  to show how happy-simulator's model maps more naturally to request processing

**Figure 2**: Side-by-side code comparison of the same M/M/1 server in SimPy
vs. happy-simulator, highlighting the difference in process expression.

### 3.4 Component Library Taxonomy

**Content sketch**: Expanded version of the JOSS component table with brief
descriptions of each category's design philosophy. Highlight composability—e.g.,
any of 10 cache eviction policies can be combined with any of 4 write policies,
yielding 40 cache configurations without additional code.

**Table 1**: Full component taxonomy (13 categories, 100+ implementations) with
representative examples and use cases.

### 3.5 Instrumentation

**Content sketch**: Describe the `Probe`, `Data`, and `TraceRecorder` subsystems
for collecting time-series metrics and event traces during simulation. Probes
sample entity state at configurable intervals. Data stores time-series for
post-simulation analysis and visualization. TraceRecorder captures event-level
detail for debugging.

---

## Section 4: Case Studies (~1,500 words)

### 4.1 Metastable Failure with Retry Amplification

**Scenario** (`examples/metastable_state.py`):
- Server: 10 req/s capacity (exponential service time, mean 100ms)
- Client: 500ms timeout, 5 max retries, immediate retry on timeout
- Load profile: 90% utilization → 200% spike (10s) → return to 90%

**What happens**:
1. During the spike, queue depth grows and latency exceeds client timeout
2. Timeouts trigger retries, adding ~4x amplified load
3. After the spike ends, external load returns to 90%, but retry load keeps
   effective arrival rate ≈ capacity
4. Queue does not drain; system remains in degraded state indefinitely
5. Recovery requires dropping external load to ~50%—far below nominal capacity

**Key insight**: The retry feedback loop creates a bistable system. The same
external load (90%) is stable when queue is empty but unstable when queue is
large. This is the defining characteristic of metastable failure as described by
Huang et al. (2022).

**Figure 3**: Four-panel plot: (a) load profile over time, (b) queue depth,
(c) goodput vs. offered load, (d) retry amplification factor. Annotate the
phase transition and sustained degradation.

### 4.2 GC-Induced Collapse at Moderate Utilization

**Scenario** (`examples/gc_caused_collapse.py`):
- Server: 10 req/s capacity (deterministic 100ms service time)
- Client: 500ms timeout, 3 max retries, 50ms retry delay
- Load: 7 req/s steady state (70% utilization, 30% spare capacity)
- Perturbation: Single 1.0s GC pause at t=30s

**What happens**:
1. GC pause (1.0s) exceeds client timeout (500ms)
2. All in-flight requests timeout simultaneously
3. Retries amplify load ~4x → effective 28 req/s vs 10 req/s capacity
4. Queue grows indefinitely; system never recovers

**Controlled comparison**:
- With retries: permanent collapse
- Without retries: brief latency spike during GC, immediate recovery

**Key insight**: 30% spare capacity sounds generous, but retry amplification
(~4x) would require reducing to ~18% utilization to recover. A single transient
event at moderate utilization causes permanent failure. This validates the
metastable failure framework and demonstrates that spare capacity is necessary
but not sufficient when retry amplification is present.

**Figure 4**: Two-panel comparison (with retries vs. without retries) showing
queue depth and goodput over time.

### 4.3 Cache Cold-Start Dynamics with Zipf Workloads

**Scenario** (`examples/cold_start.py`):
- Traffic: 1000 req/s, Zipf distribution (s=1.5) across 200 customer IDs
- Cache: LRU eviction, capacity 150 entries (75% of key space)
- Latencies: 0.1ms cache hit, 11ms cache miss (10ms network + 1ms DB)
- Perturbation: Cache reset at t=90s

**What happens**:
1. Cold start: hit rate climbs from 0% to steady state (~75-90%) over 10-30s
2. During warmup, all requests hit the datastore → 1000 req/s datastore load
3. Steady state: Zipf skew means top 20% of keys serve 60-70% of requests;
   150-entry cache captures the hot set
4. Cache reset at t=90s triggers second cold start → datastore load spike
5. Hit rate recovery follows same trajectory as initial warmup

**Key insight**: The interaction between Zipf-distributed access patterns and
finite cache capacity creates a nonlinear relationship between cache size and
hit rate. The cold-start transient generates a datastore load spike of 110x the
steady-state miss rate (1000 req/s vs ~100-250 req/s misses). In a production
system, this spike could trigger cascading failures in the datastore tier.

**Figure 5**: Three-panel plot: (a) cache hit rate over time with reset
annotation, (b) datastore reads/second showing load spike, (c) end-to-end
latency percentiles.

---

## Section 5: Evaluation (~600 words)

### 5.1 Validation Against Analytical Results

**Content sketch**: Compare happy-simulator M/M/1 queue simulation results
against closed-form predictions from queueing theory:
- Expected queue length: ρ²/(1−ρ)
- Expected wait time (Little's Law): L = λW
- Run simulations at ρ = 0.3, 0.5, 0.7, 0.9, 0.95 with sufficient warmup
- Show convergence to analytical predictions with confidence intervals
- Reference: `examples/m_m_1_queue.py` and corresponding integration tests

### 5.2 Expressiveness

**Content sketch**: Compare lines of code required to express equivalent models
in happy-simulator vs. SimPy:
- M/M/1 queue with timeout and retry
- Server with circuit breaker
- Cached datastore with LRU eviction

Hypothesis: happy-simulator requires fewer lines due to pre-built components,
and the resulting code more closely mirrors the architectural description.

### 5.3 Performance

**Content sketch**: Measure events processed per second for:
- Empty event loop (dispatch overhead)
- M/M/1 queue at various utilizations
- Complex model with multiple entity types

Report on a standard machine configuration. Note that Python performance is
acceptable for design-time simulation (not real-time) and that the primary goal
is expressiveness, not raw throughput.

### 5.4 Reproducibility

- All examples are CI-tested (`pytest` runs examples as integration tests)
- Deterministic mode via `ConstantArrivalTimeProvider` and seeded distributions
- All case study results reproducible from the repository's `examples/` directory

---

## Section 6: Discussion (~400 words)

### 6.1 Limitations

**Content sketch**:
- **Single-threaded execution**: No parallel event processing; suitable for
  design-time exploration, not real-time or hardware-in-the-loop simulation
- **Python performance**: CPython overhead limits throughput to ~10⁵–10⁶
  events/second; sufficient for most architectural models but not for
  fine-grained network simulation
- **Alpha status**: API surface may change; some components lack extensive
  validation against real-world system behavior
- **No formal verification**: Simulation results are empirical, not proofs;
  they demonstrate possible behaviors but cannot guarantee completeness

### 6.2 Design Trade-offs

- **Python for accessibility**: Chose Python over C++/Rust for lower barrier to
  entry; most distributed systems engineers already use Python
- **Integer time for correctness**: Small performance cost (nanosecond
  conversions) pays for elimination of floating-point ordering bugs
- **Batteries included**: Large component library increases maintenance burden
  but dramatically reduces time-to-first-simulation for new users

---

## Section 7: Future Work (~300 words)

### 7.1 AI-Assisted System Design

**Content sketch**: The component library provides a structured vocabulary that
could serve as a foundation for AI-assisted system modeling:
- Natural language to simulation model: "Create a system with 3 servers behind
  a load balancer, each with an LRU cache and circuit breaker"
- The library's composable components map well to LLM tool-use patterns
- Parameter sweep exploration guided by LLM-generated hypotheses
- Frame carefully as "we hypothesize" and "preliminary exploration suggests"

### 7.2 Extended Component Library

- Distributed consensus protocols (Raft, Paxos)
- Service mesh patterns (sidecar proxy, control plane)
- Cloud-native patterns (autoscaling, spot instance interruption)

### 7.3 Performance Improvements

- PyPy compatibility for 5-10x speedup
- Optional Rust core for event heap hot path
- Parallel independent event chains

### 7.4 Validation Framework

- Automated comparison against analytical results where available
- Integration with real-system telemetry for model calibration
- Sensitivity analysis tools for identifying critical parameters

---

## Section 8: Conclusion (~200 words)

**Content sketch**:

`happy-simulator` provides software engineers with a simulation tool that speaks
their language. By packaging distributed-systems patterns as composable
simulation components—circuit breakers, CoDel queues, LRU caches, token bucket
rate limiters—the library reduces the gap between architectural diagrams and
executable models.

The three case studies demonstrate that relatively simple models (100-200 lines
of Python) can reveal non-obvious emergent behaviors: retry amplification
turning transient spikes into permanent metastable failure, a single GC pause
collapsing a system with 30% spare capacity, and cache cold-starts generating
110x load amplification on backing stores. These are behaviors that resist
analytical prediction and are dangerous to discover in production.

The library is open-source (Apache 2.0), installable via pip, tested with 102
test files, and documented with 10 runnable examples. We hope it contributes to
a practice of simulation-driven distributed systems design, where engineers
routinely model and stress-test their architectures before writing production
code.

---

## References

See `paper/joss/paper.bib` for shared bibliography. Additional references for
the SoftwareX paper:

- Huang et al., "Metastable Failures in Distributed Systems," OSDI 2022
- SimPy documentation and API reference
- salabim documentation
- Nichols & Jacobson, "Controlling Queue Delay," ACM Queue 2012
- Nygard, "Release It!" (2nd ed., 2018) — circuit breaker pattern origin
- Kleinrock, "Queueing Systems, Volume 1" — analytical validation
- Banks et al., "Discrete-Event System Simulation" (5th ed.) — DES textbook
- Varga & Hornig, "OMNeT++ Overview" — network simulator comparison
- Riley & Henderson, "ns-3" — network simulator comparison
- Basiri et al., "Chaos Engineering," IEEE Software 2016
- Harchol-Balter, "Performance Modeling and Design of Computer Systems"
- Additional references TBD during full draft writing

---

## Figures Plan

| Figure | Description | Source |
|:-------|:------------|:-------|
| Fig. 1 | Architecture diagram (EventHeap cycle) | Draw from CLAUDE.md ASCII art |
| Fig. 2 | SimPy vs. happy-simulator code comparison | Write side-by-side snippets |
| Fig. 3 | Metastable failure: 4-panel (load, queue, goodput, retries) | `examples/metastable_state.py` output |
| Fig. 4 | GC collapse: with-retries vs. without-retries comparison | `examples/gc_caused_collapse.py` output |
| Fig. 5 | Cache cold-start: hit rate, DB load, latency | `examples/cold_start.py` output |
| Fig. 6 | M/M/1 validation: simulation vs. analytical predictions | New evaluation script |

---

## Appendix: Submission Checklist

- [ ] Full draft written (3,000–6,000 words)
- [ ] All figures generated from reproducible example scripts
- [ ] Evaluation benchmarks run and reported
- [ ] SimPy code comparison examples written and verified
- [ ] BibTeX entries validated
- [ ] Co-author review complete
- [ ] SoftwareX LaTeX template formatted
- [ ] Code repository cleaned and tagged for submission
- [ ] JOSS DOI obtained and referenced (if Stage 1 complete)
